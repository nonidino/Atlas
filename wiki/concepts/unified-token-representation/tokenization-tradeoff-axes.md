# The Three Tokenization Trade-Off Axes

**Type:** Concept (folder: unified-token-representation)
**Related:** [[00-token-representation-overview]], [[neural-operators]], [[patch-embedding-tokens]], [[spectral-fourier-tokens]], [[vector-quantized-tokens]], [[graph-mesh-tokens]], [[wave-particle-dual-tokens]], [[structure-preserving-tokens]], [[physics-conditioned-query-tokens]]

---

## Why Three Axes?

The fourteen representations in this folder are not an unstructured list. They cluster along three fundamental design tensions that apply to any tokenizer for physics:

1. **Continuous vs. Discrete** — what the token *is*: a real-valued vector, or an integer symbol from a finite vocabulary.
2. **Local vs. Global** — what the token *sees*: a small spatial region, or the entire domain.
3. **Grid-Tied vs. Resolution-Free** — what the token *indexes*: a cell in a fixed discretization, or a representation in continuous function space.

Every tokenizer in the folder occupies a position on all three axes simultaneously, and most failure modes — rollout instability, lost conservation, missed long-range coupling, broken resolution transfer — trace back to a bad position on one of them. Understanding the axes is understanding the design space.

---

## Axis 1: Continuous vs. Discrete

### What it means

A **continuous** token is a real-valued vector $\mathbf v \in \mathbb{R}^C$. It carries graded information: two physically similar patches produce nearby vectors; smooth perturbations in the input produce smooth changes in the token. The mapping from field to token is differentiable.

A **discrete** token is an integer $k \in \{1, \ldots, K\}$ indexing a finite codebook. The field is compressed to a *symbol* over a fixed *vocabulary* of $K$ prototypes. Two tokens are either identical or not; there is no notion of distance between them unless you embed the integers separately.

### Why the distinction matters so deeply

Physics is fundamentally continuous: field values are real numbers, PDE solutions are functions, and the governing equations are differential (infinitely sensitive to local structure). But the most powerful generative architectures — LLMs, BERT, DALL-E — are fundamentally discrete: they process integer token sequences over a fixed shared vocabulary. This makes cross-entropy training, masked-token prediction, and any-to-any inference between arbitrary modalities natural. **The continuous-vs-discrete axis is where physics and the LLM stack are most incompatible.**

### Continuous tokens: the physics-native option

Every strong PFM result in the wiki — Walrus, Poseidon, GPHyT — uses continuous tokens. The information bottleneck is minimal: a patch token is a weighted average of real field values, and a linear decoder reconstructs the field exactly (up to sub-patch detail). Smooth loss landscapes, standard gradient descent, differentiable decoders — all follow naturally.

**Where continuous tokens succeed:**
- Forward emulation (predicting the next state) — the output must be a continuous field, so the whole pipeline can be continuous and differentiable.
- Rollout — smooth latent trajectories mean errors don't discretize and amplify.
- Regression — mean-squared loss is a natural objective.
- Physics-informed training — PDE residuals, conservation diagnostics, and gradient-based physics constraints all require continuous differentiable representations.

**Where continuous tokens struggle:**
- *No shared vocabulary.* A continuous token for velocity in a fluid simulation is a different kind of object from a continuous token for magnetic-field strength in a plasma simulation. Two different PFMs trained on different physical systems have entirely different latent spaces; there is no natural shared "language" between them. The AION-1 astronomy model's key insight was that a discrete codebook *forces* heterogeneous modalities into a common symbol space — the same thing that makes text powerful.
- *Can't do masked modeling natively.* Masked language modeling (BERT) works because masking a discrete token and predicting it from context is a well-defined task over a finite distribution. Masking a continuous token requires predicting a real-valued vector, which is a regression, not classification — much harder to train stably across wildly different field scales. AION-1 does this with its FSQ/LFQ tokens; no continuum PFM does it.
- *Generation.* Autoregressive generation over a continuous space requires a density model (diffusion, normalizing flow) at every step. Over a discrete vocabulary it requires only a softmax — far cheaper and more stable. Diffusion PFMs ([[arch-diffusion-backbone]], [[pisd-physics-informed-spectral-diffusion]]) work around this but at the cost of a multi-step sampling process.

### Discrete tokens: the LLM-compatible option

**VQ/FSQ/LFQ** ([[vector-quantized-tokens]]) quantize a continuous encoding:

$$\hat z = e_{k^\ast}, \quad k^\ast = \arg\min_k \|z - e_k\|_2, \quad \text{token} = k^\ast \in \{1, \ldots, K\}$$

**Scalar CDF tokens** ([[scalar-cdf-tokens]]) discretize scalar inputs via their empirical distribution:

$$\text{token}(s) = \lceil B \cdot F(s) \rceil, \quad F(s) = \Pr[S \le s]$$

**Where discrete tokens succeed:**
- *Any-to-any inference.* Once all modalities live in a common vocabulary, you can mask any subset and condition on the rest — directly solving inverse problems, cross-modal prediction, and multi-physics coupling under one cross-entropy objective. AION-1 exploits this across 39 astronomical modalities.
- *Unified vocabularies.* A discrete token for "velocity-field regime X" can literally be the same integer as a token from a temperature field if they represent the same physical state. One codebook, one embedding table, one transformer.
- *Compositional generation.* Autoregressive generation is trivial: sample one token at a time from a softmax, just like GPT.
- *LLM integration.* Discrete physics tokens can be concatenated with text tokens and processed by an existing LLM — the foundation of eventual physics-language models.

**Where discrete tokens fail:**
- *Lossy bottleneck.* The quantization error is irreducible — it is lower-bounded by the codebook granularity. For a turbulent flow with fine-scale vortex filaments, rounding to the nearest codebook vector destroys exactly the high-frequency structure that drives the dynamics. This is tolerable for *static observational data* (AION-1's galaxy maps), but in *dynamical emulation* the error from each quantized token compounds over rollout steps ([[autoregressive-rollout-stability]]).
- *No smooth gradient structure.* The $\arg\min$ is non-differentiable; straight-through gradient estimators or reparameterization tricks are needed. The latent space becomes a discrete grid rather than a smooth manifold, making optimization delicate (codebook collapse in VQ-VAE is the canonical failure mode).
- *Physics structure is statistical, not structural.* A VQ codebook entry is a *statistical prototype* — the centroid of a cluster in embedding space. Nothing guarantees it corresponds to a physically admissible state (e.g. divergence-free velocity) or that two adjacent integers are physically adjacent. Physical constraints must be learned, not built in.

### The wave-particle resolution

[[wave-particle-dual-tokens]] refuses the trade-off: maintain both a continuous "wave" token (phase, smooth gradients, regression/rollout) and a discrete "particle" token (symbolic index, masked modeling, generation) simultaneously, coupled by a mutual-reconstruction loss. The name is physically motivated — a PDE field has both a wave description (Fourier modes) and a particle description (Lagrangian tracers), and the tokenizer should reflect both.

**[AI Inference]:** The optimal long-term solution is almost certainly a form of wave-particle duality: continuous tokens for everything involving forward emulation, physics-informed training, or fine-scale rollout; discrete tokens as a "regime label" that conditions the model on what kind of physics it's doing (laminar/turbulent/shocked, fluid/plasma/elastic), without requiring the *field values* to be discretized. This mirrors how an expert physicist thinks: a flow has a *symbolic regime* (turbulent, fully-developed boundary layer) and a *continuous state* (the velocity and pressure fields), and these two levels of description are complementary.

---

## Axis 2: Local vs. Global

### What it means

A **local** token is constructed from a *bounded region* of the physical domain — a $p \times p$ patch, a particle and its radius-$r$ neighbors, a node in a mesh. The token encodes only what is happening in a small neighborhood. Information from far away can only reach it by propagating through many layers of the network.

A **global** token is constructed from *the entire domain* — a Fourier mode (every grid point contributes to every spectral coefficient), a full-graph attention head (every node attends to every other node), a Q-Former query (cross-attends into all input patches). A single global token already "knows" about the boundary conditions on the other side of the domain.

### Why this distinction maps onto PDE type

PDEs classify by the nature of their information propagation, and this maps almost perfectly onto the local-vs-global axis:

**Elliptic PDEs** (Laplace $\nabla^2 u = 0$, Poisson $\nabla^2 u = f$, Stokes flow) are the extreme global case. The solution at any interior point depends on the boundary conditions *everywhere* on the boundary simultaneously — there is no propagation in time, only an instantaneous global equilibrium. A local tokenizer is fundamentally wrong for elliptic physics: it can only approximate the global solution by propagating information through many layers, and depth is costly.

$$\nabla^2 u = 0 \implies u(x) = \oint_{\partial\Omega} G(x, y) \frac{\partial u}{\partial n}(y)\, dS(y)$$

The Green's function $G(x,y)$ is non-local by construction — every boundary point contributes to every interior value.

**Hyperbolic PDEs** (wave equation, advection, Euler equations) have *characteristics* — information travels at finite speed $c$. At time $t$, a point $x$ only "knows about" the region of the domain within distance $ct$ of it. A local tokenizer with a small patch size is natural here, since the relevant information is concentrated nearby (at least for small $t$).

$$\partial_{tt} u = c^2 \nabla^2 u \implies u(x,t) = f(x - ct) + g(x + ct) \quad \text{(1D)}$$

**Parabolic PDEs** (heat equation, diffusion) are intermediate: information propagates globally but with exponential decay. Initially local; asymptotically global.

### Local tokens: the flexible, cheap option

Representations: [[patch-embedding-tokens]], [[spatiotemporal-tubelet-tokens]], [[hierarchical-windowed-tokens]], [[graph-mesh-tokens]], [[coordinate-implicit-tokens]] (queried locally).

$$\mathbf v_j = \text{Enc}(\text{region}_j), \quad |\text{region}_j| \ll |\Omega|$$

Self-attention cost between $N$ local tokens: $O(N^2)$ for full attention; $O(NM^2)$ for windowed attention (Swin, $M$ = window size).

**Where local tokens excel:**
- *Sharp features and discontinuities.* A shock, a phase boundary, a thin boundary layer — all are spatially localized. A local token centered on the shock carries precisely the right information about it. Global spectral tokens Gibbs-ring at shocks ([[spectral-fourier-tokens]] con).
- *Arbitrary geometry.* A radius-graph token ([[graph-mesh-tokens]]) works for turbine blades, biological tissue, fracture surfaces — anywhere a Cartesian patch cannot go. The neighborhood of a node is always well-defined regardless of the domain shape.
- *Computational efficiency.* With windowed or sparse attention ($O(NM^2)$ or $O(Nd)$ for fixed-degree graphs), local tokenizers scale linearly in domain size. Global attention at the same resolution is quadratic.
- *Compositionality.* Local tokens can be spatially concatenated, sub-sampled, or rearranged — they tile the domain like LEGO bricks. This makes mixed-resolution and multi-domain training natural.

**Where local tokens fail:**
- *Long-range elliptic coupling.* The canonical failure case: a Poisson pressure solve $\nabla^2 p = -\nabla \cdot (\mathbf u \cdot \nabla \mathbf u)$ couples every point in the domain to every other. A local tokenizer approximates this by stacking $L$ attention layers, each passing information one patch's width. For a domain of $J \times J$ patches you need $L \sim J/2$ layers to couple the first patch to the last — impractically deep for high-resolution domains. This is why [[multiscale-hierarchical-gnn]] and [[hierarchical-windowed-tokens]] build coarser levels: the coarse level carries global context cheaply (analogous to the coarse-grid solve in multigrid).
- *Invariants are hard to compute.* A global conserved quantity (total energy, total mass, total momentum) is a sum over the *whole* domain. A local token carries no information about the global sum; conservation diagnostics require aggregating all tokens, which is non-local. Structure-preserving tokens ([[structure-preserving-tokens]]) need a non-local projection step to enforce divergence-freedom.
- *Non-local boundary conditions.* Periodic BCs couple opposite boundaries; inflow BCs on one wall affect the interior globally (especially for subsonic flow where pressure information travels upstream). Local tokens see only their neighborhood and can miss these couplings.

### Global tokens: the physics-informed, expensive option

Representations: [[spectral-fourier-tokens]], [[branch-trunk-operator-tokens]] (branch encodes the whole input), [[learned-query-compression-tokens]] (each query cross-attends everywhere), [[physics-conditioned-query-tokens]].

For spectral tokens: each mode coefficient involves the whole domain:
$$\hat u(k) = \int_D u(x) e^{-2\pi i k \cdot x}\, dx$$

For full-graph attention: every node attends to every other ($O(N^2)$).

**Where global tokens excel:**
- *Elliptic physics.* A spectral or operator token natively captures global pressure-velocity coupling. The FNO spectral layer computes a global convolution in one step:
$$(\mathcal K u)(x) = \mathcal F^{-1}(R_\phi(k) \cdot \mathcal F u(k))(x)$$
  This is exactly what a Poisson solver does — one operation, all long-range information.
- *Resolution invariance.* A Fourier mode is defined in continuous function space; it means the same thing at $128^2$ and $1024^2$ ([[spectral-fourier-tokens]]). A global neural operator token is inherently resolution-free.
- *Compact representation of smooth fields.* The $k$-th Fourier coefficient decays exponentially for smooth periodic functions (Sobolev decay). Ten spectral modes can represent a smooth pressure field more faithfully than hundreds of patch tokens. PISD exploits this for a compressed spectral latent ([[pisd-physics-informed-spectral-diffusion]]).
- *Cheap derivatives.* In spectral space, $\partial_x \to 2\pi i k_x$ is a diagonal operation. Divergence, curl, Laplacian — all one-step linear projections. Physics residuals and conservation diagnostics are trivially cheap to compute globally.

**Where global tokens fail:**
- *Geometry restrictions.* The fast Fourier transform requires a periodic, rectangular, uniformly-sampled domain. Non-periodic BCs (wall-bounded flow), irregular domains (complex geometries), and non-uniform meshes all break it. Geo-FNO and SFNO (spherical harmonics) partially fix this but require specialized implementations.
- *Gibbs ringing at shocks.* Truncating a Fourier series at mode $k_\text{max}$ at a discontinuity produces oscillations that do not converge to zero with more modes. A shock is globally visible in spectral space (it affects all modes), but the spatial representation near the shock is corrupted. Local tokens with derivative channels handle shocks far better.
- *Global computation cost.* Full-graph attention is $O(N^2)$; an FFT is $O(N \log N)$; but $N$ global tokens interacting fully is expensive regardless. In 3D with $64^3$ nodes, $N = 262144$ — full attention is intractable.
- *Spatial locality is lost.* A global token carries no notion of *where* in the domain — it describes a property of the whole field. Applying a spatially-localized boundary condition ("fix $u=0$ on this patch of wall") to a global representation requires non-local surgery.

### The multiscale resolution

Both [[hierarchical-windowed-tokens]] (Swin/scOT) and [[multiscale-hierarchical-gnn]] interpolate the local-global axis the way multigrid interpolates between fine-grid (local) and coarse-grid (global) levels. The fine level handles shocks and local features (local); the coarse level handles global pressure fields and long-range coupling (global). Skipping connections bridge them. This is not truly global (the coarse level is still a grid/graph, just coarser), but it achieves near-global reach at near-linear cost.

**[AI Inference]:** The local-global axis maps cleanly onto the PDE classification (elliptic/parabolic/hyperbolic). A PFM that *routes* between local and global token representations based on the inferred PDE type — detected in-context from the trajectory statistics — would automatically match the representation to the physics. An in-context classifier (is this an elliptic, parabolic, or hyperbolic problem?) could gate between a spectral global head and a patch local head, serving both kinds of physics with one model. This is a concrete instance of [[mixture-of-experts]] routing applied at the tokenizer level.

---

## Axis 3: Grid-Tied vs. Resolution-Free

### What it means

A **grid-tied** token is defined relative to a specific discretization. The token count $N$ is a function of the grid resolution $R$ (e.g. $N = (R/p)^2$ for patch tokens with patch size $p$); the embedding weights are learned at training resolution and are *wrong* (or at least suboptimal) at a different resolution; and the model's effective capacity is implicitly tied to the grid it was trained on.

A **resolution-free** token is defined in *continuous function space* — it is a property of the function $u(x)$, not of any particular sampling of that function. Changing the grid resolution changes the computational cost of reading off the tokens, but not what the tokens mean. A model trained on $128^2$ data, given $512^2$ data, produces the same tokens and the same prediction — because the tokens are defined independent of the grid.

### The mathematical distinction: emulators vs. operators

This axis is precisely the distinction between learning a **function** (a mapping from $\mathbb{R}^n$ to $\mathbb{R}^m$, where $n$ = number of grid points) and learning an **operator** (a mapping between function spaces, $\mathcal G: \mathcal U \to \mathcal V$, where $\mathcal U, \mathcal V$ are spaces of functions defined on $\Omega$):

$$\text{Grid-tied (emulator):} \quad \mathcal N_\theta: \mathbb{R}^{J \times J \times n} \to \mathbb{R}^{J \times J \times n}, \quad J \text{ fixed at train time}$$

$$\text{Resolution-free (operator):} \quad \mathcal G_\theta: \mathcal U(\Omega, \mathbb{R}^n) \to \mathcal V(\Omega, \mathbb{R}^n), \quad J \text{ arbitrary at eval time}$$

The neural operator program ([[neural-operators]]) is precisely the program of building resolution-free models. The universal approximation theorem for operators ([[branch-trunk-operator-tokens]]) guarantees that a DeepONet can approximate any continuous operator to arbitrary accuracy, independent of the discretization.

### Grid-tied tokens: the practical, locked option

Representations: [[patch-embedding-tokens]], [[spatiotemporal-tubelet-tokens]], [[hierarchical-windowed-tokens]], [[adaptive-compute-tokens]] (partially; see below).

The patch embedding weight $\mathbf W_{\mathcal E} \in \mathbb{R}^{C \times n \times p \times p}$ is trained for patches of exactly $p \times p$ grid cells. A model trained at $J=128$ sees $(128/p)^2$ tokens per field. At $J=512$ the same patch size gives $(512/p)^2$ tokens — quadruple the count — but the embedding weights were learned for 128-resolution patches. Sub-patch structure at 512 that would have been in separate tokens at 128 is now aggregated into one token and lost.

**Where grid-tied tokens excel:**
- *Training simplicity.* A fixed grid means fixed-size tensors, standard batching, and no special treatment of variable-length inputs. This is why every strong empirical PFM uses grid-tied tokens.
- *Sub-patch structure is learnable.* The embedding weights $\mathbf W_{\mathcal E}$ learn the *statistics* of what appears in a patch at the training resolution — including correlations between nearby grid cells that resolution-free methods (which know nothing about the grid spacing) cannot exploit.
- *Decoder is exact.* A linear patch recovery decoder $\mathbf W_{\mathcal R}$ (Poseidon) recovers the field at training resolution with no approximation beyond sub-patch averaging.

**Where grid-tied tokens fail:**
- *Resolution locking.* Training at $128^2$ and evaluating at $512^2$ requires either (a) interpolating the field to $128^2$ before tokenizing (losing information), (b) retraining from scratch, or (c) adaptive stride (CSM, Walrus) — which changes the token semantics by compressing different amounts of information into each token at different resolutions. None of these is zero-cost.
- *Generalization across resolutions is fragile.* A grid-tied model has no mechanism to know that a 4-cell average at $512^2$ is the same physical patch as a 1-cell unit at $128^2$. It may learn to handle slight resolution changes through data augmentation but will fail on large resolution mismatches.
- *Aliasing compounds over rollout.* The fixed patch boundary cuts through whatever structure happens to be at that position. Over autoregressive rollout, features that drift across patch boundaries alias differently at each step, and these aliasing artifacts accumulate ([[autoregressive-rollout-stability]]). Patch jittering (Walrus) addresses this by randomizing the patch grid; [[structure-preserving-tokens]] (equivariant translation) fixes it architecturally.
- *Token count scales with grid.* $N \propto R^2$ (2D) or $R^3$ (3D). Self-attention is $O(N^2) \propto R^4$ — rapidly intractable at high resolution. This is the primary computational driver for windowed attention ([[hierarchical-windowed-tokens]]) and adaptive stride ([[adaptive-compute-tokens]]).

### Resolution-free tokens: the principled, harder option

Representations: [[spectral-fourier-tokens]], [[branch-trunk-operator-tokens]], [[coordinate-implicit-tokens]], [[learned-query-compression-tokens]], [[physics-conditioned-query-tokens]].

**Spectral tokens** achieve resolution-freedom because Fourier modes are continuous functions. The modes $\hat u(k)$ for $|k| \le k_\text{max}$ are defined by the integral

$$\hat u(k) = \int_D u(x) e^{-2\pi i k \cdot x} dx$$

which is independent of any discretization. The same $k_\text{max}$ modes describe the same physical structures regardless of the grid used to compute them. FNO trained at $64^2$ evaluates correctly at $128^2$ — because both grids resolve the same Fourier modes up to $k_\text{max}$, and those modes are what the model operates on.

**Branch-trunk tokens** achieve resolution-freedom differently: the trunk net $t_k(y)$ is a neural function of coordinates, evaluated at any query point $y$. The sensor layout for the branch can be any set of points:

$$\mathcal G(u)(y) \approx \sum_k b_k(u(x_1,\ldots,x_M)) \cdot t_k(y)$$

Change the sensor locations, change the query points — the operator is still defined. The learned basis $\{t_k\}$ is geometry-flexible in a way the Fourier basis is not.

**Q-Former tokens** achieve resolution-freedom via compression: $Q$ learned queries cross-attend into arbitrarily many input tokens $N$. As $N$ grows with resolution, the $Q$ output tokens are unchanged in count and semantics. The cost of extracting the tokens scales as $O(QN)$ (linear in $N$, not quadratic), but the downstream model always sees $Q$ tokens regardless of resolution.

**Where resolution-free tokens excel:**
- *True operator learning.* Train at low resolution (cheap), evaluate at high resolution (costly, but same model). This is the fundamental scientific use case: train on simulation data at manageable grid sizes, deploy on production grids.
- *No aliasing over rollout.* Spectral modes are defined continuously — a shifted input doesn't change the mode indices, only the mode coefficients. There is no patch-boundary aliasing because there are no patch boundaries.
- *Cross-resolution generalization.* A model trained on a distribution of resolutions generalizes because all resolutions share the same token semantics. [[poseidon-pde-foundation-model]] achieves this partially via continuous position bias; operator learning models achieve it natively.
- *Compact representation.* A few dozen Fourier modes can represent a smooth field more compactly than hundreds of patch tokens, enabling higher effective "resolution" per token budget.

**Where resolution-free tokens fail:**
- *Training cost and engineering complexity.* Variable-length inputs mean variable-length token counts; batching requires padding or masking; implementation is harder than the fixed-grid case.
- *Geometry restrictions (spectral only).* FFT requires periodic rectangular domains. Non-rectangular domains break spectral tokens; irregular geometry is only handled by graph tokens (local, not global by default) or by mapping the irregular domain to a reference domain (expensive and approximate).
- *Fixed sensor layout (branch-trunk only).* The branch net expects input at fixed sensor locations; a new measurement grid requires re-embedding or interpolation on the *input* side (even if the output side is freely queryable).
- *Opaque compression (Q-Former only).* The Q learned queries may not correspond to interpretable physical quantities; information lost in the $N \to Q$ compression cannot be recovered downstream ([[physics-conditioned-query-tokens]] addresses this).
- *Gibbs ringing (spectral only).* Discontinuous fields cannot be represented faithfully by a truncated Fourier series — the classic limitation.

### Adaptive-compute tokens: the middle ground

[[adaptive-compute-tokens]] (Walrus CSM) occupy a deliberate middle position. By varying the patch stride $s(R) = R/N_\text{axis}$ as a function of input resolution, CSM keeps the token count $N_\text{axis}^2$ fixed regardless of $R$ — mimicking the fixed token budget of a Q-Former without the cross-attention cost. But the embedding weights are still learned at specific resolutions, and the compressed-patch semantics change with $s(R)$: a token at $R=128$ represents 1 cell, at $R=512$ it represents 16 cells. The model receives the same number of tokens but they mean different things. This is resolution-flexible but not truly resolution-free.

**[AI Inference]:** The deepest connection between this axis and the physics encoding spectrum ([[00-token-representation-overview]]) is that *resolution-freedom is equivalent to operator-learning*, and operator-learning is a necessary (not sufficient) condition for a true PFM. A grid-tied PFM is not a foundation model in the operator-learning sense; it is a fast surrogate for a specific discretization. The distinction matters for deployment: a true PFM can be trained on coarse simulation data and applied directly to fine-resolution production grids. A grid-tied surrogate must be retrained for each resolution of interest. All three current empirical PFMs (Walrus, Poseidon, GPHyT) are grid-tied surrogates that approximate operator behavior via engineering workarounds (adaptive stride, continuous position bias, resolution augmentation) — not true resolution-free operators.

---

## How the Axes Interact

The three axes are not independent — certain combinations are structurally coherent and others are contradictory:

| Combination | Example | Coherent? | Notes |
|---|---|---|---|
| Continuous + Local + Grid-tied | Patch tokens (Walrus, Poseidon, GPHyT) | ✓ Strong | The dominant current approach; 3 axes all at the "easy" end |
| Continuous + Local + Resolution-free | Graph tokens (GNS, Dynami-CAL) | ✓ Strong | Local on irregular meshes; resolution-free because graph is mesh-agnostic |
| Continuous + Global + Resolution-free | Spectral / FNO / DeepONet | ✓ Strong | The neural-operator program; global + resolution-free are almost synonymous |
| Discrete + Local + Grid-tied | VQ-VAE on image patches | ✓ Exists | Standard image generation; lossy but workable for static data |
| Continuous + Global + Grid-tied | Full self-attention over all patches | ✓ Degenerate | Possible but $O(N^4)$ at high resolution; not used |
| Discrete + Global + Resolution-free | — | Rare | Difficult: global discrete tokens lose spatial info needed for reconstruction |
| Both Continuous + Discrete (dual) | Wave-particle dual | ✓ Proposed | Specifically designed to avoid committing to one side of axis 1 |
| Continuous + Local + Resolution-free (learned queries) | Q-Former / Perceiver | ✓ Proposed for physics | Content-adaptive global compression with fixed token budget; physically local if queries learn to extract local features |

**The key diagonal:** The empirically strongest PFMs occupy the corner (Continuous, Local, Grid-tied). The theoretically strongest representations occupy the opposite corner (Continuous/Discrete-dual, Global, Resolution-free). The gap between them is the gap between "what works today" and "what a true PFM requires."

---

## Scorecard Mapping

Each of the eight criteria in [[00-token-representation-overview]] traces to one or more axes:

| Criterion | Primary axis | Secondary |
|---|---|---|
| 1. Resolution invariance | **Grid-tied vs. Resolution-free** | — |
| 2. Dimensional / geometric generality | **Local vs. Global** | Grid-tied vs. Resolution-free |
| 3. Field / modality unification | **Continuous vs. Discrete** | — |
| 4. Scale awareness | Orthogonal (handled by [[structure-preserving-tokens]] coordinate/parameter injection) | — |
| 5. Structure preservation | Orthogonal (handled by equivariance / conservation constraints) | — |
| 6. Rollout stability | **Grid-tied vs. Resolution-free** (aliasing) + **Local vs. Global** (error propagation) | — |
| 7. Compute tractability | **Local vs. Global** + **Grid-tied vs. Resolution-free** (Q-Former) | — |
| 8. Invertibility | **Continuous vs. Discrete** | — |

---

## Design Recommendations

Given the analysis:

1. **For forward emulation (PDE simulation):** Stay continuous (criterion 8 + rollout stability); choose local or multiscale for irregular geometry / sharp features; move toward resolution-free via Q-Former or spectral head to unlock cross-resolution transfer.

2. **For multi-physics / any-to-any inference:** Add a discrete head (wave-particle dual) so a shared vocabulary can bridge modalities, while keeping the continuous head for emulation accuracy.

3. **For elliptic-dominated physics (Poisson, Stokes, MHD pressure):** Bias toward global tokens (spectral or multiscale hierarchy); local tokens will need many layers to approximate global coupling.

4. **For shock-dominated / hyperbolic physics (Euler, wave):** Bias toward local tokens with derivative channels; global spectral tokens will Gibbs-ring at discontinuities.

5. **For production deployment across resolutions:** The minimum viable step is **adaptive-compute tokenization (CSM)** (already in Walrus) as a near-term proxy; the principled solution is a resolution-free Q-Former front-end with physics-conditioned queries ([[physics-conditioned-query-tokens]]).

**[AI Inference]:** The three axes suggest a concrete **roadmap for tokenizer evolution**: (1) start at (Continuous, Local, Grid-tied) with patch tokens — easy to train, empirically validated; (2) add a coarse global level (multiscale Swin or graph V-cycle) to fix the long-range elliptic weakness — (Continuous, Local+Global, Grid-tied); (3) add a Q-Former or spectral head to fix resolution-locking — (Continuous, Local+Global, Resolution-free); (4) add a discrete partner head to enable cross-modal masked inference — (Continuous+Discrete, Local+Global, Resolution-free). Each step is a targeted upgrade that addresses one axis without requiring the others to change. This matches the incremental engineering philosophy of current PFMs better than a single grand architectural replacement.

---

## See Also

- [[00-token-representation-overview]] — hub; scorecard; folder map
- [[patch-embedding-tokens]] — (Continuous, Local, Grid-tied) baseline
- [[spectral-fourier-tokens]] — (Continuous, Global, Resolution-free)
- [[graph-mesh-tokens]] — (Continuous, Local, Resolution-free on irregular geometry)
- [[vector-quantized-tokens]] — (Discrete, Local, Grid-tied)
- [[learned-query-compression-tokens]] — (Continuous, Global, Resolution-free via compression)
- [[physics-conditioned-query-tokens]] — resolution-free + physically interpretable
- [[wave-particle-dual-tokens]] — resolves the continuous-vs-discrete axis
- [[hierarchical-windowed-tokens]] — interpolates local-vs-global via multiscale
- [[adaptive-compute-tokens]] — grid-tied workaround for resolution-locking
- [[structure-preserving-tokens]] — the two orthogonal axes (scale & structure) not covered here
- [[neural-operators]] — the operator-learning program = resolution-free from first principles
- [[mixture-of-experts]] — dynamic routing between local and global token modes
