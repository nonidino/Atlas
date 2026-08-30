# Graph Tokenizer (Noether 1.1)

**Type:** Concept — model portion (folder: Noether 1.1)
**Status:** Design. Carries forward the resolution-free graph tokenizer of [[graph-tokenizer]] (still the canonical deep-dive on the continuum–particle bridge and the dual-path encoder) with one substantive 1.1 change: node features are deliberately **over-complete** (redundant derived quantities), and token *emission* is deferred to a separate per-field step ([[field-token-streams-1.1]]).
**Related Concepts:** [[00-noether-1.1-overview]], [[graph-tokenizer]], [[field-token-streams-1.1]], [[field-descriptors-1.1]], [[edge-generation-1.1]], [[intelligent-patching]], [[physics-conditioned-query-tokens]], [[learned-query-compression-tokens]], [[coordinate-implicit-tokens]]
**Related Summaries:** [[gns-graph-network-simulators]], [[q-former-architecture]], [[multipole-graph-neural-operator]]

---

## What it does — intuitively

Turn *any* physical state — a grid, a mesh, a bag of particles — into nodes on a graph, where each node stands for a **patch of the domain (or a particle) and encodes what the physics is doing there**, at a fixed latent size no matter how many raw samples fell inside. The tokenizer's job is to make the state *legible and resolution-free*; it does **not** decide who interacts (that's [[edge-generation-1.1]]) or how fields couple (that's [[field-token-streams-1.1]]).

The 1.1 shift: 1.0 handed each node a single raw field value per sample point. A pattern recognizer starved of features learns a near-identity map, so 1.1 hands each node a **redundant bundle** — the field values *and* cheap derived quantities that are what a physical feature is actually made of (a shear layer *is* a velocity gradient; a plume *is* a temperature gradient plus vorticity). Redundancy here is not waste; it is what gives the encoder something to recognize.

## What it does — mathematically

Partition the domain into patches (uniform, or adaptive per [[intelligent-patching]]); each patch $i$ becomes a node at centroid $x_i$. For the point set $\{x_p\}\subset\text{patch}_i$ carrying fields $v(x_p)$, form the **augmented per-point feature**
$$\phi_p = \big[\ \gamma(x_p - x_i),\ v(x_p),\ \underbrace{\nabla v(x_p),\ \omega(x_p),\ \|\nabla v\|,\ \partial_t v(x_p),\ \partial_{tt} v(x_p),\ \dots}_{\text{redundant derived quantities (spatial and temporal)}}\ \big],$$
with $\gamma(\cdot)$ a Fourier coordinate encoding. A bank of $N_q$ learned queries cross-attends over the point set to emit a resolution-free node latent:
$$h_i = \mathrm{CrossAttn}\big(Z_q,\ \{\phi_p\}_{p\in\text{patch}_i}\big),$$
permutation-invariant over an arbitrary number of points ⇒ **same latent dimension and meaning at every resolution** ([[learned-query-compression-tokens]]). The query bank $Z_q$ is *not* one global constant in 1.1: it is **simulation-conditioned**, generated per field from a learned field descriptor $\theta_f$ and the simulation descriptor $z_{\text{sim}}$ so the tokenizer probes for the sub-patch structure each physics makes salient — the graph-tokenizer instance of [[physics-conditioned-query-tokens]], specified in [[field-descriptors-1.1]]. As in 1.0 this runs in **two paths** — a smooth latent $h_i^{\text{lo}}$ (bulk content, into the backbone) and a high-frequency residual $h_i^{\text{hi}}$ (near-cutoff content, into the decoder and the generative fill-in, [[post-decoder-diffusion-1.1]]). **"Residual" here names a role, not a literal subtraction:** $h_i^{\text{hi}}$ is a *second, independent* cross-attention pass over the same point set — its own smaller query bank $Z_q^{\text{hi}}$ ($N_q^{\text{hi}}<N_q$, output width $d_{\text{hi}}<d$) — never computed as $\phi_p$ minus the lo-path's reconstruction. It ends up carrying complementary, near-cutoff content only because training pressure (the reconstruction and rollout losses) pushes the lo-path toward the smooth, well-predicted part and leaves the hi-path free to specialize on whatever the lo-path's coarser query budget can't capture — an emergent division of labor between two parallel encoders, not an explicit subtract-and-encode step.

The derived quantities are computed once, on the input grid, by the same fixed finite-difference operators used elsewhere — they add input channels, not parameters or learned structure. For particles the "derived" bundle is per-neighbor relative velocity / relative position statistics, so the particle and field encoders remain one interface.

**How past states enter — history as derived channels (default), memory as an upgrade.** The model already ingests a short context trajectory (the node graph over the last $n$ steps, [[00-noether-1.1-overview]]); what the portion pages left unstated is *how* the backbone fuses those frames, since the backbone is otherwise spatial. The default answer is the cheapest and fits the redundancy thesis exactly: fold history into each node token as **temporal finite-difference channels** ($\partial_t v,\partial_{tt}v$ above), computed by the same fixed operators as the spatial derivatives — history becomes "more redundant derived features," the backbone stays spatial, and each token already carries velocity/acceleration. This covers the cases physics mostly lives in: a first-order-in-time PDE (NS, heat) needs one frame in principle and two to estimate $\partial_t$; a second-order one (waves) needs two (state $+$ rate). Genuinely **non-Markovian** physics — viscoelasticity, plasticity/hysteresis, memory-kernel closures — needs true memory, and there are two upgrades for it: **spatiotemporal tokens** (one token per $(\text{node},\text{field},\text{frame})$, the backbone attending across time as well as space, [[spatiotemporal-tubelet-tokens]]) or a **carried recurrent latent** across rollout steps ([[memory-augmented-physics-models]]), each gated on whether the target system is actually history-dependent. **Caveat — the 1.0 failure at the temporal level:** more history makes "copy the last frame" *easier*, so whichever fusion is used must be trained under the motion-demanding multi-step loss ([[training-scheme-1.1]]) — the same warning [[edge-generation-1.1]] raises for learned edges — or history feeds the near-identity collapse instead of curing it.

## Physics relevance

- **Band-limited losslessness.** A field cannot inject losslessly into a finite latent (counting argument), but physical fields are band-limited (viscous/Kolmogorov cutoff), so the achievable target is "lossless above $k_{\max}$." The dual path preserves the near-cutoff band the smooth query-compression drops — the mechanism against spectral-bias smoothing ([[open-architectural-problems]] Problem 3).
- **Feature-aligned tokenization.** Encoding gradients/vorticity means the coarse latent already carries the *derivatives* that dominate transport, so the backbone need not re-derive them from a coarse value field — directly targeting the RBC failure where plume-scale transport was averaged away ([[noether-1.0-rbc]] §6.6).
- **Geometry-universality.** Because a node is "a centroid + an encoding of its neighborhood," grids, meshes, scattered sensors, and particles all become the same object — the continuum–particle bridge that is the project's headline goal ([[graph-tokenizer]]).

## How fundamental constants are included

The tokenizer is **constant-light by design** — topology and node encoding should be geometric, not physics-parameterized, or the representation stops being equation-agnostic. Constants touch it in exactly two controlled ways, both upstream:

1. **Through nondimensionalization only** (the "ruler," [[conditioning-and-constants-1.1]]): the field values $v(x_p)$ fed in are already rescaled by characteristic scales built from dimensional constants, so a low- and high-viscosity state present at comparable magnitude. The tokenizer itself sees no $\nu$.
2. **Through adaptive-patch monitors, if enabled** ([[intelligent-patching]]): the refinement monitor $\eta$ may use a constant-derived threshold (e.g. a diffusive length setting the finest patch), but this changes *where patches go*, never the encoder weights.

Deliberately, **no constant is embedded into the node latent here** — that is the conditioning stage's job ([[conditioning-and-constants-1.1]]), kept separate so the same tokenizer serves every regime.

## See Also
- [[graph-tokenizer]] — the full continuum–particle-bridge deep-dive this inherits
- [[noether-1.1-fineness-and-token-density]] — how few tokens (patch count) drives the "pixelated" output, and the density fix
- [[field-token-streams-1.1]] — how the node latent becomes per-field tokens
- [[intelligent-patching]] — adaptive patch placement
- [[post-decoder-diffusion-1.1]] — consumer of the high-frequency residual path
- [[spatiotemporal-tubelet-tokens]] / [[memory-augmented-physics-models]] — the non-Markovian upgrades to history-as-derived-channels
- [[training-scheme-1.1]] — the multi-step loss that keeps history from feeding the copy-the-input collapse
- [[graph-tokenizer-atlas-0.1]] — reuses this tokenizer directly, per-agent, in the [[00-atlas-0.1-overview]] track; adds boundary-token exposure for declared-edge attachment
