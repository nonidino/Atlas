# Wave–Particle Dual Tokens (Proposed Alternative)

**Type:** Token representation — *proposed better alternative* (folder: unified-token-representation)
**Status:** Synthesis / proposal, grounded in "Wave-Particle Continuous-Discrete Dualistic Visual Tokenization" (arXiv:2511.01593). Heavy **[AI Inference]** for the physics adaptation.
**Builds on:** [[multimodal-tokenization]] (wave-particle dual tokenization), [[vector-quantized-tokens]], [[spectral-fourier-tokens]]
**Related:** [[00-token-representation-overview]], [[patch-embedding-tokens]], [[aion-1-astronomy]], [[neural-operators]]

---

## Intuition

The deepest tension in [[00-token-representation-overview]] is **continuous vs. discrete**: continuous tokens (patch, spectral, coordinate) preserve fine detail and smooth gradients but break LLM-style shared-vocabulary generation and have no compositional symbol; discrete tokens (VQ/FSQ/CDF) give a vocabulary, masked modeling, and any-to-any inference (AION-1) but throw away exactly the fine-scale detail turbulence needs. The wave–particle proposal **refuses to choose**: maintain *both* representations of every region simultaneously — a **continuous "wave" token** (phase, smooth gradients, spatial correlation) and a **discrete "particle" token** (a localized symbolic index) — and use each where it is strong. The name is not a metaphor for physics by accident: physical fields genuinely have a *wave* description (Fourier modes, PDEs) and a *particle* description (Lagrangian tracers, SPH particles), so wave–particle dual tokens are arguably *more* natural for physics than for the vision domain that proposed them.

---

## Mathematics

For each spatial region (or the whole field), produce two coupled tokens:

**Wave (continuous) token** — a continuous embedding preserving phase/gradient, e.g. a patch projection or a low-mode spectral coefficient ([[spectral-fourier-tokens]]):
$$\mathbf w = E_\text{cont}(u)\in\mathbb{R}^d\quad(\text{phase, smooth structure preserved}).$$

**Particle (discrete) token** — a quantized index over a shared codebook ([[vector-quantized-tokens]]):
$$p = \arg\min_k \|E_\text{disc}(u)-e_k\|,\quad p\in\{1,\ldots,K\}\quad(\text{symbolic, LLM-compatible}).$$

**Coupling.** The two are kept consistent by a shared mapping / mutual reconstruction so the discrete code and continuous embedding describe the *same* region:
$$\mathcal L = \|u-\text{Dec}(\mathbf w)\|^2 + \|u-\text{Dec}(e_p)\|^2 + \beta\,\|\,\mathbf w - \text{lift}(e_p)\,\|^2 ,$$
the last term aligning the particle code with the wave embedding. Downstream, the backbone can route: discrete tokens for **semantic reasoning / masked any-to-any inference / generation**, continuous tokens for **high-fidelity regression / rollout**. A physics framing: $\mathbf w$ ≈ Eulerian/spectral field description; $p$ ≈ Lagrangian/particle description.

---

## Pros

- **Resolves the continuous-vs-discrete trade-off** — fine-detail fidelity (wave) *and* shared vocabulary + masked any-to-any inference (particle) in one representation; uniquely strong on that axis of [[00-token-representation-overview]].
- **Physically motivated** — directly mirrors the wave/particle, Eulerian/Lagrangian, spectral/SPH dualities; the discrete and continuous heads can carry genuinely complementary physics.
- **Generation + understanding** — discrete tokens enable autoregressive/masked generation (inverse problems, AION-1-style fusion); continuous tokens enable accurate forward emulation — a PFM needs both ([[pfm-interface-design]]).
- **Bridges the continuum/particle gap** flagged as a major open problem in [[pfm-architecture-approaches]] — one tokenizer spanning field-based and particle-based physics.
- **Graceful degradation** — if one head is noisy (e.g. discrete quantization at a shock), the other can compensate via the coupling.

## Cons

- **Double cost & complexity** — two tokenizers, two decoders, a coupling loss; more parameters, more to tune, more failure modes.
- **Coupling is the hard part** — keeping wave and particle tokens consistent (the alignment term) is delicate; if they drift apart the representation is incoherent. The vision precedent (arXiv:2511.01593) is recent and unreplicated for physics.
- **Unproven for dynamics** — proposed for visual tokenization; no physics-FM realization. Whether the discrete head helps or just adds noise in autoregressive rollout is untested.
- **Routing decisions** — when to use which head downstream is an extra design axis (could be learned, à la MoE [[mixture-of-experts]], adding more machinery).
- **Inherits each head's weaknesses** — wave head still grid/geometry constraints (if patch/spectral), particle head still lossy.

---

## Why this is a *better* alternative

It is the only representation in this folder that **dissolves rather than trades off** the continuous-vs-discrete dilemma — letting a single PFM do high-fidelity forward emulation (continuous) *and* symbolic masked inference / generation for inverse and fusion problems (discrete). For a model that must serve [[pfm-interface-design]]'s full task set (forward, inverse, UQ, fusion), having both token types natively is arguably necessary; the open question is whether the coupling cost is worth it versus running two specialized models.

**[AI Inference]:** The wave/particle split maps cleanly onto **Eulerian/Lagrangian duality**, the central representational choice in CFD. A PFM with wave tokens (Eulerian field) + particle tokens (Lagrangian tracers) could *natively* switch frames — Eulerian for boundary-dominated flows, Lagrangian for free-surface/multiphase/mixing — addressing exactly the regime where [[gns-graph-network-simulators]] (Lagrangian) and grid emulators (Eulerian) each fail. This is the strongest physics argument for the dual representation.

**[AI Inference]:** Combine with [[physics-conditioned-query-tokens]]: use *physics-conditioned continuous queries* as the wave head and a *VQ/FSQ codebook* as the particle head, coupled by mutual reconstruction. That stacks resolution-invariance + interpretability (wave) with vocabulary + masked inference (particle) — covering six of the eight scorecard criteria in one tokenizer, at the cost of being the most complex option on the table.

**[AI Inference]:** Diffusion emulators ([[latent-diffusion-physics]], [[arch-diffusion-backbone]]) want continuous latents; masked-modeling FMs ([[aion-1-astronomy]]) want discrete tokens. A dual tokenizer is the natural shared front-end for a **hybrid diffusion-and-masked PFM** — continuous wave latent feeds the diffusion head, discrete particle tokens feed the masked-inference head, jointly trained.

---

## See also

- [[00-token-representation-overview]] — hub; the continuous-vs-discrete axis
- [[multimodal-tokenization]] — wave–particle dual tokenization origin (arXiv:2511.01593)
- [[vector-quantized-tokens]] — the particle (discrete) head
- [[spectral-fourier-tokens]] / [[patch-embedding-tokens]] — candidate wave (continuous) heads
- [[physics-conditioned-query-tokens]] — physics-meaningful wave head
- [[gns-graph-network-simulators]] — Lagrangian/particle physics
- [[pfm-architecture-approaches]] — the continuum/particle gap this addresses
- [[aion-1-astronomy]] / [[latent-diffusion-physics]] — discrete-masked vs. continuous-diffusion consumers
