# Regime-Conditioned MoE: A Multi-Axis Alternative to a Single Backbone

**Type:** Core Concept — Architecture Proposal (competing with, not yet reconciled against, [[00-noether-1.1-overview]])
**Related Concepts:** [[pfm-architecture-approaches]], [[mixture-of-experts]], [[pfm-interface-design]], [[physics-foundation-models]], [[partial-differential-equations]], [[navier-stokes-equations]], [[symmetric-attention-physics]], [[equivariant-gnns]], [[00-token-representation-overview]], [[training-scheme-1.1]]

---

## Status and provenance

This page files a synthesis produced in a separate design conversation, working from first principles across ~20 candidate physics regimes (8 deep-dived: Newtonian N-body, turbulent incompressible flow, compressible/shock flow, many-body electronic structure, kinetic/rarefied/plasma, strong-field GR, radiative EM, plasticity/fracture). It has **not** been reconciled against the concrete, partially-built [[00-noether-1.1-overview]] / [[noether-1.1-medium]] architecture — see [[implementation-log]] for the current build state. Per user decision (2026-08-07), this is filed as a documented alternative for future evaluation; it does not change Noether 1.1's status.

This architecture now has a concrete deployment target ([[pfm-purpose-and-direction]]'s multi-agent worked examples) and a build strategy ([[incremental-transfer-roadmap]]) for reaching it without a from-scratch training run.

**Core claim:** a single dense backbone with soft conditioning (Noether 1.1's approach) may be structurally wrong for the breadth a true PFM needs. The better decomposition is a **large mixture-of-experts model**, but critically, **not one gate for everything** — the regime survey below argues that at least three independent structural axes get collapsed into "one big regime label" if you're not careful, and each wants a different mechanism.

---

## Five findings from the regime survey

### 1. Local vs. global tracks equation *type*, not physics *domain*

The recurring structural split is not "fluids vs. EM vs. gravity" — it's **elliptic/constraint vs. hyperbolic/wave vs. parabolic**, and this pattern repeats verbatim across unrelated domains:

- **Elliptic / constraint (needs global coupling):** pressure Poisson equation in incompressible flow, near-field electrostatics/magnetostatics, near-field gravity, gauge-fixing constraints in EM/GR. All of these are instantaneous-in-space: a source anywhere affects the field everywhere on the current time slice.
- **Hyperbolic / wave (genuinely local/causal):** radiative EM (far field), shock propagation in compressible flow, seismic/acoustic waves, strong-field GR wave content. Information propagates at a finite speed; the correct inductive bias is causal/windowed attention or local message-passing, not global mixing.
- **Parabolic (diffusion, in between):** heat conduction, viscous diffusion, some plasma transport regimes — effectively local at short range with rapidly decaying long-range influence.

**Implication for tokenization/attention:** [[00-attention-overview]]'s catalog (windowed/causal vs. global/spectral variants) should be selected **per-region by equation type**, not fixed once for the whole model or tied to which physical domain is active. A compressible-flow region with an embedded near-incompressible pocket (e.g., a stagnation zone) genuinely needs both attention patterns simultaneously.

### 2. Regime crossovers are either smooth or threshold-like — and conflating them breaks outputs

- **Smooth crossovers:** continuous dimensionless ratios — Knudsen number (continuum ↔ kinetic), Mach number (subsonic ↔ transonic ↔ supersonic), $v/c$ (Newtonian ↔ relativistic), $r/\lambda$ (near-field ↔ far-field EM). These should be handled by **continuous conditioning** (FiLM / adaLN-style modulation) — [[pfm-interface-design]] already establishes the mechanism (dimensionless numbers as conditioning tokens), this just argues for pushing that same signal into feature-wise modulation of expert outputs rather than only at the tokenizer input.
- **Threshold-like crossovers:** yield surface onset (plasticity), shock formation (compressible flow), fracture nucleation, Mott metal-insulator transition (electronic structure). These are genuine discontinuities in the governing physics, not artifacts of coarse binning — the correct model output has a kink or jump there.

**Failure mode of conflating them:** a model trained with only continuous conditioning over-smooths genuine discontinuities (a diffused shock front, a fracture that "creeps" instead of nucleating). A model with only hard gating produces spurious discontinuities where physics is smooth (audible "regime-switching" artifacts as Re crosses a routing boundary). **These need separate mechanisms**, not one router doing both jobs.

### 3. "Simpler" asymptotic limits are not always cheaper to model

Weak-field GR, two-body orbits, linear elasticity, and near-field statics are the analytically clean limits *and* the computationally easy ones — good targets for early-curriculum, cheap-exact-data pretraining ([[training-scheme-1.1]]'s rung-based curriculum already uses this logic).

**Counterexamples that break the "simple limit → cheap" assumption:** hypersonic flow (the "simple" high-Mach limit is numerically stiffer, not easier, due to strong shock-boundary-layer coupling) and the Kn ~ 1 kinetic transition regime (neither the continuum nor free-molecular limit applies — this is the hardest part of the Knudsen axis, not an edge case). **A router or curriculum designer cannot assume "asymptotically simple" implies "cheap to route to first."** Any curriculum built on this MoE architecture needs per-regime empirical difficulty calibration, not an a-priori mapping from equation simplicity to training-order priority.

### 4. Existing AI methods already split along the equation-type fault line

PINNs and operator learning ([[neural-operators]], [[poseidon-pde-foundation-model]]) perform well on smooth elliptic/parabolic problems (elasticity, near-field EM, weak-field GR) and degrade on discontinuities (shocks, fracture, fine-scale turbulence) *unless* given explicit structural handling — jump conditions, equivariance ([[equivariant-gnns]]), antisymmetry ([[skew-symmetric-attention]]), or history-dependence for path-dependent plasticity. This is independent confirmation (from the existing-methods literature, not just the regime survey) that equation type is the right coarse axis, and that discontinuity-handling is a genuinely separate concern from routing.

### 5. Human experts triage via dimensionless numbers before solving anything

Physicists compute Re, Kn, $v/c$, $r/\lambda$ as a fast, largely discrete *mental* router, even though the underlying physics varies continuously. This is the strongest argument for feeding nondimensionalization ([[pfm-interface-design]]'s existing mechanism) **directly into the MoE gate's logits**, rather than expecting the router to rediscover regime boundaries from raw field values — the human-triage prior says the boundary information is already compressed into a handful of scalars, so the router shouldn't have to re-derive it from high-dimensional input.

---

## Proposed architecture: separate the axes instead of collapsing them into one gate

The regime survey's central architectural claim is that **local/global, smooth/threshold, and domain identity are three different axes**, and past MoE-for-physics thinking (including the AI Inference already logged in [[mixture-of-experts]]) tends to collapse them into a single hard/soft routing decision. This proposal keeps them separate:

1. **Coarse MoE, soft/expert-choice routing, split by field content / theory type** (Schrödinger-type, Navier–Stokes-type, Einstein-type, elasticity-type). Router logits driven **directly by dimensionless-number conditioning tokens** (Re, Kn, Ma, $v/c$, $r/\lambda$) rather than raw field content — per finding 5. Each expert family carries its own structurally-constrained output head matched to that theory's invariances: equivariance for particle/GNN-style experts ([[equivariant-gnns]]), antisymmetry for momentum-conserving experts ([[skew-symmetric-attention]]), gauge consistency for EM/GR experts, frame-indifference for continuum-mechanics experts. This is the two-level hierarchy [[mixture-of-experts]] already sketched (hard routing across families, softer adaptation within), now with the routing signal specified concretely.

2. **Fine-grained continuous conditioning (FiLM / adaLN) *within* each expert family** for smooth parameter variation (Re, Kn, $v/c$) — per finding 2's smooth branch. This is the mechanism that prevents the audible regime-switching artifact of pure hard routing: within one expert, output modulates continuously as the dimensionless number moves, with no seam.

3. **A separate discontinuity-handling mechanism**, orthogonal to the MoE gate — an auxiliary regularity/phase-field channel, or adaptive tokenization that refines near a detected threshold — for shocks, fracture, and yield surfaces (finding 2's threshold branch). This does *not* live in the router; it lives alongside it, triggered by threshold detection rather than by the coarse regime gate.

4. **Attention-pattern selection per region**, driven by the *same* local/global classification as the coarse gate (finding 1) — windowed/causal where the local equation type is hyperbolic, global/spectral where it is elliptic. Notably this can fire **within a single expert's forward pass**, since a single physical field (e.g., a compressible-flow domain with an embedded quasi-incompressible pocket) can contain both equation types at once.

5. **Hierarchical multi-resolution tokens** for the cases where regime changes *with scale itself* — the turbulence cascade (energy-containing range behaves differently from the inertial and dissipation ranges) and microstructure-to-component plasticity (grain-scale plasticity vs. component-scale elasticity). This is a scale axis, not a routing axis, but it interacts with the coarse gate: the router may need to operate per-scale-band rather than once globally.

6. **Curriculum:** pretrain on the analytically-tractable limits where cheap exact data exists, fine-tune into the expensive numerical regimes — but per finding 3, do **not** assume "analytically simple limit" implies "should be trained first because it's cheap." Concentrate evaluation budget on crossover zones specifically, since findings 2–3 both point to crossovers as where every failure mode concentrates (over-smoothing, spurious discontinuities, curriculum misordering).

### Open question: shared trunk vs. fully disjoint experts

Not resolved by the survey. A shared trunk with late-layer specialization would let the model exploit genuine cross-regime transfer (dimensionless-ratio crossovers and near/far-field patterns recur structurally across domains — see finding 1). Fully disjoint experts from layer one would give cleaner interpretability and avoid negative transfer between structurally incompatible theories (e.g., forcing a Schrödinger-type expert to share early layers with a GR-type expert). Likely resolvable only empirically, and only after the coarse-gate/conditioning/discontinuity split above is validated on a smaller regime set.

---

## Relationship to the existing wiki

- [[mixture-of-experts]] already proposed hard-routing-across-families + ICL/soft-adaptation-within-family as an *AI Inference*; this page is the concrete instantiation of that inference, with the routing signal (dimensionless numbers), the smooth/threshold split, and the local/global attention-selection mechanism specified.
- [[pfm-architecture-approaches]]'s "Toward a Hybrid Architecture" section already converges on several of the same components (universal tokenization for heterogeneity, particle-GNN module for discrete many-body physics) from a different angle (surveying existing published models rather than a first-principles regime survey). The overlap is corroborating, not redundant — [[pfm-architecture-approaches]] arrives at particle/continuum hybridization from what published models actually do; this page arrives at the same conclusion from what the equations structurally require.
- [[pfm-interface-design]]'s dimensionless-conditioning-token mechanism is a **prerequisite**, not a competing idea — this proposal is best read as "take the existing conditioning-token contract and use it as the routing signal for a much larger, expert-partitioned model," rather than a rejection of the interface design.
- This directly competes with [[00-noether-1.1-overview]]'s single-backbone-with-gated-discovery approach at the level of "how much should be one dense model vs. many specialized experts." No reconciliation has been attempted yet.

---

## Open Questions

1. Shared trunk vs. fully disjoint experts (see above) — likely the highest-leverage open question, since it determines whether this is closer to Noether 1.1 with an added gate, or a genuinely different architecture family.
2. How does the discontinuity-handling channel (item 3) interact with the discovered-conservation mechanism already designed for Noether 1.1 ([[discovered-conservation-1.1]])? A threshold event (fracture, shock formation) typically also breaks a locally-discovered conservation law — these two mechanisms may need to share signal.
3. At what regime-set size does routing overhead start to dominate expert compute, given that unlike LLM MoE (where tokens are homogeneous), here each expert's output head has a different, non-trivial structural constraint (different equivariance groups, different gauge structure)?
4. Can the coarse gate be trained end-to-end from dimensionless-number conditioning alone, or does it need direct supervision (labeled regime identity) at least early in the curriculum, given finding 3's warning that regime difficulty doesn't correlate simply with the numbers that would naively seed the gate?

---

## See Also

- [[mixture-of-experts]]
- [[pfm-architecture-approaches]]
- [[pfm-interface-design]]
- [[00-noether-1.1-overview]]
- [[physics-foundation-models]]
- [[00-attention-overview]]
- [[equivariant-gnns]]
- [[skew-symmetric-attention]]
- [[training-scheme-1.1]]
- [[discovered-conservation-1.1]]
- [[pfm-purpose-and-direction]]
- [[incremental-transfer-roadmap]]
- [[00-atlas-0.1-overview]] — the named, concretely-scoped architecture built from this page's proposal; its [[unet-hierarchy-atlas-0.1]] is the worked answer to this page's open shared-trunk-vs-disjoint-experts question
