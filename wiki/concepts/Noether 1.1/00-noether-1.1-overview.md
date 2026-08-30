# Noether 1.1 — Architecture Overview

**Type:** Concept — model architecture, overview / hub page (folder: Noether 1.1)
**Status:** Design. The second-generation architecture, revised after the [[noether-1.0-rbc]] as-built report exposed a systemic failure mode (predictions collapse toward a smoothed near-copy of the input) and a loss of generality (the build became a bespoke incompressible-fluid solver rather than the equation-agnostic model [[noether-1.0]] set out to be). This page is the hub: it states the design thesis, the end-to-end forward pass, and links one page per model portion. **No training has run on this revision** — treat every number inherited from [[noether-1.0]] as a design estimate.
**Related Concepts:** [[graph-tokenizer-1.1]], [[field-token-streams-1.1]], [[field-descriptors-1.1]], [[edge-generation-1.1]], [[conditioning-and-constants-1.1]], [[backbone-1.1]], [[decoder-1.1]], [[discovered-conservation-1.1]], [[post-decoder-diffusion-1.1]], [[training-scheme-1.1]], [[mixture-of-experts]], [[noether-1.0]], [[noether-1.0-rbc]], [[open-architectural-problems]]
**Related Summaries:** [[gns-graph-network-simulators]], [[poseidon-pde-foundation-model]], [[latent-diffusion-physics]], [[dynami-cal-graphnet]]

---

## Why a 1.1 at all — the thing 1.0-RBC got wrong

Noether-1.0-RBC ([[noether-1.0-rbc]]) trained, stayed stable, passed its structural gates — and learned almost nothing dynamical. The rollout keeps the initial frame nearly unchanged: the velocity **spectrum matches** while rollout-mean Nusselt is **under-predicted by 15–21%** (§6.6 there), the signature of a low-pass smoother that preserves the coarse energy envelope and kills the transport-carrying fine structure. Three compounding causes:

1. **A near-identity target.** Frames are sampled $\Delta t_{\text{out}}=0.004$ apart specifically to stay correlated, so $u_{t+1}\approx u_t$. A residual head + contractive backbone + heavy stability regularization makes "copy, lightly smoothed" the global optimum of the single-step loss.
2. **An information-starved representation.** One scalar field-value per sample point, minimal tokens, minimal edges, minimal (2-frame) context, per-field latents **summed** into one node token. The input *and* the target are near-minimal — there is almost nothing to recognize.
3. **Domain-specific hard constraints.** Div-free-by-construction, an attractor-energy projection, mandatory momentum-antisymmetry — walls that both narrow generality (they assume incompressible fluid) and act as damping biases reinforcing (1).

## The 1.1 thesis: a pattern recognizer, not a solver

A solver wants the *minimal* state that determines the next state. A pattern recognizer wants an **over-complete, redundant representation** with slack for gradients and generalization. Noether 1.1 commits to the latter, on four axes:

- **Redundant features** — nodes carry fields *and* derived redundant quantities (gradients, vorticity, strain), not one raw value.
- **Richer connectivity** — three edge families including a *learned* long-range generator ([[edge-generation-1.1]]), not one thin radius graph.
- **Separated modalities** — one token *stream per field* on a shared graph ([[field-token-streams-1.1]]); coupling is explicit learned cross-field edges, not lossy summation.
- **Constraints as discovered dials, not hard-coded walls** — the model *learns which quantities are conserved* and enforces each in proportion to how invariant it actually is in data ([[discovered-conservation-1.1]]).

The mental model: **a multimodal graph transformer** where the "modalities" are physical fields (and particle species) and "cross-modal attention" is physical coupling. This is the same architecture for continuum fields *and* N-body particle systems — generality is a first-class constraint, not an afterthought, so nothing in the core is allowed to assume "fluid."

## Portion pages (one per model stage)

Each page below follows the same four-section contract: **intuition → mathematics → physics relevance → how fundamental constants enter.**

| Stage                                         | Page                               | What changed from 1.0                                                                                                                                                                          |
| --------------------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0. Conditioning & constants                   | [[conditioning-and-constants-1.1]] | Unified treatment of *all* physical constants (scalar + vector); nondimensionalization + conditioning tokens + covariant channels                                                              |
| 1. Graph tokenizer                            | [[graph-tokenizer-1.1]]            | Resolution-free node encoder; **redundant derived node features**; dual smooth/high-freq path retained                                                                                         |
| 2. Field token streams                        | [[field-token-streams-1.1]]        | **NEW** — one token per field per site on a shared graph; no summation                                                                                                                         |
| ↳ 2b. Field descriptors & conditioned queries | [[field-descriptors-1.1]]          | **NEW substep** — learned per-field descriptor $\theta_f$ (type + force affordances) concatenated into each token; **generates simulation-conditioned query banks** $Z_q^f$ (Problem 8 / S8.4) |
| 3. Edge generation                            | [[edge-generation-1.1]]            | **NEW dedicated page** — local same-field KNN + local cross-field KNN (smaller K) + a **learned long-range edge model**                                                                        |
| 4. Backbone                                   | [[backbone-1.1]]                   | Symmetric+skew port-Hamiltonian core; momentum-antisymmetry now a **gateable channel**, not mandatory                                                                                          |
| 5. Decoder                                    | [[decoder-1.1]]                    | **General increment head is the default** (particle-compatible); div-free is *not* hard-wired                                                                                                  |
| 6. Discovered conservation                    | [[discovered-conservation-1.1]]    | **NEW** — learned invariants $\{C_k\}$; threshold-gated-hard + learned-soft enforcement; local (div-free) discovery first                                                                      |
| 7. Post-decoder diffusion                     | [[post-decoder-diffusion-1.1]]     | **NEW dedicated page** — generative fill-in of the high-frequency / gap band the deterministic decode drops                                                                                    |

## End-to-end forward pass (one step)

```
INPUT: multi-field state — a short context trajectory of the last n frames — on a grid/mesh
       OR a set of particles, + physical constants {c_k}, Δt
       (history enters as temporal-derivative channels on each token; see graph-tokenizer-1.1)

0. CONDITION      nondimensionalize by characteristic scales built from {c_k};
                  z_cond = MLP(log-scaled scalar constants) ⊕ covariant(vector constants)   [conditioning-and-constants-1.1]
1. TOKENIZE       partition → nodes at centroids/particles; per node, resolution-free
                  query-encode a redundant feature set (fields + gradients + vorticity…)     [graph-tokenizer-1.1]
2. FIELD STREAMS  emit ONE token per field per node (not one summed token)                    [field-token-streams-1.1]
2b. DESCRIPTORS   append per-field descriptor θ_f (type + force affordances); θ_f + z_sim
                  generate the field's query bank Z_q^f (sim-conditioned tokenizer)            [field-descriptors-1.1]
3. BUILD EDGES    same-field KNN  ∪  cross-field KNN (smaller K)  ∪  learned long-range edges  [edge-generation-1.1]
4. BACKBONE       L layers of symmetric+skew attention on the typed multigraph;
                  momentum-antisymmetry channel gated per edge family                          [backbone-1.1]
5. DECODE         coordinate-conditioned increment per field (general head; particle head);
                  NO hard div-free projection                                                  [decoder-1.1]
6. CONSERVE       evaluate candidate invariants {C_k}; apply soft penalty (learned λ_k) +
                  hard projection only where measured drift < ε                                [discovered-conservation-1.1]
7. DIFFUSE        generative fill-in of the high-k residual band the decode averaged out        [post-decoder-diffusion-1.1]

OUTPUT: next multi-field state (or next particle kinematic state)
```

Stages 0–5 are deterministic; stage 6 is a correction; stage 7 is generative. Stages 3, 6, 7 are the load-bearing 1.1 additions.

## Design invariants (must hold across every portion)

1. **Nothing in the core assumes a specific equation or a specific field set.** Fluid-specific facts (incompressibility, buoyancy) may be *discovered* but never *hard-coded*.
2. **Continuum and particle inputs produce the same kind of node/token** — one backbone, no fork ([[graph-tokenizer]]'s continuum–particle bridge, carried forward).
3. **Redundancy is a feature.** Where a choice is between minimal-and-structured vs. redundant-and-expressive, 1.1 chooses redundant.
4. **Physical constants enter every stage explicitly** and their handling is documented per page (§"fundamental constants" everywhere).

## Shared vs. specialized parameters — and when 1.1 becomes an MoE

A recurring design question, given how many small MLPs / weight matrices the model contains: which are *universal* (one set across all physics) and which are *specialized* per field or regime? The committed answer is **almost everything is shared weights $+$ conditioning; virtually nothing is stored per-regime.** The backbone operators, the learned edge scorer, the conservation candidate list, and the diffusion denoiser are each a *single shared function* modulated by $z_{\text{cond}}$ / $z_{\text{sim}}$; the only per-*field* (not per-regime) objects are the descriptors $\theta_f$, the field-typed encoders/decoders, and the query-bank correction — and even those are a shared base $+$ a conditioned correction ([[field-descriptors-1.1]] option B2), never a per-regime lookup table (B1), because a lookup table discards the cross-regime transfer that *is* the foundation-model premise.

The litmus test for how far to specialize any given module: **does its operator change *form* across regimes, or only *magnitude*?** Magnitude-change (buoyancy is buoyancy, just stronger at high $\mathrm{Ra}$) → keep it shared and condition it. Form-change (a shock-reading vs. a smooth-Stokes feature extractor; turbulent vs. laminar diffusion *texture*) → a routed basis is defensible, and that is where an explicit **mixture-of-experts** earns its cost: the query banks (B4 basis-mixture) and possibly the diffusion head — *not* the backbone or the conservation machinery, since specializing those would fragment the physics and kill transfer.

**[AI Inference]:** the model is already MoE-*flavoured* without a router. [[multihead-attention]] frames heads as "a basis of coupling operators composed per regime" — a dense/always-on expert set — and $z_{\text{sim}}=\operatorname{pool}_f\theta_f\Vert z_{\text{cond}}$ is *already a soft routing signal*. Making the mixing sparse/discrete (a top-$k$ over $z_{\text{sim}}$) is therefore a capacity/FLOP decision layered on infrastructure that already exists, not a change to the generality story. Being comfortable with an MoE-like model is fully consistent with 1.1 — the only discipline is to keep every expert set as *shared base $+$ sparse correction*, never disjoint per-regime experts. See [[mixture-of-experts]]; the parameter-sharing view of the conservation set is [[discovered-conservation-1.1]]'s $\sigma_k(z_{\text{cond}})$ (a conditioned function, not a stored table).

## See Also
- [[noether-1.0]] — the 1D-Burgers sibling and sizing formulas this inherits
- [[noether-1.0-rbc]] — the as-built report whose failure modes motivated this revision
- [[open-architectural-problems]] — the design-rationale ledger (Problems 4/7/8/9/10 touched here)
- [[training-scheme-1.1]] — what gets trained in what order (modules) and on which physics regimes (curriculum)
- [[pattern-recognizer-vs-solver]] — the design thesis, if broken out as its own page later (link seeded)
- [[00-atlas-0.1-overview]] — a parallel architecture track (not a successor): agents/typed-edges/experts with *imposed* conservation, vs. this page's single-backbone *discovered*-conservation thesis
