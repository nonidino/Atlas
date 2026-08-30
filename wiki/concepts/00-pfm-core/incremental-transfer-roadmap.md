# Incremental Transfer Roadmap: Bootstrapping the Regime-MoE from Existing Weights

**Type:** Core Concept — Build Strategy
**Related Concepts:** [[regime-moe-architecture]], [[pfm-purpose-and-direction]], [[transfer-learning-fine-tuning]], [[00-noether-1.1-overview]], [[edge-generation-1.1]], [[graph-tokenizer-1.1]], [[backbone-1.1]], [[mixture-of-experts]], [[code-and-papers]]

---

## The problem this page solves

[[00-noether-1.1-overview]] was built as a fully bespoke architecture — every component (tokenizer, edge generation, backbone, decoder, discovered conservation) designed and trained from scratch as one coherent design. That was the right call when the goal was validating a specific set of design commitments end-to-end. It is not the right call under the current constraint: **limited time and compute, with a target ([[pfm-purpose-and-direction]]'s five worked examples) that spans far more regimes than one from-scratch training run can plausibly cover well.**

The proposed shift: **bootstrap [[regime-moe-architecture]] from existing pretrained models, expert by expert, and progressively replace each donor's internals with native components** — rather than training one dense from-scratch backbone (current 1.1 path) or training an entirely new MoE from scratch (naive reading of [[regime-moe-architecture]]). This page is the concrete "how," reconciled against what Noether 1.1 has already built.

**Explicitly not a decision to abandon Noether 1.1.** Section "What Noether 1.1's build already gives this roadmap" below argues most of the completed engineering work is directly reusable as the native scaffold this strategy needs.

---

## What's actually available to transfer (grounded in [[code-and-papers]])

Transfer only works where weights exist. Checking every candidate donor model against the wiki's own resource catalog:

| Model | Confirmed public checkpoint? | What it can donate |
|---|---|---|
| [[poseidon-pde-foundation-model]] | **Yes** — [camlab-ethz/poseidon](https://github.com/camlab-ethz/poseidon), HF `camlab-ethz` | **Weights.** Multiscale scOT transformer, continuous lead-time conditioning, pretrained on a 6-operator fluid set with demonstrated OOD transfer to 15 downstream PDEs. Best current candidate for the elliptic/continuum expert family. |
| [[walrus-paper]] | **Yes** — [PolymathicAI/walrus](https://github.com/PolymathicAI/walrus), HF `polymathic-ai/walrus` | **Weights.** Broadest demonstrated diversity (19 scenarios, 63 fields) of any donor here — best candidate for a generalist/fallback expert or for initializing the shared conditioning trunk. |
| [[gns-graph-network-simulators]] | Code yes ([deepmind-research](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate)), pretrained weights for arbitrary new domains **not confirmed** | **Structural pattern only** — message-passing particle-graph architecture, reimplemented rather than weight-transferred. |
| [[gphyt-physics-foundation-model]] | *(find)* — unconfirmed in [[code-and-papers]] | **Structural pattern only** — derivative-prediction + numerical-integrator head, reimplemented from the paper. |
| [[dynami-cal-graphnet]] | *(find)* — unconfirmed | **Structural pattern only** — antisymmetric edge-local reference frames for exact momentum conservation; this is a closed-form architectural constraint, not a learned representation, so it doesn't need a checkpoint to reuse — it needs correct reimplementation. |
| [[aion-1-astronomy]] | *(find)* — unconfirmed | **Structural pattern only** — modality-specific tokenization scheme, not weights (astronomy-domain weights wouldn't transfer to continuum PDEs anyway). |
| [[pisd-physics-informed-spectral-diffusion]] | *(find)* — unconfirmed | **Structural pattern only** — DPS test-time guidance mechanism, relevant to the abstain/UQ mechanism (Gap 11, [[pfm-purpose-and-direction]]). |

**Actionable conclusion:** of the seven candidate donors, only **two (Poseidon, Walrus) support an actual weight-transfer bootstrap today**. Everything else in the "toward a hybrid architecture" list across [[pfm-architecture-approaches]] and [[pfm-concept-overview]] is a *structural pattern* to reimplement, not a checkpoint to load. This should directly determine what gets attempted first: **Poseidon → elliptic/continuum expert** and **Walrus → generalist/fallback expert** are the only two stage-1 moves that don't require a from-scratch training run before anything is running end-to-end. First action item, not yet done: verify current license terms and checkpoint compatibility (input channel count, resolution assumptions) for both before committing engineering time.

---

## What Noether 1.1's build already gives this roadmap

The native scaffold this strategy needs — a shared conditioning contract and an agent/edge graph interface — **already exists** in [[00-noether-1.1-overview]]'s design, and substantial parts are already implemented and tested (see [[implementation-log]]):

- **[[pfm-interface-design]]'s dimensionless-conditioning-token contract** (Re, Ma, Kn, …) is exactly the routing/conditioning signal [[regime-moe-architecture]] calls for. Already built.
- **[[graph-tokenizer-1.1]] + [[edge-generation-1.1]]** — resolution-free node encoding plus same-field/cross-field KNN + learned long-range edges — is structurally the same thing as "agents interact through edges instantiated by proximity" in [[pfm-purpose-and-direction]]. Already built, already has a real fix history (density/OOM, edge-cutoff scaling — see [[implementation-log]]'s 2026-07-15/07-19 entries), which a from-scratch MoE build would have to rediscover.
- **[[backbone-1.1]]'s Port-Hamiltonian symmetric+skew core** is the piece that changes under this roadmap — not because it's wrong, but because it's the one component being proposed for replacement-by-donor-then-gradual-nativization, per family.

**Net effect of this framing:** the incremental-transfer roadmap is not a rewrite. It inserts a **regime router between the existing tokenizer/edge layer and the backbone**, and lets what sits behind that router start as wrapped donor models before converging toward something like [[backbone-1.1]]'s native design, expert by expert. The tokenizer, edge generation, and conditioning contract do not need to change to support this.

---

## The four-stage bootstrap

### Stage 1 — Frozen donor wrap (fastest path to an end-to-end running system)

Insert the coarse MoE gate (dimensionless-number-driven, per [[regime-moe-architecture]]) directly after Noether 1.1's existing tokenizer/edge layer. Behind each gate output, wrap a **frozen** donor model (Poseidon for elliptic/continuum, Walrus for generalist) behind a thin adapter that projects the shared graph/token representation into the donor's native input format and back.

This is transfer-learning Level 0/1 in [[transfer-learning-fine-tuning]]'s existing hierarchy — closest to "fully importing" the model — deliberately accepted as an interim bootstrap step, not the end state. It's the fastest way to get a system that can attempt any of [[pfm-purpose-and-direction]]'s five worked examples at all, even badly, which is more valuable right now than a slower path to something better.

**Gate before proceeding:** does the wrapped system beat a single native dense baseline (or beat nothing, if no baseline exists yet) on at least one worked example end-to-end? If not, the adapter interface is wrong and needs fixing before any further stage is worth attempting.

### Stage 2 — Adapter fine-tuning (Level 2 transfer)

Unfreeze each donor via LoRA adapters (per [[transfer-learning-fine-tuning]]'s existing Level 2), fine-tuned per regime on whatever data is cheapest to generate for that regime (classical-solver-generated trajectories, or [[genesis-physics-engine]] where applicable). This starts adapting the donor's internal representation *toward* the shared token space, instead of forcing the shared token space to imitate the donor's native format via the Stage 1 adapter alone.

Cost note: this is the cheapest possible fine-tuning path (LoRA, not full fine-tune) precisely because resources are constrained right now — full fine-tuning is a later-stage option once a specific expert is shown to be worth the investment.

### Stage 3 — Structural distillation, submodule by submodule

For donors without usable checkpoints (Dynami-CAL, GP$_{\text{hy}}$T), reimplement their specific structural trick as a native module and swap it in behind the same expert slot: Dynami-CAL's antisymmetric edge frames for the particle/many-body expert (battery cells, docking dynamics — worked examples 1 and 3 in [[pfm-purpose-and-direction]]), GP$_{\text{hy}}$T's derivative-prediction + integrator head for whichever expert is showing the worst rollout error accumulation.

**Regression gate per swap:** each submodule replacement must be validated not to regress the LoRA-adapted baseline from Stage 2 before being kept — same discipline [[implementation-log]] already applies to Noether 1.1's own optimization passes (e.g., the 2026-07-19 bit-for-bit-identical verification after the attention/backbone speedups). This is what "bit by bit, more native" concretely means: one swap, one regression check, repeat — not a scheduled rewrite.

### Stage 4 — Shared trunk unification

Once enough experts have converged on native submodules sharing the same interface, attempt a shared trunk with late-layer specialization (resolving [[regime-moe-architecture]]'s open question empirically, per-project, rather than by a priori argument) and jointly fine-tune for the cross-regime transfer that a fully disjoint-experts design would forgo. Only attempt this once individual experts are independently validated — a premature shared trunk risks exactly the negative-transfer failure mode [[regime-moe-architecture]] flags for forcing structurally incompatible theories (e.g. Schrödinger-type and Einstein-type experts) to share early layers.

---

## Which worked example is reachable at which stage

Rough calibration against [[pfm-purpose-and-direction]]'s five worked examples — not a commitment, a planning aid:

- **Stage 1 (frozen wrap):** none of the five worked examples are safely attemptable yet — Stage 1's only goal is validating the router+adapter interface works at all, ideally on a single-regime toy case, not a full worked example.
- **Stage 2 (LoRA):** the wind farm example (worked example 5) is the best first real target — it's dominated by continuum fluid-structure and EM regimes that Poseidon's pretraining already covers reasonably, and it has no Gap 10 (composability) or Gap 11 (safety-critical abstention) requirement to solve first.
- **Stage 3 (structural distillation):** the moon/mars base and battery examples (1, 3) become reachable once the particle/many-body expert has Dynami-CAL's conservation trick natively — both need exact momentum/energy conservation at agent interfaces for physical plausibility, which a frozen or LoRA-adapted continuum-only donor cannot provide.
- **Stage 4 (shared trunk):** the reentry example (4) is the hardest target in the set — it needs smooth cross-regime transfer *within a single agent* (Knudsen-driven, sub-object resolution) which fundamentally requires the shared-trunk transfer this stage is built to test. Treat it as the validation case for whether Stage 4 was worth doing, not an early target.
- **Wildfire (2)** is deliberately last in this ordering despite being worked-example #2 in [[pfm-purpose-and-direction]] — it needs both Gap 9 (cross-boundary consistency, radiative reach isn't a simple adjacency edge) and Gap 11 (abstain/fallback, since ignition prediction is safety-critical) solved, neither of which any stage above addresses head-on. It may need its own dedicated design work beyond this roadmap.

---

## Where this led

This roadmap's donor-availability findings and 4-stage staging are now applied concretely in **[[00-atlas-0.1-overview]]** (the named architecture) via **[[training-and-bootstrap-atlas-0.1]]**, and worked through end-to-end for a specific first target in **[[case-study-rocket-ascent-2d-atlas-0.1]]**.

## Open questions

1. Are Poseidon's and Walrus's licenses actually compatible with this project's intended use (research vs. eventual deployment)? Not yet checked — first action item before Stage 1 engineering starts.
2. What's the adapter cost in practice — does projecting Noether 1.1's graph/token representation into Poseidon's grid-tied scOT format lose enough structure to make Stage 1 not worth doing at all? Unknown until attempted on a toy case.
3. Gap 9 (cross-boundary consistency) and Gap 11 (verified fallback) from [[pfm-purpose-and-direction]] aren't addressed by any stage above — they may need dedicated architecture work independent of this transfer roadmap, not something that falls out of donor bootstrapping.

---

## See Also

- [[regime-moe-architecture]]
- [[pfm-purpose-and-direction]]
- [[transfer-learning-fine-tuning]]
- [[00-noether-1.1-overview]]
- [[edge-generation-1.1]]
- [[graph-tokenizer-1.1]]
- [[backbone-1.1]]
- [[mixture-of-experts]]
- [[implementation-log]]
- [[poseidon-pde-foundation-model]]
- [[walrus-paper]]
- [[dynami-cal-graphnet]]
- [[gphyt-physics-foundation-model]]
- [[00-atlas-0.1-overview]]
- [[training-and-bootstrap-atlas-0.1]]
