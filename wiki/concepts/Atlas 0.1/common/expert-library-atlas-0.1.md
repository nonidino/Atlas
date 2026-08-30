# Atlas 0.1: Expert Library

**Type:** Core Concept — Model Portion (central design decision)
**Related Concepts:** [[00-atlas-0.1-overview]], [[port-algebra-atlas-0.1]], [[regime-moe-architecture]], [[mixture-of-experts]], [[edge-generation-atlas-0.1]], [[training-and-bootstrap-atlas-0.1]], [[case-study-rocket-ascent-2d-atlas-0.1]]

---

## The organizing question

*(The type vocabulary was closed and rebuilt as effort–flow **ports** on 2026-08-19 — [[port-algebra-atlas-0.1]] — which independently confirms this page's conclusion: `pressure`, `stress` and `shear` turned out to be three names for one traction, so "one expert per label" would have built three experts for one quantity. The argument below is unchanged and is now also structurally enforced.)*

Given a typed edge list, the naive design is one expert per type: a "heat expert," a "pressure expert," and so on, each independently predicting its quantity and combining post-hoc. **This is the wrong cut.** The right organizing axis is **governing-equation family** — the same elliptic/hyperbolic/parabolic distinction that is finding 1 of [[regime-moe-architecture]] — because that is where the physics is actually coupled, and edge-type labels routinely span quantities that are outputs of a single governing system, not independent phenomena.

## The worked contrast that motivates this

Consider two of the rocket's edges ([[case-study-rocket-ascent-2d-atlas-0.1]]):

- **$a\!-\!b$ and $e\!-\!b$ (heat + pressure, reaction/outflow → chamber):** heat and pressure at these interfaces are both outputs of the *same* compressible reacting-flow governing system — the energy and momentum equations solved together. A "heat expert" and a "pressure expert" that don't share internal state would have to agree with each other after the fact, which is strictly worse than one expert that outputs both because it solved the coupled system directly. **These two type-labels are two output channels of one expert, not two experts.**
- **$b\!-\!c$ (heat + stress, chamber wall → airframe):** heat conduction and structural stress *are* different governing equations — parabolic diffusion vs. quasi-static elasticity — weakly coupled only through thermal expansion. Engineering practice usually solves these as two separate analyses passing a coupling term between them. **This edge legitimately wants two experts talking to each other.**

The rule that falls out: **experts are cut along governing-equation boundaries, and an edge's declared types are checked against that cut — types that come from the same governing system share an expert; types that come from different governing systems get routed to different experts that exchange the coupling term explicitly.**

## The rocket's expert library

| Expert | Governs | Agents / edges it covers | Locality (per [[regime-moe-architecture]] finding 1) |
|---|---|---|---|
| **Reacting / internal compressible flow** | Combustion chemistry + confined duct flow | $a$, interior of $b$, $e$ — covers $a\!-\!b$ and $e\!-\!b$ as two channels of one expert | Elliptic-ish near the injector → hyperbolic past the throat, within the *same* expert, conditioned continuously on local Mach number |
| **External compressible flow / plume** | Aerodynamics + free-jet exhaust | fluid channel of $c\!-\!d$, $f$, $g\!-\!f$ — plausibly the same governing-equation family as the expert above (compressible NS/Euler), differently conditioned by a confined-vs-free boundary flag (open question, below) | Subsonic (near-elliptic) → supersonic (hyperbolic) as flight speed and altitude change |
| **Thermal-structural** | Conduction + stress + thermal expansion in solid material | $b\!-\!c$, stress channel of $c\!-\!d$ | Parabolic diffusion coupled to quasi-static elasticity |
| **Rigid-body trajectory** | Newtonian dynamics (3-DOF in 2D: $x, y, \theta$) | the whole-vehicle bottleneck level of [[unet-hierarchy-atlas-0.1]], not any single declared edge; integrates the net force/torque produced by every expert above, plus gravity | Global by construction — this is not a field at all, it's an ODE |

**Conservation is not a fifth expert.** The $e\!-\!f$ and $d\!-\!g$ edges are flux-continuity constraints, not new physics to compute — see [[conservation-as-constraint-atlas-0.1]] for the dedicated treatment.

## Physics-encoding-spectrum placement

Per the encoding spectrum in this vault's top-level conventions (data-driven → soft loss → test-time guidance → architectural soft bias → hard architectural constraint), the four experts above sit at different points, deliberately:

- **Rigid-body trajectory** is a strong candidate for **level 5 (hard architectural constraint)** — Newtonian 3-DOF integration in 2D is exact, closed-form classical mechanics with no learned component needed. There is no reason to spend model capacity learning something with a closed form.
- **Reacting/internal flow, external flow/plume, thermal-structural** have no closed form at the fidelity needed and sit at levels 1–4, to be decided per-expert during training (soft loss vs. architectural bias) — see [[training-and-bootstrap-atlas-0.1]].

## Open question: is internal duct flow the same expert family as external aero, differently conditioned?

Physically, both are compressible NS/Euler — the argument for merging them into one expert with a confined-vs-free boundary-condition flag (the same pattern as the existing Re/Ma conditioning contract) is strong. But donor-model reality cuts the other way: per [[incremental-transfer-roadmap]], Poseidon and Walrus were both pretrained on external/free-boundary-style data, with nothing resembling confined internal duct flow. Bootstrapping ([[training-and-bootstrap-atlas-0.1]]) will likely start these as two separately-initialized experts regardless of which is architecturally "correct," and the merge question can be revisited once both have real trained weights to compare.

## Interaction with the MoE gate in the general (non-Stage-1) architecture

In the full [[regime-moe-architecture]] vision, expert selection is driven by a dimensionless-number gate. Atlas's Stage 1 doesn't need that gate — the expert-per-agent/edge assignment is fixed by the declared graph and the table above, matching [[edge-generation-atlas-0.1]]'s "declared, not discovered" scope reduction. A learned gate becomes relevant again once agents/edges themselves become less fully declared (e.g., after RL edge instantiation is introduced, per [[edge-generation-atlas-0.1]]) or once a scenario introduces an agent whose regime isn't known in advance.

---

## See Also

- [[composition-error-theory]] — boundary-condition flexibility sharpened into a selection axis: how far up the Dirichlet-to-Neumann ladder an expert's interface can go
- [[00-atlas-0.1-overview]]
- [[regime-moe-architecture]]
- [[mixture-of-experts]]
- [[edge-generation-atlas-0.1]]
- [[unet-hierarchy-atlas-0.1]]
- [[training-and-bootstrap-atlas-0.1]]
- [[case-study-rocket-ascent-2d-atlas-0.1]]
- [[conservation-as-constraint-atlas-0.1]]
- [[plug-in-composition-theorems]] — **what a library entry must carry beyond a checkpoint**: a conformance certificate bound to the weight hash and the probe state, since nothing currently checks that a capability record is true. It also gives the library its **routing criterion** — rank by composability $\Xi$, not by benchmark accuracy — and shows that a composed cluster is itself a legal library entry
- [[interface-transfer-theory]] — the interface declaration a library entry publishes per seam: one prolongation $P_i$, from which the mapping class and the reduction are derived
