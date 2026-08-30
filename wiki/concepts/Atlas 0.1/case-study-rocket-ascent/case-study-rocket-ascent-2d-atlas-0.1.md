# Atlas 0.1: Case Study — 2D Rocket Ascent (Plan of Action)

**Type:** Core Concept — First Instantiation / Plan of Action
**Related Concepts:** [[00-atlas-0.1-overview]], [[port-algebra-atlas-0.1]], [[agent-definition-atlas-0.1]], [[edge-generation-atlas-0.1]], [[expert-library-atlas-0.1]], [[unet-hierarchy-atlas-0.1]], [[conservation-as-constraint-atlas-0.1]], [[global-fields-and-topology-atlas-0.1]], [[training-and-bootstrap-atlas-0.1]], [[pfm-purpose-and-direction]]

---

## Status: planned, not built

This page is a **plan of action**, not an as-built report — contrast with [[noether-1.0-rbc]]'s as-built status for Noether 1.0's first instantiation. Nothing here has been implemented.

The Phase 0–5 outline below is the design-level statement; each phase now has a **detailed, self-contained implementation spec** in the `Atlas 0.1 implementation` folder — see [[00-atlas-0.1-implementation-plan]] for the master plan (geometry, agent/edge tables, timestep families, model dims, milestones M0–M7) and [[impl-atlas-0.1-phase0-scope-and-data]] · [[impl-atlas-0.1-phase1-scaffold]] · [[impl-atlas-0.1-phase2-experts]] · [[impl-atlas-0.1-phase3-integration]] · [[impl-atlas-0.1-phase4-validation]] · [[impl-atlas-0.1-phase5-expansion]] for the phases.

## Why this scenario first

A 2D rocket ascent was chosen as Atlas's first concrete target (over the moon-base, wildfire, battery, reentry, or wind-farm examples in [[pfm-purpose-and-direction]]) because it's the smallest scope that still exercises every core Atlas mechanism: a multi-agent graph with typed edges, experts spanning genuinely different governing-equation families, a global non-edge field (gravity), and a large timescale range (millisecond combustion vs. minute-scale ascent) — while deliberately **not** requiring composability (no new agent type appears mid-flight), staging (excluded from scope, see [[global-fields-and-topology-atlas-0.1]]), or safety-critical abstention (Gap 11), each of which would add a dimension of difficulty this first build doesn't need to prove the core pipeline works.

## The agent graph

Seven agents, per [[agent-definition-atlas-0.1]]'s granularity heuristic:

$$
a: \text{combustion reaction}\quad b: \text{combustion chamber}\quad c: \text{airframe}\quad d: \text{atmosphere-front}
$$
$$
e: \text{combustion outflow}\quad f: \text{plume}\quad g: \text{atmosphere-wake}
$$

## The typed edge list (final, $b\!-\!g$ removed)

*Relabelled 2026-08-19 to [[port-algebra-atlas-0.1]]'s five closed port types.*

$$
\begin{aligned}
&a \!-\! b:\ \texttt{MECH},\ \texttt{THERM},\ \texttt{ADVEC}(h_0,Y_k) &&e \!-\! f:\ \texttt{MECH},\ \texttt{THERM},\ \texttt{ADVEC}(h_0,Y_k)\
&b \!-\! c:\ \texttt{MECH},\ \texttt{THERM} &&e \!-\! b:\ \texttt{MECH},\ \texttt{THERM},\ \texttt{ADVEC}(h_0,Y_k)\
&c \!-\! d:\ \texttt{MECH},\ \texttt{THERM},\ \texttt{ADVEC}(h_0) &&g \!-\! f:\ \texttt{MECH},\ \texttt{THERM},\ \texttt{ADVEC}(h_0)\
&d \!-\! g:\ \texttt{THERM},\ \texttt{ADVEC}(h_0),\ \texttt{MECH}
\end{aligned}
$$

**The relabelling corrected four of the seven edges.** Each previously carried a `heat` label across an interface with mass crossing it, and therefore omitted advected enthalpy — heat moves both by conduction (`THERM`) and by advection (a passenger on `ADVEC`). Only $b\!-\!c$, a wall, was right as written. The old `pressure`, `stress` and `fluid` labels all collapse into `MECH`, being the isotropic part, the deviatoric part, and the traction of one stress tensor. The old `conservation` label disappears: every port conserves by construction, so it was naming a property, not a quantity.

**Also newly visible:** the rocket graph has **no `ROT` or `ELEC` ports at all**, which is a correct statement about a solid-propellant vehicle and a useful contrast with the wind farm and with rung 7 of [[f1-pathmap-and-end-goal]].

See [[edge-generation-atlas-0.1]] for why $b\!-\!g$ was dropped (base-region heating should route through $f$/$c$, not a direct chamber-to-wake-atmosphere shortcut).

## The expert library

Full reasoning in [[expert-library-atlas-0.1]]; summarized:

| Expert | Covers | Bootstrap ([[training-and-bootstrap-atlas-0.1]]) |
|---|---|---|
| Reacting/internal compressible flow | $a$, interior $b$, $e$ | Train from scratch |
| External compressible flow / plume | fluid channel of $c\!-\!d$, $f$, $g\!-\!f$ | Bootstrap from Poseidon |
| Thermal-structural | $b\!-\!c$, stress channel of $c\!-\!d$ | Bootstrap from Poseidon |
| Rigid-body trajectory (bottleneck) | whole-vehicle level, plus gravity ([[global-fields-and-topology-atlas-0.1]]) | Not learned — closed-form |

Conservation at $d\!-\!g$ and $e\!-\!f$ is a flux-matching constraint ([[conservation-as-constraint-atlas-0.1]]), not a fifth expert.

## Architecture scope for v0

Per [[training-and-bootstrap-atlas-0.1]]'s scope-reduction decisions: declared edges (not RL), MLP gating (not RL gating), a 2-level U-Net — token-level plus a single whole-vehicle bottleneck (not the full 4-level hierarchy) — and direct flux-matching conservation (not learned discovery). Each is a deliberate minimization of simultaneous unknowns for a first build, with a stated trigger for revisiting it in the linked pages.

---

## Phase 0 — Scope and data

- Freeze the 2D geometry: a planar cross-section, chamber → nozzle → plume → airframe boundary layer → surrounding atmosphere.
- Generate training data per learned expert, since no existing dataset covers this combination (Gap 8, [[pfm-concept-overview]]):
  - **Reacting/internal flow:** quasi-1D/2D combustion-and-nozzle theory (Rayleigh-flow-with-heat-addition class of models) — not full turbulent-combustion CFD, since even ground-truth generation is nontrivial at this scope.
  - **External flow/plume:** a lightweight 2D compressible Euler/NS solver swept across Mach number and altitude.
  - **Thermal-structural:** a 2D conduction + linear-elasticity FEM solver.
- Define success metrics up front:
  - Per-expert accuracy against its classical baseline.
  - Full coupled-rollout stability across the whole ascent (millisecond combustion subcycling through the full flight duration).
  - Flux-residual at the $e\!-\!f$ and $d\!-\!g$ conservation interfaces.

## Phase 1 — Native scaffold

- Per-agent tokenizer ([[graph-tokenizer-atlas-0.1]], reusing [[graph-tokenizer-1.1]]).
- Declared/geometric edge instantiation ([[edge-generation-atlas-0.1]] Mechanism A).
- Typed message passing, reusing [[backbone-1.1]]'s typed-multigraph attention.
- 2-level pooling only.

## Phase 2 — Experts

Bootstrap and validate each of the three learned experts **independently** against its classical-solver baseline, per [[training-and-bootstrap-atlas-0.1]]'s training-order argument. Implement the rigid-body expert as a closed-form 3-DOF integrator (no training).

## Phase 3 — Wire it together

- Only once all three learned experts pass individual validation (per [[training-and-bootstrap-atlas-0.1]]): compose the full graph.
- MLP gating from declared edge-type to expert.
- Wire the flux-matching constraint at $e\!-\!f$ and $d\!-\!g$.

## Phase 4 — End-to-end validation

- Run the full coupled 2D ascent rollout.
- Compare against classical-solver baselines per expert, and where possible against known milestones (expected thrust-curve shape, approximate max-Q timing) as a sanity check the composed system hasn't decoupled into nonsense at the interfaces.
- **This phase is the actual test of the whole proposition**: does composing independently-bootstrapped experts through declared edges produce a stable, physically plausible rollout, or does it fall apart at the seams?

## Phase 5 — Expansion (only after Phase 4 succeeds)

- Next scenario: the offshore wind farm ([[pfm-purpose-and-direction]]) — reuses the external-compressible-flow expert almost directly for wake aerodynamics, adds a new electromagnetics expert for the generator.
- This is also the trigger point for attempting RL edge instantiation ([[edge-generation-atlas-0.1]] Mechanism B) — two scenarios' worth of regime-pairing experience is the stated minimum for it to have anything to learn from.
- Full U-Net pooling levels ([[unet-hierarchy-atlas-0.1]]) are reintroduced only when an agent count large enough to need them appears — most likely the moon-base scenario.

---

## Explicit non-goals for this phase

- Staging / topology mutation ([[global-fields-and-topology-atlas-0.1]]) — out of scope; the case study is powered ascent only.
- Composability (Gap 10, [[pfm-purpose-and-direction]]) — no new agent type appears mid-scenario.
- Safety-critical abstention (Gap 11) — not addressed; this is a research validation exercise, not a flight-software claim.
- 3D — deliberately 2D, matching the same tractability reasoning [[00-noether-1.1-overview]] used starting with 2D Rayleigh–Bénard ([[noether-1.0-rbc]]) before any 3D attempt.

## Watch item: the diversity-pretraining tension

Per [[transfer-learning-fine-tuning]]'s diversity principle (narrow pretraining looks good and generalizes badly), success on this single scenario is evidence the **architecture is soundly wired**, not evidence the **experts generalize**. This distinction should be kept explicit in any future report on Phase 4 results.

---

## See Also

- [[00-atlas-0.1-overview]]
- [[agent-definition-atlas-0.1]]
- [[edge-generation-atlas-0.1]]
- [[expert-library-atlas-0.1]]
- [[unet-hierarchy-atlas-0.1]]
- [[conservation-as-constraint-atlas-0.1]]
- [[global-fields-and-topology-atlas-0.1]]
- [[training-and-bootstrap-atlas-0.1]]
- [[pfm-purpose-and-direction]]
- [[incremental-transfer-roadmap]]
