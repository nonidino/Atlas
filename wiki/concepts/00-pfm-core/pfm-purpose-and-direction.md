# PFM Purpose & Direction: The Multi-Agent Deployment Target

**Type:** Core Concept — Vision & Worked Examples
**Related Concepts:** [[physics-foundation-models]], [[pfm-concept-overview]], [[pfm-architecture-approaches]], [[regime-moe-architecture]], [[pfm-interface-design]], [[00-noether-1.1-overview]], [[edge-generation-1.1]], [[transfer-learning-fine-tuning]]

---

## Why this page exists

[[pfm-concept-overview]] frames a PFM against an implicit benchmark target: fit diverse PDE snapshots, generalize across equation families, beat classical solvers on accuracy-per-FLOP. That target is necessary but not sufficient — it doesn't force a design toward any particular deployment shape, and as a result the wiki's architecture pages ([[00-noether-1.1-overview]], [[regime-moe-architecture]]) have been evaluated against "is this a good PFM" rather than "is this good *for something*."

This page fixes a concrete north star: **a PFM deployed as the shared physics-interaction layer for a large, compositional, safety-critical multi-agent system** — a moon/mars base being the paradigmatic case, with four further worked examples chosen to stress different axes of that requirement. Everything here is a **reframing of purpose**, not a change to [[00-noether-1.1-overview]]'s implementation status (see [[implementation-log]] for current build state) or a resolution of [[regime-moe-architecture]] vs. Noether 1.1 (still open).

---

## The core deployment shape

Physical objects in the target system — habitat modules, spacecraft, vehicles, structures, cells in a pack, turbines in a farm — are **agents**. Each agent owns a local physical state. When agents come into **proximity** (physical, radiative, wake-mediated, or otherwise — see below), they **interact through the PFM**: the model computes the coupled physics at that interface, fast enough to sit inside whatever control loop depends on the answer, instead of invoking a bespoke classical solver per interaction type.

This reframes what "foundation" needs to mean:

- **Foundation = works across regimes without being told which one**, same as the benchmark framing — but now *the regime can be different at every edge of the graph, simultaneously*, and can change type mid-simulation (a battery cell entering thermal runaway, a spot fire igniting a roof).
- **Foundation = amortizes classical-solver cost**, same as [[pfm-concept-overview]]'s Capability 7 — but now the amortization has to hold under a **latency budget dictated by the fastest control loop touching the model**, not by "faster than an offline DNS run."
- **Foundation = generalizes zero-shot**, same as Capability 2 — but the OOD case that matters most is *a new agent type appearing in the graph for the first time*, not a new parameter regime for an equation family the model has already seen.

---

## Worked Examples

Five scenarios, chosen so each stresses a different combination of spatial/temporal scale range, regime diversity, and adaptivity requirement. None have been simulated or benchmarked — these are design-target thought experiments, not validated test cases.

### 1. Moon/Mars habitat base (the paradigm case)

**Setup:** each habitat module, rover, docked spacecraft, and life-support subsystem is an agent. Agents interact through the PFM when physically adjacent or fluidically/thermally coupled (shared air loop, shared structural connection, docking interface).

- **Regimes:** structural mechanics (docking loads, regolith foundation), thermal (radiative + conductive, no convective atmosphere outside the hab), low-gravity multiphase fluid dynamics (life support, propellant), granular mechanics (regolith, dust intrusion), radiation transport (solar/cosmic, shielding).
- **Scale range:** millimeters (seal interfaces) to the full base footprint (hundreds of meters); seconds (a hull breach) to years (structural creep, radiation-damage accumulation).
- **Adaptivity requirement:** the defining one — modules get added over the base's operational life, and the model must correctly compute a never-before-co-located agent pair's interaction the moment it's docked, with no retraining window available.
- **Edge/proximity model:** physical adjacency, dominant and simplest of the five examples here — good baseline case.

### 2. Wildfire spread at the wildland–urban interface

**Setup:** each structure and vegetation cell is an agent. Full worked reasoning in the prior brainstorm; summarized here for the record.

- **Regimes:** combustion/reacting flow, radiative heat transfer, turbulent buoyant plume (compressible), structural (material state change on ignition).
- **Scale range:** the widest of the five — millimeter/millisecond combustion chemistry to kilometer/day-scale fire spread.
- **Adaptivity requirement:** a structure's material properties change discontinuously mid-simulation (unlit → burning), which is a *regime change on an existing agent*, not a new agent joining.
- **Edge/proximity model:** radiative and convective reach, not physical contact — a spot fire ignites structures 100m downwind. Directly motivates a discontinuity channel (finding 2, [[regime-moe-architecture]]) since ignition is a genuine threshold event riding on a continuously-varying heat flux.

### 3. EV battery pack thermal runaway propagation

**Setup:** each cell is an agent; physical adjacency within the pack.

- **Regimes:** electrochemistry (ion transport + reaction kinetics), thermal diffusion, structural (cell swelling/rupture), compressible gas dynamics (vented gas), all coupled within one cell's volume.
- **Scale range:** the tightest — a single 18650-format cell — but the timescale range is extreme: hours (normal cycling) to milliseconds (runaway propagation to a neighbor).
- **Adaptivity requirement:** the interaction *regime itself* flips from benign thermal diffusion to combustion-adjacent multiphysics the instant a neighboring cell triggers — the router has to re-classify an existing edge mid-simulation.
- **Why it matters most for stakes:** the clearest case in this set for **mandatory calibrated uncertainty and abstain-to-classical-solver fallback** (new Gap 3, below) — a confidently wrong propagation-risk estimate here is a pack fire, not a bad benchmark score.

### 4. Atmospheric reentry vehicle

**Setup:** a single vehicle traversing regimes along its own trajectory — the one example here that isn't primarily multi-agent, included deliberately so the architecture isn't over-fit to "physics only happens when two agents meet."

- **Regimes:** rarefied/kinetic flow (high altitude) → continuum hypersonic shock layer → ionized plasma sheath (comms blackout) → ablative thermal-protection-system chemistry/structural response → airframe structural loads.
- **Scale range:** the vehicle's own boundary layer transitions between continuum and rarefied *within itself* at certain altitudes — sub-object spatial resolution for a smooth-crossover conditioning signal.
- **Adaptivity requirement:** this is the concrete worked case for [[regime-moe-architecture]]'s finding 3 — hypersonic flow is the numerically *stiff* regime, not the cheap asymptotic limit, and it's also the highest-consequence phase of flight. A curriculum built on "simple regimes first" would badly mis-prioritize training on this trajectory.
- **Edge/proximity model:** none — single agent, multi-regime coupling *within* one object. Tests whether the coarse MoE gate can route sub-regions of a single agent's own state to different experts.

### 5. Offshore wind farm over its operating life

**Setup:** each turbine is an agent.

- **Regimes:** turbulent wake aerodynamics (fluid-structure interaction on blades), wave-structure interaction at the foundation, electromagnetics in the generator, slow structural fatigue/plasticity, electrochemical corrosion of the foundation — five regimes, five very different time constants, on one piece of hardware.
- **Scale range:** seconds (turbulent wake shedding) to decades (fatigue, corrosion).
- **Adaptivity requirement:** none in the "new agent type" sense — the stress here is structural, not novelty.
- **Edge/proximity model:** the sharpest counterexample to "proximity = adjacency" in this set. Wake interaction between turbines is long-range (advects kilometers downwind) and wind-direction-dependent, changing hour to hour — a genuinely global, hyperbolic-ish coupling per finding 1 of [[regime-moe-architecture]]. Meanwhile that *same turbine's* fatigue accumulation is purely local with **no** cross-turbine coupling at all. One agent pair can require two entirely different locality classes depending on which physics question is being asked — proximity/edge topology has to be regime-conditional, not a single fixed graph.

### Cross-example pattern

No two examples define "proximity" the same way: physical adjacency (moon base), radiative/convective reach (wildfire), a propagating internal failure state (battery), altitude-dependent regime transition within one body (reentry), and wind-direction-dependent long-range wake coupling (wind farm). **Edge generation cannot be a single fixed graph-construction rule** — it has to be regime-conditional, and in the wind-farm case, conditional *per regime for the same agent pair simultaneously*. This is a concrete addition to [[edge-generation-1.1]]'s existing same-field/cross-field KNN + learned long-range edge design: the "learned long-range edge model" needs the regime signal as an explicit input, not just geometry.

---

## Reframed gaps: [[pfm-concept-overview]]'s 8 gaps against this deployment shape

| Gap | Benchmark framing | Sharpened by the multi-agent target |
|---|---|---|
| 1. Unified representation | Handle heterogeneous field types/resolutions | Also: agents at wildly different native scales (a battery cell and a wind farm) must live in the *same* graph if the base/pack/farm contains both |
| 2. Error accumulation | Bounded rollout error over a benchmark horizon | Rollouts run for the operational life of the system (years) — this is a qualitatively harder version of the same problem |
| 3. Scale (params) | Unknown scaling laws | Unchanged — orthogonal to deployment shape |
| 4. Physics enforcement vs. generality | Equation-specific hard constraints don't generalize | Now needs to hold *per expert*, since each regime family in [[regime-moe-architecture]] wants a different constraint type |
| 5. Irregular geometry at scale | Real engineering geometry is irregular | Sharpens into **composability** (new Gap 3 below) — irregular geometry the model has *never seen the shape of before*, appearing live |
| 6. Multi-scale dynamics | No resolution-adaptive architecture | The wind farm and reentry examples both need scale-adaptivity *within a single agent's own state*, not just across the corpus |
| 7. No benchmark infrastructure | No standard PFM suite | These five worked examples are a candidate seed for a deployment-shaped benchmark, distinct from the existing PDE-accuracy suite in [[possible-architectures]] |
| 8. Training data at scale | No "Common Crawl of physics" | Sharpens further — need data for *interaction interfaces between agent types*, which no existing PDE dataset (The Well included) is organized around |

## New gaps this deployment shape surfaces (not in [[pfm-concept-overview]])

**Gap 9 — Cross-boundary consistency between independently-computed agents.** If each agent runs its own local forward pass and agents only couple through shared edges, what guarantees two agents' views of their shared interface agree? This is the ML analog of domain-decomposition / Schwarz-alternating consistency in classical solvers. No page in this wiki currently addresses it.

**Gap 10 — Composability as inference-time generalization, not a training-time property.** Distinct from Gap 5 (irregular geometry): the requirement is "handle an agent-type the model was never shown during training, correctly, the first time it's docked" — a stronger claim than OOD parameter generalization, closer to few-shot object recognition than to interpolation.

**Gap 11 — Verified fallback / abstention.** No current architecture page (Noether 1.1 included) has a designed mechanism for "recognize you're out of your depth and hand off to a classical solver or flag a human," which a safety-critical deployment (examples 1, 3 above especially) requires and a benchmark deployment never asks for.

---

## Relationship to existing architecture work

This page does not decide [[regime-moe-architecture]] vs. [[00-noether-1.1-overview]]. It does sharpen the comparison: [[regime-moe-architecture]]'s per-family expert structure is a more natural fit for Gap 9/10/11 above (isolable experts make it easier to bound what "out of depth" means, and to add a new expert without disturbing existing ones — see [[mixture-of-experts]]'s "modular update" advantage) than a single dense conditioned backbone would be. That's a point in favor of the MoE direction for *this* deployment target specifically, not a general verdict.

See [[incremental-transfer-roadmap]] for how to get from current resources to an architecture that can attempt any of the five worked examples above, without a from-scratch build. **[[00-atlas-0.1-overview]]** is the named architecture that resulted, with [[case-study-rocket-ascent-2d-atlas-0.1]] as its first concrete instantiation — a sixth worked example (rocket ascent), chosen specifically because it exercises the core mechanisms without requiring composability, staging, or safety-critical abstention up front.

---

## See Also

- [[physics-foundation-models]]
- [[pfm-concept-overview]]
- [[pfm-architecture-approaches]]
- [[regime-moe-architecture]]
- [[incremental-transfer-roadmap]]
- [[pfm-interface-design]]
- [[00-noether-1.1-overview]]
- [[edge-generation-1.1]]
- [[transfer-learning-fine-tuning]]
- [[mixture-of-experts]]
- [[00-atlas-0.1-overview]]
- [[case-study-rocket-ascent-2d-atlas-0.1]]
