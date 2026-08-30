# Atlas 0.1: Agent Definition

**Type:** Core Concept — Model Portion
**Related Concepts:** [[00-atlas-0.1-overview]], [[pfm-purpose-and-direction]], [[pfm-interface-design]], [[graph-tokenizer-atlas-0.1]], [[case-study-rocket-ascent-2d-atlas-0.1]]

---

## What counts as an agent

An **agent** is the unit Atlas treats as an independent local computation: it owns a declared geometric domain, a physical state defined over that domain, and its own conditioning signal. Two agents interact only through declared edges ([[edge-generation-atlas-0.1]]) or global fields ([[global-fields-and-topology-atlas-0.1]]) — never implicitly.

This is a **granularity choice**, not a fact about the physical system. The same rocket could be modeled as one agent (the whole vehicle) or dozens (every panel, every weld). The heuristic used throughout this project's worked examples: **an agent boundary should sit where you would want an independently computable local physics problem** — a boundary across which the governing equations genuinely change character, or across which a real engineering interface already exists (a wall, a docking seam, a free-stream boundary).

## Worked examples of the heuristic

- **Rocket ascent** ([[case-study-rocket-ascent-2d-atlas-0.1]]): seven agents — combustion reaction, combustion chamber, airframe, atmosphere-front, combustion outflow, plume, atmosphere-wake — chosen because each is where the governing physics genuinely changes (reacting flow → confined compressible flow → solid conduction/elasticity → external compressible flow → free-jet flow).
- **Moon/Mars base** ([[pfm-purpose-and-direction]]): each habitat module, rover, and subsystem is an agent — the natural granularity is the physical module boundary, since that's also where the base's real engineering interfaces (docking seams, air-loop couplings) live.
- **Offshore wind farm** ([[pfm-purpose-and-direction]]): each turbine is an agent, even though wake coupling between turbines is long-range and not adjacency-based — granularity tracks the physical object, not the interaction range (see [[edge-generation-atlas-0.1]] for how long-range edges are handled separately from adjacency edges).

## Per-agent state and conditioning

Each agent carries:
- **Field state** over its declared geometry (velocity, pressure, temperature, stress, etc. — whatever fields are relevant to that agent's physics).
- **Conditioning signal** — the dimensionless-number contract already established in [[pfm-interface-design]] (Re, Ma, Kn, …), computed per-agent from its own local state. This is what [[expert-library-atlas-0.1]]'s experts condition on, and in the general (non-Stage-1) version of [[regime-moe-architecture]] it's also what would drive the coarse routing gate.

## Stage 1 scope: declared, not discovered

In the current build ([[case-study-rocket-ascent-2d-atlas-0.1]]), the agent list and each agent's geometry are **user-declared inputs**, not something the model infers. This is a deliberate scope reduction: automatically segmenting a raw physical scene into the "right" set of agents is a genuinely hard, unsolved problem — closer to instance segmentation than to anything in [[00-noether-1.1-overview]]'s design space — and is not attempted here.

**[AI Inference]:** A future stage could plausibly turn agent segmentation into a learned step (e.g., cluster tokens by where the locally-fit governing equation changes character), but this should not be attempted before the declared-agent pipeline is validated end-to-end — see [[00-atlas-0.1-overview]]'s design invariant on minimizing simultaneous unknowns. This is speculative and not part of the current plan.

## What changes when an agent's boundary is wrong

Choosing agent granularity too coarse (e.g., merging the combustion chamber and nozzle into one agent) forces a single expert to internally represent two different equation-family regimes (confined subsonic combustion and choked/supersonic duct flow), which is exactly the failure mode [[regime-moe-architecture]]'s finding 1 warns about. Choosing it too fine (splitting the chamber wall into dozens of agents with no physical interface between them) creates edges that don't correspond to any real interface contract, inflating the graph without adding modeling power. There is no general rule for getting this right beyond the heuristic above; it is currently a human design decision made once per scenario at declaration time.

---

## See Also

- [[00-atlas-0.1-overview]]
- [[pfm-purpose-and-direction]]
- [[pfm-interface-design]]
- [[graph-tokenizer-atlas-0.1]]
- [[edge-generation-atlas-0.1]]
- [[case-study-rocket-ascent-2d-atlas-0.1]]
