# Atlas 0.1: Edge Generation

**Type:** Core Concept — Model Portion
**Related Concepts:** [[00-atlas-0.1-overview]], [[port-algebra-atlas-0.1]], [[edge-generation-1.1]], [[expert-library-atlas-0.1]], [[conservation-as-constraint-atlas-0.1]], [[case-study-rocket-ascent-2d-atlas-0.1]], [[unet-hierarchy-atlas-0.1]]

---

## Two mechanisms, both live

Atlas keeps **two edge-instantiation mechanisms on the table simultaneously** rather than committing to one: declared/geometric instantiation (the Stage 1 default, used in [[case-study-rocket-ascent-2d-atlas-0.1]]) and RL-based instantiation (a documented future direction, deliberately not attempted yet). Neither supersedes the other; which one is appropriate depends on how much cross-scenario experience exists to learn from.

---

## Mechanism A: Declared/geometric instantiation (Stage 1 default)

The user supplies the agent list and a **typed edge list** directly — which agent pairs interact, and via which interface contract(s). Edge instantiation then **materializes** the declared edges into token-level connections at shared interface geometry (matching [[graph-tokenizer-atlas-0.1]]'s boundary tokens on each side), rather than discovering them via a learned scorer.

This is a strict simplification of [[edge-generation-1.1]]'s existing design: that page's same-field/cross-field KNN plus learned long-range edge model is a *discovery* mechanism, built for when the graph isn't known in advance. Atlas's Stage 1 doesn't need discovery, because the graph is given — geometric matching at declared interfaces is enough, and it's deterministic, which makes debugging the rest of the pipeline far more tractable.

### Edge types are ports — see [[port-algebra-atlas-0.1]]

**As of 2026-08-19 the type vocabulary is closed and defined elsewhere.** An edge declares one or more **ports** drawn from a fixed set of five — `MECH`, `ROT`, `THERM`, `ELEC`, `ADVEC` — each an effort–flow pair whose product is power. [[port-algebra-atlas-0.1]] is the binding definition; this page only covers how declared ports get *materialized* into token-level connections.

The previous vocabulary (`heat`, `pressure`, `stress`, `fluid`, `momentum`, `mass`, `shear`, `conservation`) is retired. It mixed quantities, phenomena, a medium and a constraint, and — the reason it had to go — it forced the interface contract to be defined per expert-pair, which is $O(K^2)$ in the number of expert families and would have ended the program at the scale [[f1-pathmap-and-end-goal]] requires.

Two consequences land directly on this page:

- **`conservation` is no longer a type.** Every port conserves by construction: what leaves one agent enters the other. Materialization therefore attaches a residual reporter to *every* port, and a `conservative: bool` flag decides whether that residual is merely measured or actively enforced ([[conservation-as-constraint-atlas-0.1]]).
- **Mechanism A materializes ports, not labels.** The geometric matching step is unchanged; what changes is that the thing being matched is a declared port with known effort and flow variables, so the edge layer knows what to interpolate and — per preCICE practice — whether to map it **conservatively** (integral preserved: all flows) or **consistently** (pointwise values preserved: all efforts).

### The worked example: the rocket's typed edge list

$$
\begin{aligned}
&a \!-\! b:\ \texttt{MECH},\ \texttt{THERM},\ \texttt{ADVEC}(h_0,Y_k) &&e \!-\! f:\ \texttt{MECH},\ \texttt{THERM},\ \texttt{ADVEC}(h_0,Y_k)\
&b \!-\! c:\ \texttt{MECH},\ \texttt{THERM} &&e \!-\! b:\ \texttt{MECH},\ \texttt{THERM},\ \texttt{ADVEC}(h_0,Y_k)\
&c \!-\! d:\ \texttt{MECH},\ \texttt{THERM},\ \texttt{ADVEC}(h_0) &&g \!-\! f:\ \texttt{MECH},\ \texttt{THERM},\ \texttt{ADVEC}(h_0)\
&d \!-\! g:\ \texttt{THERM},\ \texttt{ADVEC}(h_0),\ \texttt{MECH}
\end{aligned}
$$

where $a$ = combustion reaction, $b$ = combustion chamber, $c$ = airframe, $d$ = atmosphere-front, $e$ = combustion outflow, $f$ = plume, $g$ = atmosphere-wake (see [[case-study-rocket-ascent-2d-atlas-0.1]] for the full case study).

**The migration exposed a real error, not just a renaming.** Four of these seven edges previously carried a `heat` label across an interface with mass crossing it. Heat moves two ways there — conduction (`THERM`) *and* advection of enthalpy (a passenger on `ADVEC`) — so those edges were **silently missing the advective half**. Only $b\!-\!c$, a wall with no mass crossing, was correct as originally written. See [[port-algebra-atlas-0.1]] §3.5 and §7.1.

**The $b\!-\!g$ edge (chamber ↔ wake atmosphere) was removed.** It described base-region heating/recirculation, but the chamber and the wake atmosphere aren't in physical contact — that interaction is actually mediated through the plume ($f$) and airframe ($c$). A direct $b\!-\!g$ edge was a modeling shortcut standing in for a two-hop physical path; removing it forces the interaction to route through the agents that are actually adjacent, which is the more physically honest graph.

### A port is an interface contract, not an expert

A critical distinction, worked out fully in [[expert-library-atlas-0.1]]: a port describes **what must be continuous across the edge**, not a dedicated compute module. Whether an edge's ports are computed by one coupled expert or several depends on whether the underlying governing equations are tightly or weakly coupled — see [[expert-library-atlas-0.1]] for the full argument and the $a\!-\!b$ vs. $b\!-\!c$ contrast that motivates it. The port algebra sharpens this: $a\!-\!b$'s `MECH` and `THERM` come out of one compressible reacting-flow system and share an expert, while $b\!-\!c$'s come from two different governing systems and route to two experts that exchange the coupling term explicitly.

---

## Mechanism B: RL-based instantiation (deferred)

Instead of a declared edge list, an RL policy proposes which agent pairs should be connected and with what type(s), trained on reward from downstream prediction accuracy and conservation-residual, accumulated across many past simulation episodes — "recognizing connections between different regimes based on past experience," as opposed to [[edge-generation-1.1]]'s differentiable KNN-plus-scorer approach.

### Why this is genuinely different from the differentiable scorer, not just a harder version of it

Which regimes should even be considered connected is, in general, a **discrete, combinatorial decision** — closer to "does an edge of this type exist at all" than to a continuously-relaxable weight. A reward-driven policy search can explore that discrete space directly; a differentiable scorer has to approximate it via a continuous relaxation (Gumbel-softmax, as [[edge-generation-1.1]] already does). RL is the more natural fit for the discovery problem in principle.

### Why it's deferred, not rejected

The RL framing's own justification — "past experience across regimes" — requires there to *be* past experience across multiple regime pairings. With exactly one scenario ([[case-study-rocket-ascent-2d-atlas-0.1]]'s rocket ascent), there is no cross-scenario diversity yet to generalize from; an RL policy trained on one scenario's edge topology has nothing to learn beyond what the declared edge list already encodes by hand, at far higher training cost and far worse sample efficiency. **Not before scenario 2** ([[incremental-transfer-roadmap]], [[training-and-bootstrap-atlas-0.1]]) is the concrete trigger condition, not an arbitrary caution.

### The ordering dependency with the U-Net hierarchy

If Mechanism B is adopted later, it introduces a real sequencing constraint noted in [[unet-hierarchy-atlas-0.1]]: the RL-proposed edge set has to be stable enough episode-to-episode for the hierarchical pooling levels above it to have something consistent to pool *over*. An edge set that changes every episode gives the supernode clustering (which assumes a roughly-fixed graph topology within a training run — see [[unet-hierarchy-atlas-0.1]]) nothing stable to learn from. RL edge discovery would need to converge (or be trained in a separate, earlier phase) before the pooling hierarchy is trained on top of it.

---

## See Also

- [[port-algebra-atlas-0.1]] — **the binding definition of the five port types**
- [[f1-pathmap-and-end-goal]] — why the vocabulary had to be closed
- [[00-atlas-0.1-overview]]
- [[edge-generation-1.1]]
- [[expert-library-atlas-0.1]]
- [[conservation-as-constraint-atlas-0.1]]
- [[case-study-rocket-ascent-2d-atlas-0.1]]
- [[unet-hierarchy-atlas-0.1]]
- [[incremental-transfer-roadmap]]
