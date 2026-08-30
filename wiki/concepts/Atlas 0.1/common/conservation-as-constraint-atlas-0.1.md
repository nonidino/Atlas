# Atlas 0.1: Conservation as an Imposed Constraint

**Type:** Core Concept — Model Portion
**Related Concepts:** [[00-atlas-0.1-overview]], [[port-algebra-atlas-0.1]], [[discovered-conservation-1.1]], [[edge-generation-atlas-0.1]], [[expert-library-atlas-0.1]], [[case-study-rocket-ascent-2d-atlas-0.1]]

---

## Two corrections to this page, 2026-08-19

**1. The idea is not new, and this page must say so.** Enforcing **flux continuity at subdomain interfaces** as the conservation mechanism is the contribution of **cPINN** (Jagtap, Kharazmi & Karniadakis, *CMAME* 2020), which decomposes the domain and enforces flux continuity in strong form along subdomain interfaces; **XPINN** generalizes it to arbitrary space–time decompositions. This page previously read as though the mechanism were novel to Atlas. It is not.

**The real distinction, stated precisely:** cPINN trains all subdomain networks **jointly against a PDE residual**, so the interface condition is a training loss on networks that grew up together. Atlas imposes it **at inference, on frozen experts that never met**. That is a genuine difference — and also precisely why Atlas's version might fail, which is what [[case-study-rbc-decomposition-atlas-0.1]] and [[spec-wind-farm-wake-atlas-0.1]] are built to find out. See [[prior-art-and-novelty-atlas-0.1]] §3.

**2. `conservation` is no longer an edge type.** Under [[port-algebra-atlas-0.1]] it was never a quantity — it is the *definition* of a port connection, not a label applied to some of them. Every port carries

$$e_A\big|_\Gamma=e_B\big|_\Gamma,\qquad f_A\big|_\Gamma=-f_B\big|_\Gamma$$

by construction. What the old label was really marking — *"check flux matching here"* — is now a residual **reported at every port**, with a separate `conservative: bool` deciding whether it is also **enforced**. Read every occurrence of "conservation-typed edge" below as "a port whose residual is enforced rather than merely measured."

## Enforce or measure: the decision rule

The port algebra makes the criterion sharp, and it is not about what the edge is called:

| Situation | Mechanism | Level |
|---|---|---|
| **One side is closed-form** (algebraic or ODE expert) | **Hard projection** onto the exact side. Minimum-norm, and where possible applied in a potential representation so other constraints survive — see [[spec-wind-farm-wake-atlas-0.1]] §7.3 for the worked case | 5 |
| **A conserved quantity has a potential representation** (e.g. mass via a stream function) | **Exact by construction**; nothing to enforce | 5 |
| **Both sides are learned and frozen** | **Measure only.** There is no exact side to project onto and no loss to penalize with. Report the residual per port | 2 |
| **Both sides are learned and trainable** | Soft interface loss — this is cPINN's regime | 2–3 |

**The third row is the honest and uncomfortable one:** Atlas's flux-matching story is exactly enforceable only where one side is closed-form, and is a *measurement* everywhere else. Saying otherwise would overclaim. The closed-form edges are what calibrate what a good residual looks like for the rest.

## The global power residual replaces the per-edge special case

Because every port is an effort–flow pair whose product is power, the whole graph admits one scalar diagnostic — [[port-algebra-atlas-0.1]] §6:

$$\mathcal R(t)=\sum_{i\in\mathcal A}\frac{dE_i}{dt}+\sum_{\Gamma\in\mathcal P}\int_\Gamma (e f)_\Gamma\,ds-\sum_i\mathcal D_i-P_{\text{ext}}$$

$\mathcal R\neq0$ means the coupling is creating or destroying energy, and the per-port breakdown localizes where. **This generalizes everything below from a per-edge special case to one number that works unchanged in every case study, present and future**, and it should be reported every macro-step from now on.

---

## The direct contrast with Noether 1.1

[[discovered-conservation-1.1]] is Noether 1.1's thesis in miniature: the model learns invariants $\{C_k\}$ from data via a threshold-gated-hard-plus-learned-soft hybrid, discovering *which* quantities are conserved rather than being told. That is the right design when agents, edges, and regimes are themselves unknown and must be inferred — Noether 1.1's whole architecture is built around not assuming a fixed structure in advance.

Atlas starts from the opposite assumption. In Stage 1 ([[case-study-rocket-ascent-2d-atlas-0.1]]), the agent graph and the typed edge list are **declared inputs** ([[edge-generation-atlas-0.1]]). If a $d\!-\!g$ edge is labeled "conservation of heat and fluid," the modeler has already told the system what must be conserved across that interface — there is nothing left to discover. **Conservation in Atlas is a flux-matching constraint imposed at declared conservation-typed edges, checked and enforced between whatever experts sit on either side, not a structural property the model has to find on its own.**

This is also the literal reason the name avoided "Noether" — conservation-law discovery isn't this architecture's forte; composing declared local experts through declared interfaces is. Atlas still respects conservation, and takes it seriously as a correctness requirement — it just gets there by a different mechanism.

## What the constraint actually does

At a conservation-typed edge (e.g., $e\!-\!f$: combustion outflow → plume, or $d\!-\!g$: atmosphere-front → atmosphere-wake), the two experts on either side ([[expert-library-atlas-0.1]]) each produce a local prediction of the flux crossing that interface. The constraint enforces:

$$
\Phi_{e \to f}(\hat{x}_e) \;=\; \Phi_{f \gets e}(\hat{x}_f)
$$

i.e., the mass/momentum/energy flux the outflow expert says is leaving must equal the flux the plume expert says is arriving — no source or sink is permitted to appear at an interface the modeler has declared conservative. This can be implemented as a hard projection (re-project both sides onto the flux-matched manifold post-hoc) or a soft penalty during training; [[training-and-bootstrap-atlas-0.1]] and [[case-study-rocket-ascent-2d-atlas-0.1]] specify a **direct flux-matching constraint** for the first build — simpler than [[discovered-conservation-1.1]]'s learned-discovery mechanism, and sufficient because the *which* question is already answered by the declaration.

## Why not just reuse [[discovered-conservation-1.1]] directly

Its hybrid mechanism (threshold-gated-hard + learned-soft, local-divergence-discovery-first) is solving a harder, more general problem than Stage 1 needs: figuring out *which* invariants exist from data, at *arbitrary* points in a graph whose structure isn't given in advance. Reusing that machinery here would import cost and failure modes (a discovery mechanism can misidentify what's conserved) for a problem Atlas's declared-graph assumption has already solved by construction. This is a case where a more general, more powerful mechanism is the *wrong* tool, not merely an unnecessary one.

## Where this could hybridize back later

**[AI Inference]:** If [[edge-generation-atlas-0.1]]'s RL-based edge instantiation is ever adopted (deferred until multiple scenarios exist), the "which edges are conservative" question stops being fully declared — an RL-proposed edge might not come pre-labeled with a conservation type the way a human-declared one does. At that point, something closer to [[discovered-conservation-1.1]]'s discovery mechanism would become relevant again, layered on top of the RL-proposed graph rather than replacing the flux-matching constraint at edges that remain declared. This is speculative and not part of the current plan — flagged here only because it's the natural point where Atlas's and Noether 1.1's conservation mechanisms would reconverge.

---

## See Also

- [[composition-error-theory]] — what conservation does *not* buy (it constrains finitely many linear functionals of the error); extends enforce-or-measure to *enforce, measure, or decline*
- [[port-algebra-atlas-0.1]] — ports, and why `conservation` stopped being one
- [[prior-art-and-novelty-atlas-0.1]] — the cPINN positioning
- [[00-atlas-0.1-overview]]
- [[discovered-conservation-1.1]]
- [[edge-generation-atlas-0.1]]
- [[expert-library-atlas-0.1]]
- [[training-and-bootstrap-atlas-0.1]]
- [[case-study-rocket-ascent-2d-atlas-0.1]]
