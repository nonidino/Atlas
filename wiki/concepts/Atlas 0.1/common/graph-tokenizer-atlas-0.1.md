# Atlas 0.1: Tokenizer

**Type:** Core Concept — Model Portion
**Related Concepts:** [[00-atlas-0.1-overview]], [[graph-tokenizer-1.1]], [[agent-definition-atlas-0.1]], [[edge-generation-atlas-0.1]], [[00-token-representation-overview]]

---

## Reused, not reinvented

Atlas's per-agent tokenizer is [[graph-tokenizer-1.1]], applied independently to each agent. Tokenization is a fundamentally agent-local operation — it encodes one agent's field state over its own geometry into a resolution-free set of tokens — and nothing about *which regime the agent belongs to* or *how it interacts with other agents* is relevant at this stage. There is no reason to design a new tokenizer for Atlas; [[graph-tokenizer-1.1]]'s resolution-free node encoder with redundant derived features (gradients, vorticity) and its dual smooth/high-frequency path apply exactly as designed there.

$$
Z_a = \text{Tokenizer}_\theta\big(x_a, \text{geom}_a\big), \qquad a \in \{\text{agents}\}
$$

Each agent's tokenizer instance runs independently; there is no cross-agent attention or mixing at this stage — that happens later, at [[edge-generation-atlas-0.1]] and [[expert-library-atlas-0.1]].

## What's new for Atlas: boundary-token exposure

The one addition Atlas needs beyond [[graph-tokenizer-1.1]]'s existing design: some subset of an agent's tokens sit on or near its declared **interface geometry** with a neighboring agent (e.g., the chamber-wall-facing tokens of the airframe agent, or the nozzle-exit-facing tokens of the plume agent). These need to be identifiable as **boundary tokens**, because [[edge-generation-atlas-0.1]]'s declared-edge instantiation step attaches edges specifically at shared interface geometry, not at arbitrary token pairs.

This is a light-touch addition — a flag on the token, derived directly from each agent's declared geometry (which faces/regions of the domain are shared with another agent), not a learned property. It does not change the tokenizer's internal representation, only what metadata accompanies each token downstream.

## Conditioning

Per-agent conditioning tokens (Re, Ma, Kn, …, per [[pfm-interface-design]] and [[agent-definition-atlas-0.1]]) are concatenated in exactly the way [[graph-tokenizer-1.1]] and [[conditioning-and-constants-1.1]] already specify. No change here either — the conditioning contract is one of the pieces of Noether 1.1's design that transfers to Atlas without modification (see [[00-atlas-0.1-overview]]'s design invariants and [[incremental-transfer-roadmap]]'s observation that this layer is already-built native scaffold).

---

## See Also

- [[00-atlas-0.1-overview]]
- [[graph-tokenizer-1.1]]
- [[agent-definition-atlas-0.1]]
- [[edge-generation-atlas-0.1]]
- [[00-token-representation-overview]]
- [[pfm-interface-design]]
