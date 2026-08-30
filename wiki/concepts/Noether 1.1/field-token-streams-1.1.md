# Field Token Streams (Noether 1.1)

**Type:** Concept — model portion, NEW in 1.1 (folder: Noether 1.1)
**Status:** Design. Replaces [[noether-1.0-rbc]] §4.3's *summed* multi-field encoder — where per-field cross-attention outputs were added into one node token, destroying field identity — with **one token stream per field on a shared graph**, coupled by explicit learned cross-field edges. This is the "separate tokens per field, shared graph" decision.
**Related Concepts:** [[00-noether-1.1-overview]], [[graph-tokenizer-1.1]], [[field-descriptors-1.1]], [[edge-generation-1.1]], [[backbone-1.1]], [[multihead-attention]], [[open-architectural-problems]]
**Related Summaries:** [[multimodal-transformers-survey]], [[aion-1-astronomy]], [[dynami-cal-graphnet]]

---

## What it does — intuitively

At each site the physics is not one thing — it is a velocity *and* a temperature *and* (elsewhere) a density, a charge, a species label. 1.0-RBC ran a separate encoder per field and then **added** the results into a single node vector, paying for the separation and then throwing it away: the backbone received a soup it had to un-mix. 1.1 keeps the fields apart. Each site emits **one token per field** — a `velocity` token, a `temperature` token — sharing the site's geometry but carrying its own latent and its own field-type tag.

The payoff is that **field-to-field coupling becomes a first-class, learnable, inspectable object** — an edge type between a velocity token and a temperature token — rather than an implicit hope inside a summed vector. Buoyancy ($T$ drives vertical $u$) and advection ($u$ transports $T$) are *literally* cross-field interactions, so representing them as cross-field edges is honest, not decorative. It also makes **variable field cardinality** trivial: add or drop a field by adding or dropping a stream, which is what a model spanning fluids, plasmas, and particle species needs.

## What it does — mathematically

For node $i$ and field $f\in\mathcal F$ (the field set of *this* input), emit
$$t_i^{f} = \big[\ \mathrm{Enc}_f\big(h_i\big)\ \big\Vert\ W_e\,\theta_f\ \big] + z_{\text{cond}},$$
where $h_i$ is the shared node latent from [[graph-tokenizer-1.1]], $\mathrm{Enc}_f$ is a small field-typed projection, $z_{\text{cond}}$ the conditioning ([[conditioning-and-constants-1.1]]), and $\theta_f$ the **learned field descriptor** (type + force affordances) — concatenated into a reserved slot, *not* added, so identity stays separable from content. The descriptor $\theta_f$ (which upgrades 1.0's opaque field-type embedding $e_f$) and its second job — generating the field's simulation-conditioned query bank $Z_q^f$ — are the subject of [[field-descriptors-1.1]]. The token multiset for the graph is $\{t_i^f : i\in V,\ f\in\mathcal F\}$ — size $|V|\cdot|\mathcal F|$ (per context frame), versus $|V|$ under 1.0's summation.

Edges are **typed by the (field, field) pair** they join and supplied by [[edge-generation-1.1]]:
- **intra-field** $t_i^f \leftrightarrow t_j^f$ — that field's transport/diffusion,
- **cross-field co-located** $t_i^f \leftrightarrow t_i^{f'}$ — pointwise coupling (buoyancy source),
- **cross-field spatial** $t_i^f \leftrightarrow t_j^{f'}$, $i\neq j$ — advective / non-local coupling (the learned edges).

The backbone ([[backbone-1.1]]) attends over this typed multigraph; each edge type has its own message parameters, so "how velocity acts on temperature" is a distinct learned operator from "how velocity acts on velocity."

**No summation anywhere.** Fields are combined only through attention over typed edges, never by adding latents — the concrete removal of the 1.0-RBC design smell.

## Physics relevance

- **Coupling terms are the physics.** In the Boussinesq system the *only* things linking the momentum and heat equations are $\Pr\mathrm{Ra}\,T\hat{\mathbf y}$ (buoyancy) and $u\cdot\nabla T$ (advection). Making each a named edge type gives the model a parameter slot for exactly the interaction the equations contain — the [[open-architectural-problems]] S4.1 typed-message idea applied *within* one continuum, not only across particle/field types.
- **Multi-physics generality.** MHD adds an $\mathbf E,\mathbf B$ stream and Lorentz-coupling edges; a reacting flow adds species streams and reaction-coupling edges — all by adding streams and edge types, never by rewriting the core. This is the multimodal-transformer reading of physics ([[multimodal-transformers-survey]]): fields are modalities.
- **Conservation stays legible.** Because momentum lives in the velocity stream and heat in the temperature stream, a discovered invariant ([[discovered-conservation-1.1]]) can be attributed to a stream and its edges, instead of being tangled in a summed latent.

## How fundamental constants are included

- **Field-type embeddings are constant-free**; they identify *which* field, not its physical scale.
- **The coupling strength between two fields is set by conditioning, not hard-coded.** The buoyancy edge does not contain $\mathrm{Ra}$; instead $z_{\text{param}}$ (carrying $\log\mathrm{Ra},\log\Pr$) is added to every token, so the *same* cross-field edge operator produces weak or strong buoyancy depending on the injected group. Turn $\mathrm{Ra}\to0$ and the learned coupling should vanish — a checkable ablation, not an assumption.
- **Vector constants enter per stream** where relevant: gravity's covariant channel ([[conditioning-and-constants-1.1]]) rides the velocity stream (it forces momentum), not the temperature stream — the constant is routed to the field it physically acts on.

**[AI Inference]:** Summation in 1.0-RBC was the worst of both worlds — full encoder cost, zero separability. Its removal is likely the single highest-leverage representational fix in 1.1: even before the learned edges, simply *not adding* velocity and temperature latents should let the backbone represent the phase relationship between them (hot fluid rising) that a summed vector cannot hold.

## See Also
- [[graph-tokenizer-1.1]] — produces the shared node latent this splits into streams
- [[edge-generation-1.1]] — supplies the intra-/cross-field edges
- [[backbone-1.1]] — attends over the typed multigraph
- [[multihead-attention]] — heads as a basis of coupling operators
