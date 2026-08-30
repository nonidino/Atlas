# Field Descriptors & Simulation-Conditioned Queries (Noether 1.1)

**Type:** Concept — model portion, substep of tokenization / field streams, NEW in 1.1 (folder: Noether 1.1)
**Status:** Design. A single learned object — a per-field **descriptor** $\theta_f$ encoding *field type + force affordances* — that (1) is appended to every token of that field's stream and (2) **generates that field's query bank** $Z_q^f$, making the tokenizer's feature-extractor simulation-specific. Fleshes out the S8.4 hypernetwork path for [[open-architectural-problems]] Problem 8 (variable field cardinality) and instantiates [[physics-conditioned-query-tokens]] ("queries = physical operators") for the graph tokenizer. Hangs off [[graph-tokenizer-1.1]] / [[field-token-streams-1.1]] the way [[intelligent-patching]] hangs off the tokenizer.
**Related Concepts:** [[00-noether-1.1-overview]], [[field-token-streams-1.1]], [[graph-tokenizer-1.1]], [[conditioning-and-constants-1.1]], [[edge-generation-1.1]], [[decoder-1.1]], [[physics-conditioned-query-tokens]], [[learned-query-compression-tokens]], [[equivariant-gnns]], [[open-architectural-problems]]
**Related Summaries:** [[q-former-architecture]], [[aion-1-astronomy]], [[dynami-cal-graphnet]], [[multimodal-transformers-survey]]

---

## What it does — intuitively

A field is more than a name. "Velocity" is a rank-1 vector that rotates with the frame, is advected, feels pressure gradients and viscosity, and couples to a scalar through buoyancy; "temperature" is a rank-0 scalar that is advected and diffuses. The 1.0 design gave each field only an opaque ID embedding $e_f$ — a lookup that says *which* field but nothing about *what it is* or *what it does*. Noether 1.1 replaces that ID with a **learned descriptor $\theta_f$** carrying two kinds of content: **how the field transforms** (its tensor/representation type) and **what interactions it can participate in** (its force *affordances*).

That one object then does two jobs, which are your two intuitions:

1. **Appended to the token** — every token of field $f$ carries $\theta_f$ in a reserved slot, so the backbone always knows what kind of thing it is looking at and what it is allowed to couple to.
2. **Generates the query bank** — the fixed learned queries $Z_q$ that probe a patch ([[graph-tokenizer-1.1]] Q1) are no longer one-size-fits-all; they are *produced from* $\theta_f$ (plus a descriptor of the whole simulation), so a shock-dominated compressible flow, a smooth Stokes flow, and an N-body cloud each get a feature-extractor tuned to what is salient in *that* physics.

Crucially none of this is a hand-labeled "simulation = gravity" switch. The simulation's identity is *derived* from the set of field descriptors present plus the physical constants — the same two channels that already distinguish systems ([[00-noether-1.1-overview]] Q2).

## What it does — mathematically

**The descriptor.** Per field $f$, a vector $\theta_f\in\mathbb R^{d_\theta}$ ($d_\theta\!\approx\!32$), split structured + free:
$$\theta_f = \big[\underbrace{\text{rank},\ \text{parity},\ \text{is-conserved},\ \text{rep-type}}_{\text{structured — supplied, physics}}\ \big\Vert\ \underbrace{u_f}_{\text{free — learned}}\big].$$
The structured block is genuine symmetry data (velocity: rank 1, even; $\mathbf B$: rank 1, odd/pseudovector; temperature: rank 0), read by the covariant machinery ([[equivariant-gnns]], [[conditioning-and-constants-1.1]]); the free block $u_f$ is a learned per-field vector for whatever the structured tags do not capture.

**Job 1 — append to the token (concatenate, don't add).** With $h_i$ the shared node latent and $z_{\text{cond}}$ the conditioning:
$$t_i^{f} = \big[\ \mathrm{Enc}_f(h_i)\ \big\Vert\ W_e\,\theta_f\ \big] + z_{\text{cond}}.$$
The reserved sub-block ($\Vert$, not $+$) is deliberate and is the *same* argument that removed the field summation in [[field-token-streams-1.1]]: adding $\theta_f$ into the content dimensions would let "which field / what affordances" collide with "what the field is doing here." A reserved slot keeps identity linearly separable from content.

**The simulation descriptor — derived, not labeled.**
$$z_{\text{sim}} = \operatorname{pool}_{f\in\mathcal F}\big(\theta_f\big)\ \big\Vert\ z_{\text{cond}},$$
i.e. the (permutation-invariant) pool of the descriptors of the fields actually present, concatenated with the constants embedding. $\{\text{velocity},\text{temperature}\}+\{\mathrm{Ra},\mathrm{Pr}\}$ pools to "convection"; $\{\text{position},\text{velocity},\text{mass}\}+\{G\}$ pools to "N-body gravity" — the simulation *type* is a function of its field set and constants, so a new system is a new combination, never a new code path.

**Job 2 — generate the query bank.** The global $Z_q$ becomes a shared base plus a low-rank, descriptor-conditioned correction (the recommended middle of the spectrum below):
$$\boxed{\,Z_q^{f} = Z_q^{\text{base}} + \Delta_q(\theta_f, z_{\text{sim}}),\qquad \Delta_q = \operatorname{reshape}\!\big(U\,V^\top\,[\theta_f\Vert z_{\text{sim}}]\big),\ \ \operatorname{rank}(UV^\top)=r\ll N_q\,}$$
with $Z_q^{\text{base}}\in\mathbb R^{N_q\times d_{\text{head}}}$ (e.g. $16\times16$). An equivalent FiLM form modulates each query row, $Z_q^f[q,:]=\gamma_q\odot Z_q^{\text{base}}[q,:]+\beta_q$ with $(\gamma,\beta)=\mathrm{MLP}([\theta_f\Vert z_{\text{sim}}])$.

**The spectrum (options, cheapest → most general):**

| Option | Mechanism | Trade-off |
|---|---|---|
| B1. Per-field discrete banks | a learned $Z_q^f$ per field (lookup table) | trivial; but unseen fields/sims need new rows — no generalization |
| **B2. Conditioned modulation (recommended)** | shared base + low-rank/FiLM correction from $[\theta_f\Vert z_{\text{sim}}]$ | cheap ($Z_q$ is only $N_q d_{\text{head}}$ numbers); **preserves cross-sim transfer** |
| B3. Hypernetwork (S8.4, most general) | $Z_q^f=H_\psi(\theta_f,z_{\text{sim}})$ generates the whole bank | fully general; least stable to train — build *after* B2 works |
| B4. Mixture of banks (MoE) | library of $M$ basis banks, router mixes by $z_{\text{sim}}$ | interpretable ("what operators does this physics use?"); middle ground |

**Reuse downstream.** $\theta_f$ is not tokenizer-only. It also (a) generates the field's decoder head ([[decoder-1.1]], same hypernetwork idea), and (b) **gates which edge types the field's tokens engage** ([[edge-generation-1.1]]) — the operational meaning of "force affordances": a field only opens the coupling edges its descriptor permits.

## Physics relevance

- **Representation-type tags = equivariance bookkeeping for free.** Encoding tensor rank + parity in $\theta_f$ tells the covariant channels how each field rotates/reflects, so equivariance is data the model *carries*, not structure baked into every layer ([[equivariant-gnns]], P9 covariant conditioning) — and it is what lets scalar, vector, and pseudovector fields share one interface without confusing how they transform.
- **Feature extraction matched to the physics.** Different physics has different salient sub-patch structure: discontinuities (shocks), smooth gradients (Stokes), neighbor statistics (particles). A simulation-conditioned $Z_q^f$ lets the tokenizer *probe for what matters here* rather than averaging a fixed set of features — the concrete graph-tokenizer instance of [[physics-conditioned-query-tokens]]'s "queries as physical operators."
- **Affordances encode the interaction repertoire.** Making "which forces a field can feel" a static descriptor (not something rediscovered every step) is the tokenizer-level seed of the typed coupling the backbone then learns — the field-internal analog of [[open-architectural-problems]] S4.1's typed cross-node messages.
- **Variable field cardinality → generality.** Because a field is *a descriptor*, adding or removing a field (or meeting an unseen one) is adding/removing/interpolating a $\theta_f$, not editing the model — the resolution of Problem 8 that keeps the same architecture spanning fluids, plasmas, and particle species ([[00-noether-1.1-overview]] design invariant 1).

## How fundamental constants are included

- **Constants enter the query bank through $z_{\text{sim}}$.** Since $z_{\text{sim}}$ contains $z_{\text{cond}}$ ([[conditioning-and-constants-1.1]]), the *same* field descriptor produces a different extractor at different Reynolds/Rayleigh/Mach — e.g. the velocity query bank shifts toward discontinuity-reading queries as Mach rises toward the compressible/shock regime. The constant tunes *how the field is tokenized*, not just how it evolves.
- **Field set + constants together = simulation identity.** This page is where "no hand-labeled simulation type" becomes concrete: $z_{\text{sim}}=\operatorname{pool}_f\theta_f\Vert z_{\text{cond}}$ is built entirely from which fields are present and which constants are non-zero — exactly the pair that physically distinguishes one system from another.
- **A vanishing constant should switch off the affordance it enables.** Buoyancy is a velocity↔temperature affordance gated by $\mathrm{Ra}$; as $\mathrm{Ra}\to0$ the conditioned gate on that coupling edge ([[edge-generation-1.1]]) should close — a checkable ablation, not an assumption, and the same discipline [[conditioning-and-constants-1.1]] applies everywhere.

## Open items / risks

- **Keep the modulation a correction, not a replacement.** The shared $Z_q^{\text{base}}$ is what lets a fluid-pretrained tokenizer transfer to a new fluid regime — the whole foundation-model premise. B1/naïve-B3 that fully replace the bank per simulation throw transfer away; B2's additive/FiLM correction preserves it.
- **Hypernetwork stability (B3).** Generating full weight matrices is powerful but brittle; the RBC log's own lesson is that elaborate architectural bets should *follow* the simple version — ship B2, earn B3.
- **Interpretability.** If you later want to *read off* which query-operators a physics uses, B4's basis mixture exposes that directly; B2 does not.
- **Descriptor grounding.** The structured block must be filled correctly per field (a mislabeled parity is a real bug); the free block $u_f$ risks absorbing spurious dataset correlations if field diversity is low — needs multi-system training to stay a genuine *field* descriptor rather than a dataset ID.

## See Also
- [[graph-tokenizer-1.1]] — the query-compression encoder whose $Z_q$ this conditions
- [[field-token-streams-1.1]] — the per-field token streams this descriptor tags
- [[physics-conditioned-query-tokens]] — "queries = physical operators," the idea instantiated here
- [[edge-generation-1.1]] — where affordances gate coupling edges
- [[decoder-1.1]] — the field's decoder head, generated from the same $\theta_f$
- [[intelligent-patching]] — the sibling tokenizer substep this page parallels structurally
