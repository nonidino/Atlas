# Normalization Scheme (initial model): Do We Need LayerNorm?

**Type:** Concept — initial model component (folder: Noether 1.0)
**Status:** Design decision. Short answer: **no standard LayerNorm; use input nondimensionalization + a structure-preserving gain normalization, and no mean-subtraction inside conservative blocks.**
**Related Concepts:** [[00-initial-model-overview]], [[initial-model-architecture]], [[symmetric-attention-physics]], [[graph-tokenizer]], [[pfm-interface-design]], [[structure-preserving-tokens]]
**Related Summaries:** [[anti-symmetric-dgn]], [[transformer-mathematical-framework]], [[equiformer-v3]], [[poseidon-pde-foundation-model]]

---

## The question

LayerNorm stabilizes training and is standard in transformers. But it does two things that are **actively harmful for a physics model** whose whole point is to preserve scale, position, and conserved quantities. So the question is not "LayerNorm: yes/no" but "what normalization preserves the physical structure we deliberately built into the tokens?"

Standard LayerNorm on a token $h \in \mathbb R^C$:

$$\mathrm{LN}(h) = g \odot \frac{h - \mu(h)}{\sigma(h)} + b, \qquad \mu(h) = \tfrac1C\textstyle\sum_c h_c,\ \ \sigma(h)^2 = \tfrac1C\textstyle\sum_c (h_c-\mu)^2.$$

Two problems:

1. **Mean-subtraction mixes physically distinct channels.** $\mu(h)$ averages across feature channels — but in a physics latent those channels encode different quantities (velocity components, pressure, the skew-symmetric "energy" directions of [[symmetric-attention-physics]]). Subtracting a cross-channel mean couples them in a physically meaningless way and, for vector/tensor features, **breaks rotation equivariance** ([[equivariant-gnns]], [[structure-preserving-tokens]]).
2. **Norm-division destroys amplitude = energy.** Dividing by $\sigma(h)$ throws away the magnitude of the state. But magnitude *is physics*: $\|v\|^2$ is kinetic energy, field amplitude is intensity. The skew-symmetric (conservative) channel is built precisely to **preserve norm across depth**; LayerNorm then immediately discards it. As [[anti-symmetric-dgn]] / [[antisymmetric-signed-attention-transformer]] note, LayerNorm "projects onto a manifold that interferes with the conservative structure."

There is also a theory view ([[transformer-mathematical-framework]]): LayerNorm is a *projection onto a constraint set* $S_1$ (the sphere). That is fine when the constraint is geometric/arbitrary; it is wrong when the radius (amplitude/energy) carries the physics.

---

## Why we still need *some* normalization

Without any normalization, deep stacks risk activation drift and the forward-Euler attention update can exceed its stable step ($\epsilon < 2/\|J\|_2$, [[anti-symmetric-dgn]]). And physical inputs span enormous dynamic ranges (the 30-orders-of-magnitude timescale problem, [[pfm-interface-design]]). So the goal is normalization that **tames range and stabilizes optimization without erasing scale, position, or equivariance.**

---

## The chosen scheme (three layers)

### 1. Nondimensionalization at the input (replaces most of LayerNorm's job)

Before tokenizing, rescale every quantity by its characteristic scale so the model sees $O(1)$ dimensionless numbers ([[pfm-interface-design]]):

$$\hat x = x/\phi_0,\quad \hat\ell = \ell/L_0,\quad \hat t = t/T_0,\quad \widehat{\Delta t} = \Delta t/T_0.$$

This is the *physically correct* way to handle dynamic range — it is what LayerNorm approximates blindly, done with the actual physical scales. The dimensionless governing numbers $(\mathrm{Re},\mathrm{Ma},\mathrm{Pr},\dots)$ are passed as conditioning. After this step the data is already well-scaled, removing most of the motivation for internal normalization.

### 2. Structure-preserving gain normalization inside the backbone (where needed)

Where internal stabilization is still required, use a normalization that **does not subtract the cross-channel mean** and **respects feature geometry**:

- **RMS-style gain on scalar features** — divide by root-mean-square, *not* by variance-after-mean-subtraction, and apply a learned per-channel gain:
$$\mathrm{RMSNorm}(h) = g \odot \frac{h}{\sqrt{\tfrac1C\sum_c h_c^2 + \varepsilon}}.$$
No mean is removed, so no spurious channel mixing; it only controls overall magnitude. Use a *grouped* RMS (per field-type group) so distinct physical quantities are normalized separately, not pooled.
- **Equivariant norm on vector/tensor features** — normalize by the rotation-invariant magnitude only, leaving direction intact (the EquiformerV3 merged-norm idea, [[equiformer-v3]]); this preserves equivariance that LayerNorm would break.
- **Conditional gain (AdaLN-style), not conditional mean** — let the gain $g$ (and step size) depend on the conditioning $(\widehat{\Delta t}, \mathrm{Re}, \dots)$, generalizing Poseidon's lead-time-conditioned LayerNorm ([[poseidon-pde-foundation-model]], [[structure-preserving-tokens]]) — but condition the *scale*, never inject a cross-channel mean shift.

### 3. No normalization inside the conservative attention blocks

Within the symmetric/skew-symmetric attention update ([[symmetric-attention-physics]]), apply **no** LayerNorm/RMSNorm: the skew-symmetric linear part already preserves the $L^2$ norm by construction, and the bounded $\tanh$ score plus the stable Euler step ($\epsilon < 2/\|J\|_2$) keep activations controlled. Normalization here would defeat the exact reason the block exists. If downstream calibration is needed, apply an *output* RMS/equivariant gain **after** the block, never inside it.

---

## Summary table

| Stage | Normalization | Rationale |
|---|---|---|
| Input | Nondimensionalization (physical scales) | correct dynamic-range handling; preserves regime info as conditioning |
| Node/edge encoder | grouped RMS / equivariant gain | tame range without channel-mixing or breaking equivariance |
| Conditioning | AdaLN-style gain (scale only) | regime-aware scaling; never a cross-channel mean shift |
| Conservative attention block | **none** | skew-symmetric part preserves norm; bounded score + stable step suffice |
| Block output (if needed) | output RMS / equivariant gain | optional downstream calibration |
| **After output gain, before decode** | **conservation projection (S1.1)** — *not* a normalization | must be the **last magnitude-affecting op**: an RMS/energy gain applied *after* it would rescale magnitude = energy and undo the projection |

### Ordering constraint with the S1.1 conservation projection

Because the structure-preserving scheme deliberately keeps **amplitude = energy** (grouped-RMS gain, no mean subtraction), the mandatory conservation projection ([[open-architectural-problems]] S1.1, [[initial-model-architecture]] Stage 5b) must run **after** any output RMS/AdaLN gain — a gain applied *after* the projection would rescale magnitude and silently break the enforced energy. Order: backbone → output gain → **conservation projection** → decode (→ diffuse residual → **re-project**). Normalization and conservation enforcement are the same concern — *magnitude is physical* — seen at two stages, so their ordering is not free.

---

## [AI Inference]

**[AI Inference]:** The deepest reason to avoid standard LayerNorm is that the initial model treats **amplitude as a state variable, not a nuisance scale.** Energy, intensity, and conserved norms live in the magnitude; a model that conserves energy cannot also normalize it away every layer. This predicts a clean ablation: replacing the structure-preserving scheme with standard LayerNorm should specifically degrade *energy-conservation* and *long-horizon* metrics (while possibly leaving single-step accuracy intact), because LayerNorm silently re-injects the dissipation the architecture was designed to remove.

**[AI Inference]:** Nondimensionalization is "LayerNorm done right." Both map wildly-scaled inputs to $O(1)$; the difference is that LayerNorm uses *empirical, per-token, cross-channel statistics* (physically meaningless) while nondimensionalization uses *physical characteristic scales* (Reynolds, Mach, characteristic time). Doing the rescaling with real physics at the input is what lets the model *drop* internal normalization that would otherwise corrupt the conserved structure — the normalization choice and the conservation goal are the same decision viewed twice.

---

## See Also

- [[00-initial-model-overview]] / [[initial-model-architecture]] — where normalization fits
- [[symmetric-attention-physics]] — why conservative blocks reject LayerNorm
- [[graph-tokenizer]] — the latents being normalized
- [[pfm-interface-design]] — nondimensionalization and dimensionless-number conditioning
- [[anti-symmetric-dgn]] / [[antisymmetric-signed-attention-transformer]] — norm-preserving skew-symmetric dynamics
- [[transformer-mathematical-framework]] — LayerNorm as projection onto a constraint set
- [[equiformer-v3]] — equivariant/merged normalization
- [[poseidon-pde-foundation-model]] — lead-time-conditioned LayerNorm (conditional gain precedent)
- [[structure-preserving-tokens]] — coordinate/scale-aware conditioning
