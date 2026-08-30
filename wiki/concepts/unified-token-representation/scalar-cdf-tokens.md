# Scalar CDF Tokens (Distribution-Aware Scalar Encoding)

**Type:** Token representation (folder: unified-token-representation)
**Used by:** [[aion-1-astronomy]] (scalar measurements)
**Related:** [[00-token-representation-overview]], [[vector-quantized-tokens]], [[pfm-interface-design]], [[structure-preserving-tokens]]

---

## Intuition

Most token-representation work is about *fields*. But a physics foundation model also needs to ingest **scalars**: dimensionless governing numbers (Reynolds, Mach, Prandtl, Péclet), material constants (viscosity, conductivity), boundary values, time. Scalars are awkward — they span many orders of magnitude, are skewed, and a raw float fed into a network is dominated by its scale. The CDF trick: **map each scalar through its own empirical cumulative distribution function, so it becomes a uniform $[0,1]$ rank**, then bin that into a discrete token. Now "the 90th percentile Reynolds number" is encoded the same way regardless of whether Re ranges over $10^2$ or $10^8$ — the representation is **scale-free and distribution-matched by construction**. This is how [[aion-1-astronomy]] folds heterogeneous scalar measurements into the same discrete vocabulary as its image/spectrum tokens.

---

## Mathematics

Let scalar $s$ have empirical CDF $F(s)=\Pr[S\le s]$ estimated from the training distribution. The CDF transform (probability integral transform) gives a uniform variable:

$$\tilde s = F(s)\in[0,1],\qquad \tilde s\sim \mathcal U[0,1]\ \text{(if $F$ is the true CDF)}.$$

**Uniform binning of the CDF** into $B$ bins yields the token index:

$$\text{token}(s) = \big\lceil B\cdot F(s)\big\rceil \in \{1,\ldots,B\}.$$

Because bins are equal in *probability mass* (uniform in CDF space), each token carries equal information and no bin is starved — automatically adapting bin *widths* in raw-value space to the data density (narrow bins where data is dense, wide where sparse). The token embeds via a standard lookup table into the shared vocabulary. Inversion (for generation) uses the quantile function $s=F^{-1}(\tilde s)$.

Contrast naive alternatives: raw $s$ (scale-dominated), $\log s$ (helps range but not skew, undefined for $s\le 0$), min-max normalization (sensitive to outliers). The CDF transform handles range, skew, and outliers in one move.

---

## Pros

- **Scale- and distribution-invariant** — arbitrary dynamic range and skew handled uniformly; ideal for dimensionless numbers spanning many decades.
- **Equal-information bins** — uniform-in-CDF binning maximizes entropy per token; no wasted or starved codes.
- **Shared-vocabulary compatible** — produces discrete tokens that live in the same space as field tokens, so a transformer treats parameters and fields identically (enables masked any-to-any: predict a parameter from fields, or condition fields on a parameter).
- **Robust to outliers** — extreme values map to the boundary bins rather than blowing up activations.
- **Directly addresses the scale-awareness gap** flagged in [[multimodal-tokenization]] — a clean way to give a PFM its governing parameters.

## Cons

- **Discretization loss** — binning rounds the scalar; fine parameter sensitivity (e.g. near a bifurcation/critical Reynolds number) is blurred unless $B$ is large.
- **Distribution-dependent / non-stationary** — $F$ is fixed from the training set; a test scalar outside the training range saturates at a boundary bin (no extrapolation), and shifting distributions require recomputing $F$.
- **Loses physical units/relations** — the CDF rank discards the *metric* structure: the model can't natively know that doubling Re is "twice as turbulent"; only the *ordering* survives unless the embedding relearns magnitude.
- **Independent per-scalar transform** — applying $F$ to each scalar separately ignores correlations between parameters (e.g. Re and Ma are not independent in a given regime).
- **Needs enough data to estimate $F$** reliably, especially in the tails.

---

## Relationship to other representations

- vs. [[vector-quantized-tokens]]: CDF tokens are the **1D, distribution-aware special case** of discrete tokenization — binning in CDF space instead of nearest-codebook in latent space.
- vs. continuous conditioning ([[poseidon-pde-foundation-model]]'s lead-time LayerNorm): Poseidon injects the scalar *time* as a **continuous affine modulation** of normalization — the opposite philosophy (continuous, metric-preserving, no binning). For parameters where smooth sensitivity matters, continuous conditioning may beat CDF binning; for vocabulary-unification and masked inference, CDF binning wins. See [[pfm-interface-design]].

**[AI Inference]:** A PFM should likely use **CDF tokens for *categorical/identity* scalars** (which PDE, which field type, which BC class — where a shared vocabulary and masked inference are the point) but **continuous conditioning (FiLM / lead-time-style LayerNorm) for *quantitative* governing numbers** (Re, Ma, Pr — where smooth metric sensitivity matters). Mixing the two — discrete tokens for type, continuous modulation for magnitude — is a concrete design for [[pfm-interface-design]]'s 5-input interface.

**[AI Inference]:** Replacing the per-scalar marginal CDF with a **joint copula transform** over correlated parameters would preserve inter-parameter structure (Re–Ma–Pr correlations), turning a bag of independent scalar tokens into a representation that respects the physical parameter manifold — a small but principled upgrade toward [[structure-preserving-tokens]].

---

## See also

- [[00-token-representation-overview]] — hub; criterion 4 (scale awareness)
- [[aion-1-astronomy]] — CDF scalar tokenization in practice
- [[vector-quantized-tokens]] — the general discrete-token family
- [[pfm-interface-design]] — how parameters/BCs/time enter a PFM
- [[poseidon-pde-foundation-model]] — continuous lead-time conditioning (the alternative)
- [[structure-preserving-tokens]] — parameter-manifold-aware encoding
