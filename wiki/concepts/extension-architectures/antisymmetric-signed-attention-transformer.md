# Antisymmetric / Signed-Edge Attention Transformer

**Type:** Architecture Concept / Proposal
**Status:** New synthesis page (2026-05-18)
**Related Concepts:** [[transformer-architectures]], [[arch-autoregressive-transformer]], [[arch-diffusion-backbone]], [[autoregressive-rollout-stability]], [[equivariant-gnns]], [[possible-architectures]]
**Related Summaries:** [[anti-symmetric-dgn]], [[transformers-particle-systems-clustering]], [[transformer-mathematical-framework]], [[dynami-cal-graphnet]]

---

## Motivation

The mathematical-physics view of transformers established by [[transformers-particle-systems-clustering]] and [[transformer-mathematical-framework]] proves two things:

1. **Tokens are particles, attention is an interaction force.** A transformer is a mean-field interacting particle system on the unit sphere $\mathbb{S}^{d-1}$, with attention playing the role of a pairwise force.

2. **Standard softmax attention is structurally wrong for physics.** Three pathologies:

   - **Purely attractive.** Softmax weights are non-negative, so the "force" $\sum_j A_{ij}(V_j - V_i)$ is always an attractive pull toward neighbors. Real physical interactions have both attractive *and repulsive* components — without repulsion, no stable structure can form.
   - **Not antisymmetric.** $A_{ij} \neq A_{ji}$ because softmax normalizes row-wise. Newton's 3rd law $\vec{F}_{ij} = -\vec{F}_{ji}$ is violated.
   - **Dissipative.** Theorem 6.1 of [[transformers-particle-systems-clustering]] proves that for $d \geq 3$ the dynamics drive all tokens to a single Dirac mass (clustering). The interaction energy $\mathsf{E}_\beta[\mu]$ is *monotonically increasing*. Information from layer 0 is gradually lost — the over-smoothing / rank-collapse pathology.

If a transformer is to be used for physics — where tokens represent particles, mesh points, or field values, and "attention" represents the interaction — these three pathologies are direct obstacles.

[[anti-symmetric-dgn]] (A-DGN) gives the rigorous mathematical fix on the graph side: replace the dissipative message-passing flow with an **antisymmetric ODE** whose Jacobian eigenvalues have zero real part (skew-symmetric structure), producing non-dissipative information propagation across arbitrary depth.

This page formalizes the translation to transformers: an **antisymmetric, signed-edge attention mechanism** that turns the transformer's token dynamics into a Hamiltonian-like flow rather than a dissipative aggregation.

---

## The Three Architectural Fixes

### Fix 1: Symmetric Attention Score (replaces softmax)

Replace the row-normalized softmax attention score with a **symmetric** function of the query and key:

$$A_{ij} = A_{ji} = s(Q_i, K_j) = s(Q_j, K_i)$$

Simplest choice: $s(Q_i, K_j) = \frac{1}{2}\left[\langle Q_i, K_j\rangle + \langle Q_j, K_i\rangle\right] / \sqrt{d_k}$, with optional normalization (e.g., divide by $\sum_k s(Q_i, K_k)$ to maintain bounded magnitudes — but apply the *same* normalization to both $A_{ij}$ and $A_{ji}$). For a strict symmetric form, set $Q = K$ (which is common in self-attention with tied weights).

This eliminates the row-stochastic asymmetry that breaks Newton's 3rd law.

### Fix 2: Antisymmetric Aggregation (Newton's 3rd law)

Replace the standard value aggregation $\sum_j A_{ij} V_j$ with an **antisymmetric** value aggregation. Two equivalent formulations:

**(a) Force-style aggregation (closer to physics):**

$$\Delta h_i = \sum_j A_{ij}\, (V_j - V_i)$$

For symmetric $A_{ij}$, the "force" $\vec{F}_{ij} = A_{ij}(V_j - V_i)$ satisfies $\vec{F}_{ji} = A_{ji}(V_i - V_j) = -\vec{F}_{ij}$ exactly. Newton's 3rd law is restored.

Total "momentum" $\sum_i \Delta h_i$ telescopes to zero:

$$\sum_i \sum_j A_{ij}(V_j - V_i) = \sum_{i,j} A_{ij} V_j - \sum_{i,j} A_{ij} V_i = 0 \quad \text{(by symmetry of } A\text{)}$$

This is **exact conservation** of the aggregate hidden-state mean — the transformer analog of momentum conservation.

**(b) Skew-symmetric value transformation (A-DGN-style):**

Apply an antisymmetric linear transformation $W - W^\top - \gamma I$ to the values before aggregation:

$$\Delta h_i = (W - W^\top - \gamma I)\, h_i + \sum_j A_{ij}\, V_j$$

The skew-symmetric linear part has purely imaginary Jacobian eigenvalues (with the $-\gamma I$ shift giving controlled slight dissipation), ensuring non-dissipative information propagation across depth. This is the direct translation of A-DGN to the transformer setting.

Combining both (a) and (b) gives an attention mechanism whose dynamics are antisymmetric at both the per-edge (Newton's 3rd) and per-layer (information-preserving) levels.

### Fix 3: Signed Coupling (repulsive interactions)

Allow $A_{ij}$ to be **signed**: $A_{ij} \in \mathbb{R}$, not constrained to $[0, 1]$. Negative $A_{ij}$ represents repulsion; positive represents attraction.

Practical parameterization: $A_{ij} = \tanh(s(Q_i, K_j))$ — symmetric, signed, bounded in $[-1, 1]$. The bound prevents runaway exponential growth that an unbounded signed attention would produce.

With signed $A_{ij}$ and the force-style aggregation, the transformer dynamics support both attractive clustering (positive $A$ between similar tokens) and repulsive separation (negative $A$ between dissimilar tokens) — exactly the structural ingredient that allows stable patterns to form in physical systems.

---

## The Resulting Attention Update

Combining all three fixes, one round of antisymmetric attention is:

$$h_i^{\ell+1} = h_i^\ell + \epsilon\, \sigma\!\left[(W - W^\top - \gamma I) h_i^\ell + \sum_{j} A_{ij}\, (V_j^\ell - V_i^\ell)\right]$$

with:

- $A_{ij} = \tanh\!\left(\frac{1}{2}(\langle Q_i, K_j\rangle + \langle Q_j, K_i\rangle) / \sqrt{d_k}\right)$ — symmetric, signed
- $W - W^\top - \gamma I$ — skew-symmetric linear part for non-dissipative information flow
- $\epsilon < 2 / \|J\|_2$ — forward Euler step size for stability
- $\sigma$ — Lipschitz activation (e.g., GELU, $\text{tanh}$)

Compare to standard self-attention:

$$h_i^{\ell+1} = h_i^\ell + \text{LayerNorm}\!\left(\sum_j \text{softmax}_j(Q_i K_j^\top / \sqrt{d_k}) V_j\right)$$

The differences: symmetric vs. asymmetric attention score; signed vs. non-negative score; force-style (relative) vs. absolute value aggregation; skew-symmetric linear part replacing the residual; explicit step size $\epsilon$.

---

## Theoretical Properties

### Information Preservation (from A-DGN Theorem)

Following the A-DGN stability analysis ([[anti-symmetric-dgn]]):

**Property 1:** The Jacobian of the antisymmetric attention update has eigenvalues with real part $\leq -\gamma$. With $\gamma = 0$, the dynamics are exactly non-dissipative; with small $\gamma > 0$, they are slightly dissipative with controllable decay rate.

**Property 2:** Information from layer 0 is preserved up to $e^{-\gamma L \epsilon}$ at layer $L$. For $\gamma L \epsilon \ll 1$, tokens retain their fine-grained distinctions across the entire stack — **no rank collapse, no over-smoothing**.

**Property 3:** Gradients are bounded; no vanishing or exploding gradients in deep stacks. The architecture trains stably at 20+ layers.

### Newton's 3rd Law (Exact Aggregate Conservation)

By construction, the force-style aggregation gives $\vec{F}_{ji} = -\vec{F}_{ij}$ for every edge. Summing over all edges:

$$\sum_i \Delta h_i = \sum_i \sum_j A_{ij}(V_j - V_i) = 0$$

The aggregate hidden state mean is **exactly conserved** across the attention update. For physics applications where tokens represent particles, this corresponds to exact conservation of total momentum.

### Repulsive Equilibria

With signed $A_{ij}$, the dynamics support equilibria that are *not* the Dirac-mass clustering of standard attention. Specifically, the modified interaction energy

$$\mathsf{E}_A[\mu] = -\frac{1}{2}\iint A(x, y)\, \langle V(x), V(y)\rangle \, d\mu(x) d\mu(y)$$

(with signed $A$) has critical points that include **structured configurations** — multiple clusters, periodic patterns, stable patterns separated by repulsive forces. This is analogous to the equilibria of physical systems with Lennard-Jones potentials (short-range repulsive + long-range attractive).

---

## Why This Matches the Wiki's "Physics GNN" Aspiration

The repeated theme across [[equivariant-gnns]], [[arch-gnn-physics-bottleneck]], and the multi-scale / Hamiltonian / Noether concept pages is that **transformer attention is the wrong shape for physics, but a fixable shape**. Specifically:

| Transformer pathology                   | Physics requirement                    | Fix                                                     |
| --------------------------------------- | -------------------------------------- | ------------------------------------------------------- |
| Attention purely attractive             | Forces signed (attractive + repulsive) | Signed $A_{ij}$                                         |
| Attention not antisymmetric             | Newton's 3rd law                       | Symmetric $A$ + force-style aggregation                 |
| Dynamics drive tokens to single cluster | Stable structure, conserved quantities | Skew-symmetric linear part                              |
| All-pairs $O(N^2)$ cost                 | Hierarchical multi-scale structure     | Sparse attention + multipole hierarchy                  |
| No conservation guarantees              | Exact conservation needed              | Combine with Dynami-CAL or Hamiltonian parameterization |

This page addresses the first three; [[multiscale-hierarchical-gnn]] addresses the fourth; [[hamiltonian-message-passing]] and [[action-based-noether-enforcement]] address the fifth.

---

## Architectural Pipeline

```
Input: token sequence h_1, ..., h_N with optional positional encoding

For each layer ℓ = 1, ..., L:
  1. Symmetric attention score
     Q_i = W_Q h_i,  K_i = W_K h_i,  V_i = W_V h_i
     A_ij = tanh((⟨Q_i, K_j⟩ + ⟨Q_j, K_i⟩) / (2√d_k))    # symmetric, signed

  2. Force-style aggregation
     F_i = Σ_j A_ij · (V_j - V_i)                          # antisymmetric per edge

  3. Skew-symmetric linear part
     L_i = (W - W^T - γI) h_i

  4. Update
     h_i ← h_i + ε · σ(F_i + L_i)                          # forward Euler

  5. Optional: feedforward MLP (residual, as in standard transformer)
     h_i ← h_i + MLP(h_i)
```

Notes:

- **No LayerNorm.** The skew-symmetric linear part already preserves L2 norm of $h_i$; LayerNorm would project onto a manifold that interferes with the conservative structure. Optional RMSNorm at the output can be applied if scale calibration is needed downstream.

- **Positional encoding.** Compatible with any positional encoding; for physics applications, RoPE or distance-based encodings are natural choices.

- **Multi-head attention.** Each head computes its own $A_{ij}^{(h)}$ — the symmetric/signed property must hold per-head. Heads are combined by averaging or by an output projection chosen to preserve antisymmetry.

- **Causal masking.** Compatible with autoregressive generation: set $A_{ij} = 0$ for $j > i$. The aggregate conservation no longer holds globally (only over the visible past), but per-edge antisymmetry still holds where edges exist.

---

## Combination with Existing PFM Architectures

### Replaces Standard Attention in Arch 1 (AR Transformer)

The AR Transformer of [[arch-autoregressive-transformer]] uses standard factorized space-time attention. Replacing this with antisymmetric attention gives a token rollout that:

- Preserves per-token distinctions across depth (no over-smoothing).
- Conserves aggregate hidden-state quantities corresponding to physical conservation laws.
- Allows signed token-token coupling (attractive between similar, repulsive between distinct — natural for distinguishing fluid regimes from solid regimes in mixed-phase simulations).

### Replaces DiT Attention in Arch 2 (Diffusion Backbone)

The DiT denoiser of [[arch-diffusion-backbone]] uses standard self-attention. Antisymmetric attention preserves token-level information across the denoising stack — preventing the score function from collapsing to a degenerate one-cluster predictor.

### Complement to Arch 5 (GNN-PB)

For transformer-style "fully-connected GNN" applications (small to medium tokens, $N \lesssim 10^4$), antisymmetric attention is the natural transformer-side analog of GNN-PB's antisymmetric edge frame. Both enforce Newton's 3rd law via different mechanisms.

---

## Open Problems

1. **Exact equivalence to A-DGN proofs.** A-DGN's stability theorem applies to the message-passing ODE; the transformer with antisymmetric attention has additional structure (the softmax-replacement, the multi-head decomposition, the FFN). Whether the stability theorem transfers exactly requires careful analysis — particularly for the symmetric-but-non-softmax attention score.

2. **Expressivity loss.** Restricting attention to symmetric scores and antisymmetric aggregation reduces the expressive class. Whether this restriction limits performance on standard NLP / vision benchmarks (and how much) is an open empirical question. For physics applications the restriction is principled (the pathologies it eliminates are real); for general-purpose ML it may be costly.

3. **Per-head antisymmetry.** Multi-head attention combines per-head outputs via a learned projection $W_O$. Preserving antisymmetry through this projection requires the output projection to be itself structured (e.g., orthogonal, or restricted to antisymmetric combinations). Practical implementations need careful design.

4. **Combination with FlashAttention.** Modern transformer training relies on FlashAttention for memory efficiency. The symmetric attention computation has the same memory pattern as standard attention; FlashAttention should adapt with minor modification. The signed-bounded $\tanh$ replaces softmax; this is mechanically simpler than softmax (no normalization across rows).

5. **Empirical demonstration.** No published transformer has the exact combination proposed here. The closest are:
   - **AntisymmetricRNN** (Chang et al. 2019, arXiv:1902.09689) — temporal sequence version
   - **A-DGN** (Gravina et al. 2023) — graph message-passing version
   - **Long Expressive Memory (LEM)** (Rusch et al. ICLR 2022) — multiscale antisymmetric ODE for RNNs
   - **Reversible Transformers** (Kitaev et al. 2020) — invertibility via Hamiltonian coupling, but no antisymmetric structure
   
   A direct implementation and empirical comparison against standard self-attention on physics benchmarks (Burgers', NS, MHD) would be the natural next experiment.

6. **Theoretical analysis of the clustering theorem with signed $A$.** [[transformers-particle-systems-clustering]]'s clustering theorem applies to softmax (purely positive) attention. The corresponding analysis for signed attention with symmetric scores has not been done. The expected result is that the energy functional becomes non-monotonic and admits structured equilibria, but rigorous proofs are missing.

---

## Benchmark Adaptation

For the 1D Burgers' benchmark of [[possible-architectures]]:

- Replace the AR Transformer's space-time attention with antisymmetric attention.
- Keep all other architectural choices (tubelet patching, position encoding, FFN, etc.) the same.
- Compare against the standard AR Transformer baseline at the same parameter count.

**Hypothesis:** antisymmetric attention transformer should match standard transformer accuracy at short rollouts, and *outperform* at long rollouts (where the over-smoothing / cumulative-error pathology of standard attention becomes the bottleneck). This is testable on Burgers' with rollout horizons of 100+ steps.

**Secondary hypothesis:** the antisymmetric architecture trains stably at 2-4× depth compared to the baseline (matching A-DGN's empirical finding on graph benchmarks).

---

## [AI Inference]

**[AI Inference]:** The antisymmetric attention transformer is what you get when you take the [[transformers-particle-systems-clustering]] particle-system analogy literally and design backward from "make the dynamics physically correct." The clustering theorem of that paper proves the *current* design is non-physical (drives to Dirac mass = collapses to one cluster); the antisymmetric design produces dynamics that admit stable patterns and conserve aggregate quantities. The clustering theorem becomes inapplicable because the energy functional is no longer monotone, and the dynamics now have the structure of a Hamiltonian-with-controlled-dissipation system (port-Hamiltonian).

**[AI Inference]:** The combination of antisymmetric attention + the [[multiscale-hierarchical-gnn]] multipole structure + the [[hamiltonian-message-passing]] potential-derived forces gives a transformer-shaped architecture that nonetheless has the conservation, multi-scale, and stability properties of a Dynami-CAL GNN. This is the "physics transformer" the wiki has been circling toward — *not* a transformer with a physics loss bolted on, but a transformer whose attention mechanism is intrinsically physics-shaped. No published work realizes this combination.

**[AI Inference]:** The mathematical-physics view also clarifies why standard transformers work as well as they do for *language* despite the clustering pathology: the metastable state described in [[transformers-particle-systems-clustering]] is exactly what language needs — distinct token clusters representing distinct semantic groups, slowly merging over depth. Language tolerates clustering because clustering corresponds to abstraction. Physics does not tolerate clustering because physics needs to preserve fine spatial structure. The architectural fix is therefore physics-specific; standard transformers should remain the right choice for language. The "transformer for physics" should be an explicitly different architecture, not just a retrained transformer.

---

## Position on the Physics Encoding Spectrum

| Component | Level |
|---|---|
| Symmetric attention score | 4 (architectural soft bias against asymmetric dynamics) |
| Force-style antisymmetric aggregation | 5 (hard aggregate-quantity conservation) |
| Skew-symmetric linear part | 4-5 (architectural enforcement of non-dissipative dynamics) |
| Signed coupling | 4 (architectural support for repulsive interactions) |

The antisymmetric attention transformer is **level 4-5** on the physics encoding spectrum — comparable to GNN-PB but on the transformer side. Combined with hierarchy (multipole) and Hamiltonian parameterization, it climbs to level 5 throughout.

---

## Cross-Links

- [[anti-symmetric-dgn]] — rigorous foundation
- [[transformers-particle-systems-clustering]] — proves the clustering pathology this concept fixes
- [[transformer-mathematical-framework]] — continuous-IDE view of transformers
- [[dynami-cal-graphnet]] — graph-side antisymmetric architecture
- [[transformer-architectures]] — broader transformer landscape
- [[arch-autoregressive-transformer]] — Arch 1 (this concept upgrades attention)
- [[arch-diffusion-backbone]] — Arch 2 (this concept upgrades DiT)
- [[autoregressive-rollout-stability]] — error accumulation problem
- [[equivariant-gnns]] — physics inductive bias taxonomy
- [[multiscale-hierarchical-gnn]] — composition partner (multi-scale)
- [[hamiltonian-message-passing]] — composition partner (Hamiltonian)
- [[action-based-noether-enforcement]] — composition partner (action-based)
- [[possible-architectures]] — broader benchmark plan
