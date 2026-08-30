# Portion 5 — Backbone

**Type:** Implementation spec — agent task (folder: Noether 1.1 / Noether 1.1 implementation)
**Portion of:** [[00-implementation-plan]] (Phase II). **Depends on:** [[impl-tokenizer-descriptors]], [[impl-edge-generation]]. **Unblocks:** [[impl-decoder]], [[impl-discovered-conservation]].
**Design pages:** [[backbone-1.1]] (authoritative), [[multihead-attention]], [[normalization-scheme]], [[skew-symmetric-attention]]. Trains in **P1** ([[training-scheme-1.1]]).

---

## Objective
The dynamics core: $L$ stacked port-Hamiltonian layers over the typed multigraph. Each layer = one forward-Euler step (depth = latent time), with per-edge-type symmetric attention (tied $Q{=}K$), a gated antisymmetric flux, a Cayley-orthogonal skew self-term, learnable dissipation $\gamma_\ell$, and an F1 semi-orthogonal FFN sub-step. **This is a deep stacked-attention model — $L{=}12$ (medium) / $24$ (large) layers.**

## Deliverables (`noether11/core/backbone.py`)
- `TypedAttention` per edge type τ: `q_i=k_i=W^{(τ,k)}h_i`, `v_i=W_V^{(τ,k)}h_i`, `A_ij=tanh(q_i·q_j/√d_head)`; message `Σ_j A_ij (v_j − g_τ v_i)`, summed over heads then up-projected by `W_O^{(τ)}`, summed over types. **Head sum scaled `1/√H` at init** (avoids tanh saturation; ReZero absorbs steady-state scale — [[backbone-1.1]]).
- `Gate g_τ = clamp(a_τ, 0, 1)` (hardtanh) — reaches **exactly 1/0 on open sets**; init `a_τ` so `g=1` on same-field edges; **frozen at 1 through P1–P2**, freed at P3 (harness controls `requires_grad`).
- `CayleySkew`: `W_skew = (I−S)(I+S)^{-1}` from learned skew `S` (orthogonal, `‖W_skew−W_skexᵀ‖₂≤2`); self-term `(W_skew − W_skewᵀ − γ_ℓ I)h_i`.
- `F1FFN`: semi-orthogonal expand→contract (`W1ᵀW1=I`, `W2W2ᵀ=I`), GELU, renorm; a second ReZero-gained Euler sub-step.
- `Layer`: `h += α_ℓ ε tanh[self + Σ_τ g_τ W_O^{(τ)} Σ_j A_ij(v_j − g_τ v_i)]`; then `h += α'_ℓ ε W2 renorm(GELU(W1 h))`. ReZero `α,α'` init 0. **No LayerNorm.**
- **Vector-constant coupling** ([[backbone-1.1]], [[conditioning-and-constants-1.1]]): the covariant channel enters (i) as invariant contractions `⟨γ_vec, x_j−x_i⟩`, `‖γ_vec‖` modulating `A^{(τ)}`/`g_τ` (anisotropy), and (ii) as an additive vector source term on the forced stream's value channel (body force) — the same site `g_τ` opens below 1. Default (no hard equivariance): vector enters as ordinary component channels.

## Interface contract
- `Backbone(cfg)(tokens, graph, z_cond) -> evolved_tokens` (same shape as input tokens; `h_hi` rides alongside untouched).
- Euler-stability invariant: `ε < 2/‖W_skew‖₂`; enforce via the ε warmup schedule from cfg.

## Build steps
1. `CayleySkew` (with the `(I+S)^{-1}` solve) + stability assert.
2. `TypedAttention` with tied Q=K, per-type params, head-sum `1/√H`.
3. `Gate` (clamp) with the freeze/free hook.
4. `F1FFN` (Stiefel init via QR; periodic re-orthogonalization every 100 steps — reuse 1.0 recipe).
5. Assemble `Layer` and stack `L`; ReZero gains.
6. Wire vector-constant entry points (guarded by a `hard_equivariance` cfg flag).

## Acceptance tests
- **Norm/stability:** with ReZero at init the block is ≈ identity; `‖W_skew‖₂ ≤ 2` holds through 1k optimizer steps.
- **Gate exactness:** a unit test that a same-field edge with `a_τ≥1` gives `g_τ==1.0` exactly and momentum-telescoping `Σ_i Δ_i ≈ 0` pre-nonlinearity.
- **F1 vs F3 ablation:** F1 shows higher effective-rank-of-latents at depth than an FFN-free variant ([[open-architectural-problems]] Problem 12).
- **P1 target (milestone M2):** single-step rel-$L^2 < 0.05$ on 2D NS.

## Pitfalls
- Summing heads is correct (forces add); **do not** divide by H — only the `1/√H` init scale, and never a *learned* per-head mix ([[multihead-attention]] $W_O$ warning; [[backbone-1.1]]).
- Conservation at `g=1` is only exact *pre-tanh* (depth, not trajectory) — the *hard* guarantee is [[impl-discovered-conservation]]'s projection. Don't over-claim here.

## See Also
- [[backbone-1.1]] (authoritative math) · [[impl-discovered-conservation]] (the hard-conservation partner) · [[impl-decoder]] (consumes evolved tokens)
