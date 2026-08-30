# Portion 8 — Discovered conservation

**Type:** Implementation spec — agent task (folder: Noether 1.1 / Noether 1.1 implementation)
**Portion of:** [[00-implementation-plan]] (Phase III). **Depends on:** [[impl-backbone]], [[impl-decoder]], [[impl-training-harness]]. **Unblocks:** [[impl-diffusion]] (ordering), [[impl-eval-benchmarks]].
**Design pages:** [[discovered-conservation-1.1]] (authoritative), [[action-based-noether-enforcement]]. Trains in **P3** ([[training-scheme-1.1]]).

---

## Objective
Let the model **discover which quantities are conserved and enforce each in proportion to how invariant it is in data** — the learned-soft penalty always on, threshold-gated-hard projection where held-out drift < ε. First cut: **local divergence discovery** on NS velocity, then RBC's global invariants.

## Deliverables (`noether11/conserve/`)
- `candidates.py` — candidate functionals `{C_k}`: generic **symbolic seeds templated on `θ_f` rank/parity** (mass-like `Σ_f 1[rank=0]∫f`, momentum-like, divergence-like) + parameterized readouts `Ĉ_k(h)=w_kᵀh_top`. **Never keyed to a named field** (Tier-B generality, [[discovered-conservation-1.1]]).
- `discovery.py` — the SFA search: minimize `D_k = E[(C_k(x_{t+1})−C_k(x_t))²]` s.t. non-triviality + orthogonality. **Whiten first** (Portion-critical, [[discovered-conservation-1.1]] "Condition the search"): form the generalized eigenproblem `(Σ_Δ, Σ)`, **shrinkage/reduced-rank whiten** `(Σ+εI)^{-1}` (the redundant features make Σ rank-deficient), solve for the smallest generalized eigenvectors. Whitening stats conditioned on `z_cond`.
- `strength.py` — learned `λ_k` and `σ_k(z_cond)` (uncertainty-weighted); the held-out-drift gate that promotes a `C_k` to hard.
- `projection.py` — the S1.1 projection `x* = x̂ − M^{-1}Aᵀ(AM^{-1}Aᵀ)^{-1}(Ax̂−b)` (reuse 1.0 `projection.py`); linear/global forms cheap, divergence via Leray only if hard-gated; **re-run every rollout step** for hard-gated `C_k`.
- Sets `decode_mode` on [[impl-decoder]] when divergence is confidently ≈ 0.

## Interface contract
- `Conservation(cfg).soft_loss(traj) -> scalar` (Group C); `Conservation.project(state, z_cond) -> state*` (hard, live at rollout).
- Strength is a **function of `z_cond`**, never a per-regime constant (Tier-A generality).
- Passive mode: measure `C_k(x_t)` through a rollout as a diagnostic without correcting.

## Build steps
1. Candidate seeds (rank/parity-templated) + parameterized readouts.
2. Discovery: assemble `Σ`, `Σ_Δ`; **shrinkage whitening**; generalized eigensolve; non-triviality (unit variance from whitening) + orthogonality; nonlinear variant = decorrelation layer + penalty.
3. `λ_k`, `σ_k(z_cond)`, held-out gate.
4. Projection (reuse + generalize 1.0); Leray path for divergence when hard-gated.
5. Wire into the harness Group C (P3) and the rollout loop (hard projection).

## Acceptance tests
- **Divergence discovery (milestone M4):** on incompressible NS, `∇·u` is discovered and its `λ` high; on **compressible** data the same machinery drives `λ_{∇·u}→0` with **no code change**.
- **Conditioning:** the eigensolve is stable despite redundant features (a test that removing whitening degrades the smallest-eigenvector recovery).
- Hard-projected mass/momentum drift < 1e-4 over full rollout (reuse 1.0 tolerance); soft-only quantities show measured, reported (not zero) drift.
- Complementarity: push-forward + hard projection beat either alone at long horizon ([[noether-1.0-rbc]] §5.2 pattern).

## Pitfalls
- **Don't hard-project a mis-discovered invariant** — gate on held-out drift, default to soft ([[discovered-conservation-1.1]] risk).
- Discovery needs P2 rollouts to be genuinely stable + trajectory-diverse (off-attractor corpus, [[impl-data-generation]]) — else spurious invariants.
- Whitening is a **correctness** step, not optional — the conserved directions are the smallest eigenvectors, exactly what poor conditioning destroys.

## See Also
- [[discovered-conservation-1.1]] (authoritative) · [[impl-diffusion]] (runs after this; re-projects generated residual) · [[impl-backbone]] (the `g_τ` gates freed alongside this)
