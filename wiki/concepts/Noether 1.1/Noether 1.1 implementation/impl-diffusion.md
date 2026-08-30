# Portion 9 — Post-decoder diffusion

**Type:** Implementation spec — agent task (folder: Noether 1.1 / Noether 1.1 implementation)
**Portion of:** [[00-implementation-plan]] (Phase III). **Depends on:** [[impl-decoder]], [[impl-training-harness]] (and ordering after [[impl-discovered-conservation]]). **Unblocks:** [[impl-eval-benchmarks]] (attractor stats).
**Design pages:** [[post-decoder-diffusion-1.1]] (authoritative), [[latent-diffusion-physics]]. Trains in **P4**, phase-separated ([[training-scheme-1.1]]).

---

## Objective
A conditional diffusion model that **generates the high-wavenumber chaotic residual** the deterministic decode averages away — curing the conditional-mean blur that made 1.0-RBC a smoother. Deterministic-encode / stochastic-predict.

## Deliverables (`noether11/generate/diffusion.py`)
- `ResidualDiffusion` `ε_θ(r_τ, τ, v̄_{t+1}, {h_i}, z_cond)`: a small denoiser conditioned on the coarse deterministic decode `v̄`, the tokens, and the constants. **ε-prediction** target; **fixed** `ᾱ_τ` schedule.
- Training: `L_diff = E‖ε − ε_θ(r_τ,…)‖²`, `r = v_{t+1} − v̄_{t+1}`, `r_τ = √ᾱ_τ r + √(1−ᾱ_τ) ε`. Trains on **frozen** deterministic-stack residuals across **all** trajectories (no chaos labels).
- Sampling (inference): the ancestral reverse loop from noise → `r_0`, then `v_{t+1} = v̄_{t+1} + r_0`; **few-step sampler** (DDIM/consistency) option for cost.
- **Ordering:** runs **after** [[impl-discovered-conservation]]; the generated `r_0` is **re-projected** onto discovered constraints (minimal-disturbance) before feeding the next step.
- Inference toggles: low-temperature (scale `σ_τ`), or skip (`r=0`) for a stable reference rollout — the GUI toggle ([[impl-gui-2d]]).

## Interface contract
- `Diffusion(cfg).sample(v̄, tokens, z_cond, n_steps) -> r_0`; `train_step(batch_of_frozen_residuals)`.
- Band-limited: generate only the **residual** (near-cutoff band `[k_det, k_max]`), not the whole field — keeps `ε_θ` and step count small ([[post-decoder-diffusion-1.1]]).
- Deterministic stack is **frozen** during P4 (lightly fine-tuned only), per phase separation.

## Build steps
1. Denoiser net (reuse the backbone block or a small DiT-style net) with the 3 conditioning inputs.
2. Forward noising + ε-prediction loss; fixed schedule.
3. Reverse sampler (full + few-step).
4. Re-projection call to [[impl-discovered-conservation]] after sampling.
5. Wire into harness Group D (P4); GUI toggles.

## Acceptance tests
- **RBC statistics (milestone M5):** with diffusion on, rollout-mean Nu within 5% and the energy spectrum's plume band restored vs. the blurred deterministic-only baseline ([[noether-1.0-rbc]] §6.6 was the diagnosis).
- Sampled state satisfies hard-gated constraints after re-projection (a conservation check post-sample).
- Few-step sampler within tolerance of full sampler at a fraction of the cost.
- Skip/low-temp toggles produce a stable reference rollout.

## Pitfalls
- **Evaluate with attractor statistics, not pointwise rel-$L^2$** — a correct sample is *not* the mean, so rel-$L^2$ penalizes it ([[post-decoder-diffusion-1.1]] open item; [[impl-eval-benchmarks]]).
- Respect the **after-conservation** ordering — generate, then re-project; not the reverse.
- Don't train jointly with the deterministic stack — the residual target is nonstationary until it converges (phase separation, [[latent-diffusion-physics]]).

## See Also
- [[post-decoder-diffusion-1.1]] (authoritative) · [[impl-discovered-conservation]] (re-projection partner, ordering) · [[impl-eval-benchmarks]] (attractor-statistic metrics)
