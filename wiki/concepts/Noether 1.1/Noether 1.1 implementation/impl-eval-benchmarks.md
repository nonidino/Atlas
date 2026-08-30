# Portion 10 — Evaluation & benchmarks

**Type:** Implementation spec — agent task (folder: Noether 1.1 / Noether 1.1 implementation)
**Portion of:** [[00-implementation-plan]] (Phase III, spans all). **Depends on:** [[impl-decoder]] (+ later portions as they land). **Unblocks:** milestone sign-off M2–M7.
**Design pages:** [[noether-1.0]] §7 (eval protocol), [[post-decoder-diffusion-1.1]] (attractor stats), [[autoregressive-rollout-stability]].

---

## Objective
The metric + benchmark suite that gates every milestone: single-step and rollout accuracy, spectral/statistical fidelity, conservation drift, and the generality ablations. Reuse 1.0's `eval/` (`rollout.py`, `ns_rollout.py`, `rbc_rollout.py`) and `scripts/eval_*`.

## Deliverables (`noether11/eval/`)
- `metrics.py` — rel-$L^2$ (single-step + horizon-resolved), energy **spectrum** error, structure functions, **rollout-mean Nusselt** (RBC), conserved-quantity **drift** (per discovered `C_k`), divergence norm.
- `rollout.py` — autoregressive rollout with toggles (projection on/off, diffusion on/off, learned edges on/off) for ablations.
- `benchmarks.py` — the per-rung acceptance tables from [[noether-1.1-medium]] §4 / [[noether-1.1-large]] §4, runnable as `scripts/eval.py --config … --system … --checkpoint …`.
- **Ablation harness** — the contrasts that are the actual evidence: projection on/off (conservation), diffusion on/off (blur vs. sharp), learned-edges on/off, F1-vs-F3 (rank collapse), hard-equivariance on/off.

## Interface contract
- `evaluate(model, dataset, toggles) -> dict[metric -> value]`.
- **Two metric regimes:** deterministic accuracy (rel-$L^2$, valid pre-Lyapunov) **and** attractor statistics (spectra, Nu — the correct target past the Lyapunov horizon, where a diffusion sample is *not* the mean). The suite must report both and not conflate them ([[post-decoder-diffusion-1.1]]).

## Build steps
1. Port 1.0 metrics; add spectral/structure-function + per-`C_k` drift.
2. Rollout with the toggle matrix.
3. Acceptance-table runner mapping to milestones M2–M7.
4. Ablation harness + a results writer (feeds the GUI's error panel and `results/`).

## Acceptance tests (these *are* the milestone gates)
- **M2:** single-step rel-$L^2 < 0.05$ (2D NS).
- **M3:** raw rollout flat to 100 steps (push-forward).
- **M4:** divergence drift low on incompressible, `λ→0` on compressible.
- **M5:** rollout-mean Nu within 5% + spectrum matched **with diffusion on**, and visibly blurred with it off (the contrast is the evidence).
- **M6/M7:** the full [[noether-1.1-medium]] / [[noether-1.1-large]] acceptance tables, including cross-regime transfer (rung $n{+}1$ doesn't degrade rung $n$).

## Pitfalls
- Don't grade a generative sample with pointwise MSE — use attractor statistics ([[post-decoder-diffusion-1.1]] open item).
- Report conservation drift **honestly** for soft-only quantities (bounded, nonzero) — matches the design's stated position ([[discovered-conservation-1.1]]).
- Keep every claim scale-honest: "faster than classical solvers" is not demonstrable at medium scale ([[noether-1.0]] §8).

## See Also
- [[noether-1.0]] §7 (the eval protocol template) · [[impl-diffusion]] (attractor-stat metrics) · [[00-implementation-plan]] §4 (milestones these gate)
