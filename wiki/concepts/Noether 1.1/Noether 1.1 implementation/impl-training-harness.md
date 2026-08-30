# Portion 7 — Training harness (phased curriculum + losses)

**Type:** Implementation spec — agent task (folder: Noether 1.1 / Noether 1.1 implementation)
**Portion of:** [[00-implementation-plan]] (Phase II). **Depends on:** [[impl-tokenizer-descriptors]], [[impl-edge-generation]], [[impl-backbone]], [[impl-decoder]]. **Unblocks:** [[impl-discovered-conservation]], [[impl-diffusion]].
**Design pages:** [[training-scheme-1.1]] (authoritative), [[training-curriculum]] (loss groups), [[autoregressive-rollout-stability]].

---

## Objective
The phased training loop that drives P0→P4, owning the **module-activation gates** ([[training-scheme-1.1]]): it decides when learned edges turn on (P2), when the antisymmetry gates unfreeze (P3), and when discovery/diffusion begin. This is the orchestrator that makes the "don't build on an unstable foundation" order real.

## Deliverables (`noether11/train/`)
- `losses.py` — Group A (per-channel homoscedastic-uncertainty-weighted residual + spectral), Group B (push-forward multi-step + context-noise injection), Group C hook (SFA discovery — implemented in [[impl-discovered-conservation]]), Group D hook (diffusion denoising — [[impl-diffusion]]). Per [[training-curriculum]].
- `curriculum.py` — the **phase controller**: a state machine P0→P4 with the gate-to-advance predicates from [[training-scheme-1.1]] §"module training order"; flips `use_learned_edges` (P2 start), `freeze_gates` off (P3), starts discovery (P3), starts diffusion (P4 — deterministic stack lightly fine-tuned, not hard-frozen).
- `loop.py` — AdamW (exclude `W_skew`, Stiefel factors from weight decay; re-orthogonalize F1 every 100 steps), ε warmup, push-forward horizon schedule $1{\to}2{\to}4{\to}8$, bf16 (+ fp32 for projection), checkpointing, logging.
- `scripts/train.py --config medium --system ns --phase-until P2` etc.

## Interface contract
- `Trainer(cfg, model, data).run(until_phase)` — advances through phases, respecting gates.
- Phase transitions are **predicate-gated**, not step-count-gated: advance only when the current phase's acceptance metric passes (e.g. single-step converged before P2).
- Emits per-phase checkpoints the GUI ([[impl-gui-2d]]) and eval ([[impl-eval-benchmarks]]) can load.

## Build steps
1. Group A losses + uncertainty weighting (Kendall-style `log σ²` per channel/domain).
2. Push-forward unroll with noise injection (reuse 1.0's validated push-forward; [[noether-1.0-rbc]] §5.2).
3. The phase controller with the exact gate predicates; wire the module flags.
4. Optimizer/schedule/checkpointing; the CLI.
5. Integrate the Group C/D hooks (call into portions 8/9 when their phase opens).

## Acceptance tests
- **P0→P1→P2 progression on 2D NS** advances only when each gate passes (a test that forces a failing metric blocks advance).
- Learned edges are provably off until P2 (assert `scorer.requires_grad == False` in P0/P1).
- Gates provably frozen at `g=1` until P3.
- Push-forward yields a flat raw rollout to 100 steps (milestone **M3**), reproducing the 1.0 result on the 1.1 stack.

## Pitfalls
- **Phase-boundary shocks** — warm each newly-activated module's gain from 0 (ReZero) and anneal in its loss weight; keep GradNorm in reserve ([[training-scheme-1.1]] open item).
- Don't advance to P3 on marginal P2 rollouts — discovery on bad trajectories finds spurious invariants ([[impl-discovered-conservation]] risk).

## See Also
- [[training-scheme-1.1]] (the phase order this enforces) · [[training-curriculum]] (loss groups) · [[impl-discovered-conservation]], [[impl-diffusion]] (the P3/P4 hooks)
