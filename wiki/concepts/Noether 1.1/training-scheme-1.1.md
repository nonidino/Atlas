# Training Scheme (Noether 1.1)

**Type:** Concept — model training master spec (folder: Noether 1.1)
**Status:** Design. The training-time companion to the 1.1 forward pass ([[00-noether-1.1-overview]]). Where [[training-curriculum]] (1.0) fixes the *loss composition* (four dependency-ordered loss groups, within-group uncertainty weighting, phase-separated diffusion), this page fixes the two things 1.1's larger, more heterogeneous parameter set makes non-obvious: **(A) the module training order** — 1.1 has many small MLPs, several typed attention operators, a *learned* edge generator, per-field descriptors and decoder heads, a conservation-discovery search, and a diffusion model, and they cannot all train from step 0; and **(B) the physics-regime curriculum** — which systems the model sees, in what order, and why. It inherits [[training-curriculum]]'s loss groups wholesale and slots the new 1.1 modules into them.
**Related Concepts:** [[00-noether-1.1-overview]], [[training-curriculum]], [[graph-tokenizer-1.1]], [[field-descriptors-1.1]], [[field-token-streams-1.1]], [[edge-generation-1.1]], [[backbone-1.1]], [[decoder-1.1]], [[discovered-conservation-1.1]], [[post-decoder-diffusion-1.1]], [[transfer-learning-fine-tuning]], [[autoregressive-rollout-stability]], [[smoke-test-bringup]]
**Related Summaries:** [[poseidon-pde-foundation-model]], [[gns-graph-network-simulators]], [[walrus-paper]], [[latent-diffusion-physics]]

---

## Why 1.1 needs its own training-order page

1.0's training story was "pretrain the tokenizer, then train the backbone under a staged loss." 1.1 breaks that into more moving parts, several of which are *unstable if trained at the wrong time*:

- The **learned edge generator** ([[edge-generation-1.1]]) is a latent-variable model trained end-to-end under the dynamics loss — but if the loss still rewards near-identity, it "learns the topology of doing nothing." It must not come online until a **motion-demanding** (multi-step) loss exists.
- The **conservation discovery search** ([[discovered-conservation-1.1]]) probes *trajectories* for invariants — meaningless until rollouts are accurate enough to carry real structure.
- The **diffusion head** ([[post-decoder-diffusion-1.1]]) denoises a *residual defined against the deterministic decode* — that residual is garbage until the deterministic stack has converged.
- The **field descriptors** $\theta_f$ and **query-bank modulation** ([[field-descriptors-1.1]]) parameterize the tokenizer's feature extractor — they must be in place *before* dynamics, or every later stage sees a shifting representation.

So the ordering principle is the same one [[training-curriculum]] closes on — *don't ask a component to do a hard job on top of an unstable foundation* — but with more components to order.

---

## The learnable inventory

Every trainable object in 1.1, and the phase it first trains in (phases defined below):

| Module                                                                                     | What it is                                                     | First trained                           | Frozen/tuned later |
| ------------------------------------------------------------------------------------------ | -------------------------------------------------------------- | --------------------------------------- | ------------------ |
| Dual-path query banks $Z_q^{\text{base}}, Z_q^{\text{hi}}$                                 | learned cross-attention query sets ([[graph-tokenizer-1.1]])   | P0                                      | lightly tuned P1+  |
| Field descriptors $\theta_f$ (free block $u_f$)                                            | per-field type + affordance vector ([[field-descriptors-1.1]]) | P0                                      | tuned throughout   |
| Query-bank modulation $U V^\top$ / FiLM MLP                                                | generates $Z_q^f$ from $[\theta_f\Vert z_{\text{sim}}]$        | P0                                      | tuned throughout   |
| Field encoders $\mathrm{Enc}_f$                                                            | small field-typed projections ([[field-token-streams-1.1]])    | P0                                      | tuned throughout   |
| Coordinate-implicit SIREN decoder heads $\mathrm{Dec}^f$                                   | per-field increment heads ([[decoder-1.1]])                    | P0 (as autoencoder) → P1 (as increment) | tuned throughout   |
| Conditioning MLPs ($z_{\text{param}}, z_{\Delta t}, \gamma_{\text{vec}}$)                  | constant embeddings ([[conditioning-and-constants-1.1]])       | P0                                      | tuned throughout   |
| Backbone: $W_{\text{skew}}$ (Cayley $S$), per-type $W^{(\tau,k)}, W_V, W_O^{(\tau)}$       | typed attention operators ([[backbone-1.1]])                   | P1                                      | core, tuned to P4  |
| Backbone gates $g_\tau$, dissipation $\gamma_\ell$, ReZero $\alpha_\ell,\alpha'_\ell$      | scalar dials, init $g{=}1$ / $\alpha{=}0$                      | P1 (frozen $g{=}1$) → P3 (freed)        | —                  |
| Semi-orthogonal FFN sub-step $W_1,W_2$ (F1)                                                | per-node nonlinearity ([[backbone-1.1]])                       | P1                                      | —                  |
| Learned edge scorer $g_\theta$                                                             | long-range edge proposer ([[edge-generation-1.1]])             | **P2**                                  | tuned P2+          |
| Conservation readouts $w_k$ / probe nets, weights $\lambda_k$, $\sigma_k(z_{\text{cond}})$ | discovered invariants ([[discovered-conservation-1.1]])        | **P3**                                  | tuned P3+          |
| Diffusion denoiser $\epsilon_\theta$                                                       | residual gap-fill ([[post-decoder-diffusion-1.1]])             | **P4**                                  | terminal           |

**Not trained — architecture, on from step 0:** the hard conservation projection ([[discovered-conservation-1.1]]), fixed finite-difference derivative operators, KNN edge families 1–2, nondimensionalization, and the structured (supplied) block of each $\theta_f$ (rank/parity tags are physics, not learned).

---

## Module training order (phases)

The phases refine [[training-curriculum]]'s Phase 0–4 into six, splitting Phase 1 to isolate the two 1.1 additions that are unsafe early (learned edges, discovered conservation).

| Phase                                                | Trains (new this phase)                                                                                                                           | Loss groups                                                 | Edges                                           | Gate to advance                                                                               |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **P0 — Tokenizer & descriptor pretraining**          | query banks, $\theta_f$, modulation map, $\mathrm{Enc}_f$, SIREN decoder *as an autoencoder*, conditioning MLPs                                   | G0 reconstruction fidelity (S3.4)                           | none (per-patch)                                | encode→decode error below the data's discretization floor, on held-out turbulent/shock data   |
| **P1 — Single-step dynamics, geometric edges only**  | backbone (skew, per-type attn, F1 FFN, $\gamma_\ell$), decoder as *increment* head; $g_\tau$ **frozen at $1$** on same-field, free on cross-field | A (residual + spectral, uncertainty-weighted)               | KNN families 1–2 only; **learned family 3 OFF** | single-step residual + spectral loss converged                                                |
| **P2 — Learned edges + temporal consistency**        | learned edge scorer $g_\theta$ (GumbelSigmoid, $\tau_{\text{gs}}$ annealed, straight-through, sparsity budget)                                    | A + B (push-forward $1{\to}2{\to}4{\to}8$, noise injection) | families 1–2 $\cup$ learned 3                   | multi-step rollout stable; learned graph sparse, non-collapsed                                |
| **P3 — Conservation discovery**                      | readouts $w_k$ + probe nets, $\lambda_k$, $\sigma_k(z_{\text{cond}})$; **free the $g_\tau$ gates**; SFA search with whitening+shrinkage           | A + B + C (SFA slowness + non-degeneracy)                   | all                                             | discovered invariants validated on held-out trajectories; hard-gated set drift $<\varepsilon$ |
| **P4 — Post-decoder diffusion**                      | $\epsilon_\theta$ on frozen residuals $r$; deterministic stack **lightly** fine-tuned, not hard-frozen                                            | A + B + C + D (denoising)                                   | all                                             | attractor statistics (spectra, rollout-mean Nu) matched                                       |
| **P5 — Coupling & new-regime extension** *(ongoing)* | new streams / edge types / $\theta_f$ for each added regime; continue training (transfer / LoRA-style), not from scratch                          | as needed                                                   | all                                             | per new-regime acceptance test                                                                |

Two placements deserve emphasis because they are the whole reason this page exists:

**Learned edges wait for P2 — after a motion-demanding loss.** Turning $g_\theta$ on in P1 (single-step, near-identity-friendly loss) would train it to *help the model stay still* — the failure [[edge-generation-1.1]] flags. Only once Group B's multi-step push-forward loss makes motion mandatory does a proposed long-range edge earn its place by *reducing rollout error*, against the sparsity budget's downward pressure. The same multi-step loss is what keeps the temporal-derivative history channels ([[graph-tokenizer-1.1]]) from feeding a copy-the-input solution.

**Gates freeze at $g{=}1$ until P3.** The antisymmetry gates ([[backbone-1.1]]) are held at their conservative init through P1–P2 so the backbone first learns transport *with* momentum structure; they are freed only in P3, alongside conservation discovery, so "which edges are non-conservative" is decided jointly with "which quantities are conserved" — the two are the same physical question. Freeing them earlier lets the model buy short-term fit by leaking momentum before it has learned the conservative dynamics that should be the default.

**Diffusion is phase-separated (P4), as in 1.0.** $\epsilon_\theta$ trains on the *frozen* residuals of a converged deterministic stack, across **all** trajectories (the S5.2 regime router is an inference-time cost decision, not a training-data filter — no chaos labels needed). The deterministic backbone is *lightly* fine-tuned rather than hard-frozen, so the diffusion head's gradients can still correct an upstream representation deficiency ([[training-curriculum]] Group D).

---

## Why this exact order — the dependency graph

Each phase's output is the next phase's precondition; the order is forced, not chosen:

- **P0 before everything** — the descriptors and modulation map *are* the tokenizer's feature extractor; dynamics trained on a shifting representation never converge. The reconstruction fidelity floor (G0) is a hard go/no-go gate, exactly as in 1.0.
- **P1 before P2** — you cannot usefully penalize multi-step drift before single-step prediction is decent, and you cannot let a latent-variable edge model optimize against a loss whose optimum is "do nothing."
- **P2 before P3** — the discovery search reads *trajectories*; it needs rollouts accurate enough (Group B stable) that a measured drift $D_k$ reflects physics, not model error.
- **P3 before P4** — the diffusion residual is defined against the deterministic decode *and re-projected onto discovered constraints* ([[post-decoder-diffusion-1.1]] "runs after conservation"); both must exist and be trustworthy first.

This is the module-level image of [[training-curriculum]]'s closing observation: learn the easy well-posed objective first (reconstruction, then single-step fit), defer the hard optimization problems (structure discovery, generation under uncertainty) until the representation beneath them is trustworthy.

---

## The physics-regime curriculum

Data order matters as much as module order. The corpus climbs a ladder that adds **exactly one generality axis at a time** on top of a model that already passed the previous rung — so a failure is attributable to the axis just added, not to a soup of new physics. This complements the [[smoke-test-bringup]] corpus ladder and realizes the "train once, deploy anywhere" theme ([[transfer-learning-fine-tuning]]).

| Rung | System                                                | New generality axis exercised              | What in the model it first stresses                                                                                                                                |
| ---- | ----------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | **1D Burgers**                                        | baseline: one field, one regime            | tokenizer, backbone Euler stability, SIREN decode — the [[noether-1.0]] smoke test                                                                                 |
| 2    | **2D incompressible NS** (decaying/forced turbulence) | +2D, +turbulence, +the *divergence* target | first learned edges (P2), first spectral loss, first **local ($\nabla\!\cdot u$) conservation discovery**                                                          |
| 3    | **2D Rayleigh–Bénard** (velocity + temperature)       | +multi-field, +buoyancy, +a gravity vector | field-token streams, cross-field/buoyancy edges, the covariant $\mathbf g$ channel, RBC's **global** invariants (attractor energy) — the 1.0-RBC target done right |
| 4    | **Compressible flow** (variable Mach)                 | +compressibility ($\nabla\!\cdot u\ne0$)   | the **compressible-vs-incompressible demo**: $\sigma_{\nabla\cdot u}(z_{\text{cond}})$ must ramp to zero along Mach with *no code change*                          |
| 5    | **Settling particles in a fluid**                     | +coupled continuum–particle                | the typed **cross-type edges** and IBM-adjoint message form ([[open-architectural-problems]] P4), the particle decoder head                                        |
| 6    | **MHD / multi-physics** (add $\mathbf B$ stream)      | +a pseudovector field, +Lorentz coupling   | "add a stream, not a code path"; pseudovector parity tags in $\theta_f$; magnetic helicity as a discovered invariant                                               |

**Conservation-discovery order tracks the ladder:** local divergence discovery on the NS velocity field at rung 2 (a learned-weight soft penalty on $\nabla\!\cdot u$ whose $\lambda\!\to\!0$ on compressible data by itself, front-loading the rung-4 generality demo), then RBC's global invariants at rung 3 — exactly the first-cut scope [[discovered-conservation-1.1]] commits to.

**Data *diversity*, not just system order, is part of the curriculum.** Discovery is underdetermined without trajectory diversity: the off-attractor, energy-diverse corpus already built for RBC ([[noether-1.0-rbc]] §5.1) is the kind of spread rung 3's discovery needs, and every rung wants a spread of characteristic numbers (a range of Re, Ra, Ma) so the $\sigma_k(z_{\text{cond}})$ maps are fit over a continuum, not a few points — the condition for the Tier-A "new regime = new point in conditioning space" generalization to hold ([[discovered-conservation-1.1]]).

**Each new rung *extends*, it does not restart (P5).** A new regime is added by introducing its streams / edge types / descriptors and *continuing* training from the shared base, leveraging transfer — the foundation-model premise ([[transfer-learning-fine-tuning]], [[poseidon-pde-foundation-model]]). Because the per-regime information lives in conditioning ($z_{\text{cond}}$, $\sigma_k(z_{\text{cond}})$) and in shared-base $+$ correction descriptors, not in per-regime weights ([[00-noether-1.1-overview]] "Shared vs. specialized parameters"), adding rung $n{+}1$ should *improve* rung $n$ via shared structure, not overwrite it — the property to verify at each step (no catastrophic forgetting of earlier rungs).

---

## Freezing / fine-tuning discipline (summary)

| Stage | Frozen | Fine-tuned | Rationale |
|---|---|---|---|
| P0→P1 | — | tokenizer lightly tuned | dynamics can still correct a reconstruction-only latent |
| P1→P2 | — | all active | edges join a moving but stable backbone |
| P2→P3 | — | all active; $g_\tau$ freed | gates and invariants decided jointly |
| P3→P4 | deterministic stack (mostly) | backbone *lightly* tuned | diffusion trains on near-frozen residuals but can still fix upstream deficiencies ([[training-curriculum]] Group D) |
| P5 | earlier-rung-critical weights protected | new streams/heads + shared base | extend without forgetting |

The tier-1 hard conservation projection runs in the **forward pass throughout all phases** — it is architecture, not curriculum ([[training-curriculum]], [[discovered-conservation-1.1]]).

---

## Open items / risks

- **Phase-boundary shocks.** Turning on a new loss group / module (learned edges at P2, discovery at P3, diffusion at P4) can shock a converged region. Mitigations: warm the new module's gain from zero (ReZero-style), anneal in its loss weight, and keep GradNorm in reserve ([[training-curriculum]]) if one group's gradients dominate on activation.
- **Learned-edge gate collapse under the sparsity budget** — all-on or all-off — standard NRI hygiene (entropy regularizer, budget tuning); watched from P2 ([[edge-generation-1.1]]).
- **Discovery quality depends on P2 rollout quality.** If Group B rollouts are only marginally stable, measured $D_k$ is dominated by model error, not physics, and the search finds spurious "invariants." The held-out-drift gate and default-to-soft are the guard, but the real fix is not advancing to P3 prematurely.
- **Curriculum forgetting.** Rung $n{+}1$ must not degrade rung $n$; needs a retained-earlier-rung eval at every step and possibly rehearsal/replay of earlier corpora (the transfer-learning diversity principle, [[transfer-learning-fine-tuning]]).
- **Descriptor grounding needs multi-system exposure early.** The free block $u_f$ of $\theta_f$ risks absorbing dataset ID rather than *field* identity if field diversity is low ([[field-descriptors-1.1]]); the ladder should introduce a second system sharing a field (e.g. velocity in both NS and RBC) before over-training on the first.

---

## See Also
- [[training-curriculum]] — the 1.0 loss composition this inherits (four groups, uncertainty weighting, phase separation)
- [[00-noether-1.1-overview]] — the forward pass and the shared-vs-specialized parameter stance
- [[edge-generation-1.1]] — why learned edges must wait for the multi-step loss (P2)
- [[discovered-conservation-1.1]] — the discovery search (P3), whitening, and hard-projection gate
- [[post-decoder-diffusion-1.1]] — phase-separated generative training (P4)
- [[smoke-test-bringup]] — the corpus ladder and implementation-readiness checklist this regime curriculum aligns with
- [[transfer-learning-fine-tuning]] — the extend-don't-restart principle behind P5 and the regime ladder
- [[poseidon-pde-foundation-model]] — operator-learning pretraining + all2all amplification, complementary to this curriculum
