# Portion 6 — Decoder

**Type:** Implementation spec — agent task (folder: Noether 1.1 / Noether 1.1 implementation)
**Portion of:** [[00-implementation-plan]] (Phase II). **Depends on:** [[impl-tokenizer-descriptors]], [[impl-backbone]]. **Unblocks:** [[impl-training-harness]], [[impl-diffusion]].
**Design pages:** [[decoder-1.1]] (authoritative), [[coordinate-implicit-tokens]]. Co-trained as autoencoder in **P0**, as increment head in **P1**.

---

## Objective
Map evolved tokens back to a physical next-state, per field, as a **coordinate-conditioned increment** decoded by SIREN heads over the dual latent paths — the general increment head is the **default**; div-free is **not** hard-wired.

## Deliverables (`noether11/core/decoder.py`)
- `SIRENHead` (reuse 1.0 `siren.py`): `sin(ω₀(Wx+b))`, `ω₀=30`, matched init (first layer `U(-1/fan_in, 1/fan_in)`, rest `U(-√(6/fan_in)/ω₀, …)`).
- `Decoder`: per field `f`, per node `i`, `δv^f(y) = Dec^f_lo(t_i^{f,(L)}, γ(y−x_i)) + Dec^f_hi(h_i^{f,hi}, γ(y−x_i))`; `v^f_{t+1}(y) = v^f_t(y) + δv^f(y)` for any query `y` in patch `i` (resolution-free).
- Per-field heads **generated from `θ_f`** (same hypernetwork idea as the query bank) so adding a field adds a head with no code change.
- **Optional structure-preserving heads (discovered-gated, off by default):** a stream-function head `δv=∇^⊥δψ` (free, exact div-free) and a Helmholtz-projected head (Poisson solve, costly) — engaged **only** when [[impl-discovered-conservation]] reports divergence confidently ≈ 0.
- Boundary conditions via fixed ghost-stencil overwrites (geometric, in the general path).

## Interface contract
- `Decoder(cfg)(evolved_tokens, query_coords) -> next_state_fields` (increment applied); works for scalar, vector, and particle-kinematic fields.
- `decode_mode: "general" | "streamfn" | "helmholtz"` — set by the conservation module, **not** hardcoded.
- Re-dimensionalize the increment via the conditioning ruler ([[conditioning-and-constants-1.1]]).

## Build steps
1. Port `siren.py`; build the dual-path additive decoder.
2. Coordinate Fourier encoding `γ` (share with tokenizer).
3. Per-field head generation from `θ_f`.
4. **P0 mode:** expose `encode→decode` for the tokenizer autoencoder (G0).
5. Structure-preserving heads behind the discovered-gate flag (stream-function first — it's free; Helmholtz later).

## Acceptance tests
- SIREN can emit high frequencies (reconstruct a sharp gradient a plain-MLP head cannot) — the spectral-bias guard ([[open-architectural-problems]] Problem 3).
- Increment is *allowed to be large* — a test that the head is **not** structurally biased toward tiny increments (the 1.0-RBC failure).
- Resolution-free: train at $128^2$, decode at $256^2$ without retraining.
- Compressible data: general head runs unchanged; div-free head **not** engaged (divergence ≠ 0).

## Pitfalls
- Div-free is **discovered, not wired** — never default the stream-function head on. It's an optimization the model *earns* ([[decoder-1.1]]).
- Predicting a *small* residual is the failure mode; pair with the motion-demanding loss ([[impl-training-harness]]).

## See Also
- [[decoder-1.1]] (authoritative) · [[impl-diffusion]] (generates the chaotic band this decode drops) · [[impl-discovered-conservation]] (sets `decode_mode`)
