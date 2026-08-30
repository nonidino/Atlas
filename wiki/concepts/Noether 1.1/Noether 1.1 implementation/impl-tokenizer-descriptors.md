# Portion 3 — Tokenizer + field streams + descriptors + conditioned queries

**Type:** Implementation spec — agent task (folder: Noether 1.1 / Noether 1.1 implementation)
**Portion of:** [[00-implementation-plan]] (Phase II). **Depends on:** [[impl-repo-scaffold]], [[impl-data-generation]]. **Unblocks:** [[impl-edge-generation]], [[impl-backbone]], [[impl-decoder]].
**Design pages:** [[graph-tokenizer-1.1]], [[field-token-streams-1.1]], [[field-descriptors-1.1]], [[conditioning-and-constants-1.1]]. Trains in **P0** ([[training-scheme-1.1]]).

---

## Objective
Turn a uniform-schema sample into the typed token multiset the backbone consumes: resolution-free node latents with **redundant derived features (spatial + temporal)**, split into **one token stream per field**, each tagged with a learned descriptor $\theta_f$, using **simulation-conditioned query banks** $Z_q^f$. This is the whole encode side.

## Deliverables (`noether11/core/`)
- `tokenize.py` — patch/partition → nodes at centroids; per-point augmented feature $\phi_p=[\gamma(x_p{-}x_i), v, \nabla v, \omega, \|\nabla v\|, \partial_t v, \partial_{tt}v, …]$ (temporal derivatives = **how history enters**, [[graph-tokenizer-1.1]]); dual-path cross-attention encoders (lo `Z_q^base`, hi `Z_q^hi`) → $h_i^{\text{lo}}, h_i^{\text{hi}}$.
- `descriptors.py` — `FieldDescriptor`: structured block (rank/parity/rep-type/conserved-flag from `field_meta`) $\Vert$ learned free block $u_f$; `z_sim = pool_f θ_f ‖ z_cond`.
- `query_modulation.py` — $Z_q^f = Z_q^{\text{base}} + \Delta_q(\theta_f, z_{\text{sim}})$, low-rank/FiLM (option **B2**, [[field-descriptors-1.1]]) — a shared base plus a conditioned correction (NOT a per-field table).
- `streams.py` — emit `t_i^f = [Enc_f(h_i) ‖ W_e θ_f] + z_cond` per field per node (concatenate descriptor into a **reserved slot**, never add). Token multiset `{t_i^f}`.
- `condition.py` (or `noether11/condition/`) — the ruler (nondimensionalize), dial ($z_{\text{param}}$ from log-π groups), arrow ($\gamma_{\text{vec}}$ covariant channel) per [[conditioning-and-constants-1.1]].

## Interface contract
- `Tokenizer(cfg).encode(sample) -> Tokens` where `Tokens` carries `{t^f_i}`, positions `x_i`, `field_ids`, `h_hi` (for the decoder/diffusion), `z_cond`, and the per-field `θ_f`.
- Resolution-free: same latent dim/meaning at any grid resolution (permutation-invariant cross-attention).
- Particle inputs produce the *same* token type (per-neighbor relative stats bundle), one interface.

## Build steps
1. Implement Fourier coord encoding `γ`, the fixed FD derivative operators (reuse `operators2d.py`), and the temporal-difference channels.
2. Dual-path query-compression cross-attention (standard softmax cross-attn, `d_head=d/N_q`, concat outputs → `h_i`), per [[noether-1.0]] §3.3.
3. `FieldDescriptor` with structured tags read from `field_meta`; learned free block.
4. B2 query-bank modulation ($UV^\top$ low-rank or FiLM); `z_sim` pooling.
5. Stream emission with reserved-slot concatenation; conditioning injection.
6. **P0 training mode:** wire the tokenizer + SIREN decoder ([[impl-decoder]]) as an autoencoder for the G0 reconstruction gate.

## Acceptance tests
- **G0 gate (milestone M1):** encode→decode reconstruction error < the data's discretization error on held-out turbulent NS frames ([[training-scheme-1.1]] P0).
- Resolution-invariance: encode a field at 2× resolution → latents match within tolerance.
- Variable field cardinality: a 1-field (NS) and a 2-field (RBC) sample both tokenize with no code change; dropping a field drops a stream.
- Descriptor separability: swapping two fields' data swaps their streams' identity slot, not their content (a unit test on the reserved-slot concatenation).

## Pitfalls
- Reserved-slot **concatenation, not addition**, for $\theta_f$ and (separately) keep the hi path a *second independent* encoder, not a subtraction ([[graph-tokenizer-1.1]]).
- The query modulation must stay a **correction on a shared base** — a per-field replacement kills transfer ([[field-descriptors-1.1]] open item).
- Keep the tokenizer **constant-light**: constants touch it only via nondimensionalization + (optional) adaptive-patch monitors, never as embedded weights.

## See Also
- [[impl-edge-generation]] (consumes the tokens) · [[impl-decoder]] (co-trained autoencoder in P0) · [[graph-tokenizer-1.1]], [[field-descriptors-1.1]] (authoritative math)
