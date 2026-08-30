# Noether 1.1 — Fineness & Token Density

**Type:** Implementation spec — failure-mode diagnosis + fix (folder: Noether 1.1 / Noether 1.1 implementation)
**Status:** As-diagnosed on the P2 checkpoint (`ckpt_latest.infer.pt`, step 11 200, medium). Root cause **confirmed empirically** on NS val; fix implemented in the `noether-1.1` branch (config default, training augmentation, inference override). Companion to [[noether-1.1-medium]] / [[noether-1.1-large]] (which now carry the `patch_domain_size` / `patch_size_jitter` numbers) and [[impl-tokenizer-descriptors]].
**Related Concepts:** [[graph-tokenizer-1.1]], [[decoder-1.1]], [[intelligent-patching]], [[coordinate-implicit-tokens]], [[post-decoder-diffusion-1.1]], [[training-scheme-1.1]], [[noether-1.0-rbc]], [[00-noether-1.1-overview]]
**Related Summaries:** [[gns-graph-network-simulators]], [[autoregressive-rollout-stability]]

---

## Symptom

Rolling out the P2 checkpoint produces a **visibly "pixelated" prediction** — the model panel is a coarse grid of near-constant blocks, while ground truth is smooth and finely structured. The blocks are large and axis-aligned; the error panel is dominated by the block boundaries.

## Root cause — the blocks *are* the tokens

The tokenizer partitions the domain into patches by **domain coordinate**, one node (token) per patch ([[graph-tokenizer-1.1]], `core/patch.py`). The number of tokens per axis is
$$n_a = \operatorname{round}\!\Big(\frac{\text{extent}_a}{\texttt{patch\_domain\_size}}\Big),\qquad P=\prod_a n_a.$$
The NS/RBC corpus is a $128\times64$ grid on the domain $[0,2]\times[0,1]$ (periodic in $x$, walls in $y$). At the shipped default $\texttt{patch\_domain\_size}=0.25$:
$$n_x=\operatorname{round}(2/0.25)=8,\quad n_y=\operatorname{round}(1/0.25)=4,\quad P=32.$$
So the whole field is represented by **32 tokens**, each standing for a $16\times16$ block of grid points. That $8\times4$ grid is exactly the block pattern seen on screen.

Why the blocks are near-*constant*, not merely coarse: the decoder ([[decoder-1.1]], `core/decoder.py::_decode_field`) has **one token per patch**, broadcast to every point in the patch; the only thing that varies *within* a patch is the Fourier-encoded offset $\gamma(y-x_i)$ fed to the SIREN head. All sub-patch structure must therefore be painted by a single SIREN from a single token. Spectral bias + a token that has to summarize a $16\times16$ region drives the head toward its patch-mean → flat blocks. This is the tokenizer/decoder-level twin of the [[noether-1.0-rbc]] §6.6 "low-pass smoother" failure: the coarse energy envelope survives, the transport-carrying fine structure is averaged away.

## Evidence (single-step NS val, same weights, density overridden)

`scripts/diagnose_token_density.py` rolls the frozen checkpoint one step at four densities. Persistence (copy the input frame) scores rel-$L^2=0.014$ — consecutive frames are ~99 % identical, so the P1 target ($<0.05$) is easy in principle.

| `patch_domain_size` | #tokens | pts/patch | 1-step rel-$L^2$ | sub-block detail | runs? |
|---|---|---|---|---|---|
| 0.25 (trained) | 32 | 256 | 1.232 | 0.275 | ✓ |
| 0.125 | 128 | 64 | 1.289 | 1.273 | ✓ |
| 0.0625 | 512 | 16 | 1.277 | 1.407 | ✓ |
| 0.03125 | 2048 | 4 | 1.282 | 1.438 | ✓ |

*"sub-block detail" = fraction of variance living within a coarse 0.25 patch; ground truth = **0.392**.*

Two readings, both load-bearing:

1. **Every density runs on the same weights** (the "runs? ✓" column). No parameter shape depends on the token count $P$ — it is a runtime graph dimension, not a weight dimension (audited across tokenizer, backbone, decoder, edge scorer; see next section). **This is the whole answer to "must I restart training": no.**
2. **More tokens alone (zero-shot) does *not* fix it.** rel-$L^2$ is flat at ~1.28 and the model is already far *worse than persistence* (1.23 vs 0.014) — the checkpoint is undertrained (P2, 11.2k steps, never cleared the P1 gate). Worse, the "detail" the finer densities add is **spurious** — sub-block detail jumps to 1.4 (vs truth 0.39), i.e. the $\omega_0{=}30$ SIREN, queried at offsets far finer than it was trained on, **rings** instead of resolving structure. Tokens are *capacity*; only training fills them with *correct* structure.

## Why no from-scratch restart is needed

`patch_domain_size` sets only $P$, the number of nodes on the graph. Weight shapes are functions of $d$, $N_q$, $d_\text{head}$, $H$, edge-type count, and per-field signature — never $P$:

- **Tokenizer** — `QueryCompression` $W_k,W_v:\mathbb R^{d}\!\to\!\mathbb R^{d_\text{head}}$; query banks sized $(N_q,d_\text{head})$; KV embed $\mathbb R^{\text{feat}}\!\to\!\mathbb R^{d}$. $P$ is a batched axis `[B,P,W,·]`.
- **Backbone** — $M,W_{qk},W_v,W_o,W_1,W_2$ all $d\times d$ (per head/type); attention is over graph edges, not a fixed $P$.
- **Decoder** — SIREN heads keyed by `(rank, ndim, rep_type)`, input width $d_\text{latent}+\gamma\text{-dim}$.
- **Edges** — the learned scorer is a function of node features; only its $O(P^2)$ candidate **cost** grows (trivial at $P\le2048$).

So the existing checkpoint **loads and rolls out at any density with no shape mismatch** — confirmed above. Changing density is a **warm-start / fine-tune**, not a re-initialization. Given the checkpoint is only at step 11.2k and hasn't passed P1, little is being thrown away either way — the backbone, conditioning, and edge scorer all transfer; the tokenizer query banks and decoder SIREN heads re-adapt to the new patch scale quickly.

## The fix (implemented)

1. **More tokens by default.** `patch_domain_size` default lowered $0.25\to0.0625$ (base + `MediumConfig`); `LargeConfig` uses $0.03125$. On the $[0,2]\times[0,1]$ corpus that is $512$ and $2048$ tokens (≈$4\times4$ and $2\times2$ grid points/patch) — fine enough that a single token per patch is an accurate local summary, so the per-patch decode stops reading as a block. *(These numbers now live in the [[noether-1.1-medium]] / [[noether-1.1-large]] spec tables — this page and those must agree.)*
2. **Multi-scale training (`patch_size_jitter`).** Per training step the shared tokenizer's density is set to $\texttt{patch\_domain\_size}\cdot m$ for $m$ drawn from a small multiplier set (default $\{0.5,1,2\}$); validation is pinned to the base density so the curriculum gate is always measured where the model deploys (`train/loop.py::_apply_patch_jitter`, `validate`). This is what makes the **resolution-free** claim ([[graph-tokenizer-1.1]], [[coordinate-implicit-tokens]]) *real* rather than aspirational: the tokenizer and SIREN heads learn to operate across offset scales, so cranking density *up* at inference yields correct detail instead of ringing.
3. **Inference-time density override.** `load_checkpoint(..., patch_domain_size=...)` and the GUI `/api/rollout` `patch_domain_size` field let any checkpoint be rendered/evaluated at a chosen density — for A/B comparison and for the diagnostic above — without touching weights.

**[AI Inference]:** the principled long-term answer to fineness is **not** uniformly-finer patches (which spend tokens evenly, including on smooth regions) but **adaptive placement** — [[intelligent-patching]]'s refinement monitor putting small patches only where detail is high. Uniform density + jitter is the cheap, immediate fix that removes the pixelation and de-risks the resolution-free promise; adaptive patching is the token-efficient successor once the uniform path is trained and stable. The two compose: jitter-trained heads are exactly what an adaptive patcher needs to trust a variable patch scale.

## What this does *not* fix

The ~1.28 rel-$L^2$ (worse than persistence) is an **accuracy/training** problem, not a tokenization one — it persists at every density. It reflects an undertrained P2 checkpoint that has not yet cleared the P1 single-step gate, compounded by the near-identity/low-pass pressure [[noether-1.0-rbc]] and [[training-scheme-1.1]] warn about. The density fix removes the *visual* coarseness and gives the model the capacity to be sharp; reaching the accuracy target still needs the curriculum to run (P0 tokenizer floor → P1 single-step → P2 rollout), now at the finer density, ideally with [[post-decoder-diffusion-1.1]] (P4) for the chaotic high-$k$ band.

## Recommended procedure

1. Pull the new config (finer `patch_domain_size` + `patch_size_jitter`).
2. **Warm-start** from the current checkpoint (do *not* re-init). Because the tokenizer/decoder must re-adapt to the new scale, re-walk the cheap early rungs: P0 (tokenizer reconstruction floor) → P1 (single-step) → P2 (rollout), then P3/P4. The backbone/conditioning/edge-scorer weights carry over.
3. Re-run `scripts/diagnose_token_density.py`: expect the block pattern gone at the base density **and** rel-$L^2$ finally below persistence — that pairing (not either alone) is the acceptance signal.

## See Also
- [[graph-tokenizer-1.1]] — where token count is set; the resolution-free encoder this relies on
- [[decoder-1.1]] — the one-token-per-patch coordinate-implicit decode that renders the blocks
- [[intelligent-patching]] — adaptive placement, the token-efficient successor to uniform density
- [[noether-1.1-medium]] / [[noether-1.1-large]] — the spec tables carrying the density/jitter numbers
- [[training-scheme-1.1]] — the curriculum the warm-start re-walks at the new density
- [[noether-1.0-rbc]] — the original low-pass-smoother failure this mirrors at the token level
