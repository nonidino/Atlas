# Noether 1.1 — Large config

**Type:** Implementation spec — model size point (folder: Noether 1.1 / Noether 1.1 implementation)
**Status:** Design estimate. The full-scale 1.1 checkpoint — the config that carries the whole regime curriculum (fluids → compressible → particles → MHD) at foundation-model capacity. **All numbers are principled design estimates.** Companion: [[noether-1.1-medium]] (which carries the full derivation this page scales). Multi-GPU / long-run Colab or a small cluster.
**Design source:** [[00-noether-1.1-overview]], [[backbone-1.1]], [[training-scheme-1.1]].

---

## Backbone depth — a deep transformer-style stack

Same skeleton as [[noether-1.1-medium]] (and as an LLM: a residual attention+FFN block repeated $L$ times, depth = capacity), but **large uses $L=24$ layers** — i.e. 24 forward-Euler integration steps per model call, each running $T{=}4$ typed graph-attention operators. Doubling depth over medium is the primary capacity increase; width doubles too. Depth is *latent time*, so a deeper large model can represent a longer/stiffer effective integration per step ([[backbone-1.1]]).

---

## 1. Scale point (vs. medium)

| Symbol | Medium | **Large** |
|---|---|---|
| $d$ (width) | 384 | **768** |
| $L$ (layers) | 12 | **24** |
| $H$ (heads/type) | 6 | **12** ($d_{\text{head}}=64$) |
| $T$ (edge types) | 4 | 4 |
| $N_q / N_q^{\text{hi}}$ | 16 / 8 | **32 / 16** |
| `patch_domain_size` | 0.0625 | **0.03125** (→ 64×32=2048 tokens; ≈2×2 pts/patch) |
| `patch_size_jitter` | (0.5,1,2) | (0.5,1,2) |
| $d_{\text{hi}}$ | 192 | **384** |
| $d_\theta$ | 32 | 48 |
| $d_{\text{diff}} / L_{\text{diff}}$ | 192 / 6 | **384 / 8** |
| $F$ (representative) | 4 | 4 (up to ~8 for MHD/multi-species) |

---

## 2. Parameter budget

Same formulas as [[noether-1.1-medium]] §2 ($(9+3T)Ld^2$ backbone, $21d^2$/layer at $T{=}4$):

| Component | Formula | Large ($d{=}768$) |
|---|---|---|
| Backbone | $21\cdot24\cdot d^2$ | **297M** |
| Field encoders | $\approx F d^2$ | 2.36M |
| Decoder SIREN heads | $\approx 4F d^2$ | 9.44M |
| Conditioning MLPs | $\approx 2d^2$ | 1.18M |
| Learned edge scorer | $\approx 3d^2$ | 1.77M |
| Query banks + modulation | $\approx 2d^2$ | 1.18M |
| Descriptors + conservation readouts | small | ~0.03M |
| **Core subtotal (deterministic)** | $[(9{+}3T)L + 5F + 7]d^2$ | **≈ 313M** |
| Diffusion denoiser | $12L_{\text{diff}}d_{\text{diff}}^2$ | ≈ 14.2M |
| **Total** | | **≈ 327M** |

At $F{=}8$ (MHD + multi-species) the $F$-terms add ~$27d^2\approx16$M, giving ~343M — the backbone still dominates, so the model stays roughly field-count-insensitive.

---

## 3. What changes from medium (deltas only)

- **Depth-stability matters more.** At $L=24$ the ReZero gains and the Cayley $\|W_{\text{skew}}\|_2\le2$ bound become load-bearing (they were safely slack at $L=12$) — the P10 stability recipe ([[open-architectural-problems]]) must be on. Keep the Euler $\epsilon$ warmup conservative ($0.1\to0.25$).
- **Full edge budget.** same-field $K=24$, cross-field $K'=8$, learned-edge budget $\le12$/node; the learned scorer's $O(N^2)$ candidate scoring is still trivial at node scale ([[edge-generation-1.1]] cost note).
- **Curriculum reaches rungs 4–6.** compressible (divergence-discovery generality demo), settling particles (typed cross-type edges / IBM message form), MHD ($\mathbf B$ stream, pseudovector parity, magnetic-helicity discovery). See [[training-scheme-1.1]] and [[impl-data-generation]].
- **Optimizer:** peak LR $1.5\times10^{-4}$ (lower for the deeper stack), batch 16–32, gradient checkpointing on the backbone to fit memory; otherwise identical to medium.

---

## 4. Acceptance targets (large)

Everything in [[noether-1.1-medium]] §4 **plus**:

| Check | Target | Source |
|---|---|---|
| Compressible generality | same weights, laminar→turbulent + incompressible→compressible via conditioning only | [[conditioning-and-constants-1.1]] |
| Continuum–particle coupling | settling-particles benchmark: momentum-consistent field↔particle exchange | [[open-architectural-problems]] P4 |
| MHD multi-physics | add $\mathbf B$ stream without a core code path; magnetic helicity discovered | [[field-token-streams-1.1]], [[discovered-conservation-1.1]] |
| Cross-regime transfer | rung $n{+}1$ does not degrade rung $n$ (no catastrophic forgetting) | [[training-scheme-1.1]] |

Corresponds to [[00-implementation-plan]] milestone M7.

## See Also
- [[noether-1.1-medium]] — the derivation and hyperparameters this scales
- [[noether-1.1-fineness-and-token-density]] — the token-density diagnosis behind this config's finer `patch_domain_size`
- [[backbone-1.1]] — the block stacked $L=24$ deep here
- [[training-scheme-1.1]] — rungs 4–6 of the regime curriculum
