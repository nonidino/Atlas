# Noether 1.1 — Medium config

**Type:** Implementation spec — model size point (folder: Noether 1.1 / Noether 1.1 implementation)
**Status:** Design estimate. The mid-scale 1.1 checkpoint: large enough to exercise multi-field coupling, learned edges, discovered conservation, and diffusion at a meaningful capacity, small enough to train on a single high-memory GPU / Colab A100. **All numbers are principled design estimates — no 1.1 training has run.** Companion: [[noether-1.1-large]]. Implements the sizing conventions of [[noether-1.0]] §4, extended for 1.1's extra modules.
**Design source:** [[00-noether-1.1-overview]], [[backbone-1.1]], [[training-scheme-1.1]].

---

## Is the backbone a deep stack of attention layers, like an LLM? — Yes.

The backbone is **$L$ stacked layers**, each a residual attention+FFN block, exactly like a transformer/LLM in gross structure — you go *deeper* to compute more, and depth is the model's main capacity knob. Three differences from an LLM layer, all from [[backbone-1.1]]:

1. **Depth = latent time.** Each layer is one forward-Euler step of a learned flow ($h^{\ell+1}=h^\ell+\alpha_\ell\epsilon\tanh[\dots]$); stacking $L$ layers integrates the dynamics ODE $L$ times. An LLM's depth is abstract; here it is literally the integrator's step count.
2. **Attention is per-edge-type over a graph, not dense over a sequence.** Each layer runs $T$ typed attention operators (same-field, cross-field co-located, cross-field spatial, learned long-range) on the token graph, with tied $Q{=}K$ (symmetric scores) and a gated antisymmetric flux — not one dense causal-masked block.
3. **No LayerNorm; ReZero gains instead** ([[normalization-scheme]]). Residual connections and an FFN sub-step per layer are the same; the normalization and the score function differ.

So: same "stack many attention blocks" skeleton as an LLM, with the per-layer block specialized for conservative graph dynamics. **Medium uses $L=12$ layers** (large uses 24 — [[noether-1.1-large]]).

---

## 1. Scale point

| Symbol | Meaning | Medium |
|---|---|---|
| $d$ | model width (token latent) | **384** |
| $L$ | backbone layers (Euler steps) | **12** |
| $H$ | attention heads *per edge type* | **6** ($d_{\text{head}}=64$) |
| $T$ | edge types | 4 (same-field, cross-field co-loc, cross-field spatial, learned) |
| $N_q / N_q^{\text{hi}}$ | query-bank sizes (lo / hi path) | 16 / 8 |
| `patch_domain_size` | token-density knob (patch edge, nondim domain units) | **0.0625** (→ 32×16=512 tokens on the 128×64 corpus; ≈4×4 pts/patch) |
| `patch_size_jitter` | multi-scale training multipliers | **(0.5, 1.0, 2.0)** |
| $d_{\text{hi}}$ | high-freq residual path width | 192 ($d/2$) |
| $d_\theta$ | field-descriptor dim | 32 |
| $d_{\text{diff}} / L_{\text{diff}}$ | diffusion denoiser width / depth | 192 / 6 |
| $F$ | *representative* active field count (variable at inference) | 4 |

---

## 2. Parameter budget

Per-layer backbone (derivation from [[backbone-1.1]]): skew self-term $d^2$ + per-type attention $T\!\cdot\!3d^2$ (tied $Q{=}K$: $d^2$, $V$: $d^2$, $W_O^{(\tau)}$: $d^2$, head-count-invariant) + F1 semi-orthogonal FFN $8d^2$ (4× expansion) $=(9+3T)d^2$. For $T{=}4$: **$21d^2$/layer**.

| Component | Formula | Medium ($d{=}384$) |
|---|---|---|
| Backbone | $(9+3T)\,L\,d^2 = 21\cdot12\cdot d^2$ | **37.2M** |
| Field encoders $\mathrm{Enc}_f$ | $\approx F d^2$ | 0.59M |
| Decoder SIREN heads (dual, per field) | $\approx 4F d^2$ | 2.36M |
| Conditioning MLPs ($z_{\text{param}},z_{\Delta t},\gamma_{\text{vec}}$) | $\approx 2d^2$ | 0.29M |
| Learned edge scorer $g_\theta$ | $\approx 3d^2$ | 0.44M |
| Query banks + low-rank modulation $UV^\top$ | $\approx 2d^2$ | 0.29M |
| Field descriptors $\theta_f$ + conservation readouts | $F d_\theta + K d$ | ~0.01M |
| **Core subtotal (deterministic)** | $[(9{+}3T)L + 5F + 7]\,d^2$ | **≈ 41.2M** |
| Diffusion denoiser $\epsilon_\theta$ (add-on) | $12 L_{\text{diff}} d_{\text{diff}}^2 + \text{cond}$ | ≈ 2.7M |
| **Total** | | **≈ 43.9M** |

*The $F$-dependent terms (encoders, decoder heads) scale with the number of active field streams; $F{=}4$ is a representative accounting figure, not a cap — the model handles variable field cardinality ([[field-descriptors-1.1]]). Backbone dominates, so total is roughly $F$-insensitive.*

---

## 3. Module dimensions & hyperparameters

**Edges** ([[edge-generation-1.1]]): same-field KNN $K=16$; cross-field KNN $K'=6$; learned-edge candidate envelope $R=4r$; GumbelSigmoid temperature $\tau_{\text{gs}}$ annealed $2.0\to0.1$; sparsity budget $\le 8$ learned edges/node.

**Backbone** ([[backbone-1.1]]): $W_{\text{skew}}$ Cayley-parameterized (orthogonal, $\|\cdot\|_2\le2$); Euler step $\epsilon$ warmup $0.1\to0.3$ (stays $<2/\|W_{\text{skew}}\|_2$); ReZero $\alpha_\ell,\alpha'_\ell$ init 0; head sum scaled $1/\sqrt H$ at init; gate $g_\tau=\operatorname{clamp}(a_\tau,0,1)$ **frozen at 1** on same-field edges through P1–P2, freed at P3; $\gamma_\ell$ learnable, init 0.01.

**Optimizer & training** (per [[training-scheme-1.1]], scaled from [[noether-1.0]] §6.2): AdamW ($W_{\text{skew}}$ + Stiefel FFN factors excluded from weight decay; re-orthogonalize $W_1,W_2$ every 100 steps); peak LR $2\times10^{-4}$, cosine decay, 5% warmup; batch 32–64 trajectories; grad-clip 1.0; bf16 (fp32 for conservation projection arithmetic); phase schedule P0→P4 with push-forward horizon $1\to2\to4\to8$.

---

## 4. Acceptance targets (medium)

| Check | Target | Source |
|---|---|---|
| G0 tokenizer reconstruction | encode→decode error < data discretization error (held-out turbulent NS) | [[training-scheme-1.1]] P0 |
| Single-step accuracy (2D NS) | rel-$L^2 < 0.05$ at $t{+}1$ | [[impl-eval-benchmarks]] |
| Rollout stability (push-forward) | raw rollout flat to 100 steps | [[training-scheme-1.1]] P2 |
| Divergence discovery | $\lambda_{\nabla\cdot u}$ high on incompressible, $\to0$ on compressible, no code change | [[discovered-conservation-1.1]] |
| RBC attractor statistics | rollout-mean Nu within 5%, spectrum matched (with diffusion on) | [[post-decoder-diffusion-1.1]] |

Trains across regime-curriculum rungs 1–3 (Burgers → 2D NS → RBC); see [[00-implementation-plan]] milestone M6.

## See Also
- [[noether-1.1-large]] — the scaled-up config
- [[noether-1.1-fineness-and-token-density]] — why `patch_domain_size` was lowered from 0.25 and what `patch_size_jitter` buys
- [[backbone-1.1]] — the per-layer block this stacks $L=12$ of
- [[training-scheme-1.1]] — the phase schedule and curriculum
- [[noether-1.0]] — the sizing conventions extended here
