# Noether 1.0 — Full Architecture, Numbers & Implementation Spec

**Type:** Concept — initial model, concrete instantiation (folder: Noether 1.0)
**Status:** The named, sized, hyperparameterized model resulting from every decision made across this folder, now specified to **implementation depth** — every tensor shape, module, and initialization recipe needed to write code. Resolves [[open-architectural-problems]] Problem 11. **All numbers below are principled design estimates, not empirically tuned** — no training has run yet; treat this as the checkpoint config to implement and validate, per the vault's preliminary-research status.
**Related Concepts:** [[00-initial-model-overview]], [[pipeline-contract]], [[training-curriculum]], [[graph-tokenizer]], [[initial-model-architecture]], [[multihead-attention]], [[normalization-scheme]], [[open-architectural-problems]], [[smoke-test-bringup]], [[intelligent-patching]]
**Related Summaries:** [[possible-architectures]] (the benchmark this sizing targets), [[gns-graph-network-simulators]], [[dynami-cal-graphnet]], [[poseidon-pde-foundation-model]]

---

## Why "Noether"

Every load-bearing mechanism in this design is a mechanized instance of **Noether's theorem** — symmetry $\Leftrightarrow$ conserved quantity. The skew-symmetric channel conserves energy because it generates a rotation (a continuous symmetry) in latent space; the S1.1 projection reads off the conserved quantities a symmetry implies and enforces them exactly; the S6.4 SFA-style discovery loss is literally *searching for the conserved quantities of symmetries the model hasn't been told about*. The name is not decoration — it names the organizing principle.

---

## 1. Scale points

Sized to land on [[possible-architectures]]'s existing benchmark sweep (100K → 1M → 10M params on 1D Burgers) rather than inventing new scale targets.

| Config | $d$ | $L$ | $H$ | $N_q$/$N_q^{\text{hi}}$ | $d_{\text{hi}}$ | Core params | Target bucket |
|---|---|---|---|---|---|---|---|
| **Noether-1.0-nano** | 64 | 2 | 2 | 8 / 8 | 32 | ≈ 126K | 100K |
| **Noether-1.0-base** | 128 | 4 | 4 | 8 / 8 | 64 | ≈ 874K | 1M |
| **Noether-1.0-large** | 256 | 10 | 8 | 8 / 8 | 128 | ≈ 8.17M | 10M |

**Primary target: Noether-1.0-base.** It maps directly onto [[possible-architectures]]'s "passing" bar — relative $L^2<0.05$ at $t=2$, training under 30 minutes on one GPU. Everything below is worked out in full for base; §4 gives the exact formulas nano/large use.

*(Numbers revised 2026-07-01 from an earlier draft of this page: the encoder budget in §3.3 uses a concatenation-based query resolution that is materially cheaper than the earlier rough estimate, and the conditioning MLP input dimension is corrected from 3 to 1 — see §3.4. Separately, the graph is now **dynamic and adaptive** rather than static — §3.2 — which changes the node/token count (63/126, not 41/82) but not the parameter budget below; and a genuine bug was caught and fixed — §3.6 — where the residual head and decoder had been silently indexed as if patches mapped 1:1 to grid points, which they don't.)*

---

## 2. End-to-end forward pass (Noether-1.0-base, 1D Burgers, one training step)

This section is the literal thing to implement — every stage of [[pipeline-contract]], with shapes, for a single training example: predict $u_{t+1}$ from $(u_t, u_{t-1})$ at 128 grid points.

```
INPUT:  u_t, u_{t-1} ∈ R^128        (field values, 2 most recent snapshots)
        ν ∈ R                       (viscosity, this trajectory's physics parameter)
        Δt = 0.01

──────────────────────────────────────────────────────────────────
STAGE 1 — Nondimensionalize ([[normalization-scheme]])
──────────────────────────────────────────────────────────────────
φ0 = L0 = T0 = 1        # domain [0,1] and IC amplitudes are already O(1) — see §3.1
û_t, û_{t-1} = u_t, u_{t-1}                        (identity for this benchmark)
z_param = MLP_param([log(1/ν)])           ∈ R^128  (§3.4)
z_dt    = MLP_dt([log(Δt/T0)])            ∈ R^128

──────────────────────────────────────────────────────────────────
STAGE 2 — Graph build (DYNAMIC — rebuilt every step from that step's
          canonical 128-point state; §3.2. Adaptive patching, decided
          2026-07-01, to test Problem 7 in the simplest possible setting)
──────────────────────────────────────────────────────────────────
η(x) = |∂u/∂x|  on the current canonical 128-point field   (shock monitor)
recursive bisection of [0,1]: start at 1 root patch; repeatedly split the
    highest-η splittable patch in half (min width = 2 grid points, max
    depth 6 — see §3.2's width-audit fix) — exactly 31 splits, always, to reach 32 active leaves
    (the token budget matching Noether-1.0-base's existing sizing)
fine nodes = the 32 active leaves (VARIABLE width, denser near the shock)
hierarchy  = the bisection tree's ANCESTOR nodes above those 32 leaves —
             a full binary tree with 32 leaves has exactly 31 internal
             nodes (leaves−1); root = the single top/global node
edges = radius-graph(active leaves, r=2 neighbors, periodic)
      ∪ restriction/prolongation edges along the SAME bisection tree
Total nodes per snapshot: 32 + 31 = 63.  Total tokens (2-step context): 126
The graph is EPHEMERAL — re-derived fresh each step from the canonical
state, never tracked/matched across steps; there is no node identity to
keep consistent, which is exactly why this composes safely with S1.1
(see the projection note in Stage 5/6 below)

──────────────────────────────────────────────────────────────────
STAGE 3 — Dual-path encoder, applied to BOTH context snapshots (§3.3)
──────────────────────────────────────────────────────────────────
for t in {t, t-1}, for each of the 32 fine patches i (4 grid points each):
    δ_p    = x_p − x_i                              (offset within patch)
    γ(δ_p) = Fourier(δ_p) ∈ R^16                    (8 bands × sin/cos)
    feat_p = concat(γ(δ_p), û(x_p)) ∈ R^17
    K = W_K · feat_p  ∈ R^{4×16}                    (d_head = d/N_q = 128/8 = 16)
    V = W_V · feat_p  ∈ R^{4×16}
    A_lo = softmax(Z_q^lo · K^T / √16) ∈ R^{8×4}     (standard cross-attn, NOT symmetric —
    out_lo = A_lo · V  ∈ R^{8×16}                     queries and points are different kinds of object)
    h_i^lo = concat(out_lo, over the 8 query slots) ∈ R^128     ← resolves graph-tokenizer's
                                                                   R^{N_q×D} into a single R^d vector
    [identical structure for the residual path, d_hi=64, d_head^hi=8]  → h_i^hi ∈ R^64

pool up: h_{L1,k}^{lo,hi} = mean(h_{4k..4k+3}^{lo,hi})  for k=0..7      (parameter-free mean-pool)
         h_top^{lo,hi}    = mean(h_{L1,0..7}^{lo,hi})                  (parameter-free mean-pool)

add conditioning to every node, both timesteps:
    h̃_i = h_i^lo + z_param + z_dt

──────────────────────────────────────────────────────────────────
STAGE 4 — Backbone, L=4 layers (§3.5) — operates on h^lo (d=128) only;
          h^hi rides alongside untouched (deterministic core doesn't drive it)
──────────────────────────────────────────────────────────────────
for layer = 1..4:
    for head = 1..4:                                  (d_head = d/H = 32)
        q_i = k_i = W^(head) h̃_i           ∈ R^32     (tied Q=K ⇒ symmetric score)
        v_i       = W_V^(head) h̃_i          ∈ R^32
        A_ij = tanh( (q_i·k_j + q_j·k_i) / (2√32) )    for (i,j) ∈ edges only
        Δ^(head)_i = Σ_j A_ij (v_j − v_i)   ∈ R^32
        Δ^(head)_i = W_O^(head) Δ^(head)_i  ∈ R^128    (per-head up-project, then SUMMED not concatenated)
    self_i = (W_skew − W_skew^T − γI) h̃_i   ∈ R^128
    h̃_i ← h̃_i + ε · tanh( self_i + Σ_head Δ^(head)_i )        # attention sub-step
    h̃_i ← h̃_i + ε · W2 · renorm(GeLU(W1 h̃_i))                # FFN sub-step (F1)

──────────────────────────────────────────────────────────────────
STAGE 5/6 — Residual prediction IS the decoder (no separate flat-per-patch
            head — see §3.6/§3.8's fix); single-step, MTP deferred, §8
──────────────────────────────────────────────────────────────────
Δ̂(y) = Dec_lo(h_i^{lo,(L)}, γ(y−x_i)) + Dec_hi(h_i^{hi}, γ(y−x_i))    for y = each of the 128
                                                                        canonical grid points
                                                                        (i = y's containing leaf)
û_{t+1}(y) = û_t(y) + Δ̂(y)                                            direct-state residual update,
                                                                        now correctly at 128-point res.

──────────────────────────────────────────────────────────────────
STAGE 5b/10 — Conservation projection (S1.1, k=1 — §3.7), ALWAYS evaluated
              at the fixed 128 canonical points, never over the current
              (adaptive, variable-width) node set — see §3.7's fix
──────────────────────────────────────────────────────────────────
M̂        = (1/128) Σ_y û_{t+1}(y)                 (discrete mean ≈ ∫u dx, fixed quadrature)
M_target = (1/128) Σ_y û_t(y)                      (previous mass — EXACTLY conserved, periodic BC)
û*_{t+1}(y) = û_{t+1}(y) − (M̂ − M_target)          for every canonical y   [uniform DC-offset correction]
    # for Burgers the projection uses this direct computation, not the learned readout Ĉ(h_top)
    # — see §3.7's readout-vs-direct note. Anchoring to the FIXED grid (not the current adaptive
    # node partition) is what makes this robust to the graph changing shape every step.

──────────────────────────────────────────────────────────────────
STAGE 7/8 — Diffusion / regime router: SKIPPED for this implementation (§8, deferred by design)
──────────────────────────────────────────────────────────────────

STAGE 11 — Re-dimensionalize: identity (φ0=1)

OUTPUT: u_{t+1} ∈ R^128
```

---

## 3. Module specifications

### 3.1 Nondimensionalization

For this benchmark, $\phi_0=L_0=T_0=1$ (identity rescaling) — the domain $[0,1]$ and sinusoidal-IC amplitudes ($O(1)$ by construction of [[possible-architectures]]'s benchmark) are already dimensionless. **This is a genuine simplification specific to Burgers**; a real multiphysics input would compute $L_0,U_0,T_0$ from characteristic scales per [[pfm-interface-design]]. Flagged explicitly so the identity choice isn't mistaken for a general rule.

### 3.2 Graph construction — dynamic, adaptive (decision 2026-07-01)

**Rebuilt every step from that step's canonical 128-point state — not cached.** An earlier draft of this page used a static, fixed-patch-size graph (valid for a plain uniform grid) and left [[open-architectural-problems]] Problem 7 (dynamic topology, momentum survival across topology churn) deferred to a future 2D benchmark. Decided instead: exercise adaptive patching **now**, in the simplest possible setting, since 1D Burgers' shocks are exactly the kind of localized moving feature intelligent patching targets.

- **Monitor function:** $\eta(x)=|\partial u/\partial x|$ on the current canonical 128-point field — a cheap, parameter-free shock detector (no new learned weights).
- **Refinement — 1D recursive bisection** (this benchmark's instance of [[intelligent-patching]]'s quadtree/octree): start from 1 root patch spanning $[0,1]$; repeatedly split the highest-$\eta$ *splittable* patch into two equal halves (minimum width **2** grid points $\Rightarrow$ max depth **6** — one level finer than the original fixed patch size; see the width-audit fix below). Reaching exactly 32 leaves from 1 root always takes **exactly 31 splits**, regardless of where they concentrate — the token budget is controlled by iteration count, not a threshold to tune.
- **Fix (2026-07-01, caught at implementation — width audit):** an earlier draft set minimum width 4 (max depth 5). That is arithmetically inconsistent with the 32-leaf budget: 32 leaves of width $\ge4$ points summing to 128 forces *every* leaf to width exactly 4 — the bisection would always terminate in the uniform partition, making the "variable width, denser near the shock" property (and §7's topology-churn ablation) vacuous. Minimum width 2 (max depth 6) restores genuine adaptivity (up to 64 min-width leaves are now representable, so the 32-leaf budget genuinely redistributes) while preserving the 32-leaf / 63-node / 126-token budget and every parameter/FLOP number on this page. The static-graph ablation baseline remains the uniform 32$\times$4 partition.
- **Fine nodes:** the 32 active leaves — variable width, dense near the shock, coarse in smooth regions.
- **Radius graph:** each active leaf connects to its 2 nearest active-leaf neighbors (periodic wraparound), regardless of width.
- **Hierarchy = the bisection tree's ancestor structure, not a fixed grouping.** A full binary tree (every split node has exactly 2 children) with 32 leaves has *exactly* 31 internal nodes (the identity $I=L-1$) — these are the supernodes, at whatever depth the tree's shape puts them; the root is the single top/global node. No separate "group by 4" rule is needed or well-defined once leaf depths vary.
- **Restriction ($R$) / prolongation ($P$):** parameter-free mean-pool / broadcast along the *same* tree edges, exactly as before — classical multigrid keeps these fixed and geometric, putting all learned physics in the relaxation step (the backbone's attention).
- **Total nodes/snapshot:** $32+31=63$. Total tokens (2-step context): **126**.
- **Ephemeral by construction:** the graph is never persisted or matched across steps — it is a pure function of that step's canonical state, recomputed from scratch every time. This is precisely what makes Problem 7's conservation worry tractable here (see §3.7).

### 3.3 Dual-path encoder — resolving graph-tokenizer's $N_q\times D$ into a single $d$-vector

[[graph-tokenizer]] specifies the per-node latent as $\mathbf h_i^0\in\mathbb R^{N_q\times D}$ — a *set* of $N_q$ query outputs, not a single vector. That's left ambiguous for how it enters a backbone that (throughout this folder) has been written as operating on a single $d$-dimensional $h_i$. **Resolved here:** set each query slot's internal width to $d_{\text{head}}=d/N_q$ and **concatenate** the $N_q$ cross-attention outputs to form the final $d$-dimensional node latent, rather than pooling or adding a separate projection layer. With $N_q=8,\ d=128$: $d_{\text{head}}=16$.

Per patch (4 points), per query slot:
$$K,V = W_K,W_V\cdot\text{feat}\in\mathbb R^{4\times16},\qquad A^{\text{lo}}=\mathrm{softmax}\Big(\tfrac{Z_q^{\text{lo}}K^\top}{\sqrt{16}}\Big)\in\mathbb R^{8\times4},\qquad \text{out}=A^{\text{lo}}V\in\mathbb R^{8\times16},$$
then `h_i^lo = flatten(out)` $\in\mathbb R^{128}$. This is standard (non-symmetric) cross-attention — queries and raw points are different kinds of object, so there's no reciprocity requirement, unlike the backbone's attention.

**Parameter count** (per path, shared across all patches and both timesteps):
$$W_K,W_V\in\mathbb R^{d_{\text{head}}\times(d_\gamma+d_u)},\quad Z_q\in\mathbb R^{N_q\times d_{\text{head}}} \;\Rightarrow\; \text{params}=N_q d_{\text{head}} + 2d_{\text{head}}(d_\gamma+d_u) = d + \frac{34d}{N_q}.$$
For $N_q{=}8,d{=}128$: $128 + 4.25{\times}128 = 672$. Residual path ($d_{\text{hi}}{=}64,N_q^{\text{hi}}{=}8$): $336$.

This is a **materially cheaper** result than an earlier draft's estimate (which implicitly assumed $d_{\text{head}}=d$, i.e. $N_q$ full-width outputs pooled by an extra projection) — the concatenation resolution needs no separate pooling layer at all.

### 3.4 Conditioning

**Correction:** an earlier draft used a 3-dimensional input (`[log Re, log Ma, log Pr]`-style) for $z_{\text{param}}$, copying the general multiphysics interface. **1D Burgers has exactly one physical parameter, $\nu$.** $z_{\text{param}}$'s input is the single scalar $\log(1/\nu)$ (a Reynolds-analog), not a 3-vector.
$$z_{\text{param}} = \mathrm{MLP}\big([\log(1/\nu)]\big):\ 1\to64\to d,\qquad z_{\Delta t}=\mathrm{MLP}([\log(\Delta t/T_0)]):\ 1\to64\to d.$$
Params: $2\times(1\times64+64\times d) = 128+128d$. For $d{=}128$: $16{,}512$. Injected additively to every node's latent (both context timesteps), per [[initial-model-architecture]].

### 3.5 Backbone layer

One shared skew-symmetric self-term ($W_{\text{skew}}\in\mathbb R^{d\times d}$, $d^2$ params) $+$ $H$-head symmetric interaction (tied $Q{=}K$, per-head value + up-projection, summed not concatenated, $3d^2$ params total, $H$-invariant) $+$ F1 semi-orthogonal FFN sub-step ($8d^2$ params at $4\times$ expansion) $= 12d^2$/layer — full derivation and the "identical to a vanilla transformer block" observation carried over from the prior draft (§7 below).

### 3.6 Residual prediction — folded into the decoder, not a separate head

**Fix (2026-07-01), caught while working through the adaptive-graph consequences below:** an earlier draft specified a flat linear head $\hat\Delta_i=W_{\text{res}}h_i$ producing one scalar per *patch* (32 values), silently indexed as if it lined up 1:1 with the 128 *grid points* — it doesn't, since a patch spans multiple points (and now a *variable* number, given §3.2). The correct design — and what [[initial-model-architecture]] Stage 5 actually specifies — is that the residual is the decoder's own continuous output: $\hat\Delta(y)=\mathrm{Dec}(h_i^{(L)},\gamma(y-x_i))$ for any queried $y$, additive across the dual path (§3.8). There is no separate flat-per-patch head to budget or implement; removing it drops the parameter table's earlier (negligible, $d$-sized) "residual head" line entirely — it was double-counting the decoder.

### 3.7 Conservation projection — exact closed form for $k=1$, anchored to the canonical grid

Given the discrete mass functional $M(u)=\tfrac1{128}\sum_y u(y)\approx\int u\,dx$ **evaluated at the fixed 128 canonical grid points**, the general S1.1 projection $x^*=\hat x-M^{-1}A^\top(AM^{-1}A^\top)^{-1}(A\hat x-b)$ specializes, for $A=\tfrac1{128}\mathbf 1^\top$ and identity metric, to a **uniform additive correction**:
$$A^\top(AA^\top)^{-1} = \mathbf 1\ \text{(a vector of all ones)}\quad\Rightarrow\quad u^*(y) = \hat u(y) - (M(\hat u) - M_{\text{target}})\ \ \forall y,$$
i.e. subtract the scalar mass excess from every grid point equally. One line of code, no matrix inverse to actually form.

**Why "canonical grid," not "current nodes" — this is the fix adaptive patching (§3.2) forces.** With uniform fixed-width patches, averaging over nodes and averaging over grid points coincide, so the earlier draft could get away with either. Once patches have **variable width** (§3.2), an unweighted average over nodes is wrong — a size-16 patch and a size-4 patch shouldn't count equally toward the mean. The fix is to **always evaluate the conserved-quantity check on the fixed 128-point canonical quadrature, post-decode**, never on the current adaptive node partition. This is what makes the projection provably indifferent to whatever tokenization happens to be active that step — a generalizable pattern, not a Burgers-only patch: *any* global-integral constraint should anchor to a fixed reference measure, never to the current adaptive partition.

**Readout vs. direct evaluation.** [[pipeline-contract]]'s general design routes the conserved-total check through a *trained* readout $\hat C(h)=W_Ch$ read from the top supernode, because in general the conserved quantity might not be cheaply recoverable from the decoded field alone. For Burgers specifically, $\int u\,dx$ **is** trivially computable from the already-decoded output — so the projection applied to the training/inference output uses the **direct discrete-mean computation** above, not the learned readout. $W_C$ is still trained (as a small diagnostic head predicting $M(u)$ from $h_{\text{top}}$) purely to check that the latent linearly encodes the right functional — a validation signal, not a load-bearing part of the projection for this benchmark.

**Does this resolve Problem 7's momentum-under-topology-churn worry?** Largely yes, for the *adaptive-patching* case specifically (a persistent-identity case — e.g. a moving-particle simulation where "particle 5" must remain particle 5 across steps — is a different, harder problem, still open). Two properties combine to make it tractable here: (1) the global check is anchored to the canonical grid, never to node topology, so it cannot be perturbed by the graph changing shape; (2) the graph itself is ephemeral (§3.2) — rebuilt fresh each step from the canonical state, with no cross-step node identity to keep consistent, so there is nothing for topology churn to corrupt. This is a genuinely checkable claim, not just an argument: running this benchmark should show mass conservation holding to the same $<10^{-4}$ tolerance as the static case (§7's eval protocol) — if it doesn't, the argument above has a hole.

### 3.8 Decoder — additive two-head SIREN, now also the residual head (§3.6)

$$\hat\Delta(y) = \mathrm{Dec}_{\text{lo}}\big(h_i^{\text{lo}},\gamma(y-x_i)\big) + \mathrm{Dec}_{\text{hi}}\big(h_i^{\text{hi}},\gamma(y-x_i)\big),$$
each a 2-hidden-layer SIREN MLP, hidden width $=d=128$, input $(d+d_\gamma)=144$, output $1$ (the predicted **residual**, added to $\hat u_t(y)$ per §2's Stage 5/6 — not the raw next state directly). See §5 for the SIREN initialization recipe this needs.

### 3.9 Diffusion / router — not implemented for this pass

Per [[smoke-test-bringup]] §0, both are deferred: moderate-$\nu$ viscous Burgers doesn't need the generative core to validate the architecture. Budgeted separately in §4.1 for when they're built.

---

## 4. Parameter budget (exact formulas + worked totals)

Let $d_{\text{head}}=d/N_q$, $d_{\text{head}}^{\text{hi}}=d_{\text{hi}}/N_q^{\text{hi}}$, $d_\gamma{=}16$, $d_u{=}1$ (fixed, Burgers-specific), $N_q=N_q^{\text{hi}}=8$ (fixed across scales), $d_{\text{hi}}=d/2$.

| Component | Formula | nano ($d{=}64$) | base ($d{=}128$) | large ($d{=}256$) |
|---|---|---|---|---|
| Backbone | $12Ld^2$ | 98,304 | 786,432 | 7,864,320 |
| Smooth encoder | $d+34d/8=5.25d$ | 336 | 672 | 1,344 |
| Residual encoder | $2.625d$ | 168 | 336 | 672 |
| Conditioning (2 MLPs) | $128+128d$ | 8,320 | 16,512 | 32,896 |
| Conserved-readout ($k{=}1$) | $d$ | 64 | 128 | 256 |
| Decoder (2 heads, hidden$=d$) | $4d^2+34d$ | 18,560 | 69,888 | 270,848 |
| Typed-edge encoder | — | **0** (not instantiated — 1 node type) | 0 | 0 |
| Hierarchy $R$/$P$ | — | **0** (parameter-free) | 0 | 0 |
| **Non-backbone subtotal** | | **27,448** | **87,536** | **306,016** |
| **Total core** | | **125,752 ≈ 126K** | **873,968 ≈ 874K** | **8,170,336 ≈ 8.17M** |

Comfortably within each target bucket. No hidden Poisson-solve or KKT-projection parameters exist (§3.7 is closed-form/analytic, free in budget though not in compute — §7).

### 4.1 Optional diffusion add-on

Unchanged in spirit from the prior draft: $d_{\text{diff}}{=}64,L_{\text{diff}}{=}2$ backbone-only $=12\times2\times64^2=98{,}304$, plus a $\approx$5K timestep-conditioning MLP $\Rightarrow$ **≈103K additional**, giving Noether-1.0-base+diff $\approx$ **977K total**. Not built for this pass (§3.9).

---

## 5. Initialization scheme

| Parameter | Recipe | Rationale |
|---|---|---|
| $W^{(h)}, W_V^{(h)}, W_O^{(h)}$ (attention) | Xavier-normal | standard |
| $W_{\text{skew}}$ (raw, pre-antisymmetrization) | i.i.d. $\mathcal N(0,1/d)$ | random-matrix theory: largest singular value of a $d\times d$ matrix with this scaling concentrates near 2, so $\|W_{\text{skew}}-W_{\text{skew}}^\top\|_2\lesssim4$ — the exact bound §6.1's $\epsilon$ schedule assumes |
| $\gamma$ | constant init $0.01$, learnable | near-conservative default; lets training discover Burgers' true $\nu$-dependent dissipation (§3.7's diagnostic identity) rather than assuming exactly $\gamma=0$ |
| $W_1,W_2$ (F1 semi-orthogonal FFN) | orthonormal init via QR decomposition of a random Gaussian matrix; **re-orthogonalize every 100 optimizer steps** (QR retraction) | standard, cheap approximate-Stiefel-manifold optimization — train with ordinary AdamW, periodically snap back to the constraint set rather than using a full Riemannian optimizer |
| $Z_q^{\text{lo}}, Z_q^{\text{hi}}$ (query banks) | i.i.d. $\mathcal N(0,0.02^2)$ | standard transformer embedding-init convention |
| $W_K, W_V$ (encoder) | Xavier-normal | standard |
| Conditioning MLPs | Xavier-normal weights, zero bias | standard |
| $W_C$ (conserved-readout) | $\mathcal N(0,0.02^2)$ | small-output regression head, generic init sufficient (it's a diagnostic, not load-bearing — §3.7) |
| Decoder (SIREN, both heads) | **first layer:** $\mathrm{Unif}(-1/{\text{fan\_in}}, 1/{\text{fan\_in}})$; **subsequent layers:** $\mathrm{Unif}(-\sqrt{6/\text{fan\_in}}/\omega_0,\ \sqrt{6/\text{fan\_in}}/\omega_0)$ with $\omega_0=30$; activation $\sin(\omega_0\cdot(\cdot))$ | the standard SIREN initialization (Sitzmann et al. 2020) — needed so the decoder can emit high frequencies at all; a generic MLP init here would reintroduce the spectral bias [[open-architectural-problems]] Problem 3 flags |

---

## 6. Hyperparameters

### 6.1 Backbone

| Parameter | Value | Rationale |
|---|---|---|
| $\epsilon$ (Euler step) | start $0.1$, linear warmup to $0.3$ over first 5% of steps | stays below $2/\|W_{\text{skew}}\|_2\approx2/4=0.5$ (§5's init bound), the Problem-10 stability requirement |
| $H$ (heads, base) | 4 | likely over-provisioned for scalar 1D Burgers (no elliptic pressure coupling, no multi-field interaction); kept at 4 for consistency with the general design, with an ablation (§7) checking whether 2 suffices |
| FFN expansion | $4\times$ | standard ratio; F1's semi-orthogonality constrains *how* $W_1,W_2$ are optimized, not the ratio itself |
| Activation $\sigma$ / $\phi$ | $\tanh$ (attention) / GeLU, renormalized (FFN) | non-expansive; keeps F1's norm-bound honest |

### 6.2 Optimizer & training

| Parameter | Value |
|---|---|
| Optimizer | AdamW, excluding $W_{\text{skew}}$ and F1's Stiefel factors from weight decay |
| Peak LR | $3\times10^{-4}$, cosine decay, 5% linear warmup |
| Batch size | 64–128 trajectories |
| Weight decay | $0.01$ (unconstrained params only) |
| Gradient clipping | global norm $1.0$ |
| Precision | bf16 for backbone/encoder/decoder; **fp32 for §3.7's projection arithmetic** |
| Curriculum | [[training-curriculum]]'s 4 phases, Burgers-scale lengths per [[smoke-test-bringup]] §5 |

---

## 7. Eval protocol & success criteria

| Check | Metric | Target | Source |
|---|---|---|---|
| **Primary accuracy** | relative $L^2$ error at $t=2$ | $<0.05$ | [[possible-architectures]] |
| **Conservation (projection on)** | relative drift of $\int u\,dx$, full rollout to $t=2$ (190 steps from the 10-step context; "90-step" in earlier drafts was [[possible-architectures]]'s now-fixed inconsistency) | $<10^{-4}$ (near machine precision) | S1.1, §3.7 |
| **Conservation (projection off)** | same, ablated | expected to drift measurably — the contrast is the evidence | — |
| **F1 vs. F3 ablation** | accuracy + effective-rank-of-latents diagnostic | F1 shows higher effective rank at depth $L$ | Problem 12 |
| **Symmetric-only vs. +skew** | energy decay rate vs. exact $-\nu\int u_x^2\,dx$ (§3.7) | symmetric-only shows *excess* decay, not "any decay" — decay itself is correct viscous physics | [[symmetric-attention-physics]] |
| **Conservation under topology churn** (new — testable now, not deferred) | same mass-drift metric as row 2, but with §3.2's adaptive graph active vs. an ablation forcing the static uniform graph | should match within noise — this is the direct empirical test of §3.7's "ephemeral graph + canonical-grid anchoring resolves Problem 7" argument | [[open-architectural-problems]] Problem 7 |
| **Inference cost** | wall-clock/step vs. RK45 | see §8 | — |

**Caveat:** viscous Burgers isn't energy-conserving in the true dynamics, weakening the symmetric-only ablation's diagnostic power. Run it on *inviscid* Burgers ($\nu=0$) as a stricter test, alongside the standard viscous run for the primary accuracy number.

**Deferred to 2D NS:** Problem 9's equivariance ablation, the P2.1/P2.2 divergence-free+BC projection, S4.1's typed-edge coupling — none exist/apply in 1D periodic Burgers. (Problem 7's *adaptive-patching* sub-case is no longer on this list — it's active and tested here, §3.2/§3.7; the *persistent-identity*, e.g. moving-particle, sub-case remains genuinely deferred.)

---

## 8. Inference cost — an honest estimate

$\approx2\times874\text{K}\times126\approx220\text{M}$ FLOPs/step (using §3.2's adaptive-graph token count, 63 nodes $\times$ 2 context steps $=126$ — up from the static case's 82, since the full 31-node ancestor set is now kept rather than 2 clean hierarchy levels) — still trivial for any GPU. **Wall-clock at this scale is overhead-bound, not FLOPs-bound**: expect 1–5 ms/step, now also including the per-step bisection-tree rebuild (§3.2) as part of that overhead, not just kernel launch. A full 190-step rollout finishes in well under a second — comparable to, not faster than, RK45 at this resolution. **"Faster than classical solvers" is not demonstrable at nano/base scale and shouldn't be claimed from it**; the claim only becomes testable at resolutions where classical solve cost (elliptic solves, mesh generation, adaptive refinement) grows faster than the model's largely-fixed per-step cost.

---

## 9. What this leaves open

**Update (2026-07-01):** the "make Noether 1.0 trainable on general physics" pass closed every problem this section previously listed as open — see [[open-architectural-problems]] Addendum 4. What was open here is now *decided* (mechanisms), leaving *implementation* for the multiphysics configs. Status of each former item:

- **Problem 4** — cross-type message form: **DECIDED** (nondimensional scale-bridge → IBM adjoint spread/interpolate → conserved-flux antisymmetry + learned residual). Implementation still reachable only once S4.4's second node type exists.
- **Problem 7** — persistent-node-identity sub-case: **DECIDED** (persistent nodes / ephemeral edges + antisymmetric conserved flux + particle-sum anchoring + smooth cutoff). The *adaptive-patching* sub-case here (§3.2/§3.7) remains this page's active, empirically-testable instance; the persistent-identity mechanism is specified for the S4.4 particle benchmark. Particle fragmentation flagged as a narrow remaining edge case.
- **Problem 8** — variable field cardinality: **DECIDED** (S8.1 typed-encoder additive latent + S8.5 universal normalization; S8.4 hypernetwork upgrade path). Burgers still has one field, so this page's $d_u=1$ encoder is unchanged; the mechanism activates for the RBC config.
- **Problems 9/10** — **DECIDED** (full scored passes): P9 dissolves the equivariance binary via covariant conditioning of symmetry-breaking fields (P9.2/P9.3, ablation-gated); P10 commits analytic spectral-norm control + ReZero gain + the standard recipe. §5's init bound and §6.1's $\epsilon$ schedule are the P10 instantiation; the P10.4 ReZero gain becomes load-bearing at the deeper RBC scale (it was safely omittable at $L{=}4$ Burgers).
- **A 2D Rayleigh–Bénard Noether-1.0 config** — the natural next artifact: the RBC analogue of this page. §3.2's graph, §7's ablation set, and the hierarchy shape are Burgers-specific; RBC additionally needs the S8.1 multi-field encoder, P2.1/S18 elliptic pressure projection (built, not just specified), the P9.3 selective-equivariant channels with gravity as covariant conditioning, and the P5 diffusion path for turbulent Ra. All mechanisms are now decided; this is an implementation-spec exercise, not open design.

---

## [AI Inference]

**[AI Inference]:** The $12d^2$/layer match to a vanilla transformer block remains the cleanest budget-controlled test available: if Noether-1.0-base underperforms a same-sized ordinary transformer on the accuracy benchmark, the physics constraints cost more expressivity than they buy in structure; if it matches or beats one, they were free.

**[AI Inference]:** §3.7's exact closed-form projection (a one-line mean correction) is a preview of a general pattern worth watching for at 2D/3D scale: whenever a conserved quantity is *cheaply computable from the decoded output directly*, the learned readout $\hat C(h)$ is better used as a training-time diagnostic (does the latent linearly encode what it should?) than as the mechanism actually enforcing the constraint. The readout only becomes load-bearing once the conserved quantity *can't* be cheaply recomputed post-decode — likely true for genuinely discovered tier-2 invariants (S6.4), where no closed form exists to fall back on.

**[AI Inference]:** §3.2/§3.7's resolution of Problem 7 generalizes past this one benchmark: the risk in "dynamic graph topology" was never really about the *topology itself* changing — it was about whether anything *load-bearing* was implicitly keyed to a specific topology. Once every global check is anchored to a fixed reference measure (here, the canonical grid) and every graph is treated as a disposable, freshly-derived function of the current state rather than a persisted object, topology churn has nothing to disturb. The harder version of Problem 7 (moving particles) is hard precisely because particle identity genuinely *is* persistent and physically meaningful across steps — a case where "just re-derive everything fresh" isn't available as an escape hatch, unlike here.

---

## See Also

- [[noether-1.0-rbc]] — the 2D Rayleigh–Bénard multiphysics config that generalizes this page (multi-field encoder, stream-function div-free decode, wall BCs, gravity-as-covariant-conditioning)
- [[smoke-test-bringup]] — the bring-up sequence and the decided quick training set (§4 there)
- [[possible-architectures]] — the benchmark sweep and success criteria this page's sizing targets
- [[pipeline-contract]] — the forward-pass stage order §2's pseudocode instantiates
- [[training-curriculum]] — the loss groups and phase schedule referenced in §6.2
- [[open-architectural-problems]] — Problem 11 (this page); Problems 9/10/12 whose defaults this page makes concrete; Problem 7, partially addressed by §3.2/§3.7's adaptive-graph + canonical-anchoring argument
- [[initial-model-architecture]] — Stage 4's backbone equation, sized in §3.5
- [[graph-tokenizer]] — the $N_q\times D$ encoder ambiguity §3.3 resolves; [[intelligent-patching]] — the hierarchy in §3.2
- [[multihead-attention]] — the general head-structure design §3.5 instantiates
- [[symmetric-attention-physics]] / [[skew-symmetric-attention]] — the math behind §5/§6.1's $\epsilon,\gamma$ bounds
- [[normalization-scheme]] — why no LayerNorm appears anywhere in this budget
