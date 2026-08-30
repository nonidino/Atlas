# Summary: Poseidon — Efficient Foundation Models for PDEs

**Source:** `new/Poseidon Efficient Foundation Models for PDEs.md` (arXiv:2405.19101v2 HTML)
**Authors:** Maximilian Herde, Bogdan Raonić, Tobias Rohner, Roger Käppeli, Roberto Molinaro, Emmanuel de Bézenac, Siddhartha Mishra (Seminar for Applied Mathematics + ETH AI Center, ETH Zurich)
**Venue:** NeurIPS 2024
**arXiv:** 2405.19101v2
**Code / Data:** [github.com/camlab-ethz/poseidon](https://github.com/camlab-ethz/poseidon); models + PDEgym datasets at [huggingface.co/camlab-ethz](https://huggingface.co/camlab-ethz)
**Date Ingested:** 2026-06-29

---

## Overview

**Poseidon** is a family of foundation models for learning the **solution operators** of PDEs. Unlike the autoregressive "next-snapshot" video-transformer paradigm of [[walrus-paper]] and [[gphyt-physics-foundation-model]], Poseidon is built explicitly on the language of **operator learning**: it learns the map $\mathcal{S}:(t,a)\mapsto u(t)$ from an initial condition $a$ (and lead time $t$) directly to the PDE solution at time $t$. Three ingredients define it:

1. **scOT (scalable Operator Transformer)** — a *hierarchical multiscale vision transformer* with shifted-window (SwinV2) attention in a U-Net encoder–decoder, equipped with **lead-time-conditioned layer norm** for continuous-in-time evaluation.
2. **all2all training** — a data-amplification strategy that exploits the **semi-group property** of time-dependent PDE solution operators to turn $K{+}1$ snapshots per trajectory into $\mathcal{O}(K^2)$ training pairs.
3. **A diverse fluid-dynamics pretraining corpus** (6 operators: 4 compressible Euler, 2 incompressible Navier–Stokes; 77,840 trajectories → ~5.11M training pairs), released as the open **PDEgym** collection.

The headline result is the **first clear positive answer to the feasibility question for PDE foundation models**: pretraining on a *very small* set of PDEs (only Euler + Navier–Stokes) yields effective representations that transfer to **15 out-of-distribution downstream tasks**, 9 of which involve PDE families *never seen in pretraining* (waves, Allen–Cahn reaction–diffusion, Poisson, Helmholtz, steady flows). Median efficiency: Poseidon-L matches FNO-with-1024-samples using **~20 samples**; in 4 tasks just **3 samples** suffice.

---

## Problem Formulation: Operator Learning, Not Next-Step Prediction

Poseidon targets a generic time-dependent PDE

$$\partial_t u(x,t) + \mathcal{L}\!\left(u,\nabla_x u,\nabla_x^2 u,\ldots\right) = 0,\quad x\in D\subset\mathbb{R}^d,\ t\in(0,T),$$
$$\mathcal{B}(u)=0 \text{ on } \partial D\times(0,T),\qquad u(0,x)=a(x).$$

The **solution operator** $\mathcal{S}:[0,T]\times\mathcal{X}\to\mathcal{X}$ gives $u(t)=\mathcal{S}(t,a)$ for $\mathcal{X}\subset L^p(D;\mathbb{R}^n)$. The **operator learning task (OLT)** is: *given any $a\sim\mu$, produce the entire trajectory $\{\mathcal{S}^\ast(t,a)\}_{t\in[0,T]}$ from the initial datum alone.*

A crucial conceptual point the paper hammers: **next-step-with-context-window prediction (MPP, DPOT) does *not* solve the OLT.** Those models map a window of $\tau$ past states to the next state; they cannot generate a full trajectory from the initial condition without bootstrapping a context they don't have at $t=0$. This distinction is why Poseidon must be finetuned with context window 1 to even compare against MPP/DPOT (SM C.6), and is a key axis on which Poseidon differs philosophically from the [[arch-autoregressive-transformer]] paradigm.

Time-independent (steady) PDEs are absorbed into the same framework as the **long-time limit** $\lim_{t\to\infty}u=\overline u$ of (1); downstream, steady PDEs (Poisson, Helmholtz, SE-AF) are finetuned at a normalized lead time of 1.

---

## Architecture: scOT (scalable Operator Transformer)

scOT is a **SwinV2 U-Net** acting on function space. For $d=2$, $D=[0,1]^2$.

### Patch partitioning and embedding (the token representation)

An input function $a\in C(D;\mathbb{R}^n)$ is partitioned into $P^2=(J/p)^2$ non-overlapping $p\times p$ patches and **linearly embedded** to a $C$-dimensional latent ($C>n$):

$$(\mathbf v_j)_i = (\mathbf b_{\mathcal E})_i + \sum_{k=1}^{n}\sum_{u,v=1}^{p}(\mathbf W_{\mathcal E})_{i,k,u,v}\,(\mathbf a^p_j)_{k,u,v}.$$

The continuous view (SM 12): $\mathbf v(x)=\hat{\mathbf E}(a)(x)=\sum_\rho \mathbf F\big(\int_{D_\rho} W(x)a(x)\,dx\big)\mathbb{I}_{D_\rho}(x)$ — a **weighted patch average** projected by learnable $\mathbf F\in\mathbb{R}^{C\times n}$. So Poseidon's tokens are **ViT-style continuous linear-patch tokens** (see [[patch-embedding-tokens]]) — but, unlike Walrus/GP$_{\text{hy}}$T, they then enter a *multiscale hierarchy* (patch merging) rather than a flat sequence. Fixed $p=4$.

### SwinV2 windowed attention

Self-attention acts only inside $M\times M$ windows ($M=16$), cyclically shifted by $M/2$ each block so information propagates across windows over depth. The discrete head:

$$\mathcal A_l(\mathbf v)=\text{Softmax}\!\left(\mathbf B_l(\mathbf v)+\frac{\cos\big((\mathbf v\mathbf W_Q^l+\mathbf 1 \mathbf b_Q^{l\top})^\top,(\mathbf v\mathbf W_K^l)^\top\big)}{\tau_l}\right)\big(\mathbf v\mathbf W_V^l+\mathbf 1\mathbf b_V^{l\top}\big).$$

Two SwinV2 specifics matter physically: (i) **scaled cosine attention** $\cos(Q,K)/\tau_l$ (bounded, training-stable — cf. QK-norm in [[walrus-paper]]); (ii) **continuous relative-log-position bias** $\mathbf B_l$ from a shared MLP $\mathcal P(\Delta x,\Delta y)=\text{ReLU}([\text{sign}(\Delta x)\log(1+|\Delta x|),\ldots]\mathbf W_{B,1}+b)\mathbf W_{B,2}$ — a smooth, resolution-portable encoding of relative grid offset.

### Hierarchical multiscale U-Net

$L=4$ levels. Encoder SwinV2 stages alternate with **patch merging** (linear $4{:}1$ downsample, latent dim doubles); decoder stages alternate with **patch expansion** (linear upsample). Same-scale encoder/decoder levels are bridged by **time-conditioned ConvNeXt** blocks (depthwise conv, kernel 7); the bottleneck is convolution-free. Output is reassembled by **patch recovery** (linear back-projection) + a kernel-5 mixup convolution. This multiscale design — rather than just model size — is shown to be decisive (CNO-FM, a multiscale CNN trained on the *identical* data, is markedly worse). See [[hierarchical-windowed-tokens]].

### Lead-time-conditioned LayerNorm (continuous-in-time)

The single most important physics adaptation. Every LayerNorm is **modulated by lead time** $t$:

$$\mathcal N(\mathbf v,t)=\alpha(t)\odot\frac{\mathbf v-\mu(\mathbf v)}{\sigma(\mathbf v)}+\beta(t),\qquad \alpha(t)=\mathbf W_\alpha t+\mathbf b_\alpha,\ \ \beta(t)=\mathbf W_\beta t+\mathbf b_\beta,$$

with $\alpha,\overline\alpha,\beta,\overline\beta$ learnable (affine in $t$; small MLPs also possible). Because $t$ enters *continuously*, a single trained scOT evaluates the solution at **any** $t\in\mathbb{R}_+$ in one shot — no fixed timestep, no rollout required (though autoregressive rollout remains optional and sometimes preferable). This is the mechanism that lets steady PDEs be treated as $t\to$ "lead time 1" limits, and that lets all2all training use arbitrary time gaps.

### Three Poseidon sizes

| Model | Params |
|---|---|
| Poseidon-T | ≈ 21 M |
| Poseidon-B | ≈ 158 M |
| Poseidon-L | ≈ 629 M |

---

## all2all Training: Semi-Group Data Amplification

The solution operator obeys the **semi-group property**

$$u(t^\ast)=\mathcal S(t^\ast,a)=\mathcal S(t^\ast-t,\,\mathcal S(t,a)),\qquad 0\le t\le t^\ast\le T.$$

So *any* snapshot in a trajectory is a valid initial condition for the snapshots that follow it. The **all2all loss** uses every ordered pair $(u(t_k),u(t_{\bar k}))$ with $k\le\bar k$:

$$\widehat{\mathcal L}(\theta)=\frac{1}{M\widehat K}\sum_{i=1}^{M}\sum_{\substack{k,\bar k=0\\ k\le \bar k}}^{K}\big\|\mathcal S(t_{\bar k}-t_k,u_i(t_k))-\mathcal S^\ast_\theta(t_{\bar k}-t_k,u_i(t_k))\big\|^p_{L^p(D)},$$

with $\widehat K=\tfrac{(K+1)(K+2)}{2}$. This converts the **linear** $K$ pairs of vanilla training (5) into **quadratic** $\mathcal{O}(K^2)$ pairs. With 11 snapshots/trajectory that is **66 input–output pairs per trajectory**, turning 77,840 trajectories into ~5.11M training examples. ($p=1$ throughout; a relative-error form is used in practice, SM C.)

**Why it works (beyond data volume):** it forces the model to learn the operator for *variable time gaps* $t_{\bar k}-t_k$ and from *non-canonical* initial states (mid-trajectory turbulence, not just smooth ICs), which is exactly what lead-time conditioning needs and what generalization to new timescales requires. Ablation (SM D.6.1, Fig. 45): all2all >> vanilla for CNO on NS-SL. Caveat: cost grows quadratically in snapshot count; subsampling snapshots trades cost for accuracy.

### Inference

Either **direct** continuous-in-time evaluation $\mathcal S^\ast_{\theta^\ast}(t,a)$, or **autoregressive rollout** $\mathcal S^\ast(t^\ast_\kappa-t^\ast_{\kappa-1},\mathcal S^\ast(\ldots\mathcal S^\ast(t^\ast_1,a)))$ over a chosen time partition — relevant to [[autoregressive-rollout-stability]].

---

## Finetuning Strategy

Parameters split as $\theta=[\widehat\theta,\widetilde\theta,\widetilde\theta^{\mathcal N}]$ with $\widetilde p,\widetilde p_{\mathcal N}\ll\widehat p$:
- $\widehat\theta$ — the large **latent backbone** (transferred from pretraining, *small* learning rate $\widehat\eta$).
- $\widetilde\theta$ — **embedding/recovery** (patch embed + patch recovery). If the downstream PDE $\lambda\notin\widehat\Lambda$ (unseen physics), these are **re-initialized from scratch** and trained at a *large* rate $\widetilde\eta\gg\widehat\eta$; if $\lambda\in\widehat\Lambda$, transferred but still high-LR.
- $\widetilde\theta^{\mathcal N}$ — **time-embedding / layer-norm gains** $(\alpha,\beta)$ — always transferred, high LR $\widetilde\eta^{\mathcal N}$.

This places the *adaptation burden on the tokenizer (embedding/recovery) and time conditioning*, while the physics representation in the latent backbone is largely preserved — a concrete instance of the [[transfer-learning-fine-tuning]] frozen-backbone idea. The **frozen-latent ablation** below makes this striking.

---

## Pretraining Corpus (PDEgym)

6 operators on $[0,1]^2\times[0,1]$, chosen to span complementary fluid phenomena:

| Operator | Equation | Phenomenon emphasized |
|---|---|---|
| CE-RP | Compressible Euler | 4-quadrant Riemann shocks (axis-aligned interfaces) |
| CE-CRP | Compressible Euler | Curved Riemann interfaces → multi-scale vortices |
| CE-KH | Compressible Euler | Kelvin–Helmholtz large vortex roll-ups |
| CE-Gauss | Compressible Euler | Gaussian-bump driven gas dynamics |
| NS-Sines | Incompressible NS | Global turbulent features (sinusoidal forcing/IC) |
| NS-Gauss | Incompressible NS | Localized turbulent mixing |

9,640 Euler + 19,640 NS = 77,840 trajectories, 11 snapshots each. Notably narrow in PDE *type* (only 2 PDE families, both convection-dominated, all periodic BC, 2D Cartesian) — which is what makes the downstream generalization surprising.

---

## Downstream Suite: 15 Out-of-Distribution Tasks

Every task is OOD vs. pretraining. Taxonomy: linear (4) / nonlinear (11); time-dependent (12) / time-independent (3); elliptic (2), parabolic (1), hyperbolic (4), mixed (8). Operator *types* also vary: IC→solution (8), coefficient/parameter→solution (5), forcing→solution (2), domain-shape→solution (1); many non-periodic BC; one non-Cartesian domain.

- **6 "same-PDE, new-distribution"**: NS-PwC, NS-BB, NS-SL, NS-SVS, CE-RPUI, CE-RM.
- **3 "new process added to NS/Euler"**: NS-Tracer-PwC (passive tracer), FNS-KF (Kolmogorov forcing), GCE-RT (gravity → Rayleigh–Taylor).
- **3 "entirely new time-dependent PDE"**: Wave-Gauss, Wave-Layer (wave equation), ACE (Allen–Cahn reaction–diffusion).
- **3 "time-independent PDE"**: SE-AF (steady Euler over airfoil, non-Cartesian via masking), Poisson-Gauss, Helmholtz.

---

## Results

**Evaluation:** relative $L^1$ error at final time; scaling curves error-vs-samples. Two metrics:
$$\mathbf{AG}_S=\frac{\mathcal E_S(\text{FNO})}{\mathcal E_S(\text{model})}\ \text{(accuracy gain, limited-compute)},\qquad \mathbf{EG}_S=\frac{S}{s}\ \text{where }\mathcal E_s(\text{model})=\mathcal E_S(\text{FNO})\ \text{(efficiency gain, limited-data)}.$$

- **Universal win:** Poseidon beats FNO on **all 15** tasks; best model on **14/15** (CNO marginally wins one time-independent task).
- **Sample efficiency:** median ~20 samples to match FNO@1024; 4 tasks need only **3** samples; 13/15 tasks need an order of magnitude fewer.
- **Accuracy:** at fixed budget, mean accuracy gain ≈ **one order of magnitude** over FNO (10%–25× range).
- **Unseen physics:** best model on 8 of the 9 never-seen-PDE tasks, including *all* unseen time-dependent PDEs.
- **Architecture matters, not just size:** Poseidon ≫ CNO-FM (same data, multiscale-CNN backbone) on 14/15; even **Poseidon-T (21M)** beats the much larger CNO-FM and MPP-B — pretraining-data + transformer-multiscale backbone, not parameter count, drives performance.
- **Scaling laws:** error follows $\mathcal E(M)\approx C\,M^{-\alpha}$ in #samples $M$; Poseidon-L has $\alpha\gtrsim 0.5$ on nearly all tasks. The paper stresses the **prefactor $C$** dominates in the realistic pre-asymptotic (few-sample) regime, not $\alpha$.
- **Biphasic scaling on far-OOD tasks:** Poisson-Gauss shows a *two-phase* law $\mathcal E(M)\approx C^w M^{-\alpha^w}$ for $M\le M^{pt}$ then $C^\ell M^{-\alpha^\ell}$ for $M\ge M^{pt}$, with $\alpha^w<\alpha^\ell$ (e.g. Poseidon-B: $\alpha^w{=}0.23\to\alpha^\ell{=}0.99$, transition $M^{pt}{=}32$). Interpreted as a **warmup phase** (slowly relating the new operator to pretraining knowledge) → **fast-learning phase**.
- **vs. DPOT:** on 7 representative tasks, mean AG Poseidon-L 8.14, Poseidon-B 6.5 vs. DPOT-L/M ≈ 4.4. DPOT **does not scale with model size** downstream and relies on its base operator's capacity (from-scratch DPOT-L AG 3.53 is only 25% below finetuned), whereas Poseidon-L is ~5× more accurate than its own from-scratch scOT — i.e. **Poseidon actually harnesses pretrained latent representations; DPOT largely does not.**

---

## Case Studies — *How* Poseidon Generalizes (the most important part)

The paper's deepest contribution is mechanistic insight into *why* a model pretrained on only fluids transfers to unrelated PDEs.

### 1. CE-RPUI (compositional reuse of pretraining operators)
CE-RPUI = Riemann problem with sinusoidally-perturbed interfaces → coexisting shocks **and** small-scale vortex roll-ups. Zero-shot is poor (confirming OOD); 1 sample recovers shock locations; **4 samples** capture vortex roll-ups; 32–128 refine. The hypothesis, supported by the less-diverse-pretraining ablation: Poseidon **composes** distinct skills learned from *different* pretraining operators — shock propagation from CE-RP, large vortices from CE-KH, *small-scale* vortices + curved shocks from **CE-CRP**. Dropping CE-CRP from pretraining specifically degrades small-scale vortex resolution. → **Diversity of pretraining operators (not raw size) supplies separable, recombinable physical primitives.**

### 2. ACE (reaction–diffusion from convection-only pretraining)
Allen–Cahn is parabolic reaction–diffusion — physics absent from the convection-dominated corpus. Yet 1 sample already yields front propagation (borrowed from shock propagation) + diffusion of localized features. **Frozen-latent test:** finetuning *only* embedding/recovery (<0.5% of params, latent backbone frozen) still learns Allen–Cahn qualitatively from 1 trajectory; at 32 trajectories error 0.031 < FNO@128 (0.037), though above full-finetune (0.014). → Fluid-pretrained latents secretly encode reaction–diffusion-usable features.

### 3. Poisson-Gauss (elliptic, steady, Dirichlet — maximally OOD)
Maps a Gaussian-superposition source to the smoothed elliptic solution; time-independent, elliptic, Dirichlet — every axis differs from pretraining. Behavior is revealing: with **1 sample the model outputs (approximately) the identity** — it does *not* hallucinate fluid dynamics, nor does it discard pretraining; by 16 samples it learns "spread/diffuse the input," by 128 "spread + smooth," by 512 it's refined. This staged learning is the empirical face of the **biphasic scaling law** (warmup → fast learning). Frozen-latent again works (512-sample error 0.11 < FNO 0.282). → Even *steady elliptic* operators are reachable from a fluids-only latent space.

**Synthesis:** Poseidon learns *transferable physical primitives* (transport/front propagation, multi-scale vortex formation, diffusion/smoothing) during pretraining and **recombines** them per downstream task; finetuning re-routes the tokenizer/time-conditioning onto these primitives rather than relearning physics. This is the paper's evidence that PDE foundation models are feasible.

---

## Relevance to the Physics Foundation Model Goal

Poseidon is arguably the **strongest existing evidence that a PFM is feasible**, and it stakes out a distinct design pole from the wiki's other foundation models:

- **Operator-learning framing** (full trajectory from IC) vs. **autoregressive next-step** ([[walrus-paper]], [[gphyt-physics-foundation-model]], MPP, DPOT). This directly bears on [[pfm-architecture-approaches]] and [[pfm-interface-design]] (how lead time is specified): Poseidon's continuous **lead-time token** is a clean, principled answer to "how does a PFM handle timescales" — one model, any $t$.
- **Multiscale Swin U-Net backbone** is a concrete, validated alternative to flat ViT for physics — it builds the multi-resolution structure of PDE solutions (boundary layers, shocks, vortex cascades) into the architecture. Strong complement to [[multiscale-hierarchical-gnn]] (continuous-grid analog of the FMM V-cycle).
- **all2all / semi-group training** is a *general, architecture-agnostic* data-amplification principle for any time-dependent-PDE PFM, including the wiki's [[arch-autoregressive-transformer]] and [[arch-diffusion-backbone]] specs.
- **Diversity > size for pretraining** corroborates the diversity-first lesson from [[walrus-paper]] with a *causal* mechanism (compositional primitive reuse), upgrading it from empirical to mechanistic.
- **Frozen-latent transfer** validates the memory/representation-reuse thesis of [[memory-augmented-physics-models]] and [[transfer-learning-fine-tuning]]: the pretrained latent is a rich physical "library," and adaptation is mostly an embedding problem — which is exactly the token-representation question explored in [[00-token-representation-overview]].

**[AI Inference]:** Poseidon's success with a *tiny, low-diversity* pretraining corpus suggests the binding constraint for PFMs may be **representational composability**, not data scale. If 6 fluid operators already yield primitives reusable for waves, reaction–diffusion, and elliptic smoothing, then a deliberately *curated basis* of operators spanning the canonical PDE behaviors (transport, diffusion, dispersion, reaction, elliptic constraint) might pretrain a far more general PFM than brute-force scraping. This reframes pretraining-set design as choosing a **spanning set of physical primitives**.

**[AI Inference]:** The continuous lead-time LayerNorm is a special case of **conditioning the normalization on a physical control variable**. The same mechanism could carry *dimensionless numbers* (Reynolds, Mach, Péclet) or a *PDE-type code* as additional affine modulations $\alpha(t,\mathrm{Re},\ldots)$ — turning scOT into a genuinely *parameter-conditioned* operator and addressing the scale-awareness gap noted in [[multimodal-tokenization]] and [[structure-preserving-tokens]].

**[AI Inference]:** all2all training is structurally identical to **trajectory-level self-distillation under a group action** (the time-translation semi-group). For PDEs with *other* exploitable symmetries (Galilean, scaling, rotational), an analogous "symmetry2symmetry" augmentation could multiply data further — connecting Poseidon's training trick to the equivariance program of [[equivariant-gnns]] and [[action-based-noether-enforcement]].

---

## Limitations

- Pretrained only on 2 PDE families (Euler, NS), all convection-dominated, periodic BC, 2D Cartesian; authors expect large gains from adding time-independent/elliptic PDEs, more timescales, and non-Cartesian geometry to pretraining.
- all2all cost is quadratic in snapshot count.
- Demonstrated in 2D; 3D and irregular geometry largely future work (SE-AF handled via masking only).
- Still a surrogate — downstream extensions to UQ, inverse problems, PDE-constrained optimization are noted as straightforward but not done here.

---

## Links

- [[physics-foundation-models]] — the central goal; Poseidon is a leading feasibility proof
- [[pfm-architecture-approaches]] — operator-learning vs. autoregressive paradigms
- [[pfm-interface-design]] — lead-time conditioning answers the timescale-specification problem
- [[neural-operators]] — Poseidon vs. FNO/DeepONet/CNO; operator-learning framing
- [[transformer-architectures]] — SwinV2 windowed/multiscale attention as physics backbone
- [[hierarchical-windowed-tokens]] — scOT's multiscale token representation (detail page)
- [[patch-embedding-tokens]] — the base linear-patch tokenizer Poseidon uses
- [[transfer-learning-fine-tuning]] — split-LR finetuning; frozen-latent transfer
- [[memory-augmented-physics-models]] — pretrained latent as reusable physical library
- [[autoregressive-rollout-stability]] — optional rollout inference mode
- [[walrus-paper]] — sister large-scale PFM (autoregressive; diversity-first)
- [[gphyt-physics-foundation-model]] — neural-differentiator PFM; in-context generalization
- [[partial-differential-equations]] — the equation families pretrained/evaluated
- [[navier-stokes-equations]] — core pretraining physics
