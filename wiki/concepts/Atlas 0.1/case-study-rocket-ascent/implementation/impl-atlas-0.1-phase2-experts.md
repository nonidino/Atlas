# Phase 2 — Experts

**Type:** Implementation spec — agent task (folder: Atlas 0.1 / Atlas 0.1 implementation)
**Phase of:** [[00-atlas-0.1-implementation-plan]]. **Depends on:** [[impl-atlas-0.1-phase0-scope-and-data]] (corpus), [[impl-atlas-0.1-phase1-scaffold]] (tokenizer/decoder/graph). **Unblocks:** [[impl-atlas-0.1-phase3-integration]].
**Design pages:** [[expert-library-atlas-0.1]] (authoritative), [[training-and-bootstrap-atlas-0.1]], [[incremental-transfer-roadmap]].
**Milestone:** M4 — each of the three learned experts passes its solo accuracy gate.
**Resource plan:** [[impl-atlas-0.1-compute-and-training-budget]] — per-expert GPU-hour estimates, and a **build order that differs from §2's numbering**: `reacting_flow` first (no donor dependency, so not blocked on §3.4's unresolved license question), then the from-scratch baselines that stage 2a's gate requires anyway, then the Poseidon bootstrap as an upgrade rather than a blocker.

---

# 1. Intuition

An expert is a specialist that handles **one governing-equation family**, not one physical quantity. This distinction is the central design decision of Atlas and the easiest thing to get wrong, so it is worth restating concretely.

The declared edges carry type labels — heat, pressure, stress, fluid, conservation. The tempting design is one expert per label. It is wrong. Consider two edges:

- At $a\!-\!b$ (reaction → chamber), the labels are *heat* and *pressure*. But heat and pressure there are **two outputs of one solve**: the compressible energy and momentum equations, coupled through the same state vector. Splitting them into a "heat expert" and a "pressure expert" would force two networks to agree after the fact about something a single network gets right by construction.
- At $b\!-\!c$ (chamber wall → airframe), the labels are *heat* and *stress*. Here they genuinely **are** different equations — parabolic conduction and quasi-static elasticity — coupled only weakly, through thermal expansion. Engineering practice solves these separately and passes a coupling term. Two experts is correct.

So: **cut along governing equations; check the declared type labels against that cut.** Atlas 0.1 ends up with four experts, one of which is not a network at all.

The second idea in this phase is **solo validation before composition**. Each learned expert is trained and gated against its own classical baseline in isolation. Only when all three pass does Phase 3 wire them together. This ordering is not fastidiousness — it is the only way a Phase 4 failure will be diagnosable. Three unvalidated experts composed through two constraint layers produce a failure mode with at least five plausible causes.

The third idea is **bootstrap where a donor exists, train from scratch where none does.** Per [[incremental-transfer-roadmap]], only Poseidon and Walrus have confirmed public checkpoints; everything else in the wiki's catalogue is a structural pattern to reimplement, not weights to load. That fact — not architectural preference — determines which expert gets which treatment.

---

# 2. Theory

## 2.1 The common expert interface

Every expert is a map on tokens, conditioned, at a fixed timestep:

$$\mathcal E_\theta:\ \left(Z\in\mathbb R^{B\times P\times d},\ \mathbf z_{\text{cond}}\in\mathbb R^{B\times n_c},\ \Delta t\right)\ \longmapsto\ Z'\in\mathbb R^{B\times P\times d}$$

Experts **do not** see raw fields and **do not** produce raw fields — the tokenizer and decoder ([[impl-atlas-0.1-phase1-scaffold]]) own that boundary. This is what allows a donor model to be swapped in behind an adapter without the rest of the pipeline knowing.

$\Delta t$ enters as a conditioning input (log-scaled), never as a fixed architectural constant, following [[pfm-interface-design]]'s contract — the same expert must serve a $10^{-3}$ s combustion step and, later, other rates.

> ✅ **This clause is what licenses all2all training** — it makes the expert a lead-time-conditioned *solution operator* rather than a fixed-step map, so every ordered pair of corpus snapshots is valid supervision. Specified as **stage A** in §3.2; the augmentations that do *not* apply to Atlas are analysed in [[data-augmentation-physics-surrogates]].

## 2.2 Expert 1 — `reacting_flow` (agents `a`, `b`, `e`)

**Governs:** compressible reacting flow with a progress variable, and confined duct flow through a choking throat.

$$\partial_t U+\nabla\!\cdot\!\mathcal F(U)=S_{\text{rxn}}(U),\qquad \partial_t(\rho Y)+\nabla\!\cdot\!(\rho\mathbf u Y)=\rho\omega$$

**Why from scratch:** neither confirmed donor was pretrained on confined reacting duct flow. There is no weight-transfer path.

> ⚠️ **This asks about donor *weights* and never about donor *data* (audited 2026-08-09, [[physics-simulation-datasets]] §3.2).** **BLASTNet 2.0** — 2.2 TB, 744 full-domain samples from 34 DNS of reacting and non-reacting compressible turbulence — exists and is public. **[AI Inference]:** pretraining this expert on 2D slices of it and then fine-tuning on the Atlas corpus should beat random init, at the cost of one download and a normalization adapter. The domain gap is real (3D DNS with detailed chemistry vs. Atlas's 2D one-step progress variable), so what transfers is a reacting-structure prior, not chemistry. Treat "from scratch" as **not yet re-evaluated**, not as settled. The one thing worth borrowing structurally is GP$_{\text{hy}}$T's **derivative prediction plus integrator** pattern ([[gphyt-physics-foundation-model]], a structural pattern only — its checkpoint is unconfirmed): the network predicts $\partial_t U$ and an explicit integrator advances the state, which yields smoother targets than direct next-state regression.

**Architecture:** $L{=}6$ typed-attention layers over the agent's own tokens (intra-agent), $d{=}256$, 8 heads, with FiLM conditioning on $(\mathrm{Ma},\mathrm{Re},\gamma,\Delta t,\ p/p_c)$. Two output heads:

$$\hat{\dot U}=\mathcal H_{\text{flux}}(Z),\qquad \hat{\dot Y}=\mathcal H_{\text{rxn}}(Z)$$

**The transonic problem.** The throat carries a genuine change of character: information cannot propagate upstream once $M>1$. A globally-attending expert can violate this and produce acausal upstream influence from the diverging section. Mitigation for 0.1: an **attention mask conditioned on the local Mach field** — for query tokens with $M>1$, mask attention to tokens downstream in $z$. This is a mild, cheap enforcement of the hyperbolic domain-of-dependence and is the one place Atlas 0.1 encodes physics architecturally rather than by loss.

## 2.3 Expert 2 — `external_flow` (agents `d`, `f`, `g`)

**Governs:** external compressible aerodynamics and free-jet plume flow — the same Navier–Stokes family as Expert 1 but with free rather than confined boundaries.

**Why bootstrapped from Poseidon:** Poseidon ([[poseidon-pde-foundation-model]]) is a 21M–629M-parameter multiscale SwinV2 transformer (scOT) pretrained on a 6-operator compressible/incompressible fluid set, with confirmed public weights (`camlab-ethz/poseidon`, PDEgym) and demonstrated transfer to 15 out-of-distribution downstream PDEs. This is the closest available match to external compressible flow.

**Adapter design.** Poseidon is grid-tied (patch tokens on a regular lattice); Atlas is graph-tokenized. The adapter must map between them:

$$Z_{\text{atlas}}\in\mathbb R^{P\times d}\ \xrightarrow{\ \text{scatter to lattice}\ }\ \mathbb R^{H\times W\times d_{\text{pos}}}\ \xrightarrow{\ \text{Poseidon}\ }\ \cdots\ \xrightarrow{\ \text{gather}\ }\ \mathbb R^{P\times d}$$

Because agents `d`, `f`, `g` are built on structured grids ([[impl-atlas-0.1-phase1-scaffold]] §3.1), the scatter/gather is an exact reshape, not an interpolation — this is precisely why keeping those agents on structured grids was worth it. Agent `d`'s wall-stretched grid is the exception: the reshape is exact in index space, and the stretching is communicated to Poseidon through the positional channel.

**Open question inherited from [[expert-library-atlas-0.1]]:** internal and external compressible flow are the same governing family and *architecturally* might be one expert with a confined-vs-free flag. Donor availability forces them apart for 0.1. Revisit after both have trained weights — do not attempt the merge now.

**Bootstrap staging** (per [[incremental-transfer-roadmap]]):
- **2a:** frozen Poseidon, train adapters only. Gate: beats a from-scratch same-parameter-count baseline.
- **2b:** LoRA on Poseidon attention/MLP blocks, rank $r{=}16$, $\alpha{=}32$. Gate: beats 2a.
- If 2b does not beat 2a, the adapter interface is the problem, not the donor — do not proceed to full fine-tuning to paper over it.

## 2.4 Expert 3 — `thermostruct` (agent `c`)

**Governs:** transient conduction coupled to quasi-static plane-stress elasticity with thermal expansion. Parabolic + elliptic — no hyperbolic content, no shocks.

**Why bootstrapped from Poseidon:** its pretraining covers smooth elliptic/parabolic operators, which is the regime where operator learning is strongest ([[regime-moe-architecture]] finding 4). Same adapter machinery as Expert 2, different LoRA adapters.

**The elliptic subtlety.** Quasi-static elasticity is an *instantaneous global* solve: a load anywhere changes stress everywhere on the shell at once. A local message-passing expert with a finite receptive field structurally cannot represent this in one pass. Two mitigations, both cheap, both used:
1. The shell is thin and long — its token graph is nearly 1D (116 tokens), so 6 layers of attention already span it.
2. Full (non-windowed) attention within agent `c`, which is affordable at this token count and is the architecturally honest choice for an elliptic operator ([[regime-moe-architecture]] finding 1: elliptic ⇒ global).

Separate output heads for the parabolic and elliptic parts, reflecting that these are weakly-coupled distinct equations rather than one system:

$$\hat{\dot T}=\mathcal H_{\text{cond}}(Z),\qquad \left(\hat{\mathbf u},\hat{\boldsymbol\sigma}\right)=\mathcal H_{\text{elast}}(Z,\hat T)$$

Note $\mathcal H_{\text{elast}}$ predicts an **absolute** field, not a rate — it is a quasi-static solve, so there is no time derivative to predict. This is the one place the increment-prediction convention from Phase 1 is deliberately broken, and the decoder must be configured accordingly.

## 2.5 Expert 4 — `rigid_body` (bottleneck) — not learned

Closed-form 3-DOF Newtonian integration, imported verbatim from `solvers/trajectory.py` ([[impl-atlas-0.1-phase0-scope-and-data]] §3.3):

$$m\dot{\mathbf v}=\mathbf F_{\text{thrust}}+\mathbf F_{\text{aero}}+m\mathbf g(h),\qquad I(m)\dot\omega=\tau,\qquad \dot m=-\dot m_p$$

**There is nothing to train.** This sits at level 5 of the physics-encoding spectrum (hard architectural constraint) because a closed form exists and is exact. Spending model capacity to approximate Newton's second law would be strictly worse in accuracy, speed, and interpretability.

Its role in the graph: it reads the bottleneck vector $\zeta$ (which carries the pooled aerodynamic and thrust loads), integrates, and writes back the updated vehicle state, which is broadcast to all tokens on unpool. Gravity enters here and only here ([[global-fields-and-topology-atlas-0.1]]).

**Differentiability:** the RK4 step must be written in `torch` (not `numpy`) so gradients flow through it during Phase 3 joint fine-tuning — the learned experts' loads feed it, and their gradients must be able to see the trajectory error.

---

# 3. Implementation

## 3.1 Module layout

```python
# model/experts/base.py
class Expert(nn.Module, ABC):
    @abstractmethod
    def forward(self, Z: Tensor, cond: Tensor, dt: float) -> Tensor: ...

class IdentityExpert(Expert):        # Phase 1 placeholder; keep for ablation
    def forward(self, Z, cond, dt): return Z
```

```python
# model/experts/reacting_flow.py
class ReactingFlowExpert(Expert):
    def __init__(self, cfg): ...     # 6 typed-attn layers, 2 heads out
    def forward(self, Z, cond, dt, mach_field=None): ...
```

```python
# model/experts/external_flow.py
class PoseidonAdapter(nn.Module):
    """atlas tokens <-> poseidon lattice. Exact reshape for structured agents."""
class ExternalFlowExpert(Expert):
    def __init__(self, cfg, ckpt: str, lora_rank: int | None): ...
```

```python
# model/experts/rigid_body.py
class RigidBodyExpert(nn.Module):     # NOT an Expert subclass: different signature
    def forward(self, zeta, rigid_state, loads, dt) -> tuple[Tensor, RigidState]: ...
```

`RigidBodyExpert` deliberately does **not** implement the `Expert` interface — it operates on the bottleneck vector and a structured state, not on tokens. Forcing it into the token interface would be a false uniformity.

## 3.2 Per-expert training (`train/train_expert.py`)

Each learned expert trains **solo**: its agents' fields in, its agents' fields out, with interface conditions supplied as *ground-truth boundary data from the corpus* rather than from a neighbouring expert. This is teacher forcing at the interface, and it is what makes solo validation meaningful — the expert is being measured on its own physics, not on its neighbours' errors.

Training runs in **two stages**, and the split is load-bearing: stage A learns the *operator* using every snapshot pair the corpus contains, stage B learns the *fixed-rate march* the multi-rate stepper will actually execute.

### Stage A — all2all (semi-group) operator pretraining

Because §2.1 makes $\Delta t$ a conditioning input rather than an architectural constant, the expert is a **lead-time-conditioned solution operator**, not a fixed-step map. For an autonomous PDE that operator forms a one-parameter semi-group,

$$\mathcal S(\tau_2)\circ\mathcal S(\tau_1)=\mathcal S(\tau_1+\tau_2),$$

so **every ordered pair of corpus snapshots is a valid training example**, not only consecutive ones ([[poseidon-pde-foundation-model]]'s all2all training; general treatment in [[data-augmentation-physics-surrogates]]). With $K$ snapshots per episode this yields

$$\binom{K}{2}=\frac{K(K-1)}{2}\ \text{pairs instead of}\ K-1$$

— at the corpus's $K=200$, **19,900 pairs per episode against 199, a $\sim\!100\times$ amplification of the most expensive artifact in the project** ([[impl-atlas-0.1-corpus-completion-plan]]).

**Sampling rule.** Draw $(i,j)$ with $i<j$ and $\tau=t_j-t_i$, but **sample $\log\tau$ uniformly, not $\tau$ and not the pair index.** Uniform sampling over pairs concentrates mass at large $\tau$ — there are ~200× more distant pairs than adjacent ones — which starves exactly the step size the expert must be best at. Cover $\tau\in[\Delta t_{\text{snap}},\,T_{\text{ep}}]$ with the design $\Delta t_{\text{model}}$ oversampled by a factor of ~4.

**Validity conditions, both of which hold here and must be re-checked if the corpus spec changes:**

1. **Lead-time conditioning** — satisfied by construction (§2.1).
2. **Autonomy.** The coupled system is non-autonomous in principle (the vehicle climbs, the freestream changes), but altitude, $\mathrm{Ma}$ and $\mathrm{Re}$ enter through FiLM, so the *extended* state is autonomous; and under route B′'s $T_{\text{ep}}=0.2$ s the vehicle moves ~0.4 m, making the residual drift negligible. **Route B′'s short horizon, chosen for cost, makes this assumption more nearly exact than the original 10 s episodes would have.** If a later corpus restores long episodes, re-examine this — do not assume it carries over.

**Applies to `thermostruct` with one exception.** Conduction is a genuine semi-group and all2all applies. The **elasticity head does not** — it is a quasi-static solve with no time derivative (§2.4), so $\hat{\boldsymbol\sigma}$ is a function of the *instantaneous* temperature field and $\tau$ is meaningless for it. Train $\mathcal H_{\text{elast}}$ on single snapshots throughout, and apply all2all only to $\mathcal H_{\text{cond}}$.

### Stage B — fixed-rate push-forward

After stage A plateaus, fine-tune at the **fixed design $\Delta t_{\text{model}}$** for that expert ($10^{-3}$ s for `reacting_flow`, $5\times10^{-3}$ s for `external_flow`, $5\times10^{-2}$ s for `thermostruct`), with the push-forward curriculum below. This stage is what the multi-rate stepper in [[impl-atlas-0.1-phase3-integration]] §3.3 actually executes, and stage A does not by itself produce a model that marches stably at one rate.

**Budget split:** ~70% of the step budget on stage A, ~30% on stage B. **[AI Inference]** — the ratio is a starting point, not a measured optimum; the signal to shift it is stage B's one-step loss failing to improve on stage A's value at the same $\tau$.

**Loss.** Stage B uses the full form below. Stage A uses the same expression with $K=1$ (no rollout term), $\Delta t$ replaced by the sampled $\tau$, and $\mathcal L_{\text{iface}}$ evaluated at $t_j$.

$$\mathcal L=\underbrace{\left\|\hat x_{t+\Delta t}-x_{t+\Delta t}\right\|_{2}^{2}}_{\text{one-step}}+\lambda_{\text{roll}}\sum_{k=2}^{K}\left\|\hat x_{t+k\Delta t}-x_{t+k\Delta t}\right\|_2^2+\lambda_{\text{spec}}\mathcal L_{\text{spec}}+\lambda_{\text{iface}}\mathcal L_{\text{iface}}$$

- **Push-forward rollout term** ($K$ ramped $1\!\to\!2\!\to\!4\!\to\!8$ as the one-step loss plateaus): the standard mitigation for error accumulation ([[autoregressive-rollout-stability]]). Substeps are fed forward **detached** — no backprop through the rollout chain. Note that autograd must still retain every substep's activations for the accumulated loss, so memory grows with $K$; use gradient checkpointing on the expert's forward at $K>1$. (This exact failure — an out-of-memory that appears only *after* the curriculum reaches a large horizon — is documented in the parallel track's [[implementation-log]], 2026-07-19.)
- **Spectral term** $\mathcal L_{\text{spec}}$: $\ell_2$ on $|\hat U(k)|$ with $1/(|k|+1)$ weighting, to stop the deterministic regression from over-smoothing the plume's fine structure.
- **Interface term** $\mathcal L_{\text{iface}}$: error on the predicted flux *at* the interface tokens, supervised by the corpus's `/iface` arrays. Weighted higher than interior error (default $3\times$), because interface accuracy is what Phase 3 composition depends on and interior accuracy is not. **Under stage A, respect `t_exch`** ([[impl-atlas-0.1-corpus-completion-plan]] D5): the slow edges ($b\!-\!c$, $c\!-\!d$ at $5\times10^{-2}$ s) are held constant for up to 50 snapshots, so a pair landing between exchanges supervises against a stale flux. Mask $\mathcal L_{\text{iface}}$ to pairs whose $t_j$ coincides with a real exchange on that edge, rather than training the model to predict a held constant as though it were live.
- **No conservation projection inside stage A.** The hard flux-matching projection ([[conservation-as-constraint-atlas-0.1]]) is a *per-step* operation that Phase 3 applies at the design rate. Applying it across a large-$\tau$ training pair imposes a single-step constraint on a jump the stepper will never take.

**Optimizer:** AdamW, lr $3\times10^{-4}$ (from-scratch) / $5\times10^{-5}$ (LoRA), cosine decay, 500-step warmup, grad clip 1.0, bf16 autocast. Stage B restarts the cosine schedule at $0.3\times$ the stage-A peak rate.

**Corpus sizing interacts with this.** All2all pairs from one episode are heavily correlated — 19,900 pairs are worth far less than 19,900 independent samples, and the number of distinct *physical regimes* is set by the sweep, not the pair count. So all2all does not reduce the episode count directly, but it does change the marginal value of the last episode: **run [[impl-atlas-0.1-corpus-completion-plan]]'s D6 sizing ablation with stage A already enabled**, or it will overstate how many episodes are needed.

## 3.2b Training inside a 5-hour session cap

*Added 2026-08-18, alongside the same cap applied to generation ([[impl-atlas-0.1-compute-and-training-budget]] §5).*

Phase 2 is where the GPU is genuinely the resource, so the cap bites differently here than it does on the corpus: it does not change *what* is trained, it changes what the harness must be able to do.

### The one hard requirement

**Every training run must be able to end at an arbitrary wall-clock instant and resume without loss.** Not "should" — a run that cannot survive its 5-hour boundary cannot be run at all under this constraint, and discovering that at hour 4 costs the whole session. Concretely, `train/train_expert.py` must, from the first commit:

- checkpoint model + optimizer + LR-schedule + dataloader epoch/step state on a **wall-clock trigger**, not only on an epoch boundary — an epoch here is long enough to straddle the cap;
- write to durable storage (object store, not the box's disk), because the box is what disappears;
- resume by *content*, the same discipline the corpus generator already uses: a checkpoint that does not match the current config fingerprint is a different run, and silently continuing from it mixes two experiments.

The corpus side already demonstrates the pattern (`data/corpus.py`: content-addressed ids, atomic staging, checksummed manifest). Reuse the shape rather than inventing a second one.

### What fits in a session

**[AI Inference]**, from §3.4 of the budget page (productive GPU-hours, before the ×3 for failures and sweeps):

| run | productive GPU-h | sessions at 5 h |
|---|---|---|
| `reacting_flow` from scratch, stage A + stage B | ~2 | **1** |
| `thermostruct` from scratch (baseline) | ~1 | **1** |
| `external_flow` from scratch (baseline) | ~3 | **1** |
| `thermostruct` + Poseidon (2a adapters → 2b LoRA) | ~10 | 2–3 |
| `external_flow` + Poseidon (2a → 2b) | ~30 | 6–8 |
| D6 sizing ablation (with/without pretraining) | ~4 | 1 |

**The three from-scratch experts each fit in one session, and they are the ones that gate progress.** §3.4's ordering — `reacting_flow` first, then the from-scratch baselines, then the donor bootstrap — was chosen to de-risk the Poseidon license question; under a session cap it earns its keep a second time, because it front-loads everything that finishes inside one sitting. The donor runs are the only multi-session work, and they are upgrades rather than blockers.

### What the small corpus changes about the schedule

Nothing about the order, one thing about stage A. With ~32 coupled episodes and the T2 sweeps, the all2all pair count is large (hundreds of thousands) while the count of distinct physical configurations is small. That is the regime where **the sampling rule matters more than the epoch count**: log-uniform lead times with the design $\Delta t$ oversampled ~4×, per §3.2, and early stopping on the *structural* gates rather than on a validation loss that a correlated pair distribution will happily drive down.

**[AI Inference]:** the practical consequence is that a session is more likely to end because the gate stopped improving than because the clock ran out — which is the right failure mode, and the reason to log gate values every checkpoint rather than only at the end.

## 3.3 Per-expert acceptance gates (M4)

All measured on the **held-out corner configs** from [[impl-atlas-0.1-phase0-scope-and-data]] §3.5, never on training configs.

| Expert | Gate |
|---|---|
| `reacting_flow` | one-step rel-$L^2$ < 0.05 on $(\rho,\rho u,E)$; **choked-throat check**: predicted $\dot m$ at $z{=}0.40$ varies < 2% as downstream $p$ is varied at fixed $p_c$ (i.e. the model has learned that a choked throat isolates the chamber from downstream conditions) |
| `external_flow` | one-step rel-$L^2$ < 0.05; 20-step rollout rel-$L^2$ < 0.15; plume shock-cell spacing within 10% of the solver's |
| `thermostruct` | one-step rel-$L^2$ < 0.03 on $T$; stress field rel-$L^2$ < 0.08; **zero-load check**: uniform temperature with free boundaries predicts $\|\boldsymbol\sigma\|\approx0$ |
| `rigid_body` | exact by construction — assert bit-agreement with `solvers/trajectory.py` |

The **choked-throat check** and the **zero-load check** are worth more than the $L^2$ numbers. They test whether the expert learned the *structure* of its regime or merely interpolated the training distribution, which is the distinction [[transfer-learning-fine-tuning]]'s diversity principle warns about. An expert that passes rel-$L^2$ but fails the choked-throat check has not learned compressible duct flow and will fail in Phase 4 in a way that looks mysterious.

## 3.4 Bootstrap procedure for donor experts

1. Download `camlab-ethz/poseidon` weights. **Verify the license permits this use before writing adapter code** — this is the first action item in [[incremental-transfer-roadmap]] and is not yet done.
2. Confirm input channel count and resolution assumptions match what the adapter will feed.
3. Freeze the whole donor; train adapters only (stage 2a); record the gate metric.
4. Add LoRA ($r{=}16$) to attention and MLP blocks; train (stage 2b); record.
5. Keep both checkpoints. If 2b regresses against 2a, the interface is wrong — fix it rather than escalating to full fine-tuning.

---

# 4. Pitfalls

- **One expert per edge-type label.** The single most likely misreading of this architecture. Re-read §1.
- **Composing before all three gates pass.** Makes Phase 4 undiagnosable.
- **Skipping the structural checks** (choked throat, zero load) because rel-$L^2$ looks good.
- **Writing `rigid_body` in numpy.** Blocks gradient flow in Phase 3.
- **Applying increment-prediction to the elasticity head.** It is a quasi-static solve; there is no rate to predict.
- **Assuming Poseidon weights will load cleanly.** Channel counts, normalization conventions, and resolution assumptions all differ; budget real time for the adapter.
- **Letting the push-forward horizon grow without gradient checkpointing.** Produces a delayed OOM that looks like a data problem.
- **Sampling all2all pairs uniformly.** Distant pairs outnumber adjacent ones ~200:1, so uniform sampling starves the design $\Delta t$ — the one step size that must work. Sample $\log\tau$ uniformly and oversample $\Delta t_{\text{model}}$.
- **Shipping stage A without stage B.** An operator that is accurate at every $\tau$ is not the same as one that marches stably at a fixed rate, and the multi-rate stepper only ever does the latter.
- **Applying all2all to the elasticity head.** It is a quasi-static solve; $\tau$ has no meaning for it. Same root error as applying increment-prediction to it.
- **Supervising $\mathcal L_{\text{iface}}$ on a stale exchange.** The slow edges hold their flux for up to 50 snapshots; without masking on `t_exch`, stage A trains the model to predict a held constant as a live signal.
- **Sizing the corpus before enabling stage A.** Overstates the episode count needed; see §3.2's closing note.

---

## See Also

- [[expert-library-atlas-0.1]] — the governing-equation-family argument in full
- [[training-and-bootstrap-atlas-0.1]] — per-expert bootstrap decisions and training order
- [[incremental-transfer-roadmap]] — donor availability; the 4-stage bootstrap
- [[impl-atlas-0.1-phase3-integration]] — where these experts get composed
- [[poseidon-pde-foundation-model]], [[gphyt-physics-foundation-model]] — donor/pattern sources
