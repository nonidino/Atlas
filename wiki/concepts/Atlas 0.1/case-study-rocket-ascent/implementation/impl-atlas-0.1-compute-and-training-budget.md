# Atlas 0.1 — Compute and Training Budget

**Type:** Implementation spec — cross-phase resource plan (folder: Atlas 0.1 / Atlas 0.1 implementation)
**Phase of:** [[00-atlas-0.1-implementation-plan]]. **Resolves:** open action item 7 (route past the M2 compute wall) with a recommendation. **Unblocks:** [[impl-atlas-0.1-phase2-experts]].
**Depends on:** measured numbers in [[atlas-0.1-implementation-log]] (2026-08-09 M2 entry, Phase 1 bench).
**Written:** 2026-08-09, in answer to "what is the plan of action from Phase 2, considering training time, on a machine that can neither train nor generate?"

> **Every number below marked _measured_ comes from [[atlas-0.1-implementation-log]] and was timed on real hardware. Every number marked [AI Inference] is an estimate from FLOP counting or published throughput, and should be re-measured before money is spent on it.**

---

# 1. Intuition

## 1.1 The headline, because it inverts the obvious assumption

The instinct on reaching Phase 2 is "I need a GPU to train the experts." That instinct is wrong for Atlas 0.1, and acting on it would waste both money and weeks:

$$\underbrace{\sim 2{,}400\ \text{CPU-core-hours}}_{\text{data generation}}\qquad\text{vs.}\qquad \underbrace{\sim 160\ \text{GPU-hours}}_{\text{all training, Phases 2--4}}$$

**Atlas 0.1 is a data-generation problem wearing a machine-learning problem's clothes.** The model is 22.5 M parameters and the corpus is at most ~29,000 samples — that is a *small* training job by 2026 standards, comfortably done on one consumer GPU in a few days. What is expensive is running a CFL-limited finite-volume solver 327 times, and **a GPU does not accelerate that at all** unless the solver is rewritten to run on it, which is a separate multi-week project.

So the answer to "should I rent a vast.ai GPU?" is: *eventually yes, and it will be cheap and easy — but it is not what is blocking you, and renting one now would leave it idle.* The first machine to rent is a **many-core CPU box**, and vast.ai is a poor venue for that.

## 1.2 Why the corpus is expensive, restated so the levers are visible

From [[impl-atlas-0.1-phase0-scope-and-data]] §3.8: 820 h per episode, 30.6 years for the corpus, on one core — **measured**. The cause is a timescale ratio. The solver must resolve *acoustic* waves:

$$\Delta t_{\text{CFL}} \sim \frac{\Delta x}{|\mathbf u| + c} \approx 3\times10^{-7}\ \text{s (nozzle)}$$

while the physics of interest evolves at the *convective* rate, and the episode spans $T_{\text{ep}} = 10$ s of flight. The total substep count is

$$N_{\text{substeps}} \;=\; \frac{T_{\text{ep}}}{\Delta t_{\text{CFL}}} \;\approx\; \frac{10}{3\times10^{-7}} \;\approx\; 3.3\times10^{7}$$

and the cost is $N_{\text{substeps}}$ multiplied by a per-substep cost that scales with cell count. **Note what is *not* in that expression: $\Delta t_{\text{macro}}$ and $\Delta t_{\text{model}}$.** The surrogate's step size is irrelevant to the cost of teaching it. That single observation is what makes §2.2's recommendation possible.

There are therefore exactly four levers, and only four:

| Lever | Mechanism | Factor available | What it costs the design |
|---|---|---|---|
| **L1** Shorten $T_{\text{ep}}$ | fewer substeps, linearly | up to $50\times$ | long-horizon trajectory evolution |
| **L2** Coarsen the grid by $r$ | $r^2$ fewer cells **and** $r\times$ larger $\Delta t_{\text{CFL}}$ ⇒ $r^3$ | $8\times$ at $r{=}2$ | patch size must halve to hold the token budget |
| **L3** Cheaper substep | GPU/vectorized port of the stencil | $30\text{--}100\times$ [AI Inference] | ~1–2 weeks of solver work; no physics change |
| **L4** Raise $\Delta t_{\text{CFL}}$ | implicit / low-Mach preconditioning (route C) | $10^{2}\text{--}10^{3}$ | a substantial solver project |

L1 and L2 are free in engineering time and expensive in fidelity. L3 and L4 are the reverse. **L3 is the only lever that costs the design nothing at all**, which is why §3.3 puts it on the critical path even though it delays nothing today.

## 1.3 The trap inside "just shorten the episodes"

Option B in [[impl-atlas-0.1-phase0-scope-and-data]] §3.8 says "10 s → ~10 ms." Taken literally — keeping 200 snapshots at $\Delta t_{\text{macro}}$ — that forces $\Delta t_{\text{macro}} = 5\times10^{-5}$ s, and then the whole two-timestep-family table has to shrink with it: gas $\Delta t_{\text{model}}$ would fall from $10^{-3}$ to $10^{-6}$ s. At that point

$$\frac{\Delta t_{\text{model}}}{\Delta t_{\text{CFL}}} = \frac{10^{-6}}{3\times10^{-7}} \approx 3$$

and **the surrogate is no longer a surrogate** — it takes steps barely larger than the solver it replaces. The entire speedup claim, which is the reason the architecture exists, evaporates. Worse, multi-rate subcycling (50 gas steps per macro step) collapses to 1, deleting the feature [[impl-atlas-0.1-phase3-integration]] is built around.

This trap is not in the spec because the spec never separates two things that are conflated in the phrase "200 snapshots at $\Delta t_{\text{macro}}$": the **snapshot cadence** (a storage decision) and the **macro step** (an architecture decision). They do not have to be equal. Untangling them is the recommendation.

---

# 2. Theory — the recommended route past M2

## 2.1 The decision, stated

**Route B′ — short time-accurate episodes with sub-macro snapshot cadence.** Formally, replace the coupled generator's single "save every macro step" rule with two independent parameters:

$$T_{\text{ep}} = 0.2\ \text{s},\qquad \Delta t_{\text{macro}} = 5\times10^{-2}\ \text{s}\ \ (\text{unchanged}),\qquad \Delta t_{\text{snap}} = 10^{-3}\ \text{s}$$

giving **4 macro steps and 200 snapshots per episode**, at a cost of $T_{\text{ep}}/10 = 1/50$ of the specified corpus.

Everything the architecture depends on survives this change:

| Property | Spec | Under B′ |
|---|---|---|
| Snapshots per episode | 200 | **200** ✅ |
| $\Delta t_{\text{macro}}$ | $5\times10^{-2}$ s | **unchanged** ✅ |
| Gas $\Delta t_{\text{model}}$ | $10^{-3}$ s | **unchanged** ✅ |
| Subcycling ratio | 50 : 10 : 1 | **unchanged** ✅ |
| Demonstrated speedup $\Delta t_{\text{model}}/\Delta t_{\text{CFL}}$ | $3.3\times10^{3}$ | **unchanged** ✅ |
| HDF5 schema, `/iface` arrays | as §3.6 | **unchanged** ✅ |
| Macro steps of trajectory evolution | 200 | **4** ❌ |

The one loss is the last row, and it is deliberately chosen as the cheapest available loss, for a reason specific to Atlas: **the `rigid_body` expert is not learned** ([[impl-atlas-0.1-phase2-experts]] §2.5). It is closed-form Newtonian integration imported verbatim from `solvers/trajectory.py` and asserted bit-identical to it. No quantity of trajectory data trains it. Trajectory data serves only (i) the Phase 3 joint loss term $\lambda_{\text{traj}}\|\hat{\mathbf s}-\mathbf s\|^2$, which needs *some* motion but not 10 seconds of it, and (ii) the Phase 4 milestone checks.

**[AI Inference]:** over $T_{\text{ep}} = 0.2$ s a vehicle at ~2 g gains ~4 m/s and moves ~0.4 m — enough for the loads→motion→freestream loop to be *measurably* closed and for `recompute_conditioning` to be exercised with a nonzero altitude delta, but far too little for max-Q or the thrust-vs-altitude milestone. Those two Phase 4 checks must be re-scoped or deferred; see §4.

Contrast with the alternatives already tabled:

- **Option A (quasi-steady gas)** buys ~$100\times$ but converts Atlas into a steady *mapping*. Every transient — the choking transient, the interface response, the whole reason multi-rate subcycling exists — is deleted. Phase 4's question ("does composing experts through declared interfaces produce a stable coupled *rollout*?") becomes unaskable. **Reject.**
- **Option C (implicit/low-Mach)** is physically the right answer and stays the long-run target, but it is a solver research project competing with Phase 2 for the same weeks. **Defer, do not abandon** — it is the correct way to eventually get 10 s episodes back.
- **B′ + L3 (GPU-ported solver)** is the combination that restores $T_{\text{ep}} \approx 1\text{--}10$ s later without touching the model. This is why §3.3 schedules the port to run in the background of Phase 2.

## 2.2 What full fidelity would actually cost, so nobody re-litigates it

Measured corpus cost: $327 \times 820\ \text{h} = 268{,}140$ core-hours. **[AI Inference]** on cloud pricing at ~\$0.02–0.04 per core-hour (dedicated/spot CPU): **\$5,000–\$11,000 and, at 64 cores, 175 days of wall clock.** Even with a successful GPU port (L3 at $50\times$) it is 5,400 GPU-hours ≈ \$2,000–2,700 and 7 months on one GPU.

**Full-fidelity 10 s $\times$ 327 episodes is out of reach for a solo budget under every combination of levers short of L4.** Recording that here so the question is answered once.

---

# 3. Implementation — the plan of action

## 3.1 Step 0: three decisions that cost nothing and gate everything

Do these before renting anything. All three are desk work on the current machine.

1. **Adopt B′** (§2.1) — closes open action item 7. Concretely: add `dt_snap` to `EpisodeSpec` alongside the existing `dt_macro`/`n_macro`/`coarsen`, record it in `/meta`, and keep the `full_fidelity: false` flag honest (B′ *is* reduced fidelity — reduced in horizon, not in physics).
2. **Verify the Poseidon checkpoint's license and I/O contract** — open action item 1, and it blocks two of three learned experts ([[impl-atlas-0.1-phase2-experts]] §3.4). Pure reading: license terms, input channel count, resolution assumptions, normalization convention. Costs an afternoon; discovering the answer *after* renting a GPU costs a week.
3. **Decide the expert/message-passing interleave** — open action item 4. [[atlas-0.1-implementation-log]] measured that with experts running once after all MP layers, nothing propagates *within* an agent during message passing, and agent `a` is unreachable from `f` at any depth. This changes the expert interface, so it must be settled *before* the experts are written, not after.

## 3.2 Step 1: generate the corpus on rented CPU — **not** on vast.ai

Cost under B′, **[AI Inference]** scaled linearly from the measured 820 h:

$$\frac{820\ \text{h}}{50} = 16.4\ \text{h/episode}$$

| Corpus | Episodes | Core-hours | Wall clock @ 48 cores | Est. cost |
|---|---|---|---|---|
| **Pilot** (recommended first) | 24 train + 8 test | 525 | **11 h** | ~\$15 |
| **Working corpus** | 120 train + 24 test | 2,360 | **49 h** | ~\$60–100 |
| Full sweep at B′ | 300 train + 40 test | 5,360 | 112 h | ~\$150–220 |

Generation is **embarrassingly parallel across episodes** — one process per core, no communication — so wall clock is core-hours divided by cores, with no scaling loss. Run the pilot first and validate M2's four criteria on it before committing to the working corpus.

**Venue: vast.ai is the wrong tool for this leg.** Its inventory is priced and provisioned around GPUs; CPU-only offers are thin, the cores attached to a GPU listing are usually few and shared, and instances are interruptible with non-durable disk. Better fits, in order:

| Venue | Why | Rough rate |
|---|---|---|
| **Hetzner dedicated / CCX** | best €/core-hour anywhere; 48 dedicated vCPU class | ~€1.2–1.6/h |
| AWS/GCP spot, `c`-family | elastic, spot-priced; needs checkpoint-on-eviction | ~\$0.5–1.5/h for 32–64 vCPU |
| OVH / Scaleway dedicated | mid-price, durable disk | similar to Hetzner |

Corpus size, **[AI Inference]**: ~70,600 cells $\times$ ~5 channels $\times$ 4 bytes ≈ **1.4 MB/snapshot**, so ~280 MB/episode raw, ~**40 GB** for 144 episodes, ~15–20 GB gzipped in HDF5. That fits on a rented box's local disk, but it must be pushed to durable object storage (Backblaze B2 ≈ \$6/TB/mo, or S3) before the box is released. **The corpus is the single most expensive artifact in the project — losing it costs 50 h of compute, losing the GPU box costs an hour of retraining.**

## 3.3 Step 2 (parallel, unattended): port `compressible2d` to the GPU

While the CPU box grinds, the highest-leverage engineering available is lever L3. The solver is six structured blocks running an identical stencil (MUSCL reconstruct → HLLC flux → viscous flux → SSP-RK2), which is close to the ideal shape for a GPU array port. **[AI Inference]:** a CuPy port is largely mechanical — `numpy` → `cupy` with the reductions in `max_stable_dt` and the sparse FEM solve handled specially — and a JAX port additionally buys `jit` fusion and `vmap` over the sweep, at the price of static shapes. Expect $30\text{--}100\times$ against one numpy core, and **budget 1–2 weeks including revalidation.**

**Every M1 oracle must be re-run on the ported solver and must reproduce the recorded numbers**, not merely pass. The 12-oracle suite already exists precisely so a change like this is gradeable; a port that passes the oracles but shifts the isentropic-vortex order from 1.86 to 1.6 has changed the physics quietly.

> ⚠️ **Corpus sizing revised down, 2026-08-09.** [[physics-simulation-datasets]] audits Phase 0's "no dataset" claim and finds it holds only for the *interface* layer. Public data (PDEgym via Poseidon; **BLASTNet 2.0** for reacting flow) covers the **interior dynamics** of both flow experts, so generated episodes should be budgeted for **interface response**, not for teaching governing families from nothing. **[AI Inference]:** 60–90 coupled episodes rather than 144 — but validate with a with/without-pretraining ablation on one expert before cutting, and add the cheap uncoupled **single-agent** runs (tier T2) that the choked-throat gate needs and that no public dataset can supply.

## 3.3b If the machine on hand is an A100 — what it does and does not buy

Written 2026-08-18, in answer to "how long does the data generation take on an A100?"

**The literal answer to the literal question is: it does not run on an A100 at all.** `solvers/compressible2d.py` is single-threaded numpy. Handing it an A100 changes nothing — the 30.6-year figure and the B′-scaled 16.4 h/episode are both *CPU-core* numbers, and the GPU sits idle. An A100 becomes relevant only *after* lever L3 (§3.3), the CuPy/JAX port, which is 1–2 weeks of work plus a full re-run of the twelve M1 oracles.

### The arithmetic the port would inherit

Derived from the log's measured per-agent table at $T_{\text{ep}} = 0.2$ s (route B′):

| agent | substeps / episode | cells | cell-updates / episode |
|---|---|---|---|
| `a` | 208 550 | 3 840 | $8.0\times10^{8}$ |
| `b` | 595 238 | 8 960 | $5.3\times10^{9}$ |
| `e` | 641 026 | 11 520 | $7.4\times10^{9}$ |
| `d` | 99 010 | 17 664 | $1.8\times10^{9}$ |
| `f` | 15 748 | 12 160 | $1.9\times10^{8}$ |
| `g` | 16 260 | 14 592 | $2.4\times10^{8}$ |
| **total** | **$1.58\times10^{6}$** | 68 736 | **$1.6\times10^{10}$** |

Implied CPU throughput: $1.6\times10^{10} / 59\,000\ \text{s} \approx 2.7\times10^{5}$ cell-updates/s/core — about 4 µs per cell-update, which is what numpy costs when the arrays are small and every stage allocates temporaries.

### Why the A100 will be latency-bound, not FLOP-bound

**[AI Inference].** The whole 7-agent state is 68 736 cells. An A100 wants millions of cells in flight; at this size the kernels finish faster than they launch. A MUSCL→HLLC→viscous→SSP-RK2 substep is ~20 kernels, so with ~8 µs of launch overhead each the floor is ~160 µs *per substep regardless of grid size*:

$$1.58\times10^{6}\ \text{substeps} \times 160\ \mu\text{s} \approx 250\ \text{s/episode}$$

Add host synchronization at the coupling exchanges and the per-episode setup and the honest band is **5–15 minutes per episode**, i.e. **$65$–$200\times$ one CPU core** — consistent with §1.2's L3 estimate of 30–100× rather than beating it. The throughput ceiling ($\sim\!10^{9}$ cell-updates/s ⇒ 16 s/episode) is *not* reachable without batching many episodes into one kernel launch (`vmap`/`jit` over the sweep, static shapes), which would amortize the launch cost and land near **30–60 s/episode**. That is a further ~10× for meaningfully more engineering; not recommended before the pilot proves the corpus.

### The whole M2 corpus on one A100, ported

| Leg | Episodes | @ 10 min/ep (naive port) | @ 45 s/ep (batched) |
|---|---|---|---|
| T2 solo sweeps (§3.2) | ~400 short runs | **< 1 h** | < 1 h |
| T3.5 pilot | 32 | **~5 h** | ~25 min |
| T3.7 production | 60–90 | **10–15 h** | 1–1.5 h |
| T3.8 Phase-4 baselines | 24 | **~4 h** | ~20 min |
| **Total** | | **≈ 20–25 A100-hours (~1 day)** | ≈ 3 h |

Against the CPU route's ~2,000–2,500 core-hours ≈ 2 days on 48 rented cores at \$60–100. **So the A100 route is not faster in calendar time once the 1–2 weeks of porting and oracle revalidation are counted** — it is faster only if the port is wanted anyway, which it is, because it is the lever that eventually restores $T_{\text{ep}} \to 1\text{–}10$ s without touching the model (§1.2, L3).

### Three things that only surface once the GPU is an A100 specifically

1. **Precision makes the A100 the *right* GPU for the solver leg, and the wrong one for training.** The finite-volume solver is fp64; the A100 does fp64 at 9.7 TFLOPS while a consumer 4090 runs it at $1/64$ rate. §3.4's "rent a 4090, not an H100" guidance is about the *training* leg and does not transfer to solver work. If one machine must do both, the A100 is the defensible compromise — but it will be badly under-used by the 22.5 M-parameter model.
2. **Agent `c` does not port and does not need to.** Its conduction is backward Euler and its elasticity a sparse bordered direct solve factorized once per mesh — no CFL limit, ~200 solves for a full history, seconds on CPU. Leave it on the host; a GPU sparse factorization here is pure engineering cost for no gain.
3. **The port must reproduce the oracle numbers, not merely pass them** (§3.3). fp64→fp32 is the tempting A100 optimization and it is the one that would quietly move the isentropic-vortex order from 1.86.

### And the training legs, for completeness

§3.4/§3.5's estimates (~100 GPU-h Phase 2, ~20–40 GPU-h Phase 3) were priced against a 4090. **[AI Inference]:** an A100 does **not** improve them much — the model is 22.5 M parameters over 1 114 tokens and Phase 3 is a chain of 50 sequential small forward passes, so both legs are latency- and memory-bound, exactly the regime where a bigger GPU's FLOPs go unused. Expect the same wall clock within a factor of ~1.5.

## 3.4 Step 3: Phase 2 training — small, cheap, and correctly ordered

### Cost estimate

**[AI Inference]**, FLOP-counted at $\approx 6\times$ params per token for a forward+backward pass, against a working corpus of $144 \times 200 = 28{,}800$ samples, ~200 epochs, and a $\times3.5$ factor for the push-forward horizon ramp $K: 1\!\to\!2\!\to\!4\!\to\!8$:

| Expert | Tokens | Params in the hot path | Est. productive GPU-h |
|---|---|---|---|
| `reacting_flow` | 380 (`a`,`b`,`e`) | 4.7 M | **~2** |
| `thermostruct` | 116 (`c`) | 4.7 M + donor | ~1 scratch / ~10 with Poseidon |
| `external_flow` | 618 (`d`,`f`,`g`) | 4.7 M + Poseidon-B ≈ 158 M | ~3 scratch / ~30 with Poseidon |
| `rigid_body` | — | 0 | **0** — asserted, not trained |

Productive total ~35 GPU-hours; **multiply by ~3 for failed runs, hyperparameter sweeps, and the LoRA stage ⇒ ~100 GPU-hours for Phase 2.** The donor dominates: the two Poseidon-bootstrapped experts cost an order of magnitude more per step than the from-scratch one, entirely because 158 M donor parameters run in the forward pass even when frozen.

Sanity anchor from **measured** data: the Phase 1 scaffold benched 1,396 ms for forward+backward at batch 8 over all 1,114 tokens on one CPU core. **[AI Inference]:** a 4090 on this shape should land near 5–15 ms — the model is small enough to be latency- and memory-bound rather than FLOP-bound, so treat published TFLOP figures as an upper bound that will not be approached.

### Ordering — and it is not the order the spec lists

[[impl-atlas-0.1-phase2-experts]] presents the experts as 1/2/3. Build them in this order instead:

1. **`reacting_flow` (from scratch).** The only learned expert with **zero donor dependency**, so it is not blocked on the Poseidon license question. Start here regardless of how step 0.2 turns out. Its gate includes the **choked-throat check**, which is the single most informative signal in Phase 2 — it tests whether the expert learned the *structure* of compressible duct flow or merely interpolated.
2. **The from-scratch baselines for `thermostruct` and `external_flow`.** These are not optional extras: stage 2a's gate is literally "beats a from-scratch same-parameter-count baseline" (§2.3), so they must exist anyway. Building them second means **the Poseidon path is de-risked before it is attempted** — if the license or the channel-count contract fails, three trained experts already exist and Phase 3 proceeds without a donor.
3. **Poseidon 2a (frozen donor, adapters only), then 2b (LoRA $r{=}16$).** Keep both checkpoints. Per §3.4's own rule: if 2b regresses against 2a, the adapter interface is wrong — fix it, do not escalate to full fine-tuning.

This reordering converts the Poseidon bootstrap from a **blocker** into an **upgrade**, which is what it should have been.

### Venue: here vast.ai is a good fit

A single **RTX 4090 (24 GB)** is sufficient and appropriate. The largest thing that must fit is Poseidon-B (~158 M params) plus adapters plus activations for a $K{=}8$ push-forward with gradient checkpointing — comfortable in 24 GB. **Do not rent an H100.** It would cost 5–8× more per hour to run a job that cannot saturate it.

| Venue | Rate (4090, approx.) | Notes |
|---|---|---|
| **vast.ai** | \$0.30–0.50/h | cheapest; interruptible hosts, **non-durable disk**, driver/CUDA quality varies by host |
| **RunPod** | \$0.40–0.70/h | more reliable, persistent volumes, better tooling |
| **Modal** | serverless, per-second | excellent fit for bursty per-expert jobs; no idle burn |
| Lambda | A100/H100 tiers | reliable but oversized for this model |

**If using vast.ai, three non-negotiables:** (i) checkpoint to object storage — not local disk — every N steps, because a host can vanish mid-run; (ii) verify the CUDA/driver/torch combination in the first five minutes before the corpus is downloaded; (iii) pull the corpus from B2/S3 with an `--onstart` script so a re-rented instance rebuilds itself unattended.

## 3.5 Step 4: Phase 3 — the one place the cost model surprises

Phase 3's joint fine-tune runs the multi-rate stepper ([[impl-atlas-0.1-phase3-integration]] §3.3), so **one macro step is 50 sequential `reacting_flow` calls, 10 `external_flow` calls, and 1 `thermostruct` call.**

**[AI Inference]:** on FLOPs alone that is ~$1.2\times10^{12}$ per sample per macro step, and a 20,000-step fine-tune at batch 4 over 2-macro-step rollouts pencils out to ~2 GPU-hours. **That number is wrong in practice, and predictably so.** A chain of 50 sequential forward passes over a 380-token model is *latency*-bound — each kernel launch is far larger than the work it dispatches, and the GPU sits idle between them. Expect **10–20× worse than the FLOP estimate ⇒ ~20–40 GPU-hours**, and expect the profiler to show low utilization that is not a bug.

Two mitigations worth knowing before it hurts: batch the 50 substeps' *agents* together where the exchange cadence allows (agents `a`,`b`,`e` step at the same rate and can be one batched call), and use CUDA graphs or `torch.compile` on the inner substep to amortize launch overhead. Also note the memory shape: the joint loss backprops through the macro step, so gradient checkpointing on the substep loop is mandatory, not optional — the parallel track already recorded an OOM that appears only after the horizon curriculum grows ([[implementation-log]], 2026-07-19).

## 3.6 Step 5: Phase 4 — remember the judge costs as much as the defendant

Phase 4 grades Atlas against the classical solver on the held-out corner configs. **Those baseline runs cost exactly what generating them cost** — ~16 h/episode under B′. Budget them: 24 held-out episodes ≈ 400 core-hours. If the corpus generation box has already been released, this means renting it again; cheaper to generate the test-set baselines *during* step 1 and store them.

Atlas's own rollout is inference-only and cheap — a few GPU-hours including ablations.

## 3.7 The whole budget on one line

| Leg | Resource | Est. cost | Est. wall clock |
|---|---|---|---|
| Corpus (B′, 144 episodes) | ~2,360 core-h | \$60–100 | 2 days @ 48 cores |
| Object storage | ~20 GB | ~\$1/mo | — |
| Phase 2 training (incl. failures) | ~100 GPU-h | \$40–70 | 4 days @ 1×4090 |
| Phase 3 joint fine-tune | ~20–40 GPU-h | \$10–25 | 1–2 days |
| Phase 4 baselines + rollouts | ~400 core-h + ~10 GPU-h | \$15–25 | 1 day |
| **Total** | | **≈ \$130–220** | **~9 days of machine time** |

**Calendar time is not machine time.** [AI Inference]: at ~9 days of pure compute, the realistic calendar is **6–10 weeks**, and the difference is writing the three experts, the training harness, and the conservation projection — plus the debugging that the M4 structural gates exist to force. The GPU bill is close to a rounding error against that; **do not optimize it, and specifically do not delay the project shopping for cheaper hardware.**

---

# 4. Pitfalls

- **Reading a 5-hour session cap as a reason to coarsen the grid.** It buys 4.7x, not the 8x the cell count suggests (§5.2, measured), and it costs the solver-tokenizer cell correspondence that body-fitting exists to guarantee. Shorten the horizon instead; cost is linear in it and independent of both timesteps.
- **Letting augmentation stand in for episodes.** all2all multiplies pairs ~100x and leaves the number of distinct physical configurations exactly where it was (§5.4). A corpus of 16 episodes is 16 configurations however many pairs it yields.
- **Renting a GPU to unblock Phase 2.** Phase 2 is blocked on *data*, which is CPU work. The GPU would idle. This is the specific error this page exists to prevent.
- **Reading option B as "10 ms episodes" literally.** It drags $\Delta t_{\text{macro}}$ and $\Delta t_{\text{model}}$ down with it and reduces the surrogate's step to ~3× the solver's, destroying the speedup claim and collapsing multi-rate subcycling to a no-op. Decouple $\Delta t_{\text{snap}}$ from $\Delta t_{\text{macro}}$ instead (§2.1).
- **Coarsening the grid without halving the patch size.** L2 is tempting ($r^3$!), but the token budget in [[00-atlas-0.1-implementation-plan]] assumes $8\times8$ patches over the current cells. At $r{=}2$ the patch must become $4\times4$ to hold the 1,114-token table; otherwise every downstream shape assertion, the materialized 3,832-edge graph, and the Phase 1 benchmark all move at once.
- **Generating the corpus on ephemeral disk without pushing to object storage.** The corpus costs 50+ hours; the trained weights cost 2. Protect the expensive artifact, not the cheap one.
- **Assuming full-fidelity 10 s episodes become reachable "once I have a GPU."** They do not — §2.2 prices it at 5,400 GPU-hours even *with* a successful port. Only route C changes that answer.
- **Building the Poseidon-bootstrapped experts before the from-scratch baselines.** The 2a gate requires those baselines anyway, and building them first means a license or channel-count failure costs nothing instead of stalling Phase 2 entirely.
- **Trusting the Phase 3 FLOP estimate.** 50 sequential small forward passes are latency-bound; the real cost is ~10–20× the FLOP count and low GPU utilization is expected, not a misconfiguration.
- **Forgetting that Phase 4's classical baselines cost solver time.** Generate the held-out corner baselines during step 1, while the CPU box is already rented.
- **Deferring Phase 4's max-Q and thrust-vs-altitude milestones silently.** Under B′ these two checks *cannot* be evaluated — 0.2 s of flight does not reach max-Q. Re-scope them explicitly in [[impl-atlas-0.1-phase4-validation]] rather than discovering it at grading time. The other physical-milestone checks (nozzle separation, shock-cell structure) are unaffected.

---

---

---

# 5. Generating under a hard 5-hour session cap

*Added 2026-08-18, in answer to "I do not have the resources to run an A100 for more than 5 hours per stage. If the corpus is small, that's fine, we can augment."*

## 5.1 The first thing to get straight: the A100 is not what generates the corpus

§3.2 already says the corpus is CPU work, but under a per-session cap it is worth restating as arithmetic. `solvers/compressible2d.py` is single-threaded numpy. On a rented A100 box, the thing doing the generation is **the vCPUs attached to that box**, and the GPU sits idle at its full hourly rate. So a 5-hour session buys

$$5\ \text{h} \times N_{\text{vCPU}}\ =\ \text{core-hours available this session}$$

— 60 on a 12-vCPU box, 110 on this 22-core development machine, 150 on a 30-vCPU one. **A CPU-only box of the same core count does the same work for a fraction of the price**, and the recommendation of §3.2 stands. But the plan below is written so it works either way, because the constraint that actually binds is the 5-hour wall, not the hardware.

## 5.2 What a session costs, measured

Core-hours per **second of simulated flight**, all seven agents, timed on the development box (22 logical cores, single-threaded numpy, 2026-08-18):

| coarsen | cells | core-h per flight-second | vs. coarsen 1 |
|---|---|---|---|
| 1 | 68,736 | **65.6** | 1× |
| 2 | 17,184 | 14.1 | 4.7× cheaper |
| 4 | 4,296 | 4.05 | 16× cheaper |

**Two things to read off this table.** First, coarsening pays far less than the $r^3$ its cell count promises (4.7× at $r{=}2$, not 8×): at these sizes numpy is overhead-bound, not flop-bound. Second — and decisive — **coarsening is not available anyway**. The solver and the tokenizer share cells by construction, which is why `b` and `e` are body-fitted at all, and interface fluxes do not survive interpolation. A coarsened corpus would have to be interpolated onto the model's grid, at the exact locations where the supervision matters most.

**So the coupled lever is the horizon, and only the horizon.** Cost is linear in $T_{\text{ep}}$ and completely independent of $\Delta t_{\text{macro}}$ and $\Delta t_{\text{snap}}$ — which is what makes the snapshot count free.

## 5.3 The plan: six portions, each sized to one session

Route B′ is kept and shortened once more: $T_{\text{ep}} = 0.1$ s (2 macro steps) with $\Delta t_{\text{snap}} = 5\times10^{-4}$ s, which is **still 200 snapshots per episode** because storage cadence costs nothing. The subcycling ratio, $\Delta t_{\text{macro}}$, and the $3.3\times10^{3}$ speedup claim are all unchanged; what halves against §2.1 is trajectory evolution, from 4 macro steps to 2 — enough for the loads→motion→freestream loop to close measurably, which is all the non-learned `rigid_body` expert needs.

| portion | kind | items | core-h each | core-h | what only this portion provides |
|---|---|---|---|---|---|
| `shell` | solo | 64 | 0.006 | **0.4** | full 10 s thermostruct histories; `c` has no CFL limit |
| `nozzle` | solo | 48 | 0.38 | **18** | back pressure swept at *fixed* chamber pressure — the choked-throat gate and G6 |
| `external` | solo | 32 | 0.43 | **14** | Mach × altitude coverage independent of shell and plume |
| `coupled_a` | coupled | 16 | 6.56 | **105** | interface supervision |
| `coupled_b` | coupled | 16 | 6.56 | **105** | more *distinct configurations*, which augmentation cannot manufacture |
| `baselines` | coupled | 8 | 6.56 | **53** | Phase-4 held-out corners, generated while the box is already rented |
| **total** | | | | **≈295** | |

At 22 workers every portion fits in one 5-hour session; at 12 workers the coupled ones take two each. **The three solo portions together are 32 core-hours** — under two hours on one machine — and they carry two of the three Phase-2 structural gates. Run them first: they are the cheapest data in the project and the coupled corpus cannot substitute for them.

Runs accumulate into one directory. Resume matches on **content** (`episode_id` hashes the sweep point, timing, decisions and config fingerprint), so a session that hits its wall mid-portion just continues next time, and a stale file with the right name is regenerated rather than trusted.

## 5.4 Augmentation: what it buys, and the one thing it does not

Every expert interface is lead-time conditioned, so for an autonomous PDE any *ordered pair* of snapshots from one episode is a valid training sample: $\binom{200}{2} = 19{,}900$ against 199 consecutive.

| | distinct configurations | training pairs |
|---|---|---|
| 32 coupled episodes, consecutive only | 32 | 6,368 |
| 32 coupled episodes, all2all | 32 | **636,800** |

**The amplification is real supervision and the configuration count is unchanged.** Pairs within one episode are heavily correlated; no augmentation invents a chamber pressure the sweep never visited. This is why `coupled_b` exists as its own portion rather than being folded into a longer `coupled_a`: the second session buys the axis augmentation cannot.

Two limits, both already recorded in [[data-augmentation-physics-surrogates]] and both live here: lead times must be sampled **log-uniformly** or the design step size is starved 200:1 by distant pairs, and **patch jittering remains unavailable** — it breaks the solver↔token cell correspondence that this whole tier structure rests on.

**[AI Inference]:** the short horizon makes the semi-group assumption all2all rests on *more* nearly exact than it was at 10 s, since the vehicle barely moves in 0.1 s and the system is closer to autonomous.

## 5.5 The training side of the same cap

The 5-hour cap applies to Phase 2 as well, and there the GPU is genuinely the resource. §3.4's ~100 GPU-hours becomes **~20 sessions**, which changes two things in the harness before it is written:

1. **Checkpoint-and-resume is mandatory, not a nicety** — every session must end with a resumable state in object storage, exactly as the corpus generator now does with its manifest. A training run that cannot survive its 5-hour boundary cannot be run at all under this constraint.
2. **Order the experts by what they unblock.** `reacting_flow` from scratch (~2 GPU-h productive, so 1 session including restarts) has zero donor dependency and carries the most informative gate. The from-scratch baselines for the other two are ~1 session each. Only the Poseidon-bootstrapped variants (~10 and ~30 GPU-h) need multi-session budgeting, and §3.4 already puts them last on purpose.

**[AI Inference]:** with a 22.5 M-parameter model and a corpus this size, a single 5-hour A100 session is enough to train `reacting_flow` through both stage A (all2all) and stage B (push-forward) — the constraint bites on the donor-bootstrapped experts, not on the from-scratch ones.

---

---

## See Also

- [[impl-atlas-0.1-phase0-scope-and-data]] §3.8 — the A/B/C routes this page chooses between
- [[impl-atlas-0.1-phase2-experts]] — the experts being budgeted; §3.4's donor bootstrap
- [[impl-atlas-0.1-phase3-integration]] §3.3 — the multi-rate stepper whose 50 sequential substeps drive §3.5
- [[impl-atlas-0.1-phase4-validation]] — the milestones B′ re-scopes
- [[atlas-0.1-implementation-log]] — the measured numbers every estimate here is anchored to
- [[00-atlas-0.1-implementation-plan]] — the two-timestep-family table whose ratio is the root cause

---
