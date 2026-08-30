# Atlas 0.1 — Corpus Completion Plan

**Type:** Implementation spec — the ordered work plan for M2 (folder: Atlas 0.1 / Atlas 0.1 implementation)
**Phase of:** [[00-atlas-0.1-implementation-plan]]. **Closes:** M2. **Unblocks:** [[impl-atlas-0.1-phase2-experts]].
**Builds on:** [[impl-atlas-0.1-compute-and-training-budget]] (route B′, venues, budget), [[physics-simulation-datasets]] (the three-tier strategy), [[impl-atlas-0.1-phase0-scope-and-data]] (solvers, sweep, schema — all built).
**Written:** 2026-08-09, answering "what do we have to do to complete our training corpora?"

---

> ## ✅ Status, 2026-08-18 — D1–D6 are settled and T3.1–T3.4 are built
>
> Everything in §2 and §3.3's first four tasks is done, in code, on the `atlas-0.1` branch. What remains of M2 is the part that needs a rented machine: T3.5 (pilot), T3.6 (gates), T3.7 (production), T3.8 (Phase-4 baselines), T3.9 (archive) — plus T1 and T2, which are unblocked and independent.
>
> | | settled as | note |
> |---|---|---|
> | **D1** | route B′ + **hold-last**, with `dt_snap` a first-class field and `interpolate` refused | a cadence that does not divide `dt_macro` now raises at spec time |
> | **D2** | throat **rounded at $R = 1.25\,h_t = 0.05$ m** | measured knee; area–Mach oracle now green (2.16% window mean, 1.50% exit) |
> | **D3** | characteristic far field **implemented, graded, NOT adopted** | its premise was falsified — see below |
> | **D4** | **10 interfaces recorded, 7 declared** | `d–f`, `a–c`, shell aft face tagged `declared: false` |
> | **D5** | `/iface_t_exch` + **both** cadences per interface | and the generator no longer couples at the model cadence — see below |
> | **D6** | `PLANS` = smoke / pilot 32 / production 72 / production_max 144 | the ablation rule travels with the code, not with memory |
>
> **Three findings in this page's own terms.** Each is written up in [[atlas-0.1-implementation-log]]:
>
> 1. **D3's premise was wrong.** A characteristic far field does not remove the Blasius shortfall: $u_e/u_\infty = 1.033$ is identical under prescribed and characteristic far fields, and applying it to the outflow as well drives $\delta_{99}/\delta_{\text{Blasius}}$ from 0.846 down to 0.650. §2's recommendation to "implement it" was right; the reason given for it was not.
> 2. **§2's D5 conflated the teacher's coupling with the surrogate's exchange rule.** Running the *generator* at $\Delta t_{\text{exch}} = \max(\Delta t_p,\Delta t_q)$ does not produce a stale record of a correct run, it produces a wrong run: at 5 ms the plume never sees the nozzle start-up (live/held mass flux $7.4\times$). The generator now exchanges every snapshot except where `c`'s macro step physically forbids it, and *those* holds are what the stamps mark.
> 3. **§4's G5 measures three things at once, and the dominant one is not lag.** Source-vs-destination integrals differ by 35–120% at $e\!-\!f$ because agent `e` resolves the exit with 96 transverse cells and agent `f` with 8 — the pointwise remap *sampled* the profile. A conservative (area-integrating) remap takes the coupling error to **exactly zero** and leaves a ~0.47 discretization jump that is a token-budget question, not a coupling one. G5 is now two metrics: `interface_lag_error` (gate) and `interface_flux_consistency` (diagnostic), both excluding the start-up transient.

---

# 1. Intuition

## 1.1 The corpus is the one artifact that must be right the first time

Everything else in Atlas can be rebuilt cheaply. A trained expert costs two GPU-hours. The scaffold rebuilds in seconds. **The corpus costs ~1,000 core-hours and cannot be patched incrementally** — if a field is missing, a geometry is wrong, or an interface was never recorded, the fix is *regenerate everything*.

So the governing discipline of this phase is not "generate data." It is: **make every irreversible decision before the first production episode runs, and record enough that a decision you got wrong can be revisited without regenerating.** §2 is the decision list; §3 is the work; §4 is how you know it worked.

[[impl-atlas-0.1-phase0-scope-and-data]] §4 already states this in one line for one case — "Saving interiors only … forces corpus regeneration. Get this right the first time." The same logic applies to at least five other decisions currently sitting open.

## 1.2 What is already done

Almost all of the *code*. From [[atlas-0.1-implementation-log]]: all solvers, twelve analytic oracles, the LHS sweep with corner-case holdout, per-episode reference freezing, the HDF5 schema, the coupled generator, and `scripts/atlas_generate_corpus.py` — plus an 11-episode reduced-fidelity corpus that proves the pipeline runs end to end with a mass-budget residual of $6\times10^{-7}$–$2.5\times10^{-5}$ against a $10^{-3}$ tolerance.

**What is missing is not code volume. It is six decisions, three small new modules, and a rented machine.**

## 1.3 The single highest-value change: record more interfaces than you declare

[[atlas-0.1-implementation-log]] open action item 5 asks whether to declare a $d\!-\!f$ edge. The partition implies one — the plume's upstream face is wider than the nozzle exit, so most of it faces the front atmosphere — and the seven-edge list omits it, along with $a\!-\!c$ and the shell's aft face against `f`.

That is currently filed as a *design* question. **It is also a data question with an asymmetric cost**, and that is not recorded anywhere:

- Declare it later, having not recorded it ⇒ **regenerate the entire corpus** (~1,000 core-hours).
- Record it now and never declare it ⇒ a few percent more storage.

**So: record interface fluxes on every geometrically real interface, and declare a subset.** The `/iface` group gets entries for $d\!-\!f$, $a\!-\!c$, and the shell aft face alongside the seven declared edges, each tagged `declared: true|false` in its attributes. The model reads only the declared ones; the data supports either answer. This also makes ablation **A4 (declared-vs-dense graph)** in [[impl-atlas-0.1-phase4-validation]] runnable at all — right now that ablation has no data to run on.

This is cheap insurance against the most expensive mistake available in this phase.

---

# 2. The decisions that must precede generation

Six items, all desk work, all irreversible-once-generating. **None requires a rented machine; all block the production run.**

## D1 — Adopt route B′ and define the snapshot policy ✅ *settled 2026-08-18: hold-last, in `EpisodeTiming`*

Per [[impl-atlas-0.1-compute-and-training-budget]] §2.1: $T_{\text{ep}} = 0.2$ s, $\Delta t_{\text{macro}} = 5\times10^{-2}$ s unchanged, $\Delta t_{\text{snap}} = 10^{-3}$ s ⇒ 4 macro steps, 200 snapshots.

**The non-obvious part.** The generator currently snapshots at macro-step boundaries, where every agent is synchronized. At $\Delta t_{\text{snap}} < \Delta t_{\text{macro}}$ it must snapshot *inside* the gas subcycle — and at those instants **agent `c` and the rigid-body state have not advanced.** Three options, and one must be chosen and recorded in `/meta`:

| Policy | Behaviour | Assessment |
|---|---|---|
| **Hold-last** | write `c` and rigid at their last macro value | **Recommended.** Physically honest — it is exactly what the multi-rate stepper does at inference ([[impl-atlas-0.1-phase3-integration]] §3.3), where `c` genuinely holds between its $5\times10^{-2}$ s steps |
| Interpolate | linearly blend across the macro step | Manufactures states the solver never computed; the surrogate would learn to reproduce an interpolation artifact |
| Sub-step `c` too | advance `c` at $\Delta t_{\text{snap}}$ | Costs little (`c` is cheap, §3.2) but **changes the physics being taught** — the model would no longer see the multi-rate structure it is built around |

Hold-last is not a compromise: it makes the training data match the inference-time behaviour of the stepper, which is what teacher forcing is supposed to do.

## D2 — Freeze the geometry, including the throat ✅ *settled 2026-08-18: rounded, R = 1.25 h_t*

The area–Mach oracle fails on the *declared* contour, and the log established by measurement that this is geometry, not discretization: the hard corner at $z_t$ (wall angle jumping $-31°\to+15°$) curves the sonic line and throws a Prandtl–Meyer fan. Smoothing halved the error; refining made it worse.

**Decide now whether to round the throat**, because the corpus bakes the geometry in permanently. **Recommendation: round it.** The reasons are asymmetric:

- A rounded throat is physically more realistic (no real nozzle has a $46°$ corner).
- It brings the one geometry-driven M1 failure into spec, so the corpus is generated by a solver that passes all its oracles rather than ten of eleven.
- The token table is unaffected — patch layout is set by the bounding grid, not the contour curvature — so nothing downstream moves.
- The alternative is a corpus permanently annotated "generated on a contour whose oracle we knowingly failed."

## D3 — Decide the far-field boundary condition for agent `d` ⚠️ *settled 2026-08-18: implemented, measured, NOT adopted*

Blasius runs 15–19% low because a prescribed-primitive far-field cannot shed a ~3.5% freestream acceleration; the fix is a characteristic (Riemann-invariant) far-field BC (open action item 8). The log correctly notes this "does not block the corpus." But it *does* set the quality of every agent-`d` sample, and `d` is the largest gas agent (17,664 cells, 276 tokens).

**Recommendation: implement it.** It is a well-understood, self-contained BC change, gradeable against the existing Blasius oracle, and it is far cheaper now than discovering in Phase 4 that the external-flow expert inherited a systematic boundary-layer bias from its training data.

## D4 — Freeze the recorded interface list (§1.3) ✅ *settled 2026-08-18: 10 recorded, 7 declared*

Record all geometrically real interfaces; declare seven. Tag `declared` in the HDF5 attributes.

## D5 — Define what `/iface` stores when edges exchange at different cadences ✅ *settled 2026-08-18, with a correction — see the status box*

The exchange cadence rule is $\Delta t_{\text{exch}} = \max(\Delta t_p, \Delta t_q)$, so $a\!-\!b$ and $e\!-\!b$ exchange at $10^{-3}$ s, the conservation edges at $5\times10^{-3}$ s, and $b\!-\!c$/$c\!-\!d$ at $5\times10^{-2}$ s. At $\Delta t_{\text{snap}} = 10^{-3}$ s the slow edges are **stale for up to 50 snapshots.**

Store, per edge, both the flux array and a **`t_exch` timestamp** of when it was last actually computed. Without it, the interface loss $\mathcal L_{\text{iface}}$ trains the model to predict a held constant as though it were a live signal, and nothing in the pipeline can tell the difference.

## D6 — Set the corpus size, and how it will be revised ✅ *settled 2026-08-18: encoded as `data.corpus.PLANS`*

Do not commit to a final episode count. Commit to a **pilot**, an ablation, and a decision rule:

1. Pilot: 32 episodes.
2. Train `reacting_flow` twice — with and without T1 public pretraining — and compare the M4 gate.
3. Set the production count from that measurement: 60–90 episodes if pretraining helps as expected, up toward 144 if it does not.

This is the ablation [[physics-simulation-datasets]] §5 says must precede any cut to the corpus size. It is cheap (two ~2 GPU-hour runs) and it converts an [AI Inference] into a measurement.

> ✅ **Also fold in all2all before sizing the corpus** ([[data-augmentation-physics-surrogates]] §3.1). The expert interface is lead-time-conditioned, so every ordered snapshot pair is a valid sample: $\binom{200}{2}=19{,}900$ per episode against 199 consecutive, a **~$100\times$ amplification for free**. It does *not* replace episodes — it multiplies time pairs *within* an episode, while the number of distinct physical regimes is set by the sweep — but it changes the marginal value of the 91st episode enough that D6's ablation should be run **with all2all already enabled**, or the measurement will overstate how many episodes are needed. B′ helps here too: over $T_{\text{ep}}=0.2$ s the vehicle barely moves, so the semi-group assumption all2all rests on is more nearly exact than it would have been at 10 s.

---

# 3. The work, by tier

## 3.1 Tier 1 — public pretraining data *(free; do it first, it runs in the background)*

Per [[physics-simulation-datasets]]:

| Task | Detail |
|---|---|
| **T1.1** Poseidon checkpoint | license + input-channel/resolution/normalization audit (open action item 1, blocks two experts) |
| **T1.2** BLASTNet 2.0 | select the reacting compressible subsets; **do not pull all 2.2 TB** |
| **T1.3** 2D slicing | extract planar slices from 3D DNS volumes. **[AI Inference]:** a $512^3$ volume yields ~1,500 usable 2D slices, so 744 full-domain samples become $\mathcal O(10^5\!-\!10^6)$ 2D training samples — far more than the coupled corpus will ever hold |
| **T1.4** Normalization adapter | map each external corpus onto Atlas's per-agent nondimensionalization. **This is the real work of T1**, and it is the same adapter machinery Phase 2 §3.4 already budgets for Poseidon |

New module: `data/external/{blastnet,pdegym}.py`.

## 3.2 Tier 2 — uncoupled single-agent runs *(cheap, and the numbers are the surprise)* ✅ *built 2026-08-18: `data/generate_solo.py`*

The measured 820 h/episode is **entirely the six gas agents run coupled, time-accurately, for 10 s of flight.** Run solo and short, the same solvers are cheap. Derived from the log's per-agent table:

| T2 corpus | Basis | **[AI Inference]** cost |
|---|---|---|
| **Nozzle back-pressure sweep** (agent `e`, and `b`+`e`) | ~10 flow-through times ≈ $1.5\times10^{-3}$ s at $\Delta t_{\text{CFL}}=3.1\times10^{-7}$ ⇒ ~4,800 substeps × 44.3 ms ≈ **3.5 min/run**; 200 runs | **~12 core-hours** |
| **External-flow sweep** (`d`, `f`, `g` solo, varied $M_\infty$, altitude) | ~7 min/run at agent `d`'s 84.5 ms/substep; 200 runs | ~25 core-hours |
| **Thermostruct sweep** (agent `c` solo) | see below | **~free** |

**Agent `c` is not in the measured cost table at all, and that is not an oversight.** Its conduction uses backward Euler (unconditionally stable) and its elasticity is a linear solve with $K$ factorized once per mesh — **there is no CFL limit.** It steps at $5\times10^{-2}$ s directly. A full 10 s shell history is 200 solves: seconds, not hours.

The consequence is worth stating plainly: **the expert with the worst public-data coverage (`thermostruct`, §3.4 of [[physics-simulation-datasets]]) has the cheapest possible data generation.** Its solo corpus can be full-horizon, full-fidelity, and *larger and more diverse than the coupled corpus* — by sweeping synthesized boundary-condition histories ($h_{\text{in}}, T_{\text{gas,in}}, h_{\text{out}}, T_{\text{gas,out}}$ as parametrized ramps and oscillations) rather than extracting them from coupled runs.

**[AI Inference]:** synthesized BCs risk teaching `c` an off-distribution boundary response. But M4 grades on held-out *corner* configs — deliberate extrapolation — and [[transfer-learning-fine-tuning]]'s diversity principle says broader BC coverage helps there. Recommended, with the coupled corpus still used for fine-tuning so the in-distribution response is anchored.

**The nozzle sweep is not optional.** The **choked-throat gate** — predicted $\dot m$ at $z{=}0.40$ varying < 2% as downstream $p$ varies at fixed $p_c$ — requires many runs at *varied back-pressure and fixed chamber pressure*. The coupled corpus does not contain that sweep at all (back-pressure there is whatever the altitude gives), and no public dataset has a choked throat. **T2.1 is the only place that gate's training and grading data can come from.**

New module: `data/generate_solo.py` + a solo sweep spec.

## 3.3 Tier 3 — the coupled corpus *(the irreducible cost)* ✅ *T3.1–T3.4 built; T3.5–T3.9 are the budgeted sessions below*

| Task | Detail |
|---|---|
| **T3.1** Implement D1 | `dt_snap` on `EpisodeSpec`, sub-macro snapshot with the hold-last policy, both recorded in `/meta` |
| **T3.2** Implement D4/D5 | extra `/iface` entries with `declared` flags and `t_exch` timestamps |
| **T3.3** Resumability | **required, not optional** — a ~50 h run on a spot/interruptible box *will* be interrupted. Per-episode checkpointing, an idempotent manifest, and a `--resume` that skips completed episodes by content hash rather than by filename |
| **T3.4** Provenance | `/meta` already carries seeds and `solver_versions`; add the git SHA, the D1–D5 decisions, and `full_fidelity: false` with a `reduction: B-prime` tag so a B′ episode can never be confused with a full-horizon one |
| **T3.5** Pilot run | 32 episodes (24 train + 8 corner-holdout), ~525 core-hours, ~11 h on 48 cores |
| **T3.6** Validate | §4's gates, including the new G5 |
| **T3.7** Production run | 60–90 episodes per D6, ~1,000–1,500 core-hours, ~2 days on 48 cores |
| **T3.8** Phase 4 baselines | generate the held-out corner *baselines* now, while the CPU box is rented — [[impl-atlas-0.1-compute-and-training-budget]] §3.6 |
| **T3.9** Archive | push to object storage (B2/S3) with checksums **before releasing the box** |

## 3.3b Bounded sessions — the shape generation actually takes

*Added 2026-08-18.* Generation now has to fit **5 hours per session**, after which the rented box goes away. That changes none of the decisions above; it changes the *unit of work* from "the corpus" to "a portion", and it re-orders the tiers.

$$\text{available this session} \;=\; 5\ \text{h} \times N_{\text{vCPU}}$$

— 60 core-hours on a 12-vCPU box, 110 on a 22-core one. The six portions, each sized to fit (full derivation and measurements in [[impl-atlas-0.1-compute-and-training-budget]] §5):

| portion | tier | items | core-h |
|---|---|---|---|
| `shell` | T2 | 64 | 0.5 |
| `nozzle` | T2 | 48 | 18 |
| `external` | T2 | 32 | 14 |
| `coupled_a` | T3 | 16 | 105 |
| `coupled_b` | T3 | 16 | 105 |
| `baselines` | T3 | 8 (corners) | 53 |

**The three T2 portions together are ~33 core-hours, and they go first.** §3.2 already established that agent `c` has no CFL limit and that the choked-throat gate has no other data source; under a session cap that stops being an interesting observation and becomes the schedule. Two of the three Phase-2 structural gates are trained and graded on data costing under two hours of one machine.

**The coupled horizon shortens once more**, to $T_{\text{ep}} = 0.1$ s at $\Delta t_{\text{snap}} = 5\times10^{-4}$ s. Still 200 snapshots — snapshot cadence is a storage decision and costs nothing — so D1's arithmetic is untouched and all2all still yields 19,900 pairs per episode. What halves is trajectory evolution, 4 macro steps to 2, which the non-learned `rigid_body` expert does not need and which the Phase-4 milestones already could not use under B′.

**Coarsening is still not the lever**, and now there is a measurement: 65.6 core-h per flight-second at coarsen 1 against 14.1 at coarsen 2 — 4.7×, not the 8× the cell count promises, because numpy is overhead-bound at these sizes. It would also break the solver↔tokenizer cell correspondence that body-fitting exists to guarantee. Horizon, not resolution.

**Sizing, against D6.** 32 coupled episodes is below the 60 that D6's with-pretraining branch names. That is a deliberate consequence of the session cap, and it is exactly what D6's ablation is for: run it on `coupled_a` alone and let the result decide whether `coupled_b` is followed by a `coupled_c`. Augmentation raises pairs, not configurations — §5.4 of the budget page keeps those two numbers apart.

## 3.4 Total

**[AI Inference]**, revised from [[impl-atlas-0.1-compute-and-training-budget]]:

| Tier | Core-hours | Wall clock @ 48 cores |
|---|---|---|
| T1 | 0 (bandwidth + adapter dev) | — |
| T2 | ~40 | < 1 h |
| T3 pilot | ~525 | 11 h |
| T3 production | ~1,000–1,500 | 21–31 h |
| T3.8 baselines | ~400 | 8 h |
| **Total** | **~2,000–2,500** | **~2 days** |

Roughly \$60–100 of compute. The schedule is set by the six decisions and three modules, not by the machine.

---

# 4. Acceptance gates — M2, revised

The four criteria in [[impl-atlas-0.1-phase0-scope-and-data]] §3.7 stand, with the episode count from D6 rather than the original 300+40. **Two new gates**, both testing failure modes the current list cannot catch:

## G5 — Interface flux consistency *(new, and the important one)*

The coupled generator uses **loose (Gauss–Seidel) coupling**: at each interface, fluxes are computed from the neighbour's *lagged* state. Phase 3 then imposes **hard flux matching** at $e\!-\!f$ and $d\!-\!g$ ([[conservation-as-constraint-atlas-0.1]]).

**If the lag error in the training data exceeds the constraint's tolerance, the constraint fights the data.** The model would be penalized for satisfying a conservation law that its own supervision signal violates — and the symptom in Phase 3 would be a conservation loss that plateaus at a stubborn nonzero floor, with no indication that the *data*, not the model, is the cause.

**Gate:** measure, on generated episodes, the mass-flux mismatch across each conservation-typed interface:

$$\varepsilon_{\text{iface}} = \frac{\left|\int_{\Gamma}\rho u_n\,d\ell\big|_{\text{src}} - \int_{\Gamma}\rho u_n\,d\ell\big|_{\text{dst}}\right|}{\left|\int_{\Gamma}\rho u_n\,d\ell\big|_{\text{src}}\right|}$$

and require it to sit **below the Phase 3 conservation tolerance with margin.** If it does not, the fix is a tighter coupling (sub-iterate the exchange to convergence within each macro step) — which costs generation time and must be known *before* the production run, not after.

This gate does not currently exist anywhere in the spec. The existing global mass-budget check ($10^{-7}$–$10^{-5}$, comfortably green) is a *whole-system* integral and can pass while individual interfaces disagree, because the errors cancel.

## G6 — Regime coverage of the T2 sweeps

The nozzle sweep must actually span choked and unchoked operation, and over- and under-expanded exit conditions — otherwise the choked-throat gate is graded on a distribution that never varies. Assert that predicted $\dot m$ is flat across the back-pressure sweep in the solver data itself (the ground truth the model will be asked to reproduce), and that the sweep contains flow separation at the over-expanded end.

---

# 5. Ordering

```
D1..D6  (desk, ~days)  ──┬──> T3.1/T3.2/T3.3/T3.4 (code) ──> T3.5 pilot ──> §4 gates ──> T3.7 production ──> T3.8/T3.9
   D2, D3 (geometry/BC)  │                                        │
        └── re-run M1 ───┘                                        └──> D6 ablation (2 GPU-h) ──> production count
T1.1..T1.4 ──────────────── (parallel throughout, free) ──────────────────────────────────>
T2 (needs D2) ────────────> ~40 core-h, run alongside the pilot ──────────────────────────>
```

Three ordering constraints that are load-bearing:

1. **D2 and D3 change the solvers, so all twelve M1 oracles must be re-run** and must reproduce the recorded numbers, not merely pass. The oracle suite exists precisely so a geometry or BC change is gradeable.
2. **T2 depends on D2** (same nozzle contour) but not on any of T3. It can run during the pilot on spare cores.
3. **T1 depends on nothing** and should start immediately — T1.1, the Poseidon license audit, is the longest-standing open action item and blocks two experts.

---

# 6. Pitfalls

- **Generating before D2/D4 are settled.** A geometry change or a newly declared edge after production means regenerating ~1,000 core-hours. Record more interfaces than you declare (§1.3).
- **Snapshotting sub-macro without a stated policy for `c` and the rigid state.** Interpolation silently manufactures states the solver never computed, and the surrogate learns the artifact.
- **Storing `/iface` without `t_exch`.** The slow edges are stale for up to 50 snapshots and nothing downstream can tell.
- **Trusting the global mass budget as an interface check.** It is a whole-system integral; per-interface errors cancel inside it. That is what G5 is for.
- **Running a multi-hour job on a spot instance without resume.** Not a hypothetical — assume interruption and make it cheap.
- **Cutting the corpus on the strength of the pretraining estimate.** Run D6's ablation; it costs two GPU-hours against a decision worth hundreds of core-hours in either direction.
- **Skipping the T2 nozzle sweep because the coupled corpus "has nozzle data."** It has nozzle data at whatever back-pressure the trajectory produced. The choked-throat gate needs back-pressure *varied at fixed $p_c$*, which only a solo sweep provides.
- **Releasing the CPU box before generating the Phase 4 baselines.** They cost the same as generation and are needed for grading; renting twice is pure waste.

---

## See Also

- [[impl-atlas-0.1-compute-and-training-budget]] — route B′, venue choice, the full project budget
- [[physics-simulation-datasets]] — the three-tier strategy and what public data covers
- [[impl-atlas-0.1-phase0-scope-and-data]] — the solvers, sweep, and schema this plan operates
- [[impl-atlas-0.1-phase2-experts]] — the consumer; the choked-throat and zero-load gates T2 serves
- [[impl-atlas-0.1-phase3-integration]] — the conservation constraint G5 protects
- [[impl-atlas-0.1-phase4-validation]] — ablation A4, which §1.3 makes runnable
- [[atlas-0.1-implementation-log]] — the measured numbers and the open action items this plan closes
