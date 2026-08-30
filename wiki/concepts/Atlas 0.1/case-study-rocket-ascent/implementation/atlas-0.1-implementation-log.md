# Atlas 0.1 — Implementation Log

**Type:** Implementation tracker — append-only (folder: Atlas 0.1 / Atlas 0.1 implementation)
**Master plan:** [[00-atlas-0.1-implementation-plan]]
**Distinct from** the vault-wide [[log]] (which records wiki operations) and from [[implementation-log]] (the parallel Noether 1.1 track's build tracker).

---

## Purpose

Append-only record of what was actually built, what broke, and what was measured — the build-layer counterpart to the design pages in `concepts/Atlas 0.1/`. Every entry should be readable by someone with no other context: state the symptom, the root cause, the fix, and the evidence the fix worked.

**Format:** `## [YYYY-MM-DD] <phase or portion> | <status> — <what happened>`
where `status ∈ {built, fixed, optimized, measured, blocked, reverted}`.

**House rules** (carried over from the parallel track's log, which earned them):
- Record the **root cause**, not just the symptom. "Fixed OOM" is not an entry; "OOM because the learned-edge candidate set was O(P²) and only fires at P2, so P0/P1 passed and the box died later" is.
- Record **measurements with numbers and hardware**, not adjectives.
- When an optimization claims to be behaviour-preserving, say how that was verified (bit-exact fingerprint, tolerance, or test name).
- Record **suggestions evaluated and rejected**, with the measurement that rejected them. This prevents the same idea being re-litigated every few weeks.

---

## Portion status board

| Phase | Spec | Status | Milestones |
|---|---|---|---|
| 0 — Scope & data | [[impl-atlas-0.1-phase0-scope-and-data]] | **built** (2026-08-09) — solvers + generator complete. **D1–D6 settled, T3.1–T3.4 and the T2 solo tier built, generation restructured into 5-hour portions; `shell` portion generated** (2026-08-18): route B′ timing, rounded throat, 10 recorded interfaces, exchange stamps, resumable content-addressed generation. Remaining M2 work is T3.5–T3.9, which need the rented box | **M1 ✅ (11 of 12 oracles — nozzle now green, Blasius still low)**, **M2 ⛔ decisions closed, generation blocked on compute** |
| 1 — Native scaffold | [[impl-atlas-0.1-phase1-scaffold]] | **built** (2026-08-09) | **M0 ✅, M3 ✅** |
| 2 — Experts | [[impl-atlas-0.1-phase2-experts]] | not started | M4 |
| 3 — Integration | [[impl-atlas-0.1-phase3-integration]] | not started | M5 |
| 4 — Validation | [[impl-atlas-0.1-phase4-validation]] | not started | M6, M7 |
| 5 — Expansion | [[impl-atlas-0.1-phase5-expansion]] | gated on M7 | — |

## Open action items carried from design

1. **Verify Poseidon checkpoint license and input compatibility** before writing adapter code ([[incremental-transfer-roadmap]], [[impl-atlas-0.1-phase2-experts]] §3.4). Not yet done; blocks Phase 2 for two of three learned experts.
2. Confirm whether Walrus is worth evaluating alongside Poseidon for the `external_flow` expert ([[incremental-transfer-roadmap]] lists both as confirmed-checkpoint donors).
3. ~~Decide the branch name and scaffold the `atlas/` package separately from `noether11/`~~ — **done 2026-08-09**: branch `atlas-0.1` off `noether-1.1`, package `src/atlas/`.

## ⛔ The blocking decision (Phase 0, M2)

**The full-fidelity corpus is not reachable — by four orders of magnitude — and the arithmetic is structural, not an optimization problem.** Measured on this machine (see the 2026-08-09 M2 entry): **820 hours per episode**, **30.6 years for 327 episodes on one core.** A 1000× speedup (GPU + full parallelism) still leaves 11 days *and* assumes the per-substep cost ports perfectly.

The cause is a timescale ratio, not slow code: $\Delta t_{\text{macro}}/\Delta t_{\text{CFL}} \approx 1.5\times10^{5}$ for the nozzle agent, and the spec asks for 200 macro steps (10 s of flight) per episode. Three ways out, none of which the build layer should pick on its own:

| Option | What changes | Cost |
|---|---|---|
| **A. Quasi-steady gas** | Re-converge the gas agents to steady state per macro step instead of marching time-accurately | ~100× cheaper; the surrogate then learns a steady *mapping*, not transient dynamics — this changes what Atlas is |
| **B. Shorter episodes** | Keep time-accurate marching; drop from 10 s of flight to ~10 ms | Feasible today; loses the trajectory coupling that motivates the rigid-body expert |
| **C. Implicit / low-Mach preconditioning** | Remove the acoustic CFL limit | The physically right answer; a substantial solver project |

Everything else in Phase 0 is done and green. A reduced-fidelity corpus (11 episodes, physically correct, `full_fidelity: false` in `/meta`) is generated and validated as proof the pipeline runs end to end.

> **Update 2026-08-18 — the decision is taken; the wall is not moved.** Route **B′** is adopted in code (`EpisodeSpec.dt_snap`, `EpisodeTiming`, `reduction: B-prime`), so the question above is answered: keep time-accurate marching, keep $\Delta t_{\text{macro}}$ and the subcycling ratio, shorten the *horizon* to 0.2 s and snapshot at 1 ms. What that buys is a $50\times$ cheaper episode, ~16 core-hours instead of 820. What it does not buy is a free corpus: the pilot is still ~525 core-hours and production ~1,000–1,500, which is a rented many-core box, not this machine. **Option A stays rejected and option C stays deferred.**

## Open action items raised by the build

4. **Decide whether experts interleave with message-passing layers.** As specified, all message passing runs first and experts run once afterwards, which means nothing propagates across an agent's interior during message passing — see the 2026-08-09 reachability entry. Blocks nothing yet; changes Phase 2's expert interface if the answer is "interleave."
5. ~~**Decide whether to declare a $d\!-\!f$ edge.**~~ — **data side closed 2026-08-18**: `recorded_interfaces` in the config records $d\!-\!f$, $a\!-\!c$ and the shell aft face with `declared: false`, so declaring any of them later is a config change rather than a regeneration, and Phase 4's A4 declared-vs-dense ablation has data. **The design question — whether to declare them — is still open** and belongs to [[edge-generation-atlas-0.1]].
6. ~~**Decide whether wall-exterior patches of `b` and `e` should be dropped.**~~ — **resolved 2026-08-09 by Phase 0**: `b` and `e` are now body-fitted, so there are no wall-exterior cells at all. Token counts unchanged.
7. ~~**Pick a route past the M2 compute wall** (A/B/C).~~ — **closed 2026-08-18: route B′ adopted in code.** `EpisodeSpec` carries `dt_snap`, `n_macro` and `reduction`; `EpisodeTiming` derives the 200-snapshot/4-macro-step episode and refuses a cadence that straddles a macro boundary; `/meta` records all of it. The compute wall itself is unchanged — this decides *what* to generate, not *where*.
8. **The Blasius shortfall is unexplained.** The characteristic far-field BC was implemented and measured on 2026-08-18 and it is **not** the cause: $u_e/u_\infty = 1.033$ is identical under prescribed and characteristic far fields, and replacing the outflow too makes $\delta_{99}$ worse (0.65 vs 0.846). Item stays open with that candidate eliminated; the BC is kept, graded, and unused.
9. **Agent `f` cannot resolve the nozzle exit profile it is fed.** 96 transverse cells on `e` against 8 on `f` over the same 0.24 m. The conservative remap makes the *integral* exact, so mass is no longer lost, but the src-vs-dst discretization jump stays at ~0.47 and Phase 3 projects onto it. Fixing it means changing `f`'s grid, which moves the token budget — a design decision ([[00-atlas-0.1-implementation-plan]]), not a build-layer one.

---

## [2026-08-18] phase 0 | fixed — the budget guard trusted linear scaling, and a first wave is exactly where that is invisible

**Symptom.** The `nozzle` portion, launched with 20 workers under a 5-hour cap, showed each worker accumulating only ~11 CPU-minutes in 80 minutes of wall clock — an effective parallelism of ~2.8 on a 16-physical-core box. The a-priori budget arithmetic (48 items x 0.38 core-h / 20 workers ≈ 0.9 h) was out by a large factor, and the guard never fired.

**Root cause, and it is structural rather than a slow machine.** `over_budget` charged each item at `core_hours / workers`, which assumes the pool scales linearly with worker count. Two things break that assumption at once: the box has 16 physical cores behind 22 logical ones, and this workload is a single-threaded memory-bound stencil that gets little from hyperthreading; and other processes on the machine take real cores. But the deeper problem is *when* the estimate is used — **the entire first wave is submitted before a single item completes**, so the one moment the guard is fully blind is the moment it commits the most work. A wave of 20 long items can overrun the session before any evidence arrives.

**Fix.** The guard now uses the **observed completion rate** — elapsed hours divided by items completed — as soon as anything has finished, falling back to the a-priori charge only before the first completion. That folds contention, core topology and whatever else the machine is doing into one measured number and needs a model of none of them. It also warns when `workers` exceeds the logical CPU count, because oversubscription cannot help a single-threaded solve and *delays the first completion*, which is precisely when the guard is blindest.

**Second lesson, recorded because it cost real time: do not edit modules while a spawn-based process pool is running.** On Windows every worker is a fresh interpreter that imports the package at start, so edits mid-run either race the workers or apply only to processes spawned later — the workers in this run were observed restarting with their CPU-time counters reset, which loses the wave's progress. Land code changes first, then start the session.

**Rejected:** raising the worker count to compensate. The measurement says the machine was already delivering less than 3 effective cores to 20 workers; adding more would have made the first wave later still.

---

## [2026-08-18] phase 0 / M2 | built — generation restructured into bounded sessions; T2 solo tier written and the shell portion generated

**Constraint that drove this.** No more than **5 hours of compute per rented session**. That is not a smaller version of the old plan, it is a different unit of work: the corpus stops being one artifact you generate and becomes a **sequence of portions you accumulate**, each sized to fit one session, each resumable, into one directory.

**First, the correction that has to lead.** The session is described as "5 hours of an A100", and the A100 does not generate anything: `compressible2d.py` is single-threaded numpy, so what does the work is the **vCPUs attached to that box** while the GPU idles at full price. A session therefore buys $5\,\text{h} \times N_{\text{vCPU}}$ core-hours — 60 at 12 vCPU, 110 on this 22-core box. The plan below works on either, but a CPU-only box of the same core count is the same work at a fraction of the cost ([[impl-atlas-0.1-compute-and-training-budget]] §3.2, restated as §5.1).

**Built.**

* `data/generate_solo.py` — the T2 tier, three groups: `nozzle` (`b`+`e` coupled only to each other, back pressure swept at **fixed chamber pressure**), `external` (`d` alone over Mach × altitude), `shell` (`c` alone on synthesized boundary-condition histories). Interface records go through the *same* `gas_face_record` the coupled generator uses — extracted to module level for exactly that reason, so an interface sample means one thing regardless of which tier produced it.
* `corpus.Portion` / `PORTIONS` / `run_portion` — six named portions with a wall-clock guard checked before each item, a process pool, and resume by content hash. A session that hits its cap stops cleanly and prints what remains; rerunning the same command continues.
* Schema and validation now distinguish `kind: solo` from a coupled episode: a solo run has no trajectory and no cross-group interfaces, may never be labelled `full_fidelity`, and is validated on its own terms.
* CLI: `--schedule`, `--portion`, `--hours`, `--g6`, `--augment-report`.
* `tests/test_atlas_solo.py` (13 tests) alongside the existing suites.

**The plan, measured.** Core-hours per second of simulated flight, all agents, this box:

| coarsen | cells | core-h / flight-second |
|---|---|---|
| 1 | 68,736 | **65.6** |
| 2 | 17,184 | 14.1 |
| 4 | 4,296 | 4.05 |

Coarsening pays **4.7× at $r=2$, not the 8×** the cell count promises — numpy is overhead-bound at these sizes — and it would break the solver↔tokenizer cell correspondence that body-fitting exists to guarantee. So the coupled lever is the horizon and only the horizon, and cost is linear in it and independent of both timesteps.

| portion | tier | items | core-h each | core-h |
|---|---|---|---|---|
| `shell` | T2 | 64 | 0.008 | **0.5** |
| `nozzle` | T2 | 48 | 0.38 | **18** |
| `external` | T2 | 32 | 0.43 | **14** |
| `coupled_a` | T3 | 16 | 6.56 | **105** |
| `coupled_b` | T3 | 16 | 6.56 | **105** |
| `baselines` | T3 | 8 corners | 6.56 | **53** |

The coupled horizon shortens to $T_{\text{ep}} = 0.1$ s at $\Delta t_{\text{snap}} = 5\times10^{-4}$ s — **still 200 snapshots**, because snapshot cadence is a storage decision that costs nothing, so D1's arithmetic and all2all's 19,900 pairs per episode are untouched. What halves is trajectory evolution: 4 macro steps to 2.

**Generated this session: the `shell` portion — 64 runs, 5 minutes wall clock on 8 workers, 0.53 core-hours, 0.9 GB, all valid.** Full 10 s flight histories across four boundary-condition modes (ramp, step, oscillation, shutdown) and an amplitude sweep. `augmentation_report`: 12,800 snapshots ⇒ **1,273,600 all2all pairs against 12,736 consecutive** — a 100× amplification, on the expert with the worst public-data coverage, for half a core-hour.

**What the ordering buys, stated plainly.** The three T2 portions together are ~33 core-hours and carry two of the three Phase-2 structural gates. The choked-throat gate in particular **cannot** be graded on the coupled corpus at all — a coupled episode has exactly one back pressure, whatever the altitude gave it — so the cheapest data in the project is also the data no other tier can substitute for. That was already written down in the corpus plan §3.2 as an observation; under a session cap it becomes the schedule.

**Not done:** the coupled portions (one session each) and the T1 public-data adapters. `--gates` refuses to grade M2 until coupled episodes exist, rather than reporting a pass over solo runs.

---

## [2026-08-18] phase 0 / M2 | built — decisions D1–D6 settled in code, T3.1–T3.4 written; four measurements changed the plan

**What.** The six irreversible decisions the [[impl-atlas-0.1-corpus-completion-plan]] says must precede generation are now *in the code and in the file*, and the three T3 modules exist. Nothing here consumes rented compute; the corpus itself is still ungenerated, so M2 stays open on compute.

| Decision | Settled as | Where |
|---|---|---|
| **D1** route B′ + snapshot policy | `dt_snap` decoupled from `dt_macro`; **hold-last**, and `interpolate` raises rather than existing | `EpisodeSpec.timing` → `EpisodeTiming`, `run()` |
| **D2** throat geometry | **rounded, $R = 0.05\ \text{m} = 1.25\,h_t$** — a circular arc whose *lowest point* is the declared throat, with the walls as tangent lines from their fixed endpoints | `geometry.contours.throat_arc`, `nozzle_inner` |
| **D3** far-field BC | characteristic (Riemann-invariant) BC **implemented and graded, but NOT adopted** — the measurement rejected its premise | `solvers.compressible2d`, `BC("farfield")` |
| **D4** recorded interface list | record **10**, declare **7**; $d\!-\!f$, $a\!-\!c$ and the shell aft face carry `declared: false` | `config.recorded_interfaces`, `IFACE_FACES` |
| **D5** exchange stamps | `/iface_t_exch` per interface, plus **both** cadences (`dt_exch` used, `dt_exch_model` per the Phase-3 rule) | `schema` v2, `coupling_cadences` |
| **D6** corpus size | refused as a constant: `PLANS` = smoke / pilot (32) / production (72) / production_max (144), with the ablation rule in the module | `data.corpus.PLANS` |

**Modules.** `data/corpus.py` (T3.3/T3.4: content-addressed `episode_id`, atomic staging, checksummed manifest, resume, provenance), schema v2 (T3.2: `/iface_dst`, `/iface_exch`, `/iface_t_exch`, `/times`, per-edge attributes), the rewritten `scripts/atlas_generate_corpus.py` (`--plan`, `--workers`, `--gates`), and `tests/test_atlas_corpus.py` (19 tests). Full repo suite green.

**Four things the measurements changed.** Each has its own entry below, because each contradicts something the plan assumed.

1. D2 radius is not free: **the exit error is minimised near $R = 0.03$–$0.05$ and rises again past it**, while the window mean keeps falling. 0.05 is both the measured knee and the standard bell-nozzle value.
2. **D3 is falsified.** A characteristic far field does not fix the Blasius shortfall — it changes nothing at the top boundary and makes matters worse when it replaces the outflow.
3. **The teacher must not imitate the surrogate exchange cadence.** Coupling the solver at $\Delta t_{\text{exch}} = \max(\Delta t_p, \Delta t_q)$ starves the plume of the entire nozzle start-up.
4. **G5 dominant term is not lag.** It is the non-matching grids at $e\!-\!f$, and a conservative remap removes it exactly.

**Re-measured with the new `--estimate`, which now reports both horizons.** The cost is set by the simulated horizon and not by either timestep, so the script prints them side by side: **755 h/episode at the spec 10 s, 15.1 h/episode under B′** on this machine. For the 32-episode pilot that is **483 core-hours, 10.1 h wall clock on 48 cores** — against the plan's projected ~525 core-hours and ~11 h, i.e. the budget page's arithmetic survives contact with the implemented timing.

**Not done (unchanged):** the corpus. T3.5–T3.9 (pilot, gates, production, Phase-4 baselines, archive) need the rented CPU box; see [[impl-atlas-0.1-compute-and-training-budget]].

---

## [2026-08-18] phase 0 | measured — D2: the throat radius trades window error against exit error, and 1.25 h_t is the knee

**Setup.** Area–Mach oracle, $120\times24$, inviscid, HLLC, graded on the quasi-1D window ($z > z_t + 0.05$) with the exit station recorded separately because that is what the thrust integral depends on.

| $R$ [m] | $R/h_t$ | window mean | window max | exit |
|---|---|---|---|---|
| 0.00 (declared corner) | 0 | 3.73% | 7.26% | 2.19% |
| 0.03 | 0.75 | 2.61% | 5.04% | **1.33%** |
| **0.05 (adopted)** | **1.25** | **2.16%** | **3.53%** | 1.50% |
| 0.08 | 2.0 | 1.91% | 5.64% | 1.74% |

**Reading.** The window mean falls monotonically with $R$ — more fillet, less Prandtl–Meyer fan, better agreement with a theory that assumes neither. But the *exit* error turns back up past 0.03, because a large fillet moves the effective area distribution away from the declared contour the area ratio is computed from. $R = 1.25\,h_t$ puts the window max at its minimum (3.53%) for 1.50% exit error, and is independently the standard throat radius of curvature for a bell nozzle. Both the table and the rationale now sit in the YAML next to the number.

**Construction, and why it is not a smoothing pass.** The arc is the circle of radius $R$ centred at $(z_t,\ h_t+R)$, so **its lowest point is exactly the declared throat**: the throat station and throat area are unchanged, and $\dot m$, the area ratio, and every oracle that depends on $A_t$ still mean what they meant. The converging and diverging walls become the *tangent lines from their fixed endpoints* to that arc, so $h(0.30)$ and $h(0.70)$ are unchanged too and the profile is $C^1$ between them. Setting `throat_round_radius: 0.0` reproduces the original piecewise-linear contour exactly, which is what makes the change reviewable.

**Consequence for a test that was quietly over-specified.** `test_nozzle_agents_are_body_fitted` compared cell centres against the closed-form contour *evaluated at the cell-centre station*. That is exact only for a piecewise-linear wall: `build_blocks` averages the contour at the two **node** stations, and on a curved wall the two differ by $O(\Delta z^2 \kappa)$. The test now compares against the node average exactly, and against the closed form to 0.1% — grading body-fitting rather than the solver discretization error.

---

## [2026-08-18] phase 0 | measured — D3 rejected: the characteristic far field neither causes nor fixes the Blasius shortfall

**Claim under test** ([[impl-atlas-0.1-corpus-completion-plan]] D3, open action item 8): "Blasius runs 15–19% low because a prescribed-primitive far-field cannot shed a ~3.5% freestream acceleration; the fix is a characteristic (Riemann-invariant) far-field BC."

**Implemented it** — `BC("farfield")`, full Riemann-invariant treatment with both supersonic branches, entropy and tangential velocity taken from the donor side by the sign of $u_n$ — and measured all four placements on the same plate ($28\times24$, $u_\infty = 60$, $\mu = 8\times10^{-4}$):

| far-field treatment | $\delta_{99}/\delta_{\text{Blasius}}$ | station spread | $u_e/u_\infty$ |
|---|---|---|---|
| prescribed (recorded baseline) | 0.846 | 0.032 | 1.033 |
| characteristic at the top only | **0.846** | 0.032 | **1.033** |
| characteristic at inflow + top | 0.842 | 0.031 | 1.033 |
| characteristic everywhere (incl. outflow) | **0.650** | **0.159** | 1.054 |

**Reading.** The edge acceleration is **identical to four digits** in every variant, including the one that replaces the boundary the hypothesis blamed. So the acceleration is not produced by the far-field treatment, and no far-field treatment removes it. Replacing the *outflow* condition as well is actively harmful: $\delta$ falls to 0.65 and the station-to-station spread — the one thing the solver was reproducing exactly — degrades fivefold.

**Decision.** D3 is **implemented, graded, and not adopted**: `EpisodeSpec.farfield` defaults to `prescribed`. The BC stays in the codebase and is now covered by the free-stream-preservation oracle (it must hold at $10^{-12}$ like every other kind), because it is the right condition for a genuinely non-reflecting far field and Phase 5 larger domains may want it. **The bar for adopting it is a far-field-sensitivity study on agent `d`** — halve and double `farfield_halfwidth`, require the solution to be invariant — not a hypothesis about a different test case.

**Open action item 8 stays open**, one candidate cause smaller. Whatever produces $u_e/u_\infty = 1.033$ on that plate, it is not the far-field boundary condition.

---

## [2026-08-18] phase 0 | fixed — the teacher must not imitate the surrogate exchange cadence

**Symptom.** With the generator exchanging at the Phase-3 rule $\Delta t_{\text{exch}} = \max(\Delta t_p, \Delta t_q)$, the $e\!-\!f$ payload refreshes every 5 ms. On a probe episode the nozzle-exit mass flux rises eightfold in the first 10 µs of start-up, and agent `f` spends the whole episode being fed the $t=0$ value — live/held reached **7.4×**.

**Root cause, and the distinction the plan did not draw.** The exchange-cadence rule is a property of the **surrogate** multi-rate stepper, where each expert is a learned operator over its own $\Delta t_{\text{model}}$. The classical solver has no such constraint: it can exchange every substep. Making the teacher imitate the stepper does not produce a *stale record of a correct run* — it produces a **wrong run**, in which the plume never sees the real nozzle exit. That plume is then what the corpus teaches.

**Fix.** `couple_every` on the spec, defaulting to `"snapshot"`: the generator exchanges as often as it snapshots, *except* where the physics forbids it — any interface touching agent `c` cannot refresh faster than `c` advances, which is $\Delta t_{\text{macro}}$. Those holds are real, and D5 stamps mark exactly them. Both cadences are recorded per interface (`dt_exch` as used, `dt_exch_model` for the Phase-3 rule), so a Phase-2 loss can mask against either. Measured maximum staleness after the change: $a\!-\!b$ and $e\!-\!f$ one snapshot, $b\!-\!c$ one macro step.

**Rejected:** keeping the model cadence in the generator and "just recording the lag". That conflates a stale *record* with a wrong *solution*; only the first is maskable.

---

## [2026-08-18] phase 0 | fixed — G5 dominant term is grid mismatch, not coupling lag, and a conservative remap removes it

**Symptom.** Gate G5 as specified — source-side integral against destination-side integral across a conservation-typed interface — reads **35–120% at $e\!-\!f$ and does not decay**, at any coupling cadence.

**Root cause.** The specified metric compares two *interior cells* either side of the interface, so it mixes three unrelated terms: exchange lag, the half-cell offset across a strong gradient, and **transverse grid mismatch**. The third dominates: agent `e` resolves the exit with 96 transverse cells, agent `f` covers the same 0.24 m with 8, and `_remap` **sampled** the source profile at the destination cell centres. On a no-slip profile the outermost destination cells — half the exit area — sample near-wall values, and the plume is fed a mass flux far short of what the nozzle expels.

**Why this matters more than its size suggests.** Phase 3 imposes *hard* mass matching at this exact interface. A constraint whose supervision violates it by 30% cannot be satisfied by any model, and the Phase-3 symptom would be a conservation loss stuck at a floor with nothing pointing at the data — precisely the failure G5 exists to pre-empt.

**Fix.** `_remap_conservative`: integrate the source profile over each destination cell instead of sampling it, so the remapped face carries the same integral by construction. Applied at $e\!-\!f$, the flux-carrying conservation edge.

**Measured**, tail of a probe episode (start-up excluded, see below):

| interface remap | lag error at $e\!-\!f$ | src-vs-dst |
|---|---|---|
| pointwise (previous) | 0.302 | 0.513 |
| **conservative (adopted)** | **0.000** | 0.465 |

Pinned as `test_the_conservation_interface_carries_the_flux_it_is_given`.

**Two metrics, not one.** `interface_lag_error` isolates the coupling term (live source integral against the payload the destination is running on) — that is the number to hold against the Phase-3 tolerance, and it is now exactly zero. `interface_flux_consistency` keeps the plan src-vs-dst definition as a *discretization* diagnostic: its residual 0.465 is agent `f` transverse resolution at the exit plane, which is a token-budget question and therefore a design decision, not a build-layer one. **Recorded as new open action item 9.**

**Both metrics now exclude the leading 25% of the episode.** Every episode starts from a uniform state, and the first snapshots contain an eightfold jump in exit flux that no coupling scheme tracks: whole-episode 0.880, tail 0.000. Grading a coupling scheme on an initial-condition artifact would have hidden the real result in both directions.

---

## [2026-08-09] phase 0 | built — classical solvers, oracles, sweep, coupled generator; M1 green on 10 of 11 oracles

**What.** All of §3's modules, on the same `atlas-0.1` branch: `solvers/{grid,thermo,riemann,compressible2d,thermostruct2d,atmosphere,trajectory,oracles}.py` and `data/{sweep,normalize,schema,generate}.py`, plus `scripts/atlas_generate_corpus.py` and 32 tests. `trajectory.py` is written as pure functions of state and loads, so Phase 2 imports it verbatim as the runtime `rigid_body` expert rather than reimplementing it.

**M1 results** (all graded against closed-form answers, never against another run of the same solver):

| Oracle | Criterion | Measured |
|---|---|---|
| Grid metric closure | machine precision | $4\times10^{-19}$ (all 9 blocks) |
| Sod shock tube | waves within 1 cell | shock 1.0 cell, contact 1.5 cells; $p^*$, $u^*$ exact to $10^{-3}$ |
| HLLC vs Rusanov | < 1% on smooth flow | 0.4% |
| Isentropic vortex | observed order ≥ 1.7 | **1.86** |
| Free-stream preservation | $10^{-12}$, every BC type | **0.0**, incl. the curvilinear nozzle block |
| Constant-pressure reactor | flame temp within 1% | 0.02% |
| **Blasius layer** | **within 10%** | **15–19% LOW — not met**, see below |
| Stokes shear layer (added) | exact viscous grading | 1.2% |
| Conduction vs erf slab | < 1% | **0.14%** |
| Free thermal expansion | zero stress | $1.3\times10^{-4}$ Pa against a $3.2\times10^{8}$ Pa scale |
| Constrained thermal stress | exact | matches $-E\alpha\Delta T/(1-\nu)$ to $10^{-6}$ |
| US Standard 1976 | < 0.5% at 0/5/11/20/30 km | < 0.1% at all five |
| Tsiolkovsky | < 0.1% | 0.02% |
| Nozzle area–Mach | **< 2% centreline** | **2.5% exit, 4.6% mean — not met**, see below |

**Reduced-fidelity corpus generated and validated**: 11 episodes, all 7 agents and all 7 interfaces present, no NaN, global mass-budget residual $6\times10^{-7}$–$2.5\times10^{-5}$ against the M2 tolerance of $10^{-3}$. Marked `full_fidelity: false` in `/meta`, so it can never be mistaken for a production corpus.

---

## [2026-08-09] phase 0 | blocked — M2 is 4 orders of magnitude out of reach; measured, not estimated

**Measured** (`scripts/atlas_generate_corpus.py --estimate`; per-substep cost timed on this machine, substep count from the CFL condition, not guessed):

| agent | cells | $\Delta t_{\text{CFL}}$ (s) | substeps / macro step | ms / substep |
|---|---|---|---|---|
| `a` | 3 840 | $9.59\times10^{-7}$ | 52 122 | 15.3 |
| `b` | 8 960 | $3.36\times10^{-7}$ | 148 675 | 28.8 |
| `e` | 11 520 | $3.12\times10^{-7}$ | 160 384 | 44.3 |
| `d` | 17 664 | $2.02\times10^{-6}$ | 24 723 | 84.5 |
| `f` | 12 160 | $1.27\times10^{-5}$ | 3 926 | 58.9 |
| `g` | 14 592 | $1.23\times10^{-5}$ | 4 055 | 65.0 |

⇒ **14 762 s per macro step**, **820 h per episode**, **30.6 years for the 327-episode corpus** on one core. Hardware: Intel Core Ultra, single-threaded numpy, no CUDA on this box.

**Root cause.** Not slow code. $\Delta t_{\text{CFL}}$ is set by the *acoustic* speed while the flow evolves at the convective one, and $\Delta t_{\text{macro}} = 5\times10^{-2}$ s is $1.5\times10^{5}$ acoustic steps for the nozzle. 200 snapshots then means 10 s of flight resolved at $3\times10^{-7}$ s. The spec's own two-timestep-family table already states this ratio; what it does not do is multiply it out.

**Not done:** silently shrinking the episode and calling M2 met. `EpisodeSpec` exposes `dt_macro`, `n_macro` and `coarsen`, all recorded in `/meta` with a `full_fidelity` flag, so a cheap run is a physically correct short run on a coarse grid and is labelled as such.

**Rejected:** capping gas substeps per macro step. It makes the gas advance less physical time than the structure and trajectory, silently desynchronizing the coupling — a wrong answer that still writes a valid-looking file.

---

## [2026-08-09] phase 0 | fixed — the nozzle oracle fails on the DECLARED contour, and it is geometry, not discretization

**Symptom.** Converged nozzle (residual drift $10^{-12}$), centreline Mach 27.8% off the area–Mach relation at the throat station, 2.2% off at the exit, 4.6% mean over the diverging section, against a 2% criterion.

**Root cause, established by measurement rather than argument.** Refining $120\times24 \to 180\times36$ moved the error the *wrong* way (max 0.193 → 0.202, mean 0.046 → 0.052): it is **resolution-independent**, so it is not discretization. Smoothing the contour (40 passes of a 3-point filter, rounding the throat corner) halved it — max 0.193 → 0.097, mean 0.046 → 0.023. The declared contour has a hard corner at the throat where the wall angle jumps from $-31°$ to $+15°$; that makes the sonic line curved and produces a Prandtl–Meyer fan, and **quasi-1D theory does not describe that flow**. The solver is not wrong; the oracle does not apply within ~15% of the throat.

**Fix.** The test grades the region where quasi-1D is a valid model (downstream of $z_t + 0.05$) and records the exit error separately, since that is what the thrust integral depends on. A rounded throat is the way to meet the criterion as literally written — it changes the declared geometry, so it is not a build-layer decision.

---

## [2026-08-09] phase 0 | fixed — five bring-up defects worth recording

**1. The injector temperature is not free, and 700 K silently deleted the hottest 45% of the sweep.** Rayleigh flow ties $T_{\text{inject}}$ to the chamber area ratio and the top of the $T_c$ range: at $A_c/A_t = 2.5$ the inlet sits at $M = 0.238$, whose choking limit is $T_0^*/T_{0,\text{in}} = 4.10$, so $T_c/T_{\text{inject}} \lesssim 3.90$ with margin. At 700 K that caps $T_c$ at 2730 K, and the screening rejected **246 of 546** candidate configs — all of them at the hot end, exactly where the physics is most interesting. At **900 K** (regeneratively preheated propellant) the declared 2200–3400 K range is fully attainable: 300 train + 40 test, **0 rejected**, minimum margin 0.125.

**2. A fixed Arrhenius pre-exponential cannot work across the sweep.** Induction time scales as $\exp(T_{\text{act}}/T_{\text{inject}})$; the first value tried burned $Y$ to 0.17 over the whole zone. $A$ is now sized per episode so the induction time is ~10% of agent `a`'s residence time, which keeps the flame inside `a` — which is what the $a\!-\!b$ interface has to carry.

**3. The constant-pressure reactor oracle needs the expansion work done explicitly.** A cell's source is a constant-*volume* energy addition ($\Delta T = q/c_v$); the oracle is constant-*pressure* ($q/c_p$). They differ by exactly $\gamma$ — measured 3506 K vs 3000 K. In the coupled solver that expansion is real physics carried by continuity and momentum; only the 0-D harness needed correcting.

**4. US1976 is tabulated in GEOMETRIC altitude; the layer formulas are in GEOPOTENTIAL.** Skipping $H = R_E Z/(R_E+Z)$ is a 2.1% pressure error at 30 km — small enough to look like solver noise, large enough to move a drag calculation. With the conversion, all five reference altitudes match to < 0.1%.

**5. Free-body elasticity: iterative deflation works on a test plate and fails on the real shell.** Jacobi-preconditioned CG on $K + cVV^\top$ exhausted 20 000 iterations on the 232×8 shell, whose elements are 20:1 slivers (20.3 mm × 1 mm). Replaced with a direct sparse bordered (Lagrange-multiplier) solve, factorized once per mesh since $K$ is constant. **Rejected:** pinning three DOFs to remove the rigid-body modes — any pin that also blocks a component of uniform thermal expansion manufactures stress in a plate that should have none, which is the free-expansion oracle itself.

**Also:** the whole test suite was aborting on this machine with `OMP: Error #15` (Anaconda MKL and torch each ship `libiomp5md.dll`) before reaching a single Atlas test. `KMP_DUPLICATE_LIB_OK` is now set in `tests/conftest.py`; this also fixes the pre-existing crash in the Noether track's data-generation fixture.

---

## [2026-08-09] phase 0 | fixed — Phase 1's `b`/`e` grids were a Phase-0 bug waiting to happen

Phase 1 built agents `b` and `e` as bounding boxes with the nozzle wall masked out. That gives the finite-volume solver a **staircase nozzle**, which cannot meet an area–Mach oracle at any resolution, and it means the solver and the tokenizer would be on different cells — so the corpus would have to be interpolated, and interface fluxes do not survive interpolation.

Both are now **body-fitted**: $y$ runs $-h_{\text{in}}(z) \ldots +h_{\text{in}}(z)$, the wall is a grid line, and every cell is inside the cavity. **Token counts are unchanged** (140 and 180 — the patch grid never moved), the ghost-token problem disappears (mean active fraction 0.89/0.67 → 1.00), and open action item 6 is closed.

Going further, `geometry/domains.py` no longer derives the model's cells at all — it takes them from `solvers/grid.build_blocks`. Two independent derivations had already drifted: a block's centroid averages the contour at the two *node* stations while a closed-form $y(z_{\text{centre}})$ evaluates it at the midpoint, and those differ wherever the contour kinks — **1.7 mm on an 8 mm shell** at the throat. One derivation, one answer, and "no interpolation between solver and model" is true by construction rather than by coincidence. (This required making `geometry/__init__.py` lazy, since `solvers.grid` imports `geometry.contours`.)

---

## [2026-08-09] phase 1 | built — native scaffold, `atlas-0.1` branch; M0 and M3 green

**What.** The full Phase 1 forward pass with identity experts, per [[impl-atlas-0.1-phase1-scaffold]]. Branch `atlas-0.1`, cut from `noether-1.1` so `src/atlas/` sits parallel to `src/noether11/` (the repo uses a `src/` layout, so the plan's top-level `atlas/` is `src/atlas/`). Nothing in `atlas/` imports from `noether11/`.

Modules: `config/` (YAML + frozen dataclasses), `geometry/{contours,domains,layout}.py`, `model/{features,tokenizer,edges,message_passing,hierarchy,decoder,atlas}.py`, `model/experts/{base,identity}.py`, plus `scripts/atlas_phase1_bench.py` and three test files (32 tests).

**Geometry reproduces the spec's token table exactly.**

| agent | a | b | e | c | d | f | g | total |
|---|---|---|---|---|---|---|---|---|
| tokens | 60 | 140 | 180 | 116 | 276 | 190 | 152 | **1114** |

The spec's "≈1,112" resolves to 1114 with `g` = 228 patches minus the 4×19 that lie entirely inside the plume. Grids are cell-centred (`z_c`, `y_c`, `area`, `active`) rather than node-based, because `c` and `d` are two-panel maps whose $y$ jumps across the vehicle — a single $[n_z{+}1, n_y{+}1]$ node array cannot express that without special cases.

**Materialized graph.** 3832 directed typed edges from 752 matched token pairs; all five declared types present; per-type slices precomputed once (never a per-layer `.any()` mask, per the spec's pitfall list).

| edge | $a\!-\!b$ | $e\!-\!b$ | $b\!-\!c$ | $c\!-\!d$ | $e\!-\!f$ | $d\!-\!g$ | $g\!-\!f$ |
|---|---|---|---|---|---|---|---|
| pairs | 22 | 14 | 104 | 396 | 24 | 40 | 152 |

**Acceptance.** M0: all import-time assertions pass ($b\!-\!g$ absent in both orders; active cell centres in exactly one domain; agents ∪ declared-unmodelled regions cover the box; every curve on both agents' boundaries; ≥4 pairs per edge; no orphan boundary tokens; `PatchLayout` bit-identical across two constructions after cache clear). M3: shapes preserved on all 7 agents; zeroed decoder ⇒ output **exactly** equals input; every parameter receives a finite, non-zero gradient; one AdamW step changes the loss; isolation test (below). Full repo suite 318 passed.

**Measured** (batch 8, `d_model` 256, 4 MP layers, 1114 tokens, CPU — Intel Core Ultra, torch 2.7.1+cpu, no CUDA on this box):

| | ms/step | process RSS |
|---|---|---|
| forward | 510 | 461 MiB |
| forward+backward | 1396 | 1774 MiB |

Parameters **8.35 M**. Below the plan's 15–25 M band, correctly: the three learned experts are absent. At $12Ld^2$ with $L{=}6$, $d{=}256$ they add ≈4.7 M each ⇒ ≈22.5 M, inside the band.

**Not built, on purpose:** the conservation constraint at $e\!-\!f$/$d\!-\!g$ and multi-rate subcycling (both Phase 3); the per-agent $\Delta t_{\text{model}}$ table is carried on the specs and unused.

---

## [2026-08-09] phase 1 | measured — token-graph reachability is **not** agent-graph reachability; the spec's isolation test cannot pass as written

**Symptom.** The M3 acceptance test "with the declared edge list, $f$ reaches $a$ only via $e$ ($f\!-\!e\!-\!b\!-\!a$), so a **single** message-passing layer must leave `a` unchanged while three layers must not" fails on the second half: at 3 layers `a` is still bit-identical, and so it is at 10.

**Root cause — two compounding facts, neither a bug.**

1. **The materialized graph has no intra-agent edges.** Mechanism A instantiates edges only at *declared interfaces*, so a token can only ever talk to a token in another agent. A signal entering agent $X$ on one interface can leave on a different interface only if some token is a boundary token of **both**. Measured overlaps: $e$'s $e\!-\!f$ and $e\!-\!b$ boundary sets are **disjoint** (0 shared tokens), as are $b$'s $a\!-\!b$ and $e\!-\!b$ sets, and $f$'s $e\!-\!f$ and $g\!-\!f$ sets. So the intended $f\!\to\!e\!\to\!b$ hop does not exist at token level at all.
2. **Experts run once, after all message-passing layers** ([[impl-atlas-0.1-phase1-scaffold]] §3.6's own pseudocode). The expert is the only thing that could mix tokens *within* an agent, and it never runs between MP layers — so this is a property of the architecture as specified, not of the identity placeholders.

**Measured token-graph first-reach hops from $f$:** $e{=}1$, $g{=}1$, $d{=}2$, $c{=}3$, $b{=}6$, $a$ **unreached within 10**. ($b$ is reached at 6 only by ping-ponging along the $c\!-\!d$ interface until the relay walks back to $z\!\approx\!0.4$.) The agent-graph diameter is 4; the token-graph's effective one is much larger and topology-dependent.

**Fix applied.** The test was rewritten to assert the *verifiable and stronger* claim, and the finding was pinned as a regression guard rather than papered over:

- `test_agent_isolation_matches_the_token_graph` — for $L\in\{1,2,3\}$, the set of agents whose output changes when $f$ is perturbed must **equal** (both directions) the set reachable in $\le L$ hops by BFS over the materialized graph. This catches undeclared leakage *and* a declared edge that silently does nothing, which the original one-directional test did not.
- `test_edges_only_connect_declared_agent_pairs` — the agent-level claim directly: materialized connections = declared pairs, exactly.
- `test_identity_experts_give_no_intra_agent_mixing` — asserts `a` is unchanged at $L{=}6$ while `c` (3 hops) is changed, with the measurement above in the docstring.

**Open design question (action item 4).** Whether Phase 2 should interleave experts with MP layers, or add an intra-agent token mixer. Interleaving would make the token graph's reachability match the agent graph's and would make the spec's original expectation true. **Not decided here** — it changes the expert interface and is a design decision, not a scaffold detail.

**Rejected:** adding intra-agent edges to `build_graph` to make the original test pass. That would make the graph no longer *declared* (Mechanism A's whole premise, [[edge-generation-atlas-0.1]]) and would silently move work that belongs to the expert into the edge layer.

---

## [2026-08-09] phase 1 | fixed — $d\!-\!g$ interface narrowed; the partition implies an **undeclared $d\!-\!f$ interface**

**Symptom.** The $d\!-\!g$ curve as written in [[00-atlas-0.1-implementation-plan]] — the plane $z{=}0.70$, $0.12<\lvert y\rvert\le1.5$ — fails this phase's own import-time assertion that every curve lies within $\epsilon_{\text{tol}}$ of *both* agents' boundaries. Distance from the curve at $\lvert y\rvert{=}0.3$ to `g`'s nearest cell is 0.30 m against a tolerance of 0.278 m; at $\lvert y\rvert{=}0.12$ it is 0.48 m.

**Root cause.** In the plan's own partition, `f` is the rectangle $\lvert y\rvert\le0.6$ over $z\in[0.70,3.0]$ (this is what fixes `g`'s token count at ~150: 228 patches minus 4×19 inside the plume — a spreading-cone plume would leave 196 and miss the spec's budget). So on the downstream side of the plane, the strip $0.12<\lvert y\rvert\le0.6$ is **`f`**, not `g`. The declared curve spans an interface `g` does not have.

**Fix.** `plane_d_g` is built over $0.6<\lvert y\rvert\le1.5$ — the sub-segment where `d` and `g` are genuinely adjacent — and the divergence is recorded in code (`InterfaceCurve.declared_span`), not only here.

**What this exposes.** A real $d\!-\!f$ interface over $0.128<\lvert y\rvert\le0.6$ at $z{=}0.70$ that the seven-edge list omits: the plume's upstream face is wider than the nozzle exit, so most of it faces the front atmosphere. Also unlisted, and much smaller: $a\!-\!c$ (the reaction zone's own lateral walls; $b\!-\!c$ covers only $z\in[0.12,0.40]$) and the shell's 8 mm aft face against `f`. **Not added** — adding an edge is a design decision ([[edge-generation-atlas-0.1]]), and the master plan's claim that "every edge has a real, non-degenerate geometric interface … must be preserved by any geometry change" says nothing about the converse. Action item 5.

---

## [2026-08-09] phase 1 | built — three implementation decisions worth recording

**1. An active-mask channel, not in the spec's channel count.** `b` and `e` are bounding-box grids with the nozzle wall inside them, so a token must be able to tell "this cell is zero" from "this cell is not my domain." One extra structural channel per agent (raw ‖ derived ‖ mask). Measured mean active fraction: `b` 0.89, `e` 0.67, `g` 0.91, others 1.00 — i.e. a third of `e`'s cells are outside the nozzle. Attention pooling additionally biases each token by $\log(\text{active fraction})$ so a token covering nothing cannot win its agent's pool.

**Consequence, not yet acted on (action item 6):** `b` and `e` keep entirely-inactive patches near the wall because the spec's token budget is the full patch rectangle (140 and 180). Dropping them would give ~118 and ~138 and save ~13% of all tokens, at the price of deviating from the spec table.

**2. Near-zero, never exactly-zero, init on no-op heads.** The tokenizer's FiLM, the hierarchy's unpool and the decoder's output all want to start as (near) identity, and the obvious way is to zero their last linear. Doing so makes the gradient of *every parameter feeding that layer* exactly zero on step 0 — indistinguishable from a module not wired to the loss, which is precisely the failure M3's "no all-zero gradient" test exists to catch. All three use $\mathcal N(0, 10^{-3})$ instead (`model/utils.near_zero_`), keeping near-identity behaviour and a live gradient. `AgentDecoder.zero_output_()` still exists for the M3 test that the increment path is an exact no-op.

**3. Message-passing depth = 4, the agent-graph diameter.** The spec's `n_layers: 6` is the per-expert backbone depth and says nothing about MP depth. 4 is the smallest depth at which every agent can reach every other exactly once ($a\!-\!b\!-\!c\!-\!d\!-\!g$ and $a\!-\!b\!-\!e\!-\!f\!-\!g$ are both 4 hops); added to the config as `n_mp_layers` and asserted against a BFS over the declared edge list, so the two cannot drift.

**Also fixed during bring-up.** On the slanted converging wall a cell centre can land exactly on $h_{\text{in}}(z)$ (e.g. $z{=}0.31875$, $\lvert y\rvert{=}0.08875$), and with closed-set domain predicates that cell belongs to both the gas agent and the shell. Gas-cavity masks are now strictly interior; the wall itself is the shell's.

---

*Build entries begin 2026-08-09. Format: `## [YYYY-MM-DD] <phase or portion> | <status> — <what happened>`, status ∈ {built, fixed, optimized, measured, blocked, reverted}.*
