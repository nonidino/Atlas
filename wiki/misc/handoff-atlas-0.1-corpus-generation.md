# Handoff Prompt — Atlas 0.1 Corpus Generation

**Type:** Working artifact — session handoff prompt (folder: `wiki/misc/`)
**Purpose:** paste the block below into a fresh chat opened on the **code repo** (`physics-foundation-model`, branch `atlas-0.1`), not on this wiki vault.
**Written:** 2026-08-09. **Companion pages:** [[impl-atlas-0.1-corpus-completion-plan]], [[impl-atlas-0.1-compute-and-training-budget]], [[physics-simulation-datasets]].

> Keep this file in sync if D1–D6 change. If the plan and this prompt disagree, [[impl-atlas-0.1-corpus-completion-plan]] wins.

---

## The prompt

```
# Task: generate the Atlas 0.1 training corpus

You are working on the `atlas-0.1` branch of a private repo implementing a physics
foundation model. Your job this session is to take the corpus from "blocked" to
"generated and validated" — this is milestone M2.

## Project background

Atlas 0.1 is a physics surrogate for a 2D planar rocket ascent. Its thesis: a
physical system can be modelled as a GRAPH OF AGENTS (regions with their own
fields and their own natural timestep) connected by TYPED EDGES (declared
interfaces stating what must be continuous where two agents meet). Each agent is
processed by an EXPERT specialized to one governing-equation family; experts
exchange information only through declared edges. Conservation is IMPOSED at two
edges as a flux-matching constraint, not discovered. One expert (rigid-body
trajectory) is not learned at all — it is a closed-form Newtonian integrator.
The system advances with MULTI-RATE SUBCYCLING: fast combustion agents step many
times per slow structural/trajectory step.

The goal is NOT state-of-the-art accuracy. It is to answer one question: does
composing independently-trained, regime-specialized experts through declared
typed interfaces produce a stable, physically plausible coupled rollout?

Seven agents: `a` combustion reaction, `b` chamber, `e` nozzle, `c` airframe
shell (solid), `d` atmosphere-front, `f` plume, `g` atmosphere-wake.
Seven declared edges: a-b, e-b, b-c, c-d, e-f, d-g, g-f.
Three PDE families: compressible reacting Navier-Stokes (a,b,e,d,f,g), transient
conduction (c), quasi-static plane-stress thermoelasticity (c). Plus a 3-DOF
rigid-body ODE, closed form.

The design and implementation specs live in an Obsidian wiki at
`C:\Users\Nauni\OneDrive\Desktop\Foundation_Model\wiki\`. THE SPECS ARE
AUTHORITATIVE — read them before writing code. Start with, in this order:

  concepts/Atlas 0.1/Atlas 0.1 implementation/impl-atlas-0.1-corpus-completion-plan.md   <- YOUR TASK LIST
  concepts/Atlas 0.1/Atlas 0.1 implementation/00-atlas-0.1-implementation-plan.md        <- binding global conventions
  concepts/Atlas 0.1/Atlas 0.1 implementation/impl-atlas-0.1-phase0-scope-and-data.md    <- the solvers/schema you are operating
  concepts/Atlas 0.1/Atlas 0.1 implementation/atlas-0.1-implementation-log.md            <- what is built, what broke, measured numbers
  concepts/Atlas 0.1/Atlas 0.1 implementation/impl-atlas-0.1-compute-and-training-budget.md

## Current state

Phases 0 and 1 are BUILT (2026-08-09). All classical solvers, twelve analytic
oracles, the LHS sweep, the HDF5 schema, per-episode reference freezing, and the
coupled generator exist and work. An 11-episode reduced-fidelity corpus has been
generated and validated end to end (global mass-budget residual 6e-7 to 2.5e-5
against a 1e-3 tolerance).

M1 passes on 10 of 11 oracles. Two known gaps, both diagnosed by measurement:
  - Nozzle area-Mach: 2.5% at exit vs a 2% criterion. Cause is GEOMETRY, not
    discretization — refining made it worse, smoothing the throat corner halved
    it. The declared contour has a hard corner at the throat (wall angle jumps
    -31deg to +15deg) which curves the sonic line; quasi-1D theory does not
    describe that flow.
  - Blasius boundary layer 15-19% low. Cause is a prescribed-primitive far-field
    that cannot shed a ~3.5% freestream acceleration.

M2 is BLOCKED. The corpus as originally specified costs 820 hours PER EPISODE and
30.6 YEARS for 327 episodes on one core — measured, not estimated. The cause is
structural, not slow code: the CFL limit follows the acoustic speed while the flow
evolves at the convective one, so dt_macro/dt_CFL ~ 1.5e5 for the nozzle agent.

## The decision that unblocks it (already made — implement, do not relitigate)

ROUTE B-PRIME: decouple the SNAPSHOT CADENCE from the MACRO STEP. These are two
different things that the original spec conflates.

    T_ep      = 0.2 s      (was 10 s)
    dt_macro  = 5e-2 s     (UNCHANGED)
    dt_snap   = 1e-3 s     (new parameter)
    => 4 macro steps, still 200 snapshots, 50x cheaper

This preserves everything the architecture depends on: dt_macro, per-agent
dt_model, the 50:10:1 subcycling ratio, the surrogate's 3.3e3 speedup ratio, the
200-snapshot schema. The only loss is long-horizon trajectory evolution, which is
the cheapest available loss because the rigid_body expert is CLOSED FORM and no
quantity of trajectory data trains it.

Do NOT implement plain "option B" (10 ms episodes). That drags dt_model down to
~3x dt_CFL and destroys the speedup claim the whole architecture exists to make.

## Your work, in order

### Step 1 — decisions to implement (all are settled; implement as stated)

D1. Add `dt_snap` to `EpisodeSpec`; snapshot inside the gas subcycle loop.
    At sub-macro snapshots, agent `c` and the rigid state have NOT advanced.
    Policy: HOLD-LAST — write their last macro value. Do not interpolate;
    interpolation manufactures states the solver never computed. Hold-last is
    also exactly what the multi-rate stepper does at inference. Record the
    policy in /meta.

D2. ROUND THE THROAT. Freeze the geometry before generating — the corpus bakes
    it in permanently and regenerating costs ~1000 core-hours. Rounding fixes
    the one geometry-driven M1 failure and does not move the token table
    (patch layout is set by the bounding grid, not contour curvature).

D3. Implement a CHARACTERISTIC (Riemann-invariant) far-field BC for agent `d`.
    Closes the Blasius gap. `d` is the largest gas agent (17,664 cells, 276
    tokens) and every sample of it inherits the current systematic bias.

D4. RECORD MORE INTERFACES THAN YOU DECLARE. The partition implies real
    interfaces the seven-edge list omits: d-f (the plume's upstream face is
    wider than the nozzle exit, so most of it faces the front atmosphere),
    a-c, and the shell's aft face against `f`. Write /iface entries for ALL
    geometrically real interfaces, each tagged `declared: true|false`.
    Rationale: declaring one later without having recorded it means
    regenerating everything; recording and never declaring costs a few percent
    of storage. This also makes Phase 4's A4 declared-vs-dense ablation
    runnable, which it currently is not.

D5. Store a `t_exch` timestamp per edge alongside each /iface flux array.
    Exchange cadence is max(dt_p, dt_q), so a-b and e-b exchange at 1e-3 s but
    b-c and c-d at 5e-2 s — the slow edges are STALE for up to 50 snapshots.
    Without the stamp, downstream training cannot tell a live flux from a held
    constant.

AFTER D2 AND D3: re-run all twelve M1 oracles. They must REPRODUCE THE RECORDED
NUMBERS in the implementation log, not merely pass. A change that passes the
oracles but moves the isentropic-vortex order from 1.86 to 1.6 has quietly
changed the physics.

### Step 2 — generator hardening

- RESUMABILITY IS REQUIRED, not optional. This runs ~50 h on a rented,
  interruptible box and WILL be interrupted. Per-episode checkpointing, an
  idempotent manifest, and `--resume` that skips completed episodes by CONTENT
  HASH, not by filename.
- Provenance in /meta: git SHA, seeds, solver_versions, the D1-D5 decisions,
  `full_fidelity: false`, and `reduction: B-prime`.
- Episode generation is embarrassingly parallel — one process per core, no
  communication. Wall clock = core-hours / cores with no scaling loss.

### Step 3 — the solo (uncoupled) corpora, which are nearly free

The measured 820 h/episode is ENTIRELY the six gas agents run coupled,
time-accurately, for 10 s. Run solo and short, the same solvers are cheap:

  - Nozzle back-pressure sweep (`e`, and `b`+`e`): ~10 flow-through times is
    ~4,800 substeps at 44.3 ms => ~3.5 min/run. 200 runs ~= 12 core-hours.
    THIS IS NOT OPTIONAL. The choked-throat acceptance gate needs back-pressure
    VARIED AT FIXED CHAMBER PRESSURE. The coupled corpus has only whatever
    back-pressure the trajectory produced, and no public dataset anywhere
    contains a choked throat. This sweep is the only possible source.
  - External-flow sweep (`d`, `f`, `g` solo, varied M_inf and altitude):
    ~7 min/run, 200 runs ~= 25 core-hours.
  - Thermostruct sweep (`c` solo): agent `c` has NO CFL LIMIT — backward Euler
    conduction plus an elastic solve with K factorized once per mesh. It steps
    at 5e-2 s directly, so a full 10 s shell history is 200 solves: seconds.
    Sweep SYNTHESIZED boundary-condition histories (h_in, T_gas_in, h_out,
    T_gas_out as parametrized ramps and oscillations) rather than extracting
    them from coupled runs. This gives the expert with the WORST public-data
    coverage the LARGEST and most diverse corpus, for free.

New module: `data/generate_solo.py` plus a solo sweep spec.

### Step 4 — run it

  1. Pilot: 32 episodes (24 train + 8 corner-holdout). ~525 core-hours, ~11 h
     on 48 cores.
  2. Validate against Step 5's gates.
  3. Production: 60-90 episodes. ~1000-1500 core-hours, ~1 day on 48 cores.
     The exact count comes from a pretraining ablation run separately — ask me
     before committing to a number.
  4. Generate the Phase 4 held-out corner BASELINES in the same session. They
     cost the same as generation and are needed for grading; renting the box
     twice is pure waste.
  5. Push everything to object storage with checksums BEFORE releasing the box.
     The corpus is the single most expensive artifact in the project — losing
     it costs 50+ hours, losing trained weights costs 2.

Venue note: rent a MANY-CORE CPU BOX (Hetzner dedicated/CCX, or AWS/GCP
c-family spot). Do NOT rent a GPU for this — data generation is CPU work and a
GPU would idle. vast.ai is a poor fit for the CPU leg (GPU-priced, thin CPU
offers, interruptible, non-durable disk); it is a fine venue later for training.

### Step 5 — acceptance gates (M2)

Existing, from the Phase 0 spec:
  - Episodes complete without NaN or CFL failure.
  - Every episode has all 7 agents, all declared interfaces, 200 snapshots.
  - Global mass budget closes: |dm_system + integral(mdot_exit)| / m0 < 1e-3.
  - Held-out corner cases show visibly different plume structure from the
    training mean.

NEW — G5, INTERFACE FLUX CONSISTENCY. This is the important one and it does not
exist anywhere in the current codebase. The generator couples LOOSELY
(Gauss-Seidel, lagged neighbour state). Phase 3 later imposes HARD flux matching.
If the lag error exceeds the constraint tolerance, the constraint fights its own
training data, and the symptom — a conservation loss plateauing at a stubborn
nonzero floor — points at the model rather than the data.

  Measure, per conservation-typed interface:
      eps = |int(rho*u_n)dl|_src - int(rho*u_n)dl|_dst| / |int(rho*u_n)dl|_src|
  Require it below the Phase 3 conservation tolerance WITH MARGIN.

  The existing global mass-budget check CANNOT catch this — it is a whole-system
  integral and per-interface errors cancel inside it. If G5 fails, the fix is
  sub-iterating the exchange to convergence within each macro step, which costs
  generation time and must be known BEFORE the production run.

NEW — G6, REGIME COVERAGE. The nozzle sweep must actually span choked and
unchoked operation and over/under-expanded exits. Assert that mdot is flat
across the back-pressure sweep in the SOLVER data (the ground truth the model
will be asked to reproduce), and that the sweep contains flow separation at the
over-expanded end.

## Working conventions

- The wiki specs are authoritative. If a spec and this prompt disagree, tell me
  rather than picking one.
- Record what you build in
  `wiki/concepts/Atlas 0.1/Atlas 0.1 implementation/atlas-0.1-implementation-log.md`,
  append-only, format `## [YYYY-MM-DD] <phase> | <status> — <what happened>`,
  status in {built, fixed, optimized, measured, blocked, reverted}. House rules:
  record the ROOT CAUSE not the symptom; measurements with NUMBERS AND HARDWARE,
  not adjectives; when an optimization claims to be behaviour-preserving, say how
  that was verified; and record suggestions EVALUATED AND REJECTED with the
  measurement that rejected them.
- My machine is slow and has no CUDA. Nothing gets trained or generated locally
  beyond smoke tests — assume rented compute for anything real.
- Do not silently shrink an episode and call M2 met. `EpisodeSpec` exposes
  dt_macro, n_macro, dt_snap and coarsen, all recorded in /meta with a
  full_fidelity flag, so a cheap run is a physically correct short run that is
  LABELLED as such.
- Known trap on this machine: Anaconda MKL and torch each ship libiomp5md.dll,
  which aborts the test suite with OMP: Error #15. `KMP_DUPLICATE_LIB_OK` is
  set in tests/conftest.py — leave it there.

## Start by

Reading the five wiki pages listed above, then telling me your plan for D1-D5
and anything in it you think is wrong. Do not start writing code until we agree
on the plan.
```

---

## See Also

- [[impl-atlas-0.1-corpus-completion-plan]] — the authoritative version of this task list
- [[impl-atlas-0.1-compute-and-training-budget]] — venue and budget reasoning behind Step 4
- [[physics-simulation-datasets]] — why the solo corpora and public data matter
- [[atlas-0.1-implementation-log]] — the build log the new session must append to
