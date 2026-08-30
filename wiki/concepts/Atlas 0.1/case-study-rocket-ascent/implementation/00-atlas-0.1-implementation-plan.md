# Atlas 0.1 — Master Implementation Plan

**Type:** Implementation spec — master plan (folder: Atlas 0.1 / Atlas 0.1 implementation)
**Design pages (authoritative):** [[00-atlas-0.1-overview]], [[case-study-rocket-ascent-2d-atlas-0.1]], [[expert-library-atlas-0.1]], [[edge-generation-atlas-0.1]], [[unet-hierarchy-atlas-0.1]], [[conservation-as-constraint-atlas-0.1]], [[global-fields-and-topology-atlas-0.1]], [[training-and-bootstrap-atlas-0.1]]
**Phase specs:** [[impl-atlas-0.1-phase0-scope-and-data]], [[impl-atlas-0.1-phase1-scaffold]], [[impl-atlas-0.1-phase2-experts]], [[impl-atlas-0.1-phase3-integration]], [[impl-atlas-0.1-phase4-validation]], [[impl-atlas-0.1-phase5-expansion]]
**Progress tracker:** [[atlas-0.1-implementation-log]]

> **Port migration notice (2026-08-19).** The edge-type labels throughout this folder (`heat`, `pressure`, `stress`, `fluid`, `conservation`) predate [[port-algebra-atlas-0.1]] and have **not** been rewritten here, because this phase is blocked at M2 and rewriting a blocked spec is churn. The mapping is [[port-algebra-atlas-0.1]] §7.1, and it is a **correction, not a rename**: four of the seven rocket edges carried a `heat` label across an interface with mass crossing it and therefore omitted advected enthalpy. **Apply the migration before this folder is unblocked, not after.**

---

## What is being built, in one paragraph

**Atlas 0.1** is a physics surrogate for a 2D rocket ascent, structured as a graph of **seven agents** (regions of the physical system, each with its own fields and its own natural timestep) connected by **seven typed edges** (declared interfaces stating what must be continuous where two agents meet). Each agent is processed by an **expert** — a sub-network specialized to one governing-equation family — and the experts exchange information only through the declared edges. Conservation is **imposed** at two of those edges as a flux-matching constraint, not discovered from data. One expert (rigid-body trajectory) is not learned at all: it is a closed-form Newtonian integrator. The whole thing advances in time with **multi-rate subcycling** (fast combustion agents step many times per slow structural/trajectory step) and is validated against classical solvers that also generate its training data.

The goal of Atlas 0.1 is **not** state-of-the-art accuracy. It is to answer one question: *does composing independently-trained, regime-specialized experts through declared typed interfaces produce a stable, physically plausible coupled rollout?* Phase 4 is that experiment; Phases 0–3 exist to make it runnable.

---

## Repository layout (target)

Code home: the `atlas-0.1` branch of [github.com/nonidino/physics-foundation-model](https://github.com/nonidino/physics-foundation-model), cut from `noether-1.1` on 2026-08-09. Atlas is a **new top-level package**, not a modification of the `noether11/` package — the two architectures are parallel tracks ([[00-atlas-0.1-overview]]) and must not share mutable state.

> *As built:* the repo uses a `src/` layout, so the tree below lives at **`src/atlas/`**, parallel to `src/noether11/`. Nothing in `atlas/` imports from `noether11/`. Phase 1 added `geometry/contours.py` (the three profiles every agent mask and interface curve is generated from), `model/features.py`, `model/message_passing.py`, `model/utils.py`, and `model/experts/{base,identity}.py`.

```
atlas/
  config/
    atlas_0_1.yaml           # single source of truth for all dims/scales
  geometry/
    domains.py               # AgentSpec, agent masks, interface curves
    layout.py                # patch layout, token centroids, boundary flags
  solvers/                   # classical: data generation AND runtime baseline
    compressible2d.py        # FV Navier-Stokes/Euler (agents a,b,e,d,f,g)
    thermostruct2d.py        # conduction + plane-stress elasticity (agent c)
    atmosphere.py            # US Standard Atmosphere 1976
    trajectory.py            # 3-DOF rigid body (ALSO the runtime expert)
  data/
    generate.py              # sweep driver
    dataset.py               # HDF5 -> torch
    normalize.py             # per-agent nondimensionalization
  model/
    tokenizer.py             # per-agent patch tokenizer
    edges.py                 # declared/geometric edge instantiation
    experts/
      base.py                # Expert ABC
      reacting_flow.py       # agents a, b, e
      external_flow.py       # agents d, f, g
      thermostruct.py        # agent c
      rigid_body.py          # thin wrapper over solvers/trajectory.py
    router.py                # agent/edge -> expert dispatch
    hierarchy.py             # 2-level U-Net (token level + bottleneck)
    conservation.py          # flux matching at e-f, d-g
    globals.py               # gravity / uniform-field injection
    atlas.py                 # top-level nn.Module + multi-rate stepper
  train/
    train_expert.py          # Phase 2: per-expert
    train_joint.py           # Phase 3: coupled fine-tune
    losses.py
    metrics.py
  eval/
    rollout.py               # Phase 4 harness
    report.py
  tests/
```

---

## Phase dependency graph

```
Phase 0 (data)  ──────────────┐
      │                       │
      ├──> Phase 1 (scaffold) ├──> Phase 3 (integration) ──> Phase 4 (validation) ──> Phase 5 (expansion)
      │                       │
      └──> Phase 2 (experts) ─┘
```

- **Phase 0** and **Phase 1** are independent and may proceed in parallel (Phase 1 can be built and shape-tested against synthetic random tensors before real data exists).
- **Phase 2** requires Phase 0 (data) and Phase 1 (the tokenizer/decoder each expert plugs into).
- **Phase 3** requires *all three* learned experts to have passed their Phase 2 gates. This ordering is load-bearing: composing unvalidated experts makes a pipeline-level failure undiagnosable ([[training-and-bootstrap-atlas-0.1]]).
- **Phase 4** is the actual experiment.
- **Phase 5** is gated on Phase 4 passing.

## Milestones

| ID | Milestone | Phase | Definition of done |
|---|---|---|---|
| **M0** ✅ | Repo scaffold + config schema loads; all tests green on an empty pipeline | 1 | `pytest` passes; `atlas.Atlas(cfg)` constructs — **done 2026-08-09** |
| **M1** ⚠️ | Classical solvers reproduce analytic oracles | 0 | **10 of 11 oracles green 2026-08-09**; area–Mach 2.5% (not 2%) and Blasius 15–19% low — both diagnosed, see [[impl-atlas-0.1-phase0-scope-and-data]] §3.7 |
| **M2** ⛔ | Full training corpus generated + normalized | 0 | **Blocked on compute**: 820 h/episode, 30.6 years/corpus on one core. Needs a decision from §3.8 — route **B′** recommended and costed in [[impl-atlas-0.1-compute-and-training-budget]] |
| **M3** ✅ | Scaffold forward pass runs end-to-end with identity experts | 1 | Shapes correct, gradients finite, one optimizer step succeeds — **done 2026-08-09** |
| **M4** | Each of 3 learned experts passes its solo accuracy gate | 2 | See per-expert gates in [[impl-atlas-0.1-phase2-experts]] |
| **M5** | Coupled forward pass + conservation constraint active | 3 | Mass-flux residual at $e\!-\!f$, $d\!-\!g$ below tolerance |
| **M6** | Full 2D ascent rollout, stable to burnout | 4 | No NaN/divergence; energy/mass drift within budget |
| **M7** | Validation report vs. classical baselines + physical milestones | 4 | Thrust curve and max-Q agreement within stated tolerance |

---

## Global conventions (binding across all phases)

These are defined once here and repeated in each phase page so that each page is self-contained. **If a phase page and this page disagree, this page wins.**

### Coordinates and frame

2D **planar (slab)** geometry — *not* axisymmetric. Body-fixed frame:
- $z$ = axial coordinate, increasing **downstream** (injector face at $z=0$, exhaust flows toward $+z$).
- $y$ = transverse coordinate, symmetric about $y=0$.
- All agent geometry below is specified in this body-fixed frame in metres. The vehicle's motion through the world frame is carried entirely by the rigid-body expert (Phase 3); the field agents never see world coordinates.

> **Why planar, not axisymmetric:** axisymmetric ($r$–$z$) formulation adds $1/r$ geometric source terms to every flux and a singular axis, which is real work for no benefit to the question Phase 4 asks. Planar slab flow is a self-consistent 2D physics problem; thrust magnitudes will not match a real engine, and **that is acceptable and expected** — Phase 4 compares Atlas against *the same planar solver*, not against flight data. Axisymmetric is a documented later upgrade (Phase 5).

### The seven agents (non-overlapping, exhaustive over the modelled domain)

| ID | Name | Domain (body-fixed, m) | Fields | Governing family |
|---|---|---|---|---|
| `a` | combustion reaction | $z\in[0,0.12]$, $\lvert y\rvert\le0.10$ | $\rho,u,v,p,T,Y$ | reacting compressible flow |
| `b` | combustion chamber | $z\in[0.12,0.40]$, inside walls (converging after $z{=}0.30$) | $\rho,u,v,p,T$ | confined compressible flow |
| `e` | combustion outflow (nozzle) | $z\in[0.40,0.70]$, diverging | $\rho,u,v,p,T$ | confined compressible flow |
| `c` | airframe (solid shell) | 8 mm shell, $z\in[-4.0,0.70]$ | $T,u_x,u_y,\sigma_{zz},\sigma_{yy},\sigma_{zy}$ | conduction + plane-stress elasticity |
| `d` | atmosphere-front | $z\in[-5.0,0.70]$, outside shell, $\lvert y\rvert\le1.5$ | $\rho,u,v,p,T$ | external compressible flow |
| `f` | plume | $z\in[0.70,3.0]$, $\lvert y\rvert\le0.6$ | $\rho,u,v,p,T$ | free-jet compressible flow |
| `g` | atmosphere-wake | $z\in[0.70,3.0]$, $\lvert y\rvert\le1.5$ minus `f` | $\rho,u,v,p,T$ | external compressible flow |

Nozzle geometry: chamber half-height $0.10$; converging $z\in[0.30,0.40]$, half-height $0.10\!\to\!0.04$; **throat at $z=0.40$**, half-height $y_t=0.04$; diverging $z\in[0.40,0.70]$, half-height $0.04\!\to\!0.12$. Planar expansion ratio $\varepsilon = A_e/A_t = 3.0$.

### The seven typed edges (final — $b\!-\!g$ removed, see [[edge-generation-atlas-0.1]])

| Edge | Types | Interface geometry |
|---|---|---|
| $a\!-\!b$ | heat, pressure | plane $z=0.12$, $\lvert y\rvert\le0.10$ |
| $e\!-\!b$ | heat, pressure | throat plane $z=0.40$, $\lvert y\rvert\le0.04$ |
| $b\!-\!c$ | heat, stress | chamber inner wall (curve from $z{=}0.12$ to $z{=}0.40$) |
| $c\!-\!d$ | stress, heat, fluid | airframe outer wall (curve, $z\in[-4.0,0.70]$) |
| $e\!-\!f$ | conservation | nozzle exit plane $z=0.70$, $\lvert y\rvert\le0.12$ |
| $d\!-\!g$ | conservation (heat, fluid) | plane $z=0.70$, $0.12<\lvert y\rvert\le1.5$ |
| $g\!-\!f$ | heat, fluid | plume shear layer (curve, $z\in[0.70,3.0]$) |

Every edge has a real, non-degenerate geometric interface. This is a consistency property of the partition above and must be preserved by any geometry change.

> ⚠️ **$d\!-\!g$, as built:** the span written above ($0.12<\lvert y\rvert\le1.5$) is **not** on `g`'s boundary. In this same partition `f` is the rectangle $\lvert y\rvert\le0.6$, so at $z{=}0.70$ the strip $0.12<\lvert y\rvert\le0.6$ faces `f`. The implemented curve is $0.6<\lvert y\rvert\le1.5$. The converse of the property above does **not** hold: the partition implies a real **$d\!-\!f$** interface that the edge list omits (also unlisted: $a\!-\!c$, and the shell's aft face against `f`). Whether to declare them is an open item — [[atlas-0.1-implementation-log]] 2026-08-09.

### Expert assignment

| Expert | Agents / edge channels | Learned? |
|---|---|---|
| `reacting_flow` | `a`, `b`, `e`; heat+pressure channels of $a\!-\!b$, $e\!-\!b$ | Yes — from scratch |
| `external_flow` | `d`, `f`, `g`; fluid channel of $c\!-\!d$; $g\!-\!f$ | Yes — Poseidon bootstrap |
| `thermostruct` | `c`; $b\!-\!c$; stress+heat channels of $c\!-\!d$ | Yes — Poseidon bootstrap |
| `rigid_body` | bottleneck level; gravity | **No — closed form** |

Conservation at $e\!-\!f$ and $d\!-\!g$ is a **constraint**, not an expert ([[conservation-as-constraint-atlas-0.1]]).

### Two timestep families (do not confuse these)

- **$\Delta t_{\text{solver}}$** — CFL-limited step used *only* inside classical solvers during Phase 0 data generation. Order $10^{-7}$–$10^{-6}$ s for the gas agents.
- **$\Delta t_{\text{model}}$** — the step Atlas advances per expert forward pass. This is what makes the surrogate a speedup, and it is **much larger** than $\Delta t_{\text{solver}}$.

| Agent | $\Delta t_{\text{model}}$ (s) | Substeps per macro-step |
|---|---|---|
| `a`, `b`, `e` | $1\times10^{-3}$ | 50 |
| `d`, `f`, `g` | $5\times10^{-3}$ | 10 |
| `c` | $5\times10^{-2}$ | 1 |
| rigid body | $5\times10^{-2}$ | 1 |

**Macro-step $\Delta t_{\text{macro}} = 5\times10^{-2}$ s.** Agents synchronize across edges exactly once per macro-step (Phase 3).

> ⛔ **This table's own ratio is what blocks M2.** $\Delta t_{\text{macro}}/\Delta t_{\text{solver}} \approx 1.5\times10^{5}$ for the nozzle agent, and 200 snapshots per episode means 10 s of flight resolved at $3\times10^{-7}$ s: **820 h per episode, 30.6 years for the corpus** on one core (measured 2026-08-09, not estimated). $\Delta t_{\text{solver}}$ is set by the *acoustic* speed while the flow evolves at the convective one. Three routes out — quasi-steady gas, much shorter episodes, or implicit/low-Mach preconditioning — are in [[impl-atlas-0.1-phase0-scope-and-data]] §3.8, and the choice changes what Atlas learns.

### Model dimensions (Atlas 0.1 reference config)

| Quantity | Value |
|---|---|
| $d_{\text{model}}$ | 256 |
| Backbone layers per expert $L$ | 6 |
| Attention heads $H$ | 8 ($d_{\text{head}}=32$) |
| Patch size (gas agents) | $8\times8$ cells |
| Patch size (solid shell `c`) | $8\times4$ cells |
| Total tokens (all agents) | $\approx 1{,}100$ (**1,114 as built**) |
| Approx. total learned params | 15–25 M (**8.35 M** scaffold-only; +≈4.7 M per learned expert) |
| Message-passing layers | **4** — the declared graph's diameter; distinct from $L$ above |

Per-agent grid and token counts are given in [[impl-atlas-0.1-phase1-scaffold]].

### Nondimensionalization

Every agent's fields are nondimensionalized before tokenization, per [[pfm-interface-design]]'s contract. Reference scales are **per-agent, per-episode constants** stored alongside the data (Phase 0), never recomputed inside the model:

$$\hat\rho=\rho/\rho_{\text{ref}},\quad \hat u = u/u_{\text{ref}},\quad \hat p = p/(\rho_{\text{ref}}u_{\text{ref}}^2),\quad \hat T = T/T_{\text{ref}},\quad \hat t = t\,u_{\text{ref}}/L_{\text{ref}}$$

Conditioning scalars passed to each expert: $\mathrm{Ma}$, $\mathrm{Re}$, $\mathrm{Pr}$, $\gamma$, $p/p_\infty$ (gas agents); Biot number, $\alpha\Delta T$, $E/\sigma_{\text{ref}}$ (solid agent).

---

## Scope reductions for 0.1 (each deferred, none rejected)

Per [[training-and-bootstrap-atlas-0.1]], with the trigger for revisiting each:

| Reduction | Deferred alternative | Trigger to revisit |
|---|---|---|
| Declared/geometric edges | RL edge instantiation ([[edge-generation-atlas-0.1]]) | Scenario 2 exists |
| MLP/fixed routing | RL expert gating ([[unet-hierarchy-atlas-0.1]]) | Scenario 2 exists |
| 2-level U-Net | Full 4-level hierarchy | Agent count ≫ 7 (moon base) |
| Imposed flux matching | Learned discovery ([[discovered-conservation-1.1]]) | Edges stop being declared |
| Planar 2D | Axisymmetric, then 3D | After M7 |

## Explicit non-goals for 0.1

Staging / graph-topology mutation ([[global-fields-and-topology-atlas-0.1]]); composability to unseen agent types; safety-critical abstention; turbulence closure beyond a fixed laminar/constant-eddy-viscosity model; real-engine quantitative fidelity.

---

## See Also

- [[00-atlas-0.1-overview]] — architecture thesis and design invariants
- [[case-study-rocket-ascent-2d-atlas-0.1]] — the phase plan this expands
- [[atlas-0.1-implementation-log]] — append-only build tracker
- [[00-implementation-plan]] — the analogous master plan for the parallel Noether 1.1 track
