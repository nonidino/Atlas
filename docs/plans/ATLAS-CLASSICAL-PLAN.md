# Atlas — the classical rocket simulation: implementation plan

**Status:** specification, written 2026-09-17, retargeted from the F1 car the
same day. Not started.
**This document is the strategic arc.** For the rung-by-rung specification of
Stage A, see **`ROCKET-POC-PLAN.md`**, which is authoritative for R0–R5.
**Compute:** `COMPUTE-PLAN.md` §1 and §7 still apply; **§2–§5 do not — the
rocket runs entirely on the development laptop.**

---

# 0. What this is, in one paragraph

**A rocket ascending, simulated as seven independent classical physics solvers
that Atlas couples entirely from their declared interfaces, with a verdict on
every seam.** The physics is not new and is not the contribution — six of the
seven solvers are already written and passing tests. The contribution is that
the coupling is *compiled from declarations with no hand-written glue*, that
every seam carries a bound or a named reason it does not, and that couplings
which would be silently wrong are **refused rather than run**. The rocket is the
stage; the verdict is the point.

---

# 1. Why the rocket replaced the car, in four numbers

Measured 2026-09-17, not assumed.

| | F1 car | rocket |
|---|---|---|
| geometry | 1,618,192 triangles; **436,926 boundary edges**; 78 of 80 parts not watertight | **3 profile functions, 16 numbers** |
| cells | 20–60 M (3-D RANS) | **72,448** |
| runs on | a rented 256 GB box | **this laptop** |
| solvers written | none | **seven, 54 tests passing** |

`C:\Users\Nauni\Downloads\f1_car.blend` is a render model, not CAD: no
parametric surfaces, sponsor decals parked 130 units off to the side, and ~8×
scale. Its repair was a project in itself, and 3-D RANS was past 15.4 GB of RAM
regardless. **Neither problem exists for a body of revolution defined by
analytic curves.**

**Nothing on the critical path was lost.** `L2/InterfaceMotion` — the rule that
refused the car's ride height and wheel rotation — is the same rule that refuses
the rocket's combustion front and plume boundary. Solving it once serves both,
and Stage C is where it happens.

---

# 2. What already exists

## 2.1 The physics — build repo, verified today

`C:\Users\Nauni\physics-foundation-model\src\atlas\solvers\` — **54 tests pass
in 324 s.**

| module | what | serves |
|---|---|---|
| `compressible2d.py` | 2-D compressible NS, MUSCL + minmod + **HLLC**, internal (walls + heat-release source) and external (freestream) modes, Strang-split reaction | `a`, `b`, `e`, `d`, `f`, `g` |
| `thermostruct2d.py` | Q1 FEM, backward-Euler conduction + quasi-static plane-stress elasticity | `c` |
| `trajectory.py` | 3-DOF planar rigid body, RK4, written to be the runtime expert verbatim | the trajectory agent |
| `atmosphere.py` | US Standard Atmosphere 1976, 0–32 km | ascent |
| `riemann.py`, `thermo.py`, `grid.py`, `oracles.py` | flux, EOS, grids, verification | all |

**The combustion is already de-scoped and that is a feature.** Agent `a` is a
**progress-variable source**, not finite-rate chemistry — the one part of a
rocket that would have been harder than a car is already simplified and tested.

## 2.2 The geometry

Everything is generated from three profiles and 16 numbers:

$$h_{\text{in}}(z),\qquad h_{\text{out}}(z) = h_{\text{in}}(z) + t_{\text{shell}},
\qquad h_p(z)$$

The interface curve and the agent mask are computed from the **same function**,
so they cannot drift, and `domains.py` asserts the partition **at import**.

## 2.3 The graph

`atlas/cases/rocket.py` — seven agents, seven typed edges, no `ROT` or `ELEC`.
Clocks $10^{-3}$ / $5\times10^{-3}$ / $5\times10^{-2}$ s, **spread 50:1**.

## 2.4 What the compiler says today

`python -m atlas rocket`, run 2026-09-17: **`refuse`, 10 refusals across 2
rules**, 49 decertifications across 21.

- **`L2/InterfaceMotion`** ×9 — combustion front (`solution_dependent`) and
  plume boundary (`prescribed`). **The one genuine research hole.**
- **`L7/R9`** ×1 — needs the time-integrated flux. **CS-11 closed this bound
  already** (W90); `boundary_response_integrated` is an existing field.

The 49 decertifications are **missing declarations, not missing science**.

## 2.5 The gap that is real work

**`rocket.py` is wired to seeded random matrices.** Every `boundary_response` is
a synthetic `linear_response(A)`. **No rocket physics has ever been run through
Atlas.** That is Stage A, rung R0.

---

# 3. The arc

| stage | what | tiers | on the critical path? |
|---|---|---|---|
| **A** | **The PoC** — R0–R5 (`ROCKET-POC-PLAN.md`) | 32–44 | **yes** |
| **B** | External solvers and scale | 25–35 | **no — deferred, see §4** |
| **C** | **`L2/InterfaceMotion`** — the research | 12–20 | yes, after A |
| **D** | Staging, and whatever the raise funds | — | no |

## Stage A — the PoC (authoritative spec: `ROCKET-POC-PLAN.md`)

| rung | what | tiers | gate |
|---|---|---|---|
| **R0** | seven real `boundary_response` implementations | 10–14 | a probe returns $\beta$, the declared null-space dimension, and $\omega$ |
| **R1** | declare what is known | 4–6 | 49 decertifications fall to a named residue |
| **R2** | multirate, time-integrated | 3–4 | `L7/R9` clears at the 50:1 ratio |
| **R3a** | static interfaces (de-scope) | 2 | **the graph compiles** |
| **R4** | march the ascent, 0–32 km | 5–8 | conserves what the joins declare |
| **R5** | the demo | 8–10 | PoC 3's dashboard re-pointed |

**≈ 2–3 weeks at the observed rate** (75 tiers in 18 days).

Two de-scopes, recorded so they are not later mistaken for oversights:
**single stage** (staging trips `E1 -> TopologyEvent`, which refuses without a
ledger), and **frozen burn state with a fixed plume boundary** (R3a). What still
moves is the trajectory — altitude, Mach, dynamic pressure — and that is the
dynamics the demo shows.

> **Say the second de-scope out loud in the demo.** *A moving interface silently
> invalidates a cached operator, which is the silent-wrongness class, and no rule
> exists — so we refuse it rather than run it.* **The refusal is the product.**

## Stage B — external solvers and scale *(deferred, and that is the change)*

The F1 plan put preCICE, OpenFOAM and SU2 on the critical path. **For the rocket
they come off it**, because the reason for them was scale: 3-D RANS on a car
needed an industrial solver and a cluster. At 72,448 cells with the physics
already written, the rocket needs neither.

**What survives, and is still worth doing after the PoC:**

1. **One external solver through `conformance.py`.** The README's central claim —
   *zero hand-written coupling code* — has only ever been tested against solvers
   written inside this project. SU2 first (LGPL, discrete adjoint, lighter build).
   **A failure here is a stopping rule, not a setback.**
2. **preCICE as an execution backend**, never as a replacement. Atlas's refusals
   happen at *compile* time on declarations, upstream of any execution engine.
   Certification stays offline ($\dim M$ solver calls per seam per regime);
   execution goes online.
3. **The homework, before any backend is designed:** preCICE's IQN-ILS builds a
   low-rank approximation of the **interface Jacobian** from coupling
   iterations, which is structurally what `probe_block` assembles. Either that is
   a free head start or there is a reason it is not, and which one must be
   settled first.
4. **`I1`, conservative-vs-consistent mapping**, declared **per port type** — a
   shipped preCICE feature and this vault's own listed gap, still "defaulted per
   edge, by accident."

**The adjoint is why B is not a replacement for A.** preCICE and OpenFOAM are not
differentiable, and differentiability is the enabling property of the design-
search story. Keep two backends: in-process (differentiable, small) and preCICE
(scale). "Differentiate the fast path, verify on the slow one" is honest and is
how surrogates are legitimately used.

## Stage C — `L2/InterfaceMotion` *(the research)*

The single rule refusing both targets, and the one thing here with no prior art
adjacent to it. **The named hole already states its own price**, so the
experiment is specified rather than open-ended:

$$\text{required: re-probe count},\quad \text{operator drift},\quad
\text{unaccounted power fraction}$$

Doing it **after** Stage A is the right order: it is far easier to motivate — and
to fund — when there is a running rocket that visibly refuses to move its own
combustion front.

## Stage D — not scheduled

Staging (`TopologyEvent`, needs a state map *and* a ledger); three dimensions;
automatic seam placement over a march (CS-9★ passed its one-interval gate at
$0.2685\times$ and **reversed when marched**, peaking at $1.59\times$ worse —
W123). Each is a real result waiting to happen. None blocks the PoC.

---

# 4. Compute

**All of Stage A runs here. No rental, no GPU.**

| | |
|---|---|
| rocket, total cells | **72,448** |
| PoC 3 car (already marching at 1.4 s/step here) | 204,599 unknowns |
| build-repo solver tests | 54 passed in **324 s**, on this laptop |

`COMPUTE-PLAN.md` §3's RAM wall was a property of 3-D RANS on a car and **does
not apply**. Its §7 operating rules still do: persist state every step, never
pipe a long run to `tail`, check `BatteryStatus` before quoting a timing, and
**quote ratios rather than wall-clock** (a ms/step figure in these docs was once
wrong by $4\times$ within hours).

Stage B, if it ever needs scale, wants **one large single-node CPU box** — not a
GPU marketplace and not a cluster. Keep the vast.ai key for training neural
experts after the raise.

---

# 5. Risks

| risk | mitigation |
|---|---|
| **The seven `boundary_response` implementations are harder than they look** | `thermal_seam.py::GasAgent` already wraps `Compressible2D` with a real Dirichlet wall channel. Start with the shell `c`: linear, quasi-static, exact response |
| **Probing a shock-capturing solver** gives a non-smooth response | measure it — `ProbeBudget.fd_step` defaults from the noise floor for this reason. **A non-smooth response at a seam crossing a shock is a finding either way** |
| **`probe_base` defaults to zeros** | on a thermal port that is probing at **0 K**; it moved $\beta$ from $4.8062$ to $0.3807$ on one seam (W74). Set it to the operating point on every thermal port |
| **`governing_family` cannot be verified and *promotes*** | a false label attributes $\tau$ under a hypothesis that does not hold, with no error and no failed gate. Name it in the certificate |
| **The static de-scope reads as a dodge** | §3 — say it out loud; the refusal is the product |
| **The build repo is a second checkout that can drift** | it is loaded under a private name (`atlas_build_solvers`) because both repos have a top-level `atlas`. **Pin its commit in the record** |

---

# 6. What Stage A achieves

> **A rocket ascending, simulated as seven independent physics solvers —
> combustion chamber, nozzle, plume, airframe shell, external flow, trajectory —
> that Atlas coupled entirely from their declared interfaces, with no
> hand-written coupling code anywhere.**
>
> **And a verdict on every seam between them: which couplings carry a proven
> error bound, and which the framework refuses to run because they would be wrong
> without telling you — which is the thing preCICE, FMI and every other coupling
> tool cannot do.**

The funding ask that follows is for the neural half: experts trained **to this
spec** — bounded receptive field, arbitrary non-rectangular geometry, one per PDE
family, all conformant — which the donor survey found do not exist, because every
continuum-fluid model with public weights is globally receptive and therefore
uncertifiable as a local decomposition.

**One number to carry into that conversation.** CS-8's substitution of a real
pretrained expert refused at all 55 cells, with $\lVert\Delta\rVert/\beta$ in
$[1.00254,\ 1.38952]$ — **the minimum a quarter of one percent from admitting**,
using a model trained for none of this.

---

# 7. Immediate next steps

1. **R0 rung 1 — the shell `c`.** `thermostruct2d` is linear and quasi-static, so
   its response is exact, and `thermal_seam.py` has already probed this solver.
2. **In parallel, R1's cheapest half** — declare `time_discretization` and
   `stencil_radius` on all seven agents. Both are known from the solver source,
   and they lift two whole decertification families.
3. **Re-run `python -m atlas rocket`** and record the verdict movement as this
   PoC's first measurement.

**Still open and unaffected by the pivot:** PoC 3's three blockers (W293, P1–P7
re-run, W275) and **36 unpushed commits**.
