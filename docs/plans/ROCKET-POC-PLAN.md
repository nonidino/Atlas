# PoC 4 — the rocket: implementation plan

**Status:** specification, written 2026-09-17. Not started.
**Supersedes:** `ATLAS-CLASSICAL-PLAN.md` (the F1 route) as the near-term target.
**Companion:** `COMPUTE-PLAN.md` §2 — **superseded for this target: the rocket
runs entirely on the development laptop. No rental.**

---

# 0. Why this is the right target, in four numbers

| | F1 car | rocket |
|---|---|---|
| geometry definition | 1.62 M triangles, **436,926 boundary edges**, 78 of 80 parts not watertight | **three profile functions and 16 numbers** |
| total cells | 20–60 M (3-D RANS) | **72,448** |
| where it runs | rented 256 GB box | **this laptop** |
| physics solvers written | none (adoption planned) | **all seven, 54 tests passing** |

The car's blocker was never the physics. It was that a render-quality mesh is
not CAD, and that three dimensions put the problem past 15.4 GB of RAM. **The
rocket dissolves both**: its geometry is analytic, and at 72,448 cells it is
*smaller than the PoC 3 car already marching at 1.4 s a step on this machine.*

---

# 1. What already exists, verified 2026-09-17

## 1.1 The physics, in the build repo

`C:\Users\Nauni\physics-foundation-model\src\atlas\solvers\` — **54 tests pass
in 324 s**, run today.

| module | what it is | serves |
|---|---|---|
| `compressible2d.py` | 2-D compressible NS finite volume. MUSCL on primitives, minmod, **HLLC** flux (Rusanov fallback). Two modes: *internal* (walls + heat-release source) and *external* (freestream). Strang-split reaction source | `a`, `b`, `e` internal; `d`, `f`, `g` external |
| `thermostruct2d.py` | Q1 FEM: backward-Euler conduction + quasi-static plane-stress elasticity, free-body modes removed | `c`, the airframe shell |
| `trajectory.py` | 3-DOF planar rigid body, RK4, **written to be the runtime expert verbatim** | the trajectory agent |
| `atmosphere.py` | US Standard Atmosphere 1976, 0–32 km, Sutherland viscosity | ascent conditions |
| `riemann.py` | Rusanov + HLLC + the exact solver as an oracle | the flux |
| `thermo.py`, `grid.py`, `oracles.py` | equation of state, grids, verification oracles | all |

**The combustion is already de-scoped and that is good news.** Agent `a` is not
finite-rate chemistry — it is a **progress-variable source, Strang-split**. That
is a modelling decision someone already made and tested, and it removes the one
part of a rocket that would have been genuinely harder than a car.

## 1.2 The geometry — the whole reason to do this

`src/atlas/geometry/contours.py`: *everything geometric is generated from three
profiles.*

$$h_{\text{in}}(z) \quad\text{(gas cavity)}\qquad
  h_{\text{out}}(z) = h_{\text{in}}(z) + t_{\text{shell}}\qquad
  h_p(z) \quad\text{(plume half-width)}$$

and the seven interface curves are slices of those three plus three constant-$z$
planes. The whole vehicle is 16 numbers:

```
chamber_halfheight 0.10   throat_halfheight 0.04   exit_halfheight 0.12
z_converge [0.30, 0.40]   z_diverge [0.40, 0.70]   z_throat 0.40
shell_thickness 0.008     shell_z [-4.0, 0.70]     plume_z [0.70, 3.0]
farfield_halfwidth 1.5    atmos_stretch_ratio 1.08 ...
```

**There is no CAD step, no defeaturing, no mesh repair, and no watertightness
problem — because the curve and the agent mask are computed from the same
function and cannot drift.** `domains.py` asserts the partition at *import*, not
in a test nobody ran.

## 1.3 The graph, in this vault

`atlas/cases/rocket.py` — seven agents, seven typed edges, no `ROT` or `ELEC`,
which is a correct statement about a solid-propellant vehicle.

```
a  reacting     48 x  80     a-b  heat, pressure
b  confined    112 x  80     e-b  heat, pressure
e  confined    120 x  96     b-c  heat, stress
c  solid       232 x  16     c-d  stress, heat, fluid
d  external    184 x  96     e-f  conservation
f  free jet    152 x  80     d-g  conservation, heat, fluid
g  external    152 x  96     g-f  heat, fluid
                 = 72,448 cells
clocks: a,b,e 1e-3 s | d,f,g 5e-3 s | c 5e-2 s | macro 5e-2 s  (spread 50:1)
```

---

# 2. What the compiler says today

Run on 2026-09-17: **`verdict: refuse`, 10 refusals across 2 rules, 49
decertifications across 21.**

The docstring in `rocket.py` says the rocket "should not be the next build"
because its refusals were research. **That was written before CS-9 through CS-14.
Most of it is no longer true.** The refusals today:

| rule | count | what it is | status |
|---|---|---|---|
| **L2/InterfaceMotion** | 9 | the combustion front `a.b` (`solution_dependent`) and the plume boundary `f.e`, `f.g` (`prescribed`) | **the one genuine research hole** — §4 R3 |
| **L7/R9** | 1 | multirate needs the **time-integrated** flux, not pointwise | **solved. CS-11 closed W90**; `boundary_response_integrated` is already a field |

And the 49 decertifications are **almost all missing declarations, not missing
science**: no `storage`, no `validity` predicate on any of the 7 agents,
`time_discretization=unknown` on all 7, no `lambda_ref` at the multiphysics
seams, no partition of unity.

**`time_discretization=unknown` is the clearest example of the gap being
paperwork.** These solvers are known — backward Euler for conduction, explicit
SSP-RK2 for the gas. Declaring it lifts both `L4/R2b/W46` and `L2/R10/halo`.

## 2.1 The gap that is real implementation

**`rocket.py` is wired to stubs.** Every agent's `boundary_response` is a
synthetic `linear_response(A)` on a seeded random matrix. The graph is a
declaration exercise; **no rocket physics has ever been run through Atlas.**

Connecting the seven real solvers is the bulk of the work, and it is not
speculative: **`thermal_seam.py` already wraps `Compressible2D` with a real
`boundary_response`**, and `window_ns.py` is a 757-line worked example of the
same pattern. The template exists.

---

# 3. Two de-scopes, taken deliberately and recorded

**De-scope 1 — single stage.** Staging trips `E1 -> TopologyEvent`, which
refuses without a state map *and* a ledger. A PoC does not need staging.
*Recorded so it is not later mistaken for an oversight.*

**De-scope 2 — quasi-steady interfaces (R3a).** Declare `motion_class=STATIC`:
a frozen burn state and a fixed plume boundary, re-probed on a schedule. The
graph then compiles. **What still genuinely moves is the trajectory** — altitude,
Mach, dynamic pressure, atmospheric state — and that is the interesting dynamics
and the thing the ascent demo shows.

> **Say this out loud in the demo rather than hiding it.** A reviewer will ask
> why the combustion front is frozen, and the honest answer is strong: *a moving
> interface silently invalidates a cached operator, which is the silent-wrongness
> class, and no rule exists — so we refuse it rather than run it.* **The refusal
> is the product.** A framework that quietly ran it would be the one with the
> problem.

---

# 4. The rungs

Estimates in tiers at this project's observed rate (75 tiers in 18 days).

| rung | what | tiers | gate |
|---|---|---|---|
| **R0** | **Real experts** — seven `boundary_response` implementations | 10–14 | a probe on one seam returns $\beta$, a null space of declared dimension, and $\omega$ |
| **R1** | **Declare what is known** | 4–6 | decertifications fall from 49 to a named residue, each with a reason |
| **R2** | **Multirate, properly** | 3–4 | `L7/R9` clears; CS-11's bound quoted at the 50:1 ratio |
| **R3a** | **Static interfaces** (de-scope) | 2 | **the graph compiles: `admit` or `admit-uncertified`** |
| **R4** | **March the ascent** | 5–8 | 0–32 km, conserving what the joins declare |
| **R5** | **The demo** | 8–10 | PoC 3's dashboard and bundle, re-pointed |
| | **PoC total** | **32–44** | **≈ 2–3 weeks** |
| *R3b* | *InterfaceMotion, solved* | *12–20* | *the research rung — see §6* |

## R0 — the seven experts *(the critical rung)*

Replace each stub with a real response: impose the prolonged boundary trace,
step the solver, return the conjugate half.

**Order, and why.** Start with **`c`, the shell** — `thermostruct2d` is linear
and quasi-static, so its response is exact and cheap, and `thermal_seam` has
already probed this exact solver. Then **`b`** (internal gas, and `thermal_seam`
wraps `Compressible2D` already). Then `a`, `e`, `d`, `f`, `g`. The trajectory
agent is **lumped** ($\dim M = 1$) and is the cheapest of all — and it exercises
the field-to-lumped path W97 closed at CS-10.

**The trap to avoid, from this vault's own record:** `PORT_SPECS[THERM]` pairs
$(T,\ q_n/T)$ so effort $\times$ flow is a power, and **the probe linearises
about `probe_base`, which defaults to zeros.** For a temperature in kelvin that
is probing at 0 K, and on one seam it moved $\beta$ from $4.8062$ to $0.3807$
(W74). **Set `probe_base` to the operating point on every thermal port.**

**Gate:** one seam, probed, returning $\beta$, the declared null-space dimension,
and $\omega$ — with the probe cost recorded in solver calls *and* seconds.
**Control:** a deliberately wrong `response_half` on one side must be caught by
`L3/C9`, and the same error on *both* sides must not be (W66's three legs).

## R1 — declare what is known

`time_discretization`, `stencil_radius`, `elliptic_subsolve`, `storage`,
`validity`, `lambda_ref`, `response_half`, and a partition of unity for `L2/C2`.
Then run `conformance.py` against every agent.

**Expect three fields to resist**, and name them rather than assume them:
`validity` is falsifiable but not verifiable; `response_half` fails silently when
both sides are wrong; and **`governing_family` *promotes* — a false label makes
$\tau$ be attributed under a hypothesis that does not hold, with no error and no
failed gate.**

## R2 — multirate

Declare `flux_matching=TIME_INTEGRATED` and supply
`boundary_response_integrated` on each side. CS-11 measured this bound at first
order in the exchange interval, $1.43\times$–$2.18\times$ loose over $2000\times$
of clock ratio. **This graph's spread is $50{:}1$ — comfortably inside the range
the bound was measured over.**

## R4 — the ascent

`trajectory.py` was written to be the runtime expert verbatim; `atmosphere.py`
gives 0–32 km. The rigid body is the **bottleneck agent**: it integrates the net
force and torque every other agent produces.

**A convention this vault already pays for:** a reported sensitivity carries its
horizon or is refused — CS-10 found a **sign change** in
$\mathrm{d}(\text{downforce})/\mathrm{d}h_0$ with the rollout horizon. Any
trajectory sensitivity here carries its horizon from the first run.

## R5 — the demo

PoC 3's dashboard, per-window switch, telemetry and bundle builder are generic.
Re-point them: seven agents instead of fourteen windows, an altitude readout
instead of a ride-height knob, and the geometry knobs become **the 16 numbers**,
which are genuinely parametric in a way the car's triangle soup never was.

---

# 5. Compute — the whole point of the pivot

**It runs here. All of it.**

| | value |
|---|---|
| rocket, total cells | **72,448** |
| PoC 3 car, unknowns | 204,599 — already marching at **1.4 s/step** on this laptop |
| build-repo solver tests | 54 passed in **324 s**, today, on this laptop |
| GPU needed | **none** |
| rented compute needed | **none** |

`COMPUTE-PLAN.md` §3's RAM wall was a property of three-dimensional RANS on a car.
**It does not apply here.** The standing operating rules from that document still
do: persist state every step, never pipe a long run to `tail`, check
`BatteryStatus` before quoting a timing, and quote ratios rather than wall-clock.

Keep the vast.ai key for training neural experts later. Nothing in R0–R5 needs it.

---

# 6. What this is worth, and the research rung

The PoC deliverable is one sentence:

> **A rocket ascending, simulated as seven coupled physics experts with a verdict
> on every seam — and one coupling the framework refuses, with the reason.**

**R3b is the research, and it is now the same rung for both targets.**
`L2/InterfaceMotion` is what refused the F1 car's ride height and wheel rotation,
and it is what refuses the rocket's combustion front and plume boundary. Solving
it once serves both. The named hole already states its own price — it requires
**re-probe count, operator drift, and unaccounted power fraction** — so the
experiment is specified, not open-ended.

Doing the PoC first and R3b second is the right order: R3b is much easier to
motivate, and much easier to *fund*, when there is a running rocket that visibly
refuses to move its own combustion front.

---

# 7. Risks

| risk | mitigation |
|---|---|
| **The seven `boundary_response` implementations are harder than they look** | `thermal_seam.py` already wraps `Compressible2D`; start with the linear shell `c`, where the answer is exact |
| **Probing a shock-capturing solver** may give a noisy or non-smooth response | measure it — `ProbeBudget.fd_step` defaults from the noise floor for exactly this reason. **A non-smooth response at a seam crossing a shock is a finding either way** |
| **A shock crossing a seam** breaks the coupling | this is a genuine test and a good demo; report it whichever way it goes |
| **The static de-scope looks like a dodge** | §3 — say it out loud; the refusal is the product |
| **`governing_family` cannot be verified** | name it in the certificate; it is the security hole of any open expert standard and stating it is publishable |
| **The build repo is a second checkout that can drift** | it is already loaded under a private name (`atlas_build_solvers`) because both repos have a top-level `atlas`. Pin the commit in the record |

---

# 8. Immediate next steps

1. **R0 rung 1: the shell `c`.** Wrap `thermostruct2d` as a real
   `boundary_response`, following `thermal_seam.py`. One agent, exact linear
   response, and it proves the pattern.
2. **In parallel, R1's cheapest half:** declare `time_discretization` and
   `stencil_radius` on all seven agents. These are known from the solver source,
   and they lift two whole decertification families.
3. **Re-run `python -m atlas rocket`** and record the verdict movement as the
   first measurement of this PoC.

**Still open from the F1 plan and unaffected by the pivot:** PoC 3's three
blockers (W293, P1–P7, W275) and **36 unpushed commits**.
