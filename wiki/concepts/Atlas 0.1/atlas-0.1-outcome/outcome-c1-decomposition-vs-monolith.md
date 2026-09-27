# Outcome C1 — classical domain decomposition against the full-domain solve

**Type:** Outcome page — **claim audit** (folder: `Atlas 0.1/atlas-0.1-outcome/`)
**Status:** compiled 2026-09-26 from existing records; no new measurement.
**Verdict (revised 2026-09-26, Tier 89):** **The owner's target is demonstrated, for one mechanism.** On a wind farm with 5 or more rotors, classical decomposition run in parallel is $3.4$–$5.7\times$ faster than the full-domain solve (5 to 21 rotors, two timing draws) and $2.1$–$3.7\times$ at 36. It costs $2.3$–$8.3\%$ of farm power ([[decomposition-speed-by-rotor-count]]). Run serially it is never faster, and it is never more accurate than a same-discretization monolith. The rocket still has no full-domain comparison.
**The target, as the owner framed it:** *when there are a large number of rotors, the decomposition works faster.* §3.3 is that measurement.
**Hub:** [[00-atlas-0.1-outcome]]
**Sources:** [[case-study-rocket-bc-seam-atlas-0.1]] §17–§20 · [[rocket-episode-pickup]] · [[poc1a-frozen-expert-results]] §7–§8 · [[atlas-proof-of-concept-1]] §10–§11 · [[learned-contribution-kill-tests]] §2 · [[case-study-scaling-ladder-atlas-0.1]] · [[poc1-retrospective-and-hybrid-roadmap]] · [[poc3-racelab-car-union]]

---

## 1. The claim, and the experiment that would test it

**As the proposal wants it:** *classical domain decomposition solves engineering problems faster and more accurately than a full-domain solve, and the rocket shows it.*

**Intuition.** Domain decomposition (DD) cuts a problem into pieces, solves each piece with its own solver, and makes the pieces agree where they meet. A monolith solves the whole domain at once. The two can be compared only if a monolith of the **same discretization** exists. That comparison isolates the price of the cut, because everything else is held fixed.

**Formally.** Let $u_M$ be the monolith's state and $u_D$ the decomposed column's state from the same initial data, same cell size, same time step. The quantity [[master-error-bound]] calls the composition error is

$$e_{\text{comp}}(t) \;=\; \lVert u_D(t) - u_M(t) \rVert .$$

A same-discretization monolith is the best a DD column can hope to match, so $e_{\text{comp}}\ge 0$ is a **cost**, not a gain. DD can be *more* accurate than a monolith only if the pieces use something the monolith cannot: a finer grid where it matters, a body-fitted grid, or a better model per region. **Comparing that honestly needs a monolith at equal total cost, not at equal cell size.** No page in this vault has run that comparison.

---

## 2. The rocket: what was built, and what it demonstrates

### 2.1 The graph

The PoC 4 rocket ascent episode ([[case-study-rocket-bc-seam-atlas-0.1]], CS-21) is a **seven-agent coupled model** whose every agent steps a real solver from the build repo, imported unmodified, on the build repo's own mesh:

| agent | role | family |
|---|---|---|
| `a`, `b` | combustion zone and chamber, to the throat | reacting compressible flow |
| `e` | nozzle | reacting compressible flow (curvilinear) |
| `f` | exhaust plume | compressible Navier–Stokes, exhaust-gas model |
| `d`, `g` | external air around the airframe, and the aft region the plume crosses | external compressible flow |
| `c` | airframe shell | conduction + quasi-static elasticity |
| trajectory | the vehicle as a rigid body | 3-DOF ODE (RK4) |

The operating point is the build repo's corner case `underexpanded_max`: $p_c = 8$ MPa, $T_c = 3400$ K, launched at $25$ km and Mach $3.5$. The episode is coarsened $4\times$ per dimension and marched 29 macro-steps of $5$ ms ($0.145$ s, about $6.5\times10^{5}$ CFL sub-steps).

### 2.2 Its correctness record (Tier 88 + W343, the current build)

This is the strongest statement in the record about a classical decomposed engineering model, and it is about **correctness, not speed**.

| account | reading | status |
|---|---|---|
| every block's content against its boundary inflows (mass, both momenta, energy), every step | $1.2\times10^{-13}$ | exact to round-off |
| mass through walls | $3.5\times10^{-17}$ of $\dot m\,\Delta t$ | exact |
| shell energy against its heat input | $1.1\times10^{-11}$ of stored energy | exact |
| injector delivery over declared $\dot m^{\ast}$ | $1.000000$ at every grid | exact after W337 |
| chamber pressure over declared $8$ MPa | $0.980$ | |
| thrust over ideal planar nozzle theory, given the inflow | $\mathbf{0.992}$ | validated to about $1\%$ |
| the two routes to the thrust (exit plane, wall forces) | agree to $2.5\times10^{-4}$ | |
| engine heat: gas lost against shell gained | $600.35$ against $600.49$ J/m per side per step | $2.4\times10^{-4}$ |
| throat seam b\|e | $+2.8\times10^{-4}$ of the mass flow, shrinking with the coupling step | a lag, after W340 |
| re-run on a second machine (Windows/MKL against Linux/OpenBLAS) | bitwise in altitude, attitude, $\omega$, mass | reproducible |
| Tier 86's 17 registered gates | 17 of 17 | pass |

**Still wrong, with named causes** ([[case-study-rocket-bc-seam-atlas-0.1]] §19.10, §20.8):

- **W341.** The plume's seams d\|f and g\|f create $23$–$38\%$ of the mass flux through them, because block `f` runs one gas model for air and exhaust alike. This is a modelling limit, not a coding slip.
- **W342.** Every wall flux (engine heat, skin friction, airframe heating, sign included) doubles per grid halving and has not converged. There is no wall model.
- **W344.** The d\|g seam leaves a $1.13\%$ floor across non-matching grids in a subsonic band, even coupled both ways.

**So the episode is a correct demonstration of coupled classical machinery.** Its bookkeeping is exact and its failures are located. The rocket page itself says it is **not** a quantitative prediction of this vehicle.

### 2.3 Why there is no full-domain comparison

- **No monolith exists for it.** The rocket page records this at its R1 ledger (§6.3): the families differ block to block, so *"no monolithic reference covers both sides"*. A reacting chamber, a two-gas plume, a conducting and bending shell and a rigid-body ODE are not one discretization of one equation.
- **Its cost was never compared with anything.** The Tier 88 march took $2879$ s on a rented 64-core box. The two-way re-march took $2384$ s on a 128-thread box. The full-resolution engine alone took $10\,388$ s for 6 steps. These are costs of the decomposed model, with no alternative timed beside them.

---

## 3. Where a monolith does exist: 2-D incompressible flow

Every decomposition-against-monolith measurement in the vault is on the wind-farm geometry, with `reference.WindowNS` windows against `RectangularNS` on the whole domain, same cell size and same viscosity.

### 3.1 Accuracy: decomposition costs accuracy

| measurement | decomposed classical column | monolith | source |
|---|---|---|---|
| farm power, K12 start layout | $2.3142$ | $2.5544$ | **$-9.4\%$** · [[poc1a-frozen-expert-results]] §7.1 |
| farm power, K25 start layout | $6.4379$ | $7.8536$ | **$-18.0\%$** · same |
| rms velocity difference after 120 steps, 2 / 6 / 12 windows | $0.0891$ / $0.0937$ / $0.0830$ | $0$ by definition | [[learned-contribution-kill-tests]] §2.2 |
| composed-defect growth exponent against the monolith, $N=1\to24$ | $+0.851$, 95% CI $[0.748, 0.954]$ | — | [[case-study-scaling-ladder-atlas-0.1]] §3 |
| same, with the physics held fixed (one turbine at every rung) | $+0.160$ | — | same, §4 |

The error is **sub-linear in the number of interfaces**: a $68\times$ increase in interfaces buys a $2.2\times$ increase in composed defect once the physics is held fixed. That is the result Claim B needs. **But it is a cost that grows slowly, not an accuracy gain.**

And one instability the monolith does not have. A decomposed column whose windows each solve their own pressure Poisson equation ("embedded" elliptic part) leaves its velocity band between macro-steps $70$ and $80$ and is not finite by $82$. The monolith holds. Exposing the elliptic part to the composition layer and applying it once fixes it ([[case-study-scaling-ladder-atlas-0.1]] §5). The mechanism is exact: a partition-of-unity blend of divergence-free fields is not divergence-free,

$$\nabla\cdot\Big(\sum_i \chi_i u_i\Big) \;=\; \sum_i \nabla\chi_i\cdot(u_i - w)\quad\text{for any } w,$$

so the assembly creates divergence exactly where two local solves disagree.

### 3.2 Speed: a wash, and it changes sign between machines

| measurement | decomposed / monolith | source |
|---|---|---|
| cost per macro-step, 2 / 6 / 12 windows, same process | $0.932$ / $1.31$ / $1.02$ | [[learned-contribution-kill-tests]] §2.3 |
| wall time for 120 steps, 2 / 6 / 12 windows | $19/21$, $78/76$, $419/438$ s | same, §2.2 |
| PoC 1a, six windows, dev box | $1.54\times$ **slower** | [[atlas-proof-of-concept-1]] §10 |
| PoC 1a demo on a Mac | $10$–$25\%$ **slower** | [[poc1-retrospective-and-hybrid-roadmap]] §1 |
| PoC 1a demo, K12, exposed classical column, dev box | $3.98\times$ **faster** | [[atlas-proof-of-concept-1]] §11 |
| classical per-agent cost, $N=2\to24$ | rises $3.84\times$ (memory hierarchy; work per agent $O(1)$) | [[case-study-scaling-ladder-atlas-0.1]] §7 |

The $3.98\times$ is one machine and one rung. Its source page marks the suspected mechanism as an **[AI Inference]**: the demo's classical column is the *exposed* solver, which does no per-window Poisson solve. It has no worklist row yet.

### 3.3 By rotor count, with the mechanisms separated (Tier 89, W346)

The owner's target, measured on CS-7's tilings extended to 80 windows, on mains power, same process, the monolith and each decomposed arm interleaved ([[decomposition-speed-by-rotor-count]]):

| rotors | cells | monolith, µs per cell | serial decomposition / monolith | **parallel decomposition / monolith, two draws** | farm power vs monolith |
|---|---|---|---|---|---|
| $1$ | $30\,720$ | $1.35$ | $0.95$ | $1.57$–$1.61$, slower (one draw; range over 2–14 threads) | $-4.3\%$ |
| $3$ | $84\,480$ | $2.15$ | $1.08$ | $0.78$ at 2 threads, $1.31$–$1.55$ at 4–14 (one draw) | $-4.1\%$ |
| $5$ | $163\,328$ | $8.79$ | $1.03$–$1.05$ | $\mathbf{0.186}$–$\mathbf{0.286}$ ($3.5$–$5.4\times$ faster) | $-2.3\%$ |
| $12$ | $319\,232$ | $8.58$ | $1.07$–$1.18$ | $\mathbf{0.176}$–$\mathbf{0.247}$ ($4.0$–$5.7\times$) | $-6.8\%$ |
| $21$ | $627\,456$ | $8.75$ | $1.12$–$1.13$ | $\mathbf{0.213}$–$\mathbf{0.230}$ ($4.3$–$4.7\times$) | $-8.3\%$ |
| $36$ | $1\,036\,032$ | $8.68$ | $1.10$–$1.25$ | $0.267$–$0.467$ ($2.1$–$3.7\times$) | not marched |

**What makes it faster is parallelism over cache-sized windows, and nothing else measured.**
- The monolith's cost per cell rises fourfold between 3 and 5 rotors, when its working set leaves the cache.
- The serial decomposed column falls off the same cliff, because it steps every window as one array.
- Spread across threads (bit for bit the serial column's answer, asserted at 24 of 24 rung-and-thread pairs), each thread's windows stay in cache.
- Local time stepping saves only 5–8% of sub-steps, because a wind farm's windows all see roughly the freestream.

**[AI Inference]:** parallelism is the textbook reason domain decomposition exists, so this is a demonstration of a known effect in this framework, not a new one. A monolith parallelised by other means (a threaded stencil library, MPI) would itself be a decomposition underneath. The proposal should present it that way.

---

## 4. What C1 can honestly become

**[AI Inference]:** the record now supports two claims, and the proposal can make both.

> *Speed at scale (measured, Tier 89).* On a wind farm of 5 to 21 rotors, a classical domain decomposition run on several cores is $3.4$–$5.7\times$ faster than the undivided solver of the same discretization, and $2.1$–$3.7\times$ at 36 rotors. Its answer is bit for bit that of the same decomposition run serially, and its farm power is within $2.3$–$8.3\%$ of the undivided solver's. The gain comes from parallelism over cache-sized windows, and appears once the undivided domain no longer fits in cache.

> *Composability (measured, Tiers 76–88).* Classical decomposition lets independently written solvers of different physical families be coupled into one model that conserves what it exchanges to round-off, and whose errors are measured and traced to causes. On a single-family problem, the price of the cut is a composition error that grows sub-linearly with the number of interfaces.

**What neither claim says.** Decomposition is not more accurate than a same-discretization monolith, and serial decomposition is not faster. The rocket's speed has never been compared with anything. It is also the premise the learned-expert half of the proposal needs: a framework into which a better expert can be dropped ([[outcome-c4-modular-multiphysics]]).

One piece of the record points toward "decomposition buys accuracy", but it is not a monolith comparison. On the car, curved body-fitted grids overlapping a Cartesian background (an overset decomposition) close the turbine join's receiving residual to $3.40\times10^{-4}$, where the porous-body Cartesian column gave $0.0292$ ([[poc3-racelab-car-union]]). That compares two decomposed columns, not a decomposition against a whole-domain solve.

---

## 5. What was NOT done, named

1. **No full-domain run of the rocket, and no equal-cost comparison.** The closest feasible test **[AI Inference]** would take the gas-only blocks (`a`, `b`, `e`, `f`, `d`, `g`, all compressible), which share one solver family. It would solve them as one block on the union mesh with a single gas model, and compare cost and seam-free accuracy against the decomposed episode at equal wall time. W341's two-gas limit would have to be declared on both sides.
2. **No equal-cost comparison anywhere.** Every monolith comparison holds the cell size fixed, which by construction cannot show DD winning on accuracy. An equal-cost comparison (a refined region inside a decomposed column against a uniformly refined monolith of the same total cost) was never run.
3. **The $3.98\times$ exposed-column speedup has no worklist row** and was measured on one machine at one rung. W346 reads the serial exposed column at $1.03\times$ on the same laptop at the same size. **[AI Inference]:** the demo's column runs in torch, whose kernels are multi-threaded, so its figure is consistent with parallelism. Not re-measured.
5. **The rotor-count speedup on another machine, and with processes instead of threads** (W346 §7). Python threads cost $1.6\times$ below 5 rotors, and the cache cliff's position is this laptop's. A many-core box, or a compiled kernel, would move both.
4. **The multi-clock question is unanswered.** A decomposed model can sub-cycle each block at its own clock. The vehicle union's clocks span $400\times$ ([[case-study-vehicle-march-atlas-0.1]]), and no march compares multirate DD against a single-clock monolith.

---

## See Also

- [[00-atlas-0.1-outcome]] · [[outcome-evidence-ledger]] · [[outcome-demos-and-artifacts]]
- [[case-study-rocket-bc-seam-atlas-0.1]] — the rocket's full record, §17–§20 especially
- [[rocket-episode-seam-audit]] — how the episode's geometry and seams were found broken and rebuilt
- [[atlas-and-standard-dd-theory]] — where Atlas sits among the classical DD families
- [[composition-error-theory]] · [[master-error-bound]] — what the composition error is, formally
