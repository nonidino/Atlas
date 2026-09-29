# Outcome C4 — a modular multiphysics framework: declare, assign, run, swap

**Type:** Outcome page — **claim audit** (folder: `Atlas 0.1/atlas-0.1-outcome/`)
**Status:** compiled 2026-09-26 from existing records; no new measurement.
**Verdict:** **Largely demonstrated, with classical experts.** Graphs across five port types and more than five governing families are built from declarations, with no hand-written coupling code. Experts are swapped per window on a live screen. The learned half of "swap anywhere" works only in 2-D incompressible flow on uniform windows. "Fast *and* accurate" has not been shown at once.
**Hub:** [[00-atlas-0.1-outcome]]
**Sources:** [[atlas-implementation]] · [[end-to-end-architecture-spec]] · [[port-algebra-atlas-0.1]] · [[plug-in-composition-theorems]] · [[joining-seam-cost]] · [[case-study-vehicle-march-atlas-0.1]] · [[case-study-wing-fsi-atlas-0.1]] · [[case-study-cooling-loop-atlas-0.1]] · [[case-study-powertrain-atlas-0.1]] · [[poc2-frontwing-results]] · [[case-study-racelab-switch-atlas-0.1]] · [[poc3-racelab-dashboard]] · [[poc3-racelab-car-solids]] · [[case-study-rocket-bc-seam-atlas-0.1]]

> **2026-09-29: a showcase, not an input to this audit.** The workbench now turns a drawn case file into a compiled graph for eight classical families. Across eleven examples that covers four port types and all four coupling styles, and each example is run against the full domain in the page ([[showcase-gallery]]). These are showcase runs, one pass each, not research records, so this page's verdict is unchanged. What they add to it is the same finding in new places: the heated strip's volumetric bond has no port type, and R10 refuses iterated one-physics cuts that the runs measure as agreeing with the full domain.

---

## 1. The claim, as three testable parts

*"Create geometries, define equations and assign experts, set a few parameters, and the simulation runs fast and accurately; add, remove and substitute experts anywhere."*

| part | test | answer |
|---|---|---|
| **(a) declarative** | Is a new coupled model a set of declarations, with no coupling code written for it? | **Yes.** §2 |
| **(b) multiphysics** | Does it couple different governing families, and do the couplings conserve? | **Yes, across five port types.** §3 |
| **(c) substitutable** | Can an expert be swapped (classical ↔ learned) without touching the rest? | **Yes mechanically, and the framework refuses swaps it cannot certify.** Only one family has a learned option. §4 |

---

## 2. (a) Declarative: the compiler

**Intuition.** Each expert publishes a *capability record*: which equations it solves, on which geometry, which ports it exposes, what time step it runs at and where it is valid. A graph is those records plus port connections. A compiler reads the graph and, before anything runs, returns one of three verdicts per seam: **admit**, **admit-uncertified** or **refuse**. It names the rule behind each.

**As built** ([[atlas-implementation]]), `atlas/` at the vault root:

| layer | job | the refusal it owns |
|---|---|---|
| L1, L1.5 | capability record; routing | a record missing a field; an empty feasible set |
| L2 | decomposition, cut placement | motion; an unledgered mutation |
| L3 | port algebra, admissibility | no declared interface space; a non-power-preserving scale set |
| L4 | transmission operator (probed) | a null space of the wrong dimension |
| L5 | interface solve, accelerator | an empty operator; a degenerate axis |
| L6 | assembly | the partition-of-unity identity |
| L7 | time integration, multirate | R9 under multirate |
| L8, L9 | emit; typing of claims | a run with no envelope stamp; a metric quoted past its horizon |

- **The plug-in claim is an executable test**: a three-agent chain nobody wrote coupling code for compiles from capability records and port connections alone.
- **A join is $O(1)$ in declarations.** Joining three subsystems into a vehicle cost a constant number of declarations at 3, 4 and 5 families, and **zero per-pair declarations** ([[joining-seam-cost]]).
- **Scale of the code today:** 36 case modules in `atlas/cases/`, 86 test files. The full suite at Tier 89 (2026-09-27) passed 1950 tests, with none failing.

**The ports are the reason it scales.** Five effort–flow pairs whose product is power ([[port-algebra-atlas-0.1]]):

$$
\texttt{MECH}\,(\boldsymbol\sigma\!\cdot\!\mathbf n,\ \mathbf v),\quad
\texttt{ROT}\,(\tau,\ \omega),\quad
\texttt{THERM}\,(T,\ q_n/T),\quad
\texttt{ELEC}\,(\phi,\ \mathbf j\!\cdot\!\mathbf n),\quad
\texttt{ADVEC}\,(\dot m;\ h_0,\ Y_k).
$$

$K$ experts need $O(K)$ port implementations rather than $O(K^2)$ pairwise adapters. The global power residual $\mathcal R(t)$ is one diagnostic that reads across every case study. The vault's own audit ([[prior-art-and-novelty-atlas-0.1]]) records that effort–flow ports date to bond graphs (Paynter, 1959). What it credits as new is compiling admissibility from the declarations ([[poc2-novelty-audit]]).

---

## 3. (b) Multiphysics: every graph built

| graph | families and ports | what it showed | page |
|---|---|---|---|
| wind farm, wake array | incompressible flow + closed-form actuator disks (field ↔ lumped) | the architecture does not break the physics; composition error flat in $N$: $P_2/P_1$ $0.7663$ / $0.7588$ / $0.7581$ / $0.7584$ at $N = 2,3,5,8$ (8 → 26 agents, 15 → 63 interfaces) | [[results-n-sweep-wind-farm]] |
| scaling ladder, CS-7 | 1 → 24 windows | composed defect sub-linear in interfaces ($+0.160$ with physics held fixed) | [[case-study-scaling-ladder-atlas-0.1]] |
| ground effect, CS-10 | a moving interface; the first design parameter | an interface whose geometry is a function of the solution | [[case-study-ground-effect-atlas-0.1]] |
| brake thermal, CS-11 | multirate THERM | a bound for the multirate lag, as a function | [[case-study-brake-thermal-atlas-0.1]] |
| wing FSI, CS-12 | flow + structure, two-way, `MECH` on a surface | energy gate holds over 480 steps with no crossing | [[case-study-wing-fsi-atlas-0.1]] |
| front wing, PoC 2 | 8 agents, 9 seams: ride height and bending live at once, one 33-equation Newton system | the compiler's per-seam verdict is a static property of the declarations; the wing reaches `admit` at a fixed shape (Tier 39) | [[poc2-frontwing-results]] |
| cooling loop, CS-13 | conduction + four lumped `ADVEC` legs; the first directed cycle | the block's own first law closes to $3.6\times10^{-7}$ W against $1787$ W | [[case-study-cooling-loop-atlas-0.1]] |
| powertrain, CS-14 | `ROT` + `ELEC`: rotor, motor-generator, bus, battery, inverter | energy balance $1.9\times10^{-15}$, KVL $2.1\times10^{-17}$ | [[case-study-powertrain-atlas-0.1]] |
| learned pair, CS-17 | Poseidon-T + NeuberNet across `MECH` | two learned experts of different families compile; costs exactly additive | [[case-study-learned-pair-atlas-0.1]] |
| **vehicle union, CS-18** | **18 agents, 5 families, 3 joins, 3 clocks** | it marches; J2's first law closes to $4.07\times10^{-9}$; J3 to $0.0399$ against a registered $0.075$, with the residual diagnosed; J1's and J3's error vectors superpose to $0.39\%$ and partly cancel | [[case-study-vehicle-march-atlas-0.1]] |
| RaceLab porous car, CS-19/20 | 13 immersed bodies, 14 fluid windows, radiator core, turbine, machine, coolant loop | a car as a graph; a live classical / learned / certified switch per window | [[case-study-racelab-switch-atlas-0.1]] |
| RaceLab body-fitted car | curved grids around each part overlapping a Cartesian background; 381,698 unknowns on ten grids | the car the user drew marches as solids at a median $1.4$ s a step; the turbine join closes to $3.40\times10^{-4}$ (porous: $0.0292$) | [[poc3-racelab-car-solids]] · [[poc3-racelab-car-union]] |
| **rocket ascent, CS-21** | reacting flow, compressible flow, conduction + elasticity, rigid-body ODE: 7 agents plus a trajectory, each a build-repo solver imported unmodified | block conservation to $1.2\times10^{-13}$, thrust $0.992$ of ideal theory, 17 of 17 gates | [[case-study-rocket-bc-seam-atlas-0.1]] · [[outcome-c1-decomposition-vs-monolith]] |

---

## 4. (c) Substitutable, and "a few parameters"

**Swapping an expert is a one-line change where the framework allows it.**
- `rollout_for(case, kind="poseidon")` replaced every classical window of the design loop with the frozen checkpoint. Nothing that composes the expert was touched ([[poc1a-frozen-expert-results]] §1).
- On RaceLab each of 14 fluid windows flips between classical, learned and certified on the live page. Each window reports its one-step error against the classical expert **on the same input state**, and both per-call costs ([[case-study-racelab-switch-atlas-0.1]]).

**And it refuses swaps it cannot support, on screen and with the reason.** The PoC 2 demo shows the compiler refusing a real neural operator ([[poc2-demo-and-novelty]]). The body-fitted dashboard draws the learned mode *refused*, and names both blocks: the uniform $128\times128$ interface against $61.0\%$ of the unknowns, and a lead of $0.025$ against the checkpoint's native $0.1$ ([[poc3-racelab-certified-screen]]).

**"A few parameters":** of the car's eleven declared parameters, the body-fitted dashboard wires eight end to end, none of them by rebinding a module constant. Ride height, rake, the duct area, the coolant flow and the ambient temperature are among them. Getting there took two tiers: moved end to end, three of the eleven first reached nothing at all, and `rake` appeared only in its own declaration ([[poc3-racelab-knobs]]). Moving ambient temperature from $300$ to $318$ K moved the marching coolant loop's return temperature by $19.31$ K ([[poc3-racelab-dashboard]]). Geometry changes re-cut the car on **commit**, in $66$–$90$ s, and the page says which car it is marching. The car's shape is **data**: the user drew it, and a declared solids rule turns 27 plates into five welded solids and two rolling wheels ([[poc3-racelab-car-solids]]).

---

## 5. What limits the claim

- **One family has a learned option.** On RaceLab, one of five governing families has a shippable learned expert, covering sixteen of twenty-six agents ([[case-study-racelab-switch-atlas-0.1]] §0). Everything else in §3 is classical.
- **No march spans a vehicle's clocks.** The union's three clocks span $400\times$ in seconds, and the block's thermal time constant is $50.1$ s: $400{,}800$ fluid macro-steps, about 21 hours on the laptop ([[case-study-vehicle-march-atlas-0.1]]).
- **Fast and accurate at once is not shown.** The fast learned column is biased ([[outcome-c2-learned-experts-in-the-loop]] §4). The accurate classical column is not faster than a monolith ([[outcome-c1-decomposition-vs-monolith]] §3).
- **Declaring is not free.** Several tiers found the declarations themselves wrong: inverted normals on two of three `car_graph` seams (W308), a `motion_class=STATIC` default in the rocket's builder that made three `L2/InterfaceMotion` refusals vanish (Tier 79), and a rotor sized for open flow that left the powertrain with no operating point in the car (W199). The compiler catches many of these, not all.
- **3-D is a classical half-car in a box** ([[poc3-racelab-3d]]), and the vehicle union is **not differentiable** through the devices (W209).

---

## 6. What was NOT done, named

1. **Learned experts in any family other than 2-D incompressible flow**: no thermal, structural, compressible, reacting or electrical learned expert has marched in a graph.
2. **Rung 8 (tyre and contact)** is open: an inequality seam compiles to the same verdict as an equality one, which is a named hole, not a result ([[inequality-seam-admissibility]]).
3. **Rung 10–11 on the vehicle graph**: design gradients and an optimiser in the loop exist on small graphs only.
4. **A real 3-D CAD geometry** was decided on 2026-09-14 and not started.
5. **The declaration workflow has no user-facing tool** beyond the car editor and the dashboards. Defining a new case study still means writing a Python case module. The path out, and what it costs, is [[outcome-c4-path-to-declarative-cases]]: the compile is already data-driven, and the blocker is that every real case writes its own time march.

---

## See Also

- [[00-atlas-0.1-outcome]] · [[outcome-evidence-ledger]] · [[outcome-demos-and-artifacts]]
- [[f1-pathmap-and-end-goal]] §3.3 — the rung each graph above sits on
- [[general-coupling-scheme]] — the seven-parameter scheme the compiler instantiates
- [[interface-transfer-theory]] — how a seam's interface space and prolongations are declared
