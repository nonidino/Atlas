# The Joining Seams — W172, and what adding a subsystem costs

**Type:** Concept page — **measurement**: three joining seams built on a real tiling beside both circuits, and the $O(K)$ declaration count re-run across them, every number beside its control (folder: `Atlas 0.1/common/`)
**Status:** built 2026-09-11, Tier 46. `atlas/cases/integration_union.py`, `scripts/w172_joining_cost.py`, `out/w172/w172.json`, `tests/test_tier46_joining_seams.py`. Worklist rows **W172** (done), **W194**, **W195**, **W196**, **W197** (opened), **W192** (priced).
**Related:** [[per-region-assembly]] · [[per-region-decomposition-axis]] · [[port-algebra-atlas-0.1]] · [[interface-transfer-theory]] · [[gap-worklist]] · [[case-study-ladder-to-f1]] · [[f1-pathmap-and-end-goal]] · [[poc2-frontwing-results]] · [[case-study-wing-fsi-atlas-0.1]] · [[case-study-cooling-loop-atlas-0.1]] · [[case-study-powertrain-atlas-0.1]] · [[case-study-wake-array-atlas-0.1]] · [[case-study-learned-pair-atlas-0.1]]

---

# 0. The result, in one paragraph

**The three joining seams exist, the union they join compiles, and the prediction they were built to test is half right.** A radiator core standing in the airflow (J1, `MECH`), the motor-generator's heat conducted into the cooled block (J2, `THERM`) and the drivetrain's rotor placed in the wing's wake (J3, `MECH`) join `front_wing`'s six-window tiling to both circuits: eighteen agents across five families, 48 ports, 24 seams, **no open port, and still zero per-pair declarations**. The joined union's only refusal is the clock rule's, which the disjoint union carries too, and re-declaring ten agents' native steps admits both. Counted join by join at three, four and five families, **a join's declaration cost does not grow with the family count**: J1 and J2 cost exactly the same in every context they were measured in. **But no join costs "two new ports and nothing else."** J1 also splits an existing tiling seam; J2 moves an existing port's response and needs one unit reconciliation; J3 re-declares the rotor's two existing ports for its new host and re-derives the powertrain's operating point — and that re-derivation reaches ports on agents **off J3's own seams: three when J2 is absent, six when J2 is present**, because J2's heat source was built on the operating point J3 moves. Adding a subsystem, in either order, adds exactly that subsystem's own Tier 39 structure plus a fixed set of graph-level reconciliations. So Tier 39's caveat is retired by the measurement and replaced by a narrower one: **declarations are $O(1)$ per join; the state a join re-derives is not local to the join, and no rule reads it** (W196).

---

# 1. What the row said, and what was predicted

Tier 39 answered the $O(K)$ question on the union's existing structure and said what its answer was not:

> It is evidence about the **algebra** and no evidence about the **integration cost** … Each new seam needs two new ports, one per side — $O(1)$ per seam by inspection, and unmeasured.

The done-when: joining seams on a union that compiles, the count re-run at more than one family count, and the page stating **separately** what existing structure costs and what adding a subsystem costs — with the caveat retired only if the measurement retires it, and a negative treated as the result.

**Intuitively.** Counting the sockets on every appliance says how standardised the plugs are. It says nothing about the wiring. This tier wires three appliances together and counts everything the wiring touched: the new cables, the sockets that had to be cut into existing walls, the labels that had to be reprinted to fit another manufacturer's standard — and the appliance whose thermostat moved because a different one was plugged in beside it.

---

# 2. What joins what — checked against the port algebra before anything was built

## 2.1 The three candidates, and two of them are not constructible

What joins what was a design decision to make and to record. Each candidate was checked against what the two sides actually declare — [[case-study-learned-pair-atlas-0.1]] §2.1's method — before any code:

| candidate | what the two sides declare | verdict |
|---|---|---|
| aero tiling ↔ cooling loop, `THERM` (a radiator in the flow) | `FSIFlowWindow` carries $u$ and $v$ and nothing else; every port on the tiling side is `MECH` | **not constructible** — the flow has no temperature, so there is no `THERM` effort to declare |
| cooling loop ↔ powertrain, `THERM` (waste heat) | the block's dry face is a real Robin face of `step_thermal`; the machine's $I^2R$ is computed from the circuit | **constructible** |
| powertrain ↔ structure, `ROT` or `MECH` | `STRUCT` is clamped at its leading edge and declares one port, `wet:MECH`, which is connected | **not constructible** — no rotational degree of freedom for `ROT`, no second loaded face for `MECH` |

The two replacements keep **one join per pair of subsystems**, which is what makes this a count of joins *between* subsystems rather than of seams inside one:

- **J1 replaces the first**: the radiator's core stands in the airflow as a porous plane, coupled by `MECH`. The heat it rejects still leaves to ambient — the air carries no temperature, which is exactly why the bond is `MECH`.
- **J3 replaces the third**: the rotor, whose two disk faces are the **only open ports in the three graphs**, placed in the flow.

A test asserts both verdicts on the records themselves.

## 2.2 The three joins

**J1 — the radiator core in the flow.** `CoreRadiator` is `cooling_loop`'s radiator with two `MECH` faces returning the actuator disk's traction with a declared loss coefficient in place of an induction,

$$\tau_{\text{up}} = +\tfrac12 K_{\text{core}}\, u^2, \qquad \tau_{\text{down}} = -\tfrac12 K_{\text{core}}\, u^2, \qquad K_{\text{core}} = 1.2,$$

and a conductance that follows the air through the core, $UA(u) = UA_{\text{rad}}\,(u/u_{\text{ref}})^{0.8}$, with $u_{\text{ref}}$ the core's own inflow at the state it is built at. At that state $UA = UA_{\text{rad}} = 42$ W/K exactly, so the loop's gain and operating point do not move; what the join adds is the **dependence** — $+7.92\%$ in $UA$ for $+10\%$ air, which is $1.1^{0.8} - 1$.

**J2 — the machine's heat into the block.** `BlockAgent` carried its heat source as a declared outer gas temperature of about $22.9$ kK behind an $8$ W/(m²K) film, because `step_thermal` takes Robin data on two faces and no volumetric source. `MountedBlock` replaces that datum with the machine's casing behind a bolted mount, $-k\,\partial_n T = h_{\text{mount}}(T - T_{\text{case}})$ with $h_{\text{mount}} = 5000$ W/(m²K), and `CooledMachine` answers with its own dissipation through the mount area $A$:

$$q = \frac{I^2\, R(T)\, p_{\text{ref}}}{A}, \qquad R(T) = R_{\text{MGU}}\bigl(1 + \alpha_{\text{ref}}(T - T_{\text{case,ref}})\bigr), \qquad \alpha_{\text{ref}} = \frac{\alpha_{20}}{1 + \alpha_{20}\,(T_{\text{case,ref}} - T_{20})},$$

with copper's $\alpha_{20} = 3.93\times10^{-3}$ /K carried exactly to the build temperature, so at the build state $R = R_{\text{MGU}}$ and the circuit's own responses do not move. Both sides return the heat crossing **into the block**, so the seam declares `effort_normal="BLOCK"` — W138, as `cooling_loop`'s own wall seam does.

**J3 — the rotor in the wake.** `wake_array.RotorDisk`, unchanged, with its inflow read from the host ring rather than `powertrain`'s declared $U_{\text{ref}} = 1$.

## 2.3 Where the devices sit, and why a device splits an existing seam

In an overlapping tiling two artificial rings face each other across the overlap, halo cells apart. A device sits **between** them — `wake_array`'s convention exactly: its `up` face meets the downstream window's `xlo` ring, which lies upstream of the device plane, and its `down` face meets the upstream window's `xhi` ring. The whole-face seam those two rings shared is therefore **split** into a `bypass` seam over the remaining 48 cells and the device's two seams over its 32 — an existing declaration re-made, and counted as one.

| device | seam split | rings | local cells | clearances |
|---|---|---|---|---|
| rotor (J3) | `x10` | `F10.xhi` at $x = 143$, `F20.xlo` at $x = 128$ | 12–44 | 8 cells behind the plate's trailing edge; clear of the $y$-overlap band $[64, 80)$ |
| radiator core (J1) | `x01` | `F01.xhi` at $x = 79$, `F11.xlo` at $x = 64$ | 32–64 (global $y$ 96–128) | above the plate; clear of the $y$-overlap band |

On a split ring the bypass segment keeps the face's **perturbation** convention — the trace is added to the ring, as every `front_wing` face does — and the device segment is **absolute**, because a device's response is a function of the inflow itself. Both split seams derive $\dim M = 25$ on 48 cells and reach **the same rules and verdicts** as the whole-face seams they replace.

## 2.4 One decision the count depends on: the operating point is solved, the power unit is held

The measured inflow at the rotor site has mean $0.9225$ — min $0.675$ in the wing's wake, max $1.070$ — not $U_{\text{ref}} = 1$. At that inflow `powertrain`'s own `CircuitSolve` finds a different operating point: shaft speed $\omega = 13.837$ against $15$, induction $0.114$ against $0.286$, loop current $0.1457$ against $0.5333$ — still generating, with balance residuals of $3\times10^{-15}$.

That forced a choice about J2's one reconciliation constant. `powertrain` is nondimensional and the block works in watts, so $p_{\text{ref}}$ — watts per nondimensional power unit — is the one number here chosen by looking at an answer: **$42{,}187.5$ W**, the value at which the machine dissipates the block's own $Q_{\text{source}} = 1800$ W at `powertrain`'s reference operating point. It could be re-fitted whenever the operating point moves, which would keep the block's source at 1800 W — and would make a unit depend on a state. **It is held.** So with J3 present the machine's heat is $134.3$ W, $0.0746$ of 1800, and the block's casing datum is re-derived from it: $362.7$ K against $396$ K. §5.4 is what that decision exposes.

---

# 3. The counter, before anything new was counted

Tier 39's table has no script behind it, so the counter had to reproduce it before counting anything new, or its definitions would not be the same ones. `scripts/w172_joining_cost.py` counts per-agent prolongations, scale sets (distinct `nondim` dictionaries) and per-pair declarations (connections carrying a seam-level `space` or `prolongations`) exactly as Tier 39 defined them:

| graph | $K$ | families | ports | per-agent prolongations | scale sets | per-pair |
|---|---|---|---|---|---|---|
| `ground_effect` | 7 | 1 | 16 | 16 | 1 | 0 |
| `thermal_seam` | 2 | 2 | 2 | 2 | 1 | 0 |
| `wing_fsi` | 7 | 2 | 16 | 16 | 1 | 0 |
| `front_wing` | 8 | 2 | 18 | 18 | 1 | 0 |
| `cooling_loop` | 5 | 2 | 10 | 10 | 2 | 0 |
| `powertrain` | 5 | 2 | 12 | 12 | 3 | 0 |
| **disjoint union** | **18** | **5** | **40** | **40** | **5** | **0** |

**Every row equals the published one.** The disjoint union is built by the three graphs' own `build()` calls and equals its parts **declaration for declaration** — every port's type, direction, scales, resolution, response half, prolongation matrix and Gram — so nothing a join costs can hide inside it. It is the control every number below is a difference against.

---

# 4. The joined union compiles

| union | $K$ | families | ports | seams | per-pair | open ports | verdict | refusals |
|---|---|---|---|---|---|---|---|---|
| disjoint, native clocks | 18 | 5 | 40 | 19 | 0 | 2, the rotor's faces | `refuse` | `L7/R9` |
| **joined, native clocks** | **18** | **5** | **48** | **24** | **0** | **0** | `refuse` | `L7/R9` |
| disjoint, clocks reconciled | 18 | 5 | 40 | 19 | 0 | 2 | `admit-uncertified` | none |
| **joined, clocks reconciled** | **18** | **5** | **48** | **24** | **0** | **0** | **`admit-uncertified`** | **none** |

**The joins add no refusal.** Every union of two or three subsystems measured — thirteen graphs, every subset of the joins — refuses at `L7/R9` and at nothing else, and that refusal is the graph's clocks ($0.0125$, $0.05$ and $0.2$), not any join (§6.4); `front_wing` alone refuses nothing. `admit-uncertified` is the ceiling, because the union declares no measured constants — `front_wing`'s were measured on `front_wing` (W190).

What the compile says on the new seams, each attributed:

- **`L1/E3` decertifies J1's two seams and J2**: they join different governing families, so no monolithic reference exists — correct, and what a multiphysics seam is. J3's seams do not decertify: the rotor declares the flow's family.
- **`L4/probe-base` decertifies the two down seams**, at $0.0016$ ($0\%$ of the base norm) for the core and $0.0196$ ($3\%$) for the rotor: a device reads its inflow from the upstream ring and declares that one base on both faces, which is `wake_array`'s own convention.
- **`L4/E7/passivity` decertifies the same two down seams**, at defects of $0.947$ and $1.858$ — and §8.2 shows that was already true of `wake_array`.

---

# 5. What each join costs, at more than one family count

## 5.1 How the cost is read

A join's cost is the difference between **two graphs that differ by that join alone**, every other join held fixed. Nothing in the table is typed. New and removed seams and ports come from the two graphs' ids. A port is **re-declared** if its declaration differs — type, direction, scales, resolution, response half, prolongation matrix and Gram, interface space. A port **moved** if its own probe base or its response at that base differs bitwise. **State** is the partner-read data the build records: the machine's speed, the rotor's inflow, the radiator's $UA$, the block's outer Robin datum, the machine's current and heat, the casing datum. Compile changes are the two decision records compared seam by seam.

Each join was measured in every context the three subsystems allow:

- **J1** at 4 families (tiling and cooling loop) and at 5, with J2 and J3 present;
- **J2** at 4 (both circuits), at 5 without J3, and at 5 with J3;
- **J3** at 3 (tiling and powertrain), at 5 without J2, and at 5 with J2.

## 5.2 The table

| column | **J1** (4, 5) | **J2** (4, 5, 5 with J3) | **J3** (3, 5 without J2) | **J3** (5 with J2) |
|---|---|---|---|---|
| joining seams | 2 | 1 | 2 | 2 |
| new ports on the joining seams | 4 | 2 | 2 | 2 |
| existing ports re-declared on them | 0 | 0 | **2** | **2** |
| of the joining sides, content taken from the partner | 2 (`RAD`) | 1 (`MGU`) | 2 (`ROTOR`) | 2 |
| existing seams split and re-declared | **1** | 0 | **1** | **1** |
| ports removed / split ports added | 2 / 2 | 0 / 0 | 2 / 2 | 2 / 2 |
| prolongations added / re-declared | 6 / 0 | 2 / 0 | 4 / 2 | 4 / 2 |
| **per-pair declarations** | **0** | **0** | **0** | **0** |
| **scale sets added** | **0** | **0** | **0** | **0** |
| declared physical constants | 2 | 2 | 0 | 0 |
| unit reconciliation constants | 0 | **1** | 0 | 0 |
| `effort_normal` declarations | 0 | 1 | 0 | 0 |
| agent records changed (weight hash) | 1 | 2 | 0 | 0 |
| existing ports moved at their own base | 0 | **1** | **6** | **9** |
| … of them on agents off the join's own seams | 0 | 0 | **3** | **6** |
| state introduced / re-derived | 0 / 0 | 3 / 2 | 0 / **2** | 0 / **6** |
| existing seams whose decisions changed | 0 | 0 | 0 | 0 |
| new refusals, directed cycles, region changes | 0 | 0 | 0 | 0 |
| new multirate seams | 2 | 1 | 2 | 2 |

**J1's column is identical at 4 and 5 families, and J2's at 4, 5 and 5 with J3.** The per-pair and scale-set rows are zero in every context of every join.

## 5.3 The prediction, against the table

*Each new seam needs two new ports, one per side — $O(1)$ per seam.* Taken clause by clause:

| clause | J1 | J2 | J3 |
|---|---|---|---|
| two new ports per new seam, **in count** | holds, $4 = 2\times2$ | holds, $2 = 2\times1$ | **fails** — two new, two **re-declared**: the rotor's open ports had to be re-cut for their host |
| **and nothing else** | **fails** — an existing seam split, two ports removed | **fails** — an existing port's response moved, a unit reconciliation | **fails** — a seam split, two prolongations re-declared, six or nine ports moved |
| constant in the family count | **holds**, at 4 and 5 | **holds**, at 4 and 5 | declarations **hold** at 3 and 5; the state columns **do not** — §5.4 |
| no per-pair declaration | holds | holds | holds in the host form; §7 prices the others |

**What each "else" is.** J1's split is structural and unavoidable: a device plane occupies part of an overlap two rings already shared, so their seam has to be re-declared over the part it does not occupy. J2's moved port is `BLOCK.wall:THERM`, the block's coolant-side response, whose outer Robin datum is now the casing behind the mount instead of a declared gas temperature — joining a heat source changes what the block's *other* face returns, which is physics and not bookkeeping. J3's re-declared ports are the rotor's two disk faces: `wake_array` declared them at its own 8-cell cutoff (9 modes on 32 cells) and a $1/32$ cell measure, while `front_wing`'s rings declare a 4-cell cutoff (17 modes) at $1/64$ — **a port's discretization is its host's, so an open port connected into a different host is not free to connect**.

## 5.4 J3's cost depends on whether J2 exists — W196

Every declaration column of J3 is the same in all three of its contexts. Three columns are not:

| J3 measured from | families | ports moved at their own base | … off J3's own seams | state re-derived |
|---|---|---|---|---|
| tiling and powertrain | 3 | 6 | **3** | 2 |
| all three subsystems, J1 present | 5 | 6 | **3** | 2 |
| all three subsystems, J1 **and J2** present | 5 | 9 | **6** | 6 |

Without J2 the three off-seam ports are the machine's own — `MGU.shaft:ROT`, `lo:ELEC` and `hi:ELEC` — because J3 moves the rotor's inflow, which moves the solved shaft speed, which moves the machine's back-EMF. **With J2 present the reach extends through the machine into the block**: `MGU.case:THERM`, `BLOCK.outer:THERM` and `BLOCK.wall:THERM` move too, and four more state data are re-derived — the machine's current ($0.533 \to 0.146$), its heat ($1800 \to 134.3$ W), the casing datum and the block's outer datum ($396 \to 362.7$ K).

**The family count did not change the cost; the join graph did.** Between the second and third rows the union has the same agents, families and pairs; the only difference is that another join reads the state J3 re-derives. **No verdict moves and no rule reads any of it**: `MachineAgent.omega` and J2's heat enter the records as declarations without provenance, so a compile of the union at a stale operating point is identical to one at the solved point.

**[AI Inference]:** in a vehicle, operating-point state is shared widely — shaft speed, coolant temperature and battery state each feed several subsystems — so the number of joins reading a re-derived state plausibly grows with the number of subsystems. If it does, a join's cost *in state* is of the order of the joins that read what it moves, which in the worst case is $O(K)$ per join, while its cost in declarations stays $O(1)$. One such dependency is measured here; the growth law is not.

---

# 6. Existing structure, and what adding a subsystem costs

## 6.1 Existing structure — Tier 39's count, unchanged by joining

The union's existing structure is §3's last row: 40 ports, 40 per-agent prolongations, 5 scale sets and 0 per-pair declarations against 153 pairs. **Joining all three seams leaves the per-pair and scale-set columns exactly where they were** — 48 ports, 48 prolongations, 5 scale sets, 0 per-pair — so the property Tier 39 measured of the algebra holds after integration and not only before it.

## 6.2 Adding a subsystem, in two orders

Both orders start from `front_wing` and end at the same joined graph. **Every step that adds a subsystem adds exactly that subsystem's Tier 39 row** — its ports, prolongations and per-pair declarations, and its own seams — in both orders; every step that adds joins adds §5's columns.

| order A | $K$ | families | Δ ports | Δ seams | what the step cost |
|---|---|---|---|---|---|
| `front_wing` | 8 | 2 | — | — | its own structure: 18 ports, 9 seams |
| + `cooling_loop` | 13 | 4 | +10 | +5 | exactly `cooling_loop`'s structure; 4 `cut_axis` declarations; 5 agents now off the macro-step |
| + J1 | 13 | 4 | +4 | +2 | §5.2's J1 column |
| + `powertrain` | 18 | 5 | +12 | +5 | exactly `powertrain`'s structure; 4 `cut_axis`; 10 agents off the macro-step; `ROTOR`'s region re-derived (W192) |
| + J2 and J3 | 18 | 5 | +4 | +3 | 4 new and 2 re-declared joining ports, 1 split, 1 reconciliation, 7 ports moved — **none off the new seams** — 3 state introduced, 4 re-derived |

| order B | $K$ | families | Δ ports | Δ seams | what the step cost |
|---|---|---|---|---|---|
| `front_wing` | 8 | 2 | — | — | its own structure |
| + `powertrain` | 13 | 3 | +12 | +5 | exactly `powertrain`'s structure; 4 `cut_axis`; `ROTOR`'s region re-derived (W192) |
| + J3 | 13 | 3 | +2 | +2 | §5.2's J3 column: 6 ports moved, **3 off its seams** |
| + `cooling_loop` | 18 | 5 | +10 | +5 | exactly `cooling_loop`'s structure; 4 `cut_axis` |
| + J1 and J2 | 18 | 5 | +6 | +3 | 6 new joining ports, 1 split, 1 reconciliation, **1** port moved, 3 state introduced, 2 re-derived |

**The totals agree and the attribution does not.** Order A adds J2 and J3 in one step, so J3's reach into J2's agents falls inside the step and its off-seam count is zero; order B adds J3 first, so its reach into the machine shows as three ports off its own seams, and J2 is then built on the operating point J3 already moved and re-derives nothing of J3's. **Which join is charged for a shared state depends on the order the joins were made in** — W196 again, seen from the other side.

## 6.3 What the union reconciles once

Adding a subsystem to a union overrides that subsystem's own value of nine graph-level fields. None of these is paid per join — no join changed a graph-level field — and each is paid once per union:

| field | the parts | the union |
|---|---|---|
| `decomposition` | `OVERLAPPING` on the tiling, `NON_OVERLAPPING` on both circuits | `OVERLAPPING`, with **8 `cut_axis` declarations** on the circuits' same-family seams (W171) |
| `partition_of_unity`, `overlap`, `overlap_cells` | one graph-scoped object on the tiling, none on the circuits | filed under the fluid region (W189) |
| `global_fields` | the tiling's pressure projection | the same, scoped to the six windows |
| `cross_points` | `("mid",)` on the tiling, `()` on each circuit | `("mid",)` |
| `loop_gains` | one per circuit | both |
| `macro_dt` | $0.0125$, $0.05$ and $0.2$ | $0.0125$ — and §6.4 |
| `measured` | `front_wing`'s constants; none on the circuits | none (W190) |

## 6.4 The clocks — W194

| union | clocks | multirate seams | agents at one | verdict | what `L7/R9` asks for |
|---|---|---|---|---|---|
| disjoint | native | **0** | 0 | `refuse`, `L7/R9` | time-integrated matching |
| disjoint | native, time-integrated declared | 0 | 0 | `refuse`, `L7/R9/quadrature` | an integrated response from **all 18** agents |
| joined | native | **5** | 8 | `refuse`, `L7/R9` | time-integrated matching |
| joined | native, time-integrated declared | 5 | 8 | `refuse`, `L7/R9/quadrature` | an integrated response from **all 18** agents |
| either | reconciled: 10 agents re-declare $dt_{\text{native}}$ | 0 | 0 | `admit-uncertified` | nothing |

**`L7/R9` refuses a union that has no multirate seam.** In the disjoint union every seam joins two agents on the same clock — the tiling at $0.0125$, the coolant circuit at $0.05$, the electrical circuit at $0.2$ — and no flux crosses between two clocks anywhere; the rule refuses anyway, because `CaseGraph.is_multirate` reads every agent's native step and not any seam's. The joins make the mismatch **real** — J1, J2 and J3 are exactly the five multirate seams — and the requirement still names all eighteen agents where eight sit at a multirate seam. `FluxMatching`'s own docstring states the requirement as *every agent at a multirate seam*; the code checks every agent in the graph. It is W136's scope defect — a rule reading every agent in the graph where its premise is about a subset — one rule along, on the clocks.

Both ways to pay are union-level and neither is per join: **re-declare ten agents' clocks** — legitimate for the circuits' closed-form elements, and a real change to the block's backward-Euler step — or **give all eighteen agents an integrated response**. A seam-scoped R9 would ask it of eight.

---

# 7. The rotor's faces, four ways — W195

J3's rotor faces meet a host whose discretization is not the one they were declared at. The same join was built four ways, against the union holding J1 and J2:

| form | rotor face declares | window face declares | `L3/C2/C3/C6` | trace ratio, rotor / window | L4 on the up seam | cost |
|---|---|---|---|---|---|---|
| **host** — re-declare the ports | 17 modes at $1/64$ | 17 modes at $1/64$ | admit, $\dim M = 17$ derived | $1.000$ | `probe-base` admits | 2 new + **2 re-declared** ports |
| **native** — the control | 9 modes at $1/32$ | 17 modes at $1/64$ | **refuse**, both seams | $0.7071$ | `probe` decertifies: no space, nothing to probe | 2 new ports, **2 refusals** |
| **per pair**, host measure | 9 modes at $1/64$, on the connection | 9 modes at $1/64$, on the connection | admit, $\dim M = 9$ | $1.000$ | `probe-base` admits | 2 new ports, **2 per-pair** |
| **per pair**, each side's own measure | 9 modes at $1/32$ | 9 modes at $1/64$ | **admit** | $\mathbf{0.7071}$ | `probe-base` **decertifies, 41%** | 2 new ports, 2 per-pair |

Three things follow.

**The algebra's zero per-pair count is a choice the host form makes, not a law.** The same join can be paid as two re-declared ports on an existing agent or as two per-pair declarations on the connections — and the per-pair form also halves the interface resolution on those seams, because `C2/C3/C6` refuses a $\dim M$ above the coarser side's declared resolution.

**The native control refuses where it should**: two ports declaring different cutoffs leave $\dim M = \min_i m_i^{\text{eff}}$ nothing to derive, and the refusal cites the missing declaration.

**And the per-pair form in each side's own measure is admitted.** Copying each port's own prolongation onto the connection gives the rotor's side a $1/32$ cell measure and the window's $1/64$, so one interface coefficient reconstructs traces a factor $1/\sqrt{2}$ apart on the two sides — the rotor would see an inflow $0.707$ times the ring's. `C2/C3/C6` checks each prolongation's adjointness **in its own Gram** and passes both; nothing compares the two sides' measures. `L4/probe-base` notices here only indirectly, as a $41$–$42\%$ disagreement between the two sides' bases, and only because both bases on a device seam are absolute. **[AI Inference]:** on a perturbation-convention seam, where both sides' bases are zero, the same mismatch would pass every check the compile runs. That is W195.

---

# 8. The physics under the declarations

## 8.1 The joins are physics, and their controls hold

| control | measured |
|---|---|
| powertrain at its reference, $u = 1$ | $\omega = 15$, induction $0.2857$, current $0.5333$, valid, balance residual $1.9\times10^{-15}$ |
| powertrain at the host inflow (J3), $\bar u = 0.9225$ | $\omega = 13.837$, induction $0.1138$, current $0.1457$, **still generating**, residual $3.0\times10^{-15}$ |
| machine heat into the block (J2), without / with J3 | $1800$ W / $134.3$ W — ratio $0.0746$, with the unit held |
| casing datum, without / with J3 | $396$ K / $362.7$ K |
| the block's first law at the release state — declared source / mount at 1800 W / mount at the joined point | relative residual $3.3\times10^{-8}$ / $4.0\times10^{-8}$ / $3.0\times10^{-8}$ |
| the coolant loop's gain with the core in the flow (J1), at the build state | identical to the loop's own |
| the core's inflow, and $UA$ at $+10\%$ air | $\bar u = 1.0045$; $UA$ up $7.92\%$, exactly $1.1^{0.8} - 1$ |

The mounted block's first law closes to the same order as the original's: joining the heat path changed where the source comes from, not whether the block conserves energy.

## 8.2 Every actuator down seam reads non-passive, and `wake_array` already did — W197

The two new down seams failing `L4/E7/passivity` could have been something the joins did. The control is `wake_array`'s own graph, compiled with the reference expert on uniform flow:

| seam | `L4/E7/passivity` |
|---|---|
| `wake_array` `R1_down`, `R2_down`, `R3_down` | decertified, defect $2.000$ each |
| `wake_array` `R1_up`, `R2_up`, `R3_up` | no defect |
| this union, `J1_core_down` / `J3_rotor_down` | decertified, $0.947$ / $1.858$ |
| this union, `J1_core_up` / `J3_rotor_up` | no defect |

**Inherited, and unrecorded**: no page, test or script in the vault mentioned a down seam's passivity before this control. It is W197. **[AI Inference]:** it looks like W138's class — the disk returns $+T(u)$ on its upstream face and $-T(u)$ on its downstream one, against rings that each report against their own outward normal, so on the down seam the two blocks may be co-oriented and the assembled operator a difference presented as a sum. Not diagnosed here.

---

# 9. What the caveat becomes

Tier 39's caveat said the zero per-pair count was evidence about the algebra and none about the integration cost. The integration cost is now measured, so **the caveat as written is retired** — and the measurement replaces it with four statements rather than with the zero:

1. **The per-pair count stays zero across every join** in the form that re-declares a port for its host, so the algebra's promise survives integration. Paying the same join per pair is possible, and costs two per-pair declarations and half the interface resolution (§7).
2. **A join's declaration cost is a small constant**, identical at every family count measured — three, four and five: two seams and four ports with one existing seam split (J1); one seam, two ports and one unit reconciliation (J2); two seams, two new and two re-declared ports and one split (J3).
3. **The constant is not "two ports and nothing else."** Every join also re-declares or moves existing structure, and one kind — re-declaring an open port for a different host — is exactly the cost that [[f1-pathmap-and-end-goal]] §4's *each rung connects a port* does not mention.
4. **What a join re-derives is not local to it.** J3 moved three ports off its own seams without J2 and six with J2; the order the joins are made in decides which join is charged; and no rule sees it (W196).

**[AI Inference]:** the honest one-line version of [[f1-pathmap-and-end-goal]] §4's *rung $n+1$ costs what rung $n$ cost* is that it holds for declarations and is unmeasured for state. Rung 9's affordability now turns on how many joins read each shared operating point, not on how many experts there are.

---

# 10. What this tier did NOT do, named

- **Nothing marches the joined union.** It is built and compiled; each port's response is evaluated at its own base; the powertrain's operating point is solved by `CircuitSolve` alone, not by a union-wide interface solve. Whether the state J3 and J2 share converges when marched is not measured.
- **W194–W197 were opened and not fixed.** `L7/R9` is not re-scoped, `C2/C3/C6` does not compare measures, no state carries provenance, and the down seams' orientation is not diagnosed.
- **The first candidate, `THERM` on the tiling, was not built.** The flow carries no temperature; a thermal passenger on the fluid windows would be a new expert, not a join.
- **The machine's circuit does not see its own temperature.** $R(T)$ enters the `THERM` response only, referenced so the `ELEC` responses do not move at the build state; the back-coupling of casing temperature into the loop current is not built.
- **The rotor disk is linearized at its declared induction, $1/3$**, while the controller's solved induction is $0.286$ in `powertrain` and $0.114$ in the host — inherited from `powertrain`, unchanged here.
- **No measured constants on the union**, so `admit-uncertified` is its ceiling (W56, W190).
- **Rung 9's integration graph — the third task — was not started.** Nothing downloaded, no machine rented, NeuberNet not loaded, nothing pushed.

## See Also

- [[per-region-assembly]] — the union Tier 45 built, and the input it handed this tier: the graph-level fields and W192.
- [[per-region-decomposition-axis]] — `cut_axis`, which the union declares on the circuits' eight same-family seams.
- [[port-algebra-atlas-0.1]] — the closed vocabulary and the per-port scale sets whose count stays at five.
- [[interface-transfer-theory]] — $\dim M = \min_i m_i^{\text{eff}}$ and the prolongation–adjoint pair §7's four forms are built from.
- [[case-study-wake-array-atlas-0.1]] — the rotor, its segment convention, and the down-seam passivity W197 found it already had.
- [[case-study-cooling-loop-atlas-0.1]] and [[case-study-powertrain-atlas-0.1]] — the two circuits, unchanged, each built by its own `build()`.
- [[poc2-frontwing-results]] and [[case-study-wing-fsi-atlas-0.1]] — the tiling the devices sit in.
- [[gap-worklist]] Tier 46 — W172 closed; W194–W197 opened.
- [[case-study-ladder-to-f1]] §20 and [[f1-pathmap-and-end-goal]] §3.3 — where rung 9 stands now.
