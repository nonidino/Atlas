# Rung 9's Gate, Restated — what the full-vehicle graph can be held to

**Type:** Concept page — **a gate re-read against the graph that exists**, with the measurements that decide each clause (folder: `Atlas 0.1/common/`)
**Status:** built 2026-09-11, Tier 47. `scripts/rung9_gate.py`, `out/rung9/rung9.json`, `tests/test_tier47_rung9_gate.py`. Worklist rows **W198** (the gate), **W199**, **W200**, **W201** (opened); **W196**, **W197** (annotated).
**Related:** [[joining-seam-cost]] · [[f1-pathmap-and-end-goal]] · [[case-study-ladder-to-f1]] · [[gap-worklist]] · [[case-study-scaling-ladder-atlas-0.1]] · [[case-study-learned-pair-atlas-0.1]] · [[port-algebra-atlas-0.1]] · [[composition-error-theory]] · [[case-study-thermal-strain-atlas-0.1]] · [[case-study-wing-fsi-atlas-0.1]] · [[case-study-powertrain-atlas-0.1]] · [[case-study-wake-array-atlas-0.1]] · [[open-problems-atlas-0.1]]

---

# 0. The result, in one paragraph

**Rung 9's gate does not fit the full-vehicle graph as written — neither clause — and a replacement is proposed rather than quietly substituted.** [[f1-pathmap-and-end-goal]] §3 gives the rung *"$\mathcal R$ closes; composition error sub-linear in $N$"*. The graph now exists — Tier 46's union of `front_wing`'s tiling with both circuits, eighteen agents, five families, joined ([[joining-seam-cost]]) — so each clause was held against it before anything else was attempted. **The clause about $N$ has no variable**: the unions this vehicle can form sit at $N = 8, 10, 13$ and $18$, $N = 13$ is two different graphs with three and four families, and every step up in $N$ adds a different subsystem's physics — the confound [[case-study-scaling-ladder-atlas-0.1]] §4 found carrying most of a headline exponent, which that ladder could remove and this graph cannot. **The clause about $\mathcal R$ cannot be assembled from the records**: no field on any record or graph can carry dissipation, while 13 of the 18 agents dissipate by their own constants; ten window faces are the fluid domain's physical boundaries and carry energy through no port; and the three subsystems' clocks are written in three unit systems nothing relates. What *can* be measured is this vault's own practice since CS-9 — **each join's receiving subsystem closes its own balance with the join's term in it and fails without it**. On the three joins: **J2 passes**, the block's first law closing to $3\times10^{-8}$ with the mount term and failing by seven orders without it; **J1 has no subject**, because no power crosses it; and **J3 fails by exactly a factor of two** — the shaft delivers the power of a rotor one diameter wide, while its face in the host is half as long, and the two declared faces carry exactly zero net power. Re-sizing the disk for its host closes that balance and leaves the declared machine **with no operating point at all**: it demands $31\times$ the largest torque a host-sized disk can deliver (W199).

---

# 1. The gate as written, and what each clause presumes

[[f1-pathmap-and-end-goal]] §3's ladder row:

> **9** — Full-vehicle graph — $\sim$15–20 agents; **Claim B under real load** — integration only — Gate: **$\mathcal R$ closes; composition error sub-linear in $N$**.

and the claim it tests, from §2: *adding the $N$th agent costs $O(1)$ integration work and adds sub-linear composition error.* [[case-study-ladder-to-f1]] schedules the rung as CS-16, *graded against CS-7's prediction*.

The instruction this tier worked under: if the gate no longer fits the graph that can actually be built, **say so and propose the replacement rather than silently substituting one** — what [[case-study-learned-pair-atlas-0.1]] §9 did for rung 3's gate (W183), and this vault's practice since.

**Intuitively.** A gate is a question a graph can answer. "Does error grow slowly as you add pieces?" needs a way to add pieces without changing what the pieces are. "Is energy conserved?" needs a ledger in which every kind of energy has a line, in one currency. This page checks whether the vehicle graph has either — and, finding neither, asks the version of each question it *can* answer.

---

# 2. "Composition error sub-linear in $N$" has no variable on this graph

Every union Tier 46 built from the three subsystems, by the number of agents it holds:

| $N$ | graphs at this $N$ | subsystems | families |
|---|---|---|---|
| 8 | 1 | `front_wing` | 2 |
| 10 | 1 | `cooling_loop`, `powertrain` | 4 |
| **13** | **2** | `front_wing` + `cooling_loop` / `front_wing` + `powertrain` | **4 / 3** |
| 18 | 1 | all three | 5 |

$N$ is not a dial on a vehicle. It moves only by adding a subsystem, so every change in $N$ is also a change in what is being composed, and at $N = 13$ "the composition error" is two different numbers for two different machines. A function of $N$ is not defined here.

The question is not unanswered; it was answered where $N$ **is** a dial. [[case-study-scaling-ladder-atlas-0.1]] grew one geometry from 1 to 24 windows and measured exponents of $+0.804$ (classical, reference-free), $+0.477$ (checkpoint) and $+0.851$ (classical, against the monolith) — and then, holding the physics fixed with one turbine at every size, $+0.138$, $+0.201$ and $+0.160$. **Most of the headline was the flow getting harder, not the cut getting longer.** That control is exactly what a vehicle cannot run: its physics grows with its size by construction.

A composition error also needs a referent. The vehicle has three kinds, and none spans it:

| where | referent | declared |
|---|---|---|
| inside the tiling | the one-window monolith (`SINGLE_TILING`) and `front_wing.referent_rollout` | yes |
| inside each circuit | the closed forms, `LoopSolve._closed_form` and `CircuitSolve._closed_form` | yes |
| J3 (one governing family) | a monolith with the disk as a body force — CS-7's classical column | not built here |
| J1, J2 (two families each) | no monolith exists; only a tightly coupled iteration of the same union | not built |

**[AI Inference]:** a vehicle-scale ladder in $N$ would have to replicate subsystems — $k$ rotors on $k$ overlaps, each with its own circuit — and would re-import CS-7's confound through the wakes the rotors shed on each other. The question rung 9 can answer is per join, not per $N$.

---

# 3. "$\mathcal R$ closes" cannot be assembled from the records

[[port-algebra-atlas-0.1]] §6's residual:

$$\mathcal R(t)=\sum_{i\in\mathcal A}\frac{\mathrm dE_i}{\mathrm dt}+\sum_{\Gamma\in\mathcal P}\int_\Gamma (e\,f)_\Gamma\,\mathrm ds-\sum_i\mathcal D_i-P_{\text{ext}} .$$

Term by term, against what the joined union's records declare:

| term | what it needs | what the joined union declares |
|---|---|---|
| $\mathrm dE_i/\mathrm dt$ | a storage function per agent | **all 18 agents declare `storage`** |
| $\int_\Gamma e\,f$ | an effort–flow pair per seam | every port, 48 of them |
| $\mathcal D_i$ | each agent's dissipation | **no field exists** on `ExpertCapabilities` or `CaseGraph` — while **13 of the 18 dissipate by their own constants**: six windows by $\nu$, three legs by $UA$ (the core also by its pressure drop), four circuit elements by $I^2R$ |
| $P_{\text{ext}}$ | the power through open ports | **no open port** — but **ten window faces** are the fluid domain's inlet, outlet, ground and top, and carry energy through no port at all |

And the currency. Each subsystem is written in its own units, and no field carries one:

| subsystem | clock `dt_native` | `L_native` | written in |
|---|---|---|---|
| `front_wing` | $0.0125$ | $1.25$ (a window) | the tiling's convective units: length 64 cells, $U_\infty = 1$ |
| `powertrain` | $0.2$ | $1.0$ | `wake_array`'s: the rotor diameter $D = 1$, $U_\infty = \rho = 1$ |
| `cooling_loop` | $0.05$ | $0.2$ | SI: seconds, metres, watts |

A global $\mathcal R$ has to put all three in one currency. Only one join declares an exchange rate — J2's $p_{\text{ref}}$, watts per power unit — and nothing relates the tiling's time or length to the others (W201). `L7/R9` compares the three clocks as bare numbers, so Tier 46's "reconciled clocks" admitted equal numbers, not equal times.

**Every $\mathcal R$ that has closed in this vault was a receiver's own balance, written by hand.** [[case-study-thermal-strain-atlas-0.1]] §6 measured a global $\mathcal R$ blind to its coupling by six orders and ruled that *the right residual is the receiving subsystem's own balance*; [[case-study-wing-fsi-atlas-0.1]] §6 closed rung 5's gate on the structure's balance — a factor of $115$ between with and without the coupling term over the march, $1.7\times10^{7}$ once settled; the front-wing demo's needle is the plate's and the spring's energy against the interface power. Rung 2's own $\mathcal R$ gate was closed as **not measurable** ([[open-problems-atlas-0.1]] OP-4), because its dissipation term had no single $\nu$ to be written with. And [[composition-error-theory]] §2.2 is explicit that $\mathcal R = 0$ proves nothing about accuracy: it is a falsifier, not a bound. That is W200.

---

# 4. The clause that can be measured: each join's receiving balance

The rule, stated for a join: **the subsystem that receives energy through the join closes its own balance with the join's term in it, and does not close without it.** The second half is the control — a balance that closes either way is not measuring the join. All three joins, at the release state, one step each.

## 4.1 J2 — the block receives the machine's heat, and passes

| accounting | with the join's term | without it | factor |
|---|---|---|---|
| **control**: the unjoined block, its declared source | $3.31\times10^{-8}$ | $1.000$ | $3.0\times10^{7}$ |
| J2 without J3: the mount carrying 1800 W | $4.05\times10^{-8}$ | $1.000$ | $2.5\times10^{7}$ |
| J2 with J3: the mount carrying the joined 134 W | $3.00\times10^{-8}$ | $0.269$ | $9.0\times10^{6}$ |

The block's first law — stored heat rate against what enters through the mount and leaves through the wetted face — closes to the same order as the unjoined block's and fails by seven orders without the mount term. **J2's clause is met at the release state.**

## 4.2 J3 — the powertrain receives the flow's power, and fails by exactly the rotor's width

The build repo's `disk.ActuatorDisk` is written with $D = U_\infty = \rho = 1$: its swept width is $1$, its thrust is $T = \tfrac12\rho A C_T' U_d^2$ and its shaft power $P = T\,U_d$. The rotor's upstream face in `front_wing`'s tiling is $32$ cells at $1/64$ — half a unit long. The power the flow gives up through that face, $\int \tau\,u\,\mathrm ds$ with $\tau = \tfrac12\rho C_T' u^2$ the traction the face returns, against the power the shaft delivers:

| geometry, inflow | face length | disk width | face power / shaft power | ÷ inflow non-uniformity $\overline{u^3}/\bar u^{3}$ |
|---|---|---|---|---|
| **control**: `wake_array`'s own, uniform | $1.0$ | $1$ | $1.000000$ | $1.000000$ |
| the host, uniform | $0.5$ | $1$ | $\mathbf{0.500000}$ | $0.500000$ |
| the host, the settled inflow at the site | $0.5$ | $1$ | $0.537342$ | $\mathbf{0.500000}$ |
| the host, the settled inflow, **the disk re-sized to its face** | $0.5$ | $0.5$ | $1.074684$ | $\mathbf{1.000000}$ |

**The shortfall is exactly the disk's declared width.** Tier 46 re-declared the rotor's *ports* for their host — the host's cutoff and cell measure — and left the rotor *itself* a one-diameter disk. The powertrain therefore receives, through its shaft, twice the power the flow gives up through the face, and its balance cannot close. Re-sized, what remains is the inflow's non-uniformity, the difference between $\overline{u^3}$ and $\bar u^{3}$ in the wing's wake — a property of the model, not a leak.

And the two declared faces carry no power at all. At the rotor's own base both are linearized at the same inflow, the upstream face returning $+\tau$ and the downstream $-\tau$: **their net power is exactly zero**, against a shaft power of $0.785$. Evaluated at the two host rings instead it is $0.0077$. **The power the shaft delivers enters through no declared port** — W197's orientation question, seen as power.

## 4.3 What closing J3 costs: the powertrain has no operating point

`CircuitSolve` finds the operating point by balancing the disk's torque against the machine's; it builds the disk as `ActuatorDisk(a=a)`, at the default width, and cannot be told otherwise. The same solve with the width declared — which at width $1$ reproduces Tier 46's operating point exactly, the control:

| rotor | shaft speed $\omega$ | induction | loop current | torque demanded | largest torque in the clamp | operating point |
|---|---|---|---|---|---|---|
| width $1$, as `powertrain` declares it | $13.837$ | $0.114$ | $0.146$ | $0.0146$ | $0.0611$ | exists — demand $0.24$ of supply |
| width $0.5$, its host face | $27.674$ | clamp edge, $0.40$ | $4.758$ | $0.476$ | $0.0153$ | **none — demand $31\times$ supply** |

A narrower disk at the same inflow spins faster — the declared tip-speed ratio fixes $\omega = \lambda U_d / r$ — so the back-EMF doubles, the loop current the circuit then carries rises thirtyfold, and the torque that current costs is thirty-one times what a half-width disk can deliver anywhere in its induction clamp. **The powertrain was sized for a rotor one diameter wide.** Closing J3's balance is not a bookkeeping fix: it is choosing a rotor and a machine that fit each other in this flow — a design decision. That is W199, and W196's reach once more: the width is one more state J3 would re-derive, and it moves the machine off its envelope.

## 4.4 J1 — no power crosses, so the clause has no subject

The radiator core's pressure drop takes $0.304$ (in the tiling's units, per unit span) of work from the air, and that energy goes nowhere any record can name: the air carries no temperature and nothing can declare the core's dissipation. What J1 carries is a *parametric* coupling — the radiator's conductance follows the air through it — so its check is the coolant loop's response:

| | the declared radiator | the core in the flow, at the build state | the core in the flow, air $+10\%$ |
|---|---|---|---|
| return temperature | $321.059$ K | $321.059$ K | $319.708$ K |
| radiator rejection | $916.8$ W | $916.8$ W | $928.6$ W |
| $UA$ | $42.0$ W/K | $42.0$ W/K | $45.33$ W/K |
| loop balance, block first law | $1.6\times10^{-14}$, $1.5\times10^{-10}$ | the same | $2.5\times10^{-14}$, $1.5\times10^{-10}$ |

Identical at the build state, the control; and $10\%$ more air lowers the return temperature by $1.35$ K and raises the radiator's rejection by $11.8$ W. The loop balance is an identity of the fixed point, as `LoopSolve`'s docstring says; the block's first law is the check that can fail, and passes.

---

# 5. The gate, proposed

Rung 9's *capability* is **Claim B under real load**: adding subsystems to a vehicle costs a constant amount of integration work and adds composition error that does not compound. A gate should test that and nothing else, on a graph that has it:

| # | clause | status |
|---|---|---|
| **G1** | **Integration work**: each join costs $O(1)$ declarations, constant in the family count | **met** — [[joining-seam-cost]], qualified by W196 |
| **G2** | **Physical validity at the joins**: each join's receiving subsystem closes its own balance with the join's term and fails without it, **over a march** | **measured at the release state only**: J2 **passes**; J3 **fails** by the disk's declared width, and cannot pass with the powertrain as sized (W199); J1 carries no power, and its parametric check passes |
| **G3** | **Composition error at vehicle scale**: each join's error against a tightly coupled referent of the same union, and whether the joins' errors add | **not built** — needs a march (§6) |
| — | ~~composition error sub-linear in $N$~~ | **withdrawn**: no variable on this graph (§2). Answered where $N$ is a dial by CS-7; the physics-held-fixed exponents are the numbers to quote |
| — | global $\mathcal R(t)$ | **kept as a monitor, not a clause**: not assemblable from the records (W200), and blind to the couplings under test wherever it has been measured |

**Rung 9 is therefore built as a compile, measured at its joins' release state, and not complete.** It may not be marked complete on G1, and G2's one failure is a design fault rather than a framework one.

---

# 6. What a march of the union needs first

Each of these is a decision, not a computation, and G2 and G3 cannot run until they are made:

1. **Time and length units** (W201) — how the tiling's convective units and `wake_array`'s relate to seconds and metres, so that three clocks can be compared at all.
2. **A rotor sized for its host, and a machine sized for that rotor** (W199).
3. **Dissipation and domain-boundary power**, declared — or each join's receiver balance written by hand, as CS-12 wrote its own (W200).
4. **The down faces' orientation** (W197), or the rotor's power path through its shaft cannot be read off its ports.
5. **The devices applied as body forces** in the march, the way `wake_array`'s own march applies its disks (W94) — the ring-segment seams are a declaration, not a forcing.
6. **The shared operating point**, solved across the union or declared per subsystem (W196).
7. **`L7/R9` scoped to the multirate seams** (W194), so the march is not refused for clocks no seam joins.

**[AI Inference]:** items 1 and 2 are not framework work at all. They are the choice of *which vehicle this is* — its speed, its front wing's chord, its rotor — and none of the three case studies was written at one physical scale. The vehicle graph exposed that the ladder built three subsystems of three different machines.

---

# 7. What this tier did NOT do, named

- **Nothing marches the union.** G2 is measured at the release state only; G3 is not built.
- **W198–W201 were opened and not fixed.** The rotor is not re-sized in `integration_union`, `CircuitSolve` is unchanged, and no unit or dissipation field was added.
- **The pathmap's rung table keeps its original gate.** The restatement lives here and in [[f1-pathmap-and-end-goal]] §3.3's row, as W183's did for rung 3.
- **The fluid's own energy budget was not attempted.** The classical tiling has a single $\nu$, so OP-4's obstruction does not apply to it, but its boundary and forcing terms have no declarations to be written from.
- Nothing downloaded, no machine rented, NeuberNet not loaded, nothing pushed.

## See Also

- [[joining-seam-cost]] — the graph this page holds the gate against, and G1's measurement.
- [[case-study-scaling-ladder-atlas-0.1]] — where "sub-linear in $N$" was measured with $N$ as a dial, and the physics-held-fixed control this graph cannot run.
- [[case-study-learned-pair-atlas-0.1]] §9 — the precedent: rung 3's gate restated rather than substituted (W183).
- [[port-algebra-atlas-0.1]] §6 — the residual whose terms §3 checks, and [[composition-error-theory]] §2.2, why it is a falsifier and not a bound.
- [[case-study-thermal-strain-atlas-0.1]] §6 and [[case-study-wing-fsi-atlas-0.1]] §6 — the receiver-balance rule G2 is written in.
- [[case-study-wake-array-atlas-0.1]] and [[case-study-powertrain-atlas-0.1]] — the rotor and the machine J3 joins, and why they do not fit the host.
- [[open-problems-atlas-0.1]] — OP-4, rung 2's $\mathcal R$ gate closed as not measurable.
- [[gap-worklist]] Tier 47 and [[case-study-ladder-to-f1]] §21 — W198–W201, and where rung 9 stands.
