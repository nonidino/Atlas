# CS-14 — a drivetrain on the wake array's open shaft

**Rung 7 of [[case-study-ladder-to-f1]]'s climb.** `atlas/cases/powertrain.py`, `tests/test_tier37_powertrain.py`. Built 2026-09-09.

`wake_array.rotor_capabilities` declares a `shaft:ROT` port and says of it, in its own note:

> unconnected: no drivetrain. The extracted power leaves the system here, and under the port algebra that is an OPEN PORT with a measurable power flow rather than an absence.

**This case study connects it.** The rotor's shaft drives a motor-generator; the machine sits in a DC circuit with a bus, a battery and an inverter; and the power that used to leave the system is followed to where it is stored and to where it is lost.

```
   ROTOR --ROT--  MGU  --ELEC-->  BUS  --ELEC-->  BATT
                   ^                                |
                   +---------- INV <----ELEC--------+
```

---

## 1. What only this case study can say

### 1.1 The port vocabulary closes, and two types were open rather than one

**Censused rather than recalled.** `MECH`, `THERM` and `ADVEC` had carried real connections. `ROT` had been declared **three** times — `wind_farm`, `wind_farm_real`, `wake_array` — and every one of those declarations carries the note *"unconnected: no drivetrain"*, so the type had never appeared in a `Connection`. `ELEC` had never been declared **at all**: `rocket.py`'s own docstring says so in its fourth line, and correctly, because a solid-propellant rocket has no electrical bond.

This graph gives both their first seam. **All five of the closed vocabulary are now exercised**, and the two that were open were opened by the same seam pair.

**The cost was one scale set and one prolongation each.** That is the whole content of the affordability claim — the port algebra promises that the twentieth expert costs what the fourth did, $O(K)$ declarations rather than $O(K^2)$ adapters — and until this session that promise had never been tested past two governing families. It now runs on four: incompressible Navier–Stokes, plane-stress elasticity, heat conduction, and a lumped DC circuit.

One thing the type forced, and it is a real declaration rather than a formality: `PORT_SPECS[ELEC]`'s flow is a **current density** and its power is per unit area, so a lumped circuit terminal needs a declared cross-section for the power identity $s_e s_f = s_P$ to close. A real terminal has one; declaring it is what lets `check_scales` verify the set from the port list alone.

### 1.2 The first cross-domain energy balance that is not an identity

CS-13's loop balance closes to $2\times10^{-14}$ and reduces, on substitution, to $T_{\text{return}} = T_{\text{return}}$ — [[case-study-cooling-loop-atlas-0.1]] §2.2 says so rather than leaving it implied. **This one does not reduce.**

The mechanical side is read off the build repo's own `disk.ActuatorDisk`, which contains no circuit. The electrical side is solved from Kirchhoff's laws, which contain no wake. They meet through exactly one thing:

$$k_e \;=\; k_t$$

the machine's back-EMF constant and its torque constant, equal because both are the same air-gap flux linkage seen from the two sides of it. So

$$T\omega \;=\; (k_t I)\,\omega \;=\; (k_e \omega)\,I \;=\; V_{\text{emf}} I$$

is a **theorem about the machine**, not a definition anybody chose — and a theorem the code cannot break is a theorem the code is not resting on. Pulling the constants apart breaks the balance in exact proportion: at $k_e/k_t = 0.95$ and $0.90$ the bridge residual comes out at $0.05$ and $0.10$ to machine precision.

| quantity | value |
|---|---|
| shaft speed $\omega$ | $15.0$ |
| induction $a$, **solved** | $0.285714$ |
| torque, rotor (from `disk.py`) | $0.0533333$ |
| torque, machine ($k_t I$) | $0.0533333$ |
| mechanical power $T\omega$ | $0.800000$ |
| $\sum_i I^2R_i$ | $0.0853333$ |
| $V_{oc}I$, into the chemistry | $0.714667$ |
| **energy-balance residual** | $\mathbf{1.9\times10^{-15}}$ |
| **bridge residual** ($k_e = k_t$) | $\mathbf{1.8\times10^{-15}}$ |
| KVL residual | $2.1\times10^{-17}$ |
| closed form vs relaxation | $1.4\times10^{-14}$ |

### 1.3 W163's second graph, on unrelated physics

A DC circuit is a loop — current leaves the machine, crosses the bus, passes the battery and returns through the inverter — so the four electrical agents form a **directed cycle**, exactly as CS-13's coolant does.

**Compiled with the return conductor and without it, the graph gets the same verdict, the same rule set and the same per-agent decisions.** The only difference in the whole decision record is the extra seam's own per-seam rows. CS-13 found this on a coolant loop with `ADVEC` and a conduction solver; this finds it on a DC loop with `ELEC` and an actuator disk. One graph is an anecdote; **two on unrelated physics is the statement that no layer of the compiler looks at loop topology**, which is what W163 said it was waiting for.

The rule is still not written. Writing it is the next tier's decision, not a side effect of this one.

### 1.4 The first `validity` predicate in this package to catch a real defect

Most `validity` callables in this vault are `return True`. Two here are not, and one of them earned its keep on the first build.

**`MachineAgent.validity`** requires $k_e\omega > V_{oc}$: a generator can only push current into a battery whose open-circuit voltage its back-EMF exceeds, and below that speed the loop current reverses and the machine motors instead. The first version of this module gave the machine SI constants against `wake_array`'s **nondimensional** shaft, and the predicate declined — $k_e\omega = 0.19$ against a $3.6$ V battery. **It caught a units error across a seam**, which is precisely what the nondimensionalisation half of the port algebra exists for. The module is now nondimensional throughout, on both sides of the shaft, and the dimensional reading lives in the scale set.

**The rotor's own bound is the donor's.** `disk.py` clamps the induction factor at $0.4$ — momentum theory breaks down in the turbulent-wake state and the donor refuses to extrapolate — so an operating point outside the clamp does not exist. The result reports `rotor_valid = False` rather than returning the nearest edge as though it were a solution.

---

## 2. The operating point is a solve, not a pair of numbers

The rotor's torque is a function of the induction its controller sets; the machine's is a function of the shaft speed, through the circuit. The operating point is where they agree:

$$T_{\text{rotor}}(a, u) \;=\; k_t\,\frac{k_e\omega - V_{oc}}{R_{\text{total}}}, \qquad \omega = \lambda u$$

solved by bisection on `disk.py`'s own closed form, over the donor's own clamp. `V_{oc}` is the one constant here chosen by looking at the answer, and the reasoning is on the record in the module: at $1.30$ the equilibrium lands exactly on the disk's reference induction $a = 1/3$ and the search returns its starting point — an untested code path passing; at $1.28$ the demand exceeds what the clamp can deliver and no operating point exists. $1.34$ puts it at $a \approx 0.286$, inside with room and away from the reference.

The circuit itself is **solved rather than swept**, for CS-13's reason carried to a second cycle, and its closed form is *composed from the elements' own coefficients* rather than written out, so it cannot drift from the elements it is the closed form of. The relaxation runs beside it and the pair is the control.

---

## 3. What the compile said, including three findings about a module this one only reads

The graph compiles to **`admit-uncertified`** with no refusals. Three of its decisions are about `wake_array`'s rotor and could only be surfaced by connecting a port that module left open:

**`L4/block-share` — the shaft seam is one-sided by construction.** `RotorDisk.respond` ignores the trace on `shaft:ROT` entirely: an actuator disk at fixed induction and fixed inflow delivers a torque that does not depend on the speed you ask of it, because its $\omega$ is tied to the inflow rather than to the load. So the rotor's block is identically zero and the machine carries the whole seam operator. That is **W97**'s class — a seam one-sided by construction, whose substitution certificate is therefore blind — arriving on a `ROT` port.

**`L4/probe-base` — the two sides linearize about different speeds.** `RotorDisk.probe_base` returns zeros for `shaft:ROT`, which is a *stalled* turbine, while the machine declares the operating speed. **W74** on a third seam, and it belongs to `wake_array` rather than to this module. Left as it is rather than repaired, because the instrument that would settle it exists: the rotor's response is constant in the trace, so `probe.base_sensitivity` would clear the decertification outright. That it is not wired into this compile is the finding.

**`L2/R10` — W160 on a third graph.** An actuator disk declares the flow's `governing_family` (correctly — an algebraic closure inside a continuum problem is not a different continuum problem) with `stencil_radius = 0`, and the rule infers a hidden pressure solve. The front wing's suspension and CS-13's four coolant legs are the other two. **A row that fires on three unrelated graphs is about a class**, not about one agent.

**And one finding is this module's own.** The first version flipped the sign on each element's `lo` terminal, which made one of the two blocks at every node negative and left `L4/E7/passivity` reporting a defect of $1.5\times10^{-5}$ on `mgu_bus` — **W138's shape on a third port type, caught by the rule this session taught to see it.** With each terminal reporting in its own element's sense, the assembled operator at a node is the *series* resistance of the two elements meeting there, which is positive because a resistive network is passive. No `effort_normal` is declared on the `ELEC` seams: this really is the own-outward convention.

---

## 4. What this does not buy

**Four of the five agents are algebra.** Only `ROTOR` is a real solver, and it is the build repo's zero-parameter closed form rather than anything trained. `L4/operator-content` decertifies every seam here, correctly: a lumped block is a scalar and a scalar is trivially a multiple of the identity.

**No constant is measured on this graph.** $L$, $\sigma$, $\tau$, $C_\mu$ and $\lVert\mathcal A\rVert$ are all on `unmeasured` and the record says so rather than carrying another graph's numbers.

**The circuit is quasi-steady.** No inductance, no switching, no state of charge dynamics. That is honest for a graph whose purpose is a port type and a topology, and it is why the battery's `stored` is $V_{oc}I$ rather than an integral.

---

## Links

- [[case-study-ladder-to-f1]] — the climb this is rung 7 of
- [[case-study-cooling-loop-atlas-0.1]] — rung 6, and W163's first graph
- [[gap-worklist]] — Tier 37; **W163** gets its second graph, **W160** its third
- [[port-algebra-atlas-0.1]] — `ELEC`'s bond, and the $O(K)$ declaration count this tests
- [[case-study-wake-array-atlas-0.1]] — the module whose open port this connects
- [[master-error-bound]] — the constants this graph does not measure
