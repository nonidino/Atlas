# CS-13 — a closed coolant circuit, and the first directed cycle

**Rung 6 of [[case-study-ladder-to-f1]]'s climb.** `atlas/cases/cooling_loop.py`, `tests/test_tier36_cooling_loop.py`. Built 2026-09-09.

A solid block is cooled by a **closed** liquid circuit — passage, hot line, radiator, cold line and pump — that returns to the passage it left. One real expert (`thermostruct2d.ThermoStruct2D.step_thermal`, the build repo's Q1 conduction solver, imported unmodified) and four **lumped** legs of the `disk.ActuatorDisk` class: closed form, zero fitted parameters, four lines of algebra each.

```
   BLOCK  ──THERM──  PASS  ──ADVEC──▶  HOT  ──ADVEC──▶  RAD
                      ▲                                   │
                      └──────── COLD ◀────ADVEC───────────┘
```

---

## 1. What only this case study can say

**It is the first graph in this package in which information circulates.**

[[spec-wind-farm-wake-atlas-0.1]] §5.1 raised path-dependence of message passing as a real concern — *"$I\to N\to F\to W$ and $I\to B^+\to\ldots\to W$ are two distinct paths between the same agents. Information can circulate"* — and then the wind-farm implementation log **withdrew** the one measurement that had been read as evidence for it: the mirror residual was the frozen operator's own asymmetry, reproduced with no graph, no ports and no agents anywhere in the loop. The log's own words are that sweep-order path dependence *"may still be present — nothing here rules it out. What is ruled out is that it is NEEDED."*

So the concern has been open since 2026-08-23 with **no graph that can test it**, and the reason is specific. Every cyclic graph this package has built is an **overlapping tiling**, where the cycle is undirected and the scheme is **additive**. `front_wing`'s six fluid windows contain the 4-cycle $F00 - F10 - F11 - F01$, and it tests nothing: all six windows step from the same assembled field and are then blended, so no window's input is another window's output, there is no sweep, and there is no order for an answer to depend on.

A coolant loop is the opposite, and it is the minimal case. The cycle is **directed**, it is carried by `ADVEC` — a mass flux, which has a direction — and every agent's inlet is another agent's outlet. **There is no agent that can go first.** That is a property of the topology rather than of any expert.

---

## 2. What was measured

| quantity | value | what it says |
|---|---|---|
| spread over 8 sweep orders, **1 sweep per macro-step** | $\mathbf{1.54\ \mathrm K}$ ($4.8\times10^{-3}$ relative), 5 distinct answers | a composition layer that sweeps a cycle once per step has made a modelling choice **nothing in the declaration licenses** |
| the same, after 8 sweeps | $1.32\ \mathrm K$ | not a transient anybody outruns in practice |
| the same, **at the fixed point** | $3.5\times10^{-12}\ \mathrm K$ ($1.1\times10^{-14}$ relative) | the cure is not an ordering rule |
| per-sweep contraction of the spread | $0.97405$ (4 legs), $0.9618577$ (3 legs) | the path dependence decays at a rate the circuit's **own dissipation** sets |
| loop balance $Q_{\text{block}} + W_{\text{pump}} = \sum_i Q_{\text{rejected},i}$ | residual $2.0\times10^{-14}$ | the ladder's gate — **and it is an identity of the fixed point** |
| the block's own first law | residual $3.6\times10^{-7}\ \mathrm W$ against $1787\ \mathrm W$ | **the gate that can fail** |
| iterated vs closed-form fixed point | $1.7\times10^{-16}$ relative | the positive control on the solve |

### 2.1 The finding: a swept cycle is order-dependent and a solved one is not

Eight equally defensible schemes — four rotations of the circuit, two directions — give **five distinct answers** at one Gauss–Seidel sweep per macro-step, spread over $1.54$ K. Nothing in any declaration prefers one of them; the choice belongs to whoever wrote the loop.

Iterate any of the eight to convergence and they agree to $1.1\times10^{-14}$ relative. **So the cure is not an ordering rule: it is solving the loop instead of sweeping it**, which is [[general-coupling-scheme]] §4.2's rule — *write the residual in the port's own conjugate variables and solve it there* — applied to a cycle rather than to a seam.

**[AI Inference]:** that generalises only as far as the loop map being a contraction. Here it is, and demonstrably so: the composed gain is the product of the legs' own $a$ coefficients, strictly below one because the radiator dissipates. A circuit with no dissipative leg has a unit-gain loop map, no isolated fixed point, and nothing for an iteration to converge to — `LoopSolve._closed_form` raises there rather than returning a number, which is a statement about the circuit and not a solver failure. Where the boundary between those regimes sits is **not measured here**.

### 2.2 The gate is an identity, and the page says so

$Q_{\text{block}} + W_{\text{pump}} = \sum_i Q_{\text{rejected},i}$ closes to $2.0\times10^{-14}$, and it is worth being exact about its status: **substituting the legs' own definitions turns it into $T_{\text{return}} = T_{\text{return}}$.** It is a consistency check on the arithmetic — a sign error in any leg's rejection breaks it, and that is asserted — and it is **not** evidence that the coupled physics is right.

What can fail on physics is the block's first law,

$$\frac{\mathrm d}{\mathrm dt}\int_{\Omega}\rho c_p T\,\mathrm dV \;=\; Q_{\text{outer}} - Q_{\text{wall}},$$

taken across one `step_thermal` through the same face machinery the loop uses. Its floor was **measured rather than assumed**: swept over four decades of $\Delta t$ the relative residual runs $8.1\times10^{-14}$, $3.3\times10^{-8}$, $4.1\times10^{-12}$, $5.3\times10^{-11}$ — **non-monotone**, which is the signature of an iterative solver's tolerance and not of a discretisation error, since the latter would have a trend.

The gate's teeth and its blind spot are both pinned. A **sign error** on the wall flux and a dropped **out-of-plane width** each move the residual by $10^6$ times the residual the gate passes at. Reading the **outer** face where the wetted one was meant moves it by $1.7\%$ — the block is nearly uniform through its thickness at the release state — so this gate does **not** catch that one, which is stated in the test rather than left to be assumed.

### 2.3 A seven-figure identification, measured and then withdrawn

On the **three**-leg circuit the per-sweep contraction of the inter-order spread matched $\sqrt{\text{loop gain}}$ to **seven figures**: $0.9618577$ measured against $0.9618576$. That reads as an exact identification and was very nearly written down as one.

On **four** legs it does not hold. The rate is $0.97405$ over 256–512 sweeps and $0.97381$ over 512–1024, against $g^{1/3} = 0.973926$ — agreeing to about $10^{-4}$ and **drifting** — and the spread underflows to bitwise zero by 1536 sweeps before any asymptote is reached.

**The match was true as far as it was marched.** What survives is a bracket: the rate is strictly inside $(g,\,1)$ and within $10^{-3}$ of $g^{1/(n-1)}$ at both leg counts, which is all the claim in §2.1 needs.

---

## 3. What the compiler said, including what it did not

The four-leg circuit compiles to **`admit-uncertified`** with no refusals. The decisions are the ones a multiphysics graph with no measured constants should get: `L1/E3` fails at the wall (a genuine two-family seam), `L2/R10` fires on all four lumped legs (**W160**, the same false alarm as the front wing's suspension, arriving unprompted on a graph it was not diagnosed on), `L4/operator-content` decertifies every seam, and `L8/W56` reports five unmeasured constants.

### 3.1 The compiler cannot tell a circuit from a chain — **W163**

Compile the graph with the return seam and compile it without: **same verdict, same rule set, same per-agent decisions.** The only difference in the entire decision record is the extra seam's own per-seam rows.

So nine layers of admissibility currently see a directed cycle as a list of independent seams — while the measured consequence of the cycle is $1.5$ K of order dependence at one sweep per macro-step, which is larger than several defects the framework does refuse over.

**No rule was written for it here, deliberately.** A rule invented the same day the first cyclic graph appeared is a rule validated on one graph, which is the failure mode this whole direction exists to avoid. **[AI Inference]:** the shape it probably wants is not an ordering constraint but an admissibility condition on the loop map — *a directed cycle is admissible when its composed gain is a contraction* — which is decidable from the same declared response the probe already builds, and which `LoopSolve._closed_form` already raises on when it fails. Not built, and it should be written against a second cyclic graph.

### 3.2 A triangle is not a cross-point — **W162**

The **three**-leg version of this circuit is **refused** at `L2/I2/G1` for a multi-valued shared cell. `CaseGraph.detected_cross_points` infers a cross-point from a triangle in the agent adjacency — a good proxy on a tiling, where three subdomains meeting pairwise really do share a corner, and wrong on a circuit, where three legs joined by three **distinct planes** share no point at all.

And the graph has no way to say so. The escape is `cross_points`, and an empty tuple is falsy, so `cross_points=()` falls through to detection: **a graph can declare which cross-points it has and cannot declare that it has none.**

Four legs is also the more honest circuit — a cooling loop has a hot line and a cold line, not a radiator bolted to a pump — so the fix was the better model. The three-leg build is kept reachable as `cooling_loop.build(n_legs=3)`, because a finding whose reproduction is a paragraph is not reproducible.

---

## 4. The four declaration errors the first compile found

Recorded because they are the case study's own first result, and because each is a thing [[atlas-implementation]]'s guide warns about, arriving:

1. **`L3/C4`** — `ADVEC` is a **multibond** and needs a conjugate pair *per passenger* on top of the base triple. With one passenger the two coincide, which is exactly why omitting it looked harmless; a two-passenger port written the same way would silently price the second at the first's scale.
2. **`L4/null-space`** — 16 modes declared on a **lumped** leg with one state, so the probed block had 15 null directions and the rule reported all fifteen. `PORT_SPECS[ROT]`'s own note says the shape: *"a lumped port is the case $\dim M = 1$"*. This is the guide's *"claiming resolution nobody measured"* with the arithmetic to prove it.
3. **`L7/R9`** — a $50\!:\!1$ clock ratio the legs do not have. $T_{\text{out}} = T_{\text{in}} + Q/(\dot m c_p)$ contains no time; a quasi-steady algebraic leg's native step **is** the macro-step.
4. **`L4/probe-base`** — **W74** again. `probe_base` is one callable per *expert* and takes the *port*, and a leg carries two kinds of port whose traces are different physical quantities. One base for both put $320$ K against $500\ \mathrm{kg/(m^2\,s)}$ on the same seam, and the rule reported the two sides as $56\%$ of the base norm apart — which they were.

**The compiler found all four before anything ran**, which is the one claim [[poc2-novelty-audit]] §2 says survives, exercised on a graph written after that audit.

---

## 5. What this does not buy

**It adds a topology and a port type at scale, not physics.** The four legs are closed-form algebra; only `BLOCK` is a real solver, and its seam is a conjugate-heat-transfer seam of the kind [[case-study-thermal-strain-atlas-0.1]] and `brake_thermal` already have. `L4/operator-content` decertifies the wall seam for the same reason it decertifies `thermal_seam`'s — the film coefficient dominates the conduction operator by $10^3$ — which is **W68** reproducing on a third seam rather than a new result.

**No constant is measured on this graph.** $L$, $\sigma$, $\tau$, $C_\mu$ and $\lVert\mathcal A\rVert$ are all on `unmeasured`, and the record says so rather than carrying another graph's numbers.

---

## Links

- [[case-study-ladder-to-f1]] — the climb this is rung 6 of
- [[gap-worklist]] — Tier 36 opens **W162** and **W163**; **W160** reproduces here
- [[general-coupling-scheme]] §4.2 — solve the residual in the port's own variables
- [[port-algebra-atlas-0.1]] — `ADVEC`'s passenger list, and `THERM`'s true bond
- [[spec-wind-farm-wake-atlas-0.1]] §5.1 — the path-dependence concern this graph tests
- [[poc2-novelty-audit]] — the audit whose one surviving claim this exercises
- [[case-study-thermal-strain-atlas-0.1]], [[tier0-measurements]] §14 — the film-coefficient result this reproduces
