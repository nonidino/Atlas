# The body-fitted column declared: a volumetric device gets a port

*Case study, PoC 3. New 2026-09-15. Tier 65.*

---

# 0. The result, in one paragraph

**Tier 64 marched the devices and the joins on the body-fitted car; nothing about that march was declared, so the compiler had never seen the column and gate clause P1 could not be stated on it. It can now: the body-fitted column compiles, and it refuses at `L7/R9` and nothing else — CS-18's table, on a third graph whose fluid is ONE expert instead of fourteen windows.** Getting there was the tier's content, because three obvious declarations all fail and the failures are informative. An expert with no ports is refused by the record layer itself; declaring each overset grid as an agent would be a lie, and the composite's $12{,}078$ interpolation rows measure why; and putting the device's two faces on the fluid — the porous column's own shape — is a seam from an agent to itself, which the graph layer refuses by name. **What does work is a strip port**: a device applied as a body force presents the traction its force integrates to, $T/W$, against the band's mean velocity, and their product is the power the force takes out of the fluid. That is the first expressible bond for a volumetric device in this package (**W94**, open since Tier 18), and it works here because *this host accepts a body force and reports the work it does*. Six of seven registered predictions held. The seventh is the sharpest thing in the tier: the declared bond reproduces the march's shaft power to **sixteen digits** when it is evaluated where the march evaluated it, and is $2.4\%$ low at the single state the record is written at — a capability record describes an expert at a state, and this one is quoting a window.

---

# 1. Three declarations that do not work, and what each one measures

## 1.1 An expert with nothing to declare

The composite's couplings are a rolling road, an inlet, a top and an outflow — all domain boundaries, which carry no port (W200's standing hole) — plus two devices, which are body forces. That is zero ports, and `capability.ExpertCapabilities` raises before any compile:

> `expert 'FLUID' declares no ports`

The refusal is right and it is a wall: a column that couples to everything through body forces and to nothing through a boundary cannot enter a graph at all.

## 1.2 One agent per grid

The porous column declares its fluid-fluid overlaps as ordinary artificial boundaries, because each window is marched by its own expert and the results are blended. **The overset composite is one implicit solve**: its interpolation equations are rows of the same matrix as its momentum equations, so a grid cannot be stepped alone and there is nothing to exchange, lag or blend. Measured on the car's own composite:

| | |
|---|---|
| grids | $12$ (a Cartesian background and eleven body-fitted) |
| unknowns | $377{,}267$ |
| interpolation rows | $\mathbf{12{,}078}$, $3.2\%$ of them |
| donor entries those rows read | $108{,}702$ |
| distinct grid pairs coupled | $40$ |

**That is the answer to W268, and it is a decision rather than a measurement of one**: an overset overlap is not a seam, it is inside the expert. What the declaration then cannot say is that the overlap exists at all.

## 1.3 The device's two faces

On the porous column the tiling is cut AT each device plane, so `integration_union` connects the device's up and down faces to two different windows. Nothing cuts the composite there, so the same declaration is a seam from the fluid to itself — and the graph layer refuses it by name:

> `seam 'J3_rotor_self' connects agent 'FLUID' to itself`

---

# 2. What does work: the strip port (W94)

A device applied as a body force over a strip of thickness $\Delta$ presents the traction its force integrates to:

$$\text{effort} \;=\; \frac{T}{W}\ \ [\text{Pa}], \qquad \text{flow} \;=\; \langle u\rangle \text{ over the band}\ \ [\text{m/s}], \qquad \text{effort} \times \text{flow} \times W \;=\; T\,\langle u\rangle,$$

which is exactly the power Tier 64 measured against the shaft's claim to $3.4\times10^{-4}$. So the strip **is** a MECH bond — the one `wake_array`'s disk has never had against a frozen operator (W94, open since Tier 18) — and the reason it is expressible here is that this host accepts a body force and reports the work it does.

Two things the declaration had to get right, and the compiler caught both:

- **both sides of a MECH seam return the EFFORT** in this package. Declared the other way round (the fluid returning its velocity), the compile refuses at `L3/C9`: the two responses are added, so they must be the same half.
- **the port's trace is the DEVICE's resolution, not the grid's**: `CoreRadiator` and `wake_array.RotorDisk` both refuse any length but $32$, while the band they span is $29$ grid cells. So this port is **non-conforming by construction** — $32$ device cells over $29$ grid cells, $15$ modes each side — which is what `effective_resolution` and the declared prolongation are for, and what the porous column never had to say because its band was $32$ cells of its own lattice.

The second of those is still in the record as evidence: `racelab14.json` carries a `stage_errors.power` traceback ending *"ROTOR: `u_ref` has 29 cells, expected 32"* from the attempt that found it. The stage was repaired and re-run — `stage_wall_s.power` is the successful one, and the `power` block beside it is complete — so the traceback is **provenance, not a live failure**; the record keeps its failures rather than editing them out.

The fluid's own response is **direct forcing**: given a velocity across the band, the traction that would put it there within one step, $\rho\,(u - u_{\text{now}})\,\Delta/\mathrm{d}t$. It is the march's arithmetic read backwards — `car_union` hands the fluid a force and reads the velocity that follows — and it is explicit, which is named as a hole rather than hidden.

---

# 3. The compile

| graph | agents | seams | verdict | refusals |
|---|---|---|---|---|
| **body-fitted, joined** | $\mathbf{11}$ | $\mathbf{13}$ | refuse | $\mathbf{[L7/R9]}$ |
| body-fitted, disjoint | $11$ | $10$ | admit-uncertified | none |
| body-fitted, clocks reconciled | $11$ | $13$ | admit-uncertified | none |
| body-fitted, J1 only / J3 only | $11$ | $11$ | refuse | $[L7/R9]$ |
| **porous column (Tier 51), joined** | $26$ | $36$ | refuse | $[L7/R9]$ |
| porous column, disjoint | $26$ | $33$ | admit-uncertified | none |

**The refusal is the clocks and not the joins** — reconciling them removes it, and one join alone still earns it — which is CS-18's own finding reproduced on a graph that shares none of its fluid structure. The joined column's **$189$ decisions are $156$ admits, $32$ decertifications and one refusal**; **no seam earns a refusal of its own**, the two strip seams included, and what they do earn is the package's standing decertification set (`L1/C8`, `L3/C8`, `L4/operator-content`, `L4/probe-base`).

> **Corrected 2026-09-21 (Tier 82, W308), and the earlier numbers were $191$/$34$ with `L4/E7/passivity` in that set.** Two of this graph's three seams declared `effort_normal` **inverted** — `J1_core_strip` named `RAD` and `J3_rotor_strip` named `ROTOR`, each making the assembled operator *negative definite*, which is passive only up to a global sign that $S\lambda = \chi$ is indifferent to and `L4/E7/passivity` is not. `J2_heat` already named the right agent, and that is the control that made it a finding rather than a blanket flip. Naming the fluid side on both makes all three **positive definite with every singular value bitwise unchanged** (measured: maximum relative difference $0.0$), so **every $\beta$ and $\kappa$ on this page is untouched** — only E7's test moves, and it stops firing on both strip seams. The refusal was never theirs: it is the clocks, as the paragraph above says. Found by W306 at Tier 78, deferred there so this sentence would be changed deliberately rather than as a side effect of a diagnostic tier, and audited before it was.

**The declaration is less than half the size of the porous column's** — $11$ agents against $26$, $13$ seams against $36$ — and every seam it loses is a fluid one: thirteen window-to-window seams, the wetted seam and the mount seam all vanish into the single expert, and what is left on the fluid side is one seam per device.

---

# 4. What the compile found in the compiler (W285)

R10 refuses an agent that embeds a global elliptic solve **when the graph decomposes the domain**, and since W114 it checks that premise through a proxy: *another agent of the same `governing_family`*. On this column the proxy was met by the **actuator disk**, which declares `incompressible-navier-stokes-2d` with `stencil_radius` $0$ and owns no region at all — so R10 refused a composite that nothing decomposes.

W160 narrowed the branch beside this one for exactly this reason, by exactly this signature, and left this one. It is now narrowed the same way: **an agent with no spatial operator is not a piece of a family's region**. The price is measured rather than assumed:

- the porous column's compile is **unchanged**: $26$ agents, $36$ seams, refusals $[L7/R9]$, its disjoint union refusing nothing;
- the whole suite passes;
- of the forty captured artifacts the live control recompiles, **one moves**: `front_wing`, where the lumped suspension joins the structure in the halo rule's `uncut_agents`. **No verdict moves anywhere.** It is named in `MOVED_BY_W285` with the reason and the control asserts it moved *for that reason*, which is the discipline W194 set.

---

# 5. The bond against the march it describes

D5 asked whether the declared bond reproduces Tier 64's number, and the answer is two numbers:

| | ratio to the march's shaft power |
|---|---|
| the record's effort at the band it is written at | $\mathbf{0.9762}$ |
| the same bond, evaluated step by step over the window | $\mathbf{0.9999999999999999}$ |

**The declaration is the march's arithmetic exactly** — it calls the same `operating_point` on the same disk — and the $2.4\%$ is entirely *one state against a window*: the window's inflow moves $1.00$ to $1.035$ of the sizing and the disk's induction with it, from $0.082$ to $0.202$, so a power that is cubic in the inflow cannot be read off its mean. D5 registered $1\%$ at one state and **failed**; that failure is W229's theme (a quantity evaluated at a build state and assumed over a trajectory) measured on a capability record rather than on an envelope.

---

# 6. The predictions

Registered before the stages ran; the exploration that preceded them is in the record's `read_before_this_run`.

| id | claim | outcome |
|---|---|---|
| D1 | the narrowing leaves the porous column's compile exactly as it was | **held** — $26$ agents, $36$ seams, $[L7/R9]$ |
| D2 | the whole suite passes with the narrowing in | **held** — $1628$ passed, $0$ failed (and see below) |
| D3 | the body-fitted column refuses $[L7/R9]$ and nothing else; disjoint refuses nothing; reconciling the clocks removes it | **held** |
| D4 | $11$ agents and $13$ seams against $26$ and $36$, no fluid-fluid seam | **held** |
| D5 | the strip bond's power is the march's shaft power to $1\%$ | **failed** — $0.9762$ at one state, $1.0000$ step by step (§5) |
| D6 | the two strip seams earn no refusal of their own | **held** — both admit-uncertified |
| D7 | a seam from the fluid to itself is refused by name | **held** |

### D2 is a fixed point, not a reading

**D2's evidence is the suite's own tally, and this tier's tests are inside that suite** — one of them asserts that every registered prediction has a verdict, and D2 has none until a tally is written. So the suite cannot go green until the number is recorded, and the number cannot be read until the suite is green. The first run reported $1626$ passed and $2$ failed, and **both failures were this tier's own bookkeeping**: every other file in the package was green on that run, which is the part of D2 the narrowing is actually answerable for.

The number the record carries was therefore *solved for and confirmed*, not read once: the tally was written, the whole suite was re-run against the record carrying it, and that run reported $1628$ passed and $0$ failed — exactly the tally in the record. Writing the green number without that confirming run would have been a fabrication dressed as a measurement.

One of the two first-run failures was **not** a fixed-point artefact and was repaired rather than predicted away: the test that checks this page against the record looked for `12078` after stripping commas, which can never match the vault's LaTeX `$12{,}078$` — the page was right and the assertion was wrong, and it now asserts the LaTeX form the way Tier 62 does.

---

# 7. What this tier did NOT do, named

- **Nothing was marched.** The declaration is checked against Tier 64's record, not against a new run, and no seam operator here is probed — the fluid's strip response is declared and never called by the compile.
- **The overlap is still undeclared** (W268 answered by decision): twelve grids are inside one expert, so the compiler cannot see the interpolation that joins them, and nothing checks it.
- **The domain's boundaries still carry no ports** (W200): the road, the inlet, the top and the outflow are where the column's momentum actually enters and leaves, and the graph says nothing about them.
- **The strip port's geometry is a strip, not a surface**: its outward normal is a declared direction and its measure is the band's width. Nothing checks that against the solver.
- **The fluid's response is explicit** — one step of direct forcing, with no pressure reaction — so it is a declaration of the right shape and not a probed operator.
- **P2–P7 are not re-run on this graph**: Tier 64 measured the balances the march can measure, and the gate's clauses are still written against the porous column.
- **No learned expert and no demo**: the background's rectangular sub-regions could carry Poseidon windows (W270) and nothing declares them.
- **Nothing was downloaded or installed**, no machine was rented, the unlicensed structural checkpoint was not loaded, and nothing was pushed.

```
python scripts/tier65_car_graph.py --out out/racelab14 --stages record,compile,power,summary
python scripts/run_suite.py            # D2 has no verdict yet, so the record test fails here
python scripts/tier65_car_graph.py --out out/racelab14 --stages summary --suite 1628 0
python scripts/run_suite.py            # 1628 passed, 0 failed -- the confirming run
python -m pytest tests/test_tier65_car_graph.py tests/test_tier45_region_assembly.py -p no:cacheprovider
python scripts/vault_scan.py wiki
```

About two and a half minutes, nearly all of it the composite the `record` stage builds to count its interpolation rows. The record is `out/racelab14/racelab14.json`.

---

## See Also

- [[poc3-racelab-car-union]] — Tier 64, the march this declaration describes, and the power the bond reproduces.
- [[case-study-vehicle-march-atlas-0.1]] — CS-18, the union's compile and the `L7/R9` table this reproduces.
- [[case-study-racelab-graph-atlas-0.1]] — CS-19, Tier 51: the porous column's graph, its 26 agents and its gate (P1–P7).
- [[atlas-implementation]] — the compiler whose R10 premise this tier narrows.
- [[gap-worklist]] — W285, and the rows this tier annotates: W94, W268, W200, W229.
