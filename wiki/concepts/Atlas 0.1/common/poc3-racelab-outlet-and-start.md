# The outlet opened, the start settled, and the drawn car's arms

*Case study, PoC 3. New 2026-09-14. Tier 59.*

---

# 0. The result, in one paragraph

**Both faults Tier 58 found outside the drawing are repaired, and the car the user drew runs its five arms inside every declared envelope for CS-19's full 600 macro-steps, with every gate threshold inherited and unchanged.** *The outlet (W258):* RaceLab's column pinned the domain's outflow column, so nothing moved it but the global projection; it now relaxes convectively — `WindowNS`'s own Orlanski condition, applied by the composition layer once per exchange — and on the repaired column the drawn car's fluid envelope declines on **no** macro-step, at either sizing, against $544$ of $600$ on the pinned one, while the duct flow at release moves by $0.03\%$. *The start (W259):* the release state had been settled against a machine the march never runs; settled for $120$ macro-steps against the machine W228 sizes, and sized again by W228 on that state, the machine stays inside on all $600$ macro-steps at both sizings, with its inflow ratio between $0.9809$ and $1.0258$ of a window that runs from $0.9731$ to $1.1408$. **Then Tier 56's own driver, run on the repaired column with the settle as a stage, found the first admissible horizon the drawn car has had — $600$ — verified it enforced, and marched all five arms with no macro-step outside:** P1, P2, P3, P4, P5 and P7 pass, P6 splits as it always has (lagged $0.428$ s under the $0.5$ s ceiling, tight $2.17$ s over), and the demo now releases from that settled state at the verified sizing. Three of eighteen registered predictions failed, and each is named: the outlet explains only $2.7$ of the $16.6$ points the duct flow fell, the repaired column is still sensitive to the box's length at the $2\%$ level, and the settled duct flow still drifts $4.4\%$ over the horizon.

---

# 1. What was asked, and the order chosen

After Tier 58 the user said: *measure whichever candidates in whichever order, and fix the issue.* The order followed the evidence rather than the list:

1. **The outlet first**, because it was the only fault with a mechanism in the code ([[poc3-racelab-car-fixes]] §7), and because a wake that cannot leave the box was also a candidate for the duct flow's unexplained fall (W256).
2. **The start second**, once the fall had been re-measured on the repaired column — the start transient was the only machine fault left to separate from it.
3. **Then the pipeline itself**, unchanged except for the two repairs, so the arms would be judged by the same stages, thresholds and gate Tier 56 was.

Nothing in `car_geometry.json`, the gate (`racelab.GATE`), the sizing rule (W228) or any threshold, clamp or bound was changed.

---

# 2. W258 — the outlet, repaired

## 2.1 What was wrong, from the code

Every RaceLab window is built by `ground_effect.solver_for` with `transmission="dirichlet"`, and under that setting `WindowNS._pin` overwrites all four faces of each window's ring from its input on every sub-step. On an interior seam that is right — the neighbour supplies the data. On the **domain's outlet** no neighbour supplies anything: the last column is frozen through every window step, only the global Leray projection changes it, and nothing relaxes what the projection adds. `WindowNS`'s own documentation calls a pinned outflow over-constrained for an advection-dominated flow, and the class implements the remedy it adopted, `_convect_outflow`.

## 2.2 The repair

`racelab.OUTFLOW = "convective"`, applied in `RaceRollout.relax_outflow` — called first in both `RaceRollout._advance` and `racelab_switch.MixedRollout._advance`, on the assembled field before it is cut, so every expert behind the outlet, classical or learned, receives the same condition:

$$a_{\text{last}} \leftarrow a_{\text{last}} - C\,\big(a_{\text{last}} - a_{\text{last}-1}\big), \qquad C = \operatorname{clip}\!\left(\frac{u_{\text{last}}\,\Delta t_{\text{ex}}}{h},\,0,\,1\right),$$

for both velocity components wherever $u_{\text{last}} > 0$, out of place. It is the expert's formula at the expert's cadence: a window takes $\lceil u_{\max}/2 \rceil$ sub-steps per exchange here, so one while the speed stays under two free-stream speeds.

`"pinned"` is the old column and is kept as the control. **The default changed, and every driver of a record measured before this tier now names the pinned column at every march** (`scripts/tier51_racelab_graph.py`, `tier52_racelab_switch.py`, `tier53_racelab_rerun.py`, `tier58_car_fixes.py`, and `tier54_traced_car.py` through its record registry), so each committed record stays reproducible by its own script. A settled field now records the outlet it was settled on, and a field for the other column is refused by the driver and not taken for this column's by the demo.

## 2.3 The measurements

`scripts/tier59_outflow_and_start.py`, stages `control` and `repair`, into `out/racelab7`:

| arm | outlet | box | FLUID declined | $u_{\text{rotor}}$ at release | its fall over probe 1 |
|---|---|---|---|---|---|
| control | pinned | $672$ | from $56$ (as Tier 58) | $0.531725$ | — ($60$ steps) |
| **the drawn car** | **convective** | $672$ | **none, at either sizing** | $0.531572$ | $14.0\%$ |
| box-length control | convective | $800$ | none | $0.527644$ | $15.3\%$ |
| box-length control | pinned | $800$ | from $265$, on $153$ steps | $0.527675$ | $15.4\%$ |

- **The control is bitwise**: the pinned column through the new code reproduces Tier 56's settled field and the first $60$ macro-steps of Tier 58's drawn arm exactly.
- **On the repaired column the fastest cell is never on the outflow column**, and the largest speed anywhere over $600$ macro-steps is $1.859$ against the bound's $2.048$.
- **The car's own flow hardly moves**: $u_{\text{rotor}}$ at release changes by $0.03\%$, the largest interior speed by $2.2\%$.
- **The box-length control reads two ways.** On the pinned column a longer box moves the breach from $56$ to $265$ — the outlet deciding the envelope, as Tier 58 found. On the repaired column neither box breaches — but the duct flow in the longer box still falls $1.3$ points more, and differs from the shorter box's by up to $2.2\%$ over the probe, where $1\%$ was predicted (W260).

---

# 3. W259 — the start, repaired

## 3.1 What was left

On the repaired column, W228's sizing from the unsettled state ($0.456800$) put the machine outside on macro-steps $0$ to $2$ only — the inflow starting $16\%$ above the sizing, above the window's upper edge — and inside for the other $597$. The duct flow fell $0.531 \to 0.460$ over the first $100$ macro-steps and by only $0.26\%$ per $100$ over the last $300$. The fast part is the release: the spin-up marches the machine `powertrain` declares, which at the release state sits on its lower clamp and motors, and the probe switches in a sized machine whose disk's $C_T' = 4a/(1-a)$ is $33$ times larger at that sizing, in one step.

## 3.2 The repair

**The release state is settled against the machine the march will run** — Tier 52's own practice (`settled_field(host_inflow=...)`, one Picard step), which Tier 54's driver had dropped:

1. spin up for $240$ macro-steps with the declared machine, as before;
2. size by W228 from that state ($0.456800$);
3. **settle** $120$ macro-steps at that sizing — chosen from the repair stage's probe, where the duct flow had covered $84\%$ of its whole fall by macro-step $100$, and not tuned afterwards;
4. size by W228 again **from the settled state**, and release everything from it.

## 3.3 The measurements

Stage `start`, prediction registered after stage `repair`'s record was read and before the stage ran:

| from the settled state | sized for | machine declined | FLUID declined | inflow ratio over $600$ |
|---|---|---|---|---|
| probe at its release sizing | $0.457249$ | **none** | none | $0.9746$ to $0.9997$ |
| probe at W228's sizing | $0.445657$ | **none** | none | $0.9809$ to $1.0258$ |

The settle itself carries the start the probes used to — outside on its own first three macro-steps — and releases from $u_{\text{rotor}} = 0.457249$. W228's second sizing is $2.4\%$ below its first, and at it the machine's inflow ratio stays $0.8\%$ above the window's lower edge and $11\%$ below its upper one.

---

# 4. The drawn car's arms

`scripts/tier54_traced_car.py --out out/racelab8`, Tier 56's driver with the record registered as the repaired column and `stage settle` inserted after `size`, in three runs of under two hours each: `spinup,size,settle,verify` ($20$ min), `arms` ($90$ min), `slow,gate,compare`.

## 4.1 Size, settle, verify

The driver lands on Tier 59's numbers exactly — release $u_{\text{rotor}}$ $0.531572$, first sizing $0.456800$ (no admissible horizon, as before), settled state $0.457249$, second sizing $0.445657$ — and then: **safe horizon $600$, CS-19's**, and `verify` marches all $600$ macro-steps **enforced** at $0.445657$ and admits them, the disk's induction between $0.050$ and $0.181$ and the loop current positive on every step.

## 4.2 The arms and the gate

| arm | coupling | macro-steps outside |
|---|---|---|
| referent | tight | $\mathbf{0}$ of $600$ |
| repeat | tight | $0$ |
| all lagged | lagged | $0$ |
| null J3 | tight | $0$ |
| null J1 | tight | $0$ |

| clause | verdict | the number |
|---|---|---|
| **P1** the compile | **pass** | joined refuses `L7/R9` and nothing else ($26$ agents, $36$ seams); disjoint and clocks-reconciled refuse nothing |
| **P2** J3's receiving balance | **pass** | $0.029210$ against $0.075$; null $1.0000$ |
| **P3** J1's parametric check | **pass** | $9.79\times10^{-8}$ against $10^{-6}$; null's UA pinned at $42$, its fluid bitwise the referent's |
| **P4** J2's receiving balance | **pass** | $5.59\times10^{-9}$ against $10^{-6}$; without the mount term $0.9978$ |
| **P5** the repeat floor | **pass** | bitwise in every crossing quantity and in the field |
| **P6** the macro-step cost | **splits** | lagged $0.428$ s under the $0.5$ s ceiling; tight $2.17$ s over it |
| **P7** the body force | **pass** | worst relative residual $6.2\times10^{-16}$ against $10^{-12}$ |

**P2's residual against the ring-to-plane velocity gap** (W231): the gap predicts $0.032200$ and the balance closes at $0.029210$, so the gap over-predicts by $1.102\times$ — beside the traced car's $1.098\times$ (Tier 54), Tier 53's $1.68\times$ and CS-19's five figures. Not predicted, and not diagnosed.

Over the settle window the referent's machine generates on every macro-step and sheds $1.26$ W of heat, and the block's thermal time constant is $50.9$ s — within $1.5\%$ of CS-18's $50.1$ s.

## 4.3 The demo

`atlas/demo_racelab/engine.py` is sized for the verified value, `U_DUCT = 0.44565659437257343`, and releases from `out/racelab8`'s settled state, matched by the car's fingerprint **and** the outlet it was settled on. Opened and marched: at macro-step $59$ the page carried no `OUTSIDE THE MODEL` stamp — Tier 56's page stamped from $20$ — with the induction at $0.119$ and the machine generating. Six hundred macro-steps are what was verified; the page says so, and will stamp a longer march when the drift in §5 takes the machine out.

---

# 5. The prediction, registered, and the three that failed

| id | the prediction | outcome |
|---|---|---|
| R1 | the pinned column through this file is Tier 58's, bitwise | **confirmed** |
| R2 | the repair removes W258 at both sizings | **confirmed** — no FLUID decline, the fastest cell never on the outflow column |
| R3 | the repaired column is box-length independent to $1\%$; the pinned one's breach moves $\ge 100$ steps | **falsified** — the pinned half holds ($56 \to 265$); the repaired $u_{\text{rotor}}$ differs by up to $2.2\%$ (W260) |
| R4 | the car's own flow moves little | **confirmed** — release $u_{\text{rotor}}$ $-0.03\%$ |
| R5 | the outlet explains at least $3$ points of the $16.6\%$ fall | **falsified** — $2.7$ points; the rest was the start and the out-of-envelope machine |
| R6 | the start is not the outlet | **confirmed** — declined at $0$ to $2$ on the repaired column |
| T1 | settled and re-sized, the machine is inside all $600$ | **confirmed** |
| T2 | the second sizing is within $3\%$ and keeps $0.5\%$ above the lower edge | **confirmed** — $2.4\%$ and $0.8\%$ |
| T3 | at the settled state's release sizing, inside on $\ge 550$ | **confirmed** — $600$ |
| racelab8 1–2 | reproduces stage start; verify admits $600$ enforced | **confirmed** |
| racelab8 3 | every arm inside **and** the referent's $u_{\text{rotor}}$ moves $< 3\%$ | **falsified in its second half** — every arm is inside, and the duct flow moves $4.4\%$ over the horizon (W261) |
| racelab8 4–9 | P1, P7, P2, P3 and P4 with their nulls, P5, P6 splits | **confirmed** |

---

# 6. What this tier did NOT do, named

- **The box-length sensitivity is not explained** (W260). The global spectral projection pads a box of a different length, and the wake develops over a different distance; neither was separated.
- **The settled duct flow still drifts** — $4.4\%$ over $600$ macro-steps at W228's sizing, leaving $0.8\%$ above the machine's lower clamp at the end (W261). A longer march will meet it; nothing here claims past $600$.
- **The bundle is stale** (W262). The build at `C:\Users\Nauni\poc3_build` ships the pinned column's release state, and the builder, the launcher's self-test and the templates still point at `out/racelab5`; it was not rebuilt or re-verified, and nothing was pushed.
- **A drawing change still moves the machine** (W257, unchanged): the device planes are chosen by the cut objective.
- **The learned column was not re-measured** on the repaired outlet; CS-20's numbers stand for the pinned one.
- **Nothing was downloaded, no machine was rented, the unlicensed structural checkpoint was not loaded, and nothing was pushed.**

```
python scripts/tier59_outflow_and_start.py --out out/racelab7 --stages control,repair,summary
python scripts/tier59_outflow_and_start.py --out out/racelab7 --stages start,summary
python scripts/tier54_traced_car.py --out out/racelab8 --stages spinup,size,settle,verify
python scripts/tier54_traced_car.py --out out/racelab8 --stages arms
python scripts/tier54_traced_car.py --out out/racelab8 --stages slow,gate,compare --compare-with out/racelab4/racelab4.json
```

---

## See Also

- [[poc3-racelab-car-fixes]] — Tier 58, where both faults were found outside the drawing.
- [[poc3-racelab-drawn-car]] — Tier 56, the refusal these repairs lift.
- [[poc3-racelab-traced-car]] — Tier 54, the driver re-used and the traced car's arms compared above.
- [[case-study-racelab-graph-atlas-0.1]] — CS-19, the arms and the gate.
- [[case-study-racelab-switch-atlas-0.1]] — CS-20, where the machine was sized for its host (W222, W228) and the spin-up first settled against it.
- [[poc3-racelab-demo]] — the dashboard, now inside its envelopes for the verified horizon.
- [[gap-worklist]] — W258 and W259 closed; W260 to W262 opened.
