# The drawn car, and the arms that were not run

*Case study, PoC 3. New 2026-09-13. Tier 56.*

---

# 0. The result, in one paragraph

**The car the user drew in Tier 55 replaced the traced one, so every marched number in `out/racelab4` described a vehicle that no longer existed — and Tier 56 re-ran Tier 54's script for the drawn car, into `out/racelab5`. The gate refused, and the five arms were not run.** Two declared envelopes decline over the horizon, and neither can be moved by the one thing the procedure is allowed to move. **The machine (W244):** the duct flow falls $16.6\%$ over the $600$ macro-steps, against $2.0\%$ on the traced car; sized for the horizon's minimum the disk is on its *upper* clamp at macro-step $0$, sized for the release state it reaches its *lower* clamp at macro-step $20$, and no constant spans both. **The fluid (W245):** the window expert's cell-Reynolds bound of $8$ is breached at macro-step $56$ and stays breached on $544$ of $600$ macro-steps, with $u_{\max}$ climbing from $1.82$ to $3.30$ — and the fastest cell is **not at any body**. It sits on the **outflow column**, where the car's wake reaches the edge of the box and the boundary column is dragged downward at $v = -3.00$. The pipeline did exactly what Tier 54 rebuilt it to do: `verify` declined, and `arms` refused to spend ninety minutes on a sizing already known to fail. **The most useful measurement here is where the fluid breaks, and the most useful correction is to this tier's own first reading of it**, which blamed the nearest body — 155 cells away.

---

# 1. What was re-run, and what the script had to learn first

`scripts/tier54_traced_car.py` is Tier 54's driver, re-used. Before it could be re-used it needed six changes, each for a reason a re-run would otherwise have hit:

| change | why |
|---|---|
| **`--out` is required** | it wrote to `out/racelab4` unconditionally; re-running it for a new car would have overwritten Tier 54's committed record |
| the car's description is **measured from the model**, with a fingerprint | the record hard-coded the traced car ("freesvg.org id 48844", "22 plates") and would have stamped it on the drawn car |
| a prediction **registry keyed by record** | a new record would otherwise have inherited Tier 54's prediction |
| **probe 2 at the arms' sizing** when probe 1's first decline is the machine | probe 1 is sized for the release state and the arms are not; choosing a horizon from a decline the re-sizing removes reads a different quantity from the one that predicts the arms' failure |
| probes keep their **whole trajectory**; "no admissible horizon" is a result | this tier's first run recorded endpoints only, and Tier 54's chooser returned its floor — "horizon 60" — from a probe that had declined at macro-step $0$ |
| a stage that raises is **recorded before it propagates** | the first run's refusal is in the record, not only in a console |

Two stages were added for the gate — `slow` (P4 with its null, Tier 53's recipe imported) and `gate` (all seven clauses judged in code, thresholds read from `racelab.GATE`) — and two for the diagnosis below, `trace` and `ablate`.

**A settled field now carries the car it was settled around** (`racelab.geometry_fingerprint`, a SHA-256 over the *built* bodies). The demo had released the traced car from the hand-drawn car's field in Tier 54 and the drawn car from the traced car's field after Tier 55 — the same defect twice, because a directory order says which cache is newest and not which car it belongs to. Every stage that releases from `settled.npz` now refuses a field for a different car, and so does the bundle builder. The fingerprint repeats, ignores a caption, and changes on a move of $10^{-9}$ cells and on a knob.

---

# 2. The prediction, registered before the spin-up

Recorded in `PREDICTION_DRAWN` and written to the record with its timestamp before the first stage ran:

| # | the prediction | outcome |
|---|---|---|
| 1 | $u_{\text{rotor}}$ at release differs from the traced car's $0.667914$; direction not predicted | **confirmed** — $0.531725$, $20\%$ lower |
| 2 | the car leaves the fluid envelope before macro-step $554$, so the horizon is shorter than $530$ — *because* its near-normal plates accelerate the flow past their ends | **the conclusion is confirmed in its strongest form, and the reason is not what was measured** — there is no admissible horizon at all, and the fluid breaks on the outflow column, not beside a plate (§4) |
| 3 | $u_{\text{rotor}}$ still falls, and sizing for the horizon's minimum again admits the arms | **falsified** — it falls, and no sizing admits them (§3) |
| 4–7 | P1 and P7 pass; P2's residual is not the velocity gap; P3 and P4 pass with their nulls; P5 bitwise and P6 splits | **not measured** — P2–P6 need the arms. P1 and P7 need only the release state; see §6 |

---

# 3. W244 — the sizing has no admissible answer on this car

W228's procedure is to size the machine for the horizon's **minimum** duct inflow, not for one instant. On the two earlier cars it admitted every macro-step. On this one:

| | Tier 53 (hand-drawn) | Tier 54 (traced) | **Tier 56 (drawn)** |
|---|---|---|---|
| $u_{\text{rotor}}$ at release | $0.6692$ | $0.6679$ | $\mathbf{0.5317}$ |
| its fall over $600$ macro-steps | $6.3\%$ | $2.0\%$ | $\mathbf{16.6\%}$, still falling at the last step |
| sized for | $0.63000$ | $0.654077$ | — |
| admitted | $600$ | $530$ | **none** |

| probe | sized for | first decline | outside |
|---|---|---|---|
| **1**, at the release state | $0.531725$ | macro-step $\mathbf{20}$, ROTOR — induction on the **lower** clamp, $0.02$ | $580$ of $600$ |
| **2**, at the horizon's minimum | $0.442762$ | macro-step $\mathbf{0}$, ROTOR — induction on the **upper** clamp, $0.4$ | $553$ of $600$ |

The machine is sharp: on the traced car a $2\%$ range of inflow already moved the induction from $0.07$ to $0.17$ over Tier 54's verify, which is the sensitivity CS-18 §7.2 measured for a machine sitting just above its battery's open-circuit voltage. A constant sizing can be inside its clamps only while the inflow stays within a few percent of the value it was sized for, and this duct's inflow moves by a sixth. **Sizing for the release state buys twenty macro-steps; sizing for the minimum buys none.** The two trajectories say how: at the release sizing the induction is $0.1105$ at macro-step $0$ and pinned at $0.02$ from macro-step $20$ to the end, and at the minimum's sizing it is pinned at $0.4$ from macro-step $0$ and comes down through $0.151$ at $100$ to $0.027$ at $500$ only as the inflow falls toward the value it was sized for — by which time the fluid has long since broken (§4). Probe 2 also shows the minimum is not a fixed target: the machine sized for it changes the thrust and the minimum falls another $2.6\%$, to $0.431106$ — W228's one Picard step, not converging.

The procedure is not mis-applied here; it is insufficient, which is what Tier 54's own prediction said the informative outcome would be. What would fix it is a *declaration*, not a threshold — a machine whose sizing follows its inflow — and that is a new model, not taken here.

---

# 4. W245 — the fluid leaves its envelope at the outflow boundary

`stage trace` marches the drawn car by hand at the release sizing and reads the **model's own** envelope after every macro-step — the declined list and the cell Reynolds number `RaceRollout._absorb` computed — together with where the fastest cell is.

| macro-step | $0$ | $50$ | $56$ | $100$ | $200$ | $300$ | $400$ | $500$ | $599$ |
|---|---|---|---|---|---|---|---|---|---|
| $u_{\max}$ | $1.8189$ | $2.0073$ | $2.0531$ | $2.2956$ | $2.4879$ | $2.9216$ | $3.0277$ | $3.2565$ | $\mathbf{3.3025}$ |
| cell Re | $7.105$ | $7.84$ | $\mathbf{8.020}$ | $8.967$ | $9.718$ | $11.41$ | $11.83$ | $12.72$ | $\mathbf{12.90}$ |

**The first decline is at macro-step $56$, and the flow never comes back inside.** From there on the fastest cell is on the **outflow column** — $x = 671$ of $672$ — at $y = 100$ to $118$, which is where the top of the car's wake meets the box's edge and also where the tiling's horizontal seam band starts. At macro-step $599$ that cell carries $u = 1.384$ and $v = -2.999$: the boundary column is being pulled down into the wake. One column upstream the largest speed is $2.02$, and forty columns upstream it is $1.70$. The speed field shows why: the drawn car sheds a slow wake that fills the box from the road to $y \approx 110$ all the way from the rear wing to the outlet.

**So the check fired correctly on a state the expert does not stand behind, and the state is a boundary artifact rather than a body's.** The fix is not a threshold — the bound is the expert's own declaration — and it is not a different machine. It is a longer box behind the car, a treatment of the outflow boundary, or a car whose wake is shallower. Tier 54's traced car breached first at macro-step $554$, by $0.1\%$; where, it did not record.

> **[AI Inference]:** the traced car's late breach is probably the same mechanism with a weaker wake — its body tapered and it had no rear endplate standing $61$ cells tall. That was not measured: Tier 54 recorded the step and the speed, not the cell.

## 4.1 The first reading was wrong, and how

The trace's first summary said the fastest cell was at `RW_ENDPLATE` on all $544$ declined macro-steps. It was computing the **nearest** body, and a nearest neighbour always exists: `RW_ENDPLATE` is the most downstream body and sat **$155$ cells** from the cell. The summariser now names a boundary when the cell is on one and a body only when it lies within $8$ cells, and the record keeps both readings side by side (`ablate.trace_summary_reread`). Had this page repeated the first reading it would have told the user to redraw a plate that is not where the flow breaks.

## 4.2 The ablation, and what it can and cannot say

`stage ablate` marches the same car for $200$ macro-steps with bodies **removed**, holding the window layout at the drawn car's, beside the intact car through the same code path. The control agrees with `stage trace` **bitwise** over those steps.

| removed | FLUID first | ROTOR first | MGU first | $u_{\max}$ at $199$ |
|---|---|---|---|---|
| nothing (control) | $56$ | $20$ | $27$ | $2.4864$ |
| `RW_ENDPLATE` | $58$ | $\mathbf{123}$ | $\mathbf{146}$ | $2.4298$ |
| `DIFF_EXIT` | $57$ | $94$ | $123$ | $2.4631$ |
| the closed rear box: `RW_ENDPLATE`, `RW_ENDPLATE_LO`, `DIFF_EXIT` | $59$ | **none in $200$** | **none in $200$** | $2.3994$ |

**It cannot attribute the fluid breach to any plate, and saying so is the result.** Every arm starts from the intact car's settled field, whose wake already reaches the outlet at macro-step $0$, and $200$ macro-steps is about a quarter of a transit of the box: removing a body does not remove a wake it has already made. The fluid column moves by at most three macro-steps in any arm, and that is uninformative in both directions. **The machine is different, because the duct flow responds within a few macro-steps.** The endplate alone moves the rotor's first decline from $20$ to $123$, the diffuser exit alone to $94$, and the closed box the two form with the endplate's lower edge keeps the rotor and the machine inside their clamps for **all $200$ macro-steps marched** — "none in $200$" is not "never", it is as far as the arm went. **So the rear box the user drew is what pulls the duct's inflow down over those $200$ macro-steps** — the fall W244 is about — though whether the car without it stays within one constant's reach over the full $600$ was not marched. A discriminating ablation of the fluid breach would re-spin each car from the freestream — about five minutes an arm — and was not run.

---

# 5. What the demo does with this car

- **It releases the drawn car from its own settled field**, matched by fingerprint (`demo_racelab.engine.find_release`), and says which field it used and whether it belongs to this car.
- **It is sized for the release state**, $U_{\text{duct}} = 0.5317$, because nothing better exists: that buys twenty macro-steps inside every envelope, and the page stamps `OUTSIDE THE MODEL` from then on. The traced car's $0.654$ would have the machine motoring from macro-step $0$.

---

# 6. The gate, as far as it can be judged

`stage gate` judges every clause with its threshold read out of `racelab.GATE`, and a clause whose measurement does not exist is `None` — not `fail`, and not `pass`.

| clause | verdict | the number |
|---|---|---|
| **P1 · the compile** | **pass** | the joined union refuses at `L7/R9` and nothing else ($26$ agents, $36$ seams); the disjoint union refuses nothing; the clocks-reconciled control refuses nothing |
| P2 · J3's receiving balance | `None` | needs the referent and J3's null arm |
| P3 · J1's parametric check | `None` | needs the referent and J1's null arm |
| P4 · J2's receiving balance | `None` | needs the referent's machine heat |
| P5 · the repeat floor | `None` | needs the referent twice |
| P6 · the macro-step cost | `None` | needs the lagged and tight arms |
| **P7 · the body force conserves** | **pass** | worst relative residual $7.5\times10^{-15}$ against $10^{-12}$, over every body at the release state |

## 6.1 Two controls that came for free

- **The record is reproducible bitwise.** `size` was marched twice — once before the script learned to keep trajectories, once after — and both probes' minima, first declines and outside counts agree exactly, to the last digit of $0.44276205468922647$.
- **The hand-stepped trace takes the same trajectory as `racelab.march`.** Probe 1's $600$ per-step values of $u_{\max}$, recorded through `march`, equal `stage trace`'s, recorded by stepping the rollout by hand, bitwise. That is what licenses reading the envelope off the rollout in §4 as a statement about the march the arms would have run.

---

# 7. What this tier did NOT do, named

- **The five arms were not run.** P2–P6 are unmeasured on the drawn car. `out/racelab4` stands as the record of the traced car and is not rewritten, and nothing here is a comparison of settled quantities between cars.
- **No threshold, clamp or bound was touched.** The cell-Reynolds bound of $8$ is the window expert's declaration and the induction clamps are the disk's.
- **No sizing procedure beyond W228's was tried.** A machine that re-sizes itself as its inflow moves would be a new declaration.
- **The fluid breach is not attributed to any part of the car** (§4.2), and no longer box or outflow treatment was tried.
- **The car was not redrawn.** Whether to shorten the rear endplate, lengthen the box, or ship a demo stamped after twenty macro-steps is a decision for the person who drew it.
- **Nothing was downloaded** except PyPI's release metadata for torch, which was read; no machine was rented; the unlicensed structural checkpoint was not loaded; nothing was pushed.

---

## See Also

- [[poc3-racelab-traced-car]] — Tier 54, the car before this one and the script re-used here.
- [[case-study-racelab-switch-atlas-0.1]] — CS-20, where the machine was first sized for its host (W228).
- [[case-study-racelab-graph-atlas-0.1]] — CS-19, the arms and the gate.
- [[poc3-racelab-demo]] — the dashboard this car now appears on.
- [[gap-worklist]] — W244 to W247.
