# Rake and the duct area wired, and the knob layer criterion 2 needs

*Case study, PoC 3. New 2026-09-16. Tier 70.*

---

# 0. The result, in one paragraph

**Tier 69 measured section 3.3's eleven parameters against what they claim to reach and found four wired, three dead, one unwireable, two reaching only through a module global, and one reaching half of what it claims. This wires the two that were dead and repairable, and builds the state a screen would drive.** `rake` now pitches the floor group **rigidly** about its own leading edge — five bodies move, the pivot does not, and every plate gains the same $0.0244^\circ$ at $\text{rake} = 0.15$. `duct_area` scales each opening about its centre, taking the duct's total achieved span from $0.210$ to $1.050$ across its declared range. **The check that mattered most was neither**: wiring a knob must not move the car every cached field and record was built on, and `rake`'s default had to go from $0.10$ to $0.0$ in the same change — because while it reached nothing, *every car ever built here was the rake $=0$ car*, whatever the default said. The fingerprint is unchanged at `a1f67a8e`. Four of five predictions held; **M4 failed because it contradicted M2, registered beside it**.

---

# 1. Rake is a rigid pitch, not a staircase

Section 3.3 scopes rake to "floor, diffuser", and `DIFF` and `DIFF_EXIT` are both in the `floor` group, so one group is the whole of it. The floor runs $x \in [170.0, 521.97]$ cells, so `rake` is the height the **rear** of the floor gains over its front.

Raising each plate by its own share of the rake and leaving the angles alone would be a staircase pretending to be a ramp. A rigid pitch moves both:

$$y \mathrel{+}= \text{rake}\,\frac{x - x_0}{L}, \qquad \alpha \mathrel{+}= \arctan\!\frac{\text{rake}}{L}$$

Measured at $\text{rake} = 0.15$, $L = 351.97$:

| body | $x$ | rise | angle gained |
|---|---|---|---|
| `FLOOR_LE` | $170.0$ | $\mathbf{0}$ — it is the pivot | $0.024418^\circ$ |
| `FLOOR_STEP` | $181.0$ | $0.0047$ | $0.024418^\circ$ |
| `FLOOR` | $191.6$ | $0.0092$ | $0.024418^\circ$ |
| `DIFF` | $485.0$ | $0.1342$ | $0.024418^\circ$ |
| `DIFF_EXIT` | $495.0$ | $0.1385$ | $0.024418^\circ$ |

`DIFF`'s rise is $0.15 \times (485 - 170)/351.97 = 0.134244$ to nine figures, the pivot does not move at all, and the angle is identical on every plate — which is what makes it one rotation rather than five translations.

---

# 2. The duct area scales about each opening's centre

The openings were fixed literals: a `CHASSIS` inlet over $[0.25, 0.75]$ of its panel and a `POD_UP` outlet over $[0.80, 1.00]$. `duct_area` scales each span about its own centre, so the inlet stays where it was drawn and only opens or closes.

| `duct_area` | inlet span | outlet span | total |
|---|---|---|---|
| $0.3$ | $0.150$ | $0.060$ | $\mathbf{0.210}$ |
| $1.0$ | $0.500$ | $0.200$ | $0.700$ |
| $1.5$ | $0.750$ | $0.300$ | $\mathbf{1.050}$ |

**An opening that would run off the end of its panel is shifted, not truncated.** At $1.5$ the outlet's centred span would be $[0.75, 1.05]$; it becomes $[0.70, 1.00]$, keeping the full $0.300$ rather than losing the overhang. `car_solids.openings_report` returns nominal beside achieved and a `clamped` flag, so the one case where a span genuinely cannot fit is visible rather than assumed away.

---

# 3. The check that mattered most: the nominal car did not move

`geometry_fingerprint` is what the settled fields, the raster cache and every record of Tiers 62–68 are keyed on. Wiring a knob that is *on* by default would have pitched the floor under all of them silently.

| | fingerprint |
|---|---|
| HEAD, before this tier (rake unwired) | `a1f67a8e…` |
| wired, at the new defaults | **`a1f67a8e…`** |
| wired, at the old default $\text{rake} = 0.10$ | `44eb4bdc…` |

So the default moved $0.10 \to 0.0$ in the same change, and `duct_area`'s default is $1.0$. **This is not a tidy-up; it is the difference between a knob and a silent geometry change.** `racelab` already carries the same note beside `diffuser_deg` and `front_flap_deg`, recording that a default disagreeing with the drawing once meant "the editor showed one car and the march ran another".

**[AI Inference]:** the general rule is that wiring a previously-dead parameter is a **two-part** change — the wiring and the default — and the default half is the dangerous one, because it is invisible in the diff of the logic. Any parameter that has been declared-but-unread has an implied value of "whatever the code did without it", and that is what the new default must reproduce.

---

# 4. What a screen would drive

`car_knobs.KnobState` is criterion 2's machinery: `set` applies one value and returns the subsystems that **responded**, measured, so a page can show the response rather than assert it.

| knob | moved to | responded |
|---|---|---|
| ride height | $0.60$ | `DIFF`, `FW_FLAP`, `FW_MAIN`, `FW_UPPER` |
| rake | $0.15$ | `DIFF`, `DIFF_EXIT`, `FLOOR`, `FLOOR_LE`, `FLOOR_STEP` |
| diffuser / front flap / rear wing | $15^\circ$ / $30^\circ$ / $30^\circ$ | `DIFF` / `FW_FLAP` / `RW_FLAP` |
| duct area | $1.5$ | duct geometry, core inflow, $UA$ |
| coolant mass flow | $0.30$ | `cooling_loop` |
| ambient temperature | $318$ K | `cooling_loop`, `brake_thermal` |
| battery state of charge | $0.0$ | open-circuit voltage |

Two things it refuses, at the point a value enters rather than after:

- **A knob that reaches nothing.** `road_speed` and `battery_power` raise with their reason, and the stored value does not move. §3.3's rule is worth nothing if a dashboard can still set it.
- **A value outside a declared range**, and an unknown knob name.

And a geometry knob is flagged `needs_regrid`. **It does not re-cut the car.** A composite rebuild costs $66$–$90$ s (Tier 68), so the change is recorded, `pending_regrid` names it, and `BodyFittedColumn.set_knob` reports `marching: "the car BEFORE this change"` — §4.1 asks for a visible recompiling state, and a demo that marched the old car while the sliders showed the new one is the defect this project keeps finding.

---

# 5. The predictions

| id | claim | outcome |
|---|---|---|
| M1 | the nominal car is unchanged at the wired defaults | **held** — `a1f67a8e`, and $\text{rake}=0.10$ gives a different one |
| M2 | the tally becomes $6$ wired, $2$ global, $1$ partial, $1$ dead, $1$ absent | **held** |
| M3 | every offerable knob reports at least one subsystem that responded | **held** — nine of nine |
| M4 | each of the **four** unofferable knobs is refused where a value enters | **failed** — there are two (§5.1) |
| M5 | rake is a rigid pitch: pivot fixed, rear rises, one angle | **held** |

## 5.1 M4 contradicted M2, and both were registered in the same breath

M4 says *four* unofferable knobs. M2 says the tally becomes `dead = 1, absent = 1` — **two**. The four was Tier 69's count, and **this tier is the thing that changed it**: wiring `rake` and `duct_area` moved them out of that set before M4 could be judged.

Everything M4 was actually about held. Both remaining unofferable knobs are refused at the point a value enters, with their reason, the stored value does not move, and out-of-range and unknown names are refused too. Only the number was wrong, and it was wrong against a sibling prediction sitting three lines above it.

**This is the third tier running with a prediction refuted by something already written beside it** — Tier 68's K1 by its own `why`, Tier 69's L1 by its choice of baseline, and now M4 by M2. All three are cheap to catch: one division, one question about the baseline, one glance at the neighbour. M4 is left as registered and judged as registered.

---

# 6. What this tier did NOT do, named

- **Criterion 2 is still not met**, and this is the honest line: there is **no page**. `KnobState` is the machinery a screen drives and `BodyFittedColumn.set_knob` is its entry point, but nothing renders a slider, so nobody can yet *move a parameter and watch*. The dashboard is the next tier.
- **`regrid()` is not implemented.** A geometry knob marks `pending_regrid` and the column keeps marching the car it has; rebuilding the composite from the new knobs is written down and not written.
- **The cooling knobs still reach through module globals** (W291): `KnobState` sets them, but `loop_return` gets there by rebinding `cooling_loop.MDOT` and `T_AMB`, which is unsafe under the demo's background thread and moves a leg's `weight_hash` with nothing noticing.
- **`road_speed` and `battery_power` are still unwired** — the first needs a dimensional scale, the second a different powertrain model. They are refused rather than shown.
- **Nothing was marched**, and no knob has yet been moved on a live column.
- **Nothing was downloaded or installed**, no machine was rented, the unlicensed structural checkpoint was not loaded, and nothing was pushed.

```
python scripts/tier70_knobs_wired.py --out out/racelab19 --stages wire,respond,nominal,summary
python -m pytest tests/test_tier70_knobs_wired.py tests/test_tier69_car_knobs.py -p no:cacheprovider
python scripts/vault_scan.py wiki
```

The record is `out/racelab19/racelab19.json`.

---

## See Also

- [[poc3-racelab-knobs]] — Tier 69, the measurement this repairs two of.
- [[poc3-racelab-bodyfitted-demo]] — Tier 68, the column `set_knob` reports against.
- [[gap-worklist]] — W290 narrowed here, W291 still open.
