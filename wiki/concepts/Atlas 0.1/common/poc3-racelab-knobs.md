# Section 3.3's parameters, and what each one actually reaches

*Case study, PoC 3. New 2026-09-16. Tier 69.*

---

# 0. The result, in one paragraph

**Criterion 2 asks that a person move a parameter and watch every coupled subsystem respond, and its purpose is to show the coupling is real — that one knob propagates through every subsystem the graph says it reaches, so RaceLab is one coupled system rather than several models sharing a screen.** Section 3.3 gives the rule that makes it testable: *a slider that moves a number nothing reads is worse than no slider*. So every one of the eleven parameters was moved from one end of its declared range to the other and the quantity it **claims** to reach was measured at both ends. **Four reach their subsystem as declared. Three reach nothing at all. One has nothing in the model to reach. Two reach theirs only by rebinding a module-level constant. And one reaches half of what it claims** — the state-of-charge knob moved the battery's EMF while the motor-generator drawing from it went on reading a module constant, including in the `validity` predicate that W282's envelope is written on. That last one is repaired here. The sharpest single finding is `rake`: it is declared as a geometry knob and **appears exactly once in the entire package — its own declaration**.

---

# 1. The table, as measured rather than as written

| knob | group | claims to reach | measured | verdict |
|---|---|---|---|---|
| ride height | vehicle | front wing, floor, diffuser | moves `DIFF`, `FW_FLAP`, `FW_MAIN`, `FW_UPPER` | **wired** |
| diffuser angle | aero | floor geometry | moves `DIFF` | **wired** |
| front flap angle | aero | front wing geometry | moves `FW_FLAP` | **wired** |
| rear wing angle | aero | rear wing geometry | moves `RW_FLAP` | **wired** |
| coolant mass flow | cooling | `cooling_loop` | return $310.156 \to 310.865$ K | **global** |
| ambient temperature | cooling | `cooling_loop`, `brake_thermal` | return $288.133 \to 325.778$ K | **global** |
| battery state of charge | powertrain | open-circuit voltage | battery moves, **the machine did not** | **partial** |
| **rake** | vehicle | floor, diffuser | **moves nothing** | **dead** |
| **radiator duct area** | cooling | duct geometry, core inflow, $UA$ | **moves nothing** | **dead** |
| **road speed** | vehicle | the freestream, the scale's $U_0$ | **moves nothing** | **dead** |
| **battery power draw** | powertrain | `powertrain`, J2's heat | **nothing to reach** | **absent** |

Amendment 13.1 had already omitted brake duty, brake duct opening and plate stiffness; `car_knobs.OMITTED` keeps those beside these so the hole list is one list.

## 1.1 The three dead ones, and why each is dead

**`rake`** is the one that matters most, because it looks wired. `CarParams` declares it with a default of $0.10$ and a comment describing it as the rear-to-front ride-height difference — and a search of the whole package finds the string exactly once, in that declaration. No body, grid, force, record or capability reads it. Moving it from $0.00$ to $0.15$ changes **zero** of the car's $51$ built bodies.

**`radiator duct area`** has nothing to scale. The duct's openings are fixed literals in `car_solids.SOLIDS_DEFAULT` — a `CHASSIS` inlet from $0.25$ to $0.75$ and a `POD_UP` outlet from $0.8$ to $1.0$ — and no code multiplies them by anything.

**`road speed`** cannot reach a field because the column is nondimensional: `ground_effect.U_INF` is $1.0$ and the Reynolds number is the tiling's. Road speed sets a **scale**, so moving it renames the axes and changes no number the march computes.

## 1.2 The absent one is a statement about the model, not a gap in the wiring

Section 3.3 asks for a battery power draw of $0$–$350$ kW reaching `powertrain` and J2's heat. **This circuit has no demand input at all.** The machine is a *generator*: its loop current follows the shaft speed through `current_at(omega)`, and the battery receives whatever is pushed into it. There is no quantity a power draw could set without a different powertrain model, so the knob is not "unwired" — it is unwireable as specified, and that is a fact about the case study worth stating rather than a task waiting to be done.

---

# 2. The defect the measurement was built to find

`BatteryLeg.v_oc` is a real field and its `emf()` honours it. `MachineAgent` had **no `v_oc` field at all**: `current_at` and `validity` read the module constant `V_OC`. So a state-of-charge knob moved the battery and left the machine drawing against a battery that was no longer there.

That is not merely untidy. **`validity` is the predicate W282's envelope is written on** — the machine declines when $k_e\,\omega \le V_{oc}$ — so at a lower state of charge the machine would have declined, or failed to decline, at the wrong shaft speed.

Repaired by giving the machine its own `v_oc` and `r_total`, defaulting to the module constants:

| | default | $v_{oc} = 0.80$ |
|---|---|---|
| `current_at(20)` | $\mathbf{2.200000}$ — unchanged | $4.000000$ |
| declines below | $\omega = 13.400$ | $\omega = 8.000$ |
| `validity` at $\omega = 10$ | **False** | **True** |

The last row is the coupling made visible: **the same shaft speed is outside the envelope on a full battery and inside on a depleted one**, which is physically right — a generator needs less speed to push into a lower voltage — and was not expressible before.

---

# 3. The reach test committed the defect it detects

The first version called a knob wired whenever its two readings differed. **`nan != nan` is `True` in Python**, so a probe that returned `nan` at both ends — because it was calling an API that did not exist — reported its knob as **moving**. A broken probe read as a working knob, which is precisely the failure section 3.3 is about, committed by the detector.

`car_knobs.moved` therefore requires both readings to be **finite and different**, and a test asserts it refuses a `nan` pair, an `inf`, and non-numbers.

**[AI Inference]:** the general shape is that any "did it change?" check written as `a != b` over floating point silently treats *absence of a measurement* as *evidence of a change*, and the direction of that error is always permissive. A reach test, a regression check and a cache validator all share it.

---

# 4. The price of the repair

A change to a tested module is measured the way Tier 45 measures one: every captured artifact recompiled and compared byte for byte.

Against the Tier 45 capture, **16 of 40 artifacts differ** — and **none of them because of this tier**. The capture is a baseline from Tier 45 and it *accumulates*: W194 moved the two union fixtures, W285 moved `front_wing`, and other tiers moved the rest. Running the identical stage against `HEAD`'s `powertrain.py`, before this tier's edit, produced **the same sixteen**. So what this tier adds is **none**, and what it heals is none.

---

# 5. The predictions

| id | claim | outcome |
|---|---|---|
| L1 | the repair leaves **every** captured artifact byte-identical | **failed** — $16$ differ, none added by this tier (§5.1) |
| L2 | the machine now follows its battery: threshold $13.4 \to 8.0$ rad/s | **held** |
| L3 | the tally is $4$ wired, $3$ dead, $1$ absent, $2$ global, $1$ partial | **held** |
| L4 | `moved` refuses a pair of `nan`s | **held** |
| L5 | `rake` still reaches nothing after the repair | **held** |

## 5.1 L1 failed on the wrong baseline, and the failure is mine

L1 asked whether the repair leaves *every* captured artifact byte-identical. Sixteen are not — but sixteen were not before the repair either, because the comparison is against a **Tier 45** capture that accumulates every deliberate change since.

The claim as registered is therefore false, and it is judged false. **The test it should have stated is the one the record now carries**: `added_by_this_tier`, which is empty. A prediction that compares today's tree against a months-old baseline measures the project's history, not the change in front of it — and it fails in the *alarming* direction, which is at least the safe one, but it is still the wrong question.

L1 is left as registered and judged as registered, because a criterion rewritten after seeing the data is not a pre-registered criterion. The right question is recorded beside it rather than substituted for it.

---

# 6. What this tier did NOT do, named

- **No knob is exposed on any screen.** This measures what the knobs reach and repairs one subsystem; wiring them into the demo, with the "recompiling" state a geometry change needs, is the next tier and criterion 2 is **not** met by this one.
- **`rake`, `duct_area`, `road_speed` and `battery_power` are still not wired.** They are named, with reasons, and `must_not_be_shown` keeps them off a dashboard until they are.
- **The cooling knobs still go through module globals.** They reach the loop, but by rebinding `cooling_loop.MDOT` and `T_AMB`, which ten call sites read directly — unsafe under the demo's background march thread, and a leg's `weight_hash` is built from `MDOT`, so moving it moves a capability record's identity with nothing noticing. That refactor is its own tier.
- **Nothing was marched** and the body-fitted column was not touched.
- **Nothing was downloaded or installed**, no machine was rented, the unlicensed structural checkpoint was not loaded, and nothing was pushed.

```
python scripts/tier69_car_knobs.py --out out/racelab18 --stages reach,machine,control,summary
python -m pytest tests/test_tier69_car_knobs.py tests/test_tier37_powertrain.py -p no:cacheprovider
python scripts/vault_scan.py wiki
```

The record is `out/racelab18/racelab18.json`.

---

## See Also

- [[poc3-racelab-bodyfitted-demo]] — Tier 68, the column these knobs will eventually move.
- [[poc3-racelab-car-union]] — Tier 64 and W282, whose envelope the machine's `validity` predicate carries.
- [[gap-worklist]] — W290, W291 and the rows this tier opens.
