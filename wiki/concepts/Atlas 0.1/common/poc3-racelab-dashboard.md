# The dashboard, and the coolant knob that finally reaches the march

*Case study, PoC 3. New 2026-09-16. Tier 71.*

---

# 0. The result, in one paragraph

**The body-fitted column has a screen, and building it immediately proved the one thing the measurement could not.** Tiers 69 and 70 wired section 3.3's knobs and reported, for each, the subsystems that responded — but the dashboard showed the coolant loop "responding" while the marching circuit's return temperature stood at **exactly $313.633745$ K across every step**. The knob reached a *probe* that rebound a module constant, measured, and put it back; the running `CarUnion` had already read the old value and never saw it. That was **W291**, and it is now closed: `cooling_loop.LoopSettings` is threaded through every leg, `LoopSolve`, `SweepStudy` and `CarUnion`, and moving ambient temperature from $300$ to $318$ K on the live dashboard took the marching circuit's return from $312.361655$ to $\mathbf{331.668647}$ K — **a $19.31$ K rise on a number that could not move before.** The knob tally goes from six wired and two global to **eight wired and none global**. The page itself found a second defect that no test would have: every WebSocket upgrade was refused with **403** while the page loaded and its API answered, so a viewer would have watched a dead frame forever.

---

# 1. Rebuild on commit, not on slider move

The user's decision (2026-09-16). A geometry change re-cuts the car, which costs $66$–$90$ s for the composite and up to $63$ s for the probe operator. It cannot run inside a frame and it must not run on every drag of a slider. So the page has three states and says which one it is in:

| | what happens | what the page shows |
|---|---|---|
| a **cooling or powertrain** knob moves | applied to the running march at once | the subsystems that responded, in the event log |
| a **geometry** knob moves | recorded; **the car is not re-cut** | the knob goes amber and the header reads `MARCHING THE CAR BEFORE THESE CHANGES` |
| **Commit** | the car is re-cut from the committed knobs | a recompiling overlay with the stage and a progress bar |

That header line is the point. Section 4.1 asks for a visible recompiling state, and a demo that marched the old car while the sliders showed the new one is the defect this project keeps finding — `racelab` already carries a note recording the time "the editor showed one car and the march ran another".

---

# 2. W291: a knob that reported a response the march did not have

`cooling_loop` declared `MDOT` and `T_AMB` as module constants and ten call sites read them directly. A knob could only reach the loop by **rebinding** them, which Tier 69 recorded as `how="global"` with the note that it is unsafe under a background march thread.

**The dashboard turned that from a code smell into a measurement.** The page reported `ambient_t → responded: cooling_loop, brake_thermal`, and the telemetry's coolant return did not move — not by a little, but by exactly nothing, $313.633745$ K on every step, because the probe's rebinding was undone before the marching circuit ever looked.

A knob that reports a response the march does not have is worse than a knob that does nothing: the first misleads, the second merely disappoints. Section 3.3's rule says as much about sliders, and this is the same rule one level down.

## 2.1 The fix, and why it is additive

`LoopSettings` is a frozen operating point — `mdot`, `t_amb`, `ua_rad`, `w_pump`, `cp` — **each defaulting to the constant it replaces.** `_Leg` carries one, and the legs read `self.s.mdot` where they read `MDOT`. `make_legs`, `LoopSolve`, `SweepStudy` and `CarUnion` all take one and pass it down.

Because the defaults are the constants, every existing leg, graph, record and weight hash is unchanged; what moves is that a caller can now say *which* operating point, and the leg carries it.

| | without touching a constant |
|---|---|
| $\dot m = 0.05$ | $310.155908$ K |
| $\dot m = 0.30$ | $310.865178$ K |
| $T_{\text{amb}} = 273$ | $288.133248$ K |
| $T_{\text{amb}} = 318$ | $325.778299$ K |

Those reproduce the old rebinding probe's numbers exactly, which is what says the refactor changed the route and not the physics.

## 2.2 The record moves with the knob

A leg's capability record was built with

```
weight_hash = f"loop-leg-{agent_id}-mdot{MDOT:.6g}"
```

read from the **module**. So moving the coolant flow moved the physics and left the record claiming the machine that used to run — the compiler would have been reasoning about an expert that was not there. The hash is now built from the leg's own settings tag, so it moves with the knob.

## 2.3 The proof is on the running march, not in a unit test

| | coolant return |
|---|---|
| before the fix, every step | $313.633745$ K — frozen |
| after the fix, baseline at step $400$ | $312.361655$ K |
| after `ambient_t` $300 \to 318$, at step $807$ | $\mathbf{331.668647}$ K |
| | $\mathbf{+19.31}$ K |

The circuit sub-cycles every $400$ fluid steps, so the reading is not continuous — which is why the check waited for the next sub-cycle rather than reading straight after the slider moved.

**[AI Inference]:** the $331.67$ K here is above the $325.78$ K the standalone probe gives at the same ambient, and the gap is expected rather than a discrepancy: the probe solves a circuit with `q_machine = 0`, while the marching block carries the machine's winding heat through J2. A page that showed the probe's number as though it were the march's would be a third version of the same defect, one level further out.

---

# 3. What opening the page found, and no test would have

**Every WebSocket upgrade was refused with 403** — while `/` returned the page and `/api/meta` answered $200$. A socket check that only asked whether the server was up would have passed. The page loaded, rendered its sliders, and would have shown a dead frame forever.

The cause is worth writing down because it is invisible:

`bodyfitted_server` carries `from __future__ import annotations`, so the handler's `sock: WebSocket` is the **string** `"WebSocket"`. FastAPI resolves annotations with `get_type_hints` against the **module's globals** — and FastAPI had been imported *inside* `create_app`. The name did not resolve there, the parameter was not recognised as the socket, and the handshake was refused.

The existing `demo_racelab/server.py` works because it imports FastAPI at module level. A test now asserts that this one does too, because moving the import back inside the function would restore the defect silently and nothing except opening the page would notice.

---

# 4. The commit, and the three defects behind one `dimension mismatch`

The first end-to-end commit failed with a bare `ValueError: dimension mismatch`, raised inside a device's ring probe two hundred lines from the cause. Chasing it produced the tier's sharpest findings.

**A $0.12$-cell pitch of the floor changes the composite's unknown count.**

| | unknowns | fingerprint |
|---|---|---|
| $\text{rake} = 0.00$ | $377{,}267$ | `a1f67a8e2792` |
| $\text{rake} = 0.12$ | $\mathbf{377{,}176}$ | `1b39f94d6479` |

Ninety-one fewer, because moving the floor moves which background cells are holes. So a field written on one car is not merely *stale* on another — it is the **wrong length**.

**`load_state` never checked.** It handed the old car's $377{,}267$-long vectors to a composite with $377{,}176$ unknowns, `set_state` accepted them without complaint, and the mismatch surfaced later somewhere that says nothing about what went wrong. It now raises `StateBelongsToAnotherCar` naming both counts. The unknown count is a cheap and exact proxy for "same composite", and the guard lives in `load_state` rather than in the one caller that happened to be bitten.

**And the honest consequence, which is the part worth keeping.** A newly committed car has no prefix of its own, so settling it means marching from the solver's initial field — which is **rest**. Tier 62 measured that start failing on this car: the impulsive start put the first step's divergence at $2.3\times10^{3}$ with a momentum solve that did not converge, and the from-rest start grew a spurious road boundary layer. **Only the walls-ramped spin-up works, and it is a script, not something a dashboard can do between frames.**

So the column re-cuts the car and **declines to march it**:

```
fingerprint a1f67a8e2792 -> 1b39f94d6479   (changed)
unknowns    377267 -> 377176 (-91)
prefix refused: holds 377267 unknowns and this composite has 377176
released:   NOT SPUN UP
declined to march: True
```

`step()` raises rather than drawing. A demo that marched a field known to blow up and called the pictures physics is precisely the artifact this project exists not to produce — **W293** carries the gap.

**[AI Inference]:** the general shape is that a cache key which identifies *which* car is not the same as a check that the cached object *fits* this car. The fingerprint in the filename answered "whose is this?"; it took a length check to answer "can this be used at all?", and the two failed in different places. A cache that is keyed but not validated is the same class of defect as Tier 68's raster cache, one level up.

---

# 5. The predictions

| id | claim | outcome |
|---|---|---|
| N1 | the default `LoopSettings` is the module constants field for field | **held** |
| N2 | a full reach report moves no module constant | **held** |
| N3 | the leg's weight hash moves with the coolant flow | **held** |
| N4 | threading the settings adds nothing to Tier 45's moved set | **held** — $16$ moved, $0$ added |
| N5 | a geometry commit lands on a different fingerprint and an **honest** release | **held** |

N5 was re-registered before the run, not after it. Its first form asked for "a release state named for that car", which the commit stage showed is not always achievable — a car with no spun-up field has no such state and cannot be given one between frames. The claim that survives is the one that is actually the requirement: a release is honest when it either names *this* car's file or refuses to march at all, and never when it silently reuses another car's field.

---

# 6. What this tier did NOT do, named

- **The porous column's dashboard is untouched.** This is a second page at its own route; requirements 13.3 keeps both columns because the porous one is the only place section 12's window-flipping criteria can be told over fourteen windows.
- **No checkpoint is loaded**, so the learned switch, the presets and the referent still have nothing to act on — and the page says so rather than offering a control that does nothing.
- **The state-of-charge knob still reaches the battery and not the marching machine.** Tier 69 gave `MachineAgent` its own `v_oc`, so a machine *can* now be told which battery it draws from; what is missing is the wiring, because `racelab.machine_for_host` builds the marching machine without the knob.
- **`road_speed` and `battery_power` are still refused**, with their reasons on the page: the first needs a dimensional scale, the second a different powertrain model.
- **A committed car cannot be spun up** (W293). The commit re-cuts it and the column **declines to march**, because the only start that works on this car is the walls-ramped spin-up and that is a script. So a geometry change is something a viewer can make and see refused, with the reason — not something they can watch run.
- **Nothing was downloaded or installed**, no machine was rented, the unlicensed structural checkpoint was not loaded, and nothing was pushed.

```
python scripts/tier71_dashboard.py --out out/racelab20 --stages settings,control,commit,summary
python -m pytest tests/test_tier71_dashboard.py tests/test_tier36_cooling_loop.py -p no:cacheprovider
python -m uvicorn atlas.demo_racelab.bodyfitted_server:create_app --factory --port 8014
python scripts/vault_scan.py wiki
```

The record is `out/racelab20/racelab20.json`.

---

## See Also

- [[poc3-racelab-knobs]] — Tier 69, where W291 was measured and named.
- [[poc3-racelab-knobs-wired]] — Tier 70, `rake` and `duct_area`, and `KnobState`.
- [[poc3-racelab-bodyfitted-demo]] — Tier 68, the engine this page drives.
- [[gap-worklist]] — W291 closed here; W292 and the state-of-charge wiring still open.
