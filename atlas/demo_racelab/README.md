# PoC 3 — RaceLab, the live demo

A car simulated as a graph of interchangeable physics experts, with every fluid
window flippable between a classical solver and a learned one, and the speed and
accuracy consequences shown per window and for the whole machine.

```bash
python -m atlas.demo_racelab --open
```

Then open <http://127.0.0.1:8013/>. On Windows PowerShell the same command works;
there is no shell script to invoke.

It needs `fastapi`, `uvicorn` and `pillow` beside the package's own
dependencies, and it needs the build repo for `reference.WindowNS` exactly as
`atlas/cases/wing_fsi.py` does. The learned column additionally needs
Poseidon-T in the local Hugging Face cache and `scOT` installed; **without them
the classical column still runs** and the page says the learned switch is
unavailable rather than failing.

Run this first if `out/racelab2/cache/settled.npz` is absent:

```bash
python scripts/tier52_racelab_switch.py --stages spinup
```

Without it the demo releases from a uniform freestream, **says so on screen**,
and shows a transient whose first 58 macro-steps leave the disk's induction
clamp — which the envelope stamp will report in red. That is `demo_frontwing`'s
own rule and the reason it exists.

---

## What is on the screen

**The centre** is the car: the field overlay as a PNG layer, the thirteen bodies
drawn from `racelab.car_bodies` — the same call the physics makes, so the
picture is the model's geometry and not a second one — the fourteen window
boxes, the overlap bands shaded, and the two device planes in green. A window's
tint is its current mode. Click one to select it.

**The left panel** carries the switch: four presets, the field selector, the
march controls, and — first, because it is the thing a viewer most needs and
would otherwise never see — **the lead the composition layer asks for**.

**The right panel** is the window inspector: the selected window's mode, its
one-step error against `WindowNS` **on the same input state**, the lead that
measurement was taken at, both per-call costs, and the graph's compile verdict.

**The bottom** is the global telemetry: speed against the all-classical column,
rms field error against it, the mode ledger, and the machine's operating point.

---

## The three things this demo is built to make impossible to miss

### 1. The lead ratio

The composition layer exchanges four times per macro-step, and one 128-cell
window spans 2.0 tiling length units, so a learned window is asked for a step
**1/32 of the one Poseidon-T was trained to take**. CS-20 measured the
per-window error across leads:

| lead / native | 1/8 | 1/4 | 1/2 | **1** | 2 |
|---|---|---|---|---|---|
| median error | 0.361 | 0.562 | 0.730 | **0.171** | 0.241 |

**A minimum at the native lead**, and the error *grows* as the step shrinks
below it. So a viewer who flips a window to *learned* is watching an expert
asked for the thing it is worst at, and **the page says so beside the switch**
rather than reporting the resulting error as the expert's.

That is **W225**: this demo shows a true result and, taken alone, a poor
demonstration. What it must never do is show the error without the ratio.

### 2. The envelope stamp

Every declared predicate — `MachineAgent.validity`, the disk's induction clamp,
the fluid window's cell-Reynolds bound — is consulted by
`racelab.RaceRollout._absorb` at the end of every macro-step.

The engine runs with `enforce=False` **by design**: a dashboard that stops
reports nothing. What it does instead is **stamp**. The moment a predicate
declines, the stage border turns red, a banner names the expert and the reason,
and the banner says that every number below it is a measurement of a state the
model declines to stand behind. That is `demo_frontwing`'s `OUTSIDE THE MODEL`
pattern, and W145 is why both demos have it.

The all-classical column stays inside. **Every learned column leaves within a
handful of macro-steps**, which is the demo's most honest frame.

### 3. Four of five families have no learned option

The left panel lists them with the reason for each — no shippable learned
structural expert (NeuberNet is unlicensed and local-only and must not appear in
a bundle), no vendored conduction operator, a 1-D advection leg with a
closed-form decay, and an algebraic circuit. **One of five governing families on
this graph has a learned option, and sixteen of twenty-six agents.** A viewer
should leave knowing that, because it is a true and important fact about the
field rather than about this code.

---

## The accuracy story, as it is actually computed

1. **The per-window one-step error needs no referent at all**, and is the number
   the inspector leads with. Press *measure windows*: each expert takes one
   macro-step from the **same input state** and the learned one is differenced
   against the classical one. Exact, cheap, and independent of any march.
2. **The global rms error is against a real referent** — an all-classical
   composed march held beside the live one from the same release state. It costs
   a second march, and the frame reports both step times so the price is visible.
3. **The composed classical answer is not the physics.** It carries its own
   composition defect against a single-domain monolith. **That second referent
   is not run**, here or in CS-19, and neither claims it.
4. **Certified mode has a proof, not a measurement** — and it is *not a
   per-macro-step mode*. Defect correction certifies a **fixed point**, and an
   explicit time step has none to certify, so selecting it does not march: the
   page says so and the assignment does not change. It is measured per window by
   `racelab_switch.certified_window`, where CS-20 found the checkpoint bought
   5 and 7 classical calls out of 270 and 165 at 3x the wall time. **W226.**

---

## What this demo is not

- **It is not evidence that a learned expert pays.** It is the measurement that
  says it does not, here, and why. [[case-study-ladder-to-f1]] §14.3's sentence
  is unchanged.
- **The sliders of the requirements' §3.3 are not wired.** Phase 1 wired three
  geometry knobs into `CarParams` and no further; a parameter that moves a body
  can move it across a window boundary, which the static decomposition (E1, a
  fixed graph) forbids without a visible rebuild. **A slider that moves a number
  nothing reads is worse than no slider**, so none is shown.
- **No 3-D.** That is phase 4 and is optional to ship.
- **No bundle.** That is phase 5.
- **One Reynolds number, one grid, one incidence**, and every body porous with
  no boundary layer and no Kutta condition — `poc2-frontwing-results`' own list,
  now thirteen times over instead of once.

## Layout

| file | what it is |
|---|---|
| `engine.py` | the persistent march, the mode switch, the field encoder, the compile worker and everything the frame carries |
| `server.py` | FastAPI: binary websocket frames for the field, JSON for the rest |
| `static/index.html` | the page |

## See also

- `wiki/concepts/Atlas 0.1/common/case-study-racelab-graph-atlas-0.1.md` — phase 1: the car, the decomposition, the graph, the march
- `wiki/concepts/Atlas 0.1/common/case-study-racelab-switch-atlas-0.1.md` — phase 2: the switch, and every number this page displays
- `atlas/demo_frontwing/README.md` — the PoC 2 demo this is modelled on
- `POC3-RACELAB-REQUIREMENTS.md` — the specification, and §11's hole list
