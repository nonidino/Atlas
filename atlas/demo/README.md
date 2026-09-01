# Live wind-farm design demo (PoC 1a)

A browser UI over the composed graph from [`atlas/cases/wind_farm_design.py`](../cases/wind_farm_design.py).
Built for someone with no fluid dynamics and no machine learning, standing in
front of a screen for three minutes.

**It shows two things, and they are different claims.**

| | what it shows | which solver |
|---|---|---|
| **① Optimise** | the layout improves because the whole coupled simulation is differentiated: one pass gives the derivative of farm power with respect to every position and every angle at once | coupled graph **only** — the classical solver returns a number, not a gradient |
| **② Check** | for any *single* layout, the coupled graph and an undivided classical solver are run side by side, and both the answers and the per-step costs are compared | both, alternately, in a second window |

The honest story is the pair: search cheaply and differentiably with the
composed graph, then check the answer against the classical stack.

## Run it

```bash
python scripts/w112_farm_demo.py
```

Then open <http://127.0.0.1:8011/>. Equivalently `python -m atlas.demo`.
The head-to-head window is at `/compare`, or the button on the main page.

```bash
python scripts/w112_farm_demo.py --domain medium --turbines 12 --horizon 6 --open
```

| flag | default | what it does |
|---|---|---|
| `--domain` | `small` | `small` (6 windows, 11.0 x 7.5 D), `medium` (12), `large` (24) |
| `--turbines` | 6 | 4-25, clamped to what the box holds at the spacing limit |
| `--horizon` | 7 | macro-steps differentiated per optimiser iteration |
| `--steps` | 30 | macro-steps per classical-vs-coupled comparison |
| `--device` | `cpu` | `cuda` moves the **coupled** side only; the classical monolith is numpy and stays on the CPU either way, which the panel says out loud |
| `--threads` | min(8, cores) | torch CPU threads |
| `--open` | off | open a browser once the server is up |
| `--port` | 8011 | |

### Requirements

`torch`, `numpy`, `scipy`, `pillow`, `fastapi`, `uvicorn`. The fluid expert
itself lives in the build repo; set `ATLAS_BUILD_REPO` if it is not at the
default checkout path. To run this on a machine that has neither repo, use the
self-contained bundle on the `poc1-windfarm-demo` branch — see
[`PACKAGING.md`](PACKAGING.md).

No GPU required. CPU is the default because the point is that this runs on a
laptop.

## The three-minute demo script

1. **Let it settle.** Point at the picture. *"Grey is undisturbed wind. Blue is
   slow air — that's the wake behind a turbine. Orange is air speeding up as it
   squeezes past."* The power number falls while the wakes form: the farm is
   eating its own wind.
2. **Drag one turbine out of the row behind another.** Its wake moves with it and
   the power number rises. *"Nothing was retrained. The wake is being computed."*
3. **Press Optimise.** Every few seconds the turbines step to new positions and
   angles and the power trace climbs. *"It isn't trying layouts one at a time. It
   differentiates the whole coupled simulation and knows which way to move every
   turbine at once."*
4. **Press Stop, then the play button on the Replay row.** The farm walks back
   through the layouts the optimiser tried, wakes and all. Pause anywhere and the
   flow keeps settling around that layout.
5. **Open the head-to-head window.** The same layout is solved by the coupled
   graph and by one undivided classical solver, alternately, and both fields are
   drawn as they go. *"That's the honest part: we search fast with this model and
   check with the old one."* Read the per-step costs off the bars.
6. **Drag the wind-direction slider.** The validity panel turns red. *"It tells
   you when you've left the range it was checked in, instead of confidently
   making something up."*

## What is on screen

- **Plan view** of the farm, wind blowing left to right, with the streamwise
  velocity field behind it. The colour ramp is the one the wake-array viewer
  uses, stop for stop: dark and cool in a wake, **neutral grey at the
  freestream**, warm where the flow speeds up. It is **fixed** — the same colour
  always means the same speed, so frames are comparable, including between the
  two panels of the head-to-head window.
- **Turbines** as bars, drawn to scale (a rotor is 1 D). Drag the bar to move
  one; drag the small white tip to angle it into the wind.
- **Total farm power**, and what fraction of the wind arriving at the farm that
  represents.
- **A power trace** over the last few hundred macro-steps.
- **Replay** with play/pause and a scrubber, described below.
- **A validity panel** and **a classical head-to-head**, both described below.

## Replay

Every optimiser iteration records its design vector. The replay walks back
through them: `▶` plays, `⏸` holds, and the slider seeks.

Only the layout is recorded, not the flow field — a stored field per iteration
would be about 1.3 MB a step. So the replay **re-marches**: the turbines jump to
where they were at step *i* and the wake re-forms around them. That is why the
cursor advances once every `REPLAY_HOLD` (2) macro-steps rather than once per
frame — any faster and the layout runs ahead of the field that is supposed to
explain it. Pausing does not stop the fluid: the wakes go on settling around the
layout you paused on, which is the only way to see what that layout was actually
doing rather than what it looked like in passing.

## The head-to-head: one layout, two solvers, timed

The second window (`/compare`) marches the layout on screen twice from still air:

- through the **coupled graph** — the domain cut into overlapping windows, each
  solved by its own instance of the fluid expert, re-assembled every step; and
- through `scaling_ladder.reference_monolith` — the **same discretization over
  the whole domain with no cut at all**.

Because the two differ *only* by the cut, the difference between their power
numbers is the price of splitting the domain up and re-assembling it, not a
disagreement about physics. The largest velocity difference anywhere in the
field is reported alongside it.

**They alternate, one macro-step each.** Running them concurrently would make a
better animation and a worthless measurement: they would be competing for the
same cores, and each per-step number would be a function of what the other one
happened to be doing. Alternating gives every timed region the machine to
itself, and because both sides sit at step *i* at the same moment, their two
fields are directly comparable — which is where the field-difference number
comes from.

For the same reason **the live march is held still while a comparison runs**,
and the window says so while it is doing it. A per-step cost measured against a
moving load is a measurement of the load. The engine parks itself and resumes
on its own the moment the timing is over; results are cached by layout hash, so
asking the main window's *Check this layout* button for a layout already checked
is instant.

**One thing the panel will show you and the demo does not hide:** at these farm
sizes the undivided classical solver is *not* slower than the coupled graph.
Splitting a domain into windows and re-assembling them costs more than solving
it in one piece. The coupled graph's advantage is not forward speed — it is that
it can be **differentiated end to end**, which is what makes tens of optimiser
runs beat the hundreds a derivative-free search needs.

## How fast is it? The demo measures itself

**This file used to give a table of per-step costs and the table was wrong.** It
said 212 ms per composed macro-step at 6 windows. The same code, on the same
box, unchanged, later measured ~890 ms — because the box is a Core Ultra 7 155H
and it had clocked itself down from 3.8 GHz to 1.4 GHz.

So no wall-clock constant is quoted here. `Engine.eta_s` reports an exponential
moving average of what iterations have actually cost **on the machine you are
running on**, the *seconds / optimiser step* readout shows it, and it is marked
*(estimate)* only until the first real iteration has been timed. The head-to-head
window times both solvers per step, holds out the first step (which pays for FFT
plans and allocation), and reports the median.

What *is* stable is the shape: one gradient macro-step costs about 5.5 forward
ones, and both scale with the window count. A forward macro-step is a fraction
of a second to about a second per window-grid on a laptop-class CPU, which is
what makes the field real-time enough to drag a turbine and watch its wake move.

## The validity panel

Every row is a predicate something already declared, with its number, its limit,
and (click a row) why the limit exists.

| row | limit | source |
|---|---|---|
| cell Reynolds number (freestream) | 8 | `reference.WindowNS`'s own validity predicate; the wake array sits at 7.97, so the wind-speed slider has almost no headroom above 1.0 |
| cell Reynolds number (live max) | 8 | **diagnostic, not a gate** — a developed wake is always above it (~12), which the case study records rather than hides |
| max abs u inside the stability band | 3.0 | `w100_scaling_ladder`'s band; past it the composed rollout is on a trajectory that does not stay finite |
| inflow yaw off the x axis | 2 deg | the global Leray projection declares its hypothesis as "the march holds the inlet and both laterals at (U_INF, 0)" |
| minimum turbine separation | 2 D | the packing limit below which actuator-disk momentum theory has no near-field model |
| turbines inside the domain box | — | `FarmCase.box` keeps rotors clear of the freestream band and the outlet |

When a **gate** fails the panel goes red and says which one. The point of
exposing knobs that can break it — wind direction especially — is that the
framework can say *"outside the range I was checked in"* instead of returning a
confident number.

## Architecture

```
atlas/demo/engine.py         the simulation: one worker thread owns the field and
                             the design vector; every UI message is queued and
                             applied between macro-steps. The head-to-head runs on
                             its own thread with its own solvers, and parks the
                             worker while it is timing.
atlas/demo/server.py         FastAPI. The main page is a websocket: JSON state and
                             control both ways, field frames down as binary PNG at
                             a capped rate, so a slow client throttles itself
                             rather than the physics. The head-to-head window is
                             plain polled HTTP, so closing the main window does not
                             interrupt a comparison in flight.
atlas/demo/static/index.html the main page — one self-contained file, no build
                             step, no CDN.
atlas/demo/static/compare.html  the head-to-head window.
atlas/demo/cli.py            flags, shared by `-m atlas.demo` and scripts/.
```

**Nothing here modifies the optimiser.** The one class that extends it,
`engine.DemoRollout`, overrides only `band` and `project` so the freestream can
be pointed and scaled — and at the default inflow it is asserted **bitwise
identical** to `wind_farm_design.Rollout` on a real macro-step
(`tests/test_tier22_demo.py`). That assertion is the licence for the subclass to
exist: it is what says the animation is the column the PoC measured.

## Tests

```bash
python -m pytest tests/test_tier22_demo.py -q
```

29 assertions. The last two groups boot the real server, open the real
websocket, take optimiser steps, and drive a full classical-vs-coupled
comparison through the same HTTP endpoints the second window uses — including
checking that the live march really did park while the timing ran.
