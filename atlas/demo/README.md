# Live wind-farm design demo (PoC 1a)

A browser UI over the composed graph from [`atlas/cases/wind_farm_design.py`](../cases/wind_farm_design.py).
Built for someone with no fluid dynamics and no machine learning, standing in
front of a screen for three minutes.

**It makes two claims on one screen and measures both of them on the machine it
is running on.**

| | what it shows | which columns |
|---|---|---|
| **① Faster** | the domain cut into overlapping windows, each marched by its own copy of a **frozen** fluid expert and re-assembled every step, costs about a third of what one **undivided** classical solver over the same grid costs per macro-step | all of them, taking turns |
| **② Differentiable** | one backward pass through the whole coupled simulation returns the derivative of farm power with respect to every position and every yaw angle at once, so an optimiser moves them all together | the composed graph **only** — the classical solver returns a number and no derivative |

and then it prices the result honestly: the layout the optimiser found, scored
by the composed column, by the same cut with a classical solver in the windows,
and by the **undivided classical solver**, which is the accuracy authority.

## The expert, named, with its licence

The default fluid expert is **Poseidon-T** (`camlab-ethz/Poseidon-T`), a frozen
20.8 M-parameter neural operator this project did not train and does not
fine-tune. **Its weights are CC-BY-NC-4.0: research use only, no commercial
use.** That is stated in the page header beside the expert's name, because the
speed claim is a property of what is inside the windows and a viewer who cannot
tell which expert is running cannot read the number. `--expert
reference_exposed` puts the classical solver in the windows instead, and the
whole screen re-labels itself.

## Run it

```bash
python scripts/w112_farm_demo.py
```

Then open <http://127.0.0.1:8011/>. Equivalently `python -m atlas.demo`.
The full-size head-to-head is at `/compare`, or the link in the setup panel.

```bash
python scripts/w112_farm_demo.py --domain large --turbines 25 --steps 30 --open
```

| flag | default | what it does |
|---|---|---|
| `--domain` | `medium` | `small` (6 windows, 352x240), `medium` (12, 464x352), `large` (24, 688x464) |
| `--turbines` | 12 | 4-25, clamped to what the box holds at the spacing limit |
| `--expert` | `poseidon` | `poseidon` (frozen checkpoint) or `reference_exposed` (classical windows) |
| `--horizon` | 5 | macro-steps differentiated per optimiser iteration |
| `--steps` | 40 | macro-steps per column per measurement region; below about 30 the unoptimised layout's wakes have not developed and the verified gain is invisible |
| `--device` | `cpu` | `cuda` moves the **composed** columns; the monolith is numpy and stays on the CPU, which the panel says out loud |
| `--no-third-column` | off | skip the classical-windows column (a third of a measurement's cost) |
| `--no-measure-on-start` | off | do not time the columns on the starting layout at boot |
| `--threads` | min(8, cores) | torch CPU threads |
| `--open` | off | open a browser once the server is up |
| `--port` | 8011 | |

**Why the default is `medium` and not `small`.** The rung is load-bearing for
what the screen says. [[poc1a-frozen-expert-results]] §8 measured the composed
Poseidon column at $3.15\times$ the monolith's forward speed at 464x352 / 12
windows and $3.64\times$ at 688x464 / 24; §10 of [[atlas-proof-of-concept-1]]
measured the *classical* composed column at $1.54\times$ **slower** at 352x240 /
6. A demo that opened at six windows would be showing the case where the cut has
not yet paid for itself. `small` is still there, and reports whatever it measures
there.

### Requirements

`torch`, `numpy`, `scipy`, `pillow`, `fastapi`, `uvicorn`, and — for the frozen
column — `transformers` and `scOT`, plus the Poseidon-T checkpoint. The fluid
expert itself lives in the build repo; set `ATLAS_BUILD_REPO` if it is not at the
default checkout path. To run this on a machine that has neither repo, no
checkpoint and no environment, use the self-contained bundle on the
`poc1-windfarm-demo` branch — see [`PACKAGING.md`](PACKAGING.md). That bundle
pins every version, carries the checkpoint, and is one command from a bare
clone.

No GPU required. CPU is the default because the point is that this runs on a
laptop.

## The three-minute demo script

1. **Let the first measurement finish.** It starts on its own. Both pictures fill
   in together, one macro-step at a time — *"the same farm, solved twice: on the
   left cut into twelve pieces with a frozen neural network in each, on the right
   in one piece by a classical solver. Grey is undisturbed wind, blue is a wake,
   orange is air squeezing past."* The speed panel fills in as it goes.
2. **Read the ratio.** *"Same machine, same problem, one after the other so
   neither is competing with the other for cores. The cut-up one is about three
   times faster per step — and that number was produced here, in the last
   minute, not read off a slide."*
3. **Read the accuracy panel.** *"And it is not free: the worst cell in the field
   is off by this much, and the farm power by this much. The undivided solver is
   the referent — we are not claiming to be more accurate than it, we are
   claiming to be cheap enough to search with."*
4. **Drag one turbine out of the row behind another.** Its wake moves with it and
   the power number rises. *"Nothing was retrained. The wake is being computed."*
5. **Press Optimise.** Every few seconds the turbines step to new positions and
   angles and the power trace climbs. *"It isn't trying layouts one at a time. It
   differentiates the whole coupled simulation and knows which way to move every
   turbine at once. The classical solver on the right cannot do this at all —
   that is the point of the whole architecture."*
6. **Press Stop, then Time & score both columns again.** *"Now the honest part:
   the layout it found, priced by the solver we trust. That gain is confirmed by
   something with no composition error in it."*
7. **Press play on the Replay row.** The farm walks back through the layouts the
   optimiser tried, wakes and all. Pause anywhere and the flow keeps settling
   around that layout.
8. **Drag the wind-direction slider.** The validity panel turns red. *"It tells
   you when you've left the range it was checked in, instead of confidently
   making something up."*

## What is on screen

- **Two plan views, side by side**, wind blowing left to right, on **one fixed
  colour ramp** — the one the wake-array viewer uses, stop for stop: dark and
  cool in a wake, **neutral grey at the freestream**, warm where the flow speeds
  up. Fixed, never auto-scaled, and identical in both, so the same colour always
  means the same speed and the two pictures can be compared directly.
- **Turbines** as bars drawn to scale (a rotor is 1 D). **A bar's rotation is its
  yaw angle** — a design variable, not a decoration. Drag the bar to move one;
  drag the small white tip to angle it.
- **The speed panel**: ms per macro-step for every column, and the ratio.
- **The accuracy panel**: worst single-cell velocity difference against the
  undivided solver, and the difference in farm power.
- **Total farm power** and its trace, from the live march.
- **The optimiser**, with a replay of every layout it tried.
- **The three-way scoring table.**
- **The validity panel**, whose every row is a declared predicate.

## The measurement region — where every number comes from

One layout, marched from still air by every column, **one macro-step each in
turn, with the live march parked**. It is the same discipline the `/compare`
window has always used, promoted to the main screen, and it is what makes all
three panels readable at once: they are read off one march rather than three.

- **Alternating** rather than concurrent, because two solvers competing for the
  same cores measure the contention and not themselves.
- **The live march parks**, and the page says so while it is doing it. A per-step
  cost measured against a moving load is a measurement of the load. The engine
  parks itself, waits for the worker to have *actually* stopped rather than
  assuming the message arrived, and resumes on its own when the timing is over.
- **Every column sits at step $i$ at the same moment**, which is what makes the
  worst-cell difference between their fields meaningful.
- **The first step of each column is held out** of the headline: it pays for FFT
  plans, allocation and the first CUDA launch, and on a 20-step march it is
  5-10x the others.
- Results are cached by layout hash, so asking twice for a layout already
  measured is instant.

One runs automatically at boot, on the starting layout, so the headline is on
screen without anyone having to know to ask for it — and so the score book has
its reference row.

## How fast is it? The demo measures itself, and quotes no constant

**This file used to give a table of per-step costs and the table was wrong.** It
said 212 ms per composed macro-step at 6 windows. The same code, on the same
box, unchanged, later measured ~890 ms — because the box is a Core Ultra 7 155H
and it had clocked itself down from 3.8 GHz to 1.4 GHz. During the rebuild of
this demo, two measurement regions forty minutes apart on that same box differed
by a factor of three again.

So **no wall-clock constant is quoted here or written into the code**. There is
no cold-start table left to fall back on: `Engine.speed` starts empty, is filled
only by measurement regions, and the panel says "not measured yet" until one
lands. `Engine.eta_s()` returns `None` rather than a guess, and the screen marks
the optimiser estimate *(estimate)* until a real iteration has been timed.
`tests/test_tier22_demo.py` asserts all of that, including that the removed
table has not come back.

What *is* stable, and what should transfer between machines, is the **ratio**.
The published ones, measured elsewhere and quoted with their source, are
$3.15\times$ (12 windows) and $3.64\times$ (24) — carried in the payload's
`published` field, never mixed with a measured one.

## The three-way scoring table

[[poc1a-frozen-expert-results]] §7.2, in the form a live demo can reach. Rows are
layouts — the one the demo started from and the one the optimiser reached —
and columns are the three solvers. The gain that counts is the **monolith's**,
because it is the only one of the three with no composition error in it.

**The capture fraction is withheld unless this session earned it.** §7.2's 71 %
compares an optimum found on the frozen column against one found on the
classical column, under that same verifier. The demo reports it as a *measured*
number only when both optima have actually been scored here — switch the expert,
optimise again, score again, and the panel replaces the citation with this
machine's own number. Otherwise it shows the verified gain it did achieve and
cites 71.2 % / 70.9 % with the page and section they came from.

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

## Two estimators, and the demo keeps them apart

The big power number is the **live march**: warm-started from whatever field is
on screen and differentiated over a short horizon, so it carries the history of
every layout you have dragged through and sees less downstream interaction than
a full rollout does. It is what makes the thing interactive and it is a demo
number.

The **measurement region** is the defensible one: every column marches the layout
from still air under one protocol, which is what §9 of
[[atlas-proof-of-concept-1]] measured. The two will not agree, the page says
which is which, and the accuracy and scoring panels quote only the second.

## Architecture

```
atlas/demo/engine.py         the simulation: one worker thread owns the field and
                             the design vector; every UI message is queued and
                             applied between macro-steps. A measurement runs on
                             its own thread with its own solvers, and parks the
                             worker while it is timing.
atlas/demo/server.py         FastAPI. The main page is a websocket: JSON state and
                             control both ways, the live field down as binary PNG
                             at a capped rate, so a slow client throttles itself
                             rather than the physics. A measurement's fields are
                             served over plain HTTP, so both pictures can be drawn
                             from the same step and closing the main window does
                             not interrupt a measurement in flight.
atlas/demo/static/index.html the main page — one self-contained file, no build
                             step, no CDN.
atlas/demo/static/compare.html  the full-size head-to-head window.
atlas/demo/cli.py            flags, shared by `-m atlas.demo` and scripts/.
atlas/demo/bundle/           the launchers, pins and paperwork for the
                             self-contained branch; assembled by
                             `scripts/w112_build_bundle.py`.
```

**Nothing here modifies the optimiser.** The two classes that extend it,
`engine.DemoRollout` and `engine.DemoPoseidonRollout`, override only the
freestream (and, for the checkpoint, the translation kernel that follows from
it) — and at the default inflow each is asserted **bitwise identical** to its
parent in `wind_farm_design` on a real macro-step (`tests/test_tier22_demo.py`).
That assertion is the licence for the subclasses to exist: it is what says the
animation is the column the PoC measured.

## Tests

```bash
python -m pytest tests/test_tier22_demo.py -q
```

The last groups boot the real server, open the real websocket, take optimiser
steps, and drive a full three-column measurement through the same endpoints the
pages use — including checking that the live march really did park while the
timing ran, that the speed panel quotes nothing before it has measured
something, that the score book survives a Reset and not a change of conditions,
and that the capture fraction is withheld until this session has earned it.
