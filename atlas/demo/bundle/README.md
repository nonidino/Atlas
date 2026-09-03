# Atlas PoC 1a — a composed physics graph, faster and differentiable

A wind farm, simulated by a **graph of frozen, independently-pretrained physics
experts** coupled at typed interfaces — and, beside it, the same problem solved
by **one undivided classical solver**. The demo makes two claims on one screen
and measures both of them on your machine:

1. **The composed graph is usually faster, and the page measures whether it is
   here.** The domain is cut into overlapping windows, each marched by its own
   copy of a frozen neural operator, re-assembled every step. Against the
   undivided classical solver over the same grid that was **2.1x to 3.4x**
   faster per macro-step on the machine this was built on, **1.07x** on one
   rented Linux box, and **0.43x** — slower — on another whose CPU shares 192
   cores with other tenants. On that same box the composed column on the GPU was
   **6.4x** the CPU monolith. So the margin is a property of the machine and the
   page reports what it finds here, with the columns taking turns so each has
   the machine to itself.
2. **The composed graph can be optimised, and the classical one cannot.** One
   backward pass through the entire coupled simulation returns the derivative of
   farm power with respect to every turbine position and every yaw angle at
   once. Press *Optimise* and watch the wakes re-steer and the power climb.

Then it scores the layout the optimiser found **three ways** — by the composed
column, by the same cut with a classical solver in the windows, and by the
undivided classical solver, which is the accuracy authority — so the gain is a
number a verifier confirms rather than one the model that proposed it reports.

This branch is a **self-contained copy**. Everything needed to run it is here,
including the fluid solver (which normally lives in a different repository) and
the 83 MB frozen checkpoint (which would otherwise be downloaded). It runs on
macOS, Linux and Windows, on a laptop, with no GPU.

---

## Run it

**macOS / Linux**

```bash
git clone --branch poc1-windfarm-demo --single-branch https://github.com/nonidino/Atlas.git poc1
cd poc1
./run.sh
```

**Windows** — in PowerShell, which is what Windows 11 opens by default:

```powershell
git clone --branch poc1-windfarm-demo --single-branch https://github.com/nonidino/Atlas.git poc1
cd poc1
.\run.cmd
```

The `.\` is not decoration. PowerShell does not run programs from the current
directory, so a bare `run.cmd` fails with *"The term 'run.cmd' is not
recognized"* even though the file is right there. In `cmd.exe` either form
works.

Then open **<http://127.0.0.1:8011/>**.

That is the whole procedure, from a machine with nothing set up. The script
finds a Python, builds a virtual environment in `.venv` beside itself, installs
every dependency at a **pinned version** (choosing a CPU or CUDA build of torch
by whether an NVIDIA driver answers), runs a self-test that marches both columns
and prints what they cost here, and only then starts the server. If any package
is missing it says which one and stops. Nothing is installed outside `.venv`;
delete that folder to undo all of it.

The first run takes a few minutes because torch is a large download. Later runs
start in seconds. `./run.sh --reinstall` rebuilds the environment from scratch.

No `git`? Download the branch as a zip from the repository's **Code → Download
ZIP** with the `poc1-windfarm-demo` branch selected, unpack it, and run the same
script.

On Windows, if `git clone` reports **"Filename too long"**, the checkout path is
too deep for the bundled checkpoint's own directory name: clone somewhere
shorter (`C:\poc1`), or run `git config --global core.longpaths true` first.

### Requirements

- **Python 3.10, 3.11 or 3.12.** The pins have no wheels outside that range and
  the launcher checks rather than discovering it halfway through an install. To
  point at a specific interpreter: `PYTHON=/usr/bin/python3.12 ./run.sh`.
- About **2.5 GB of disk** for the virtual environment on a CPU-only install,
  most of it torch; more with the CUDA wheel.
- **No GPU needed.** If you have one, `./run.sh --device cuda` moves the composed
  columns onto it. The undivided classical solver is numpy and stays on the CPU
  either way, which the speed panel says out loud.
- **Network at install time only**, for pip. The checkpoint is already here.

### Is it working?

```bash
./run.sh --check
```

Imports every dependency and names any that is missing, loads the frozen
checkpoint from this checkout, marches a few macro-steps of each column
alternately, and prints what they cost on this machine. About a minute.

---

## What is on screen

- **Two velocity fields, side by side, on one fixed colour ramp.** Cool and dark
  is slow air — a wake. Neutral grey is undisturbed wind. Warm is air speeding
  up around the array. The scale never changes and is identical in both, so the
  same colour always means the same speed. Turbines are bars drawn to scale
  (a rotor is 1 D); a bar's **rotation is its yaw angle**, which is a design
  variable the optimiser moves.
- **A speed panel**, in ms per macro-step per column with the ratio between
  them. Every number is measured here — see *How fast* below.
- **An accuracy panel**: the worst single-cell velocity difference against the
  undivided solver, and the difference in farm power. The first is a bound over
  the whole field and is always much the larger; the second is what the turbines
  integrate and what the optimiser maximises.
- **The optimiser**, with the power trace climbing and a replay of the layouts
  it tried.
- **A three-way scoring table** for the layout it found.
- **A validity panel** whose every row is a predicate something already declared,
  with its number, its limit and (click a row) why the limit exists.

## How fast — the demo measures itself, and quotes no constant

An earlier version of this file gave a table of per-step costs. It said 212 ms
per composed macro-step. The same code, on the same box, unchanged, later
measured **~890 ms** — because the box is a Core Ultra 7 155H and it had clocked
itself down from 3.8 GHz to 1.4 GHz.

So no wall-clock constant is quoted here. A **measurement region** parks the live
march and runs each column one macro-step at a time in turn, so each is timed
with the machine to itself; the panel reports an exponential average over
regions, with the first step of each column held out because it pays for FFT
plans and allocation, and says how many regions it rests on. Until the first one
lands the panel says so instead of showing a number.

What is stable, and what should transfer between machines, is the **ratio**.

## The claim, and exactly what it does not say

- **Not more accurate.** The undivided classical solver is the authority here and
  the composed column is measured against it, not the other way round.
- **Not a replacement for the classical stack.** It is a **pre-screen**: search
  wide and cheap with the composed model, verify the shortlist with the
  classical one. The scoring table is that pipeline, on screen.
- **Not the whole gain.** Scored by the verifier, a layout found entirely inside
  the frozen-expert column is worth about **71 %** of what a layout found on the
  classical column is worth (measured at two problem sizes; see
  `docs/poc1a-frozen-expert-results.md` §7.2). The demo shows the verified gain
  it actually achieves and says where that fraction comes from.
- **Not production-licensed.** The expert is **Poseidon-T**, `camlab-ethz`,
  20.8 M parameters, frozen — this project did not train it and does not
  fine-tune it. Its weights are **CC-BY-NC-4.0: research use only, no commercial
  use.** That is stated on screen and in `vendor/POSEIDON-T-LICENCE.md`.
- The live power number is a **demo number**: warm-started from whatever field is
  on screen and differentiated over a short horizon, so it carries the history of
  every layout you have dragged through. The measurement region is the defensible
  one, because every column marches it from still air under the same protocol.
  The page says which is which.

## Options

```bash
./run.sh --domain large --turbines 25 --steps 30 --open
```

| flag | default | what it does |
|---|---|---|
| `--domain` | `medium` | `small` (6 windows), `medium` (12), `large` (24). The published speed ratio was measured at 12 and 24 windows; at 6 the composed column has fewer windows to amortise over and the ratio is smaller or inverted, which the panel will show. |
| `--turbines` | 12 | 4–25, clamped to what the box holds at the spacing limit |
| `--expert` | `poseidon` | `poseidon` (the frozen checkpoint) or `reference_exposed` (the classical solver in the windows) |
| `--horizon` | 5 | macro-steps differentiated per optimiser iteration |
| `--steps` | 40 | macro-steps per column per measurement region; below about 30 the unoptimised layout's wakes have not developed and the verified gain is invisible |
| `--device` | `cpu` | `cuda` moves the composed columns; the monolith is numpy and stays on the CPU |
| `--no-third-column` | off | skip the classical-windows column in a measurement (a third of its cost) |
| `--no-measure-on-start` | off | do not time the columns on the starting layout at boot |
| `--threads` | min(8, cores) | torch CPU threads |
| `--open` | off | open a browser once the server is up |
| `--port` | 8011 | |

## Tests

```bash
.venv/bin/python -m pytest tests/ -q      # Windows: .venv\Scripts\python -m pytest tests\ -q
```

Three files, several minutes.

- `tests/test_tier21_wind_farm_design.py` — the optimiser: bitwise equality
  against the original numpy column at every layer, the actuator disk
  integrating to its own thrust, the cos³γ yaw law emerging rather than being
  imposed, and the adjoint checked against central finite differences.
- `tests/test_tier26_poseidon_design.py` — the frozen column: the taped
  checkpoint asserted **bitwise** against the untaped adapter, zero parameter
  gradients, and the gradient checked against finite differences to the
  precision a float32 forward pass allows.
- `tests/test_tier22_demo.py` — the demo: the classes it layers on the optimiser
  asserted bitwise identical to their parents on a real macro-step at the
  default inflow; the validity panel green at the default and red when the wind
  is turned; and the last groups boot the real server, drive a full measurement
  region through the same endpoints the page uses, and check that the live march
  really did park while the timing ran.

---

## Layout of this branch

```
run.sh / run.cmd     one command; builds .venv, installs pinned, self-tests, starts
run.py               the launcher proper; --check runs the self-test alone
requirements.txt     every dependency, pinned
atlas/               the Atlas 0.1 framework and case studies
  cases/wind_farm_design.py    the differentiable design case (the optimiser)
  demo/                        the browser demo: engine, server, two pages
vendor/              the fluid solver, and the frozen checkpoint, each in the
                     layout its own loader expects
scripts/             the batch drivers behind the measured results
tests/               the three test tiers above
docs/                the pages the results rest on
PROVENANCE.md        where every file came from, and why it is where it is
```

**This branch is a copy.** Develop in `atlas-0.1`, not here.
