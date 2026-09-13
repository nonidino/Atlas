# Atlas PoC 3 — RaceLab

A racing car simulated as a **graph of interchangeable physics experts** — a
fluid solver in fourteen overlapping windows along the car, the front wing's
structure and suspension, a radiator core, a recovery turbine driving a motor
generator, the block it heats, and the coolant loop that cools it — with a live
dashboard where **every fluid window can be flipped between a classical solver
and a learned neural operator**, and the speed and accuracy consequences of each
flip shown per window and for the whole machine.

**The car is the stage; the point is the switch.** And the page says what the
switch is worth here, including when the answer is bad: a learned window is asked
for a time step a thirty-second of the one its checkpoint was trained to take,
and an all-learned column leaves the fluid expert's own declared envelope within
a handful of macro-steps, which the page stamps in red rather than hiding.

This branch is a **self-contained copy**. Everything needed to run it is here,
including the solvers (which normally live in a different repository), the 83 MB
frozen checkpoint (which would otherwise be downloaded), and the settled flow
field the demo releases from. It runs on macOS, Linux and Windows, on a laptop,
with no GPU.

---

## Run it

**macOS / Linux**

```bash
git clone --branch poc3-racelab-demo --single-branch https://github.com/nonidino/Atlas.git poc3
cd poc3
./run.sh
```

**Windows** — in PowerShell, which is what Windows 11 opens by default:

```powershell
git clone --branch poc3-racelab-demo --single-branch https://github.com/nonidino/Atlas.git poc3
cd poc3
.\run.cmd
```

The `.\` is not decoration. PowerShell does not run programs from the current
directory, so a bare `run.cmd` fails with *"The term 'run.cmd' is not
recognized"* even though the file is right there. In `cmd.exe` either form
works.

Then open **<http://127.0.0.1:8013/>** (or add `--open`).

That is the whole procedure, from a machine with nothing set up. The script
finds a Python, builds a virtual environment in `.venv` beside itself, installs
every dependency at a **pinned version** — the packages this demo names in
`requirements.txt`, and everything *they* pull in through `constraints.txt` —
runs a self-test that marches both columns and prints what they cost here, and
only then starts the server. If a package is missing it says which one and
stops. Nothing is installed outside `.venv`; delete that folder to undo all of
it.

The first run takes a few minutes because torch is a large download. Later runs
start in seconds. `./run.sh --reinstall` rebuilds the environment from scratch.

No `git`? Download the branch as a zip from the repository's **Code → Download
ZIP** with the `poc3-racelab-demo` branch selected, unpack it, and run the same
script. (A zip does not carry the executable bit: on macOS or Linux run
`bash run.sh` instead of `./run.sh`.)

On Windows, if `git clone` reports **"Filename too long"**, the checkout path is
too deep for the bundled checkpoint's own directory name: clone somewhere
shorter (`C:\poc3`), or run `git config --global core.longpaths true` first.

### Requirements

- **Python 3.10, 3.11 or 3.12.** The pins have no wheels outside that range and
  the launcher checks rather than discovering it halfway through an install. To
  point at a specific interpreter: `PYTHON=/usr/bin/python3.12 ./run.sh`.
- **On Debian and Ubuntu**, the system Python ships without its `venv` module;
  the launcher says so and names the package (`sudo apt install python3.12-venv`).
- **On macOS, an Apple-silicon Mac.** The torch version pinned here publishes no
  wheel for Intel Macs.
- **No GPU.** Every column in this demo runs on the CPU — the composed march is
  float64 and the learned expert is stepped on the CPU inside it — so the
  launcher installs the CPU-only torch wheel even when a GPU is present, rather
  than a few gigabytes this demo would never touch.
- **Network at install time only**, for pip and for one source archive (below).
  The checkpoint is already here, and nothing is fetched at run time.

### Is it working?

```bash
./run.sh --check          # Windows: .\run.cmd --check
```

Imports every dependency and names any that is missing; loads the solvers and
**asserts they came from this checkout's `vendor/`**; checks that the settled
field was settled around **this** car; compiles the graph; marches the
all-classical column and then the all-learned column beside it, through the
demo's own engine, with the per-window one-step error; and finally **asks the
demo's own server, on a real local port, for a frame over the WebSocket the page
uses** — because a self-test that only drives the engine once passed on a bundle
whose page could never receive a frame. A minute or two. The exit code is `0`
when both columns marched and the page can receive them, `3` when the classical
column marched, the page can receive it, and the learned expert is not
available here, and anything else when something is broken.

### With no network, or without `scOT`

`scOT`, the model class the checkpoint is loaded through, is the one thing the
launcher fetches from outside pip's own index (see *Licence*). If that fetch
fails, **the classical column still runs**: the self-test exits `3`, the
launcher says so and starts the server anyway, and the page greys the learned
switch out **with the reason** rather than offering a control that would do
nothing. The next `./run.sh` tries the fetch again.

---

## What is on screen

- **The centre** is the car: the flow field as an image, the car's bodies drawn
  from the same call the physics makes, the fourteen window boxes with their
  overlap bands, the two device planes in the radiator duct, and each window
  tinted by its current mode. Click a window to select it.
- **The left panel** carries the switch — four presets, a field selector, the
  march controls — and, first, **the lead the composition layer asks a learned
  window for**, because it is the thing a viewer most needs and would otherwise
  never see.
- **The right panel** is the window inspector: the selected window's mode, its
  one-step error against the classical solver **on the same input state**, both
  per-call costs, and the graph's compile verdict.
- **The bottom** is global telemetry: speed against the all-classical column,
  the rms field error against it, the mode ledger, and the machine's operating
  point.
- **The header** names both experts and their licences at all times.
- **A red stamp** — `OUTSIDE THE MODEL` — the moment any declared envelope
  declines the state being shown.
- **2-D / 3-D.** The 3-D toggle marches a half-car in a box, classical only, and
  says why the learned switch is greyed out there.

`atlas/demo_racelab/README.md` explains every number and what it rests on.

## Licence

The bundled weights are **CC-BY-NC-4.0 — research use only, no commercial use**.
`vendor/POSEIDON-T-LICENCE.md` carries the attribution and says so; the page
says so in its header.

**`scOT` is not in this branch.** `github.com/camlab-ethz/poseidon` publishes no
licence file, so its code is not ours to redistribute; the launchers install it
from the upstream archive at a pinned commit, with `--no-deps`.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q      # Windows: .venv\Scripts\python -m pytest tests\ -q
```

These are the same files, character for character, as the ones in the `Atlas`
repository — which is the only reason passing them here means anything. A test
that needs something a machine does not have (the learned expert, say) skips
and says why; a skip is not a pass, and the count is printed.

## Layout of this branch

```
run.sh / run.cmd     one command; builds .venv, installs pinned, self-tests, starts
run.py               the launcher proper; --check runs the self-test alone
requirements.txt     the packages this demo names, pinned
constraints.txt      everything those pull in, pinned
atlas/               the Atlas 0.1 framework and case studies
  cases/racelab.py              the car as a graph, and its march
  cases/car_geometry.json       the car's shape, as data
  demo_racelab/                 the dashboard: engine, server, page
vendor/              the build repository's solvers, and the frozen checkpoint,
                     each in the layout its own loader expects
out/racelab5/        the settled field the demo releases from, and the recorded
                     run every sizing in the demo was measured in
out/racelab*/        the earlier recorded runs the tests read
scripts/             the batch drivers behind the recorded runs
tests/               the test files for PoC 3, unchanged
wiki/                the case-study pages those tests quote, at their own paths
PROVENANCE.md        where every file came from, and why it is where it is
SOURCE_COMMITS       the exact commits and the checkpoint snapshot this was built from
```

**This branch is a copy.** Develop in `atlas-0.1`, not here.
