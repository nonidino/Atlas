# Atlas Workbench

Build a simulation by hand: choose a kind of physics, draw the domain, cut it
into pieces, and run the pieces against the undivided domain. The page shows
whether they agree, which one was faster **on your machine**, and what the
compiler says about joining the pieces. There are eight kinds: a wind farm,
heat conduction, an electric film, a river carrying a pollutant, sound crossing
from air into water, a loaded plate, a heated bimetal strip, and a block cooled
by a water channel. The **Fast example** button opens a case that shows each
kind's best honest speed-up, and says what limits it when there is none.

This branch is a **self-contained copy**: everything it needs is here or is
installed by one command, it runs on a laptop with no GPU, and nothing is
installed outside this folder.

---

## Run it

**Windows**, in PowerShell (what Windows 11 opens by default):

```powershell
git clone --branch atlas-workbench --single-branch https://github.com/nonidino/Atlas.git atlas-workbench
cd atlas-workbench
.\run.cmd
```

The `.\` is not decoration: PowerShell does not run programs from the current
directory, so a bare `run.cmd` fails with *"The term 'run.cmd' is not
recognized"*. In `cmd.exe` either form works.

**macOS or Linux**:

```bash
git clone --branch atlas-workbench --single-branch https://github.com/nonidino/Atlas.git atlas-workbench
cd atlas-workbench
./run.sh
```

That is the whole procedure. The script finds a Python, builds a virtual
environment in `.venv` beside itself, installs every package at a **pinned
version** (the ones the workbench names in `requirements.txt`, and everything
*they* pull in through `constraints.txt`), runs a self-test once, and opens the
page in your browser. Later runs go straight to the page.

No `git`? Download the branch as a zip (**Code → Download ZIP** with the
`atlas-workbench` branch selected), unpack it and run the same script. A zip
does not carry the executable bit, so on macOS or Linux run `bash run.sh`.

### Options

| | |
|---|---|
| `--no-open` | serve without opening a browser |
| `--port 8031` | another port (by default, the first free one from 8020) |
| `--check` | the self-test alone |
| `--reinstall` | throw `.venv` away and build it again |
| `--no-torch` | skip torch, about 200 MB, which only the learned case uses |
| `--no-gmsh` | skip Gmsh, which only *File → Import geometry from Gmsh* uses |

### Requirements

- **Python 3.10, 3.11 or 3.12.** The pinned numpy, scipy and pandas publish no
  wheels for 3.13, so the launcher checks the version rather than discovering
  it halfway through an install. On a Mac, `brew install python@3.12` or the
  installer from [python.org](https://www.python.org/downloads/); to point at a
  particular interpreter, `PYTHON=/path/to/python3.12 ./run.sh` (Windows:
  `set PYTHON=C:\path\to\python.exe`).
- **On Debian and Ubuntu**, the system Python ships without its `venv` module;
  the launcher says so and names the package to install.
- **macOS: not yet run on a Mac.** `run.sh` is written for the bash macOS
  ships (3.2), every pinned package publishes macOS wheels for Apple silicon
  and Intel, and the script has run on Linux; until it has run on a Mac, treat
  macOS as expected to work, not as verified.
- **No GPU.** Every arm runs on the CPU. torch, when it is installed, is the
  CPU-only wheel.
- **Network at install time only.** After that the page needs none: every
  script and stylesheet it loads is served by its own server.

### Is it working?

```bash
./run.sh --check          # Windows: .\run.cmd --check
```

In about a minute:

1. every package imports, and the optional ones are reported separately;
2. the wind farm's solver loads **from this folder's `vendor/`** (the path is
   printed and asserted: a solver loaded from anywhere else would be a
   different test);
3. **each of the eight kinds** compiles, through the same compile the page's
   *Compile* runs, and marches a step or two of every arm it offers through
   the page's own runner; the fields must be finite inside the domain, and
   every exact control the kind registers must hold (its "bit for bit"
   identities, such as the threaded pieces equalling the serial ones);
4. **the page**: the workbench's own server is started on a free local port,
   the page is fetched as a browser would fetch it, every script and
   stylesheet it names is fetched too, and its document is pulled over the
   page's live connection (its WebSocket).

The exit code is `0` when all of that passed, `3` when it passed and an
optional package is absent, and anything else when something is broken, in
which case the launcher does not start the server.

### Optional parts

- **torch** (CPU-only) is for the learned case, which is not in this version
  yet. On a Mac it needs Apple silicon: torch 2.7.1 publishes no Intel-macOS
  wheel. Without it, everything else runs.
- **Gmsh** is for *File → Import geometry from Gmsh*. It is GPL-licensed, so it
  is installed on your machine from PyPI and is not part of this branch.
  On Linux it may install and still not load (it needs the system's OpenGL and
  X libraries); the self-test then reports it absent, and everything else runs.

If an optional install fails, the launcher says so, starts the page anyway,
and tries again next time (or pass `--no-torch` / `--no-gmsh`).

---

## The speeds are your machine's

Every run is timed where it runs, with the arms taking turns so that no arm
always runs first, and the record of each run keeps the machine's state beside
its numbers (on Windows, whether it was on battery, and which other Python
processes were running). The speed-ups the project quotes were measured on a
22-thread laptop on mains power; a machine with fewer cores will show smaller
ones for the cases that run their pieces on threads. Compare ratios, not
seconds.

## Tests

```bash
.venv/bin/python -m pytest tests -q        # Windows: .venv\Scripts\python -m pytest tests -q
```

These are the workbench's test files, character for character the ones in the
`Atlas` repository, which is the only reason passing them here means anything.
A test that needs something this folder does not carry skips and says why: one
cross-checks the workbench's finite elements against the build repository's
structural solver, which is not vendored here. A skip is not a pass, and the
count is printed.

## Layout of this branch

```
run.cmd / run.sh     one command: builds .venv, installs pinned, self-tests once, serves
run.py               what they run; --check is the self-test alone
requirements.txt     the packages the workbench imports, pinned
constraints.txt      everything those pull in, pinned
atlas/               the Atlas 0.1 framework, its case studies, and the workbench
  workbench/           the page, the eight kinds, the runner, the compiler's view
vendor/              the one package of the build repository the workbench loads
                     (the wind farm's fluid solver), where its loader looks;
                     vendor/LICENSE is that repository's MIT licence
tests/               the workbench's tests, unchanged, and the helper they share
scripts/             the scripts those tests import
SOURCE_COMMITS       the exact commits this was built from, and the pins
```

What you make goes in `out/workbench/cases/`, each run's record beside its
case. Delete `.venv` (or the whole folder) to undo the install; pip keeps its
usual download cache in your user folder, which `pip cache purge` empties.

**This branch is a copy.** Develop in `atlas-0.1`, not here.

## Licences

- **Atlas** (everything outside `vendor/`): no licence has been chosen yet.
  This branch is private until the project's website launches.
- **`vendor/`**: from the build repository, under its MIT licence
  (`vendor/LICENSE`).
- Every package the launcher installs comes from PyPI (torch from the PyTorch
  index) under its own licence. **Gmsh is GPL-2.0-or-later**, installed on your
  machine and not redistributed here.
