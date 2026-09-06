# A simulation that tells you when it doesn't know

A Formula 1 front wing, simulated with two physical couplings live at once — it
rides up and down on its suspension, and it bends under its own aerodynamic
load, and each of those changes the other. The whole thing is differentiable, so
an optimiser can search the design space instead of sampling it.

**That part is not new.** Coupled aero–structural simulation is routine
industrial practice, and it has been for decades.

What is new is on the right-hand side of the screen: **before it runs anything,
the software reads a written description of every component and every joint and
decides, joint by joint, whether it is willing to certify the result** — and when
it is not, it names the exact rule that failed and what nobody has measured.

```bash
git clone --branch poc2-frontwing-demo --single-branch https://github.com/nonidino/Atlas.git poc2
cd poc2
./run.sh                # Windows PowerShell: .\run.cmd  (the .\ is required)
```

Then open <http://127.0.0.1:8012/>. macOS, Linux and Windows. Python 3.10–3.12.
No GPU needed. **Nothing is downloaded at run time and nothing is downloaded at
install time except the pinned packages** — there is no model checkpoint here,
which is a property of what this demo shows and not a convenience. The whole
bundle is about five megabytes.

`./run.sh --check` runs the self-test only, which answers "does this work here"
in about a minute.

---

## Why anyone should care, in four sentences

A fast simulator that is *confidently wrong* is worse than a slow one that is
right, because an optimiser will find and exploit precisely the places where it
is confidently wrong. Every serious attempt to speed up engineering simulation
with machine learning runs into that, and the usual answer is to validate the
model afterwards on cases somebody chose. This project's answer is different:
make the composition layer **read what each component declares about itself** and
refuse, up front, the assemblies whose components cannot support the theory the
coupling scheme rests on. It is the difference between a tool that gives you a
number and a tool that gives you a number *and its own opinion of that number*.

---

## What is on screen, in five beats

Each has a **plain-language** description and a **technical** one, side by side;
there is a toggle in the header for the plain one alone.

### ① Offer it a real AI model and watch it refuse

Poseidon-T is a frozen 20.8-million-parameter neural operator trained by someone
else on a great deal of fluid dynamics — exactly the kind of model this whole
approach is supposed to run on. The panel compiles the same graph three times:

| description handed to the compiler | verdict |
|---|---|
| **WindowNS**, the classical solver actually running | 2 seams refused, 7 uncertified |
| **Poseidon-T, as its own wrapper declares it** | 2 refused, 7 uncertified — but with **three new complaints** at every fluid seam |
| **Poseidon-T, with the elliptic part declared** | **9 of 9 refused** |

Nothing about the model changes between those rows. Only the *description* does.
The reason is one line long: **you cannot cut a problem into pieces if each piece
secretly depends on all the others** — and a neural operator's output at any point
depends on its input everywhere, by construction of the architecture.

That is not a rigged failure. The measurement behind it is `W93`:
`probe.support_reach` poked a delta at a real wake seam and found Poseidon-T's
response nonzero in **every one of the 128 seam cells**, 64 cells from the poke,
where its record declares a stencil radius of **2**. A factor of 32, at every
amplitude from 1 to 1e-3.

It is also a negative result about the whole compose-pretrained-operators
programme, including this project's own, and it is why the roadmap changed:
climb the ladder with classical solvers, and price each learned substitution one
seam at a time.

> **This beat needs six windows.** With the single-window referent there is no
> decomposition, so the rule that refuses has no premise to fire on, and it
> correctly stays quiet. The panel says which column it is in.

### ② Turn off one check and watch the optimiser lie

Both components state, in writing, what they are valid for: the wing is a
small-deflection model and stops being one past a stated bend; the suspension
stops being one below a stated ride height. Until 2026-09-04 that check ran on
the plain march and **not on the path the optimiser takes**. So the search walked
the car through its own floor and reported a **27.5%** gain for a design the model
does not stand behind.

With the check live it is **19.3%**, the optimiser is refused eight times out of
thirty-one, and the screen names which component objected and at which macro-step.

Untick *enforce declared envelopes* and the whole page turns red and stamps every
number `OUTSIDE THE MODEL`. Both recorded columns are shown side by side: same
code, same box, same budget, same seed, one check between them.

The reason this matters past this one bug: with fifteen to twenty components, the
chance that at least one is outside its training regime on a *novel* design
approaches one — and a novel design is the entire point. Knowing when to abstain
is the load-bearing safety property of the whole idea.

### ③ Freeze one joint and see what it was worth

Make the spring infinitely stiff and the wing is bolted in place — the graph
becomes CS-12, one of the two case studies this was assembled from. Make the wing
infinitely stiff and it cannot bend — the graph becomes CS-10, the other one.
March each and read the downforce.

At the starting design the suspension joint is worth **0.49%**. At the optimised
design it is worth **8.16%**, because the optimum rides much closer to the ground,
where a small change in height is a large change in force. The elastic column at
the optimum **cannot be measured at all**: a rigid wing there makes *more*
downforce and drives itself through the floor, so the model declines.

That measurement also falsified a claim this project had written down — the
prediction was that stiffening the wing shrinks what the *elastic* joint is
worth. It went the other way, on the other joint, by a factor of seventeen.

### ④ Race the gradient against the population, on one clock

One backward pass through the whole coupled rollout returns how much the
downforce changes for each of four design knobs at once. The alternative — what
you do without gradients — is to try many designs and keep the good ones. Both
run here on the same problem, one evaluation each in turn, with the live march
held so neither is timing the contention.

The honest result is smaller than this project used to quote: **1.8× to 4.3× in
wall-clock**, not the 18×–22× a naive rollout count reports, and the two columns
land within **1.003×–1.008×** of each other in the objective. The gradient does
not find a better design; it finds an equally good one sooner. Two errors
compounded in the old number and both are named on screen.

The live race is short and **can go either way** — its budgets are *scaled* from
the recorded ones rather than chosen, and below about nine gradient iterates Adam
has not turned yet.

### ⑤ One needle for energy at the seams

If a coupling is quietly creating or destroying energy, everything downstream of
it is wrong in a way no plot will show you. Three accountings run live and the
gaps between them are the point: a static one reads `1.0000` (the normalising
level), adding the interface power the two seams carry takes it to `5.93e-2`, the
half-step correction to `3.30e-2`, and once settled it is `3.32e-8`.

It is the one diagnostic defined identically at every rung of the roadmap, so a
regression on a twenty-agent graph is comparable with a measurement on a
two-agent one.

---

## What this does **not** claim

- **Nothing here is certified.** Not one joint is green, and none ever has been
  in this project: three constants the bounds rest on have never been measured
  (W1, W3, W49), and while that is true every graph comes back
  `admit-uncertified`.
- **This is not faster than the solver it replaces.** Every component here is a
  classical solver. The four-to-six orders of magnitude that make the whole idea
  worth having arrive only with learned components — and beat ① is the framework
  refusing the most obvious candidate. That tension is the honest state of the
  programme, not a footnote.
- **The physics is a demonstration, not an engineering model.** One Reynolds
  number (125), one grid (208×144), one angle of attack (20°). The wing is a
  porous inclined plate with no boundary layer of its own and no circulation
  condition; the floor is a rolling road, so there is no viscous choking of the
  gap. The plate is a 32×2 Q1 plane-stress cantilever and Q1 elements lock in
  bending. **No stress number here is a statement about an alloy** — `E*` is a
  design knob in flow units calibrated to a deflection.
- **The big question is unanswered.** Does composition error grow faster than the
  number of joints? Nobody has measured it, here or anywhere. This assembly has
  nine joints; a car has fifteen to twenty subsystems.

---

## What is in this bundle, and how each thing resolves

| what | where | how it resolves |
|---|---|---|
| the framework and the demo | `atlas/` | on `sys.path` from `run.py` |
| the fluid expert `reference.WindowNS` | `vendor/src/atlas/cases/windfarm/` | `window_ns.load_reference` reads `$ATLAS_BUILD_REPO/src/atlas/cases/windfarm`; `run.py` points it at `./vendor` |
| the structural expert `ThermoStruct2D` | `vendor/src/atlas/solvers/` | `thermal_strain.load_solvers` reads `$ATLAS_BUILD_REPO/src/atlas`, same variable |
| the settled flow field | `out/w141/settled.npz` | the demo releases from it, so the first frame is the state every recorded number was measured at |
| **the recorded run** | `out/w141/w141.json` | every "recorded" figure on screen is read out of it at load time |
| the driver that produced it | `scripts/w141_poc2_frontwing.py` | `python scripts/w141_poc2_frontwing.py --stages ...` |
| the tests | `tests/` | `pytest tests/` — `conftest.py` sets `ATLAS_BUILD_REPO` |
| the write-ups | `docs/` | the results page, the novelty assessment, both parent case studies |

**No source file is modified for the bundle.** That is deliberate and it is what
earns the copy its credibility: the tests here are the same files, character for
character, as the ones in the development tree, so passing them here means the
same thing as passing them there.

The two repositories both have a top-level package called `atlas` and they do not
collide, because the build repo's package is loaded under a private module name
through `importlib.util.spec_from_file_location` rather than being put on
`sys.path`.

**The demo never writes `w141.json`** and never falls back to a hard-coded copy of
a number in it. If the artifact is absent the recorded columns say so. A
wall-clock figure in one of this project's own documents was wrong by a factor of
four the next day on unchanged code, which is why numbers are read from the run
that produced them or not shown at all.

---

## Flags

```bash
./run.sh --open --tiling six --race-grad 14 --race-pop 93
```

| flag | default | what it does |
|---|---|---|
| `--tiling` | `six` | `six` = the composed column, six overlapping windows; `single` = the **referent**, one window over the whole domain, differing by the cut and nothing else. Beat ① needs `six` |
| `--coupling` | `tight` | `tight` solves both seams as one 33-equation Newton system; `split` solves them in turn (the ablation); `lagged` solves once per macro-step and holds |
| `--horizon` | 12 | macro-steps differentiated per live optimiser iteration. Warm-started and short — **a different estimator** from the results page's, and the screen says so |
| `--ablation-steps` | 60 | macro-steps per column in beat ③. Recorded: 120 |
| `--race-steps` | 16 | macro-steps per objective evaluation in beat ④ |
| `--race-grad` / `--race-pop` | 14 / 93 | beat ④'s budgets. Scaled from the recorded 30 : 200, not chosen |
| `--no-enforce` | off | start with beat ②'s envelope check **off**. Everything produced is stamped `OUTSIDE THE MODEL`; this is not a mode anything should be measured in |
| `--threads` | 1 | torch CPU threads. **One by default and it is measured**: this workload is 1.9–2.0× *slower* on eight, and the two agree bitwise, so the thread count is a clock rather than a variable |
| `--port` | 8012 | |

---

## Licence and provenance

Both source repositories are private, and so is this branch. Every solver here is
first-party unpublished work; there are no third-party weights in this bundle and
no non-commercial licence attached to anything in it. `SOURCE_COMMITS` records
which commit of each repository this copy was taken from.
