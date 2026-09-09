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

**The order is the argument, and it was re-cut on 2026-09-08.** It used to open
on the substitution refusal — beat ④ below. That is this project's strongest
*claim* and it is what a specialist should look at first, but it is a **negative
result**, so opening on it meant the first thing anyone saw was the machine
saying no. The positive result was already measured and already here, third and
fourth. Nothing was re-measured to change the order and no figure moved; the
refusal is not softened, it is sequenced, and *What this does not claim* below
still carries the tension whole.

### ① A better wing, and it gets there four times sooner

Four knobs — how stiff the wing is, how thick it is, how stiff the spring is, how
high it rides — and one number to push up: downforce. The search finds a wing
with **19.3%** more of it than the one it started from, inside both the stress
and the deflection ceilings.

What lets it is that the whole coupled simulation, air and metal and suspension
together, is differentiable end to end: one backward pass returns how the
downforce moves for all four knobs at once. The alternative — what you do without
a gradient — is to try many designs and keep the good ones. Both run here on the
same problem, one evaluation each in turn, with the live march held so neither is
timing the contention.

Recorded: **4.31× in wall-clock**, then 22.2× in rollouts, with the two columns
landing within **1.003×** of each other in the objective. The wall-clock figure
is quoted first and the rollout figure is labelled flattering, because the
rollout ratio prices a gradient iterate at one forward evaluation and it is 4.9
of them on this graph. The gradient does not find a *better* design than the
population does; it finds an equally good one sooner.

Both columns are **warmed before either clock starts**. Unwarmed, a three-step
race reads a 25× adjoint premium where a warmed one reads 5.1.

The live race is short and **can go either way** — its budgets are *scaled* from
the recorded 30 : 200 rather than chosen, and below about nine gradient iterates
Adam has not turned yet.

### ② And the win is real, not a constraint quietly walked through

The first thing worth asking about a number like 19.3% is whether the search
cheated to get it.

Both components state, in writing, what they are valid for: the wing is a
small-deflection model and stops being one past a stated bend; the suspension
stops being one below a stated ride height. Until 2026-09-04 that check ran on
the plain march and **not on the path the optimiser takes**. So the search walked
the car through its own floor and reported a **27.5%** gain for a design the model
does not stand behind.

With the check live it is **19.3%**, the optimiser is refused eight times out of
thirty-one, and the screen names which component objected and at which macro-step.
The smaller number is the one that is inside the model.

Untick *enforce declared envelopes* and the whole page turns red and stamps every
number `OUTSIDE THE MODEL`. Both recorded columns are shown side by side: same
code, same box, same budget, same seed, one check between them.

The reason this matters past this one bug: with fifteen to twenty components, the
chance that at least one is outside its training regime on a *novel* design
approaches one — and a novel design is the entire point. Knowing when to abstain
is the load-bearing safety property of the whole idea, and an optimiser is the
worst adversary it will ever have, because a search finds and exploits precisely
the places where a model is confidently wrong.

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

### ④ How you know when not to trust the fast answer

A search that wins on the clock is only worth having if you know when *not* to
believe it. This is that check, running on the same graph — and here it is
correctly declining a case it should decline.

Poseidon-T is a frozen 20.8-million-parameter neural operator trained by someone
else on a great deal of fluid dynamics — exactly the kind of model this whole
approach is supposed to run on, and the one that would make beat ① orders of
magnitude cheaper. The panel compiles the same graph three times:

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

**Underneath the three verdicts: what the compiler measured.** The rule behind
them used to rest on a yes-or-no — does a component's influence reach past the
cut, or not? Asked that way it answers *yes* for every component in this project,
the plainest classical solver included, which is the check being unavailable
rather than a fact about any of them. **W153** replaced the indicator with a
measurement of how far influence actually reaches, and the three separate:

| agent | measured reach | influence past the cut |
|---|---|---|
| `WindowNS`, the local classical solver | **0.186** | **bitwise zero** past 20 cells, its declared reach, over all 13,924 cells beyond it |
| `WindowNS` carrying its own pressure solve | **0.496** | still **26%** of its peak at 20 cells out, and 12% at 60 — *still falling, slowly* |
| **Poseidon-T** | **0.790** | still **61%** at 20 cells out, and **61%** at 60 — *it has stopped falling off* |

The figure in the third column is each agent's response **normalised by its own
peak**, worst over the three probed states. The raw far-field magnitude would
have been the obvious thing to print and it ranks the three the *other* way round
— 76.07 for the embedded classical solve against 9.78 for Poseidon-T, because it
is a size in each agent's own units while the bar is measuring *decay*. It is
sampled at two distances because one would let the panel call both non-local rows
flat, and only one of them is.

On a scale where 1 is *everywhere*. **This changes no verdict and could not
have**: Poseidon-T is refused before the measurement and after it, and where the
overlap is narrower than the domain of dependence the refined envelope returns
the old indicator to 1.000000, so no past verdict can have moved silently. What
it changes is *why* — the compiler is not pattern-matching "neural network,
therefore no". It measured a graded property and put the three in the order the
physics says they belong. Read from `out/w153/w153.json`, which travels with this
bundle; the demo does not re-run it.

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
  (W1, W3, W49) ON THIS GRAPH until W157 measured them here; they are now\n  declared and `unmeasured` is empty, and it is still not green -- five named\n  rules stand, two of which (W136, W138) are recorded faults in the checker\n  rather than facts about the wing. Formerly, while a constant was missing, every graph came back
  `admit-uncertified`.
- **This is not faster than the solver it replaces.** Every component here is a
  classical solver, and beat ①'s 4.31× is one *search method* against another on
  the same solvers — not a claim about the solvers themselves. The four-to-six
  orders of magnitude that make the whole idea worth having arrive only with
  learned components — and beat ④ is the framework refusing the most obvious
  candidate. That tension is the honest state of the programme, not a footnote,
  and moving that beat from first to fourth does not soften it.
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
| **Tier 33's measured reach** | `out/w153/w153.json` | beat ④'s three reach bars are read out of it at load time. Carried whole (~0.46 MB) rather than reduced to the six numbers the screen quotes, because a six-number excerpt *is* the hard-coded copy the rule below bans |
| the driver that produced the first | `scripts/w141_poc2_frontwing.py` | `python scripts/w141_poc2_frontwing.py --stages ...` |
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

**The demo never writes either artifact** and never falls back to a hard-coded
copy of a number in one. If `w141.json` is absent the recorded columns say so; if
`w153.json` is absent beat ④ shows its three verdicts without the reach bars
underneath them, which is the state that beat was in before the measurement
existed. A wall-clock figure in one of this project's own documents was wrong by
a factor of four the next day on unchanged code, which is why numbers are read
from the run that produced them or not shown at all.

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
