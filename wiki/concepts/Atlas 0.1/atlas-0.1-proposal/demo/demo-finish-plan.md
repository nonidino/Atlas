# Finishing the demo workbench — the header, the branching river, and holes that cross the edge

**Type:** Concept page — **plan, with the decisions made and the ones left to the owner** (folder: `Atlas 0.1/atlas-0.1-proposal/demo/`)
**Status:** written 2026-09-30, before any code. Read from `atlas/workbench/` at commit `fc9a182` (`spec.py`, `layout.py`, `flow.py`, `families/plume.py`, `app.py`, `registry.py`, `starter.py`) and its README. **Nothing here has been run**: the planning session's container had no numpy, and installs need the owner's approval. Every diagnosis below is read from the code and marked as a hypothesis until the next chat reproduces it in the served page. **Reproduced on 2026-09-30 by the demo chat**: §2.2 and §3.2 say which hypotheses held. **Steps 1–7 of §6.1 done by 2026-10-01**: the one-command install (§4.1) is verified on fresh clones on Windows and Linux and in the served page, not on macOS. **Step 8 done on 2026-10-01**: the learned case was evaluated once against its registered gate. **G5 failed**, and the case's card says so ([[demo-learned-case-plan]] §9). The installer now carries its weights.
**Hub:** [[00-proposal-workstreams]] · **Sibling plans:** [[demo-fast-examples-plan]] (item 1.4) · [[demo-learned-case-plan]] (item 1.5)
**Built on:** [[showcase-library-plan]] · [[showcase-gallery]] · [[outcome-c4-path-to-declarative-cases]] · `atlas/workbench/README.md`

---

## 0. What the demo is for, and what that changes

The owner's purpose statement, 2026-09-30: *"A way for people to actually interact and become convinced of the legitimacy of the idea and its current progress."*

Until now the workbench was built for the owner, who knows what a style, a seam and a compile verdict are. A visitor does not. **So the demo's job changes from showing everything to getting a stranger to one convincing result in under a minute.** Three consequences run through all five items:

1. **The first screen chooses a physics, not a case.** Names, descriptions and the 21-example gallery are the owner's library. A visitor needs the kind of simulation and a button that runs a good one (items 1.1 and 1.4).
2. **What the visitor draws must work.** A river that forks and a hole that bites into the edge are the first things a curious person tries. A refusal there reads as a toy (items 1.2 and 1.3).
3. **What it shows must survive a sceptic.** Every speed ratio stays measured, with its machine's state. A fast example that compiles to `refuse` at R10 undercuts itself, so W348 matters here (§5).

---

## 1. Item 1.1 — the header picks the simulation type; a new type starts a new case

### 1.1 What the owner asked

> *"There's no need to have names and descriptions for the individual cases, remove that. Instead, the header should have an option to change the type of simulation itself. When the simulation type changes, the full geometry should reset."*

### 1.2 What is there now

- The header's left slot is the case's **name button** (`app.py` `name_btn`, click to rename). The *Rename...* dialog edits `CaseSpec.name` and `CaseSpec.description` (`spec.py:388–389`).
- The physics is chosen in the **Physics layer**. A change of physics keeps the drawn shape (`spec.adapt_to_family`) and adopts the family's first example's setup (`spec.family_setup`).
- *Case > New case...* asks what the case simulates and starts it on the whole grid (`starter.new_case`).
- *Case > Examples...* lists the 21 `EXAMPLES`, each with a label and a description.

### 1.3 The decisions

| # | decision | why |
|---|---|---|
| D1 | The header's left slot becomes a **simulation-type selector**: a dropdown of the eight families by their short labels (`registry.SHORT`: wind farm, heat conduction, current in a plate, river plume, sound, loaded structure, heated structure, cooled block). The physics chooser leaves the Physics layer, so there is one place to choose | the owner's request; one control, one place |
| D2 | **Choosing a type starts a new case of that type**: `starter.new_case(family)`, ready to run on the whole grid. Geometry, materials, boundaries, windows, rotors and circuit all reset. This replaces `adapt_to_family`'s keep-the-shape behaviour for the header's switch | the owner's request: "the full geometry should reset" |
| D3 | **No confirmation dialog. One Undo brings the old case back**, and a notification says so: *"Started a new river plume case. Undo brings back the wind-farm case."* | a confirmation is friction on the most-used control; `Workbench.edit` already makes the switch one undoable edit |
| D4 | **`name` and `description` leave the case file.** Schema `case@0.6`: the migration drops both from 0.1–0.5 files. The *Rename...* dialog goes. *Save as...* keeps its file-name field (a file needs a name; a case does not) | the owner's request. Keeping dead fields invites them back into the UI |
| D5 | *Case > New case...* becomes *Case > Start over*: a new case of the **current** type | with D1, choosing a type already starts a new case; resetting without changing type still needs a home |
| D6 | *Case > Examples...* **leaves the UI.** The header gains **Fast example** (item 1.4) and, for the wind farm only, **Learned experts** (item 1.5). `EXAMPLES` stays in `spec.py`: the tests pin every example's verdict, and [[showcase-gallery]] cites them | the gallery lists names and descriptions, which is what D4 removes |

**Decided (O1, the owner, 2026-09-30): a small *More examples* list under the Fast example button**, one line per example saying what it shows, with no name. For example, *"Heat round a bend on three curved windows"*. The list shows the current type's examples. It keeps the drawn examples (`bend-3`, `s-channel`, `ring-film` and the rest) within a visitor's reach, since they are the best evidence that irregular geometry works.

### 1.4 What changes, file by file

| file | change |
|---|---|
| `spec.py` | `case@0.6`; the migration drops `name` and `description`; `slug()` gets its file name from the family and a time stamp when there is no name |
| `app.py` | the header's name button replaced by the type selector; `file:rename` removed; `file:new` becomes `file:start-over`; the type change dispatched as `case:type:<family>`, one `Workbench.edit`, with the notification of D3 |
| `inspector.py` | the physics chooser leaves the Physics layer, and the layer keeps the time, the grid and the family's parameters |
| `starter.py` | `new_case(family)` is the one entry point for a new case of a type (it already exists) |
| tests | `tests/test_workbench_shell.py` and `tests/test_workbench_starter.py`: the header has no name; the selector switches type; one Undo restores the previous case bit for bit (`model_dump` equality); a 0.5 file with a name loads and loses it |

### 1.5 Done when

- In the served page at $1090\times620$: pick each of the eight types from the header. Each is *Ready to run* (six of eight) or names its one owner-only decision (a wind farm's air reaching both grid edges; a cooled block's channel), exactly as `test_workbench_starter.py` already requires.
- Draw a shape, switch type, press Undo: the shape is back.
- No control anywhere shows a case name or a description.

**Built on 2026-09-30 (the demo chat), and walked in the served page at $1090\times620$.** D1–D6 and O1 are as decided:
- the header's type selector;
- a new case per type, with one Undo;
- `case@0.6`, with no name or description;
- *Case > Start over*;
- *More examples*, one line each, from `spec.SHOWS`.

Checked in the page:
- all eight types picked from the header read *Ready to run*;
- a drawn outline, a switch to the river, then Undo: the outline is back;
- the header's last control ends at 1,079 px of 1,090;
- a fresh load at $1090\times520$ puts nothing past the window.

What the plan did not foresee:
- **Undo must bring back the case's file, not only its content.** Undo's history held the case's JSON alone, so undoing a type switch would have left a saved case reading as never saved. Each Undo step now carries the case, its file and its key (`app._Version`). The key replaces the name where a file needs one: the file's name, the example's key, or the kind and the moment it was started (`app.new_key`).
- **The cooled block starts from its example, and a shape drawn without its water was refused outright.** Its windows are cut one per physics, and that cut needs water in the domain. A newly drawn domain is now never refused for its windows' sake: they are cut again from the new shape, and the note says why. The windows fix for Dirichlet–Neumann also cuts one piece per material, or per coolant and block, when the domain has two media.
- The header was 100 px too wide at first: *Case* and *?* ran past 1,090 px. The save state now sits under the selector.

---

## 2. Item 1.2 — a river that forks into two outlets

### 2.1 What the owner asked

> *"In the stream pollutant case, it doesn't let me have a geometry with a bifurcation, and two outlets."*

### 2.2 What the code allows, and where it probably fails

**The flow itself is not the obstacle.** `flow.potential_flow` solves $\nabla^2\phi=0$ on the river's cells with $\phi=1$ on every inlet face and $\phi=0$ on **every** outlet face, and refuses only a piece with no inlet or no outlet of its own (`flow.py:73–86`). Two outlet edges are two sets of $\phi=0$ faces, and the discharge splits between them by the channel's conductance. `families/plume.py` sums its outflow series over every outlet face. The check (`spec._check_transport`, `spec.py:1042–1060`) asks that at least one inlet and one outlet exist, not exactly one.

**Three places where a fork can fail**, in order of likelihood. **All three are hypotheses from reading the code; the next chat reproduces the owner's scenario in the served page first.**

| # | where | what goes wrong on a Y-shaped river | evidence in the code |
|---|---|---|---|
| H1 | **the automatic windows** (`layout.py`) | `ends()` picks the **two** edges where the Fiedler vector is lowest and highest. A Y has three ends. The `along` coordinate is harmonic from one end to the other with no flux elsewhere, so in the third branch, a dead end for that problem, it is nearly constant. Cutting at equal cell counts then puts the third branch in one piece, or slices it lengthwise where the values tie (ties go by cell index). Growth to full weight can then fail (`LayoutError: no reach up to N cells gives every cell a window at full weight`), or a window comes out in two disconnected parts | `layout.py:211–227` (`ends`), `:258–266` (`_cut`), `:347` (the growth limit) |
| H2 | **the starting values** (`starter.py`) | the starter marks the leftmost edge the inlet and the rightmost the outlet. A fork has a second outlet the starter does not know about. Its branch carries no flow (a dead end in the potential problem), and nothing tells the visitor why the plume will not go down it | `starter.py:136–146`, `:317–318` |
| H3 | **the compile** | a window whose seam with its neighbour is two separate curves (a band that spans both branches) may break `compile.curve_order`, which orders one seam's faces along one curve | `compile.curve_order`; not read in full |

**Reproduced on 2026-09-30 (the demo chat), headless and in the served page. The hypotheses, corrected.** I drew a Y: the stem from the left, two branches to the right, straight edges (and again with smooth ones). It runs. With both branch ends marked as outlets, the served page ran it for 6,000 steps:
- mass closed to $7.3\times10^{-14}$ ($10^{-9}$);
- the full river agreed to $1.2\times10^{-15}$ of the peak ($10^{-10}$);
- threaded equalled serial;
- the compile read `admit-uncertified`.

At 2, 3, 4 and 6 automatic windows every window was one connected piece. So:
- **H1 held, in a milder form than written.** There was no `LayoutError` and no window in two parts. But the Fiedler ends of a Y are its two branch tips, so the stem is the dead end: the along coordinate there lies between $0.497$ and $0.503$. The two windows therefore split the stem lengthwise. All 126 faces between them in the stem lie along the flow, and none cross it.
- **H2 held.** The starter marks one branch's end as the outlet: the rightmost edge, the first of the two tied ends. The other stays a bank. The case then reads *Ready to run*, and nothing says the second branch carries no water.
- **H3 did not show.** The compile admitted the Y, uncertified.
- **A fourth way, found in the probe.** A river split into two outlets by an island, drawn as a hole that reaches the river's downstream end, is refused by the hole rule: *"hole 0 is not inside the domain's outline"* (§3.2).
- **Which of these the owner met is not known.** The page writes a record only when a run starts. The owner's two river records from that morning (`out/workbench/cases/new-river-plume.results/20260930-120218.json` and `-120733.json`) are single channels. The build below removes all four.

The served page's run is `out/workbench/records/demo-repro/fork-y-two-outlets-20260930-161125.json`.

### 2.3 The design

**1. For a family that carries a flow, the windows follow the flow, not the Fiedler vector.** The river and the coolant already solve a potential $\phi$ from the inlets ($\phi=1$) to the outlets ($\phi=0$). Its level sets cross every branch from bank to bank, and it decreases monotonically along every streamline, because a harmonic function has no interior extrema. So $1-\phi$ is the natural `along` coordinate of a network of channels of any topology: a fork, a braid, a delta.

$$\text{along}(x) \;=\; 1-\phi(x),\qquad \nabla^2\phi = 0,\quad \phi\big|_{\text{inlets}}=1,\quad \phi\big|_{\text{outlets}}=0,\quad \partial_n\phi\big|_{\text{banks}}=0 .$$

**2. A band between two cuts may be several pieces; each connected piece is its own window.** Cut at equal cell counts in `along`. Then label each band's connected components (`scipy.ndimage.label`, already imported in `layout.py`), and make each component a window. A band that spans both branches gives two windows. Growth to full weight (styles A and B) runs per window, as now.

**3. For a family with no flow, a branched shape uses recursive spectral bisection.** Split the domain's cell graph at the median of its own Fiedler vector, then split each half the same way, until the count is reached. This is the textbook graph partitioner (Pothen, Simon & Liou 1990). It needs no notion of two ends, its pieces are connected in practice, and on an elongated shape the first cut falls across the middle. So it reproduces the present `along` cut where that works. **[AI Inference]:** the along/across layout should stay the default for a shape with two ends, because its pieces are "rectangles in the domain's own coordinates", which is what [[chart-operator-architecture]] wants from a chart. Bisection is the fallback for the rest.

**4. When to switch.** Count the domain's ends as the boundary's local maxima of the graph distance from the domain's most central cell. Two ends: the present layout. More: bisection, or the flow for a flow family. The Windows layer says which it used and why.

**5. A branch with no outlet is a problem the visitor can fix in one click.** After the flow solve, a connected region of the river where the speed is below $10^{-3}$ of the mean is a dead end. The check reports it as a warning, *"a branch of the river carries no water: nothing lets it out"*, with a **Fix**: *"Make the end of this branch an outlet"*. The fix picks the region's edge farthest from the inlet along the boundary.

**6. The split is shown, not set.** The Run & results tab reports each outlet's share of the discharge and of the pollutant carried out. **[AI Inference]:** a potential flow splits by geometry alone, which is a stated guess, as the inlet-on-the-left default already is. Setting the split by hand (a flux condition on one outlet) is a later option, not part of this item.

### 2.4 Checks, registered before the first run

| check | tolerance | why |
|---|---|---|
| mass released = held + carried out through **both** outlets | $10^{-9}$ per step | the family's existing balance, now over two outlets |
| the full river against the decomposed one | $10^{-10}$ of the peak | unchanged from the family: style A with one explicit step per exchange is the full domain to round-off |
| threaded equals serial | bit for bit | unchanged |
| the flow's continuity residual | $10^{-12}$ of the discharge | `flow.py` already reports it |
| **N = 1 control**: one window over the forked river equals the full river | bit for bit | the standing rule for every new code path |

**A new example**, `river-fork`: a river drawn with splines that forks round an island into two outlets, windows following the flow. It is the fork a visitor would draw, and a test pins it.

### 2.5 Done when

The owner's scenario, walked in the served page: choose *river plume*, draw a Y with the stem on the left, press the Fix the problems list offers, press Run. The plume goes down both branches; both outlets show their share; every check passes.

**Built on 2026-09-30 (the demo chat).** Two parts of the design above were corrected on the way.

- **When to switch (point 4).** The rule of counting local maxima of the graph distance round the boundary calls a rectangle's four corners four ends, so it would have re-cut every rectangle and every example. The switch is instead: **the harmonic coordinate between the two ends leaves a dead region**, cells where it changes by under $10^{-3}$ of its mean (`layout.dead_regions`).
  - Into a branch that neither end is in, the coordinate decays like $e^{-\pi x/w}$, so past about $2.2\,w$ it is dead.
  - Not one drawn example, rectangle or L has a single slow cell. The Y's stem has 2,464, and a T with a stub one width deep has none. So every earlier case keeps its windows.
  - The same measure on a river's potential is the dead-branch test of point 5.
- **The Fix of point 5 is made at once for a newly drawn fork**, as every starting value is, and said in the note. It stays in the problems list for a branch that loses its outlet later. So the walk needs no press of a Fix.

**Walked in the served page at $1090\times620$:**
- the river chosen, a smooth Y drawn with its stem on the left;
- ready at once, with inlet, outlet, second outlet, outfall and step each said in the note;
- Run: 6,000 steps, every check passing;
- the line over the cards: *"B3 takes 50.0% of the water and 54.0% of the pollutant; B6 takes 50.0% of the water and 46.0% of the pollutant."*

**The first walk found a fifth fault.** The starter placed the outfall before the river had its ends, at the middle of its cells, which on a Y is the fork. The release sat on the dividing streamline and went down one branch alone: 100% and 0%. The starting values now run in a fixed order, a river's ends first. A drawn river's outfall goes mid-river, a fifth of the way down its flow.

**`river-fork`**, compiled and run in the page:
- `admit-uncertified`, with six agents and five seams: one from the fork to each branch, so H3 did not show;
- 2,000 steps: mass $2.7\times10^{-14}$, the full river $2.1\times10^{-16}$ of the peak, threaded equal to serial, continuity $2.2\times10^{-14}$ (registered at $10^{-12}$ before its first run);
- the wider branch took 53.6% of the water and 57.3% of the pollutant;
- the one-window control is bit for bit (`tests/test_workbench_fork.py`).

Records: `out/workbench/records/demo-fork/`.

---

## 3. Item 1.3 — holes that cross the domain's edge, or reach past it

### 3.1 What the owner asked

> *"Holes should be allowed to intersect the boundary's edge. Also, they should also be allowed to extend beyond the geometry's edge, as long as the geometry itself is a closed shape."*

### 3.2 What refuses it now

Two rules in `spec._check_drawn` (`spec.py:901–936`):
- *"hole {h} is not inside the domain's outline"*: every point of the hole's ring must lie inside the outline;
- *"hole {h} reaches past the grid"*: a hole is held to the grid like the outline.

And one in `layout.coordinates` (`layout.py:241–243`): *"cutting across needs a domain without holes"*.

**Reproduced on 2026-09-30 in the served page, exactly as written.** I cut a circle hole across the end of a drawn river's branch and on past the grid's right edge: centre $(344, 184)$ cells, radius 22. It made two errors, neither with a Fix:
- *"hole 0 is not inside the domain's outline"*;
- *"hole 0 reaches past the 352 x 240 grid; draw it inside, or make the grid bigger (Model, Physics)"*.

Headless, each rule fires on its own:
- a hole past the grid, on the whole grid, gives the second error alone;
- a hole across an inset outline, inside the grid, gives the first alone.

The layout's rule did not show: the page cuts one piece across by default, so `across` is 1.

### 3.3 Why lifting them is mostly safe

**The domain is already a set of cells.** `geo.domain_mask` takes the cells whose centres lie inside the outline and outside every hole. A hole that crosses the outline simply removes cells, and a hole past the grid removes cells that were never there. **The solvers never see the drawing, only the mask.** So every family's arithmetic is unchanged. What changes is the bookkeeping that refers back to the drawing.

### 3.4 What must change

| # | what | the change |
|---|---|---|
| 1 | the two refusals | removed. A hole is any closed, non-self-intersecting shape, anywhere |
| 2 | **boundary faces near a crossing** | `geometry.void_faces` gives each face to the void the condition of the **nearest drawn edge**. Near a crossing, the nearest edge can be the part of the outline that the hole removed. **Fix:** the candidates are only the parts of edges that are still boundary: the outline's samples outside every hole, and a hole's samples inside the outline. It is the same nearest-edge rule on a filtered set, so a plain case's arithmetic does not move |
| 3 | **an edge the hole swallows whole** | a boundary condition on an edge that no longer bounds anything is a problem, *"the inlet is on an edge the hole removed"*, with a Fix that moves it to the nearest surviving edge |
| 4 | **the layout's "no holes" rule** | read topology from the mask, not from the drawing. A **topological hole** is a connected component of the void that touches neither the grid's border nor the outline's exterior. A drawn hole that crosses the outline is a notch, not a hole, and cutting across is allowed |
| 5 | **a hole that cuts the domain in two** | refused, with the reason: *"the hole cuts the domain into 2 separate pieces; a case is one connected domain"*. The layout already refuses a disconnected domain; the check now says so where the hole is drawn |
| 6 | **the canvas** | the outline and the holes are drawn as they are, the void is hatched over both, and the solver's staircase is drawn under them, as now. No polygon clipping is needed to draw it |
| 7 | holes in **windows** | the same rules, since windows may have holes too |

### 3.5 Checks

- **N = 1 control** on a domain with a crossing hole, for every family: one window equals the full domain, bit for bit.
- A plate whose hole crosses the right edge, clamped on the left and loaded on what remains of the right edge: the supports balance the load to $10^{-6}$.
- A conduction case where a hole cuts off the right edge's middle: the energy balance to $10^{-6}$, and the heat through each surviving edge piece reported.
- The existing 242 workbench tests pass unchanged: a case whose holes sit inside the outline gives the same faces, conditions and windows as before.

### 3.6 Done when

In the served page, drag a circle hole across the outline's edge and on past the grid's. The case stays ready, with a problem only if an edge carrying a condition was swallowed, and its Fix clears it. Run, compile and the checks all behave.

**Built on 2026-09-30 (the demo chat), as §3.4 designed.** The pieces:
- the two refusals are gone;
- `geometry.live_edges` is the filtered nearest-edge set, and it hands back `drawn_edges`' own arrays wherever nothing is cut;
- a condition left on a swallowed edge is a problem with a Fix, *Move it to the nearest edge left*, which is never made unasked;
- a hole that cuts the domain in two is refused;
- `layout.topological_holes` reads holes from the mask;
- the heat through each piece of a cut edge goes into the conduction record.

**Walked in the served page at $1090\times620$, with real pointer events.** A circle hole drawn inside a heat-conduction outline was dragged by its diamond to $(344, 120)$: across the outline's edge at 336 cells and past the grid's at 352.
- The case stayed *Ready to run*.
- The run closed energy to $4.5\times10^{-12}$ and agreed with the full domain to $8.4\times10^{-11}$ (tolerance $10^{-6}$ each).
- The cold edge's two pieces each carried $-1444.13$ W/m.
- The compile read `refuse` at R10, the W348 refusal of §5.

A river whose whole inlet end a hole took showed *"the river-inlet B4 is on outline:3, an edge the hole removed"*. Its Fix moved the inlet onto the hole's arc. That raised the flow's speed near it, so the next problem was the time step, which has its own Fix. The river then ran every check.

**Checks (§3.5), in `tests/test_workbench_holes.py`:**
- the one-window control on a crossing hole is bit for bit for the river, the plate, the channel and the farm;
- the split and the sound's pieces match their references to the bit;
- the circuit and the cooled block pass their own checks;
- the plate balances to $10^{-6}$;
- the wall reports both pieces of its cut edge.

271 workbench tests pass. One old test changed, because it pinned the refusal the owner lifted.

---

## 4. Hosting: who can actually interact with it

**This decides whether the demo's purpose can be met, and it belongs to the owner (O2).** The workbench is a Panel server on the owner's laptop. A stranger cannot reach it.

| option | what the visitor gets | speed claims | cost | notes |
|---|---|---|---|---|
| **A. recorded runs on the website** | a replay of real records: the fields, the timings, the verdicts, animated from the JSON records | exact, with the machine's state | a static host, free | no interaction beyond scrubbing and choosing a case |
| **B. one-command local install** | the full workbench on their own machine | measured on their machine, labelled so | free | the RaceLab bundle is the precedent: a fresh clone installed, self-tested and served on Windows 11 and Linux ([[poc3-racelab-bundle]]) |
| **C. a hosted instance** | the full workbench in a browser tab | **not comparable**: a small cloud machine has few cores, and threads are the wind farm's whole speed-up ([[decomposition-speed-by-rotor-count]]) | a free tier with few CPUs, or a paid machine by the hour | one run at a time per server, by design (`runner.py`), so it does not serve a crowd |
| **D. in the browser, compiled to WebAssembly** (`panel convert`) | the workbench with no server | none: WebAssembly in a browser has no threads for the parallel arm | free | numpy and scipy run under Pyodide; the threaded arm and the timings do not |

The first draft recommended A + B.

**Decided (O2, the owner, 2026-09-30): B, a one-command local install that installs every package it needs and works on Windows and on a Mac**, reusing the proofs of concept's bundle code where it helps. The website links to it from every *Try the demo* ([[website-outline]] §3). It shows statistics and real screenshots, not replays.

### 4.1 The installer

**The precedent, to reuse, not rewrite.** PoC 3's bundle ([[poc3-racelab-bundle]]) installs, self-tests and serves on a fresh clone. It had four rounds of verification, and each of the first three passed every check it had while hiding a defect a later check found. Its parts:
- `scripts/build_racelab_bundle.py`, the builder, with `w146_build_frontwing_bundle.py` and `w112_build_bundle.py` before it;
- the launcher templates in `atlas/demo_racelab/bundle/`: `run.sh`, `run.cmd`, `run.py`, `requirements.txt`, `constraints.txt`, `conftest.py`, `gitattributes`;
- `scripts/verify_racelab_bundle.py` and `scripts/verify_racelab_offline.py`, which check a fresh clone.

Its lessons carry over directly:

| lesson (PoC 3) | what the workbench installer does |
|---|---|
| the launchers **set** the paths, never default them; the self-test asserts where each solver was loaded from (a pre-set variable would silently test another copy) | the same: the build repository's solvers the workbench needs (the wind farm's `WindowNS` through `atlas/cases/window_ns.py`, and whatever else a family imports) are **vendored** under `vendor/`, and `run.py` points at them |
| top-level packages in `requirements.txt`, **everything they pull in** pinned in `constraints.txt`, generated from verified installs | the same, from a verified Windows install and a verified macOS install |
| macOS ships bash 3.2: no array expansion under `set -u` | the same `run.sh` discipline |
| **torch on macOS means Apple silicon** (the pinned wheel had no `x86_64` build) | torch is needed only by the learned case (§1.5), so it is an **optional** step: if it fails to install, the other seven types and the classical arms still run, and the page says why the learned case is unavailable |
| a WebSocket library must be pinned, or the page never receives a frame | Panel serves through Bokeh's Tornado server, which has its own WebSocket; **the self-test opens the served page anyway**, since "nothing short of opening the page could have found it" |
| `run.sh` must be `100755` in the git index, and line endings are set by `.gitattributes` | the same |

**What it installs:** Python packages in a fresh virtual environment (`.venv`), pinned: Panel 1.5, Bokeh 3.6, numpy, scipy, pydantic and whatever the workbench imports, torch CPU-only for the learned case, and **Gmsh from PyPI, optional**. Gmsh is GPL: it is installed on the visitor's machine from PyPI, not redistributed, and only the *Import geometry from Gmsh* item needs it. The installer creates no system-wide state.

**One command per system:**
- Windows: `run.cmd` in a fresh clone;
- macOS: `./run.sh`.

Each creates the environment, installs, self-tests, then serves `python -m atlas.workbench --open`. A second run skips straight to serving.

**What must not be in it:**
- NeuberNet, in any form: the builder refuses, as PoC 3's did;
- Poseidon's weights: the learned case uses the owner's own weights (§1.5);
- anything under `out/` except the records the tests read.

**How it ships:** a branch or a GitHub release built from a commit, like PoC 3's orphan branch, with the builder never editing a source file. It stays private until the website launches (O9).

**Verification** on a **fresh clone**, on Windows (the owner's laptop) and on **macOS**:
- the self-test passes;
- the page opens and a run finishes;
- the classical arms still run with torch absent.

**The owner may have no Mac.** In that case the demo chat asks before using a GitHub Actions `macos-latest` runner: it uses the account's minutes, and macOS minutes count at a multiple on a private repository.

**Done, 2026-10-01 (the demo chat's step 7): `8d14565`, then `f008c51` and `6232415`.** Three commits, because the branch is built from a commit and each round of verification on a fresh clone found something that changed what ships.
- **What a stranger runs.** `.\run.cmd` on Windows or `./run.sh` on macOS and Linux, in a clone of the branch `atlas-workbench`. `scripts/build_workbench_bundle.py` makes that branch from a commit, in `out/workbench-bundle`, and never pushes it (O9). The launcher:
  - finds Python 3.10, 3.11 or 3.12 and builds `.venv`;
  - installs the pins, then two optional steps that may fail without stopping anything: torch, CPU-only, and Gmsh from PyPI;
  - runs the self-test once, then serves the page on the first free port from 8020 and opens a browser. A second run goes straight to the page.
- **What the branch carries:** 135 files, 4.3 MB.
  - `atlas/`, without the three proofs of concept's demo packages, which nothing else imports;
  - the one build-repository package the workbench loads, the wind farm's `WindowNS`, at `vendor/src/atlas/cases/windfarm/`, with that repository's MIT licence;
  - every workbench test, and the three scripts they import, found by reading their imports.

  The builder refuses NeuberNet (no file and no use; its name appears in prose 4 times), Poseidon's weights and any binary file (none), a vendored scOT, and `out/`.
- **The self-test** (`run.py --check`, 27–34 s on AC here):
  - every package imports;
  - the solver is loaded from `vendor/`, asserted by its path;
  - each of the eight types compiles through the page's own compile and marches every arm through the page's own runner, finite inside the domain, with every exact control the type registers holding;
  - the page is served on a free port and fetched with its 25 scripts and stylesheets, none from the network. Its document is pulled over the WebSocket: 281 models, with the *Fast example* and *Run* buttons.
- **What verification found, in order:**
  1. **A split by physics differs by design.** The runner's generic "threaded equals serial" flag is false for the heated structure: its serial and threaded arms are a synchronous and a lagged split. The self-test asks each type for its own exact controls instead.
  2. **The first clean install served HTTP 500 on every page load, while all eight types ran.** It resolved param 2.4.2. From param 2.3.0, with Panel 1.5.2, a widget's `name` is constant, and the header relabels its buttons by setting it. Measured: 2.1.1, 2.2.0 and 2.2.1 allow it; 2.3.0, 2.4.0 and 2.4.2 raise.
     - So `constraints.txt` pins every package that install pulled in at the development environment's version, where the whole suite passed.
     - `scripts/verify_workbench_pins.py` resolves those pins with pip's own resolver, binary wheels only, for Windows, Apple-silicon and Intel macOS, and Linux, on Python 3.10, 3.11 and 3.12. Each resolves to 42 packages with nothing unpinned; torch adds 10 (none on Intel macOS) and Gmsh 1.
  3. **Help named wiki pages the install does not carry.** It now names them only where they are.
  4. **Behind a proxy, the self-test called a working page broken.** With pip pointed at a closed proxy, the optional installs failed as designed. But the self-test's fetch of `http://127.0.0.1:…` went to the proxy too, because urllib honours `HTTP_PROXY` for loopback unless `NO_PROXY` names it, so the launcher would have refused to serve. The page is now fetched with no proxy (`f008c51`, pinned by a test).
  5. **On Linux, Gmsh installs and will not load.** On Ubuntu without a desktop its import raises `OSError: libGLU.so.1: cannot open shared object file`. `pytest.importorskip` skips only an `ImportError`, so four Gmsh tests failed instead of skipping; and `gmsh_import.read_msh` caught only `ImportError`, so the import dialog would have met a raw `OSError`. It now says Gmsh is installed but will not load, and why; a test simulates the failure, and the tests skip with the reason (`6232415`). The branch's README names `libglu1-mesa`.
- **Verified on a fresh clone of the build from `6232415`, Windows 11, Python 3.12.7, on AC** (`out/workbench/records/installer/verify_windows-amd64_release3.json`; the earlier builds' records are beside it):
  - **Static checks:** `run.sh` `100755` and LF, `run.cmd` CRLF; all 134 copied files identical by blob hash to `6232415` and the build repository's `98df350`.
  - **The launcher:** exit 0 in 227 s.
  - **The bundle's tests:** 319 passed, 2 skipped. One skip is the `fe.py` cross-check against the unvendored ThermoStruct2D; the other is the installer's own test, whose templates live here.
  - **Without torch:** exit 3, all eight types ran.
  - **With no network:** exit 0, and no connection left the machine.
  - **With both optional installs failing behind a closed proxy:** exit 3; both failures said so, no stamp was written, and all eight types ran.
  - **Size:** `.venv` is 2.0 GB with torch and Gmsh, 613 MB once both are removed. Its deepest path is 141 characters, under Windows' 260.
- **Verified on a fresh clone of the same build on Linux**, Ubuntu under WSL 2 on the same laptop, Python 3.10.21 (`verify_linux-x86_64_release3.json`):
  - **Static checks:** the same; all 134 copied files identical by blob hash.
  - **The launcher:** `./run.sh` built `.venv`, installed everything, and its self-test passed every type and the page, exiting 3 in 54 s. The 3 is because Gmsh installed and will not load (`libGLU.so.1`).
  - **The bundle's tests:** 315 passed, 6 skipped: the two above and the four Gmsh tests, each with its reason. Before `6232415` those four failed (`verify_linux-x86_64_release.json`).
  - **Without torch:** exit 3. **With no network:** exit 3, and no connection left the machine. **With both optional installs failing behind a closed proxy:** exit 3, each failure said so, and all eight types ran.
  - **Size:** `.venv` is 1.4 GB.
- **In the served page** (the clone's own `run.cmd` at 1090×620, on the `8d14565` build; the `f008c51` build was served the same way and its page loaded; the later commits changed no page code but the Gmsh dialog's error):
  - the header fits: no horizontal overflow, and its last control ends at 1081 px;
  - *Sound*, then *Fast example*, then *Run*: 130 macro-steps in 32 s. The card read 5.55× for parallel pieces in cache, with both bars met; the field *decomposed minus full* was 0 (on AC; the run's record is `walk-windows-fast-sound-20261001-041539.json`);
  - *Help* said this copy does not carry the wiki.
- **Not verified on macOS**: there is no Mac (the owner's answer). What stands in for it:
  - `run.sh` follows bash 3.2's rules, pinned by a test, and `bash -n` parses it;
  - the resolver installs every pin for both kinds of Mac;
  - `run.sh` ran end to end on Linux (above).
- **Open:**
  - Bokeh's client logs `reference … isn't known` while the page rebuilds after a type switch, an example or a run. It does so in the development page as much as in the installed one, and the page still draws everything.
  - Whether calling Gmsh's API from distributed code needs more than installing Gmsh from PyPI is the owner's to settle before publishing (O9).
  - Atlas has no licence yet.
  - ~~The learned case's weights will be the first binary file the builder allows (step 8).~~ Done: see *Rebuilt for step 8* below.

**Rebuilt for step 8 (2026-10-01), from `f8bbc9b`** (`build-f8bbc9b.json`):
- **What changed:** the branch carries the learned case, 146 files and 8.4 MB. It holds the trained weights and the stored truth, the only binary files the builder allows, each by exact path; the registration, the training log and the gate's record; and nothing else of `out/`.
- **The self-test's new step:** the weights load and step `farm-12` once, finite and divergence-free, and the registered verdict is printed. Without torch the learned case is reported absent, which is optional.
- **Verified on a fresh clone on Windows**, on AC with no other Python process (`verify_windows-amd64_release4.json`):
  - **Static checks:** all 145 copied files identical by blob hash, the six of `out/` among them; no unexpected binary file.
  - **The launcher:** exit 0 in 229 s. The learned step read divergence $3.8\times10^{-18}$, and the verdict printed was *"not every bar met (G5)"*.
  - **The bundle's tests:** 333 passed and 2 skipped (the same two skips as before).
  - **Without torch:** exit 3, with torch reported absent. **With no network:** exit 0, and no connection left the machine. **With both optional installs failing:** exit 3, each failure said so.
  - **Size:** `.venv` is 2.0 GB with torch and Gmsh.
- **Verified on a fresh clone on Linux**, Ubuntu under WSL 2, Python 3.10.21 (`verify_linux-x86_64_release4.json`):
  - **Static checks:** the same; all 145 copied files identical by blob hash.
  - **The launcher:** exit 3 in 55 s, because Gmsh installs and will not load (`libGLU.so.1`), as in step 7. The learned step read divergence $3.8\times10^{-18}$, with the same verdict printed.
  - **The bundle's tests:** 329 passed and 6 skipped: the two above, and Gmsh's four.
  - **Without torch, with no network, and with both optional installs failing:** exit 3 each. No connection left the machine, and each failure said so.
  - **Size:** `.venv` is 1.4 GB.
- **Not on macOS**, as before.
- **The final build, from `eccf0a2`** (§5's T3 citation): it differs from the build verified above only in the compiler's message strings, one README line and `SOURCE_COMMITS` (a `git diff` of the two branch commits). A fresh clone's static checks pass: all 145 copied files identical by blob hash (`verify_windows-amd64_release5-static.json`). The launcher and the tests were not run again on it.

---

## 5. W348 is a demo problem now

Every style-B and style-C example compiles to **`refuse`** at `L2/R10`, though each seam is admitted and each agrees with the full domain to $3.6\times10^{-9}$ or better ([[showcase-gallery]] §4; [[gap-worklist]] W348). In the owner's own library that is an honest note. **On a fast example shown to a stranger, a red "refuse" beside a correct, faster answer reads as a failure.**

Several fast examples in [[demo-fast-examples-plan]] are steady elliptic problems solved by substructuring, which is exactly R10's refused class. **So W348 should be settled before those examples are recorded.** The mathematical fact it needs, that the fixed point of an iterated or directly-solved linear coupling is the undivided discrete solution, is theorem **T3** in [[formal-proofs-plan]], and it is one of the cheapest there to machine-check. The recommended order is to prove T3, then change R10 by W348's option (a), then record the fast examples.

**Revised 2026-09-30, when the chats were planned to run at the same time** ([[00-proposal-workstreams]] §4): **the demo chat owns the R10 change**, because its fast examples need it and the compiler is code it already tests. It cites T3 by its classical source, the fixed-point consistency of restricted additive Schwarz (Frommer & Szyld, *SIAM J. Numer. Anal.* 39, 2001), and switches to the Lean theorem's name when the proofs chat has checked it.

**Done, 2026-09-30 (the demo chat's step 5): option (a).**
- **A piece now declares where its embedded solve's boundary data comes from** (`ExpertCapabilities.elliptic_data_from_ports`, false by default and written to a record only when true). The workbench's style-B and style-C pieces declare it: each is re-solved with its neighbours' values on its cut until the interface agrees. Tier 0's windows could not have: their pressure Poisson problem took homogeneous Neumann data on the cut.
- **R10 decides a cut piece that declares it after the scheme** (`compiler._r10_scheme`), as R13 is decided:
  - a `direct-schur` scheme decertifies;
  - a sweeping accelerator with a stated $\varepsilon_{\text{tol}}$ decertifies, citing Frommer & Szyld (2001);
  - a scheme with neither refuses, with R10's own sentence.

  The decertification names W168's cost, $149\times$ in coupling sweeps, as a cost and not an error. [[formal-proofs-plan]] shows no Lean written yet, so the classical source is cited.
- **The negative controls, measured** (`scripts/w348_controls.py`, `out/workbench/records/w348/`):
  - Tier 0's four windows as built, and CS-S1 with the pressure solve embedded, still refuse at `L2/R10`; neither declares the flag;
  - `wall-2` with the declaration withdrawn is refused at L2 as before (a test).
- **The census** (`scripts/w348_census.py`, a pytest plugin over the full suite): 307 compiles before, 311 after (the 4 more are the new test's).
  - **Only the intended rows moved**: the workbench's 9 compiles of its iterated conduction and elasticity graphs, each from `L2/R10` refuse to `L5/R10` decertify, and nothing else.
  - Four front-wing compiles were credited to different tests, because a compile shared within a process belongs to the test that runs it first and the new test file regrouped the suite. Paired by graph, their verdicts and decisions are identical (`out/workbench/records/w348/census/`).
  - The suite after the change: 2262 passed, 0 failed (on battery, 2703 s).
- **The W189 byte control** (`scripts/w348_byte_control.py`): HEAD's compiler against the working tree's, with the same hash seed; all 40 artifacts are identical.
- **The seven examples that refused now compile `admit-uncertified`**, with R10 at L5: the wall, the insert, the bracket, the bend, the round insert, the S-channel and the plate with a hole.

**T3's Lean names, 2026-10-01** (`eccf0a2`). The proofs chat machine-checked T3, and found its restricted-additive-Schwarz part false as this plan stated it: not every fixed point of a Schwarz sweep solves the undivided system ([[formal-proofs-record]] §5.1). So the decertification now says what the checked theorems say:
- a direct Schur solve: `Atlas.schur_iff`;
- Dirichlet–Neumann: `Atlas.TwoPieces.dn_fixedPoint_solves`;
- restricted additive Schwarz: a sweep that converges from every start has one fixed point, the solution (`Atlas.Schwarz.fixedPt_eq_solution`). The counterexample is named, and convergence is Frommer & Szyld's for M-matrices and measured for elasticity;
- the truncation: at most the tolerance over $1-\rho$, since the driver stops on the update (`Atlas.Schwarz.error_le_update`).

The change is text only: HEAD's and the new `compiler.py` have identical syntax trees once string constants are blanked. The W348 tests pin the new citations.

---

## 6. Build order for the next chat

1. **Reproduce** the owner's fork scenario and a crossing hole in the served page. Record what actually fails, and correct §2.2 and §3.2 if the hypotheses were wrong.
2. Item 1.1 (the header): small, and it changes what every later screenshot looks like.
3. Item 1.3 (holes): lift the two rules, filter the edge candidates, then the checks.
4. Item 1.2 (the fork): the flow as the along coordinate, components as windows, the dead-branch warning, `river-fork`.
5. The workbench suite (`python -m pytest tests -k workbench -q`, about 7 minutes on battery), then the full suite (`python scripts/run_suite.py`), checking every log's mtime is after the start.
6. The newcomer's first scenario in the served page at $1090\times620$, for all eight types.
7. The README, [[showcase-gallery]] for `river-fork`, [[index]] and [[log]].

8. The installer (§4.1), verified on a fresh clone on Windows and on macOS.

**O1 and O2 are decided** (§1.3, §4). The chat's own prompt is [[proposal-chat-prompts]] §1.

### 6.1 The demo chat's eight steps (2026-09-30)

The chat's prompt turned the order above into eight steps, each one commit, with a report to the owner after each.

| step | what it does | done when | state |
|---|---|---|---|
| 1. Reproduce | the forked river and the crossing hole, in the served page; §2.2 and §3.2 corrected | both recorded | done, `0588c3c` |
| 2. The header (item 1.1) | a simulation-type selector replaces the case's name, and a new type starts a new case that one Undo restores; name and description leave the case file (`case@0.6`, with a migration); *Case > Start over*; the Examples dialog leaves the page | §1.5 | done, `31ee474` |
| 3. Holes (item 1.3) | holes may cross the outline and reach past the grid | §3.6 | done, `5c9f8b3` |
| 4. The fork (item 1.2) | the flow as the along coordinate, each connected band a window, spectral bisection for other branched shapes, the dead-branch warning and its Fix, the example `river-fork` | §2.5 | done, `2dec7e9` |
| 5. W348 | R10's case for a cut embedded family moves after `_l5_l7_scheme`. Where the coupling supplies the elliptic solve's boundary data and is solved directly (`direct-schur`) or iterated to a stated tolerance, the graph is decertified with W168's cost instead of refused. It cites Frommer & Szyld (2001), and T3's Lean name once [[formal-proofs-plan]] shows it checked | Tier 0's four windows and CS-S1 still refuse, measured; the refusal set of every graph the suite compiles, diffed before and after, moves only in the intended rows; the W189 byte control holds; the pinned verdicts are updated | done, `f4c7b5b` (§5); T3's Lean names cited once checked, `eccf0a2` |
| 6. Fast examples (item 1.4) | micro-benchmarks first: does P (parallel pieces) scale on threads, does scipy's `splu` release the GIL, and where does each family's core leave cache. Then the bars are registered in each family module before the first timed run; a sweep; a confirmation run from a fresh start; the header's *Fast example* button, with *More examples* under it (O1); each card's mechanism line and O3's fixed line; a [[showcase-gallery]] section | [[demo-fast-examples-plan]]'s gates | done: four of eight meet both bars ([[showcase-gallery]] §8) |
| 7. The installer (O2) | `run.cmd` (Windows) and `run.sh` (macOS): a `.venv`; pinned requirements and constraints; torch CPU-only and optional; Gmsh optional, from PyPI; the build repository's solvers vendored, with `run.py` setting their paths; a self-test that asserts where each solver was loaded from and opens the served page. The builder refuses NeuberNet and Poseidon's weights | §4.1, on a fresh clone on Windows | done, `8d14565`, `f008c51` and `6232415`: verified on fresh clones on Windows and Linux, and in the served page; not on macOS (§4.1) |
| 8. The learned case (item 1.5) | a random network's step is timed first (step 0); G1–G6 and the held-out split are registered before any data exists. Then the data is generated, a GPU rented, the network trained and the GPU destroyed, each announced with its expected time or cost (O6). It is evaluated once, built into a read-only case, and its weights are added to the installer | [[demo-learned-case-plan]] §5's gate, reported whatever it shows | done, `a7a300a`, `60dd865` and `f8bbc9b` ([[demo-learned-case-plan]] §9). G1–G4 and G6 pass. **G5 fails**: a classical solver on a grid twice as coarse is closer to the truth. The learned arm matches the classical decomposition's accuracy, not the full domain's. It is a menu item, not a header button. The installer carries its weights (§4.1) |
| End | [[gap-worklist]] W348–W351, the plan pages' status lines, [[showcase-gallery]], the workbench README, [[index]] and [[log]]; a short report per step | | done, 2026-10-01 |

**The owner's answers, 2026-09-30.** The other two chats are not idle, but they are writing documentation and theory, with nothing running on the demo. Every timed record says so. No Mac is available, so `run.sh` is the chat's best guess. It follows the RaceLab bundle's macOS discipline (bash 3.2, `100755` in the index) and is **not verified on macOS**.

---

## See Also

- [[00-proposal-workstreams]] — the four workstreams and their order
- [[demo-fast-examples-plan]] — item 1.4, the one-click fast example per type
- [[demo-learned-case-plan]] — item 1.5, the one learned case
- [[showcase-library-plan]] — the plan this one continues
- [[showcase-gallery]] — what every example measured
- [[gap-worklist]] — W348, and this plan's rows
