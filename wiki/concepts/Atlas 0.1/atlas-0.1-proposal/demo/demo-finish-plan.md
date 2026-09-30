# Finishing the demo workbench — the header, the branching river, and holes that cross the edge

**Type:** Concept page — **plan, with the decisions made and the ones left to the owner** (folder: `Atlas 0.1/atlas-0.1-proposal/demo/`)
**Status:** written 2026-09-30, before any code. Read from `atlas/workbench/` at commit `fc9a182` (`spec.py`, `layout.py`, `flow.py`, `families/plume.py`, `app.py`, `registry.py`, `starter.py`) and its README. **Nothing here has been run**: the planning session's container had no numpy, and installs need the owner's approval. Every diagnosis below is read from the code and marked as a hypothesis until the next chat reproduces it in the served page.
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

**Owner decision needed (O1):** D6 removes 19 hand-built examples from the visitor's reach, among them the drawn ones (`bend-3`, `s-channel`, `ring-film` and the rest), which are the best evidence that irregular geometry works. **The recommendation is to keep them reachable through Fast example's neighbours:** a small *More examples* list under the Fast example button, one line per example that says what it shows, with no name. For example, *"Heat round a bend on three curved windows"*. Removing them entirely is one line of code either way.

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

---

## 3. Item 1.3 — holes that cross the domain's edge, or reach past it

### 3.1 What the owner asked

> *"Holes should be allowed to intersect the boundary's edge. Also, they should also be allowed to extend beyond the geometry's edge, as long as the geometry itself is a closed shape."*

### 3.2 What refuses it now

Two rules in `spec._check_drawn` (`spec.py:901–936`):
- *"hole {h} is not inside the domain's outline"*: every point of the hole's ring must lie inside the outline;
- *"hole {h} reaches past the grid"*: a hole is held to the grid like the outline.

And one in `layout.coordinates` (`layout.py:241–243`): *"cutting across needs a domain without holes"*.

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

---

## 4. Hosting: who can actually interact with it

**This decides whether the demo's purpose can be met, and it belongs to the owner (O2).** The workbench is a Panel server on the owner's laptop. A stranger cannot reach it.

| option | what the visitor gets | speed claims | cost | notes |
|---|---|---|---|---|
| **A. recorded runs on the website** | a replay of real records: the fields, the timings, the verdicts, animated from the JSON records | exact, with the machine's state | a static host, free | no interaction beyond scrubbing and choosing a case |
| **B. one-command local install** | the full workbench on their own machine | measured on their machine, labelled so | free | the RaceLab bundle is the precedent: a fresh clone installed, self-tested and served on Windows 11 and Linux ([[poc3-racelab-bundle]]) |
| **C. a hosted instance** | the full workbench in a browser tab | **not comparable**: a small cloud machine has few cores, and threads are the wind farm's whole speed-up ([[decomposition-speed-by-rotor-count]]) | a free tier with few CPUs, or a paid machine by the hour | one run at a time per server, by design (`runner.py`), so it does not serve a crowd |
| **D. in the browser, compiled to WebAssembly** (`panel convert`) | the workbench with no server | none: WebAssembly in a browser has no threads for the parallel arm | free | numpy and scipy run under Pyodide; the threaded arm and the timings do not |

**Recommendation: A + B, with C as an optional extra.** The website carries recorded runs, every number traceable to its record. *Try it* offers the local install. A hosted instance, if the owner wants one, is labelled "for trying the interface: speeds here are not the demo's". **[AI Inference]:** a free hosted tier's core count would turn the wind farm's $4.8\times$ into something much smaller, and a visitor who saw that would reasonably distrust the rest.

---

## 5. W348 is a demo problem now

Every style-B and style-C example compiles to **`refuse`** at `L2/R10`, though each seam is admitted and each agrees with the full domain to $3.6\times10^{-9}$ or better ([[showcase-gallery]] §4; [[gap-worklist]] W348). In the owner's own library that is an honest note. **On a fast example shown to a stranger, a red "refuse" beside a correct, faster answer reads as a failure.**

Several fast examples in [[demo-fast-examples-plan]] are steady elliptic problems solved by substructuring, which is exactly R10's refused class. **So W348 should be settled before those examples are recorded.** The mathematical fact it needs, that the fixed point of an iterated or directly-solved linear coupling is the undivided discrete solution, is theorem **T3** in [[formal-proofs-plan]], and it is one of the cheapest there to machine-check. The recommended order is to prove T3, then change R10 by W348's option (a), then record the fast examples.

---

## 6. Build order for the next chat

1. **Reproduce** the owner's fork scenario and a crossing hole in the served page. Record what actually fails, and correct §2.2 and §3.2 if the hypotheses were wrong.
2. Item 1.1 (the header): small, and it changes what every later screenshot looks like.
3. Item 1.3 (holes): lift the two rules, filter the edge candidates, then the checks.
4. Item 1.2 (the fork): the flow as the along coordinate, components as windows, the dead-branch warning, `river-fork`.
5. The workbench suite (`python -m pytest tests -k workbench -q`, about 7 minutes on battery), then the full suite (`python scripts/run_suite.py`), checking every log's mtime is after the start.
6. The newcomer's first scenario in the served page at $1090\times620$, for all eight types.
7. The README, [[showcase-gallery]] for `river-fork`, [[index]] and [[log]].

**What the next chat asks the owner first:** O1 (keep *More examples* or not) and O2 (hosting).

---

## See Also

- [[00-proposal-workstreams]] — the four workstreams and their order
- [[demo-fast-examples-plan]] — item 1.4, the one-click fast example per type
- [[demo-learned-case-plan]] — item 1.5, the one learned case
- [[showcase-library-plan]] — the plan this one continues
- [[showcase-gallery]] — what every example measured
- [[gap-worklist]] — W348, and this plan's rows
