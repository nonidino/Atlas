# Atlas Workbench

Build a domain decomposition by hand, compile it with the Atlas compiler, run it,
and compare it with the full-domain solve. Built:
- **one screen to model and one to run** (rebuilt on 2026-09-29, planned first on a
  Claude Design canvas), the case file and the **geometry section**;
- **starting values**: a drawn shape or a newly chosen physics is ready to run, with
  every assumption said, and the problems list fixes what has a default in one click
  (2026-09-30);
- **a case is what it simulates** (case file 0.6, 2026-09-30): the header's first
  control chooses the kind of simulation, and another kind starts a new case of it
  (one Undo brings the old one back); a case has no name or description; *More
  examples* lists the chosen kind's examples, one line each saying what it shows;
- the **runner**, with eight physics families, one per showcase case: the wind
  farm (style A), heat conduction in several materials (styles B and C), a
  resistive plate on a circuit (style D), a pollutant plume down a river (style A,
  explicit), sound through two media (style C, explicit), a loaded bracket (style
  B), a heated strip split by physics, and a heated block cooled by a channel
  flow (style C at a seam between two physics);
- the **compile**: each family turns a case into the Atlas compiler's `CaseGraph`,
  and the Run & results tab shows the compiler's verdict per seam;
- the **showcase gallery**: every example is in its kind's *More examples* list, and
  `wiki/concepts/Atlas 0.1/atlas-0.1-outcome/showcase-gallery.md` records what
  each one measured in the served page.

The runner is plan step 3 of
`wiki/concepts/Atlas 0.1/atlas-0.1-outcome/outcome-c4-path-to-declarative-cases.md`,
and the compile is the graph half of its step 2. The library of cases is
`showcase-library-plan.md` in the same folder.

```bash
python -m atlas.workbench --open
```

Then open <http://127.0.0.1:8020/>. `--port` and `--cases-dir` are optional; cases
are saved to `out/workbench/cases/` by default. Add `?case=<file>.json` to the
URL to open a saved case on load (only a file name inside the cases folder is
accepted).

## Built on what was already installed

**Panel 1.5** and **Bokeh 3.6**, both BSD-3-Clause, were already in the Python
environment. No existing tool draws a domain decomposition and compares it with the
full domain, so none was reused whole. What these two libraries supply is the
part of a GUI that is hardest to hand-write well: menus, dialogs, form widgets,
notifications, and a canvas whose Tap events and `PointDrawTool` select, drag
and delete shapes.

**Gmsh 4.15** (GPL-2+, already installed) is the geometry path for anything that
is not a rectangle on the grid. The workbench *calls* its Python API to read a
`.msh` file; nothing of Gmsh is copied into this code. Its linking exception does
not cover this code, which matters only if the workbench is ever distributed.

**Nothing is downloaded, and the page makes no outside request.** The page uses
Panel's **Bootstrap** template. The Fast template was tried first and dropped for two
reasons. Its web components follow the operating system's dark mode on their own
while the page around them stays light (dark, magenta-bordered inputs and invisible
sidebar text on a dark-mode PC). And its design loads a font from Google. A test pins
both properties of the replacement.

## Layout

The owner, 2026-09-29: "go through the entire UI to make it much more simple and
intuitive. This is way too complicated of a UI currently. plan it out in claude
design first, and then implement." Seven screens were drawn on a Claude Design
canvas first. The six-step workflow, its menus and tables became two tabs, one
header, and one rule: **pick a layer, click on the canvas, edit what is selected
in the inspector.**

| part | what it holds |
|---|---|
| **header** | **what it simulates** (a selector of the eight kinds: another kind starts a new case of it) and whether it is saved; the two tabs; **More examples** (the chosen kind's examples, one line each saying what it shows); the **status chip**; **Run**; **Case** (*Start over*, open, save, save as, revert, import and export JSON, import geometry from Gmsh, this session's history); **?** (how it works, where things are documented, about). Undo and Redo sit in the canvas's toolbar, so the header fits a laptop |
| **status chip** | *Ready to run*, *Ready · N warnings*, *N problems* or *Marching k of n*. Click it for every problem in plain words, each with a **Show me** button that opens the layer where it is fixed (`app.layer_for`) and, where the workbench has a default for it, a **Fix** button (below) |
| **Model** tab | a **layer rail** on the left, the **canvas** with its tools in the middle, the **inspector** on the right. It takes the window's height, and the canvas keeps its cells square at any size |
| **Run & results** tab | Run and Stop with the settings folded away, the Atlas compiler's verdict, and once a run ends four **cards** (speed, agreement, balance, checks) over the details and the fields |

**The layers** are the ones the family reads: Shape, Materials, Boundaries,
Windows, Physics, and Rotors or Circuit for the families that have them. Each is
labelled with its state (*the grid* or *drawn*, the count of materials, windows or
rotors, *auto* when the windows follow the shape). Below the layers the rail lists
the active layer's items; a click selects one.

**The tools**, one set on every drawing layer:
- **Select**: a click selects what is under it on the active layer; the selected
  shape gets its handles (below);
- **Draw**: click to place points, double-click (or *Finish*) to close; edges start
  straight or smooth;
- **Rectangle**, **Circle**: two clicks each (two corners; the centre, then a point
  on the circle);
- **Pan**.

A tool adds one shape, then hands back to Select with the new shape selected. The
hint under the canvas says what a click does now.

**The inspector** edits what is selected: a shape's edges (straight, arc with its
bulge, or smooth), a region's material and stacking, a material's properties (or a
new one, from the library or by name), a window's name and place, a boundary's
condition and span, a rotor's position and diameter, a circuit part's value and
nodes. With nothing selected it holds the layer's settings: how the windows are cut
(*Automatic* or *My own*) and joined, and the physics, the time and the grid. An
edit the case refuses is not applied, and every field goes back to the case.

**Run & results.** Run marches the arms in turns. The compile card sits under the
controls, because a finished run or compile rebuilds the tab at its top. The cards
are read first: the speed ratio, the agreement with the full domain and the balance
against their tolerances, and how many checks passed. The details (every arm, every
check, the notes, the record's path) are folded under them, and the fields, the
time per step and the convergence below. If the open case is not the one that ran,
or not the one compiled, the tab says so rather than lend it the verdict.

## Starting values and one-click fixes (`starter.py`)

The owner's first scenario, 2026-09-30: the river chosen, a smooth blob drawn as the
river, and four problems. "These four problems don't make sense. I just drew the
geometry and filled in some of the other settings." They were:
- 13,539 cells with no material;
- the outfall outside the river. It stood at the river's default place, (200 m,
  100 m), but the grid had kept the wind farm's cell size, so the river was 11 m ×
  7.5 m;
- no inlet;
- no outlet.

The page also ran past a laptop's screen, so Run needed a scroll.

**A case runs the moment its shape is drawn.** Drawing the domain, or choosing the
physics, gives the case what it lacks (`fill_defaults`), and only that. A repair
runs only for a thing the check finds missing:

| missing | what it gets |
|---|---|
| a material | one region over the whole grid, first in the stack, so any region drawn later lies on top. Its material is a solid one the case already has, or else the first solid in the family's library |
| a river's inlet, outlet | the outline's leftmost edge (by the mean $x$ of its drawn points) and its rightmost |
| the outfall in the water | the domain's cell nearest a point a fifth of the way from the inlet to the outlet |
| a plate's electrodes, circuit | E1 on the leftmost edge and E2 on the rightmost; a 12 V battery (0.5 ohm inside) and a 1 ohm resistor through the plate, the showcase circuit's values |
| a structure's clamp, load | the leftmost edge clamped; the rightmost loaded with the family's value on a rectangle (-5 MPa, $y$) |
| a held temperature | the family's own, as on a rectangle: 400 K on the leftmost edge, and 300 K on the rightmost for conduction (the heated strip has a hot end only) |
| a stable time step | 90% of the explicit limit, rounded down to two figures (the river and the sound) |
| windows | the automatic layout along the shape: one for a field on a circuit or a split, two for Dirichlet-Neumann, two to six for the overlapping styles; the wind farm on the whole grid gets the scaling ladder's 3 × 2 rectangles, the tiling its compile recognises |

Each assumption is listed in one notification and in the log. Each can be changed in
the layer it names, and one Undo takes them all back.

**A new kind of simulation starts a new case, at its example's scale.** The new case
takes the setup of the kind's first example: the cell size, the time step, the steps,
the mode, the coupling and the number of windows (`spec.family_setup`,
`spec.adapt_to_family` on a blank case). So a river is cut in cells of 5 m, not the
farm's 0.03125 D. Until 2026-09-30 a change of physics kept the drawn shape; the
owner's rule since then is "When the simulation type changes, the full geometry
should reset", so nothing of the old case is kept, and one Undo brings it all back.

**The problems list fixes what has a default.** The check marks each problem it has
a repair for (`Issue.fix`). The list shows a button for it beside **Show me**, such
as *Make the rightmost edge the outlet*, and *Fix the N that can be fixed* repairs all
of them in one edit. Each problem says what is wrong in plain words, and where it is
fixed.

**The header's type selector** starts a new case of the chosen kind, ready to run on
the whole grid, with no confirmation: one Undo brings the old case back, with its
file (`starter.new_case_from`, `Workbench.start_new`; Undo restores the case, its
file path and its key). The note says what it was given to start. The cooled block
starts from its example, because its channel must be drawn. **Case > Start over** is
a new case of the same kind.

**A newly drawn domain is never refused for its windows' sake.** If its windows cannot
be cut the way they were (a cooled block drawn without its water cannot be cut one
piece per physics), they are cut again from the new shape, and the note says why. For
Dirichlet-Neumann, the windows fix cuts one piece per material (or the coolant and the
block) when the domain has two media, and along the shape otherwise.

**Two things only the person can decide**, which the problems say plainly:
- a wind farm's air must reach the grid's left and right edges, where the flow
  enters and leaves;
- a cooled block needs its channel.

**The check never raises.** A newly drawn cooled block made the conjugate-heat check
raise, which left the case unusable. Each part of the check now reports its failure
as a problem: "the case could not be checked (...)".

**It fits a laptop.** The Model tab takes the window's height, and a long list in the
rail or the inspector scrolls inside itself. Checked in the served page at 1,090 × 620
CSS pixels on both tabs, and at 1,090 × 520 on the Model tab: nothing runs past the
window, and Run is at the top.

The canvas stretches into the space it is given and widens one of its ranges so
that its cells stay square (`editor.SQUARE_JS`). The hint and the summary under it
have fixed heights: a hint that grew a line as points were placed made the canvas
shrink under the pointer, and eight clicks drew a three-vertex outline.

## The runner

`runner.py` marches a **committed** case: Run copies the case, so an edit made while it marches cannot reach it, and the page names the version that is marching. Every macro-step each ticked arm takes one step, in an order that rotates by one arm per step. The wind farm's arms:

| arm | what runs |
|---|---|
| Decomposed, serial | the windows stepped as one batch per window shape (W346's `E`) |
| Decomposed, parallel | the same windows split across threads, every chunk at the whole tiling's sub-step count, so it is **bit for bit** the serial arm (checked every step; W346's `Ep`) |
| Full domain | the same discretization on the undivided domain (W346's `F`) |

The other families have the same three arms where their style allows them. A style-C case has no parallel arm: Dirichlet–Neumann is sequential by construction, explicit or iterated. The Run & results tab names every arm it drops, and why. A steady case repeats the same solve from the initial state, so its "macro-step" is a **timed repeat**, and the page says so. A family may name its own arms: the heated strip's are the synchronous split, the lagged split (on two threads) and the unsplit solver.

It runs in a background thread; the page reads its progress every 400 ms and draws the fields, their difference, the time per step, the family's series (farm power; outflow; the energy's share in the second medium; the largest stress), and, for an iterated style, the iteration's convergence curve. A field that spans decades (the plume) is drawn on a log scale, for display only. **Stop** (on the Run & results tab, or the header's Run while a run marches) stops after the arm-step in progress and keeps whole macro-steps only. One run at a time per server, because two runs on one machine would time each other.

**What the timer holds:** the family adapter's `step` and nothing else. Diagnostics, the bitwise comparison, the snapshot and the record are taken after it stops.

**The record** goes beside the case file, in `<case>.results/<time>.json`: the case as it marched, every per-step time, the metrics, each check with its tolerance and registration date, each arm's field difference from the full domain (rms, largest, and rms relative to the full field's), and the machine's state before and after (power source, other Python processes; `machine.py`).

**The family adapters** are in `families/`, one per family, named by the registry (`registry.Family.adapter`). Every check below was registered before its family's first run, and the record carries the date.

| family | adapter | styles | its checks |
|---|---|---|---|
| wind farm | `families/windfarm.py`: W346's march, lifted, on any rectangles (`tiling.RectangleTiling`, whose weights are `ArrayTiling.weights` to the bit on the measured tilings) | A | incompressibility closes to 1e-9 in each arm's enforcing operator; farm power within 25% of the full domain (W346's B5); threaded equal to serial bit for bit (W346's B1) |
| heat conduction | `families/conduction.py` on `fv.py` | B, C | the energy balance to 1e-6; the full domain to 1e-6 of the temperature span; a steady wall of layers against its closed form $q=\Delta T/\sum_i L_i/k_i$ to 1e-6; style B threaded equal to serial bit for bit |
| resistive plate on a circuit | `families/electric.py` on `fv.py` | D | Kirchhoff's current law at every node to 1e-6; the batteries' power equal to the heat in the resistors, inside the batteries and in the plate, to 1e-6; the joint system (plate and circuit as one sparse solve) to 1e-6 |
| pollutant plume down a river | `families/plume.py` on `fv.py` | A, explicit | mass released = held + carried out, to 1e-9 per step; the full domain to 1e-10 of the peak concentration; threaded equal to serial bit for bit |
| sound through two media | `families/acoustics.py` (a staggered-grid leapfrog) | C, explicit | the leapfrog's energy kept to 1e-10 in the closed domain; the reflected pulse's integral against $R=(Z_2-Z_1)/(Z_2+Z_1)$ to 1e-6; the two pieces equal to the full domain bit for bit |
| loaded bracket | `families/elasticity.py` on `fe.py` | B | the supports balance the loads to 1e-6; the full domain's displacements to 1e-6; threaded equal to serial bit for bit |
| heated strip | `families/thermoelastic.py` on `fe.py` | split by physics | heat in = heat stored to 1e-9 per step; the synchronous split equal to the unsplit solver bit for bit; the lagged split equal to the synchronous one a step late, bit for bit |
| cooled block | `families/cooling.py` on `fv.py` | C at a two-physics seam | heat generated = heat the coolant carries out, to 1e-6; the full domain to 1e-6 of the rise above the inlet |

## The coupling styles (`styles.py`)

The solver-family interface was derived from two real families, the wind farm and conduction, not designed ahead of them. `styles.py`'s docstring tabulates what fills each slot in each family: the state, restriction, a step given boundary traces, the full-domain equivalent, and assembly with a balance.

| style | what couples the pieces | the full-domain arm |
|---|---|---|
| A | overlapping windows, one exchange per macro-step. The wind farm marches several sub-steps on one exchange, so it is not the full domain (W346: 2–8% in farm power); the plume takes one explicit step per exchange, so it is the full domain to round-off. **With `run.exchange_every = k`** (2026-09-30; the river and, as its own style A, the sound), each window takes k explicit steps between exchanges on its own cells and a halo k cells deep read at the exchange. The stencil reaches a cell a step, so its own cells stay exact (the river to round-off, the sound to the bit) and stay in the processor's cache across the k steps | the same scheme on the undivided domain, k steps |
| M | pieces meeting at faces, each at its own stable explicit step (multirate, 2026-09-30, conduction): the pieces whose limit allows the macro-step take it whole, the others sub-cycle reading their slow neighbours interpolated in time, and each face between them is refluxed (Berger & Colella, *J. Comput. Phys.* 82, 1989), so the energy balance closes to round-off. Its agreement with the full domain is the time interpolation's, judged on its own registered check (1e-3 of the span). With equal steps it is the full domain to the bit | the same explicit update at the most restrictive cell's step, everywhere |
| B | overlapping windows iterated to a tolerance: restricted additive Schwarz, a sparse LU per window, blended in window order, so threads change no bit. On a structure, windows are the nodes of their cells | the same discretization solved directly |
| C | two pieces meeting at an interface. Iterated (conduction, the cooled block): Dirichlet–Neumann with a stated first relaxation factor, Aitken's rule optional, and the 1-D contraction factor $\rho=k_DL_N/(k_NL_D)$ on the page; unrelaxed, it diverges when $\rho>1$. A piece that would float as the Neumann side (nothing of its own sets its level) takes the Dirichlet side. Explicit (sound): the pieces trade the interface's pressures and velocities once a step, and are the full domain to the bit | the same discretization on every cell |
| D | a field's boundary integral is a lumped part's port variable: the electrodes' currents drive a circuit (nodal analysis; ideal and Norton batteries), and the circuit's potentials hold the electrodes | the plate and the circuit as one sparse system |
| split | one mesh, two physics: a conduction agent and an elasticity agent with the whole temperature field crossing between them, run synchronously or lagged a step on two threads | the unsplit solver |

`fv.py` is one finite-volume core for conduction, the electric potential and the families to come. It has harmonic-mean faces (exact for a wall of layers), upwind advection, and any set of cells. A face that leaves the set is closed in one of three ways (`neighbour`, `dirichlet`, `neumann`), and those three are what the styles differ in. The full domain is the same call on every cell, so **a one-window decomposition is the full-domain system to the bit**, and a test checks it steady and transient.

`fe.py` is the structural families' core: Q1 elements with a material per element, the build repo's `ThermoStruct2D` element (2 x 2 Gauss points, its shape functions and plane-stress matrix). On one material its stiffness, conduction and mass matrices are that solver's own to 1e-12, and its thermal load gives that solver's clamped displacement to 1e-10; a test checks both when the build repo is present.

**At showcase sizes the full domain is faster** for every family but the wind farm. On 4,000–29,600 cells a sparse direct solve, or one explicit step on the whole grid, beat every decomposed arm in the served page on AC power, by 1.5× (the sound's two pieces) to 520× (plain Schwarz on the bracket, serially, 616 iterations). The heated strip's three arms tie within the machine's noise, about 10%. These cases show the coupling, not a speedup, and the timing table gives the ratio either way.

## The compile (`compile.py`)

**Compile** (on the Run & results tab) puts the case as it stands to `atlas/compiler.py`, in a background thread. The case is copied first, so an edit made while it compiles cannot reach it. No run and no compile start while the other is active, because each would be timed with the other. The record goes beside the case file, in `<case>.results/compile-<time>.json`: every decision, the compiler's report, and the case as compiled.

**A case becomes a graph through its family.** Every family module has a `case_graph(spec)`:
- one agent per window or piece, each with a capability record;
- a connection wherever a cut face of one window opens into another (across an overlap, or across a shared face in style C);
- the cross-points, always declared (W162): named where three or more windows overlap, `()` where there are none.

Each port declares a real Fourier prolongation over its faces (up to 8 modes; one on an electrode), and the compiler derives the seam's common space.

**The responses are the families' own arithmetic.** The compiler's L4 probe perturbs each port's trace and reads the response, so every agent's `boundary_response` is a real local solve:

| family | agent's response |
|---|---|
| conduction, the cooled block, the plume | `FVAgent`: a finite-volume window's Dirichlet-to-Neumann map through its own sparse solve, or, for an explicit step, the flux the step would send. It is the flow *into* the window: the first draft returned the flow out, and the compiler read a passivity defect of 299 on the wall |
| the plate on a circuit | two agents meeting at each electrode, one number per electrode: the plate's current into itself from its own field solve, and the circuit's from nodal analysis, with the other electrodes held |
| the bracket | a window's Schur complement: its reaction to prescribed displacements on its cut nodes |
| the sound | the explicit update's velocity response to the interface pressure |
| the wind farm | the vault's own graph for the tiling: `atlas/cases/scaling_ladder.py`'s `build`, with W346's exposed windows and the rotors as agents. A drawn tiling that is not one of that ladder's rungs is refused before the compiler, with that reason |
| the heated strip | none: the whole temperature field crosses between two agents on one mesh, a volumetric bond, and `VOLUMETRIC` is not a port type. The package's `NamedHoleError: PortAmendment` refuses it before the compiler |

**Each seam's verdict is the worst of the compiler's decisions about it.** The page shows every rule that is less than `admit`, and the case's verdict is the compiler's own.

**What the examples compile to** (the gallery page has the times and the rules):
- `admit-uncertified`: the three farms, the plume, the plate on a circuit, the sound, the cooled block, and, since W348 (2026-09-30), the two-layer wall, the insert and the bracket;
- refused before the compiler: the heated strip.

No case can earn a plain `admit`: the master bound's constants are unmeasured on every graph (W56). **Until W348, R10 refused the wall, the insert and the bracket at L2**: two embedded agents of one physics cut its region, and the rule ran before the scheme existed. Their pieces take every datum on their cut from the coupling (`elliptic_data_from_ports`), and the compiled scheme solves the interface directly (`K_accelerator = direct-schur`). For a linear problem that is the undivided solution, so R10 now decides them after the scheme, at L5, and decertifies with the cost of keeping each solve inside its piece (W168's $149\times$ in sweeps) instead of refusing. The graphs R10 was measured on, Tier 0's four windows and CS-S1, still refuse at L2 (`scripts/w348_controls.py`). The Run step measures the pieces within $3.6\times10^{-9}$ of the full domain or closer.

## The showcase gallery

Each kind's examples are under the header's **Fast example** button: the button opens the kind's Fast example (below), and its arrow lists the rest, one line each saying what it shows and no name (the owner's decision O1, 2026-09-30; until then *Case > Examples...* listed all twenty-one with their names and descriptions). Every example was opened, checked, compiled and run in the served page on 2026-09-29. `wiki/concepts/Atlas 0.1/atlas-0.1-outcome/showcase-gallery.md` has a row per case: its style, port type, compile verdict, its two checks' measured values against their tolerances, and its runtime. The records are in `out/workbench/records/step-f/`. Every run finished under 2 minutes, the longest in 70 s. The 21-rotor farm's compile takes 95 s, so its compile plus its run is 165 s, which the owner accepted on 2026-09-29, since the compile is its own step. Each row is labelled a showcase, not a research record.

## The Fast example (demo item 1.4, 2026-09-30 to 10-01)

One complete case per kind, chosen to show the pieces running faster than the whole (`spec.FAST_EXAMPLES`). **The bars are registered in each family module** before its first timed run (`FAST`, judged by `fast.py`):
- the fastest decomposed arm at least 3× the whole domain, $s=t_{\text{full}}/t_{\text{arm}}$;
- agreement with the whole domain within $10^{-9}$ of the field's scale for the same algebra, $10^{-3}$ for multirate, 3% in farm power for the farm.

A sweep chose each configuration and a confirmation run from a fresh start measured it (`scripts/fast_examples.py`). **Four meet both bars:**

| kind | example | how | $s$ |
|---|---|---|---|
| wind farm | `fast-farm` | 12 windows on 4 threads, the 5-rotor rung | 4.62 |
| heat conduction | `fast-heat` | style M: a copper spreader sub-cycling 9 steps to the steel's one | 5.58 |
| river | `fast-river` | 6 windows on 6 threads, 24 steps per exchange on a halo | 5.85 |
| sound | `fast-sound` | style A: 4 windows on 4 threads, 16 leapfrog steps per exchange | 4.67 |

The structure, the plate on its circuit and the cooled block are one direct solve here, factored once and reused, so their pieces lose (0.004–0.275×). The heated structure's split by physics can at most halve the time (1.17×). Each of the four loads its fastest honest setup, and its card says what limits it (O3).

**The card** comes first in Run & results: the mechanism in one line with the run's own numbers, the run against the bars, and the fixed line every card carries. The farm's example runs the threaded windows and the whole domain, because its power needs 28 macro-steps to come within 3% and three arms would take over two minutes; opening it sets Run settings so, and says so. The gallery's §8 has the table, the step-0 micro-benchmarks and what the sweeps found.

## The geometry section

The hybrid the owner chose on 2026-09-28: rectangles on the grid are drawn in the
page; anything else comes from Gmsh. **Superseded in part on 2026-09-29**: any
shape with straight, arc or spline edges is now drawn in the page too (see
"Drawn shapes" below), and Gmsh stays the path for a mesh made elsewhere.

**One canvas, a layer at a time** (the Model tab, since 2026-09-29's rebuild; the
tables of the six-step page went into the inspector).

| layer | on the canvas | in the rail and the inspector |
|---|---|---|
| **Shape** | the whole grid until an outline is drawn, with holes cut in it (*Draw the outline*, *Cut a hole*, *Use the whole grid*); the outline is selected on arrival, its handles ready | the outline's edges, each with its kind and the condition on it; a selected edge drawn straight, as an arc or smooth |
| **Materials** | regions drawn with any tool, in the material chosen in the rail; imported polygons (with holes) are shown but not moved. Regions **stack in list order**: a later one takes the cells it covers, so a plate with an insert needs no cutting | the materials in use and the regions; a region's material, stacking and place; a material's properties, one from the library, or a new one by name. Offered to a family that reads regions, or while a case holds some |
| **Boundaries** | drawn along the domain's edges, coloured by kind; a click selects an edge | its condition, value and span, and *Split it in two*. Fixed when the family's solver fixes its boundary (the wind farm, the river and the acoustics do), with the reason |
| **Windows** | drawn with any tool, or generated from the shape (*Automatic*); the live warnings below | how they are cut and joined, the blend width; a window's name and place; *Lay a grid of rectangles...* |
| **Rotors** | *Place rotors* adds one per click; drag to move | position and diameter |
| **Circuit** (lumped parts) | a schematic: each electrode is a node beside its edge segment, the circuit's other nodes sit in a row below the domain, and each battery or resistor runs between its two nodes | a part's value, a battery's internal resistance, its two nodes; *Add a battery*, *Add a resistor*. For a family that reads no circuit, a circuit left in the case is shown with that said, and nothing can be added |

**The selected shape's handles**: a **square** per vertex (drag to move it,
double-click to smooth the curve through it or make it a corner again, Backspace to
remove it), a **circle** per edge (click to select the edge, drag to bend it into an
arc, double-click to add a vertex), a **diamond** to move the whole shape (Backspace
deletes it), and on a rectangle four corners to resize it.

Every edit snaps to the chosen step (1, 8 or 16 cells; 8 by default), goes
through the same validation, undo and save as any other edit, and is drawn back
from the case, so the canvas never shows a shape the case does not hold.

**The live warnings are the check's own rules** (`geometry.analyse_windows`):

- **red hatching**: cells with no window at full weight. A window's weight ramps
  from 0 at each artificial face over the ramp width (`wake_array.ArrayTiling.weights`);
  every measured tiling gives each cell one window at weight 1. Touching windows,
  overlaps under twice the ramp, and windows thinner than two ramps all fail it. An error.
- **grey hatching**: cells in no window. An error.
- **thin seams**, named only where the red hatching is: a warning.
- **diamonds**: cross-points, where three or more windows overlap (information; the
  compiler's rules for them are L2/I2/G1).

**Gmsh.** *Case > Import geometry from Gmsh (.msh)...*. Name physical groups
`domain`, `region:<material>`, `window:<id>`, and `bc:<kind>` or `bc:<kind>=<value>`
(on edge curves); mesh in 2-D; save the mesh. Layers the file defines replace the
case's; the others are kept. Only `.msh` is read, because a `.geo` file is a
script and its language can run commands.

**The six geometry decisions** in the showcase plan were not answered one by one;
the build took the plan's recommendation in each, and each is a small change:
snapping 8 cells (selectable); a tiling generator plus hand editing; corner
handles plus an editable table; warn on the canvas and let the check refuse; all
four layers now.

## Drawn shapes: irregular domains and windows of any shape (case file 0.4)

The owner's request of 2026-09-29: draw irregular geometry edges, and have windows
that are not rectangles. The owner chose **the cells inside the shape**: the grid
stays Cartesian, every shape is drawn exactly, and a solver gets the cells whose
centres the shape contains. So every existing solver and check carries over, and
a curve is a staircase at the cell size.

**A shape is vertices and edges** (`shapes.py`). Each edge is a straight `line`, a
circular `arc` or a `spline`:
- an arc is stored by its bulge $b=\tan(\theta/4)$, so it keeps its shape when an
  end moves ($|b|=1$ is a semicircle);
- a spline is a centripetal Catmull-Rom segment, so a run of spline edges is one
  smooth curve through its vertices.

One kind of shape serves the domain's outline, holes cut in the domain, material
regions, and windows (which may have holes too).

**On the canvas** (the Shape, Materials and Windows layers): **Draw**, **Rectangle**
and **Circle** add a shape (on the Shape layer, the outline or a hole), and
**Select** reshapes the selected one with its handles, as "The geometry section"
above describes. A selected edge's kind and bulge are also set in the inspector.
What lies outside a drawn domain is drawn pale and dotted, stepped as the solver
sees the domain's cells, with the exact outline over it.

**What changes underneath:**
- `fv.Field` takes the domain's mask. A face between the domain and the void is a
  boundary face, and it takes the condition of the **drawn edge nearest it**
  (`geometry.void_faces`). A boundary names its edge as `outline:<k>` or
  `hole<h>:<k>`, and when an edge is split both halves keep its condition.
- A drawn window holds the domain's cells whose centres it contains. Its
  partition-of-unity weight ramps with the distance from its artificial faces
  (`geometry.ramp_weight`: $(\mathrm{clip}((d-\tfrac12)/r,0,1))^2$, the rectangles'
  profile with Euclidean distance) and is certified like the rectangles'
  (`tiling.MaskTiling`).
- A curved seam's faces are ordered along the seam for the compiler's Fourier
  ports (`compile.curve_order`).
- **A case with no drawn shape runs on exactly the arithmetic it did before**
  (`geometry.is_plain`): the same analysis, `RectangleTiling`, face order and sums,
  and all 141 earlier tests pass unchanged.

**Checked** (`tests/test_workbench_drawn.py`, and the records in
`out/workbench/records/step-g/`):
- one window over a drawn domain equals the full domain bit for bit, steady and
  transient;
- **the bend** (`bend-3`: a quarter ring drawn with arcs, three ring-sector windows,
  style B) agrees with the full domain to $3.07\times10^{-10}$ of its span, closes
  its energy to $1.24\times10^{-9}$, and threaded equals serial bit for bit. Its
  staircase moves the heat flow by -0.97% from the ring's continuum
  $k\,\Delta T\ln(r_o/r_i)/\theta$;
- **the round insert** (`insert-round`: Dirichlet-Neumann across a circle of cell
  faces, style C) agrees to $1.01\times10^{-10}$ and closes to
  $9.86\times10^{-12}$;
- both compile with every seam admitted, and R10 decertifies them after the scheme,
  as it does their rectangular counterparts (W348).

## Holes anywhere (2026-09-30)

The owner: "Holes should be allowed to intersect the boundary's edge. Also, they
should also be allowed to extend beyond the geometry's edge, as long as the geometry
itself is a closed shape." Two rules refused both; they are lifted. **A hole is any
closed shape that does not cross itself, anywhere**: across the outline (a notch) or
past the grid (it removes only the grid's cells).

The domain is its cells, and a hole only removes cells, so no solver changed. What
changed is the bookkeeping that refers back to the drawing:
- **each face takes the condition of the nearest edge that still bounds the domain**
  (`geometry.live_edges`): the outline's parts inside a hole, and a hole's parts
  outside the outline, past the grid or inside another hole, are no one's edge. Where
  nothing is cut, `live_edges` returns `drawn_edges`' own arrays, so every earlier
  case labels its faces to the bit (a test checks every drawn example);
- **a condition on an edge a hole took whole is a problem** ("the river-inlet B4 is on
  outline:3, an edge the hole removed: ... it does nothing"), with a Fix that moves it,
  id and all, to the nearest edge left. A newly drawn shape never makes that move
  unasked (`starter.ASKED_ONLY`): it moves the person's own choice;
- **a hole that cuts the domain in two is refused**, with the reason;
- **cutting across reads holes from the mask** (`layout.topological_holes`): a hole
  that bites the outline is a notch, and the layout may cut across it;
- **the heat through each piece of an edge** a hole cuts is reported by the conduction
  family (`heat_through_pieces`), in the record and the run's notes.

Checked (`tests/test_workbench_holes.py`): the one-window control on a domain with a
crossing hole, bit for bit, for the river, the plate, the channel and the farm; the
split and the sound's pieces equal their references to the bit; the circuit and the
cooled block pass their own checks. A plate pulled on what a hole leaves of its edge
balances to $10^{-6}$, and a wall whose cold edge a hole cuts reports both pieces.
In the served page, a circle hole dragged across an outline's edge and past the grid
left the case ready; its run closed energy to $4.5\times10^{-12}$ and agreed with the
full domain to $8.4\times10^{-11}$, and a river whose whole inlet a hole took showed
the problem, and its Fix cleared it.

## A river that forks (2026-09-30)

The owner: "In the stream pollutant case, it doesn't let me have a geometry with a
bifurcation, and two outlets." Reproduced first: a Y ran, but it had three quieter
faults.
- The starter gave it one outlet, and nothing said the other branch carried no water.
- The automatic windows ran between the Y's two branch tips, so they split its stem
  lengthwise.
- An island that split a river into two outlets was refused as a hole outside the
  outline (fixed by "Holes anywhere", above).

**A shape branches when the coordinate between its two ends leaves a dead region**
(`layout.dead_regions`): cells where it changes by under a thousandth of its mean. Into
a branch that neither end is in, a harmonic coordinate decays like $e^{-\pi x/w}$, so
past about $2.2\,w$ it is dead. A convex corner is not: no drawn example, no rectangle
and no L has a single slow cell (a test runs every example). The plan's rule, local
maxima of the distance round the boundary, would have called a rectangle's four corners
four ends.

**A branched river's windows follow its own flow**: the along coordinate is $1-\phi$,
$\phi$ the river's potential (1 on its inlets, 0 on its outlets), cut into bands at
equal cell counts, and **each connected piece of a band is a window**, so a band across
both branches is two windows. Any other branched shape is cut by **recursive spectral
bisection** (Pothen, Simon and Liou, 1990). A shape with two ends is cut as before, to the
bit. The Windows layer says which cut it took (`Layout.how`).

**A branch that carries no water is a warning with a Fix**, *Make the end of this branch
an outlet*: the bank beside its dead water farthest from the inlets along the river's
own paths. A newly drawn fork gets it at once, said in the note, as every starting value
is. **The outfall of a drawn river goes mid-river a fifth of the way down its flow**: it
had gone to the middle of its cells, which on a Y is the fork, and a release on the
dividing streamline went down one branch alone (the served page measured 100% and 0%).
The starting values now run in a fixed order, a river's ends first.

**The run reports each outlet's share** of the water and of the pollutant leaving, in a
line over the cards and in the record. A potential flow splits by the channels' shapes
alone, which is a stated guess. **A fourth check**, registered before the fork's first
run: the flow's continuity, every cell's flows summing to $10^{-12}$ of the discharge.

**The example `river-fork`**: a Y drawn with splines, its stem entering on the grid's
left edge and two branches of different widths leaving by two outlets on its right, on
six windows along its flow. In the served page it compiled `admit-uncertified` (six
agents, five seams, one from the fork to each branch) and ran 2,000 steps with every
check passing: mass $2.7\times10^{-14}$, the full river $2.1\times10^{-16}$ of the peak,
threaded equal to serial, continuity $2.2\times10^{-14}$. The wider branch took 53.6% of
the water and 57.3% of the pollutant. **The owner's scenario, walked in the page**:
river plume chosen, a smooth Y drawn with its stem on the left, Run. It was ready at
once with two outlets, and the plume left by both (54.0% and 46.0%), every check
passing. Records: `out/workbench/records/demo-fork/`. `tests/test_workbench_fork.py`
pins all of it, the one-window control bit for bit among it.

## Windows that follow the domain (case file 0.5)

The owner's next question, the same day: "it doesn't make sense for the windows
themselves to be rectangles, does it? Why doesn't their shape itself ADAPT to the
geometry of the curve smoothly?"

They did not adapt because nothing generated a window from the geometry. A window
was whatever was drawn, and a generated tiling was rectangles. Now a case can
have a **layout** (`layout.py`): the windows are generated from the domain's own
shape, and generated again whenever that shape changes.

**The domain's own coordinates.**
1. Its **Fiedler vector** (the Laplacian's first non-constant eigenvector on its
   cells) runs from one extreme of the domain to the other. The boundary edges
   where it is lowest and highest are the domain's two **ends**.
2. The **along** coordinate is harmonic in the domain: 0 on one end, 1 on the
   other, no flux through the walls. It is one `fv.assemble` solve with unit
   conductivity. Its level curves cross from wall to wall and meet each wall at a
   right angle.
3. The **across** coordinate is harmonic between the two sides.

Cutting at equal cell counts gives pieces that are rectangles in those
coordinates, so they bend with the domain. On the quarter ring the along
coordinate is the angle, to 0.0075 (the staircase), and the across coordinate is
$\ln(r/r_i)/\ln(r_o/r_i)$, to 0.006. So the cuts fall at exactly 30 and 60
degrees, and on the arc of equal areas $r=\sqrt{(r_i^2+r_o^2)/2}=76.2$.

**One per material** is the other cut: its Dirichlet-Neumann interface is the
material interface itself.

For styles A and B each piece grows inside the domain, never across a gap, by
the least reach that gives every cell a window at full weight. For C, D and the
split the pieces meet along faces.

**In the page** (Model, Windows, nothing selected): *Automatic* or *My own*, how to
cut (*along the shape's length* or *one piece per material*), and how many pieces
along and across.
- Drawing a domain whose windows are still rectangles makes them give way to
  generated ones.
- Reshaping the domain generates the windows again (`Workbench.edit`).
- Editing a window by hand makes the windows the case's own, with a note.

A generated window is stored as its cells (`shape = "cells"`, runs along rows), so
a record holds the windows it marched on. It is drawn as the smooth curves where it
ends inside the domain: the 1/2 contour of its smoothed indicator, traced by a
small marching-squares routine (`geometry.contour_lines`), so no new dependency.

**The example**, `s-channel`: a steel channel drawn with splines, on four windows
generated from its shape. It agrees with the full domain to $1.8\times10^{-9}$ of
its span, threaded equals serial, and every seam is admitted. R10 decertifies it
after the scheme (W348). Moving a vertex of its wall regenerates the windows to fit.

## Drawn shapes in every family

The owner, the same day: "make the smooth domain/spline/other stuff available for
all cases". Until then only conduction ran on a drawn domain; each other family
refused one with its reason. Now all eight run on the cells a drawn shape contains,
with nothing new in the case file (a 0.5 file loads unchanged). What each needed:

| family | what a drawn domain needed |
|---|---|
| electric | electrodes on drawn edges: an electrode's faces are the boundary faces its edge holds (`geometry.boundary_faces`), and the plate is solved on the domain's cells |
| transport | **a flow that follows the banks**: the potential flow from the inlets to the outlets (`flow.py`), divergence-free face by face and zero through every bank; the explicit limit read off the assembled diagonal |
| acoustics | **rigid drawn walls**: a face to the void never opens, so the domain stays closed; two pieces of any shape, each owning its faces, so they equal the full domain to the bit |
| elasticity | an element mask: the void carries no stiffness and its nodes leave the solve; clamps and loads on drawn edges; windows on the node grid by the mask partition of unity |
| thermoelastic | the same mask for both agents; fixed temperatures and a heat flux on drawn edges; the free body's rigid-body modes on the domain's nodes |
| conjugate heat | the coolant's flow solved through its own cells (`flow.py`), so a winding channel works; cut by physics (block and channel) |
| wind farm | **the solid held at rest by penalization**: every sub-step, just before the projection, the velocity outside the domain is set to zero. Each window marches its box, the window as drawn with the solid in it held at rest, and the windows blend by the mask partition of unity |

**The families that fix their outer boundary** keep doing so. On a drawn domain,
each drawn edge takes the condition of the grid edge it lies along, and the
family's default elsewhere (`spec.derived_kind`). So the wind farm's terrain is a
wall and its top edge a freestream, and a river drawn from the grid's left edge to
its right edge enters on the left and leaves on the right. For the wind farm and the
sound these follow the geometry on every edit (`spec.derive_boundaries`); the river
starts so, and any of its edges can be made an inlet or an outlet.

**A flux or a traction on a drawn edge is per unit length of the edge as drawn.**
Its staircase of faces is up to $\sqrt2$ longer than a slanted edge, so each face
carries the value times the edge's true length over its faces' length
(`geometry.staircase_scale`). The edge then carries its whole load and no more.

**Seven examples, one per family** (`ring-film`, `river-bend`, `sound-lens`,
`plate-hole`, `bimetal-arc`, `cooled-winding`, `farm-hill`). Each is clean, passes
every registered check, and is pinned in `tests/test_workbench_drawn_families.py`.
The records are in `out/workbench/records/step-i/`, made headless on battery power,
so their times are not quoted. Each was compiled and run again on AC power on
2026-09-29 and 30, with the gallery's columns (`out/workbench/records/step-j/`, and
`showcase-gallery.md` in the vault); every compile is pinned in the same tests. What
they measured:

| example | family | what it measured (registered tolerance) |
|---|---|---|
| `ring-film` | electric, D | Kirchhoff $5.4\times10^{-13}$, energy $9.6\times10^{-14}$, current against the joint solve $6.0\times10^{-14}$ ($10^{-6}$ each). The film's resistance, 0.5770 ohm, is +0.98% from the ring's continuum $\theta/(\sigma t\ln(r_o/r_i))$ = 0.5714 ohm: the staircase |
| `river-bend` | transport, A | 2,000 explicit steps: mass $3.3\times10^{-14}$ ($10^{-9}$), concentration against the full river $1.8\times10^{-16}$ ($10^{-10}$), threaded equals serial. The solved flow's continuity residual is $1.4\times10^{-14}$ of the discharge, and no face of a bank carries any flow |
| `sound-lens` | acoustics, C | 1,500 leapfrog steps: energy $4.2\times10^{-16}$ ($10^{-10}$), the two pieces equal the full domain bit for bit; no textbook reflection on a curved surface |
| `plate-hole` | elasticity, B | forces $7.7\times10^{-12}$, displacements against the full plate $3.2\times10^{-12}$ ($10^{-6}$ each), threaded equals serial |
| `bimetal-arc` | thermoelastic, split | 60 steps: heat $2.2\times10^{-10}$ ($10^{-9}$); the synchronous split is the unsplit solver bit for bit, and the lagged one is it a step late, bit for bit |
| `cooled-winding` | conjugate heat, C | every watt leaves in the water, $4.7\times10^{-11}$ ($10^{-6}$); temperatures against the full domain $1.5\times10^{-9}$ of the rise ($10^{-6}$); the coolant's flow is continuous to $7.1\times10^{-14}$ |
| `farm-hill` | wind farm, A | 40 macro-steps: mass $2.2\times10^{-16}$ ($10^{-9}$), threaded equals serial, and farm power **19.7%** from the full domain: inside W346's registered 25%, and well above the 2.3-8.3% the plain farms measured. Its velocity differs from the full domain's by 0.20 U rms, against 0.047 U for the plain three-rotor farm, and its windows take 26 sub-steps a macro-step against the full domain's 32. Both controls hold bit for bit: one window over the drawn farm is its full domain, and a drawn farm that is the whole rectangle is the plain farm, both arms, both arrangements. So the departure is the hill's, under one exchange per macro-step, not the drawn machinery's |

**Not built:** the potential flow has no viscosity and no inertia. It turns a bend
without separating, which is right for a scalar's path and wrong for a separated
flow. The wind farm's penalization is first order in the sub-step, because the
projection lets a little velocity back into the solid each sub-step.

**Fixed on 2026-09-30, found by recording the examples for the gallery:**
- **A drawn farm's windows did not hold their ground.** Each window marched the box
  of its fluid cells alone, so a window over the terrain had its box's edge where
  the solid began, and one window over the whole drawn farm was not its full domain
  (0.61 U after one macro-step). A window's box is now the window as drawn, and the
  penalization holds the solid in it at rest, as the full domain's does. The example
  moved by 0.1 point of farm power: its departure was the hill's all along.
- **The drawn river did not compile**: the compile's velocity scale divided by the
  river's shallowest depth, which is zero past a drawn river's banks.
- **A drawn farm compiled as the plain one.** Its graph is the scaling-ladder rung's,
  a full rectangle of fluid, so the terrain was nowhere in it. A drawn farm is now
  refused before the compiler, with that reason.

## Code

| file | role |
|---|---|
| `spec.py` | the case file (`atlas-workbench/case@0.6`, with no name or description; 0.1 to 0.5 files migrate on load), `check()` with each family's rules, and the twenty-two examples, each with the one line *More examples* shows (`SHOWS`): three farms read from `atlas/cases/scaling_ladder.py`'s real tilings, one or two per other family, two drawn ones (the bend and the round insert), one whose windows are generated from its shape (the S-channel), one drawn example per other family, and the forked river |
| `flow.py` | a flow that follows a drawn domain: the potential flow from its inlet faces to its outlet faces, for the river and the coolant |
| `layout.py` | windows generated from the geometry: the domain's ends from its Fiedler vector, its harmonic along and across coordinates, equal-count cuts, one piece per material, growth to full weight, and regeneration when what they follow changes; for a shape that branches (`dead_regions`), a river's own flow with each connected band a window, or recursive spectral bisection (`bisect`); holes read from the mask (`topological_holes`) |
| `shapes.py` | drawn shapes: vertices joined by lines, circular arcs (by their bulge) or centripetal Catmull-Rom splines; sampling, crossing and area checks, and the edits the canvas makes (move, bend, split, remove, smooth) |
| `geometry.py` | the geometry rules as plain functions: snapping, tilings, the full-weight analysis, region masks and stacking; for drawn shapes, their cell masks, the distance ramp, the analysis on masks, and a drawn domain's boundary faces labelled by the nearest edge that still bounds it (`live_edges`) |
| `editor.py` | the Model tab's canvas: its layers, the five tools, what a click selects on each layer (`hit`), the selected shape's handles, and the edits they make |
| `inspector.py` | the Model tab's rail and inspector: the layer buttons, the active layer's list, and the panel for what is selected (or the layer's settings) |
| `gmsh_import.py` | `.msh` to case geometry, by physical-group names |
| `registry.py` | the physics families: what each runs on, which layers it reads, which boundaries it can impose, its short name, and what is missing before the workbench can run it |
| `starter.py` | what a new shape or a new kind starts with: the repairs keyed by the check's problems (`FIXES`, `apply_fix`), `fill_defaults`, and the header's new case of a kind (`new_case`, `new_case_from`) |
| `app.py` | the GUI: the header (the type selector, *More examples*), the two tabs and the dialogs (the problems list, open, save, Gmsh). Every menu item and button goes through `Workbench.dispatch(action)`, every edit through `Workbench.edit`; Undo restores the case with its file and key (`_Version`) |
| `tiling.py` | any rectangles as a partition of unity, certified by `GridPartitionOfUnity` |
| `runner.py` | the run: arms in turns, the timer, Stop, the snapshots, the record |
| `runview.py` | the Run & results tab: the controls, the compile card, the four cards, the live fields, timings and convergence |
| `checks.py` | a check as data: what is measured, its tolerance, when it was registered |
| `machine.py` | the machine's state beside every timing, and the keep-awake request |
| `families/windfarm.py` | the wind-farm family's adapter (style A) |
| `fv.py` | the finite-volume core: harmonic-mean faces, upwind advection, any set of cells, three ways to close a cut face |
| `styles.py` | styles B, C and D as drivers: restricted additive Schwarz, Dirichlet–Neumann with Aitken, a field on a lumped network; style M's two-rate explicit step, refluxed (`TwoRate`) |
| `fast.py` | the Fast example's bars (`FastBars`, registered in each family's `FAST`), the run's agreement read from its checks, the judgement, the card's mechanism lines and the fixed line (O3) |
| `families/conduction.py` | heat conduction in several materials (styles B and C, steady or transient) |
| `families/electric.py` | a resistive plate on a battery-and-resistor circuit (style D) |
| `fe.py` | Q1 finite elements with a material per element: plane-stress stiffness, conduction and mass, the thermal load, rigid-body modes, stresses |
| `families/plume.py` | a pollutant plume down a river of two reaches (style A, one explicit step per exchange) |
| `families/acoustics.py` | sound through two media on a staggered grid (style C, explicit) |
| `families/elasticity.py` | a two-material bracket under load (style B on a structure) |
| `families/thermoelastic.py` | a heated bimetal strip, conduction and elasticity split by physics |
| `families/cooling.py` | a heated block cooled by a channel flow (style C at a seam between two physics) |
| `compile.py` | the compile: a case's `CaseGraph` through its family's `case_graph`, the finite-volume window as an agent (`FVAgent`), seams from the geometry with Fourier prolongations, the declared cross-points, the verdict per seam, and the background job with its record |
| `tests/test_workbench_shell.py`, `tests/test_workbench_geometry.py`, `tests/test_workbench_runner.py` | the generator against the measured tilings, the full-weight rule against the assembly's own weights, the Gmsh path on a real mesh, the canvas driven as the page drives it (a click is a Tap at `_on_tap`, a handle's drag the data Bokeh's `PointDrawTool` sends, a field a widget; helpers in `tests/workbench_ui.py`), every header control and menu action (nothing is left unbuilt), the status chip's list and its *Show me*, and the runner: its three arms against W346's own columns bit for bit, the one-window control, Stop, the record with its field differences, every decomposed arm's farm metrics, and the stale-case labels |
| `tests/test_workbench_fv.py`, `tests/test_workbench_families.py` | the finite volumes against closed forms (a layered wall, a series circuit), one window equal to the full domain to the bit, styles B, C and D against the full domain, the unrelaxed Dirichlet–Neumann diverging exactly when $\rho>1$, both new families end to end, and the page's family, physics and materials controls |
| `tests/test_workbench_fast.py` | the Fast examples: every family's bars registered, every kind's example ready to run, style M's N = 1 control to the bit and its balance and agreement, the river's and the sound's halo exchange against the full domain (the sound to the bit), the river's leaner step against the code it replaced, the header's button and the card |
| `tests/test_w348_r10_after_scheme.py` | R10 after the scheme (W348): its three outcomes, `wall-2` decertified, the same graph refused at L2 with the declaration withdrawn, and Tier 0's four windows still refused |
| `tests/test_workbench_starter.py` | the owner's first scenario in every family (ready to run once the shape is drawn, or the one thing only the person can decide said plainly), a new kind starting from its example's setup and one Undo restoring the old case, defaults that fill only what is missing, the Fix buttons one by one and all at once, every kind from the header ready to run and *Start over*, the windows fix at a Dirichlet-Neumann interface, a check that never raises, the canvas holding still while a shape is drawn, and the header holding what it simulates. `tests/test_workbench_shell.py` also pins case@0.6 (no name; an older file loses its own), *More examples*, the type switch and its one Undo with the file, and that no control shows a case name |
| `tests/test_workbench_cases.py` | the five step-C families: each one-window (or one-piece) control, each registered check, their positive controls (no interface reflects nothing; a plain strip heated uniformly carries no stress), `fe.py` against `ThermoStruct2D`, the floating-piece rule, the circuit layer, and each family in the page. And the compile: every example's verdict pinned, the split refused by the port vocabulary, the farm through its own graph (and a moved window refused), a seam's response rising with its trace, the Run & results tab's per-seam table, and the gallery opening every example |

## Known limitations

- **Light theme only, whatever the OS is set to.** Panel's theme switches reload the page, which starts a new session and drops the open case.
- Undo covers case edits, not view or tool changes.
- **Regions are read by every family but the wind farm**, which ignores them, and the check says so.
- Timings are taken inside the workbench server process, which also serves the page; the page's updates share the process with every arm alike.
- **The circuit is drawn, not dragged**: its parts are set in the inspector and the canvas draws the schematic.
- **At showcase sizes the decomposed arms are slower than the full domain** for every family but the wind farm (above). This is measured and shown, not hidden.
- **Plain Schwarz has no coarse level**, so it converges slowly on a bending structure (616 iterations on the bracket); a coarse space or a Krylov wrapper would be the remedy, and neither is built.
- **The wind farm compiles only on a scaling-ladder rung, on the full rectangle.** Its graph is the vault's `scaling_ladder.build`, which declares a regular tiling of fluid. A drawn tiling that is not a rung, or a drawn domain, runs, but is refused before the compiler, with that reason.
- **A drawn river's or channel's flow is a potential flow** (`flow.py`), with no viscosity and no inertia: it carries a scalar the right way round a bend, and does not separate. **The wind farm's drawn walls are penalized**, first order in the sub-step. **A drawn sound case reads no textbook reflection**: that needs two uniform media at a straight cut. Its energy and the pieces' agreement are checked.
- **A drawn curve is resolved to the grid's cells.** The solvers see the cells whose centres a shape contains, so a curve is a staircase at the cell size. The canvas draws both, and on the bend the staircase moves the ring's heat flow by -0.97% from its continuum value. Body-fitted grids would remove that; they are not built.
- **The 21-rotor farm's compile takes about 95 s** (124 seams, each probed through its agents' solves), so its compile plus its run is about 165 s. Each is under 2 minutes, but together they are not. The owner accepted this on 2026-09-29: the compile is its own step, and the two-minute rule is per run.
- **Sound's style A has no declared graph yet.** Its windows step a halo as deep as their steps between exchanges, which no graph here declares, so its compile is refused before the compiler, with that reason (the run is the full domain to the bit).
- **R10 decertifies the iterated one-physics cases rather than admitting them** (styles B and C: the wall, the insert, the bracket; W348). The pieces' declaration that the coupling supplies their boundary data cannot be checked by the compiler, and the fact it rests on, that the interface solution of a linear problem is the undivided one (T3 in the vault's formal-proofs plan), is cited from Frommer & Szyld (2001), not yet machine-checked.
- **The starting values are stated guesses.** A river's inlet is the shape's leftmost edge and its outlet its rightmost, and so for a plate's electrodes, a structure's clamp and load, and a hot and a cold end. A river that runs another way needs its edges set by hand. Every guess is listed when it is made.
- **Two things have no default**: a wind farm's air must reach the grid's left and right edges, and a cooled block needs its channel drawn. The problems list says so.
- **A tool adds one shape**, then hands back to Select; there is no key that keeps it armed, so placing five rotors is five presses of *Place rotors*.
- **A finished run or compile rebuilds the Run & results tab**, and the page scrolls back to its top; that is why Run and the compile's verdict are placed there.
