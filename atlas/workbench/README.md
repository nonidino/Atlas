# Atlas Workbench

Build a domain decomposition by hand, compile it with the Atlas compiler, run it,
and compare it with the full-domain solve. Built:
- the menus, the workflow, the case file and the **geometry section**;
- the **runner**, with eight physics families, one per showcase case: the wind
  farm (style A), heat conduction in several materials (styles B and C), a
  resistive plate on a circuit (style D), a pollutant plume down a river (style A,
  explicit), sound through two media (style C, explicit), a loaded bracket (style
  B), a heated strip split by physics, and a heated block cooled by a channel
  flow (style C at a seam between two physics);
- the **compile**: each family turns a case into the Atlas compiler's `CaseGraph`,
  and the Check step shows the compiler's verdict per seam;
- the **showcase gallery**: *File > Showcase gallery* lists every example, and
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
part of a GUI that is hardest to hand-write well: menus, dialogs, tables,
notifications, and a canvas whose `BoxEditTool` and `PointDrawTool` draw, drag
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

| part | where | what it does now |
|---|---|---|
| menus | header | **File** (new, the showcase gallery, thirteen examples, open, save, save as, import and export JSON, import geometry from Gmsh), **Edit** (undo, redo, generate a window tiling, revert), **View** (grid, overlaps, labels, activity log), **Run** (check; compile with the Atlas compiler; run decomposed, full, or the ticked arms; stop; show the results), **Help** |
| workflow | sidebar | six steps, each labelled with its status (`ok`, `N errors`, the compile's verdict, the run's state; `before a change` once the open case is no longer the one compiled or run), and a live case check |
| case bar | top of the workspace | which case is open, and whether and where it is saved |
| workspace | main | the active step |
| activity | bottom | a timestamped log of every action |

The six steps:
1. **Case**: name, description, physics family, domain size. A family that cannot run yet is listed and disabled. Works.
2. **Geometry**: the canvas and its six layers, with drawn shapes on the Domain, Windows and Regions layers (lines, arcs and splines). Works; see below.
3. **Physics & coupling**: the family's own parameters (from the registry), the run mode (steady or transient), the macro-step, the coupling style and its settings (ramp width; tolerance and iteration cap; first relaxation factor, Aitken, and which piece takes the Dirichlet side; none of these for a family whose exchange is explicit), and the materials table, editable or picked from the family's library. Works.
4. **Check**: every rule, with where to fix it, and **Compile with the Atlas compiler**: the verdict, the agents and seams, the declared cross-points, a per-seam table with the rules behind each verdict, and the decisions about the graph as a whole. See "The compile" below.
5. **Run & compare**: works for all eight families; see below.
6. **Results**: from the run's record, each arm's time per step and its ratio to the full domain, the family's metrics, the displayed field's rms difference from the full domain, and the sanity checks against their registered tolerances. If the open case is no longer the one that ran, the step says whose results these are.

## The runner

`runner.py` marches a **committed** case: Run copies the case, so an edit made while it marches cannot reach it, and the page names the version that is marching. Every macro-step each ticked arm takes one step, in an order that rotates by one arm per step. The wind farm's arms:

| arm | what runs |
|---|---|
| Decomposed, serial | the windows stepped as one batch per window shape (W346's `E`) |
| Decomposed, parallel | the same windows split across threads, every chunk at the whole tiling's sub-step count, so it is **bit for bit** the serial arm (checked every step; W346's `Ep`) |
| Full domain | the same discretization on the undivided domain (W346's `F`) |

The other families have the same three arms where their style allows them. A style-C case has no parallel arm: Dirichlet–Neumann is sequential by construction, explicit or iterated. The Run step names every arm it drops, and why. A steady case repeats the same solve from the initial state, so its "macro-step" is a **timed repeat**, and the page says so. A family may name its own arms: the heated strip's are the synchronous split, the lagged split (on two threads) and the unsplit solver.

It runs in a background thread; the page reads its progress every 400 ms and draws the fields, their difference, the time per step, the family's series (farm power; outflow; the energy's share in the second medium; the largest stress), and, for an iterated style, the iteration's convergence curve. A field that spans decades (the plume) is drawn on a log scale, for display only. **Run > Stop** stops after the arm-step in progress and keeps whole macro-steps only. One run at a time per server, because two runs on one machine would time each other.

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
| A | overlapping windows, one exchange per macro-step. The wind farm marches several sub-steps on one exchange, so it is not the full domain (W346: 2–8% in farm power); the plume takes one explicit step per exchange, so it is the full domain to round-off | the same scheme on the undivided domain |
| B | overlapping windows iterated to a tolerance: restricted additive Schwarz, a sparse LU per window, blended in window order, so threads change no bit. On a structure, windows are the nodes of their cells | the same discretization solved directly |
| C | two pieces meeting at an interface. Iterated (conduction, the cooled block): Dirichlet–Neumann with a stated first relaxation factor, Aitken's rule optional, and the 1-D contraction factor $\rho=k_DL_N/(k_NL_D)$ on the page; unrelaxed, it diverges when $\rho>1$. A piece that would float as the Neumann side (nothing of its own sets its level) takes the Dirichlet side. Explicit (sound): the pieces trade the interface's pressures and velocities once a step, and are the full domain to the bit | the same discretization on every cell |
| D | a field's boundary integral is a lumped part's port variable: the electrodes' currents drive a circuit (nodal analysis; ideal and Norton batteries), and the circuit's potentials hold the electrodes | the plate and the circuit as one sparse system |
| split | one mesh, two physics: a conduction agent and an elasticity agent with the whole temperature field crossing between them, run synchronously or lagged a step on two threads | the unsplit solver |

`fv.py` is one finite-volume core for conduction, the electric potential and the families to come. It has harmonic-mean faces (exact for a wall of layers), upwind advection, and any set of cells. A face that leaves the set is closed in one of three ways (`neighbour`, `dirichlet`, `neumann`), and those three are what the styles differ in. The full domain is the same call on every cell, so **a one-window decomposition is the full-domain system to the bit**, and a test checks it steady and transient.

`fe.py` is the structural families' core: Q1 elements with a material per element, the build repo's `ThermoStruct2D` element (2 x 2 Gauss points, its shape functions and plane-stress matrix). On one material its stiffness, conduction and mass matrices are that solver's own to 1e-12, and its thermal load gives that solver's clamped displacement to 1e-10; a test checks both when the build repo is present.

**At showcase sizes the full domain is faster** for every family but the wind farm. On 4,000–29,600 cells a sparse direct solve, or one explicit step on the whole grid, beat every decomposed arm in the served page on AC power, by 1.5× (the sound's two pieces) to 520× (plain Schwarz on the bracket, serially, 616 iterations). The heated strip's three arms tie within the machine's noise, about 10%. These cases show the coupling, not a speedup, and the timing table gives the ratio either way.

## The compile (`compile.py`)

**Compile** (step 4, or *Run > Compile with the Atlas compiler*) puts the case as it stands to `atlas/compiler.py`, in a background thread. The case is copied first, so an edit made while it compiles cannot reach it. No run and no compile start while the other is active, because each would be timed with the other. The record goes beside the case file, in `<case>.results/compile-<time>.json`: every decision, the compiler's report, and the case as compiled.

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
- `admit-uncertified`: the three farms, the plume, the plate on a circuit, the sound and the cooled block;
- `refuse` at R10: the two-layer wall, the insert and the bracket. Each of their seams is admitted;
- refused before the compiler: the heated strip.

No case can earn a plain `admit`: the master bound's constants are unmeasured on every graph (W56). R10 refuses two embedded agents of one physics that cut its region. It does not read that styles B and C iterate until the pieces agree, and the Run step measures them within $3.6\times10^{-9}$ of the full domain or closer. The page says so beside the refusal and does not change it. The same compile records derive a direct Schur solve for the interface (`K_accelerator = direct-schur`), and R10, at L2, runs before that scheme exists. The question is open as W348 in the vault's gap worklist.

## The showcase gallery

*File > Showcase gallery* lists every example with its description, family and style, and an **Open** button. Every example was opened, checked, compiled and run in the served page on 2026-09-29. `wiki/concepts/Atlas 0.1/atlas-0.1-outcome/showcase-gallery.md` has a row per case: its style, port type, compile verdict, its two checks' measured values against their tolerances, and its runtime. The records are in `out/workbench/records/step-f/`. Every run finished under 2 minutes, the longest in 70 s. The 21-rotor farm's compile takes 95 s, so its compile plus its run is 165 s, which the owner accepted on 2026-09-29, since the compile is its own step. Each row is labelled a showcase, not a research record.

## The geometry section

The hybrid the owner chose on 2026-09-28: rectangles on the grid are drawn in the
page; anything else comes from Gmsh. **Superseded in part on 2026-09-29**: any
shape with straight, arc or spline edges is now drawn in the page too (see
"Drawn shapes" below), and Gmsh stays the path for a mesh made elsewhere.

**One canvas, six layers, one editable at a time.**

| layer | on the canvas | in the table |
|---|---|---|
| **Domain** | the whole grid until an outline is drawn (Draw shape), with holes cut in it; Reshape moves its vertices and bends its edges; *Reset to the whole grid* | the outline and each hole: vertices, edges by kind, cells |
| **Windows** | Move & draw: drag to move, Shift+drag to draw, click then Backspace to delete. Reshape: drag a corner handle, or a drawn window's handles. Draw shape: a window of any shape. *Generate a tiling* fills the domain with rows x columns at a chosen overlap | id, shape, position, size, cells; the Edges table for drawn ones |
| **Regions** (materials) | the same, for rectangles; imported polygons (with holes) are shown but not moved. Regions **stack in list order**: a later one takes the cells it covers, so a plate with an insert needs no cutting | id, material, stacking order |
| **Devices** (rotors) | click to add, drag to move, click then Backspace to delete | id, position, diameter, yaw |
| **Boundaries** | drawn along the domain's edges, coloured by kind | edge, segment, kind, value. Read-only when the family's solver fixes its boundary (the wind farm, the river and the acoustics do) |
| **Circuit** (lumped parts) | a schematic: each electrode is a node beside its edge segment, the circuit's other nodes sit in a row below the domain, and each battery or resistor runs between its two nodes | id, kind, value, a battery's internal resistance, its two nodes; *Add battery*, *Add resistor*. Offered only to a family that reads a circuit |

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

**Gmsh.** *File > Import geometry from Gmsh (.msh)*. Name physical groups
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

**On the canvas** (Domain, Windows and Regions layers):

| tool | what it does |
|---|---|
| **Draw shape** | click to place vertices, double-click (or *Close shape*) to close. Edges start straight or smooth. *Circle*, *Ellipse* and *Hexagon* add a ready-made shape. On the Domain layer the shape is the outline or a hole |
| **Reshape** | drag a **square** to move a vertex, a **circle** to bend its edge into an arc (back onto the line to straighten it), a **diamond** to move the whole shape. Double-click a circle to add a vertex, a square to smooth the curve through it (again for a corner). Backspace on a square removes the vertex; on a diamond, the shape |

The **Edges** table sets any edge's kind or bulge. What lies outside a drawn domain
is drawn pale and dotted, stepped as the solver sees the domain's cells, with the
exact outline over it.

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
- both compile with every seam admitted, and are refused at R10 alone, as their
  rectangular counterparts are (W348).

## Code

| file | role |
|---|---|
| `spec.py` | the case file (`atlas-workbench/case@0.4`; 0.1, 0.2 and 0.3 files migrate on load), `check()` with each family's rules, and the thirteen examples: three farms read from `atlas/cases/scaling_ladder.py`'s real tilings, one or two per other family, and two drawn ones (the bend and the round insert) |
| `shapes.py` | drawn shapes: vertices joined by lines, circular arcs (by their bulge) or centripetal Catmull-Rom splines; sampling, crossing and area checks, and the edits the canvas makes (move, bend, split, remove, smooth) |
| `geometry.py` | the geometry rules as plain functions: snapping, tilings, the full-weight analysis, region masks and stacking; for drawn shapes, their cell masks, the distance ramp, the analysis on masks, and a drawn domain's boundary faces labelled by edge |
| `editor.py` | the Geometry step: the canvas, its tools (Move & draw, Draw shape, Reshape, Pan & zoom), its tables (with a Domain table and an Edges table) |
| `gmsh_import.py` | `.msh` to case geometry, by physical-group names |
| `registry.py` | the physics families: what each runs on, which layers it reads, which boundaries it can impose, and what is missing before the workbench can run it |
| `app.py` | the GUI. Every menu item and button goes through `Workbench.dispatch(action)` |
| `tiling.py` | any rectangles as a partition of unity, certified by `GridPartitionOfUnity` |
| `runner.py` | the run: arms in turns, the timer, Stop, the snapshots, the record |
| `runview.py` | the Run & compare and Results steps |
| `checks.py` | a check as data: what is measured, its tolerance, when it was registered |
| `machine.py` | the machine's state beside every timing, and the keep-awake request |
| `families/windfarm.py` | the wind-farm family's adapter (style A) |
| `fv.py` | the finite-volume core: harmonic-mean faces, upwind advection, any set of cells, three ways to close a cut face |
| `styles.py` | styles B, C and D as drivers: restricted additive Schwarz, Dirichlet–Neumann with Aitken, a field on a lumped network |
| `families/conduction.py` | heat conduction in several materials (styles B and C, steady or transient) |
| `families/electric.py` | a resistive plate on a battery-and-resistor circuit (style D) |
| `fe.py` | Q1 finite elements with a material per element: plane-stress stiffness, conduction and mass, the thermal load, rigid-body modes, stresses |
| `families/plume.py` | a pollutant plume down a river of two reaches (style A, one explicit step per exchange) |
| `families/acoustics.py` | sound through two media on a staggered grid (style C, explicit) |
| `families/elasticity.py` | a two-material bracket under load (style B on a structure) |
| `families/thermoelastic.py` | a heated bimetal strip, conduction and elasticity split by physics |
| `families/cooling.py` | a heated block cooled by a channel flow (style C at a seam between two physics) |
| `compile.py` | the compile: a case's `CaseGraph` through its family's `case_graph`, the finite-volume window as an agent (`FVAgent`), seams from the geometry with Fourier prolongations, the declared cross-points, the verdict per seam, and the background job with its record |
| `tests/test_workbench_shell.py`, `tests/test_workbench_geometry.py`, `tests/test_workbench_runner.py` | the generator against the measured tilings, the full-weight rule against the assembly's own weights, the Gmsh path on a real mesh, the canvas driven through the data Bokeh's tools send, every menu action (nothing is left unbuilt), and the runner: its three arms against W346's own columns bit for bit, the one-window control, Stop, the record with its field differences, every decomposed arm's farm metrics, and the stale-case labels |
| `tests/test_workbench_fv.py`, `tests/test_workbench_families.py` | the finite volumes against closed forms (a layered wall, a series circuit), one window equal to the full domain to the bit, styles B, C and D against the full domain, the unrelaxed Dirichlet–Neumann diverging exactly when $\rho>1$, both new families end to end, and the page's family, physics and materials controls |
| `tests/test_workbench_cases.py` | the five step-C families: each one-window (or one-piece) control, each registered check, their positive controls (no interface reflects nothing; a plain strip heated uniformly carries no stress), `fe.py` against `ThermoStruct2D`, the floating-piece rule, the circuit layer, and each family in the page. And the compile: every example's verdict pinned, the split refused by the port vocabulary, the farm through its own graph (and a moved window refused), a seam's response rising with its trace, the Check step's per-seam table, and the gallery opening every example |

## Known limitations

- **Light theme only, whatever the OS is set to.** Panel's theme switches reload the page, which starts a new session and drops the open case.
- Undo covers case edits, not view or tool changes.
- **Regions are read by every family but the wind farm**, which ignores them, and the check says so.
- Timings are taken inside the workbench server process, which also serves the page; the page's updates share the process with every arm alike.
- **The circuit is drawn, not dragged**: its parts are set in the table and the canvas draws the schematic.
- **At showcase sizes the decomposed arms are slower than the full domain** for every family but the wind farm (above). This is measured and shown, not hidden.
- **Plain Schwarz has no coarse level**, so it converges slowly on a bending structure (616 iterations on the bracket); a coarse space or a Krylov wrapper would be the remedy, and neither is built.
- **The wind farm compiles only on a scaling-ladder rung.** Its graph is the vault's `scaling_ladder.build`, which declares a regular tiling. A drawn tiling that is not a rung runs, but is refused before the compiler, with that reason.
- **Drawn shapes run on one family so far: heat conduction.** The others refuse a drawn domain or window in the check, each with its reason (`registry.Family.drawn_why`): the wind farm's window solver marches rectangles; the plume's river flow is given per reach rather than solved; the sound's leapfrog runs two full-height pieces; the two structural families' elements cover the whole grid; the cooled block's coolant fills whole rows; the plate on a circuit reads its electrodes on the grid's four edges.
- **A drawn curve is resolved to the grid's cells.** The solvers see the cells whose centres a shape contains, so a curve is a staircase at the cell size. The canvas draws both, and on the bend the staircase moves the ring's heat flow by -0.97% from its continuum value. Body-fitted grids would remove that; they are not built.
- **The 21-rotor farm's compile takes about 95 s** (124 seams, each probed through its agents' solves), so its compile plus its run is about 165 s. Each is under 2 minutes, but together they are not. The owner accepted this on 2026-09-29: the compile is its own step, and the two-minute rule is per run.
- **R10 refuses the iterated one-physics cases** (styles B and C: the wall, the insert, the bracket), because it does not read whether a coupling iterates. The refusal is shown as the compiler gave it. Open as W348 in the vault's gap worklist.
- Bokeh gives each gesture to one tool, so moving and resizing are two tools, not one.
