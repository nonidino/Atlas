# Atlas Workbench

Build a domain decomposition by hand, run it, and compare it with the full-domain
solve. Built so far:
- the menus, the workflow, the case file and the **geometry section**;
- the **runner**, with three physics families: the wind farm (style A), heat
  conduction in several materials (styles B and C) and a resistive plate on a
  circuit (style D).

The runner is plan step 3 of
`wiki/concepts/Atlas 0.1/atlas-0.1-outcome/outcome-c4-path-to-declarative-cases.md`.
The library of cases it runs is `showcase-library-plan.md` in the same folder.

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
| menus | header | **File** (new, six examples, open, save, save as, import and export JSON, import geometry from Gmsh), **Edit** (undo, redo, generate a window tiling, revert), **View** (grid, overlaps, labels, activity log), **Run** (check; run decomposed, full, or the ticked arms; stop; show the results; compile says why it is not built yet), **Help** |
| workflow | sidebar | six steps, each labelled with its status (`ok`, `N errors`, `not built`), and a live case check |
| case bar | top of the workspace | which case is open, and whether and where it is saved |
| workspace | main | the active step |
| activity | bottom | a timestamped log of every action |

The six steps:
1. **Case**: name, description, physics family, domain size. A family that cannot run yet is listed and disabled. Works.
2. **Geometry**: the canvas and the four layers. Works; see below.
3. **Physics & coupling**: the family's own parameters (from the registry), the run mode (steady or transient), the macro-step, the coupling style and its settings (ramp width; tolerance and iteration cap; first relaxation factor, Aitken, and which piece takes the Dirichlet side), and the materials table, editable or picked from the family's library. Works.
4. **Check**: every rule, with where to fix it. Compile with the Atlas compiler is visible and disabled, with its reason.
5. **Run & compare**: works for all three families; see below.
6. **Results**: each arm's time per step and its ratio to the full domain, the family's metric, the field difference, and the sanity checks against their registered tolerances, from the run's record.

## The runner

`runner.py` marches a **committed** case: Run copies the case, so an edit made while it marches cannot reach it, and the page names the version that is marching. Every macro-step each ticked arm takes one step, in an order that rotates by one arm per step. The wind farm's arms:

| arm | what runs |
|---|---|
| Decomposed, serial | the windows stepped as one batch per window shape (W346's `E`) |
| Decomposed, parallel | the same windows split across threads, every chunk at the whole tiling's sub-step count, so it is **bit for bit** the serial arm (checked every step; W346's `Ep`) |
| Full domain | the same discretization on the undivided domain (W346's `F`) |

The other families have the same three arms where their style allows them. A style-C case has no parallel arm: Dirichlet–Neumann is sequential by construction. The Run step names every arm it drops, and why. A steady case repeats the same solve from the initial state, so its "macro-step" is a **timed repeat**, and the page says so.

It runs in a background thread; the page reads its progress every 400 ms and draws the fields, their difference, the time per step, the family's series (farm power; heat in), and, for styles B, C and D, the iteration's convergence curve. **Run > Stop** stops after the arm-step in progress and keeps whole macro-steps only. One run at a time per server, because two runs on one machine would time each other.

**What the timer holds:** the family adapter's `step` and nothing else. Diagnostics, the bitwise comparison, the snapshot and the record are taken after it stops.

**The record** goes beside the case file, in `<case>.results/<time>.json`: the case as it marched, every per-step time, the metrics, each check with its tolerance and registration date, and the machine's state before and after (power source, other Python processes; `machine.py`).

**The family adapters** are in `families/`, one per family, named by the registry (`registry.Family.adapter`). Every check below was registered before its family's first run, and the record carries the date.

| family | adapter | styles | its checks |
|---|---|---|---|
| wind farm | `families/windfarm.py`: W346's march, lifted, on any rectangles (`tiling.RectangleTiling`, whose weights are `ArrayTiling.weights` to the bit on the measured tilings) | A | incompressibility closes to 1e-9 in each arm's enforcing operator; farm power within 25% of the full domain (W346's B5); threaded equal to serial bit for bit (W346's B1) |
| heat conduction | `families/conduction.py` on `fv.py` | B, C | the energy balance to 1e-6; the full domain to 1e-6 of the temperature span; a steady wall of layers against its closed form $q=\Delta T/\sum_i L_i/k_i$ to 1e-6; style B threaded equal to serial bit for bit |
| resistive plate on a circuit | `families/electric.py` on `fv.py` | D | Kirchhoff's current law at every node to 1e-6; the batteries' power equal to the heat in the resistors, inside the batteries and in the plate, to 1e-6; the joint system (plate and circuit as one sparse solve) to 1e-6 |

## The coupling styles (`styles.py`)

The solver-family interface was derived from two real families, the wind farm and conduction, not designed ahead of them. `styles.py`'s docstring tabulates what fills each slot in each family: the state, restriction, a step given boundary traces, the full-domain equivalent, and assembly with a balance.

| style | what couples the pieces | the full-domain arm |
|---|---|---|
| A | overlapping windows, one exchange per macro-step (the wind farm) | the same scheme on the undivided domain |
| B | overlapping windows iterated to a tolerance: restricted additive Schwarz, a sparse LU per window, blended in window order, so threads change no bit | the same finite volumes on every cell |
| C | two pieces meeting at an interface, iterated Dirichlet–Neumann. The first relaxation factor is stated, with Aitken's rule optional, and the page shows the 1-D contraction factor $\rho=k_DL_N/(k_NL_D)$: unrelaxed, the iteration diverges when $\rho>1$ | the same finite volumes on every cell |
| D | a field's boundary integral is a lumped part's port variable: the electrodes' currents drive a circuit (nodal analysis; ideal and Norton batteries), and the circuit's potentials hold the electrodes | the plate and the circuit as one sparse system |

`fv.py` is one finite-volume core for conduction, the electric potential and the families to come. It has harmonic-mean faces (exact for a wall of layers), upwind advection, and any set of cells. A face that leaves the set is closed in one of three ways (`neighbour`, `dirichlet`, `neumann`), and those three are what the styles differ in. The full domain is the same call on every cell, so **a one-window decomposition is the full-domain system to the bit**, and a test checks it steady and transient.

**At showcase sizes the full domain is faster** for B, C and D. On 6,400–15,360 cells, one sparse direct solve beat every decomposed arm in the served page, on AC power. These cases show the coupling, not a speedup, and the timing table gives the ratio either way.

## The geometry section

The hybrid the owner chose on 2026-09-28: rectangles on the grid are drawn in the
page; anything else comes from Gmsh.

**One canvas, four layers, one editable at a time.**

| layer | on the canvas | in the table |
|---|---|---|
| **Windows** | Move & draw: drag to move, Shift+drag to draw, click then Backspace to delete. Resize: drag a corner handle. *Generate a tiling* fills the domain with rows x columns at a chosen overlap | id, position, size |
| **Regions** (materials) | the same, for rectangles; imported polygons (with holes) are shown but not moved. Regions **stack in list order**: a later one takes the cells it covers, so a plate with an insert needs no cutting | id, material, stacking order |
| **Devices** (rotors) | click to add, drag to move, click then Backspace to delete | id, position, diameter, yaw |
| **Boundaries** | drawn along the domain's edges, coloured by kind | edge, segment, kind, value. Read-only when the family's solver fixes its boundary (the wind farm does) |

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

## Code

| file | role |
|---|---|
| `spec.py` | the case file (`atlas-workbench/case@0.3`; 0.1 and 0.2 files migrate on load), `check()`, and the examples: three farms read from `atlas/cases/scaling_ladder.py`'s real tilings, two plates and a plate on a circuit |
| `geometry.py` | the geometry rules as plain functions: snapping, tilings, the full-weight analysis, region masks and stacking |
| `editor.py` | the Geometry step: the canvas, its tools, its tables |
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
| `tests/test_workbench_shell.py`, `tests/test_workbench_geometry.py`, `tests/test_workbench_runner.py` | the generator against the measured tilings, the full-weight rule against the assembly's own weights, the Gmsh path on a real mesh, the canvas driven through the data Bokeh's tools send, and the runner: its three arms against W346's own columns bit for bit, the one-window control, Stop, the record |
| `tests/test_workbench_fv.py`, `tests/test_workbench_families.py` | the finite volumes against closed forms (a layered wall, a series circuit), one window equal to the full domain to the bit, styles B, C and D against the full domain, the unrelaxed Dirichlet–Neumann diverging exactly when $\rho>1$, both new families end to end, and the page's family, physics and materials controls |

## Known limitations

- **Light theme only, whatever the OS is set to.** Panel's theme switches reload the page, which starts a new session and drops the open case.
- Undo covers case edits, not view or tool changes.
- **Regions are read by the conduction and electric families.** The wind-farm family ignores them, and the check says so.
- Timings are taken inside the workbench server process, which also serves the page; the page's updates share the process with every arm alike.
- **Lumped attachments** (a circuit wired to electrodes, showcase case 5) are in the schema since 0.3 and checked, but the page has no layer or table for them yet: the circuit is set in the case file.
- **At showcase sizes the decomposed arms of styles B, C and D are slower than the full domain** (above). This is measured and shown, not hidden.
- Bokeh gives each gesture to one tool, so moving and resizing are two tools, not one.
