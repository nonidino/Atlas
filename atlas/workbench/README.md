# Atlas Workbench

Build a domain decomposition by hand, run it, and compare it with the full-domain
solve. Built so far: the menus, the workflow, the case file, and the **geometry
section**. The runner is plan step 3 of
`wiki/concepts/Atlas 0.1/atlas-0.1-outcome/outcome-c4-path-to-declarative-cases.md`;
the library of cases it will run is `showcase-library-plan.md` in the same folder.

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
| menus | header | **File** (new, three examples, open, save, save as, import and export JSON, import geometry from Gmsh), **Edit** (undo, redo, generate a window tiling, revert), **View** (grid, overlaps, labels, activity log), **Run** (check; compile and run show why they are not built yet), **Help** |
| workflow | sidebar | six steps, each labelled with its status (`ok`, `N errors`, `not built`), and a live case check |
| case bar | top of the workspace | which case is open, and whether and where it is saved |
| workspace | main | the active step |
| activity | bottom | a timestamped log of every action |

The six steps:
1. **Case**: name, description, physics family, domain size. Works.
2. **Geometry**: the canvas and the four layers. Works; see below.
3. **Physics & coupling**: viscosity, freestream, macro-step, ramp width, assembly, pressure solve. Works.
4. **Check**: every rule, with where to fix it. Compile with the Atlas compiler is visible and disabled, with its reason.
5. **Run & compare**: settings, arms, and three field panels. Disabled, with its reason.
6. **Results**: the table a run will fill. Placeholder.

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
| `spec.py` | the case file (`atlas-workbench/case@0.2`; 0.1 files still load), `check()`, and examples read from `atlas/cases/scaling_ladder.py`'s real tilings |
| `geometry.py` | the geometry rules as plain functions: snapping, tilings, the full-weight analysis, region masks and stacking |
| `editor.py` | the Geometry step: the canvas, its tools, its tables |
| `gmsh_import.py` | `.msh` to case geometry, by physical-group names |
| `registry.py` | the physics families: what each runs on, which layers it reads, which boundaries it can impose, and what is missing before the workbench can run it |
| `app.py` | the GUI. Every menu item and button goes through `Workbench.dispatch(action)` |
| `tests/test_workbench_shell.py`, `tests/test_workbench_geometry.py` | 48 tests, including the generator against the measured tilings, the full-weight rule against the assembly's own weights, the Gmsh path on a real mesh, and the canvas driven through the data Bokeh's tools send |

## Known limitations

- **Light theme only, whatever the OS is set to.** Panel's theme switches reload the page, which starts a new session and drops the open case.
- Undo covers case edits, not view or tool changes.
- **Regions are only read by the planned conduction family** (showcase case 2). The wind-farm family ignores them, and the check says so.
- **Only the wind-farm family can be chosen**, so boundaries are fixed in practice; the editable boundary table and its rules are there for the families that come next.
- Lumped attachments (a circuit wired to an edge, showcase case 5) are not in the schema yet.
- Bokeh gives each gesture to one tool, so moving and resizing are two tools, not one.
