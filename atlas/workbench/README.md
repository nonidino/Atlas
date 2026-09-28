# Atlas Workbench

Build a domain decomposition by hand, run it, and compare it with the full-domain
solve. This is the **shell**: the menus, the workflow, and the case file. The
geometry editor is being designed with the owner next, and the runner is plan step
3 of `wiki/concepts/Atlas 0.1/atlas-0.1-outcome/outcome-c4-path-to-declarative-cases.md`.

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
notifications, and a plot canvas whose `BoxEditTool` draws, drags and deletes
rectangles.

**Nothing is downloaded, and the page makes no outside request.** The page uses
Panel's **Bootstrap** template. The Fast template was tried first and dropped for two
reasons. Its web components follow the operating system's dark mode on their own
while the page around them stays light (dark, magenta-bordered inputs and invisible
sidebar text on a dark-mode PC). And its design loads a font from Google. A test pins
both properties of the replacement.

## Layout

| part | where | what it does now |
|---|---|---|
| menus | header | **File** (new, three examples, open, save, save as, import, export), **Edit** (undo, redo, revert), **View** (grid, overlaps, labels, activity log), **Run** (check; compile and run show why they are not built yet), **Help** |
| workflow | sidebar | six steps, each labelled with its status (`ok`, `N errors`, `not built`), and a live case check |
| case bar | top of the workspace | which case is open, and whether and where it is saved |
| workspace | main | the active step |
| activity | bottom | a timestamped log of every action |

The six steps:
1. **Case**: name, description, physics family, domain size. Works.
2. **Geometry**: a read-only preview of the domain, its windows, their overlaps and the rotors, with tables of both. **The editing tools are the next piece, designed together.**
3. **Physics & coupling**: viscosity, freestream, macro-step, ramp width, assembly, pressure solve. Works.
4. **Check**: the structural check. Compile with the Atlas compiler is visible and disabled, with its reason.
5. **Run & compare**: settings, arms, and three field panels. Disabled, with its reason.
6. **Results**: the table a run will fill. Placeholder.

## Code

| file | role |
|---|---|
| `spec.py` | the case file (`atlas-workbench/case@0.1`, pydantic), `check()`, and examples read from `atlas/cases/scaling_ladder.py`'s real tilings |
| `registry.py` | the physics families, each with what it runs on and what is missing before the workbench can run it |
| `app.py` | the GUI. Every menu item and button goes through `Workbench.dispatch(action)` |
| `tests/test_workbench_shell.py` | 16 tests, including that every menu item reaches a known action |

## Known limitations of the shell

- **Light theme only, whatever the OS is set to.** A dark theme is a polish item. Panel's theme switches reload the page, which starts a new session and drops the open case.
- Undo covers case edits, not view toggles.
- The structural check does not yet enforce overlap-against-ramp or cross-point rules. Those belong to the geometry section.

## Open decisions for the geometry section

1. **Snapping:** to single cells, or to a coarser step (e.g. 16 cells, half the overlap)?
2. **Creating windows:** draw freely, generate a tiling (rows, columns, window size, overlap) and then adjust it, or both?
3. **Resizing:** Bokeh's `BoxEditTool` draws, drags and deletes but has no resize handles, so exact sizes would be typed in the table. Plotly (MIT, also installed) has editable shapes with resize handles, if that matters more.
4. **Overlap rules:** enforce at least twice the ramp where windows meet, or warn and let the check refuse it?
5. **Devices:** place rotors by clicking, or from a table; which parameters (diameter, yaw) appear on the canvas.
