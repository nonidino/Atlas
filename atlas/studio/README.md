# Atlas studio

One app to draw a domain, choose its physics, cut it into pieces, run it, and
read the guarantees: the integrated UI of
`wiki/concepts/Atlas 0.1/atlas-0.1-proposal/demo/integrated-ui-plan.md`, which
replaced the Panel workbench's page (its modules were deleted on 2026-11-18, once
their tests drove this protocol instead). The engine under it is still
`atlas/workbench/`: the case file, the families, the runner and the compile.

```bash
python -m atlas.studio --open        # http://127.0.0.1:8021/
```

`--port`, `--host` and `--cases-dir` are optional; cases are saved to
`out/workbench/cases/`, the folder the workbench used, so a case saved in either
opens in the other. `--fetch-fonts` downloads Ranade from Fontshare and exits.

## What is built

| phase | what | state |
|---|---|---|
| P0 | the shell: FastAPI and one JavaScript page over a WebSocket; the header, the two themes, hover explanations; open and save of case files bit for bit; the vendored libraries pinned by hash; the bundle's pins and self-test | built |
| P1 | geometry I: case file 0.7's feature tree; the Paper.js editor (select, pen, rectangle, ellipse, arc, measure, pan); Add, Cut and Region; smart guides, grid snap, typed sizes; combine (union, subtract, intersect); materials; the start screen; imports of `.msh`, STEP, DXF and SVG; the server's exact evaluation and the Check view; a minimal Run; the one-command install serves the studio | built |
| P2 | geometry II: parameters and formulas; the Dimension tool and planegcs constraints with the degrees-of-freedom count; pattern, mirror, split and fillet; coefficient fields (formula, image); the Channel tool; seeded scatter (circles, channels, both); image tracing; sweeps from Run | built |
| P3 | physics: the equation registry (every PDE of the family catalog, its type, guarantee class, properties, and the solver that runs it or why none does); the equation screen; the regime numbers with their definitions | built |
| P4 | pieces and experts: automatic cutting (rectangle grids at material interfaces, polar strips, channel strips, the O-H layout around an insert), hand cutting (chords, merges, corners); TFI, Laplacian and a new numerical conformal chart; Proposition 3's certificates; an expert per piece, trained networks offered only when their sha256 matches their record; seam verdicts | built |
| P5 | run: Single and Compare on the pieces (full domain or chart operator, mixed or one expert on every piece), steady or marched; the field, difference, per-piece and seam views; price as a ratio with its A/A spread; E4's bar, balance and the certificate | built |
| P6 | records: every registered evaluation under `out/chartop/`, its gates with their registered titles and bars, its prices by problem set, ρ per test piece; Open into Compare; a new kind's registration appears with no change to the page | built |

## Geometry (P1)

**The tree.** A case's shapes are features (`document.py`, case file 0.7): a
rectangle, an ellipse, a path (lines, arcs by their bulge, cubic curves, the engine's
splines), or a combine of others. Each adds to the domain, cuts it, is a material
region inside it, or is a guide. The domain is what adds, minus what cuts; regions
stack in order. A 0.6 file opens as the features its own polygons are, and a tree
that adds nothing to them saves as 0.6 again, byte for byte.

**The truth is the server's** (`geometry_eval.py`). Shapely does the booleans and
every segment of the result is traced back to the side it came from, so the engine
receives the sides as drawn: a rectangle's four lines, a circle's four arcs by their
bulge, a pen curve as short lines. A boundary condition set on a side follows it
through later edits (a side cut in two carries it on both halves). All 27 examples
evaluate back to their own geometry, edge for edge and cell for cell. Exact areas
come from Gmsh's OpenCASCADE kernel (`occ.py`) when Gmsh is installed.

**The page** (`static/js/geo/`): everything moves locally at frame rate and is sent
as one edit when a gesture ends. A tool stays on until V. Smart guides snap to other
shapes' edges, centres and points (cyan, with a diamond); Shift squares a rectangle,
makes an ellipse a circle and locks the pen to 15 degrees; Alt draws from the centre,
or flips Add and Cut while drawing; Tab types a size. The pen: click for a corner,
drag for a curve (Alt breaks the handle), Backspace for the last point, click the
first point to close, Enter to leave it open as a guide, or type a length and Enter.

**Check** (K, `geocheck.py`): the cells the solver sees at n = 32, 64, 128, 256 or
the case's own grid, each region in its material's colour; the checks (closed, no
crossings, holes and regions inside); thin features, slivers and near-touching edges,
each a numbered pin with its fixes (a finer grid, widen to one cell, keep); and each
region's exact area against its cells' area.

**Start** (`starts.py`): plate, plate with hole, L-bracket, bend, S-channel, annulus,
high contrast (a seeded draw: the same seed gives the same layout) and blank, from a
width and a height; the three held-out benchmarks; recent files; a drop zone.

**Imports**: `.msh` through the workbench's Gmsh reader (it replaces the drawing),
STEP through OpenCASCADE in metres (a solid is cut at half height), DXF and SVG read
in the page (lines, arcs, circles, polylines with bulges, ellipses, splines, Bezier
paths), the outermost loops adding and the loops inside them cutting.

## Geometry (P2)

**Parameters** are named numbers or formulas of each other (`formula.py`, a small
parser: numbers, names, `+ - * / ^`, `exp log sqrt abs sin cos tan tanh min max`, pi
and e; never Python `eval`). A shape's field can be **bound** to a formula: type it in
the inspector (the *fx* mark says so) or let the Dimension tool write it. Before the
tree is evaluated, `resolve.py` gives every parameter its value, sets every bound
field and places every scatter's inclusions; a saved file holds both the drivers and
what they drove. `formula.js` is the same grammar for the page's previews; a test
runs both on the same formulas.

**Dimension (D)** clicks an edge of a rectangle (its width or height becomes `W`,
`H`, ...), a hole (its diameter `d`), a straight edge of a path (a driving length,
solved by planegcs) or two points (the second shape's position bound to the first's,
`cx = W - x_h`; Shift sets both directions). **Fix** pins a shape where it is. A
dimensioned size cannot be dragged by accident. The bar counts the **degrees of
freedom** with planegcs (`geo/gcs.js`): a rectangle has 4, a dimensioned and fixed
plate 0, "fully defined". A path's horizontal, vertical, parallel, perpendicular and
equal constraints hold while its points are dragged (planegcs solves each move).

**Operations:** Pattern (in a line, around a centre, in a grid), Mirror (about a
vertical or horizontal line) and Fillet (a tangent arc at a corner of two straight
edges) keep the shape editable; Split cuts one shape by another into what lies
outside and the overlap, a region to fill. **Channel (C)** draws a centre line that
gets a width: the thin strips Stage 0b needs. **Scatter** places inclusions from a
seed (count, size range, gap, shape, material, a drawn Area): NumPy's PCG64 in a fixed
order, so the same seed gives the same inclusions, byte for byte in the case file.

**Coefficient fields:** a region's coefficient is its material's constant, a formula
of `x`, `y` and the parameters, or an image whose grey levels map onto a range (log or
linear). The server bins the field into at most 16 materials, each level a set of
whole cells (`geometry_eval.field_regions`), and reports the range it found; the page
evaluates the same formula at the same cells and shows *same range* when they agree.

**Trace** puts a picture under the drawing at a typed width, on this page only.
**Sweep:** mark a parameter in the Parameters panel (from, to, runs); Run's *Run sweep*
runs the case once per value, each on its own copy, and lists agreement, price and
checks per value.

## Physics (P3)

**The registry** (`registry_eq.py`) is data: every PDE of the family catalog's groups
A to L, each with its type, its group's chart-operator family and guarantee class
(G0, G0′, G1, G2, G3), whether a chart-operator expert exists for it (CO: CO-D's
alone, today), its properties (order, linearity, self-adjointness, energy, maximum
principle, what it conserves, signal speed, which equations share its operator, where
Atlas uses it), and either the workbench kind that runs it or why nothing here does.
Eight of the 44 run; the other 36 are greyed, the reason on hover. Clicking one that
runs starts a case of its kind (Undo brings the old case back).

**The regime numbers** (`registry_eq.regime`, pushed with every state), each with its
definition on hover: the contrast of the materials in use; the Fourier number per cell
(κΔt/ρch²) and per piece (κΔt/ρc|Ω|, what the CO-D card's range [0.01, 100] reads,
marked ✓ or ✗ against it); the Péclet number per river reach (u = q/depth); the wind
farm's Re_D = UD/ν (lengths in rotor diameters: the referent's 255); and kh for sound,
k = 3/w for a Gaussian pulse of standard deviation w. A test checks each against a
hand computation (`tests/test_studio_physics.py`).

**The equation screen:** the equation with every symbol explained on hover; steady or
transient where the kind has both; each material's properties, editable; Δt, steps,
the kind's own parameters, threads and the coupling style; the boundaries grouped by
condition, each explained on hover.

## Pieces and experts (P4)

`pieces.py` cuts a heat case's domain into four-sided pieces. **Automatically**, for the
domains whose layout is a strip: a rectangle on a grid, cut again at every edge of a
rectangular region (one material per piece); an annular sector in polar strips at
equal log r and angle (log-polar charts, the exact conformal map); any four-sided
channel at equal values of its harmonic coordinate (conformal charts; end pieces by
Winslow where the CO-D card's envelope refuses them); a rectangle with one round
insert in the O-H layout. Anything else is refused with the reasons, and cut by hand.
These repeat `atlas/chartop/benchmarks.py`'s constructions on the case's own outline:
the three held-out benchmarks cut here are that module's pieces, chart for chart
(max |ΔX| = 0; tested). **By hand:** a cut across a piece runs on through every piece
across it (a chord, so seams stay conforming); a chord made by hand can be merged
away; an inside corner where straight cuts meet can be dragged.

**Charts** per piece: TFI (the Coons patch), Laplacian (Winslow) or conformal
(`conformal.py`, new: the conformal module and boundary correspondence from two
harmonic measures, with the log-polar map recognised where it applies), or the cut's
own. **Certificates**: Proposition 3 (`det J > 0` at every cell corner, a simple
positive boundary); the smallest relative `det J` and the largest skew are shown, and
a folded piece is refused with its `det J`. **Experts** (`experts.py`): classical, H2r
on squares of one material, TL0, and the trained networks TL-s0, TL-s1, S′-s0 and
S′-s1, each offered only when its checkpoint's sha256 equals its evaluation record's;
*Best admitted for all* gives each piece H2r where it applies, the two-level network
where its card admits it, else the classical solve. **Seams** are the host's own:
conforming seams admitted, a mortar admitted uncertified; every side's port is THERM.
*Check the coupling* solves the pieces with the classical expert against
`host.monolithic`: round-off with each side's whole trace, and the experts' 16-mode
floor beside it.

## Run on the pieces (P5)

`corun.py` runs Single (one window against the full domain) or Compare (windows A and
B) **in turns**: each window is the full domain (the pieces merged into one mesh) or the
chart operator (mixed experts, or one expert on every piece); steady, or marched from
each window's own previous field (up to 40 steps). It reports B's energy-norm error,
the worst piece's, the largest nodal error, the jump in T across seams and the heat
balance; E4's bar applied to this run (one frame of the registration's two); the
price as B's time over A's with A's own spread and the machine's state; and the
certificate. On bend-3, TL-s0 against the full domain gives the two-level record's
energy error, max error and smallest eigenvalue to every digit (tested, and in the
page).

## Records (P6)

`records.py` reads `out/chartop/` and never runs anything: each `evaluation-*.json`
and each `registration-*.json` without an evaluation yet, newest first; a kind's gates
from `atlas/chartop/gates_<kind>.py` (or, for a kind added later, its registration's
own list), their verdicts from wherever the record keeps them, its prices under their
problem sets, and ρ per test piece as a histogram. *Open* takes a held-out problem into
Compare with the record's network. A test reads both the JSON and the page's data.

## Page checks

`scripts/studio_drive.py WALK` drives the real page in headless Edge with **trusted**
input (the DevTools protocol's mouse and key events), checks what each step should
change, fails on any console error, and writes `out/studio/<walk>/walk.json` with
screenshots. The app's own browser pane cannot take input while it is hidden, and a
page check must not depend on a window being in front. `scripts/studio_walks/p2.py`
is P2's: 30 checks, among them the four done-when items; `p3.py` is P3's: 19;
`p4.py`, `p5.py` and `p6.py` check P4's, P5's and P6's done-when items the same way.

## How it works

**One page, one socket.** The server (`server.py`) serves `static/index.html`, its
plain JavaScript modules (no build step, nothing from npm) and the vendored files,
and holds one `Session` per open page. Every action the page takes is a message
over `/ws` (`protocol.py`): the server checks the message against the handler's
arguments, applies it through the session, and answers with a reply, after sending
what changed, the case with its problems and the geometry to draw (`state`).
Notices, run progress, fields and compile verdicts are pushed as they happen.

**The session** (`session.py`) is the retired workbench's `Workbench` without its
GUI library: a case is what it simulates (another kind starts a new case of it);
every edit is validated as a whole case and refused whole; windows that follow the
domain are cut again when it changes; a newly drawn shape or physics gets what it
lacks to run, each thing said; one Undo restores a whole case with its file and
key; a run is a commit, and the page says which version is marching.

**Edits are data.** `edit.apply` carries a list of operations on the case as JSON
(`set`, `insert`, `remove`, `move`, each with a path from the root); the result is
validated as a `CaseSpec` before anything is kept. A test drives every message
without a browser (`tests/test_studio_shell.py`).

**The price is a ratio.** A run's record keeps every per-step time; the page is
sent only what `runcards.py` derives from it: the full domain's mean step time over
each decomposed arm's, each arm's own spread (90th over 10th percentile, the A/A
noise), and the registered checks with their tolerances. No raw seconds reach the
page.

**The page reconnects to its own case.** A dropped socket or a reload in the same
tab asks for its session back (kept for an hour), so nothing is lost by a theme
switch, which never reloads anyway.

## The page

| file | role |
|---|---|
| `static/index.html` | the shell: the header (tabs, status chip, Run, theme, Case) and the five views |
| `static/css/tokens.css` | the two themes as tokens: the owner's palette, the dark values from the mockup canvas |
| `static/css/app.css` | the layout at 1090 x 620: 48 px header, 52 px rail, 8 px gutters |
| `static/js/net.js` | the socket: requests with replies, pushes, reconnection |
| `static/js/store.js` | what the page knows (catalog, state, run frames) and the edit helper |
| `static/js/ui.js` | the element builder, hover explanations, notices, menus, dialogs, the theme |
| `static/js/glossary.js` | every shorthand the page shows, explained (G0 to G3, TL0, H2r, det J, Fo, Pe, A/A, ...) |
| `static/js/header.js` | the tabs, the status chip and its problems list with one-click fixes, Run, the Case menu |
| `static/js/canvas.js` | a Paper.js stage in the physics' length unit: zoom, pan, fit, grid |
| `static/js/views/*.js` | the five views: geometry, physics, pieces, run, records |

## The vendored files

Paper.js 0.12.18 (MIT), planegcs 1.3.0 (LGPL-2.1), and the JetBrains Mono and
Instrument Sans fonts (OFL), each unmodified with its licence beside it and pinned
by sha256 in `vendor.py`; the root `THIRD-PARTY-NOTICES.md` lists them. **Ranade is
not committed**: its licence (ITF Free Font License, section 02) forbids distribution
through a repository, so `fonts.py` fetches it from Fontshare at install time and the
page falls back to Instrument Sans without it. The server writes the page's font CSS
from the files present, so a missing font is never a failed request.

## Tests

`tests/test_studio_shell.py`: every vendored file against its hash; the page and every
file it and its modules name served from this machine with the MIME types a browser
requires; no file of the page names the network; every message the page sends has a
handler, and every handler refuses arguments it does not take; edits, refusals, undo;
every example opened and saved bit for bit; a run over the socket ending in cards with
no raw seconds. The workbench's own tests (`tests/test_workbench_*.py`) drive the same
protocol on a `Session` (`tests/studio_driver.py`): every edit, refusal, undo, example,
run, compile and fix the retired Panel page's tests checked. The installer's self-test
(`run.py --check`) starts the studio's own server and runs a case over its WebSocket.

## Known limitations

- **A geometry edit waits for the engine.** On a 300 x 240 grid a drawn edit takes
  about 5 to 10 s: the engine cuts the windows again (a Fiedler vector and harmonic
  coordinates) and checks the case twice, exactly as the retired workbench did on
  every drawn edit. The page says *Rebuilding…* meanwhile and keeps drawing locally.
- **Not built in P5's views:** the coarse basis view, flux arrows and a probe; the dual
  bound is held by the registered evaluations' records and not computed in a run. The
  full domain in a run is the pieces' own mesh at the cut's resolution, not an
  independent grid of 32 to 256.
- **Automatic cutting covers strips** (a rectangle, a sector, a channel, a plate with
  one insert) and says so for anything else; the general quad layout is open.
- **Pieces belong to the session**, not the case file: a saved case is cut again.
- **NeuberNet is never listed**: it is not a chart-operator expert, and it is
  unlicensed and local-only.
- **Tangent constraints** are not offered: the pen's arcs are bulges, not planegcs
  arcs. A fillet is tangent by construction.
- **The server never runs planegcs.** The page solves a path's constraints and writes
  the solved points into the tree; a constraint the page did not solve (a file edited
  by hand) is kept but not enforced until the path is next edited in the page.
- **On Linux without libGLU** (WSL's Ubuntu, for one) Gmsh does not load: STEP import
  and the exact OpenCASCADE areas are unavailable, and the Check view's areas come from
  Shapely. The installer says so.
- **A cubic curve reaches the engine as short lines** (it has no cubic edge), so a
  boundary condition on a pen curve is held by each of those edges.
- **Case files are written in text mode** (`CaseSpec.save`, unchanged), so a file is
  CRLF on Windows and LF elsewhere; open-then-save is bit for bit on each.
