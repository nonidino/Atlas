# The integrated Atlas UI — one app to draw, choose the physics, cut into pieces, run, and read the guarantees

**Type:** Concept page — **plan** (folder: `Atlas 0.1/atlas-0.1-proposal/demo/`)
**Status:** written 2026-11-16 from a planning chat with the owner. **Building, phase by phase** (`atlas/studio/`, its README says what each phase built): **P0 built** 2026-11-16; **P1 built** 2026-11-16, its install verified on fresh clones on Windows and on Linux (WSL) 2026-11-17; **P2 built** 2026-11-17. The owner's licence answers the same day: Paper.js 0.12.18, planegcs 1.3.0 and the OFL fonts vendored; Ranade never committed (its licence forbids a repository) and fetched at install; no LICENSE for Atlas yet; a general conformal chart to be built (§5). The design decisions in §0 are the owner's, made in that chat with questions and three rounds of canvas mockups. Every effort estimate is labelled **[AI Inference]**. After the UI, the project returns to Stage 0b of the chart operator: a learned coarse space $P_\theta$ on high-contrast CO-D ([[chart-operator-stage-0-rework]] §20.4). The UI must show those results too (§7).
**Hub:** [[00-proposal-workstreams]] · **Siblings:** [[demo-finish-plan]] · [[demo-fast-examples-plan]] · [[demo-learned-case-plan]] · **What it extends:** [[outcome-c4-path-to-declarative-cases]] (step 4, the GUI) · [[showcase-library-plan]] · [[showcase-gallery]] · **What it must show:** [[chart-operator-family-catalog]] · [[chart-operator-stage-0-rework]]

---

## 0. The request, and the decisions already made

The owner asked for one integrated app, a much better version of the workbench, in which a person can:
1. create geometry;
2. choose the equations and the regime;
3. decompose the domain and give each piece an expert;
4. run the case;
5. see the solution fields, the price and the guarantees.

**Decided in the planning chat** (each by the owner, by question):

| decision | the owner's choice |
|---|---|
| framework | **FastAPI and one JavaScript page** with a 2-D canvas over a WebSocket (§2). Panel is retired |
| phase 1 | **the new shell and the geometry**, for the existing eight families first; pieces and experts come after |
| look | today's layout (rail, canvas, inspector), with a **light and a dark theme** that switch without a reload |
| palette | `#E4E3EC` ground, `#191927` text and header, `#8E72EE` actions and selection, `#00D7D2` ready, guides and the conductor material |
| font | **Ranade** (Fontshare), with JetBrains Mono for numbers; Ranade's licence is checked before it enters the repository (§10) |
| wording | as few words as possible on screen; **every shorthand explains itself on hover** (G0–G3, TL0, H2r, det J, Fo, Pe, A/A, every gate) |
| geometry base | **Paper.js** for drawing, editing and booleans, and **planegcs** (FreeCAD's 2-D constraint solver, in WebAssembly) for dimensions and constraints. Not written from scratch (§3) |
| run | two modes, **single** (one window) and **compare** (two windows), each window choosing **full domain** or **chart operator**, a solver and a resolution (§6) |
| pieces | the chart method is a choice: **conformal, Laplacian or TFI** (§5) |

**The mockups.** The screens were drawn first on a private Claude Design canvas, at 1090×620:
- eight geometry boards (start, workspace, drag to draw, pen, combine, dimensions, materials, check);
- the physics, pieces, run (single and compare) and records screens;
- a dark-theme board.

The canvas is the owner's: <https://claude.ai/artifact/LhZGBYt6WQZLzRP5qGbVfh>. This page describes the screens in words, so it stands without the canvas.

---

## 1. What is wrong with the current workbench

The workbench was run on 2026-11-16 at 1090×620 (`python -m atlas.workbench`, port 8020, loaded with `?v=1`), and every control was clicked in the page. The findings below are **from the page**, not from tests.

**Defects seen:**
- **The tabs go out of step.** The header's Run switches the content to Run & results, but *Model* stays highlighted. Clicking *Model* then does nothing until *Run & results* is clicked first.
- **19 Bokeh errors** in the console ("reference p3693 isn't known"), from Panel rebuilding layouts. The server log showed none, so only the page reveals them.
- **Drawing drops clicks.** Five outline points were clicked; three landed. The double-click that closes the shape took about 3 s. Meanwhile two "place at least three points" notices appeared, because every click is a round trip to the server.
- **The canvas moves under the cursor.** The Draw sub-toolbar appears above the canvas when drawing starts and pushes it down mid-gesture.
- **The Case menu runs off the right edge** at 1090 px ("Import geometry from G…").
- **Notices stack over the inspector** and hide what they describe.
- **The Physics panel opens part-scrolled.**

**Awkward, or missing for the new work:**
- No regime numbers (contrast, Péclet, Fourier) and no guarantee class anywhere.
- The results table shows **raw seconds per arm** beside the ratio.
- The compile verdict (*admit-uncertified*) is a word in a card and is not drawn on the canvas.
- No per-piece error, no seam mismatch, and no window outlines on the field plots.
- **No four-sided pieces, charts, certificates, expert choice or chart-operator records.** On `bend-3`, the workbench's windows are overlapping blended windows. `atlas/chartop/benchmarks.py` rebuilds the same layouts as charted pieces separately, so the chart operator has never been in the page.
- **Geometry is tedious:**
  - every shape is a polygon clicked point by point;
  - holes and regions are separate modes;
  - there are no booleans, dimensions or constraints, and no typed sizes;
  - no starting shapes.

---

## 2. Architecture: FastAPI and one page, the engine kept

**What stays.** Everything below the page is kept and called as it is:
- `spec.py` (the case file);
- `families/` (the eight families);
- `compile.py`, `runner.py`, `checks.py`, `machine.py`, `starter.py`, `gmsh_import.py`;
- the whole of `atlas/chartop/`.

The new server is a thin layer over them. **Modules hash-pinned by registered evaluations are not edited:**
- the squares pin of 2026-11-08 and the two-level pin of 2026-11-10, which include `expert.py`, `coarse.py`, `twolevel.py`, `train.py` and the box scripts;
- new code goes in new modules.

**What goes.**
- `app.py`, `editor.py`, `inspector.py` and `runview.py`, the Panel and Bokeh page (about 5,200 lines).
- Their tests that drive Panel widgets are rewritten against the new protocol (§9).

**The new pieces:**

| part | what it is |
|---|---|
| `atlas/studio/server.py` | FastAPI: serves the page and a WebSocket. Every edit is a message: the server validates it through `spec.check`, applies it, and answers with the new case. Runs stream progress and fields as messages |
| `atlas/studio/static/` | one HTML page, plain JavaScript modules (no build step), the CSS tokens of both themes, and the vendored libraries (below) |
| `atlas/studio/geometry_eval.py` | the feature tree evaluated exactly with Gmsh's OpenCASCADE kernel, then turned into cell masks with Shapely (§3.3) |
| `atlas/studio/registry_eq.py` | the equation registry: each equation's properties, regime numbers, guarantee class and refusals, as data (§4) |
| `atlas/studio/records.py` | the chart-operator records reader, read-only (§7) |

**No npm at install time.** The JavaScript libraries are **vendored** into `static/vendor/`, each file pinned by sha256, with its licence notice beside it:
- Paper.js;
- planegcs (its `.wasm` and loader);
- the Ranade woff2 files, if their licence allows (§10).

The one-command install (`run.cmd`, `run.sh`, `run.py`) stays Python-only.

**The trade-offs that decided it:**

| | Panel (fix it) | Panel with a JS canvas | **FastAPI and one page** |
|---|---|---|---|
| install | works today | works; two UI systems | Python only, with the libraries vendored; FastAPI, uvicorn and websockets added to the pins |
| one-command bundle | unchanged | unchanged | `requirements.txt` and `constraints.txt` gain the server packages at the versions verified (dev: FastAPI 0.115.5, uvicorn 0.32.1, websockets 15.0.1) |
| drawing | a server round trip per click (§1) | local drawing, committed on finish | local drawing, snapping and booleans at frame rate |
| testability | widgets in shadow DOM; a click beats a typed value | both problems remain on the forms | **plain DOM**: `find` and `read_page` work; protocol tests without a browser |
| risk | low | medium | a rewrite of the page layer, **[AI Inference]** 3–5 tiers for phase 1 |

**Lessons from earlier demos that bind this one** ([[outcome-c4-path-to-declarative-cases]] §2, Step 4):
- rebuild on commit, not on drag;
- the page says which configuration is running;
- price is the ratio measured on the machine in front of the viewer;
- open the page and click every control.

---

## 3. Geometry

### 3.1 The model: a feature tree, not a polygon list

A case's geometry becomes a **feature tree** (case file 0.7; 0.6 files migrate as a plate built from their polygons). Each feature has a kind, its inputs, and named parameters:
- **sketch shapes:** rectangle, ellipse, arc, pen path (lines, arcs, Bézier curves), channel (a centre line with a width), imported curves;
- **operations:** union, subtract, intersect, split, pattern (line, circle, grid), mirror, fillet;
- **roles:** each shape **adds** to the domain, **cuts** a hole, or is a **region** inside it. One switch (Add, Cut, Region) sets the role of the next shape, and Alt flips it while drawing;
- **regions:** a material, or a coefficient field;
- **generators:** *scatter inclusions*: count, size range, material, minimum gap, shape (circles, channels, both) and a **seed**, so a layout is reproducible from its numbers.

**Parameters** are named values (W, H, r, d, w). A parameter can be a formula (`x_h = W − 0.15`). Any parameter can be marked for a **sweep**, which runs the case once per value.

### 3.2 The tools

All are in one tool rail, with keyboard shortcuts shown on hover:

| tool | how it works |
|---|---|
| **Select** (V) | click a shape; drag handles; Shift-click adds to the selection |
| **Pen** (P) | click for a corner, drag for a curve; Alt breaks a handle; Backspace removes the last point; click the first point to close; Enter leaves it open |
| **Rectangle** (R), **Ellipse** (E) | drag a box; Shift for a square or a circle; Alt from the centre |
| **Arc** (A) | start, end, then pull the bulge |
| **Channel** (C) | draw a centre line; it gets a width: the thin high-contrast strips Stage 0b needs |
| **Dimension** (D) | click an edge, a hole or two points; the size becomes a driving dimension |
| **Measure** (M) | read a distance or an angle without changing anything |
| **Pan** (H, or Space-drag), **Import** (I) | |

**While drawing:**
- **smart guides** snap to edges, centres and midpoints (cyan dashed lines with a diamond at the snap);
- live size labels follow the cursor, and **Tab types an exact size**;
- a tool **stays on** until V is pressed. The old one-shape-then-Select rule made five rotors five presses.

**Combine** is a floating bar over the selection: Union, Subtract, Intersect, Split, Pattern, Mirror. A preview hatches what will be cut, and the cutters stay editable in the feature tree.

**Start screen.** A new case opens on:
- starting shapes, each fully parametrised: plate, plate with hole, L-bracket, bend, S-channel, annulus, high-contrast, blank;
- a drop zone for STEP, DXF, `.msh` and an image to trace;
- the three held-out benchmarks as read-only shapes, each with an editable copy.

### 3.3 Two kernels, one truth

**The page.** Paper.js evaluates the tree in the page, for drawing, hit-testing and the previews; booleans are Paper.js's `unite`, `subtract`, `intersect` and `divide`.

**The server.** It evaluates the same tree **exactly** with Gmsh's OpenCASCADE kernel (Gmsh 4.15 is already a dependency): booleans, fillets, STEP import. Shapely 2.1 then turns the boundary into each region's cells. **The server's result is the truth.** Two kernels disagree at the round-off level, and Paper.js has known floating-point trouble with large coordinates and overlapping segments, so:
- coordinates are kept in metres near unit scale;
- the Check view compares the two: drawn area against area in cells, per region.

**Constraints.** planegcs solves the dimensions and constraints in the page:
- the constraints: parallel, perpendicular, tangent, equal, coincident, fixed;
- a **degrees-of-freedom count**, shown as "fully defined" when it is zero.

The solved values are written back to the feature tree's parameters, so the server never needs the solver. B-splines are not yet fully supported by planegcs, so the pen's curves are Bézier segments and arcs.

### 3.4 Materials and coefficient fields

- **Materials:** a palette of swatches (name, κ, ρc); click a region to fill it; *New* from the library or by name.
- **Coefficients:** constant, **formula** in $x$, $y$ (and $T$ for nonlinear families), or **image** (greyscale mapped onto a κ range).
  - Formulas are parsed by a small expression parser on the server (numbers, $x$, $y$, $T$, parameters, `+ − × ÷ ^`, `exp`, `log`, `sin`, `cos`, `tanh`, `min`, `max`), **never Python `eval`**.
  - A live preview shows the field's range.
- **Contrast:** $\kappa_{\max}/\kappa_{\min}$ is shown in the status bar and on Physics.

### 3.5 Check

Grid resolution $n$ is chosen (32, 64, 128, 256) and the geometry is drawn **as the solver sees it**: each region's cells. The checks:
- the outline is closed;
- no self-crossings;
- holes and regions are inside the domain;
- **thin features** (narrower than one cell);
- **slivers** (a gap narrower than one cell);
- **near-touching** edges.

Each note is a numbered pin on the canvas with one-click fixes: a finer grid, widen to one cell, or keep. *Keep* is right when the chart operator will resolve the feature on its own pieces. No note blocks a run unless `spec.check` refuses.

### 3.6 Imports

| format | path |
|---|---|
| `.msh` | the existing `gmsh_import.py` (physical groups `domain`, `region:<material>`, `bc:<kind>`), unchanged |
| STEP | Gmsh's OpenCASCADE importer; a 2-D section is taken |
| DXF, SVG | parsed in the page into pen paths, then normal features |
| image | shown under the canvas at a set scale, to trace with the pen |

---

## 4. Equations and regime

**The equation list** comes from the family catalog's groups ([[chart-operator-family-catalog]] §2). Each equation is one row, with:
- its type: elliptic, parabolic, hyperbolic or mixed;
- a **CO** tag where a chart-operator expert exists (CO-D only, today);
- its group's guarantee class (G0, G0′, G1, G2, G3; catalog §1.3).

An equation with no solver in Atlas is shown greyed, with the reason on hover.

**The equation screen:**
- the equation, typeset;
- steady or transient, with Δt, steps and the initial state;
- parameters per region;
- boundary conditions per edge, each condition explained on hover (Dirichlet, Robin with its formula, insulated).

**The properties panel** is data in `registry_eq.py`, one entry per equation:
- type, order, linearity, self-adjointness;
- the energy (for heat, $\tfrac12\int\kappa|\nabla T|^2-\int fT$);
- whether a maximum principle holds, what is conserved, the signal speed;
- which equations share its operator (heat, Darcy and electrostatics share CO-D's operator);
- where Atlas uses it.

**Regime numbers,** computed from the case:
- contrast $\kappa_{\max}/\kappa_{\min}$;
- the Fourier number $\mathrm{Fo}=\kappa\,\Delta t/(\rho c\,h^2)$;
- the Péclet number $\mathrm{Pe}=|u|L/\kappa$ for transport;
- the wavenumber $k$ for waves.

Each is shown against the selected expert's card envelope.

**What the UI refuses, and why.** Every refusal names its rule on hover:

| refused | why |
|---|---|
| a learned expert on a piece outside its card's envelope (contrast, Fourier number, chart quality) | the card's host rules (`atlas/chartop/card.py`) |
| H2r on a piece that is not a square of one material | `reuse.py`'s premise |
| a seam the compiler refuses (for example rule G1 at a cross-point) | the compiler's verdict, named |
| an equation with no Atlas solver | listed in the catalog, not built |
| NeuberNet in any solver list unless it is present locally | unlicensed, local only |
| the wind farm's compile on a drawn domain, sound's style A compile | today's documented refusals, unchanged |

---

## 5. Pieces and experts

**Chart method,** chosen per case or per piece:

| choice | built in `atlas/chartop/chart.py` |
|---|---|
| **TFI** (transfinite interpolation, the Coons patch) | `transfinite`; cheapest; can fold on strongly curved pieces |
| **Laplacian** (harmonic coordinates) | `harmonic` (one sparse solve) and `winslow` (the harmonic chart by Picard iteration) |
| **Conformal** | only where the map is known in closed form (`from_map`), or approximately as `harmonic` with a conformal boundary correspondence. A general numerical conformal map onto a square is **new work** |

The quality readout shows the smallest $\det J$ over the cells and the largest skew.

**Cutting:**
- **By hand:** draw a cut across a piece, merge two neighbours, and drag a chart corner.
- **Automatic:**
  - cut at material interfaces (so every piece has one material, as the catalog's Kirchhoff observation needs);
  - prefer squares (so H2r can reuse a response);
  - a target piece count.
- **[AI Inference]:** automatic cutting of an arbitrary drawn domain into four-sided pieces is a known hard problem (quad layout). The first version should cover the starting shapes and domains cut into strips, and say plainly when it cannot cut a domain, leaving the rest to hand cutting.

**Certificate.** Each piece shows Proposition 3's certificate (`certificate.py`): $\det J>0$ at every cell corner, and a simple, positively oriented boundary. The case shows *16 / 16 charts certified* or names the piece that fails.

**Expert per piece:**
- Classical;
- TL0, the untrained floor;
- the trained networks found in the weights registry (TL-s0, TL-s1, S′-s0), each checked by sha256 against its record before it is offered;
- H2r, on squares of one material.

*Best admitted for all* gives each piece the cheapest expert the compiler and the card admit.

**Seams and ports.**
- Every seam is drawn in its verdict's colour (admitted, admitted-uncertified, refused).
- Every side shows its port (`THERM` for heat).
- The verdicts are the compiler's own, as the workbench's compile card already reads them.

---

## 6. Run and visualisation

**Two modes:**
- **Single:** one window.
- **Compare:** windows A and B side by side, with one time slider and an optional shared colour scale.

Each window chooses:
- **method:** *full domain* (one solve, no pieces) or *chart operator* (pieces, an expert each);
- **solver:**
  - full domain: classical at $n$ = 32, 64, 128 or 256;
  - chart operator: *mixed* (as assigned on Pieces), or one expert on every piece;
- **resolution:** $n$, plus $n_c$ for the two-level experts.

**Views:**
- **Field**, with the pieces outlined;
- **Difference** (B − A);
- **Error per piece** against the undivided solve;
- **Seams:** the jump in $T$ and in flux across each seam;
- **Coarse basis:** the two-level expert's coarse functions, one at a time;
- **Modes** and **Convergence**;
- **Flux:** arrows of $-\kappa\nabla T$;
- a **probe**: click a point, see $T$ against time.

**Numbers:**
- **Price** is a ratio only: B over A in compare mode, against $n=256$ in single mode. It is timed in turns on this machine, with the A/A spread beside it and the machine's state recorded (`machine.py`). **No raw seconds on screen.**
- **Accuracy:** the energy-norm error of the worst piece, the seam mismatch and the balance.
- **Checks and gates:** pass or fail chips, each explained on hover.
- **Certificate:** the guarantee class, the dual bound where DUAL applies (`dual.py`), and the chart count.

---

## 7. The chart-operator records

A read-only **Records** screen reads `out/chartop/` and is never a way to re-run anything:
- the registered evaluations, newest first: Stage 0b (not registered yet), two-level, squares, hybrid, forms;
- for the selected one, its stamp, code commit and pin status;
- **its gates**, each a chip whose hover text is the gate's registered title and bar, read from `gates_*.py`;
- **its prices**, each bar labelled with its problem set (a geometric mean over all priced problems is never set beside a held-out-only number as if comparable);
- **$\rho$**, defined as $\rho = 1-\varepsilon_{TL}/\varepsilon_{TL0}$, the share of TL0's error the network removes (1 perfect; below 0 worse than TL0), as a histogram over the test pieces with its median and 10th percentile.

**Open** links each held-out problem into Run's compare mode.

**For Stage 0b** the same screen needs no new layout. The registration adds:
- a contrast-bin axis: $\rho$ by contrast;
- the price per arm against the classical solve.

The arms (TL0, $P_\theta$, classical, H2r where it applies) come from the registration file, not from code in the page.

---

## 8. Phasing

Each phase ends with a check **in the real page**, not only in tests:
- restart the server and load with `?v=N`;
- confirm the GET in the server log;
- click every control the phase added, at 1090×620 and in the dark theme;
- read the console: **zero errors**.

The general rules from earlier demos apply to every phase:
- [[demo-finish-plan]] §4's installer checks;
- the first scenario must run;
- verify the page, not the engine;
- lazy imports are undeclared dependencies;
- pin the verified environment.

**P0 — the shell**
- **Builds:**
  - the FastAPI server and the WebSocket protocol;
  - the header, the two themes and the hover explanations;
  - open and save of case 0.6 files (round trip bit for bit);
  - the bundle's pins (FastAPI, uvicorn, websockets, Shapely), the vendored libraries with their hashes, and the font decision.
- **Done when:** a fresh clone installs with one command on Windows and on Linux (WSL). The page opens with the network cut, every existing example opens and saves unchanged, and the installer's self-test opens the page itself, not just the engine.

**P1 — geometry I, and the first scenario**
- **Builds:**
  - the Paper.js canvas;
  - Select, Pen, Rectangle, Ellipse and Arc;
  - Add, Cut and Region; guides, snapping and typed sizes;
  - Combine (union, subtract, intersect); the feature tree, undo and redo;
  - materials; the Check view;
  - imports of `.msh`, STEP, DXF and SVG;
  - the server's exact evaluation and the Check view's drawn-against-cell-area comparison;
  - **a minimal Run** for the eight families (the existing runner, shown as cards and fields).
- **Done when:** choose a family, draw a shape with a hole by dragging (no point-by-point clicking needed), press Run, and see the result, with stated defaults and one-click fixes, all inside 1090×620. Every family's showcase example still opens and runs, its checks unchanged. The page check also drives the pen with real clicks and drags and confirms every click lands.

**P2 — geometry II**
- **Builds:**
  - planegcs dimensions and constraints, parameters and formulas;
  - Pattern, Mirror and Fillet;
  - coefficient fields (formula and image);
  - Channel and Scatter (seeded);
  - image tracing; sweeps.
- **Done when:**
  - a dimensioned plate reaches "fully defined";
  - changing W moves every dependent shape;
  - the same seed gives the same inclusions twice (byte for byte in the case file);
  - a formula field's κ range matches the server's evaluation.

**P3 — physics**
- **Builds:** the equation registry and screen, properties, regime numbers, guarantee class and refusals.
- **Done when:** every equation in the list either runs or says why not on hover, and every regime number is checked against a hand computation on one case.

**P4 — pieces and experts**
- **Builds:**
  - chart methods (TFI, Laplacian; conformal where available);
  - hand cutting and automatic cutting for strip-cuttable domains;
  - certificates;
  - the expert per piece with card checks;
  - the compiler's seam and port verdicts on the canvas.
- **Done when:**
  - the three held-out benchmarks, cut in the page, reproduce `benchmarks.py`'s pieces;
  - a coupled classical solve matches `host.monolithic` to round-off;
  - a piece made to fold is refused with its $\det J$ shown.

**P5 — run**
- **Builds:**
  - single and compare modes;
  - the chart-operator path through `host.Layout` and `Coupled`;
  - the views of §6, the price ratio with its A/A check, gates and the certificate.
- **Done when:** a compare run of TL-s0 against the classical full domain on `bend-3` shows the same accuracy numbers the two-level record holds for that problem. Any difference is explained (machine, threads) before it is shown.

**P6 — records, and Stage 0b ready**
- **Builds:** the Records screen of §7.
- **Done when:** each existing evaluation's gates and prices on screen match its JSON record, checked by a test that reads both. A Stage 0b registration file, once it exists, appears with no change to the page's code.

**[AI Inference]:** P0–P1 about 4–6 tiers, P2 2–3, P3 1–2, P4 3–5 (automatic cutting is the open part), P5 2–3, P6 1.

---

## 9. Testing

- **Engine and protocol tests** in Python (pytest):
  - every message the page can send, applied to a case and checked through `spec.check`;
  - the server's geometry evaluation against closed forms (areas, a plate minus a circle);
  - Paper.js and the server agree on a fixed set of shapes, within a stated tolerance.
- **Page checks** in the real page, with the built-in browser, for every phase (§8): plain DOM, so `find` and `read_page` reach every control without shadow-DOM walking.
- **Kept:**
  - the workbench's engine tests (families, runner, compile, starter);
  - the installer tests;
  - the chart-operator suite.
- **Replaced:** the tests that drive Panel widgets (`workbench_ui.py`), because their subject is gone.

---

## 10. Risks and open questions

- **Licences.**
  - **planegcs**' licence is listed as MulanPSL-2.0 on its npm listing; FreeCAD's PlaneGCS is LGPL-2.1. Confirm from the package's own LICENSE before vendoring.
  - **Paper.js** is believed MIT; confirm.
  - **Ranade's** Fontshare licence must allow its files in a public repository. If it does not, the installer fetches it at install time and the page falls back to Instrument Sans.
  - **The Atlas repository has no LICENSE file.** Vendoring third-party code means keeping their notices, and the owner may want to choose a licence for Atlas first.
- **Automatic quad cutting** of arbitrary domains is research-grade (§5). The plan does not depend on it.
- **Two geometry kernels** can disagree; the server is the truth, and the Check view shows the difference.
- **Pinned modules.** The registered evaluations hash `atlas/chartop` modules; the UI calls them and never edits them.
- **The machine.** Prices are taken in the server process that also serves the page, as today. A price shown in the page is a guide, and a registered price comes only from the registered protocol.

---

## 11. What this page does not decide

- The exact WebSocket message schema; P0 writes it.
- Whether sweeps run in the page or as a background job with a record (P2).
- Stage 0b's own registration: its arms, gates and contrast bins belong to the Stage 0b chat. §7 only makes room for them.

---

## See Also

- [[outcome-c4-path-to-declarative-cases]] — the four-step path this is step 4 of, and the earlier demos' lessons
- [[showcase-library-plan]] — the workbench's case library, kept as the examples
- [[showcase-gallery]] — what each example measured, the regression bar for P1
- [[demo-finish-plan]] — the header, the forked river, holes, and the installer this plan keeps
- [[demo-fast-examples-plan]] — the fast example per type, still offered
- [[demo-learned-case-plan]] — the learned wind farm, still offered read-only
- [[chart-operator-family-catalog]] — the equation groups and guarantee classes of §4
- [[chart-operator-stage-0-rework]] — TL0, TL and $\rho$, and §20.4's carry-over to Stage 0b
- [[chart-operator-architecture]] — the expert the Pieces and Run screens put in front of a person
