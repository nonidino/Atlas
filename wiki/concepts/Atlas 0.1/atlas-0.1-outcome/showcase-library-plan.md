# The classical showcase library: plan

**Type:** Outcome page — **build plan** (folder: `Atlas 0.1/atlas-0.1-outcome/`)
**Status:** written 2026-09-28 on the owner's request; **nothing on this page is built**. It extends [[outcome-c4-path-to-declarative-cases]] from one physics family to a library, and plans the workbench's geometry section to fit it. Runtimes and effort are estimates, marked **[AI Inference]**.
**Hub:** [[00-atlas-0.1-outcome]] · **Parent claim:** [[outcome-c4-modular-multiphysics]] · **Code:** `atlas/workbench/`

---

## 1. Why a library, and why it has been small

**The question (owner, 2026-09-28):** can't Atlas be used with any classical decomposed simulation, and why not expand the library with easy cases that show its diversity and run live in a couple of minutes?

**In principle, yes.** The compiler reads capability records, not solvers ([[atlas-implementation]]): what an expert declares about its ports, clock and boundary behaviour. Three things kept the library small, and none of them is a limit of the framework:

1. **No general runner.** Every real case study has its own hand-written time loop: `FSIRollout`, `GroundRollout`, `BrakeRollout`, `FrontWingRollout`, `RaceRollout`, `UnionRollout`, `CarUnion`, `wind_farm_design.Rollout`, and the rocket's `CoupledEpisode`. The generic `atlas/solve.py` runtime is used only by `thermal_seam`. So a new case has cost hundreds of lines of bespoke code.
2. **Every case so far was a research study,** with registered predictions, audits and follow-up tiers.
3. **Several cases were chosen to probe learned experts or theory gaps, not to show breadth.** The rocket is the extreme: combustion on millisecond scales, flight on minute scales, and reacting compressible flow ([[case-study-rocket-bc-seam-atlas-0.1]]).

**[AI Inference]:** once a general runner exists, a simple classical case costs a solver adapter (often 50–150 lines of numpy), a case file, and two sanity checks.

---

## 2. What qualifies as a showcase case

| criterion | why |
|---|---|
| **simple regimes:** one clock, or a small integer clock ratio; no shocks, no combustion | keeps the multi-clock problems out ([[case-study-vehicle-march-atlas-0.1]]: a 400× clock span) |
| **runs in under 2 minutes** on the laptop, ideally seconds, at about 100k cells or fewer | computed live in the showcase |
| **shows something the others do not:** a port type, a coupling style, a kind of decomposition | breadth is the point |
| **has a reference:** the same solver on the whole domain, or a textbook answer | the Run & compare step needs one |
| **carries two checks:** a balance that closes (mass, energy, charge), and agreement with the reference within a stated tolerance | a live demo must not show a wrong number |
| **labelled "showcase", not "research record"** | the depth of the evidence differs, and the label says so |

---

## 3. The four coupling styles the runner must support

| style | used for | mathematics | status |
|---|---|---|---|
| **A. Overlapping windows, one exchange per step** (time-dependent) | wind farm, waves, plumes | blend by a partition of unity $\sum_i \chi_i = 1$, then the global operators; the Schwarz waveform relaxation run for one sweep ([[atlas-and-standard-dd-theory]]) | built (wake array; W346's harness, serial and threaded) |
| **B. Overlapping windows, iterate until they agree** (steady problems) | heat, elasticity, groundwater | alternating or additive Schwarz: solve each window with its neighbours' latest trace until $\lVert u^{k+1}-u^{k}\rVert < \varepsilon$; the convergence curve is itself a display | new, small |
| **C. Material interfaces without overlap** | copper against steel; air against water | Dirichlet–Neumann (or Robin) iteration: continuity of the field and of the normal flux, $u_1 = u_2$ and $k_1\,\partial_n u_1 = k_2\,\partial_n u_2$ on $\Gamma$ | new |
| **D. Field joined to lumped parts** | circuits, coolant loops, rotors | the field's boundary integral is the lumped part's port variable (current, heat, torque), exchanged per step or solved tightly | built (CS-13 [[case-study-cooling-loop-atlas-0.1]], CS-14 [[case-study-powertrain-atlas-0.1]], actuator disks) |

---

## 4. The cases

All 2-D. Runtimes and code sizes are **[AI Inference]** estimates.

| # | case | physics and port | style | what only it shows | reference and checks | runtime | new code |
|---|---|---|---|---|---|---|---|
| 1 | **Wind farm, 1–21 rotors** | incompressible flow + actuator disks; `MECH`, `ROT` | A | the measured parallel speedup, live | full domain ([[decomposition-speed-by-rotor-count]]) | 10 s – 2 min | none: the runner only |
| 2 | **Two-material plate** (copper insert in steel), steady and transient | heat conduction; `THERM` | C, then B | materials and resolution per region; the convergence plot | full domain; the two-layer wall in closed form, $q = \Delta T / \sum_i L_i/k_i$; energy balance | seconds | ~100 lines |
| 3 | **Pollutant plume down a river** in two reaches (different speed and mixing) | advection–diffusion; `ADVEC` | A, one-way | transport across a seam | full domain; mass balance | seconds | ~80 lines |
| 4 | **Sound through two media** (air-like into water-like) | acoustic wave equation; pressure–velocity (`MECH`) | C, explicit | reflection and transmission at a seam | textbook $R = (Z_2 - Z_1)/(Z_2 + Z_1)$ with $Z = \rho c$; energy balance | seconds | ~100 lines |
| 5 | **Current spreading in a resistive plate** fed by a battery and a resistor | $\nabla\cdot(\sigma\nabla\phi) = 0$ joined to a lumped circuit; `ELEC` | D | the electrical port on a spatial field | full domain; Kirchhoff and energy balance, $\sum I = 0$, $P = I^2R$ | seconds | ~120 lines |
| 6 | **Two-material bracket under load** | plane-stress elasticity (`ThermoStruct2D.solve_mechanical`); `MECH` | B | structures, not only flows | full domain; force balance | seconds | ~80 lines of wiring |
| 7 | **Heated plate that expands** | conduction → elasticity, same mesh | co-located split | the split by physics rather than space | full domain ([[case-study-thermal-strain-atlas-0.1]]) | seconds | wiring |
| 8 | **Heated block cooled by channel flow** | advection–diffusion fluid + conduction solid; `THERM` | C at a two-family seam | two physics families meeting at one seam | full domain; energy balance | < 1 min | ~120 lines |
| + | **Coolant loop, powertrain** | CS-13, CS-14 | D | loop and circuit topologies | their existing balances ($2\times10^{-14}$, $1.9\times10^{-15}$) | seconds | adapters |
| opt. | **Pattern-forming reaction–diffusion** (Gray–Scott) | nonlinear reaction–diffusion | A | nonlinearity crossing seams; visual | full domain | seconds | ~60 lines |

**Coverage:** all five port types (`MECH`, `ROT`, `THERM`, `ELEC`, `ADVEC`), all four coupling styles, and the co-located split. **Not in the showcase:** the rocket and the vehicle union, which stay research records.

---

## 5. What to build, in order

| step | what | effort **[AI Inference]** | depends on |
|---|---|---|---|
| **1** | **Geometry section** of the workbench (§6) | 1–2 sessions | — |
| **2** | **General runner for style A**, the wind-farm family: march a case file, compare with the full domain, live fields (plan step 3 of [[outcome-c4-path-to-declarative-cases]]) | 1–2 sessions | 1 |
| **3** | **A solver-family interface** (state, step given boundary traces, the full-domain equivalent, restriction and assembly) **plus styles B, C, D** | ~1 session | 2 |
| **4** | **Cases 2–8**, about half a session each; 2 and 3 first, then 5 | ~3–4 sessions | 3 |
| **5** | **Showcase gallery:** File > New from example lists every case; each opens, runs and compares in under 2 minutes | small | 4 |
| **6** | **Compile verdict per case** (plan step 2's loader), so Check shows the compiler's per-seam output | ~1 session | 3 |

**Risks.**
- The general runner is where heterogeneous cases will expose design gaps. Keep a per-family adapter rather than one universal stepper.
- Cross-points (three or more windows meeting) break naive blending. The compiler has rules for them (`L2/I2/G1`), and those rules have been wrong once, on a circuit ([[case-study-cooling-loop-atlas-0.1]]).
- Keep every showcase case to one clock.

---

## 6. The geometry section, planned to fit the library

The library changes what geometry must express. Windows alone are not enough, so the case file moves to **version 0.2** (version 0.1 files still load), with four layers on one canvas, each with an editable table:

| layer | what it holds | needed from |
|---|---|---|
| **Domain** | size and cell size | built |
| **Regions** | rectangles (polygons later) assigned a material or physics | case 2 |
| **Windows** | the decomposition | case 1 |
| **Attachments and boundaries** | rotors; lumped components wired to a region's edge; boundary conditions on the domain edges (inlet, outlet, wall, fixed temperature, fixed potential) | cases 1, 3, 5 |

**Interaction, as proposed (Bokeh, installed):**
- a mode selector (Select/move · Draw window · Draw region · Place device · Set boundary);
- `BoxEditTool` to draw, drag and delete; corner handles from `PointDrawTool` to resize;
- tables kept in sync with the canvas both ways;
- snapping to a selectable step (1, 8 or 16 cells);
- a tiling generator (rows × columns, window size, overlap defaulting to twice the ramp width);
- live warnings on the canvas (uncovered cells, thin overlaps, cross-points);
- everything through the existing undo, redo and save.

**Built on Bokeh in the page, with a Gmsh import path beside it** (the owner chose the hybrid on 2026-09-28; see §8).

**Decisions for the owner** (recommendation first):
1. **Snapping default:** 8 cells, or single cells?
2. **Creating windows:** tiling generator plus hand editing, or free drawing only?
3. **Resizing:** corner handles plus editable table, or table only?
4. **Overlap rule:** warn on the canvas and let Check refuse, or block the edit?
5. **Layers:** all four now (case 2 needs regions), or windows only first?
6. **Which cases first after the wind farm:** 2 and 3, then 5?

---

## 7. What this plan does not decide

- The solver-family interface's exact shape. It is decided by building style A first and generalizing from two real families, not in advance.
- Whether showcase cases get pages of their own in the wiki. The proposal: one row each in a gallery page, with the two checks' numbers.
- 3-D. Out of scope for the showcase.

---

## 8. Existing editors that could replace a hand-built geometry section

The owner's question (2026-09-28): is there an existing 2-D visual editor or CAD tool that could be integrated instead of building from scratch? Surveyed that day, with each tool's licence and whether it is already installed checked.

**The short answer:** many tools draw geometry, and **none knows what a window, an overlap ramp, a seam or a port is.** Whatever draws the shapes, the Atlas-specific part (mapping shapes to layers, snapping to the grid, validating overlaps and cross-points, feeding the compiler) has to be written either way. For rectangles on a Cartesian grid, the drawing is the smaller part.

| tool | licence | installed? | what it gives | what it lacks here |
|---|---|---|---|---|
| **Gmsh 4.15** (desktop GUI + Python API) | GPL-2+; its linking exception covers only Netgen, METIS, OpenCASCADE and ParaView | **yes** | a real geometry kernel (OpenCASCADE: curves, booleans); **named physical groups** for regions and boundaries (checked: `region:steel`, `region:copper`); mesh **partitioning** into subdomains, with an option for ghost (overlap) cells; meshing for body-fitted work later | a separate desktop window, not embeddable in the page; its partitions are graph cuts of an unstructured mesh, not rectangles on a grid. meshio could not read a *partitioned* file (checked), so import goes through Gmsh's own API. GPL matters only if the code is ever distributed |
| **draw.io / diagrams.net** (web, embed mode) | Apache-2.0 | no | a polished editor in an iframe: resize handles, grid snapping, layers, per-shape data, undo; returns XML to the host page | embed mode is officially supported only on `embed.diagrams.net` (online), so offline use means downloading the web app. It is a diagram tool, with no units |
| **Excalidraw** | MIT | no | an embeddable React canvas | freehand whiteboard with no units or precision; needs a React build |
| **tldraw** | proprietary SDK licence (production needs a licence key) | no | a capable React canvas | the licence; excluded |
| **FloorspaceJS** (NREL) | open source (licence not checked) | no | the closest in spirit: a web geometry editor aware of zones and boundary conditions | built for buildings (stories, spaces); adapting it is heavier than building |
| **Any 2-D CAD via DXF** (LibreCAD GPL-2, QCAD, FreeCAD, AutoCAD) + **ezdxf** | ezdxf MIT | ezdxf no; no CAD installed | precise drafting in a familiar tool; layers map to windows, regions, boundaries | a separate app and a re-import after every edit; ezdxf needs installing |
| **SVG from Inkscape** (the geomIO pattern) | Inkscape GPL | no | layers of drawn shapes, parsed as geometry | same round-trip; no units |
| **Bokeh / HoloViews drawing tools** (`BoxEdit`, `PolyDraw`, `PolyEdit`, `PointDraw`) | BSD-3 | **yes** | draw, drag and delete rectangles and polygons in the page, wired straight to Python | resize handles have to be composed; it is building, but from high-level parts |

**Recommendation [AI Inference]: a hybrid, not a single tool.**
1. **In-page editing on Bokeh/HoloViews** for the showcase's geometry: rectangles on a grid. It is the only option that can show Atlas's live warnings (thin overlap, cross-point, uncovered cells) while you draw, and it needs nothing new installed.
2. **A Gmsh import path** for anything curved or non-rectangular: draw in Gmsh's GUI, name physical groups by convention (`region:<material>`, `bc:<kind>`, `window:<id>`), and import into the case file. Gmsh is already installed, it is the one surveyed tool that understands regions and boundaries, and it is also the road to body-fitted grids ([[poc3-racelab-body-fitted-grids]]).
3. **Not a diagram tool** (draw.io, Excalidraw, tldraw): no units, and they need network access, a large download or a licence key.

**Owner's decision (2026-09-28): the hybrid.** In-page Bokeh editing for rectangles on the grid, and a Gmsh import path by physical-group naming for anything else.

---

## See Also

- [[outcome-c4-path-to-declarative-cases]] — the four-step path this extends
- [[outcome-c4-modular-multiphysics]] — every graph built so far
- [[decomposition-speed-by-rotor-count]] — case 1's measured answer
- [[atlas-and-standard-dd-theory]] — where each coupling style sits in classical domain decomposition
- [[port-algebra-atlas-0.1]] — the five port types the cases cover
