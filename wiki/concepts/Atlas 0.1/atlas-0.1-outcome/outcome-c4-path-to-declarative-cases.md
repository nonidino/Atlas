# Outcome C4 — the path to case studies without a Python module, and a decomposition GUI

**Type:** Outcome page — **design plan** (folder: `Atlas 0.1/atlas-0.1-outcome/`)
**Status:** written 2026-09-26 on the owner's question: *how can a new case study be set up without a full Python module, and how hard is a GUI that lets a person draw the domain decomposition, run it, and compare it with the full-domain solve (classical only)?* This page reads the code as it stands and proposes; **nothing on it is built.** Estimates are marked **[AI Inference]**.
**Hub:** [[00-atlas-0.1-outcome]] · **Parent claim:** [[outcome-c4-modular-multiphysics]]

> **2026-09-27: the shell of step 4 is built** — `atlas/workbench/` (`python -m atlas.workbench --open`). It has the menus, the six-step workflow, the case file of step 2 (`spec.py`, schema `atlas-workbench/case@0.1`), and a registry catalogue for step 1 (`registry.py`, not yet a factory). It is built on Panel and Bokeh (BSD-3, already installed), with no new download and no outside request. The geometry editor is being designed with the owner, and the loader (step 2's graph half) and the runner (step 3) are not built. `atlas/workbench/README.md` lists the open geometry decisions.
>
> **2026-09-28: extended to a library.** [[showcase-library-plan]] plans eight fast classical cases across all four coupling styles, the geometry section to fit them, and a survey of existing editors.
>
> **2026-09-28: the geometry section is built**, as the hybrid the owner chose: rectangles drawn on a Bokeh canvas (windows, material regions, rotors, boundaries), a tiling generator, and import from Gmsh by physical-group names. The case file is now `atlas-workbench/case@0.2`. The loader and the runner are still not built. Details: [[showcase-library-plan]] §6.
>
> **2026-09-28: step 3's executor is built for the wind-farm family** (`atlas/workbench/runner.py`, `families/windfarm.py`): W346's march lifted onto any rectangles, run from the page decomposed (serially and on threads) and on the full domain in turns, with Stop, live fields and a record per run. On the example tilings its arms are W346's own columns to the bit, and W346's speed result reproduces live at 21 rotors (threaded 4.0–4.6× faster than the full domain, serial 0.90–0.91×). The loader (step 2's graph half) is still not built. Details: [[showcase-library-plan]] §5.

> **2026-09-29: the executor runs three families and four coupling styles.** The interface every family fills (state, restriction, a step given boundary traces, the full-domain equivalent, assembly with a balance) was derived from the wind farm and heat conduction, as this page's step 1 asked. Heat conduction runs as overlapping windows iterated to agreement (style B) or as two materials meeting at an interface (style C, Dirichlet–Neumann). A resistive plate on a circuit is the first field joined to lumped parts (style D). The case file is now `atlas-workbench/case@0.3`, with lumped attachments. Every check was registered before its family's first run. At these sizes the full domain is faster than every decomposed arm of the new styles, and the page says so. The loader (step 2's graph half) is still not built. Details: [[showcase-library-plan]] §5.

> **2026-09-29, later: eight families, one per showcase case.** Five more families: a plume down a river (style A with an explicit step), sound through two media (style C, explicit), a loaded bracket (style B on a structure), a heated strip split by physics, and a cooled block (style C at a seam between two physics). Every one of their checks passes on its example, each registered before its family's first run. The cooled block's reference check failed twice first, and that was mended in the method (a floating piece takes the Dirichlet side), not in the tolerance. The loader is still not built. Details: [[showcase-library-plan]] §5.

> **2026-09-29, last: the loader's graph half is built, and every example compiles from the page.** It is built per family rather than through step 1's registry: each family's `case_graph(spec)` declares one agent per window or piece, derives a seam wherever two meet, and declares the cross-points (W162). `atlas/compiler.py` then judges it, and the Check step shows the verdict per seam.
> - Seven examples come back `admit-uncertified`.
> - The two-material conduction cases and the bracket are refused at R10, a proxy that does not read that their coupling iterates to agreement.
> - The heated strip is refused before the compiler: its volumetric bond is not a port type.
>
> Every example also ran in the page in under 2 minutes. The 21-rotor farm's compile adds 95 s to its 70 s run. Details: [[showcase-gallery]] and [[showcase-library-plan]] §5.

---

## 1. What a case study is today, layer by layer

A case study is one file, `atlas/cases/<name>.py` ([[atlas-implementation]]; `atlas/CASE-STUDY-GUIDE.md`). Reading five of them (`wake_array`, `scaling_ladder`, `cooling_loop`, `powertrain`, `wing_fsi`) splits that file into four jobs. Only two of them are really case-specific.

| job | what it is | generic today? | evidence |
|---|---|---|---|
| **1. geometry** | the domain, the windows, where the devices sit | **partly.** The car's shape is already data: `atlas/cases/car_geometry.json`, edited in a browser by `scripts/car_editor.py`. The wind-farm tilings are code: `ArrayTiling` takes `n_col`, `n_row`, a fixed 128-cell window and a fixed 112-cell stride | `scripts/CAR_EDITOR.md`; `wake_array.ArrayTiling` |
| **2. declarations** | agents with `ExpertCapabilities` records, ports, `Connection`s, the `CaseGraph` | **nearly.** The records are dataclasses of enums and numbers, but each also carries **callables**: `boundary_response`, `validity`, `storage` | `atlas/capability.py`, `atlas/graph.py` |
| **3. the compile** | the nine layers and the three-verdict compiler | **yes.** No case branches anywhere; `graph.py`'s own rule is that a case study adds no code elsewhere | [[atlas-implementation]] |
| **4. the run** | the time march: cut, local solves, exchange, assembly, projection, devices, diagnostics | **no.** Every real case study has its own rollout class: `FSIRollout`, `GroundRollout`, `BrakeRollout`, `FrontWingRollout`, `RaceRollout`, `UnionRollout`, `CarUnion`, `wind_farm_design.Rollout`, and the rocket's `CoupledEpisode` in the build repo. A generic runtime exists (`atlas/solve.py`: `coupled_step`, `rollout`, driven only through `boundary_response`), but only `thermal_seam` and the tests use it | `grep` of `atlas/cases/` |

**So the blocker is job 4, not job 2.** The compiler already reads data. What stops a person from declaring a case without Python is that **running** it means writing a march.

**There is a second, smaller blocker in job 2:** the callables. A solver's `boundary_response` is code, but it is code *per solver*, not *per case*. `window_ns`, `wake_array` and `cooling_loop` each wrap the same few build-repo solvers.

---

## 2. The path, in four steps

### Step 1 — an expert registry: Python once per solver, never per case

A table from a name to a factory that returns a configured solver *and* its capability record:

| registry name | solver | ports it can expose | exists today as |
|---|---|---|---|
| `incompressible/window-ns` | `reference.WindowNS` (and its exposed variant) | `MECH` on window faces | `wake_array.reference_solver`, `exposed_reference_solver`, `window_ns` records |
| `incompressible/monolith` | `RectangularNS` on the whole domain | — | `scaling_ladder.reference_monolith` |
| `device/actuator-disk` | `disk.ActuatorDisk` as a body force | `ROT` shaft, the volumetric bond | `wake_array` rotor capabilities |
| `conduction/q1-block` | `thermostruct2d.step_thermal` | `THERM` on a face | `cooling_loop.BlockAgent`, `thermal_seam` |
| `lumped/coolant-leg`, `lumped/radiator`, `lumped/pump` | the affine legs | `ADVEC`, `THERM` | `cooling_loop` legs |
| `lumped/machine`, `bus`, `battery`, `inverter` | the circuit elements | `ROT`, `ELEC` | `powertrain` |

Adding a solver to the registry is the one place Python remains. Adding a *case* never needs it.

### Step 2 — a case spec file (JSON)

Everything a case module declares in code, as data. A sketch for the wind-farm family:

```json
{
  "domain": {"shape": "rectangle", "cells": [688, 464], "dx": 0.03125},
  "physics": {"expert": "incompressible/window-ns", "nu": 0.00392, "u_inf": 1.0},
  "windows": [
    {"id": "W0", "origin": [0, 0], "size": [128, 128]},
    {"id": "W1", "origin": [112, 0], "size": [128, 128]}
  ],
  "overlap": {"ramp_cells": 8, "assembly": "projected"},
  "devices": [{"id": "R1", "expert": "device/actuator-disk", "x": 3.5, "y": 1.75, "diameter": 1.0}],
  "run": {"macro_dt": 0.2, "steps": 60, "start": "freestream"},
  "compare": {"reference": "incompressible/monolith", "metrics": ["farm_power", "rms", "time_per_step"]}
}
```

A loader (`atlas/spec.py`) turns this into a `CaseGraph` by calling the registry. It derives the connections from geometry: two windows that overlap get a seam; a device inside a window gets its bond. The compiler then decides admissibility exactly as it does now.

### Step 3 — a generic executor for one family: overlapping Schwarz with devices

The march every wind-farm-family case already runs, lifted out of `scripts/w100_scaling_ladder.composed_step`, `wake_array.assemble_conservative` and Tier 89's `scripts/w346_rotor_count_speed.py`:

$$
\text{cut} \;\to\; \text{local solves (serial, threaded, or local time stepping)} \;\to\; \text{blend by } \chi_i \;\to\; \text{one global projection} \;\to\; \text{band} \;\to\; \text{devices}.
$$

The one generalization it needs is **arbitrary rectangles instead of a regular tiling.** `ArrayTiling` assumes a fixed stride. The assembly layer already does not: `assembly.GridPartitionOfUnity` takes any set of index blocks and weights, and certifies the partition-of-unity identity for them. So the change is a `RectangleTiling` whose weights come from ramps at each overlap, checked by the identity residual the compiler already computes.

The **monolith comparison** is well defined here and only here: one solver, one discretization, the undivided domain ([[outcome-c1-decomposition-vs-monolith]] §1). The executor should refuse "compare to full domain" when a spec mixes families, and say why.

### Step 4 — the GUI

The project has built this kind of page four times already (`atlas/demo`, `atlas/demo_frontwing`, `atlas/demo_racelab`, `scripts/car_editor.py`). The pattern is a FastAPI server plus one HTML page, a background engine thread, and a WebSocket for frames. A decomposition editor on the same pattern:

| panel | what it does | built from |
|---|---|---|
| **canvas** | draw the domain; draw, drag and resize windows snapped to the cell grid; place rotors; see overlaps, ramps and the partition-of-unity residual live | `car_editor.html`'s drag handles; the tiling overlay RaceLab already draws |
| **inspector** | per window: size, overlap with each neighbour, cells, which rotors it holds; per seam: port type and the compiler's verdict with its rule | the RaceLab window inspector; `compile_report` |
| **run** | "Run" marches the decomposition and the monolith **in turns, on the same machine**; progress, then side-by-side fields and their difference | the PoC 1a demo's timed three-way column; W346's harness |
| **results** | time per step for each arm, speedup, farm power, rms difference, the composed defect over time; save and load the spec | W346's record format |

**Lessons the earlier demos already paid for, which this GUI must inherit:**
- A picture of the state is a measurement instrument. W98 was found by watching an animation ([[case-study-wake-array-atlas-0.1]]).
- A control that reaches nothing is worse than no control. Every knob must be traced end to end ([[poc3-racelab-knobs]]).
- Rebuild on commit, not on drag. The page must say which configuration is marching ([[poc3-racelab-dashboard]]).
- Quote the speed ratio measured on the machine in front of the viewer, not a published one ([[atlas-proof-of-concept-1]] §11).
- Open the page and click every control; API tests miss what the page never draws ([[poc3-racelab-demo]] §5.4).

---

## 3. How hard, by scope

**[AI Inference]:** estimated from the size of the pieces that already exist, in the project's own unit, a tier (one working session with its record, tests and wiki pages). These are estimates, not measurements.

| scope | what it takes | estimate |
|---|---|---|
| **A. The owner's ask: one family, classical, rectangles** (2-D incompressible flow with actuator disks, windows drawn by hand, compared with the monolith) | the registry for 3 entries; the spec loader; `RectangleTiling` over `GridPartitionOfUnity`; the executor lifted from W346; the GUI | **3–4 tiers**: spec and loader 1; executor and rectangle tiling 1; GUI 1–2 |
| **B. Add a second family on the same canvas** (a conduction block with a coolant loop, or a structure under a wing) | registry entries with real `boundary_response`s; an executor that marches heterogeneous states on different clocks, via the generic `solve.rollout` or a new driver | **4–8 more tiers**, and research-grade: every multiphysics case so far found a declaration or coupling defect (W308, W199, the Tier 79 `motion_class` default) |
| **C. Non-rectangular geometry** (body-fitted or overset grids around drawn shapes) | `overset`, `overset_multi` and `overset_ns` exist for the car; generalizing grid generation from a drawn outline is new | **several tiers**; the car took Tiers 60–68 |
| **D. Learned experts selectable per window** | registry entries for learned experts, the switch RaceLab already has | small once A exists for Poseidon-T on uniform windows; blocked by [[outcome-c5-requirements-for-dd-native-experts]] everywhere else |

**What makes A tractable:**
- The monolith exists.
- The window solver, the assembly, the projection, the disk and the timing harness all exist and are tested.
- The GUI pattern has been built four times.

**What could make it slower:**
- Arbitrary rectangles can create **cross-points** (three or more windows meeting) that the regular tiling never produced. The compiler has rules for them (`L2/I2/G1`; `detected_cross_points`), and they have been wrong once already, on a circuit.
- A ramp between two windows needs overlap on both sides. A GUI that lets a person draw windows that barely touch must refuse them rather than blend nothing.

---

## 4. What this page does not decide

- Whether the spec is JSON, YAML or TOML, and whether it lives beside `car_geometry.json`.
- Whether the executor reuses `atlas/solve.py`'s generic `rollout` (probed-DtN transmission, built for non-overlapping seams) or lifts the overlapping-Schwarz march. For scope A the second is the shorter path; for scope B the first is the more general one.
- Who owns the registry's capability declarations. They carry decisions (`response_half`, `orientation`, units) that earlier tiers found wrong and fixed one at a time.

---

## See Also

- [[outcome-c4-modular-multiphysics]] · [[outcome-c1-decomposition-vs-monolith]] · [[00-atlas-0.1-outcome]]
- [[end-to-end-architecture-spec]] — the nine layers the loader must feed
- [[general-coupling-scheme]] — the scheme tuple an executor instantiates
- [[per-region-assembly]] — the partition of unity as a per-region object
- [[poc3-racelab-bundle]] — what shipping a GUI to another machine took
