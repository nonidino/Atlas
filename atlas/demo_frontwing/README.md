# PoC 2 — the live front-wing demo

A front wing riding at height `h` on a moving floor, bending under its own aero
load, with **both interfaces live at once** and **every seam's certification
verdict on screen**, taken from the compiler rather than from a rule of thumb.

```bash
python -m atlas.demo_frontwing --open
```

Then open <http://127.0.0.1:8012/>. On Windows PowerShell the same command works;
there is no shell script to invoke.

It needs `fastapi`, `uvicorn` and `pillow` beside the package's own
dependencies, and it needs the build repo for `ThermoStruct2D` — the structural
expert — exactly as `atlas/cases/wing_fsi.py` does. Set `ATLAS_BUILD_REPO` if the
checkout is not at `~/physics-foundation-model`.

Run `python scripts/w141_poc2_frontwing.py --stages setup` first if
`out/w141/settled.npz` is absent: without it the demo releases from the
freestream, says so on screen, and shows a transient that no number on
[[poc2-frontwing-results]] was measured at.

---

## What is on the screen

**The flow field** — speed, colour-mapped, over the whole `208x144` domain, with
the rolling road drawn along the bottom. The composed column blends six
`reference.WindowNS` windows and applies one global spectral Leray projection per
exchange; the `referent` button puts a single window over the whole domain, which
differs from the composed column **by the cut and by nothing else**.

**The structure**, drawn from `FlexWing.stations(delta, h)` — the same call the
physics makes, so the picture is the model's geometry and not a second one — as
32 quads, each coloured by the peak von Mises over the two elements through the
thickness at that station. The stress ramp is anchored on the **ceiling**, not on
the frame's own maximum: a ramp that renormalises every frame makes a design at
20% of the ceiling look identical to one at 99%, which is the one thing this
panel exists to distinguish. The dashed white line is the undeflected chord, so
the bending is visible against something; the blue leader is the ride height.

**The readout** — downforce, drag, L/D, ride height, tip deflection, peak von
Mises against its ceiling, peak deflection against its ceiling, the 33-equation
interface residual, and the measured cost of one step.

**The per-seam certification panel**, which is the point of this demo.

---

## The certification panel, and the two things it must not overstate

Every one of the nine seams gets its own light. The colour is
`compile_scheme`'s own verdict, grouped by the seam each decision's `subject`
reaches — a decision reaches a seam when its subject **is** the seam, or is one
of the two ports the seam pairs, or is one of the two agents it joins. Graph-level
subjects (`<graph>`, `<assembly>`, `<run>`) reach every seam. The grouping is one
function, shared with `scripts/w141_poc2_frontwing.py`, and a test asserts the
driver and the demo agree on this graph.

| light | means |
|---|---|
| **red** | at least one **refusal** reaches this seam. The composed march may still run — and does — but the compiler declines to certify it, and the run is a search instrument rather than a verified one |
| **amber** | no refusal, at least one **decertification**. The bound applies with a constant nobody measured, or a hypothesis is `unchecked` |
| **green** | neither. **Nothing in this graph is green**, and the panel says why: `L`, `sigma` and `C_mu` are unmeasured (W1, W3, W49) and a non-empty `unmeasured` list has forced `admit-uncertified` on every graph in this package since **W56** |

**1. The colour is a property of the declaration, not of the design point.**
Measured, not assumed: over the sixteen corners of the design box, **zero**
produce a different per-seam verdict map. So the lights change when the interface
is *declared* to move — the two surface seams go **red** at
`L2/InterfaceMotion` and the seven fluid–fluid ones stay amber — and they do not
flicker as a knob turns. The demo re-runs the compile on **every** design change
anyway, on its own thread, because the honest way to show that is to run it.

**2. What moves with the design is the quantity underneath each light**, and that
is what a designer would actually watch: each surface seam's one-sidedness ratio,
the interface residual, and the two constraint margins.

The two surface seams carry different decertifications and the reasons are on
hover:

- **`wet`** (fluid ↔ structure, 32 stations) — `L2/R10/halo` and
  `L4/E7/passivity`. The halo rule over-fires here (**W136**): this seam is a
  *physical* boundary of the solid with no overlap for anything to outrun. The
  passivity defect is an **orientation** artefact (**W138**): the two blocks are
  added where the interface residual subtracts them, and flipping one sign takes
  the defect to exactly zero.
- **`mount`** (fluid ↔ lumped suspension, the ride height) — `L2/R10` and
  `L4/E7/passivity`, CS-10's seam unchanged.

---

## The optimiser, and why its number is a demo number

Press **optimise** and Adam steps the four design knobs — wing stiffness `E*`,
plate thickness `t/c`, spring rate `k`, free ride height `h0` — up the penalised
objective: maximise settled downforce subject to a stress ceiling and a
deflection ceiling, both entered as margins normalised by their own ceiling so
the two are commensurable without a weight nobody measured.

**The optimiser's rollout *is* the live march.** It starts from the state on
screen and the state it ends at is the state that stays, so the field animates
inside the gradient step instead of freezing until it lands. That makes it a
**warm-started, short-horizon** estimator and **not** the one
`scripts/w141_poc2_frontwing.py` measures with — a 120-macro-step rollout from a
settled field, which is a clean function of the design alone. The screen says so.
Read the demo for the mechanism and the results page for the numbers.

**No wall-clock figure is quoted anywhere in this package.** The PoC 1a demo
shipped a table of milliseconds-per-step that was wrong by a factor of four the
next time anyone measured it, on unchanged code, because the box was in a
different power state. The engine measures itself instead and the screen marks
the figure as an estimate until the first real one lands.

---

## The scheme controls

| | |
|---|---|
| **six windows / referent** | the composed column against the single-window referent. They differ by the cut alone, and over 480 macro-steps the cut's own defect is `9.0e-5` in the tip deflection and `2.2e-5` in the ride height, against the seams' lag at `5.4e-2` and `2.3e-3` — factors of 602 and 106 |
| **joint / split / lagged** | `joint` solves both seams as one 33-equation Newton system; `split` solves them in **turn**, which is the ablation; `lagged` solves once per macro-step and holds |

Switching the tiling rebuilds the field; switching the coupling does not.

---

## What this demo is not

- **No stress number here is a statement about an alloy.** The plate is a `32x2`
  Q1 plane-stress cantilever, Q1 elements lock in bending, and `E*` is a declared
  design knob in **flow units** calibrated to a deflection rather than taken from
  a material. The ceilings are levels the reference design sits inside and the
  downforce-seeking direction runs into, which is the only property a ceiling
  needs for a constrained search to be about the constraint.
- **One Reynolds number, one grid, one incidence.** `Re_c = 125`, `208x144`
  cells, `20` degrees. Nothing here is a grid-convergence study.
- **The wing has no Kutta condition, no bound circulation and no boundary layer
  on its own surface.** It is CS-10's porous inclined plate, which has *thickness*
  — it blocks the gap, which is the mechanism ground effect runs on — and an
  exact discrete momentum exchange.
- **The floor is a rolling road**, so there is no floor boundary layer and no
  viscous gap choke. That is why CS-10's measured ground-effect curve is monotone
  with no turnover, and it is why the optimiser will push the ride height down
  until something other than the aerodynamics stops it — and **measured, that
  something is the suspension expert's declared floor, not either ceiling.**
  The results page's primary search ends $2.3\%$ under the stress ceiling and at
  half the deflection ceiling, declined eight times by `Suspension.validity`.
- **Nothing here reaches `admit`**, and no panel on this screen should be read as
  if it did.
- **The optimiser can be stopped by an expert declining, and that is not a
  crash.** Both experts carry a declared envelope — a small-strain bound on the
  deflection and a floor under the ride height — and the optimiser drives toward
  the floor, because in this flow model downforce rises all the way down. When
  it reaches one the march stops, the screen says *which* expert declined and at
  which macro-step, and the state resets. **This check was missing from every
  path but `run` until 2026-09-04** (**W145**): the demo's march and its
  optimiser both drive `macro_step` directly, so the wing could be walked
  through the floor and the screen would have shown a downforce for a design the
  model declines. It now goes through one check in `Engine._absorb`, which every
  path funnels through.

## See also

- `wiki/concepts/Atlas 0.1/common/poc2-frontwing-results.md` — what is proved,
  what is not, the cost, the ablation and the limitations
- `atlas/demo/README.md` — the PoC 1a demo this is modelled on
- `atlas/cases/front_wing.py` — the assembly; `atlas/CASE-STUDY-GUIDE.md` — how a
  graph is declared
