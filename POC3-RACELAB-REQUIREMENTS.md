# PoC 3 — RaceLab: requirements

**Status:** specification, written 2026-09-12. Not yet built. Multi-session.
**Branch it will ship on:** `poc3-racelab-demo` (self-contained bundle, as PoC 1 did).
**Predecessors:** PoC 1a (`atlas/demo/`, the wind farm), PoC 2 (`atlas/demo_frontwing/`, the front wing).

---

# 0. What RaceLab is, in one paragraph

**A race car, simulated as a graph of interchangeable physics experts, with a
live dashboard where every window can be flipped between a classical solver, a
learned neural operator, and a learned operator held to the classical answer by
a theorem — and the speed and accuracy consequences of each flip shown per
window and for the whole machine.** The car is the stage. The point is the
switch. Nothing in this project's literature ships a per-window
classical/learned/certified selector with live error attribution, and that is
the novelty; the car exists so that a viewer can see at a glance which part of a
real machine each expert is responsible for.

---

# 1. Why this system, and what it inherits

RaceLab is the cash-in on rung 9. Tier 49 already marches an eighteen-agent
union across five governing families — fluid, structure, conduction, coolant
advection, lumped circuit — at a declared vehicle scale, with every join's
receiver balance measured over a march beside its null arm. RaceLab adds
geometry, interactivity and a third expert mode; it does not need a new
governing family, a new port type, or a new coupling rule.

**What already exists and must be reused rather than re-implemented:**

| asset | module | what it gives RaceLab |
|---|---|---|
| the fluid window | `reference.WindowNS` via `atlas/cases/window_ns.py` | the classical 2-D incompressible expert |
| the six-window tiling and its composition layer | `atlas/cases/wing_fsi.py` (`FSIRollout`: cut, blend, project, band) | the window machinery, unchanged |
| the front wing with all four design knobs | `atlas/cases/front_wing.py` (`FrontWingRollout`) | plate structure, suspension, ride height |
| the cooling loop | `atlas/cases/cooling_loop.py` | block, coolant legs, radiator, pump |
| the powertrain | `atlas/cases/powertrain.py` | MGU, bus, battery, inverter, `CircuitSolve` |
| the brake | `atlas/cases/brake_thermal.py` | vented disc, cooling duct, real gas properties |
| rotors as actuator disks | `atlas/cases/wake_array.py` | the device-as-body-force pattern |
| the joined union | `atlas/cases/integration_union.py` | J1/J2/J3, the joining seams |
| **the vehicle scale** | `atlas/cases/vehicle_march.py` (`VehicleScale`) | metres and seconds, and the 400:1 clock ratio |
| **defect correction** | `atlas/defect_correction.py` | the certified third mode |
| the compiler | `atlas/compiler.py` | per-seam verdicts to display |
| the demo pattern | `atlas/demo_frontwing/` | FastAPI + thread engine + WebSocket + one `index.html` |
| the bundle builder | `scripts/w146_build_frontwing_bundle.py` | the packaging template |

**The learned expert is Poseidon-T**, 2-D incompressible Navier–Stokes,
CC-BY-NC-4.0, already vendored in PoC 1's bundle.

---

# 2. Hard constraints

These are not negotiable and every design decision below is downstream of them.

1. **NeuberNet must not appear in the bundle.** It is unlicensed and local-only.
   No structural learned expert ships. If a structural learned option is wanted
   it must be trained from scratch and licensed by us.
2. **Poseidon-T is CC-BY-NC-4.0** — research use, no commercial use. It must be
   named on screen with its licence, as PoC 1 does, and `vendor/POSEIDON-T-LICENCE.md`
   must travel with it.
3. **`scOT` has no licence file** and cannot be redistributed. The launcher
   installs it from the pinned upstream commit with `--no-deps`, exactly as PoC 1's
   does. **With no network, the classical column must still run.**
4. **Every solver in the project is 2-D.** A 3-D version is a new solver, and
   Poseidon cannot serve it.
5. **The build repo is private.** It is vendored into the bundle under `vendor/`,
   and `window_ns.load_reference` is pointed at it by `run.py` — no source file
   is edited to make the bundle work.
6. **No source file may be modified for the bundle.** The tests on the demo
   branch must be character-identical to the ones on `atlas-0.1`, because that
   is what makes them mean the same thing.

---

# 3. The physical system

## 3.1 The 2-D car — a centreline slice

A streamwise–vertical slice through the car on a moving ground plane. It must be
recognisable as a car at a glance and must not be a rectangle.

| feature | physics | source |
|---|---|---|
| **front wing**, main plane + flap, in ground effect | immersed body force in the fluid; quasi-static plate structure; suspension spring at the mount | `front_wing`, unchanged |
| **floor and diffuser ramp** | immersed body; the ramp angle is a parameter | new geometry, existing machinery |
| **radiator duct** with the core inside it | duct walls as immersed bodies; the core as a porous plane returning $\tfrac12 K u^2$ | `integration_union.CoreRadiator` (J1) |
| **sidepod / body** | immersed body | new geometry |
| **rear wing** | immersed body force | `front_wing`'s plate, re-sited |
| **wheels**, front and rear | immersed bodies with a rotating-surface boundary condition; wakes | new, simplest defensible form |
| **moving ground** | the floor boundary moves at road speed | a band condition, new |
| **brake discs** behind the wheels | vented disc + cooling duct | `brake_thermal` |
| **coolant loop** | block, pump, radiator, lines | `cooling_loop` |
| **battery / MGU / inverter** | lumped circuit | `powertrain` |

**Domain.** Wider and taller than the front-wing case's $208\times144$. Target
roughly $384\times192$ cells at the same $\mathrm{d}x = 1/64$ in tiling units,
i.e. $6.0 \times 3.0$ length units — at the declared scale
($L_0 = 0.50$ m) that is $3.0 \times 1.5$ m, which holds a car and its near wake.
**The exact figure is a measurement, not a guess:** the first session must time a
march at the candidate size and set it so a macro-step stays under ~0.5 s.

## 3.2 The 3-D car

The same body revolved into three dimensions is wrong — a car is not
axisymmetric — so the 3-D version is a **half-car in a box** with a symmetry
plane: body, front wing with endplate, floor, diffuser, one sidepod with its
duct, one front and one rear wheel.

**3-D is a separate phase (Phase 4) and the demo must be complete and shippable
without it.** See §9.

## 3.3 Parameters exposed to the user

Grouped, with the subsystem each one reaches:

| group | parameter | range | reaches |
|---|---|---|---|
| **vehicle** | road speed | 20–90 m/s | everything; sets the fluid's freestream and the scale's $U_0$ |
| | ride height | 0.15–0.60 (tiling units) | front wing, floor, diffuser |
| | rake (front-to-rear ride-height difference) | 0–0.15 | floor, diffuser |
| **aero** | diffuser ramp angle | 0–15° | floor geometry |
| | front wing flap angle | 0–30° | front wing geometry |
| | rear wing angle | 0–30° | rear wing geometry |
| **cooling** | radiator duct area | 0.3–1.5 of nominal | duct geometry, core inflow, $UA$ |
| | coolant mass flow | 0.05–0.30 kg/s | `cooling_loop` |
| | ambient temperature | 273–318 K | `cooling_loop`, `brake_thermal` |
| **powertrain** | battery power draw | 0–350 kW | `powertrain`, and J2's heat into the block |
| | battery state of charge | 0–1 | `powertrain` open-circuit voltage |
| **brakes** | brake duty | 0–1 | `brake_thermal` heat input |
| | brake duct opening | 0.3–1.5 of nominal | `brake_thermal` duct velocity |
| **structure** | plate stiffness $E^\star$ | as `DESIGN_REF` | front wing structure |

**Every parameter must be honestly wired.** A slider that moves a number nothing
reads is worse than no slider: if a parameter cannot reach its subsystem, it is
omitted and the omission is recorded in §11's hole list.

---

# 4. Windows and the expert switch

## 4.1 Windows are derived from geometry and are static

The window decomposition is computed **once** from the car's geometry at
startup and does not change when a parameter moves. That is the envelope
hypothesis the whole framework assumes (E1: a fixed graph), and RaceLab must not
quietly violate it. A parameter that would change the decomposition — a ride
height that moves a body across a window boundary — is either clamped or the
window set is rebuilt with a visible "recompiling" state.

**Target: 12–20 fluid windows**, laid out along the body rather than as a uniform
grid, plus lumped windows for each non-fluid agent. Overlapping tiling with the
same halo and partition of unity `wing_fsi` already uses.

## 4.2 The three expert modes, per window

This is the centre of the PoC.

| mode | what runs in that window | speed | accuracy |
|---|---|---|---|
| **classical** | `WindowNS` | baseline | the referent |
| **learned** | Poseidon-T's composed column on that window | faster per call where the checkpoint is cheap | measured, and visibly wrong in places |
| **certified** | Poseidon-T inside defect correction against the classical window step | slower than learned, faster than classical *only if it pays* | **provably the classical answer** (Theorem 1) |

**The certified mode is the differentiator and it must be honest.** Tier 48 and
Tier 50 measured that in this project's slot the learned expert does *not*
currently pay against classical coarsening, and that its learned content is
worth about 13 classical calls against a 69-call weight-noise spread. **RaceLab
must not imply otherwise.** It shows the mechanism working and reports the price,
including when the price is bad.

## 4.3 Where a learned option does not exist

Structure, conduction, coolant advection and the circuit have **no shippable
learned expert**. Those windows show the switch **greyed out with a reason**, not
hidden. A viewer should learn from the dashboard that four of five families have
no learned option — that is a true and important fact about the field, and
`physics-simulation-datasets` §3.4 already records why.

---

# 5. Outputs

## 5.1 Field overlay

A field selector over the fluid domain, drawn as a PNG layer the way
`demo_frontwing.engine.field_png` already does it:

- streamwise velocity $u$
- vertical velocity $v$
- velocity magnitude
- vorticity
- pressure (from the projection)
- **error against the all-classical referent**, per cell
- temperature, where a thermal agent owns the region

Plus body outlines, window boundaries, and a per-window tint showing its current
mode.

## 5.2 Per-window telemetry

For each window, live:

- current mode
- **one-step error against the classical expert on the same input state** — this
  is exact, cheap, and needs no global referent
- wall time per call, and calls per second
- the compiler's verdict for the seams that window touches (`admit`,
  `admit-uncertified`, `refuse`) with the rule number
- for certified mode: outer iterations, inner cheap calls, and the residual

## 5.3 Global telemetry

- **speed**: macro-steps per second, and the ratio against the all-classical march
- **accuracy**: rms field error against the all-classical march at the same
  macro-step, plus the settled integral quantities (downforce, drag, L/D,
  coolant return temperature, disc temperature, battery current)
- a small ledger: how many windows are in each mode, and the net predicted
  speed-up from the per-call costs

## 5.4 The accuracy story, stated honestly

The user raised this and it needs a precise answer.

1. **There IS a referent for the composed answer.** The all-classical composed
   march exists and is affordable. Every mixed configuration is scored against
   it. This is what CS-12 and CS-18 already do.
2. **The composed classical answer is not the physics.** It carries its own
   composition defect against a single-domain monolith. Where the monolith is
   affordable (the fluid alone, on a single tiling) that second referent is also
   run and the gap is reported. Where it is not, the demo says so.
3. **Per-window one-step error needs no referent at all** and is the number to
   trust most.
4. **Certified mode has a proof, not a measurement.** Its limit is the classical
   settled state whatever the learned map does. That is the one accuracy claim in
   the whole demo that does not depend on a measurement.
5. **Never show a single-draw number as an effect size.** Tier 50's lesson. Any
   randomised quantity gets a replicate count and a spread.

---

# 6. The dashboard

Single-page, vanilla HTML/CSS/JS in one `static/index.html`, no build step — the
pattern PoC 2 uses at 58 KB. FastAPI + uvicorn server, a background engine
thread marching continuously, frames pushed over a WebSocket.

**Layout:**

- **centre**: the car, the field overlay, window boundaries, mode tints
- **left**: parameter sliders, grouped as §3.3
- **right**: the window inspector — click a window, get §5.2
- **bottom**: global telemetry, the speed/accuracy pair, and the mode ledger
- **header**: the expert names and their licences, always visible

**Interactions:** click a window to select; a three-way toggle to set its mode;
"all classical" / "all learned" / "all certified" presets; a field selector;
pause/step/reset.

---

# 7. Packaging

Identical in structure to PoC 1's bundle, which is the only one proven to run on
a machine that has nothing.

```
git clone --branch poc3-racelab-demo --single-branch https://github.com/nonidino/Atlas.git poc3
cd poc3
./run.sh            # Windows PowerShell: .\run.cmd
```

**Requirements:**

- macOS, Linux, Windows. Python 3.10–3.12. No GPU required; CUDA used if present.
- The launcher builds a `.venv` beside itself, installs every dependency at a
  **pinned** version, installs `scOT` from its pinned upstream commit with
  `--no-deps`, runs a self-test that names any missing package and marches both
  columns, and only then starts the server.
- Nothing is installed outside that folder.
- `./run.sh --check` runs only the self-test.
- **`run.sh` must be mode 755 in the git index.** A Windows-built branch ships it
  644 and it dies `exit 126` on Linux — verify by cloning into a fresh directory.
- Bundled: the framework, `vendor/src/atlas/cases/windfarm/` (the build repo's
  solvers), `vendor/hf-cache/` (Poseidon-T), `vendor/POSEIDON-T-LICENCE.md`.
- A `scripts/build_racelab_bundle.py` that constructs it, in the shape of
  `scripts/w146_build_frontwing_bundle.py`. **It does not push.**

---

# 8. What must be measured and recorded, not just displayed

RaceLab is a demo, but it is this project's demo, and the vault's standards
apply. Each phase produces a wiki page with:

- every number on screen traceable to a recorded run in `out/`
- controls beside every measurement
- named holes for everything not done
- `**[AI Inference]**` on anything speculative
- an entry in `wiki/log.md`, rows in `wiki/index.md` and `gap-worklist.md`
- tests in `tests/test_tier<N>_<topic>.py`
- `python scripts/vault_scan.py wiki` → 0 problems
- the full suite green, with the total recorded

**A demo that shows a number the vault cannot reproduce is a defect.**

---

# 9. Phases

Each phase is one to two sessions and ends shippable.

| phase | deliverable | ends with |
|---|---|---|
| **1 — the graph** | the car's geometry, the window decomposition, the assembled `CaseGraph`, and a headless march of the whole thing at the declared scale | a compile with its verdicts, and a march that runs and conserves what CS-18's joins conserve |
| **2 — the switch** | per-window mode selection, classical/learned/certified, with per-window one-step error and per-call cost | a headless script that runs any mode assignment and reports §5.2 and §5.3 |
| **3 — the dashboard** | the server, the page, the field overlay, the inspector, the sliders | `python -m atlas.demo_racelab --open` works on this machine |
| **4 — three dimensions** | the 3-D half-car, a 3-D window solver, classical-only | the same dashboard with a 3-D toggle, and the learned switch greyed out in 3-D with the reason |
| **5 — the bundle** | `poc3-racelab-demo` branch, launchers, self-test, licences | a clone-and-run on a machine that has nothing, verified on Windows **and** one of macOS/Linux |

**Phase 4 is optional to ship.** Phases 1–3 and 5 are a complete PoC. If 3-D
proves expensive, ship without it and say so.

## 9.1 The open decision: a learned expert in 3-D

**Not decided.** Poseidon is 2-D, so Phase 4 as specified is classical-only.
The alternative is to **train a 3-D window operator** on data generated by the
new 3-D solver, on rented A100s — which would also be the rung-5 expert the
ladder has been waiting for since it was written, so it counts twice.

**Default: Phase 4 ships classical-only and the demo says so on screen.** The
training route is a separate decision to be taken after Phase 4 exists, when the
3-D solver's cost per sample is known rather than guessed. It must not be taken
before, because the data-generation budget cannot be estimated without it.

---

# 10. Risks, named in advance

| risk | why | mitigation |
|---|---|---|
| **the domain is too slow to be interactive** | $384\times192$ with 12–20 windows is 3–4× the front-wing case | measure a macro-step in Phase 1 *before* committing to the size; the demo may march a coarse field for display and a fine one on demand |
| **Poseidon is out of distribution on this geometry** | it was trained on neither cars nor ground effect | this is expected and is a *result*, not a failure — report the per-window error and let certified mode show the repair |
| **the learned mode does not pay** | Tiers 48 and 50 measured that it does not, against classical coarsening | say so on screen; the demo's claim is that the mechanism works and is measurable, not that it is fast |
| **wheels and moving ground are new physics** | no existing case has either | keep both to the simplest defensible form and name the simplification |
| **windows crossing bodies** | a ride-height change can move a body across a boundary | clamp, or rebuild the decomposition with a visible state |
| **the 400:1 clock ratio** | Tier 49 measured that no single march spans the fluid and the coolant clocks | sub-cycle as `vehicle_march` does; the dashboard shows each subsystem's own clock |
| **bundle size** | Poseidon-T is 83 MB; the build repo and a 3-D solver add more | keep 3-D fields out of the bundle; ship the generator, not the data |

---

# 11. Explicitly out of scope

- **No optimizer and no design search.** The user asked for none, and rung 10's
  gradient work is a different question.
- **No claim that this replaces CFD.** The deployment story is unchanged: search
  wide and cheap, verify the shortlist classically.
- **No compressible flow, no combustion, no tyre contact model.**
- **No new governing family and no new port type.** If RaceLab needs one, that is
  a finding to record, not a thing to add quietly.
- **No training run** without an explicit decision (§9.1).
- **NeuberNet anywhere near the bundle.**

---

# 12. Definition of done

RaceLab is done when:

1. A person with a laptop, a network connection and no prior setup can clone one
   branch, run one command, and see a car being simulated — on Windows and on
   macOS or Linux.
2. They can move a parameter and watch every coupled subsystem respond.
3. They can click any fluid window, flip it to a learned expert, and see its
   error and its speed change — per window and globally.
4. They can flip it again to certified and watch the error go to the classical
   answer, and see what that cost.
5. Every number on screen is traceable to a recorded run in the vault.
6. The dashboard names which families have no learned option, and why.
7. The full suite is green and `vault_scan` reports 0 problems.

---

# 13. Amendments

## 13.1 2026-09-14 — the user's decisions after the Tier 59 review

Everything above stands as written; where these conflict with it, these win.
Each row says what is built, so the document never describes a demo that does
not exist.

| section | as written | decided | status |
|---|---|---|---|
| §4.1 windows | 12–20 fluid windows laid out along the body, with `wing_fsi`'s halo and partition of unity — which means rectangles | **Body-fitted grids.** Curved grids wrapped round each part of the car, overlapping a Cartesian background, with the bodies as real walls rather than porous body forces. The background keeps rectangular windows, and the learned expert runs only there, because Poseidon-T accepts nothing but a uniform 128 × 128 grid. | **Built through the car's first march**: the grids, the overlap and the pressure solve (Tier 60), the flow solver (Tier 61), and the drawn car as solids marching on them (Tier 62, [[poc3-racelab-car-solids]]). The devices, joins, agents and demo on them are not. |
| §4.2 certified mode | a per-window mode | **Live.** An implicit fluid time step, so that a macro-step has a fixed point for defect correction to certify; the classical column becomes that implicit step (W226's second option, not an amendment of §4.2 to a steady-state mode). | not started |
| §3.3 parameters | thirteen sliders | **Powertrain and cooling** — state of charge, battery power draw, coolant mass flow, ambient temperature — **and geometry** — ride height, rake, diffuser angle, front flap, rear wing, radiator duct area — with a visible recompiling state for geometry. Brake and plate-stiffness knobs stay omitted: there is no brake subsystem and the structure is rigid. | not started |
| §3.2 and §9, phase 4 | a half-car in a box, classical only | **A real Formula One car's CAD model**, after the two-dimensional work is done. Its licence is checked and its download approved before anything is fetched. | not started |

## 13.2 2026-09-15 — the solids, and what the first march found

| section | as written | decided | status |
|---|---|---|---|
| §3.1 the car | plates of zero thickness, porous | **Solids by a declared, editable rule**: wing elements aerofoil-like at about 12% of the chord, body and floor panels a few cells thick, touching shell plates welded into one closed body, every thickness in `car_geometry.json`; **wheels about 2% of their radius above the road, rolling at road speed**. | **Built** in Tier 62. |
| §3.4 cooling duct | a duct fed by the flow through porous plates | **Openings cut by the rule.** With solid walls the drawn pod has no inlet or outlet and its duct flowed backwards in Tier 62 (W278). The solids rule cuts an inlet in the body shell ahead of the duct and an outlet behind it, sized by editable numbers in `car_geometry.json`, and the duct's flow is measured before the radiator core and the turbine are put in it. | not started |
| §3.1 front wing | the drawn plates | **A better grid generator**, so the front wing keeps its drawn shape. Welded, its endplate outline could only be gridded after a fillet filled it 92%, and the wedge that marched made lift (W275, W279). Grids that wrap the notch without filling it are built and verified, and the car is re-gridded on them. | not started |
