# PoC 2 — the live front-wing demo

A front wing riding at height `h` on a moving floor, bending under its own aero
load, with **both interfaces live at once** and **every seam's certification
verdict on screen**, taken from the compiler rather than from a rule of thumb.

```bash
python -m atlas.demo_frontwing --open
```

Then open <http://127.0.0.1:8012/>. On Windows PowerShell the same command works;
there is no shell script to invoke.

It needs `fastapi`, `uvicorn`, `pillow` and `cma` beside the package's own
dependencies, and it needs the build repo for `ThermoStruct2D` — the structural
expert — exactly as `atlas/cases/wing_fsi.py` does. Set `ATLAS_BUILD_REPO` if the
checkout is not at `~/physics-foundation-model`.

Run `python scripts/w141_poc2_frontwing.py --stages setup` first if
`out/w141/settled.npz` is absent: without it the demo releases from the
freestream, says so on screen, and shows a transient that no number on
[[poc2-frontwing-results]] was measured at. `out/w141/w141.json` supplies every
**recorded** figure on screen and its absence is reported rather than papered
over — see *the recorded run*, below.

**To hand this to somebody else**, build the self-contained bundle:

```bash
python scripts/w146_build_frontwing_bundle.py --commit
```

which produces a ~4.9 MB directory that clones and runs on macOS, Linux or
Windows with one command and **no network access at run time**. See
`bundle/README.md` for what is in it and how each piece resolves.

---

## The five beats

The page is a tab per beat over a march that never stops. Each carries a
**plain-language** description and a **technical** one side by side, authored in
`explain.py` rather than in the template so they can be reviewed, diffed and
tested — `tests/test_tier31_frontwing_demo.py` checks every number they quote
against `out/w141/w141.json` and `out/w153/w153.json` and asserts the vocabulary
this project has decided it has not earned is absent.

**The order is the argument, and it is `explain.BEATS`'s order alone.** The tab
nav, the panel sequence and each panel's heading and ordinal are all built from
that tuple at load time; nothing in `static/index.html` writes a beat's position
down a second time, and `::test_the_pages_own_tab_nav_produces_the_new_beat_order`
runs the page's own `tabsOf` through node to read the order back off the screen
rather than out of Python.

It was **re-cut on 2026-09-08**. It used to open on the substitution refusal —
beat ④ below — which is this project's strongest claim and also a *negative
result*, so the first thing a reader who has not read the vault saw was the
compiler saying no. The positive result was already measured and already here,
third and fourth. Nothing was re-measured to change the order and no figure
moved: the refusal is not softened, it is sequenced, and *what this demo is not*
still carries the tension whole.

### ① `race.py` — a better wing, and it gets there four times sooner

Adam on the adjoint against CMA-ES on the same objective, **one evaluation each,
in turn**, with the live march held for the duration — this beat reports a
wall-clock ratio, and a 12-frames-per-second march in the background would be
measured instead of the search. Both columns are **warmed before either clock
starts**: unwarmed, a three-step race reads a 25× adjoint premium where a warmed
one reads 5.1, which is the same mistake `stage_cost` made when it published 9.47
against a re-measured 5.4–5.8.

The two budgets are **scaled** from the recorded 30 : 200 rather than chosen.
Below about nine gradient iterates Adam has not turned yet and the population
column wins on wall-clock; the screen says that rather than tuning it away.

Recorded: the search finds **+19.3%** downforce inside both ceilings, and gets
there **4.31× in wall-clock**, 22.2× in rollouts, quality 1.003× — and W143 is
why the first of those is quoted first.

### ② `DemoConfig.enforce` — and the win is real, not a constraint walked through

The first thing worth asking about a number like 19.3% is whether the search
cheated to get it. This beat is the receipt.

**put it on the floor** sets the free ride height and the spring rate to the
bottom of their range. That design settles below `Suspension`'s declared floor.
With the check on, the march stops at macro-step 0 and names the expert; with it
off, the same design marches to `h = 0.0067` against a floor of `0.0703` and
reports the largest downforce on the screen — within a percent of the recorded
pre-**W145** optimum, which is the number that fix removed.

Everything produced with the check off is stamped `OUTSIDE THE MODEL`: the page
turns red, the field gets a red border, and the readout flags four stats. The
switch does not remove the check — it records what the check *would* have said
into `outside` and marches on.

The recorded columns are side by side: **+19.3% enforced** against **+27.5%
unenforced**, same code, same box, same budget, same seed. `declined` and
`infeasible` are separate rows on purpose — 39 of the population's 200
evaluations were declined by an expert and 66 were infeasible, and merging them
would overstate W145 by a factor of 1.7.

**The live optimiser will not take you there on its own**, and the panel says so:
it is a warm-started twelve-step estimator and it ascends in a different
direction from the 120-step one the recorded columns used.

### ③ `ablation.py` — freeze one joint and see what it was worth

Three rollouts: the live design, `k → ∞` (which is CS-12), and `E* → ∞` (which is
CS-10). **The height the frozen suspension is pinned at is measured inside the
call**, from the live column it marches first, rather than passed in — the caller's
instantaneous ride height is not the height a fresh rollout settles to, and at the
optimum the two differ by 37%, which would be read as the seam's worth. A hint may
still be passed and its error is reported beside the measurement.

Recorded, the suspension seam goes **0.49% → 8.16%** between the reference design
and the optimum, and the elastic column at the optimum **declines**: a rigid wing
there makes more downforce and drives itself through the floor.

### ④ `substitution.py` — how you know when not to trust the fast answer

A search that wins on the clock is only worth having if you know when *not* to
believe it. Same graph, compiled three times with three **descriptions** of the
fluid expert. Nothing about any model changes; only the capability record does.

| description | verdict |
|---|---|
| `WindowNS`, the classical incumbent | 2 seams refused, 7 uncertified |
| Poseidon-T, as `poseidon_capabilities` declares it | 2 refused, 7 uncertified, with **three new rules** at every fluid seam |
| Poseidon-T with `elliptic_subsolve=embedded` | **9 of 9 refused** |

**The verdict is a property of the declaration, so this beat needs no weights** —
which is why the bundle carries no checkpoint and touches no network. The
measurement behind the middle row is **W93** and is quoted with its provenance
rather than re-run: `probe.support_reach` found Poseidon-T's response nonzero in
every one of the 128 seam cells, 64 from the poke, where the record declares 2.

The third row is **declared, not measured**, and the panel says so: W60 records
that `elliptic_signature` measures non-normality and was read as measuring
globality, so this field is genuinely undeclarable for a learned operator.

> **It needs six windows.** On the single-window referent the fluid family has
> one member, nothing is decomposed, and `L2/R10`'s premise does not hold — so
> the rule correctly stays quiet however the expert is declared. That is
> **W114**'s premise check working, and the payload carries a sentence saying so
> rather than letting a quiet rule read as a passing one.

**Underneath the three verdicts, `engine.load_horizon` — what was measured.**
The rule behind them rested on a 0/1 indicator: does influence reach past the
cut? At one exchange per macro-step that reads $\Pi = 1.0000$ for *every* agent
in this vault, which is the check being unavailable rather than a fact about any
of them. **W153** replaced it with the sensitivity it stands for, and the three
separate — `WindowNS` exposed at $\Pi_w = 0.186$, the same solver carrying its
own elliptic part at $0.496$, Poseidon-T at $0.790$, on a scale where 1 is
*everywhere*. The panel calls it **measured influence reach** in the plain voice
and $\Pi_w$ in the technical one.

The local column's far field beyond $b = 20$ — its declared $\rho s = 2\times10$
— is **bitwise zero** over all $13{,}924$ cells, and it is asserted across every
probed state rather than one, because `d_eff` moves between 19 and 20 over the
spin-up while `exactly_zero` holds throughout.

**This changes no verdict and could not have**: Poseidon-T is refused before the
measurement and after it, and at halo 1 the ratio $\Pi_w/\Pi$ is $1.000000$, so
no past verdict can have moved silently. What it changes is *why* — the compiler
is not pattern-matching "neural network, therefore no". Read from
`out/w153/w153.json`, never transcribed, and absent, the beat shows its three
verdicts alone.

> The mapping from bars to columns is deliberately **not** one-to-one: both
> Poseidon descriptions light the same bar, because the verdict is a property of
> the declaration and the reach is a property of the model. The embedded
> classical row lights none — it is the control that separates *global* from
> *global because it is learned*.

**The figure beside each bar is a decay, not a magnitude.** The far-field
Frobenius norm is the obvious thing to print and it ranks the three the *other*
way round — $76.07$ embedded against $9.78$ for Poseidon-T, because it is a raw
size in each agent's own units — so a bigger bar beside a smaller number would
read as a broken panel. What is printed is `profile_max` normalised by each
agent's own peak, worst over the probed states, at **two** distances: one would
let the panel call both non-local rows flat and only one of them is. At $b = 20$
and $b = 60$ the embedded solve reads $26\%$ then $12\%$ — *still falling,
slowly* — and Poseidon-T reads $61\%$ then $61\%$ — *it has stopped falling off*.
`::test_the_figure_beside_each_bar_ranks_the_same_way_the_bar_does` asserts the
ordering and that the Frobenius really does invert.

### ⑤ `balance.py` — one needle for energy at the seams

Three accountings of `R(t)`, live, per macro-step: the energy change alone, with
the interface power the two seams carry, and with the half-step term. Both
receivers' energies at release are carried, **and the spring's is the larger** —
the plate starts flat so its strain energy is identically zero, and forgetting
the spring's put the first macro-step at 1.81 against a static accounting's 1.00.

Recorded over 480 macro-steps: `1.0000` → `5.93e-2` → `3.30e-2`, and `3.32e-8`
once settled. **Press reset to see it**: after a march has settled all three
agree, because there is nothing left moving for the motion terms to account for.

---

## What is on the screen the whole time

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

**The per-seam certification panel**, which is where the whole idea lives.

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
**Beat ④ is where a colour actually changes**, and it changes because the
declaration does.

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

## The recorded run

`out/w141/w141.json` is the driver's own artifact and every **recorded** figure
on screen is read out of it at load time by `engine.load_recorded`. The demo
never writes it, and **never falls back to a hard-coded copy of a number in it**:
if it is absent, the recorded columns say so.

That rule is not fussiness. The PoC 1a demo shipped a table of
milliseconds-per-step that was wrong by a factor of four the next time anyone
measured it, on unchanged code, because the box was in a different power state.
A number and the run that produced it have to travel together or the number rots.

The live engine measures itself for everything else and the screen marks a figure
as an estimate until the first real one lands.

---

## The scheme controls

| | |
|---|---|
| **six windows / referent** | the composed column against the single-window referent. They differ by the cut alone, and over 480 macro-steps the cut's own defect is `9.0e-5` in the tip deflection and `2.2e-5` in the ride height, against the seams' lag at `5.4e-2` and `2.3e-3` — factors of 602 and 106. **Beat ④ needs six** |
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
- **This is not faster than the solvers it is made of.** Every expert here is a
  classical solver. The orders of magnitude that make the whole idea worth having
  arrive only with learned experts — and beat ④ is the framework refusing the
  most obvious candidate. That tension is the honest state of the programme.
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

## Layout

| file | what it is |
|---|---|
| `engine.py` | the persistent march, the certification worker, the three on-demand beat workers, `load_recorded` and `load_horizon` |
| `server.py` | FastAPI: binary websocket frames for the field, JSON for everything else |
| `substitution.py` | beat ④ — the candidate declarations and the verdict diff |
| `ablation.py` | beat ③ — freeze each seam, measuring its own operating point |
| `race.py` | beat ① — the two search columns, warmed, on one clock |
| `balance.py` | beat ⑤ — the running power-residual accounting |
| `explain.py` | **the prose**, in both voices, with the vocabulary check |
| `static/index.html` | the page |
| `bundle/` | the launchers and paperwork for the standalone branch |

## See also

- `wiki/concepts/Atlas 0.1/common/poc2-frontwing-results.md` — what is proved,
  what is not, the cost, the ablation and the limitations
- `wiki/concepts/Atlas 0.1/common/poc2-demo-and-novelty.md` — what this demo is
  for, and the novelty argument it is built to carry
- `atlas/demo/README.md` — the PoC 1a demo this is modelled on
- `atlas/cases/front_wing.py` — the assembly; `atlas/CASE-STUDY-GUIDE.md` — how a
  graph is declared
