# Outcome — interactive pages, runnable demos and figures

**Type:** Outcome page — **materials index** (folder: `Atlas 0.1/atlas-0.1-outcome/`)
**Status:** compiled 2026-09-26. Every item below existed before this page, except the evidence board. Each item carries the tier it was built at, because some were built before a later tier changed what they show.
**Hub:** [[00-atlas-0.1-outcome]]

---

## 1. Published interactive pages

These are private artifacts on the owner's account. A viewer can open one only after it is shared from the page's Share menu.

| page | what it shows | built | link | caveat |
|---|---|---|---|---|
| **Atlas 0.1 Outcome** (evidence board) | the five claims with verdicts; **classical decomposition's cost against the monolith by rotor count, serial and parallel (Tier 89)**; per-window cost crossover; cost against the monolith with the coarse competitor; learned error against lead; farm power against the monolith; the body-fitted car's reachable share; what is not done | 2026-09-26, this folder; version 2 on 2026-09-27 after Tier 89 | [claude.ai/artifact/VLuXJcetzkYuwFA2eChmw3](https://claude.ai/artifact/VLuXJcetzkYuwFA2eChmw3) | a synthesis page; every number is sourced on the claim pages |
| **Rocket Ascent Episode** | the seven-agent coupled march, field by field and step by step: seams, loads, trajectory, gates | Tier 88 + W343 (the current build, `98df350`) | [claude.ai/artifact/FkMdFnycRPNobgAzH5Vj6s](https://claude.ai/artifact/FkMdFnycRPNobgAzH5Vj6s) | the published page carries the same records as `out/w321/rocket-episode.html` (checked 2026-09-26: both name the W343 build). [[rocket-episode-pickup]] also names an earlier copy on another account |
| **Scaling Ladder** | composition error from 1 to 24 windows, classical and learned columns beside the monolith; animated fields; wall time per agent | Tier 19, 2026-08-30 | [claude.ai/artifact/L8WARWRmwrqq8e92in4Ctt](https://claude.ai/artifact/L8WARWRmwrqq8e92in4Ctt) | its "does the rollout survive" panel presents a global projection added to the classical column as the fix. Marched to 120 steps the same day, that repair was **falsified**; the surviving arrangement exposes the elliptic part instead ([[case-study-scaling-ladder-atlas-0.1]] §5) |

---

## 2. Local copies of the self-contained viewers (`interactive/`)

Each is a single HTML file with its data embedded. Open it in any browser; only the fonts come from the network.

| file | source | what it is | caveat |
|---|---|---|---|
| `interactive/atlas-outcome-board.html` | written for this folder | the evidence board above | |
| `interactive/rocket-episode.html` | `out/w321/rocket-episode.html` | the rocket episode (same file as the published page's content) | about 2.4 MB |
| `interactive/scaling-ladder.html` | `out/w100/scaling-ladder.html` | the scaling ladder | the falsified-repair caveat above |
| `interactive/wake-array.html` | `out/w93/wake-array.html` | three turbines in an L over six frozen Poseidon-T windows: the geometry, the agent graph, the fields and the array loss ($25.27\%$ and $14.62\%$) | built at Tier 18. [[case-study-scaling-ladder-atlas-0.1]] §5 later found the classical column's published state sits on a trajectory that diverges past macro-step 70; numbers measured *at* that state stand |

---

## 3. Runnable demos (local servers)

These need Python, the vault's `atlas/` package and, for the classical fluid solver, the build repo. The learned columns also need Poseidon-T in the local Hugging Face cache. Without it the classical columns still run and the page says why the learned one is greyed out.

| demo | what a viewer does | run | standalone branch | page |
|---|---|---|---|---|
| **PoC 1a — wind-farm design** | drag turbines; press optimise and watch a gradient taken through the whole coupled simulation move every turbine; compare composed, same-cut classical and undivided solvers, **timed on the viewer's own machine** | `python scripts/w112_farm_demo.py`, then open `http://127.0.0.1:8011/` (or `python -m atlas.demo`) | `poc1-windfarm-demo` (pushed) | [[atlas-proof-of-concept-1]] §11 |
| **PoC 2 — front-wing certification** | five beats: a wing with ride height and bending live at once, a constrained search, and the compiler's per-seam verdict, including a real neural operator **refused on screen** | `python -m atlas.demo_frontwing --open` | `poc2-frontwing-demo` (pushed; 4.9 MB, one command, macOS / Linux / Windows) | [[poc2-demo-and-novelty]] · [[poc2-novelty-audit]] |
| **PoC 3 — RaceLab, porous column** | a car as 13 bodies and 14 fluid windows; flip each window between classical, learned and certified; per-window one-step error and cost | `python -m atlas.demo_racelab --open`, port 8013 | `poc3-racelab-demo` (**built locally, not pushed**) | [[poc3-racelab-demo]] · [[case-study-racelab-switch-atlas-0.1]] |
| **PoC 3 — RaceLab, body-fitted dashboard** | the car the user drew, on curved overlapping grids with the duct, radiator and turbine; eight live knobs; the certified mode with its two prices; `learned` drawn refused with both reasons | `python -m atlas.demo_racelab --column body-fitted --open`, port 8014 | same branch | [[poc3-racelab-dashboard]] · [[poc3-racelab-certified-screen]] · [[poc3-racelab-showable]] |

**[AI Inference]:** for a live pitch, the PoC 1a demo shows C3 and C4 best, since it re-measures its speed ratio on the machine in front of the audience. The body-fitted dashboard shows C4 and why C5 is needed: the learned mode is refused on screen, with both reasons.

---

## 4. Figures (`figures/`)

| file | source | shows |
|---|---|---|
| `figures/poc1a-frozen-fields-K12.png` | `out/w118/fields_K12.png` | 12 turbines over 12 frozen-expert windows, before and after 30 gradient steps: farm power $2.478 \to 7.024$ ($+183\%$ as measured by the composed model; $+266\%$ when the undivided classical solver scores the layout) |
| `figures/poc1a-frozen-traces-K12.png` | `out/w118/traces_K12.png` | farm-power trace, evaluations to tolerance against CMA-ES on a log axis, and the yaw angles converging |
| `figures/racelab-car-solids.png` | `out/racelab11/car_solids.png` | the car the user drew as the solids that marched (Tier 62), with mean forces per part |
| `figures/racelab-overset-grids.png` | `out/racelab9/tier60_grids.png` | body-fitted overset grids: a body grid overlapping the background, generated grids around a NACA 0012 and a plate, and the pressure solve converging at second order (Tier 60) |

![[poc1a-frozen-fields-K12.png]]

![[racelab-car-solids.png]]

![[racelab-overset-grids.png]]

---

## 5. Not included, and why

- **The rocket seam PNGs** in `out/w321_prefix/`, `out/w321_seams_only/` and `out/w321_pre_w337/` show builds before the Tier 86–88 fixes.
- **`out/w100/look_N24.png`** renders with half its canvas blank and clipped titles.
- **A shared page titled "Wake Atlas"** appears in the owner's artifact list as shared from outside the organisation. It describes the wind farm's $N$-sweep on branch `atlas-0.1-windfarm` at `0a407b7`. It is not linked here until its origin is confirmed.

---

## See Also

- [[00-atlas-0.1-outcome]] · [[outcome-evidence-ledger]]
- [[poc3-racelab-bundle]] — how the RaceLab branch was verified from a fresh clone on Windows and Linux
- [[wind-farm-agent-graph-figure]] — the wind farm's to-scale agent graph (its HTML sits beside it)
