# The website's evidence — every number it may show, with its source and how sure we are of it

**Type:** Concept page — **evidence register for the proposal website** (folder: `Atlas 0.1/atlas-0.1-proposal/website/`)
**Status:** written 2026-09-30. §1 is quoted from the vault's measured pages, each backed by a record file. §2–§3 are **published results by others**, collected from web searches in the planning session: **arXiv and most publishers were blocked there**, so no paper was read in full. Each row carries a status, and **nothing marked S or U goes on the site until someone reads the paper and records the exact quotation** in `site/data/literature.json` ([[proposal-website-plan]] §3.1).
**Hub:** [[00-proposal-workstreams]] · **Plan:** [[proposal-website-plan]] · **The vault's own ledger:** [[outcome-evidence-ledger]]

**Status legend:** **R** measured in this vault and read from its page and record · **S** from a search result's summary of the source, which is to be read in full · **U** from general knowledge, unverified in the planning session.

---

## 1. Atlas's own numbers, available today

| # | claim, as the site could word it | number | conditions and caveat | source | status |
|---|---|---|---|---|---|
| A1 | *Solving a wind farm in pieces is faster than solving it whole* | **$4.83\times$** (12 rotors, 4 threads); **$4.55\times$** (21 rotors, 8 threads) | the served workbench, 2026-09-29, on AC power; the wakes' start-up only (12 and 6 macro-steps); labelled "showcase, not research record" | [[showcase-gallery]] §3; `out/workbench/records/step-f/` | R |
| A2 | *…across farm sizes, in a controlled measurement* | **$3.5$–$5.7\times$** from 5 to 21 rotors over two timing draws; $2.1$–$3.7\times$ at 36; farm power within $2.3$–$8.3\%$ of the undivided solve | Python threads on the dev laptop; the threaded arm is bit for bit the serial one. **Quote $3.5$, as the record rounds $3.499$**; the outcome hub's "$3.4$" is the same lower bound written as "at least" | [[decomposition-speed-by-rotor-count]]; `out/w346/w346.json` | R |
| A3 | *The pieces agree with the undivided answer to parts per billion* | every iterated example within **$3.6\times10^{-9}$** of its scale: $2.7\times10^{-13}$ (the wall), $3.6\times10^{-9}$ (the insert), $3.7\times10^{-11}$ (the bracket), $3.1\times10^{-10}$ (the bend), $1.8\times10^{-9}$ (the S-channel) | against the same discretization solved whole; the scale is the temperature span or the largest displacement | [[showcase-gallery]] §2 | R |
| A4 | *Nothing is lost at the seams* | every registered balance (mass, energy, charge, force) passes, from $10^{-16}$ to $10^{-8}$ against tolerances of $10^{-9}$–$10^{-6}$ | per example | [[showcase-gallery]] §2 | R |
| A5 | *Eight kinds of physics, one framework* | 8 families; styles A–D and a split by physics; 4 of the 5 port types compiled | the classical showcase | [[showcase-gallery]] §2 | R |
| A6 | *A new case is declarations, not code* | **zero** hand-written coupling code per case | the compiler builds each scheme from declarations | [[outcome-c4-modular-multiphysics]] | R |
| A7 | *Tested* | 2,195 automated tests passing at the planning session's head | the full suite; the count changes with every build | `scripts/run_suite.py`; [[log]] 2026-09-30 | R |
| A8 | *A learned column can already outrun the classical solver…* | $3.15\times$ and $3.64\times$ faster than the undivided classical solver (12 and 24 windows) | **the ratio is a property of the hardware**: $0.43\times$ on a share of a 192-core server. The same checkpoint is biased and loses to a $2\times$-coarse classical solver. **Use only beside the missing-piece section's caveat, or not at all** | [[outcome-c3-learned-speed-at-scale]] §3 | R |
| A9 | *…but today's learned experts are not built for this* | S1–S12: for example, *accepts only a uniform $128^2$ grid*; *$61\%$ of a car's unknowns unreachable*; *runs at $0.067\times$ classical speed at the coupling's cadence* | the measured specification | [[outcome-c5-requirements-for-dd-native-experts]] §2 | R |
| A10 | *Measured, not asserted* | 89 tiers of pre-registered measurement | — | [[00-atlas-0.1-outcome]] | R |

**To come**, each only when its record exists:
- the fast example per type ([[demo-fast-examples-plan]]);
- the learned case with its competitor ([[demo-learned-case-plan]]);
- the fork and the crossing hole ([[demo-finish-plan]]).

---

## 2. Classical decomposition, published

| # | result, as the site could word it | source | status |
|---|---|---|---|
| L1 | *Domain decomposition runs on the largest machines ever built*: nonlinear FETI-DP/BDDC scaled to about **786,000 cores** | Klawonn, Lanser & Rheinbach, *Toward extremely scalable nonlinear domain decomposition methods for elliptic PDEs*, *SIAM J. Sci. Comput.* 37 (2015); a 2015 talk titled *Scaling nonlinear FETI-DP domain decomposition methods to 786452 cores*. **The core count and the machine (Mira, or JUQUEEN) to be confirmed from the paper** | S |
| L2 | *…and an FE$^2$ multiscale solver on all **458,752** cores of JUQUEEN* | Klawonn, Lanser & Rheinbach, inexact reduced FETI-DP for FE$^2$; publication to be identified | S |
| L3 | *With the right coarse space, iteration counts do not grow with the number of pieces or the material contrast* | Spillane, Dolean, Hauret, Nataf, Pechstein & Scheichl, GenEO, *Numer. Math.* 126 (2014) | U |
| L4 | *Better interface conditions make convergence much faster*: $1-O(h^{1/2})$ per iteration against $1-O(h)$ | Gander, *Optimized Schwarz methods*, *SIAM J. Numer. Anal.* 44 (2006); already cited by [[atlas-and-standard-dd-theory]] | U (in the vault) |
| L5 | *Each piece can take its own time step without losing energy* | Diaz & Grote, *Energy conserving explicit local time stepping for second-order wave equations*, *SIAM J. Sci. Comput.* 31 (2009) | U |
| L6 | textbooks for the section's fine print | Toselli & Widlund (Springer, 2005); Dolean, Jolivet & Nataf (SIAM, 2015); Quarteroni & Valli (Oxford, 1999) | U |

---

## 3. Neural operators and neural decomposition, published

| # | result, as the site could word it | source | status |
|---|---|---|---|
| N1 | *Learned solvers match classical ones run at 8–10× finer resolution: **40–80× faster** at equal accuracy* (2-D turbulence) | Kochkov, Smith, Alieva, Wang, Brenner & Hoyer, *Machine learning–accelerated computational fluid dynamics*, *PNAS* 118(21), e2101784118 (2021). The summary quotes: *"as accurate as baseline solvers with 8 to 10× finer resolution in each spatial dimension, resulting in 40- to 80-fold computational speedups"* | S |
| N2 | *A car's drag estimated **26,000× faster** than GPU CFD*, at about 3% error, from 500 training shapes | Li et al., *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*, *NeurIPS* 2023 | S |
| N3 | *Learned operators that take the geometry as input: heart simulations on personal computers* | Yin et al., DIMON, *Nature Computational Science* (2024). Its press summaries say "thousands of times" faster; **the exact figure and its baseline must come from the paper** | S |
| N4 | *Learned pieces assembled on unseen domains: **1–3 orders of magnitude** faster than a physics-informed network* | Wang et al., Mosaic Flows, *CMAME* 389 (2022). **The baseline is a PINN, not a classical solver; say so** | S |
| N5 | *Neural operators inside Schwarz iteration generalise to new geometries, with convergence theory* | Huang et al., *Operator Learning with Domain Decomposition for Geometry Generalization in PDE Solving*, *ICLR* 2026 | S |
| N6 | *…and scale to large 3-D solids from tiny training patches* | Secchi et al., Neural-Schwarz Tiling, arXiv 2605.12343 (2026) | S |
| N7 | *Neural operators fix what classical iterations are slow at: a uniform convergence rate* | Zhang, Kahana, Turkel, Zhang & Karniadakis, HINTS, *Nature Machine Intelligence* 6, 1303–1313 (2024) | S |
| N8 | *Machine-learned interface conditions for decomposition on unstructured grids* | Taghibakhshi et al., *NeurIPS* 2022 | S |
| N9 | *Local neural operators power system-level analysis* | Fabiani, Vandecasteele & Goswami, *Nature Machine Intelligence* 8, 1127–1141 (2026) | S |
| N10 | *Neural operators on general geometries* | Li, Huang, Liu & Anandkumar, Geo-FNO, *JMLR* 24 (2023); Wu et al., Transolver, *ICML* 2024 | U |
| N11 | *A review of neural operators' speed-ups in science and design* | Azizzadenesheli et al., *Nature Reviews Physics* 6 (2024) | U |
| N12 | *(context)* learned weather models forecast in minutes what took hours | Lam et al., GraphCast, *Science* 382 (2023) | U; optional |

**Framing rule for this section.** These are other people's results on other problems. The site shows them as **the field's momentum**, never beside Atlas's numbers as if they were comparable, and never summed.

---

## 4. Charts on the site

Every chart is built from `claims.json` with one palette and one set of rules. **The website chat loads the `dataviz` skill before drawing the first chart.** Ratios are drawn on a log scale when they span decades (A3's agreements). A speed-up is a bar against a line at $1\times$, never a pie.

---

## 5. Sources for the design notes of [[proposal-website-plan]] §2

- **antigravity.google**: design summaries at `design.withfudge.com/share/antigravity.google-design` and `fontofweb.com/tokens/antigravity.google` (search results, 2026-09-30): Google Sans Flex; primary $\#121317$; white surface; black pill actions; particle fields; 4 px spacing.
- **googlebook.google**: the launch described by Google's blog (*Introducing Googlebook, designed for Gemini Intelligence*) and by 9to5Google (2026-09-21).
- **simscale.com**: its title, *"AI-Native Engineering Simulation Software in the Cloud"*, and its press pages on engineering AI agents (2026).

---

## See Also

- [[proposal-website-plan]] — how these are shown
- [[dd-neural-prior-art-2026]] — the same neural works read for positioning rather than for display
- [[outcome-evidence-ledger]] — the vault's full ledger, with caveats
- [[showcase-gallery]] — the demo's records
