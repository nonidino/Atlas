# The website's evidence — every number it may show, with its source and how sure we are of it

**Type:** Concept page — **evidence register for the proposal website** (folder: `Atlas 0.1/atlas-0.1-proposal/website/`)
**Status:** written 2026-09-30. §1 is quoted from the vault's measured pages, each backed by a record file. §2–§3 are **published results by others**, collected from web searches in the planning session: **arXiv and most publishers were blocked there**, so no paper was read in full. Each row carries a status, and **nothing marked S or U goes on the site until someone reads the paper and records the exact quotation** in `site/data/literature.json` ([[proposal-website-plan]] §3.1).
**Updated 2026-10-02 by the website chat (W354, step 2):** §1.1 adds the rows that now have records (the Fast examples, the learned case, the proofs, the architecture document's measurements, the installer), each read from its record file. **§3.1 records the verification of the literature**: each row the site may show was read at its source on 2026-10-02, and its exact words, location and DOI are written there and in `site/data/literature.json`. **Rows that could not be read are off the site** (§3.1's last table). The claim *"no machine-checked domain-decomposition theory was found before this"* was searched again and is reworded (§3.2).
**Hub:** [[00-proposal-workstreams]] · **Plan:** [[proposal-website-plan]] · **The vault's own ledger:** [[outcome-evidence-ledger]] · **Design notes:** [[website-design-notes]]

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

*(These now exist: §1.1.)*

### 1.1 Added 2026-10-02, each read from its record

**Machine for every timed row of A11 and A12:** the owner's Windows 11 laptop, 22 logical processors, on mains power, no other Python process at the start (each record's `machine` block). **A1 is superseded on the site by A11's farm row**: the Fast example is the demo's own farm and is the number a visitor can reproduce. A2 stays as the controlled measurement across farm sizes. **A7 and A10 stay off the site**: neither has a run record (the test count lives in the log, and changes with every build; the tier count is the vault's internal bookkeeping).

**A11, the Fast examples, one per kind** (`out/workbench/records/fast/confirm-*.json`; mechanism P = parallel pieces that stay in cache, M = multirate time steps; $s=t_{\text{full}}/t_{\text{arm}}$):

| # | kind, example | $s$ | agreement with the full domain (bar) | conditions | the mechanism, or the limit | record |
|---|---|---|---|---|---|---|
| A11a | wind farm, `fast-farm` | **4.62** | farm power 1.84% (3%) | 163,328 cells, 12 windows, 4 threads, 30 macro-steps, 48.9 s | P | `confirm-fast-farm-20260930-232234` |
| A11b | heat conduction, `fast-heat` | **5.58** | $1.22\times10^{-4}$ of the 100 K span ($10^{-3}$) | 245,760 cells, 2 pieces, 1,000 steps, 35.3 s | M: the copper spreader takes its own short steps | `confirm-fast-heat-20260930-232435` |
| A11c | river plume, `fast-river` | **5.85** | $4.7\times10^{-17}$ of the peak ($10^{-9}$) | 460,800 cells, 6 windows, 6 threads, 200 steps, 42.4 s | P, 24 steps per exchange | `confirm-fast-river-20260930-233019` |
| A11d | sound, `fast-sound` | **4.67** | the full domain bit for bit ($10^{-9}$) | 296,000 cells, 4 windows, 4 threads, 130 steps, 26.6 s | P, 16 steps per exchange | `confirm-fast-sound-20260930-232408` |
| A11e | loaded structure, `plate-hole` | 0.008 | $3.2\times10^{-12}$ | 12,800 cells | **limit:** the whole is one direct solve, factored once; the pieces iterate. Substructuring could win and is not built | `confirm-plate-hole-20260930-230850` |
| A11f | current in a plate, `plate-circuit` | 0.275 | $1.1\times10^{-13}$ | 12,800 cells | **limit:** as A11e | `confirm-plate-circuit-20260930-230910` |
| A11g | heated structure, `bimetal-arc` | 1.17 | the split's own controls, bit for bit | 11,664 cells | **limit:** split by physics, which can at most halve the time | `confirm-bimetal-arc-20260930-230912` |
| A11h | cooled block, `cooled-block` | 0.004 | $2.6\times10^{-10}$ | 12,000 cells | **limit:** as A11e | `confirm-cooled-block-20260930-230915` |

**How the site words it:** *four of eight kinds run $4.6$–$5.9\times$ faster in pieces; the other four are faster solved whole, and each card names why.* The ratios belong to these 2-D cases on this laptop ([[showcase-gallery]] §8).

**A12, the learned case** (`out/learned-case/gate-20261001-154054.json`, 40 macro-steps of `farm-12`, layout held out of training, evaluated once; errors against the truth run $T$, the full domain at twice the resolution, at macro-step 12):

| arm | what runs | mean macro-step | farm power vs $T$ | velocity vs $T$ |
|---|---|---|---|---|
| F | the full domain | 3.70 s | 0.51% | 0.15% |
| Ep | the classical decomposition, 4 threads | 0.757 s | 7.53% | 3.81% |
| **L** | **the same windows, each stepped by the network** | **0.497 s** | **7.50%** | **3.82%** |
| Cc | the classical decomposition on a grid twice as coarse | 0.169 s | 3.63% | 3.15% |

- **Passed:** G1 speed ($t_{Ep}/t_L=1.52$, $t_F/t_L=7.45$), G2 accuracy, G3 conservation ($5.9\times10^{-18}$), G4 stability, G6 held out.
- **Failed: G5.** The coarse classical decomposition is closer to $T$ than L in both measures at both horizons (12 and 40), in about a third of L's time.
- **How the site words it:** *the learned pieces matched the classical pieces and ran 1.5 times faster. A classical solver on a coarser grid did better still, so the gate failed, as it was registered to.*

**A13, the proofs** (`out/lean/axioms.json`, `out/lean/builds.jsonl`, `out/lean/t3_counterexample.json`, the Lean sources; [[formal-proofs-record]] §7.11–§7.12):

| # | claim | number | source |
|---|---|---|---|
| A13a | declarations whose axioms were checked by the kernel, none failing | **142**, 0 failing | `axioms.json` (2026-10-01 19:51) |
| A13b | `sorry` in the project | **0** | `axioms.json` (`with_sorry`) and the clean build's log |
| A13c | axioms used | only the three standard ones: `propext`, `Classical.choice`, `Quot.sound` | `axioms.json` |
| A13d | theorems the owner approved before they were proved | **105** | [[formal-proofs-record]] §7.11's table |
| A13e | `theorem` and `lemma` declarations in the source | 140, in 26 files | counted from `lean/AtlasProofs/` at build time |
| A13f | a clean re-check of the whole project | 100.4 s, the laptop **on battery** | `builds.jsonl`, `verify-integration-clean` |
| A13g | the finding | T3 as first written is **false** for restricted additive Schwarz; a counterexample is checked, and the corrected statement needs a convergent sweep | `t3_counterexample.json`; [[formal-proofs-record]] §5.1 |

**A14, the architecture document's own measurements.** **These are not training results: no chart operator has been trained.** They measure an untrained network's cost and what the energy Gram does with errors put in by hand.

| # | claim | number | conditions | record |
|---|---|---|---|---|
| A14a | the forward cost of the proposed network | 16.9 ms per $64^2$ piece | the 1.95 M-parameter trunk, **untrained**, one piece at a time, 4 threads of a cloud container's CPU | `out/arch/backbone_timing.json` |
| A14b | errors in the modes enter the port matrix squared, and the pieces stay positive | with field errors of 10%: the Gram's port matrix is off by 0.90% and **every piece stays positive semidefinite**; predicting the matrix directly is off by 10% and **12 of 16 pieces turn indefinite** | 4 × 4 steady pieces of $32^2$ cells, 16 modes per edge; **errors are synthetic perturbations of exact modes** | `out/arch/superelement_gram.json` |

**A15, the installer** (`out/workbench/records/installer/`): a fresh clone's `run.cmd --check` on Windows exits 0 in 229.4 s, which installs everything and runs the self-test on all eight kinds (pip's download cache already warm, so this is not a first-download time); `./run.sh --check` on Linux installs and self-tests in 55.3 s and exits 3 because the optional Gmsh will not load there (`libGLU.so.1`); **macOS: not verified, no Mac was available.**

**A15, updated 2026-10-03: the install runs from `atlas-0.1`.** The install branch `atlas-workbench` was never pushed, so the site's commands failed for the owner (*"Remote branch atlas-workbench not found"*). The workbench now runs from the root of `atlas-0.1`. The launchers are copied unchanged from `atlas/workbench/bundle/`. The wind farm's solver sits in `vendor/`, copied from the public `poc1-windfarm-demo` and `poc2-frontwing-demo` branches, where the copies are identical. The unpushed bundle used build-repo commit `98df350` instead, which the machine that placed this copy could not reach; whether the two differ is not known (`vendor/README.md`). **Checked 2026-10-03 on a cloud machine** (Linux, Python 3.12): a fresh `--depth 1` clone of the branch ran `./run.sh --check`. It built `.venv`, installed the pinned packages and CPU torch, and ran all eight kinds, the learned case (the trained weights load and step) and the page. It exited 3, because the optional Gmsh would not load (`libGLU.so.1`), as on the Linux record above. **That run wrote no record file**, so the site's timed installer numbers stay the first branch's, and their caveat says so. Windows has not yet been run from `atlas-0.1`.

**A16, the wind farm as it grows (added 2026-10-03).** Every rung of `out/w346/w346.json` gives decomposition's speed over the undivided solve, at the best thread count of each timing draw. Claims `w346.r<n>.a` hold the first timing and `w346.r<n>.b` the repeat; they are drawn as the chart beside A11's top three. The speeds by farm size:

| rotors | first timing | repeat timing | accuracy record |
|---|---|---|---|
| 1 | **0.64×**: the whole is faster | not taken | yes |
| 3 | 1.29× | not taken | yes |
| 5 | 5.37× | 3.50× | yes |
| 12 | 5.69× | 4.06× | yes |
| 21 | 4.35× | 4.70× | yes |
| 36 | 3.75× | 2.14× | **none** |

A2's range, 3.5×–5.7× over 5 to 21 rotors, is the minimum and maximum of the same rows.

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

### 3.1 Verified on 2026-10-02

Each row below was read at its source on 2026-10-02: the arXiv abstract or full text, the publisher's or the author's archive, and Crossref for the volume, pages and DOI. **Status V** means the words quoted are the source's own. The same entries, with the date read, are `site/data/literature.json`.

| # | what the site may say | the source's own words | where | reference |
|---|---|---|---|---|
| L2 | *A domain-decomposition solver ran on all 458,752 cores of a supercomputer* | "scalability on all 458,752 cores of the JUQUEEN BlueGene/Q system at Forschungszentrum J&uuml;lich is demonstrated" | the abstract | M. Lanser, *Nonlinear FETI-DP and BDDC Methods*, PhD thesis, University of Cologne (2015), `kups.ub.uni-koeln.de/6304` |
| L4 | *Better conditions at the seams make the pieces agree faster* | "They converge uniformly faster than classical Schwarz methods" | the abstract | M. J. Gander, *Optimized Schwarz methods*, SIAM J. Numer. Anal. 44(2), 699–731 (2006), doi:10.1137/S0036142903425409 |
| N1 | *Learned corrections inside a fluid solver: 40–80× faster at equal accuracy* (2-D turbulence) | "our results are as accurate as baseline solvers with 8-10x finer resolution in each spatial dimension, resulting in 40-80x fold computational speedups" | the abstract | D. Kochkov et al., *Machine learning–accelerated computational fluid dynamics*, PNAS 118(21), e2101784118 (2021), doi:10.1073/pnas.2101784118 |
| N2 | *A car's drag coefficient 26,000× faster than GPU CFD, trained on 500 shapes* | "The cost-accuracy experiments show a $26,000\times$ speed-up compared to optimized GPU-based computational fluid dynamics (CFD) simulators on computing the drag coefficient"; "using only five hundred data points" | the abstract | Z. Li et al., *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*, arXiv:2309.00583 (2023) |
| N3 | *A heart's electrical wave on a new patient's geometry: under 1 s on one GPU, against 5 hours for the finite-element solver* | "it only requires less than 1 s on a single GPU to predict the ATs and RTs for a new patient, whereas it takes our solver openCARP 5 hours on 48 CPU nodes" | the preprint, §3.7 (its Table 1 says "48 CPU cores"; the site quotes the sentence and does not repeat the count) | M. Yin et al., arXiv:2402.07250 v1 (2024); published as *A scalable framework for learning the geometry-dependent solution operators of partial differential equations*, Nat. Comput. Sci. 4(12), 928–940 (2024), doi:10.1038/s43588-024-00732-2. **The published text was not reachable; the site cites the preprint for the words** |
| N4 | *Learned pieces assembled on domains 1,200 times larger than they were trained on* | "domains of unseen shapes and BCs that are, respectively, 1200 and 12 times larger than the training domains" (Laplace and Navier–Stokes) | the abstract | H. Wang, R. Planas, A. Chandramowlishwaran, R. Bostanabad, *Mosaic flows*, Comput. Methods Appl. Mech. Eng. 389, 114424 (2022), doi:10.1016/j.cma.2021.114424 |
| N5 | *Neural operators inside Schwarz iteration, with convergence theory, generalise to new geometries* | "we devise an iterative scheme Schwarz Neural Inference (SNI) … we provide a theoretical analysis of the convergence rate and error bound" | the abstract | J. Huang, K. Zhang, Y. Wu, Z. Cheng, arXiv:2504.00510 (v2, ICLR 2026) |
| N6 | *A solver learned on $3\times3\times3$ patches, tiled to large 3-D solids* | "NEST learns a neural operator on minimal voxel patches ($3 \times 3 \times 3$) … and evaluate it on large, geometrically complex 3D domains far outside the scale of the training patches" | the abstract | P. Secchi, D. S. Balint, M. Maurizi, arXiv:2605.12343 (2026) |
| N7 | *A learned operator paired with a classical iteration converges at a uniform rate* | "resulting in a uniform convergence rate and hence exceptional performance of the hybrid solver overall" | the abstract (arXiv:2208.13273) | E. Zhang, A. Kahana et al. (seven authors), *Blending neural operators and relaxation methods in PDE numerical solvers*, Nat. Mach. Intell. 6(11), 1303–1313 (2024), doi:10.1038/s42256-024-00910-x |

**Corrections to §2–§3 found by reading:**
- **L1** (786,000 cores) was not found in the paper: a search summary places the 2015 SISC paper's largest run at 524,288 cores of Mira, and only the group's web page names 786,432. **L1 stays off; L2's thesis abstract replaces it.**
- **N2's "about 3% error"** is not in the abstract and was not checked in the body, so the site does not say it.
- **N7's authors** are seven, not five as §3 lists.
- **N3's "thousands of times faster"** came from press summaries; the paper's own sentence (above) is what the site shows.

**Not verified, so not on the site:** L1 (above); L3, GenEO (the publisher's page and HAL refused the request; Crossref has no abstract); L5 and L6 (not attempted); N8–N12 (not attempted).

### 3.3 Verified on 2026-10-03, for the page's two halves

Read at the source on 2026-10-03, for the owner's restructure ([[website-outline]], revision of 2026-10-03). The NASA records were read through the NASA Technical Reports Server's own API (the record and its abstract). The arXiv entries were read through the arXiv API's abstract. **Status V.** The entries are in `site/data/literature.json`.

| # | what the site may say | the source's own words | where | reference |
|---|---|---|---|---|
| L7 | *For flow around complex moving bodies, overlapping simple grids are among the most cost-effective routes to an accurate answer* | "For modeling flows with viscosity about geometrically complex bodies in relative motion, the Chimera-overset-grid method is among the most computationally cost-effective methods for obtaining accurate aerodynamic results." | the abstract | W. M. Chan, S. E. Rogers, S. M. Nash, P. G. Buning, R. Meakin, *Chimera Grid Tools*, NASA Tech Briefs, December 2005; NTRS 20110016440 |
| L8 | *Overlapping grids, one per component, computed the flow around the full Space Shuttle launch vehicle* | "Application of the collar grid scheme to the Orbiter fuselage and vertical tail intersection in a computation of the full Space Shuttle launch vehicle demonstrates its usefulness for simulation of flow about complex aerospace vehicles." | the abstract | S. J. Parks, P. G. Buning, W. M. Chan, J. L. Steger, *Collar grids for intersecting geometric components within the Chimera overlapped grid scheme*, conference paper, 1991; NTRS 19910056138 (the record names no venue, so the site names none) |
| N13 | *The Fourier neural operator ran up to three orders of magnitude faster than traditional solvers* (shown as 1,000×, labelled as three orders of magnitude) | "It is up to three orders of magnitude faster compared to traditional PDE solvers." | the abstract | Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar, arXiv:2010.08895 (2020); the arXiv record carries no journal reference, so the site cites the preprint |
| N14 | *Most learned solvers map a whole problem to a whole answer for one family of geometries: fast inside it, hard to reuse on new domains* | "Most learned PDE solvers follow a global-surrogate paradigm … This has enabled fast inference within fixed problem families, but limits reuse across new domains" | the abstract | Secchi, Balint, Maurizi, arXiv:2605.12343 (2026), as N6 |
| N15 | *The core obstacle is that a trained neural operator does not transfer to new geometries* | "At the core of the challenge lies the absence of transferability of neural operators to new geometries." | the abstract | Huang, Zhang, Wu, Cheng, arXiv:2504.00510 (v2, ICLR 2026), as N5 |

**Groups on the site:** L2, L4, L7, L8 under *What others have shown* in the decomposition half; N1, N2, N3, N13 in the learned-experts half; N14 and N15 cited in the chart operator's *why today's neural operators aren't built for the job*; N4–N7 under *Others have started joining the halves*.

### 3.2 "A first": searched again

A fresh search on 2026-10-02 (proof assistants Lean, Coq/Rocq and Isabelle against domain decomposition and Schwarz convergence) found **no machine-checked theory of domain decomposition**. It did find close relatives: the convergence of stationary iterative methods such as Jacobi formalised in Rocq, a Lean 4 formalisation of the Halpern and Krasnoselskii–Mann fixed-point iterations (arXiv:2602.17064), and a Coq proof of a wave-equation scheme's error (Boldo et al., arXiv:1005.0824). **So the site does not say "a first".** It says: *"We found no earlier machine-checked theory of domain decomposition (search of 2026-10-02); machine-checked proofs of related iterations exist."*

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
