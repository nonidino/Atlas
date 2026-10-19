# Wiki Build Log

Append-only record of wiki operations. Format: `## [YYYY-MM-DD] operation | description`

**Archive.** Entries before 2026-10 were moved, verbatim and in order, into one file per month on 2026-10-17: [[log-2026-04]] · [[log-2026-05]] · [[log-2026-06]] · [[log-2026-07]] · [[log-2026-08]] · [[log-2026-09]]. New entries are appended here.

---

## [2026-10-01] build | Proofs set-up: Lean 4 on Mathlib's cache, timed; the statements of steps 1 and 2 drafted; T3 for Schwarz is false as written

[[gap-worklist]] W353; the record is [[formal-proofs-record]], the plan [[formal-proofs-plan]] (§6 holds the status lines). **Nothing is proved yet**: every theorem below is a Lean statement that typechecks with a `sorry` for its proof, awaiting the owner's OK.

- **Installed:** elan 4.2.4, Lean 4.34.1, Mathlib `v4.34.1` from its prebuilt cache (never compiled here), `leanblueprint` 0.0.20. 13.2 GB of disk. The dependency graph needed no separate Graphviz.
- **The build runs outside OneDrive**, in `C:\Users\Nauni\.cache\atlas-lean`, mirrored from `lean/` by `scripts/lean_build.py`, which times every `lake` run into `out/lean/builds.jsonl`.
- **Timed, on battery at 1.4 GHz:** the one-time download 698 s; a file costs about 26 s before its first proof (10 s loading Mathlib's compiled files, 16 s running their start-up code); the nine statement files built in 81.6 s. Files that do not import each other build side by side, so the rule is a short import chain.
- **Mathlib's names confirmed** against the installed version. T1's distance bound is already a Mathlib lemma; Krasnoselskii–Mann is not in Mathlib.
- **Drafted, in Lean and in the blueprint:** T1 (with the defect needed only at the classical answer), T7 (with W208's lesson as the hypothesis, and the operator norm as the least certifying constant), T4 (recursion, three-term split, accumulation, three regimes), T3 (direct Schur, Dirichlet–Neumann in block form and in the workbench's face form, restricted additive Schwarz).
- **A finding: T3 for restricted additive Schwarz is false as the plan wrote it.** A three-unknown problem with exact window solves and a partition of unity has a fixed point that is not the solution, and the workbench's own `styles.schwarz` reports it `converged` after one sweep (`scripts/lean_t3_counterexample.py`). A symmetric positive definite example with three windows does the same in exact rational arithmetic, though its sweep does not converge. What holds: a sweep that converges from every start has one fixed point, the undivided solution; and stopping on the update leaves an error of the update over $1-\rho$.
- **An instrument that was wrong:** the build script's first `sorry` count read 0 on a log holding 32. Fixed and recounted; the claim "no `sorry`" rests on `scripts/lean_axioms.py`, which asks the kernel and whose positive control saw all 32.

**Not found:** the theorem drafts T24–T26 and the "tier 2" scoping the owner announced. No "For the proofs chat" heading was in [[chart-operator-architecture]] when this was written.

**Added:** `lean/` (the project, nine statement files, the blueprint), `scripts/lean_build.py`, `scripts/lean_axioms.py`, `scripts/lean_t3_counterexample.py`, `out/lean/builds.jsonl`, `out/lean/t3_counterexample.json`, `out/lean/axioms.json`, [[formal-proofs-record]]. **Changed:** [[formal-proofs-plan]], [[index]], [[gap-worklist]].

## [2026-10-01] build | Proofs: the statements of step 3 drafted (T5, T10, T11), none proved

[[formal-proofs-record]] §4.3; W353. Four more Lean files that typecheck with a `sorry` for each proof, and two blueprint chapters. Still awaiting the owner's OK on every statement.

- **T5, defect correction:** Theorem 1 (any limit is a fixed point of the classical map), also for inexact inner solves, which is what the code runs; Corollary 1 (a constant cheap map gives the classical march); Corollary 2 (the shrunk map's eigenvalues are $\alpha+(1-\alpha)\mu$, with real part at least $\alpha$). A precision: Theorem 1 needs **both** maps continuous at the limit, where [[defect-correction-learned-operator]] §2.1 names only $G_\Psi$.
- **T10, symmetry averaging:** the group average of any map is exactly equivariant; it returns an equivariant map unchanged; and averaging over $\{\mathrm{id},M_x,M_y\}$, not a group, is not equivariant.
- **T11, passive interconnection:** the connection rule conserves power pairwise, the connected ports' powers sum to zero over any graph, passive agents make a passive system, and incrementally passive agents make a non-expansive step ([[master-error-bound]] §6.1's theorem, in one step's energy balance).

The four new files built in 102.6 s on battery; the axiom check's positive control saw all 48 `sorry`. **Added:** `lean/AtlasProofs/DefectCorrection.lean`, `DefectCorrectionShrink.lean`, `SymmetryAveraging.lean`, `Passivity.lean`, two blueprint chapters. **Changed:** [[formal-proofs-record]], [[formal-proofs-plan]], [[index]].

## [2026-10-01] build | Demo step 6: a Fast example per simulation type

Item 1.4 of [[demo-finish-plan]], built to [[demo-fast-examples-plan]] (its §6 records where the build departs from the plan). The header's *Fast example* opens the chosen kind's case; its arrow lists the kind's other examples (O1). Each family registered its bars in its module before its first timed run. A sweep chose each configuration and a confirmation run from a fresh start measured it, on AC power. The other two chats were writing documentation and theory.

**Four of eight kinds meet both bars** ([[showcase-gallery]] §8):
- the wind farm, 4.62× (P), farm power within 1.84%;
- conduction, 5.58×, by a new style M: each piece at its own explicit step, refluxed so the balance closes to round-off, and the full domain to the bit with equal steps;
- the river, 5.85×, and the sound, 4.67× (P): the windows take several steps per exchange on a halo as deep, so they stay in cache, and their cells are the full domain's (the sound's bit for bit).

**The other four** load their fastest honest setup, and the card says what limits them (O3). The structure, the plate and the cooled block are one direct solve, which beats their iterated pieces (0.004–0.275×). The heated structure's split by physics can at most halve the time (1.17×).

**What the sweeps found:**
- With one exchange a step, the river's windows are memory-bound (0.84–1.02×).
- Multirate would change explicit upwind's numerical diffusion by percents.
- Subnormal floats in the plume's tail slowed a long march 3.4×; every arm now flushes them under $10^{-30}$ g/m$^3$.
- The farm needs 28 macro-steps for 3%, so its example runs two arms.
- The first confirmations (river 2.74×, sound 2.93×, the farm over two minutes) are superseded and kept.

**Step 0** (`scripts/fast_step0.py`): numpy windows scale to 3.6–3.8× from $256^2$ cells, the river's sparse step to 4–9×, and SuperLU releases the GIL up to 16,384 unknowns.

**Added:** `atlas/workbench/fast.py`, `scripts/fast_step0.py`, `scripts/fast_examples.py`, `tests/test_workbench_fast.py`, `out/workbench/records/fast/`. **Changed:** `styles.py` (`TwoRate`), `tiling.py`, `fv.py`, `spec.py` (four examples, `FAST_EXAMPLES`, `run.exchange_every`, style M), `app.py` (the button), `runview.py` (the card), every family module (`FAST`), the conduction, river and sound families, three test files, the README, [[demo-fast-examples-plan]], [[demo-finish-plan]], [[showcase-gallery]], [[gap-worklist]].

## [2026-10-01] build | Demo step 7: a one-command install of the workbench

The owner's O2, built to [[demo-finish-plan]] §4.1 from PoC 3's bundle code. A stranger runs `.\run.cmd` on Windows or `./run.sh` on macOS and Linux in a clone of the branch `atlas-workbench`. The launcher finds Python 3.10–3.12, builds `.venv`, installs the pins, then torch (CPU-only) and Gmsh as optional steps, runs a self-test once and serves the page. `scripts/build_workbench_bundle.py` makes the branch from a commit and never pushes it: it stays private until the website launches (O9).

**What verification on fresh clones found**, each fixed before the next round:
- **The first clean install answered HTTP 500 on every page load**, while every type ran. From param 2.3.0, with Panel 1.5.2, a widget's `name` is constant, and the header relabels its buttons by setting it. So the constraints pin the versions the suite passed with, and pip's own resolver installs exactly those, binary wheels only, for Windows, both kinds of Mac and Linux, on Python 3.10–3.12.
- **Behind a closed proxy, the self-test's fetch of its own page went to the proxy**, and a working page was called broken; it now goes direct.
- **On Linux, Gmsh installs and will not load** (`libGLU.so.1`). Four tests failed instead of skipping, and the import dialog would have met a raw `OSError`; both now say why.
- **A split by physics runs two arms that differ by design**, so the self-test asks each type for its own exact controls.
- **Help named wiki pages the install does not carry.**

**Verified** on the build from `6232415`:
- **Windows 11, Python 3.12.7:** every copied file identical by blob hash; the self-test passed; the bundle's tests 319 passed and 2 skipped; it passed without torch, with no network, and with both optional installs failing.
- **Ubuntu under WSL, Python 3.10.21:** the same checks; the tests 315 passed and 6 skipped, Gmsh's four among the skips.
- **In the served page at 1090×620:** the sound's Fast example ran to its card, 5.55× and bit for bit.
- **Not on macOS:** there is no Mac.

**Added:** `atlas/workbench/bundle/` (the templates), `scripts/build_workbench_bundle.py`, `scripts/verify_workbench_bundle.py`, `scripts/verify_workbench_pins.py`, `tests/test_workbench_installer.py`, `out/workbench/records/installer/`. **Changed:** `app.py` (Help), `gmsh_import.py`, two test files, the workbench README, `.gitignore`, [[demo-finish-plan]], [[gap-worklist]], [[index]].

## [2026-10-01] build | Proofs: the implementation plan, the mains timings, and the blueprint seen in a browser

[[formal-proofs-implementation-plan]], written at the owner's request; W353. **Still nothing proved**, and the owner's OK on the statements of batches 1 to 3 is still pending.

- **The plan:** about seventy statements to machine-check for the proposal, in six batches, each with its Lean route, size, effort estimate and place in the import chain; what is left for later with the reason and an estimate; what is cited, proved on paper or certified numerically, and why. Every effort figure is an estimate labelled as one, with a rule for replacing it by the measured rate after batch 1.
- **The architecture's version 1 was found on `origin/atlas-0.1`** (commit `02da03d`), not merged into the local branch, and read from the remote: three coupling tiers, the new tier-0 theorems T24 to T26 (the Gram superelement, its energy optimality, the perturbation of the interface solve, the label-free objective), T1, T8, T9 and T22's incomplete-solve term scoped to tier 2, and the SNI counterexample as a remark beside T1.
- **An expected finding, [AI Inference] until checked:** T9's two-level form, as version 0 worded it, asks only that each learned map contract on the fine space. A two-dimensional linear example has that property and diverges under an exact coarse solve, so T9 is planned on the composite sweep.
- **Measured on mains:** a full re-check of all 13 statement files takes 136.5 s on the first load of the day and 34.0 s once the files are cached; a first-level file costs 12 to 15 s against 26 to 30 s on battery.
- **The blueprint was opened in a browser** through a new `lean-blueprint` entry in `.claude/launch.json` (the owner's OK): mathematics typeset, the dependency graph drawn, the console clean.

**Not done:** the index row for the new page, because [[index]] and [[gap-worklist]] hold another chat's uncommitted edits; it is added once those are committed. **Added:** [[formal-proofs-implementation-plan]]. **Changed:** [[formal-proofs-plan]], [[formal-proofs-record]], `.claude/launch.json`, `out/lean/builds.jsonl`.

## [2026-10-01] note | The chart operator: the prior art read in full, three probes, and the owner's first eleven design decisions

The architecture chat (task 3), in a cloud container with open network, from 2026-09-30.

**Read in full:** the fifteen papers [[dd-neural-prior-art-2026]] named and seven more (SNAP-DDM, Mao and Fan's multilevel Maxwell solver, DD-DeepONet, the non-overlapping hybrid FE–neural operator, learned AGDSW coarse spaces, wave-domain Douglas–Rachford coupling of port-Hamiltonian subsystems, Robin–Robin coupling of operator-inference ROMs). The record is `out/arch/prior-art-reading.md`. **N1, N4, N5 and N6 stand; N2 and N3 narrow.** Robin transmission and coarse spaces with learned subdomain solvers are published. **SNI's convergence theorem (ICLR 2026) needs one more hypothesis**: a one-subdomain counterexample meets both of its hypotheses and cycles for ever (`scripts/arch_sni_counterexample.py`).

**Three probes**, all headless, classical, not timings of the workbench:
- `scripts/arch_chart_maps.py`: cutting the layout at equal along/across values gives every piece modulus about one (0.98–1.07, measured), and at a 68 and 148 degree corner the conformal chart's $J$ ratio is 291 where Winslow's is 3.2 and transfinite interpolation's 4.1, at the price of shear.
- `scripts/arch_coupling_cost.py`: rounds of local solves per coupling strategy with exact local solves. Steady 8 × 8 pieces: traditional Schwarz 1,864, version 0's wave exchange 1,124, with GMRES 83, with a coarse space 70, the superelement none. A monolithic direct solve beats every decomposition at these 2-D sizes.
- The first session's `scripts/arch_chart_probe.py` (the layout's pair is conformal) stands.

**The owner's decisions** (A2, B2, C1 on 2026-09-30; eleven in all by 2026-10-01): 2-D first with a named 3-D path; a chart rule; the unit square with topology owned by the decomposition; geometry as anisotropy, the fields $(\log J,\mu,\log\kappa)$; equal-value cuts; an exact discrete chart certificate and an envelope; charts of the smooth outline; **a three-tier coupling stack** (superelement, tangent, black box); GMRES or Anderson in tier 2 with a coarse space from 16 pieces; ordered sweeps in tier 2 only; reuse, warm start and energy-safe early termination in time. Under discussion: the expert's output format, the backbone and its size, the training route, the first family and gate, the name.

**Added:** [[chart-operator-design-decisions]] (the running record, with a draft theorem T24 for the proofs chat), [[coupling-cost-and-complexity]] (the measured cost model).

**Changed:** [[chart-operator-architecture]] (a status line pointing to the decisions; version 1 waits for the owner's confirmation), [[dd-neural-prior-art-2026]] (a status line: read in full, N2 and N3 narrowed), [[index]].

## [2026-10-01] note | The chart operator, version 1: the design confirmed, the proposal document written, the wiki rewritten

The architecture chat, phase 3 of [[proposal-chat-prompts]] §2. **No network was trained and no data generated**; the one learned measurement is the forward cost of untrained backbones. Nothing was published (O9).

**The owner's decisions:** the output format 6a–6e and the backbone 10a–10b as recommended; then **"Yes to the design and defaults"** for 11–19, the author line *Naunidh Singh*, the title left to the chat, MathJax from a CDN.

**Measured** (`scripts/arch_superelement_gram.py`, `scripts/arch_backbone_timing.py`): the port matrix formed as an **energy Gram** of the expert's constraint modes stays PSD and errs by about $0.9\,\varepsilon^2$, where a direct matrix head turns indefinite; coupled error at $\varepsilon=0.01$ 0.27% against 9.2% (steady). Forward cost per piece at $n=64$: 0.49 M parameters 6.0 ms alone / 3.8 ms batched, 1.95 M 16.9 / 7.7 ms, 6.2 M 30 / 25 ms, against 0.85 ms for a classical local solve and 68 ms to recompute a port matrix classically.

**Proved on paper** (not machine-checked): the Gram is PSD for every network output, its error is $E^\top A_IE$, the assembled system can only gain stiffness, and the coupled answer is Céa-optimal over the learned modes; corollary, **a label-free training objective with the same gradients as supervised energy-norm training**; a corner bound $K\ge\max(\tan\frac\alpha2,\cot\frac\alpha2)$ and an exact discrete chart certificate.

**Two refinements found while writing, reported to the owner:** step-doubling consistency does not hold for the backward-Euler step the tier-0 networks learn, so that penalty is kept for flow networks only; cosine port modes jump at the square's corners (finite but $\log n$-growing discrete energy), registered as comparison C6.

**Added:** `proposal/architecture/index.html` — *Chart Operators: Learned Superelements for Domain Decomposition on Curved Geometry*, the arXiv-style document (11 sections, 3 appendices, 11 numbered statements with proofs, 13 tables, 7 SVG figures from the workbench's s-channel and the probe outputs, 59 references with DOIs or arXiv ids checked against Crossref and arXiv); `scripts/arch_paper_figures.py`, `scripts/arch_paper_build.py`.

**Changed:** [[chart-operator-architecture]] rewritten as **version 1** (what changed from version 0 and why; §7 *For the proofs chat*: T24–T26, T1/T8/T9 now tier 2 only, T22's $\rho^k$ term tier 2 only); [[chart-operator-design-decisions]] closed (11–20 decided); [[chart-operator-training-and-cost]] (the energy-norm and label-free objectives, measured cost components, gates E1–E6, SE1–SE3, comparisons C1–C6); [[dd-neural-prior-art-2026]] §3.1 (verdicts N1–N7); [[gap-worklist]] W352 done for the proposal, build items open; [[index]].

## [2026-10-01] build | The architecture chat's branch merged; the proofs work prepared for two chats at once

At the owner's request, `origin/atlas-0.1` (the architecture chat's twelve commits) was merged into the local branch, merge commit `eb75132`.

- **Two conflicts, both in shared pages, each resolved by keeping both sides' text:** in [[gap-worklist]] the W352 row from the architecture chat and the W353 row from the proofs chat; in this log the ten local entries, then the architecture chat's two (which is why those two sit above this entry, after the local ones of the same day). [[index]] merged by itself. No file under `atlas/`, `tests/` or `lean/` changed.
- **Another chat's uncommitted edits to six wiki pages were kept as they were.** The merge was built in a temporary index and the working tree updated around those pages; for the two pages the merge also changes, the uncommitted edits were carried across and compared line for line, 80 edit lines before and after.
- After the merge: the vault scan reads 0 problems over 286 files, the link scan 0 new and 0 stale.

**Prepared for two chats proving different batches at the same time** ([[formal-proofs-implementation-plan]] §3): one proves batches 1 to 3, the other states and proves batches 4 to 6. `ATLAS_LEAN_TAG` gives each its own build, profile and axiom record files; `lean/decls-batches-1-3.txt` lists the first chat's declarations; [[formal-proofs-plan]] §6 and [[formal-proofs-record]] (§5A and §7) each have one block per chat, so their branches merge without touching the same lines. **Still nothing proved.** **Changed:** `scripts/lean_build.py`, `scripts/lean_axioms.py`, [[formal-proofs-plan]], [[formal-proofs-record]], [[formal-proofs-implementation-plan]]. **Added:** `lean/decls-batches-1-3.txt`.

## [2026-10-01] build | Formal proofs, batch 4: the tier-0 guarantee is machine-checked (T25, T24, T26)

**The first theorems of the Lean project are proved.** The chat that owns batches 4 to 6 stated tier 0 (the learned superelement), the owner approved the statements, and they are now checked: **28 theorems in five files, no `sorry`, each resting on Lean's three standard axioms only.** Record, statements and findings: [[formal-proofs-record]] §7.

- **T25 (i) to (iii), the Gram port matrix** (`lean/AtlasProofs/GramPort.lean`): for any fields a network returns, the port matrix the host computes as an energy Gram is symmetric positive semidefinite; it exceeds the exact one by exactly the Gram of the field errors in the $A_I$ inner product; the assembled interface matrix dominates the exact one and inherits every lower bound it has.
- **T25, energy optimality** (`EnergyOptimality.lean`, `Superelement.lean`): the host's interface system is the Galerkin system of the learned trial set, so the rebuilt field is the energy-closest field the network's modes can represent to the undivided discrete solution, and its error is at most a reference error plus the energy of the field errors. The document's three assumptions are named hypotheses.
- **T24** (`InterfacePerturbation.lean`): the perturbation bound of the interface solve, with the constant taken from the learned matrix or, by T25 (iii), from the exact one.
- **T26** (`LabelFree.lean`): the label-free objective equals the supervised energy-norm loss plus a constant with no network output in it.

**Measured, on battery:** a clean build of the whole project 56.0 s (23 files; the five new ones 16 to 23 s each); the axiom check 17 s. **The profile lists no tactic step of ten seconds**, but its count is not zero: ten `import took 10 to 12.5 s` lines, five of them in files that hold statements only, which is the battery's clock speed. It has to be re-run on mains.

**Findings** ([[formal-proofs-record]] §7.3): no assumption beyond the architecture document's was needed for its theorems; the optimality identity uses two of its three assumptions, the third only for uniqueness and solvability; the error bound holds for any reference modes and coordinates; the fixed coordinates' term of the host's right-hand side is written out.

**Also in this commit:** batch 5 (tier 2: T8, T9a, T9b with the example that contracting the fine space is not enough, the SNI remark) is **stated and approved**, 14 theorems with `sorry` proofs in four files ([[formal-proofs-record]] §7.4 to §7.6); two new blueprint chapters. **Not done here, and why:** the row for [[formal-proofs-implementation-plan]] in [[index]] and the status of W353 in [[gap-worklist]] are not written, because both pages hold another chat's uncommitted edits. **Changed:** `lean/AtlasProofs.lean`, `lean/blueprint/src/content.tex`, `lean/blueprint/lean_decls`, [[formal-proofs-record]] §7, [[formal-proofs-plan]] §6, [[formal-proofs-implementation-plan]] §3.2. **Added:** nine Lean files, `lean/blueprint/src/chapters/tier0.tex` and `tier2.tex`, `lean/decls-batches-4-6.txt`, `out/lean/*-batches-4-6.*`.

## [2026-10-01] build | Formal proofs, batch 5: the tier-2 theory is proved (T8, T9a, T9b, the SNI remark), with the two-level finding

**Batch 5 is proved: 14 theorems in four files, no `sorry` of their own.** Twelve are checked on Lean's three standard axioms. Two, the contracts of T9a and T9b, quote T1 (`Atlas.perturbed_contraction`), which the other proofs chat is proving on its own branch, so the kernel still reports `sorryAx` for them through that one name. Record: [[formal-proofs-record]] §7.4 to §7.6.

- **T8** (`lean/AtlasProofs/Cayley.lean`): $\lVert e-Zf\rVert^2\le\bigl(1-\tfrac{4Zc}{(1+ZM)^2}\bigr)\lVert e+Zf\rVert^2$ for any effort and flow, with no linearity; in finite dimensions $I+Z\Lambda$ is invertible and the scattering map has that norm bound; and **the Gram port matrix of any network output has a non-expansive scattering matrix for every impedance**, so a tier-0 expert serves at tier 2 with no certificate.
- **T9a** (`ContractionContract.lean`): $\rho$-Lipschitz experts and a non-expansive exchange give a $\rho$-Lipschitz sweep in the $\ell^2$ product; with T1, one answer, rate $\rho$, within $\bigl(\sum_i\delta_i^2\bigr)^{1/2}/(1-\rho)$ of the classical one.
- **T9b** (`TwoLevel.lean`): the contract from a certified contraction of the **composite** two-level round, whose answer is a fixed point of the plain sweep. **The finding is now a checked theorem**, `Atlas.two_level_fine_contraction_diverges`: the sweep $G=\begin{pmatrix}0&\tfrac12\\10&0\end{pmatrix}$ contracts the fine space by $\tfrac12$, has the single fixed point $0$, is followed by an exact coarse solve, and its two-level iterates from $(1,2)$ are $(5^k,\,2\cdot5^k)$. So "each learned map contracts on the fine space" does not give convergence, and a tier-2 card must certify the composite round.
- **The SNI remark** (`SniCounterexample.lean`): $T(u)=\tfrac12u$ contracts, $\tilde T(u)=\tfrac12(u-\tanh10u)$ is within $\tfrac12$ of it everywhere, and $\tilde T$ has a two-cycle $\pm x$ with $0.3\le x\le0.34$, by the intermediate value theorem.

**Measured, on battery:** clean build of the whole project 51.6 s; axiom check 17 s. **The profile reads zero this time:** 22 files, 307.2 s, no step at or over ten seconds, so the ten `import` lines of the batch-4 run were the machine's state at that moment, not the proofs.

**For the architecture chat, through the owner:** the sentence of the architecture document's §5.3 that the contraction hypothesis "is needed only on the complement of the coarse space" is contradicted by the checked example; and T9a's distance carries a $\sqrt n$ when $n$ experts are each within $\delta$. Neither document was touched from here. **Changed:** the four Lean files, `lean/blueprint/src/chapters/tier2.tex`, [[formal-proofs-record]] §7, [[formal-proofs-plan]] §6, [[formal-proofs-implementation-plan]] §3.2.

## [2026-10-01] build | Formal proofs, batch 6: Krasnoselskii–Mann, the learned master bound per tier, existence, and the three headline theorems

**Batches 4 to 6 are finished on this branch: 57 theorems, no `sorry` in any of this chat's thirteen files.** 46 are checked on Lean's three standard axioms. 11 quote theorems of batches 1 to 3 (T1, T4, T4c, T5), which the other proofs chat is proving on its own branch, and become checked at the merge; they are listed by name in [[formal-proofs-record]] §7.10. The owner approved batch 6's statements before the proofs.

- **T9′** (`lean/AtlasProofs/KrasnoselskiiMann.lean`), from scratch, since it is not in Mathlib: a map that lengthens no distance and has a fixed point, in a finite-dimensional inner-product space; the averaged iteration $x_{n+1}=(1-\omega)x_n+\omega T(x_n)$, $0<\omega<1$, converges to a fixed point. Checked.
- **T22, once per tier** (`LearnedMasterBound.lean`). Tier 2: each step's defect gains $C_\mu\delta/(1-\rho)$ and $C_\mu\rho^kD$ inside the master bound. Tier 0: no incomplete-solve term; the learned part of the defect is the Galerkin error, and a second statement puts T25's bound in its place.
- **T2, conditional** (`Existence.lean`): a map within $\eta$ of the classical sweep on a ball, in value at the answer and in increments, contracts there at $\rho+\eta$ and has its answer within $\eta/(1-\rho-\eta)$, provided $\eta\le(1-\rho-\eta)\,r$. That such a network exists is a hypothesis, the cited approximation theorem.
- **The headline statements** (`Headline.lean`), each one theorem that only quotes others: tier 0 (checked), tier 2, defect correction.

**Measured, on battery:** clean build of the whole project 79.3 s, 27 files in four levels of imports; axiom check 16 s; profile 26 files, 434.2 s, no step at or over ten seconds. All fifteen proofs compiled on the first build. T9′, which the plan held to be the risk, is about 115 lines of proof.

**Findings** ([[formal-proofs-record]] §7.9): none false, none short of a hypothesis after the statements were written. Stated with the theorems: T2 needs room in the ball; the tier-2 headline's $\delta$ is the experts' defects combined in $\ell^2$ and it has no coarse space in it; tier-0 T22 takes the step's stability in the energy size as a hypothesis; the defect-correction headline needs the classical problem to have at most one answer.

**Still open, and whose:** the full build and the full axiom check after the merge with the branch of batches 1 to 3 (on the owner's word that it is ready); the row for [[formal-proofs-implementation-plan]] in [[index]] and the W353 status in [[gap-worklist]], both pages still holding another chat's uncommitted edits. **Changed:** `lean/AtlasProofs.lean`, `lean/AtlasProofs/ContractionContract.lean` (one helper lemma), `lean/blueprint/src/content.tex`, `lean/blueprint/lean_decls`, `lean/decls-batches-4-6.txt`, [[formal-proofs-record]] §7, [[formal-proofs-plan]] §6, [[formal-proofs-implementation-plan]] §3.2. **Added:** four Lean files, `lean/blueprint/src/chapters/rollout.tex`.

## [2026-10-01] build | Proofs batch 1 checked: T1, T7, T4 and T4e, in a cloud container

The chat that proves batches 1 to 3, on its own branch `claude/atlas-proofs-batches-1-3-ztrott` from `428e4aa`, in a cloud container (Linux, 4 virtual processors, 15 GB), not on the laptop. The owner approved the statements of batches 1 to 3, T3's restatement included, before any proof.

- **Set-up:** elan 4.2.4 by the official script; `lake update` 153.6 s (the toolchain from `releases.lean-lang.org`, Mathlib `v4.34.1` and its dependencies cloned, its cache of 8908 files from `cache.mathlib.org`); the first build 17.2 s, 48 `sorry` warnings, no Mathlib module compiled; the axiom check's positive control saw all 48.
- **Checked:** the 18 declarations of batch 1, every statement as written: T1 (perturbed contraction), T7 (the certificate, its constant an operator norm), T4 (the master bound, its recursion, split, accumulation, three regimes and attained example) and T4e (the transmission term). The build 7.6 s with no `sorry` in a batch-1 file; a clean re-check of all 13 files 13.0 s; the axiom check 18 of 18 on `propext`, `Classical.choice` and `Quot.sound` only; the profile no step of ten seconds or more.
- **Rate:** 150 lines of proof, 6 min 11 s from the first proof edit to the batch verified, 18 min 42 s counting the reading and the set-up; the estimate was 2.5 to 4.5 hours. [[formal-proofs-implementation-plan]] §6.2 is rewritten from it, the old estimate kept beside it.
- **No finding.** The rows of batches 2 and 3 now read *approved*.

**Changed:** the four batch-1 files under `lean/AtlasProofs/`; `lean/blueprint/src/chapters/contraction.tex` and `master-bound.tex` (`\leanok` in 18 proofs, and a one-line proof for the two-mode example, which had none); [[formal-proofs-record]] (§3's cells, §5A); [[formal-proofs-plan]] (§6, the block of batches 1 to 3); [[formal-proofs-implementation-plan]] (§6.2). **Added:** `lean/decls-batch-1.txt`, `out/lean/builds-batches-1-3.jsonl`, `out/lean/axioms-batches-1-3.json`, `out/lean/profile-batches-1-3.json`.

## [2026-10-01] build | Proofs batch 2 checked: T3, its finding's counterexample included, in a cloud container

The same chat, branch and container as batch 1's entry. **T3's status line** in [[formal-proofs-plan]] §6 reads *checked* and was committed alone first (`bd7ddfe`), as the demo chat's compiler rule requires.

- **Checked:** the 14 theorems of batch 2 and the 7 definitions they rest on, every statement as written: T3a (a direct Schur solve gives the undivided solution), T3b (every fixed point of Dirichlet–Neumann solves the undivided system, in block form and in the face form `styles.dirichlet_neumann` runs, and conversely), T3c (the restricted additive Schwarz sweep in residual form; agreement of every window gives the solution; a stationary blend needs $M$ one-to-one; a convergent sweep has one fixed point, the solution, and converges to it), T3d (the error is at most the update over $1-\rho$, by T7), and the counterexample of [[formal-proofs-record]] §5.1: $A$ invertible over $\mathbb Q$, the sweep fixes $u=(1,-1,-1)$, and $Au=(0,0,-2)\ne0$. **The finding is now machine-checked.**
- The build 14.6 s with no `sorry` in a batch-2 file; a clean re-check of all 13 files 16.9 s; the axiom check 21 of 21 on the standard axioms only; the profile no step of ten seconds or more (the counterexample's largest step, a `simp` on the matrices, about 1 s).
- **Rate:** 135 lines of proof in 7 min 22 s from the first proof edit to the batch verified; the estimate was 3.25 to 5 hours. Batch 1's record-keeping took 9 minutes.
- **No finding.** Imports added: `Mathlib.Tactic.Abel` in three files, and `AtlasProofs.Certificate` in `SchwarzConvergence.lean` for T3d.

**Changed:** the five batch-2 files under `lean/AtlasProofs/`; `lean/blueprint/src/chapters/consistency.tex` (`\leanok` in 14 proofs); [[formal-proofs-record]] (§3's cells, §5A.3); [[formal-proofs-plan]] (§6, T3's line, in its own commit); [[formal-proofs-implementation-plan]] (§6.2, batch 2's measurement). **Added:** `lean/decls-batch-2.txt`. **Updated:** `out/lean/builds-batches-1-3.jsonl`, `axioms-batches-1-3.json`, `profile-batches-1-3.json`.

## [2026-10-01] build | Proofs batch 3 checked: T5, T10 and T11; batches 1 to 3 complete, in a cloud container

The same chat, branch and container as batch 1's entry.

- **Checked:** the 16 theorems of batch 3 and the 3 definitions they rest on, every statement as written: T5 (any limit of defect correction is a fixed point of the classical map, also with inexact inner solves; a constant cheap map gives the classical march; the shrunk map's derivative, its eigenvalues and their real parts), T10 (the group average of any map is exactly equivariant, leaves an equivariant map alone, and fails for the non-group $\{\mathrm{id},M_x,M_y\}$) and T11 (the port rule conserves power, connected powers cancel, passive agents make a passive system, incrementally passive agents a non-expansive step).
- **Batches 1 to 3 complete:** the project builds with **no `sorry`** (a clean re-check 16.7 s); the axiom check over the 58 names of `lean/decls-batches-1-3.txt`, run once as the check over all three batches, finds **58 of 58** on `propext`, `Classical.choice` and `Quot.sound` only; the profile no step of ten seconds or more. The blueprint's dependency graph shows all 46 theorem nodes fully proved.
- **Rate:** 108 lines of proof in 8 min 5 s from the first proof edit to the batch verified; over the three batches 48 theorems and 393 lines in 21 min 38 s, against an estimate of 9 to 13 hours. [[formal-proofs-implementation-plan]] §6.2 now holds all three measurements.
- **No statement false or short of a hypothesis.** Two smaller findings, hypotheses stronger than needed, found by Lean's linters and recorded in [[formal-proofs-record]] §5.2: T5 Corollary 2's real-part bound does not use $0\le\alpha$, and T10's idempotence does not use `SMulCommClass`. The statements were not changed.
- Imports added: `Mathlib.Analysis.Normed.Group.Uniform` in `DefectCorrection.lean`; `Mathlib.Algebra.BigOperators.GroupWithZero.Action`, `Mathlib.Algebra.Module.NatInt` and `Mathlib.Tactic.NormNum.Eq` in `SymmetryAveraging.lean`.

**Changed:** the four batch-3 files under `lean/AtlasProofs/`; `lean/blueprint/src/chapters/defect-correction.tex` and `structure.tex` (`\leanok` in 16 proofs, and a one-line proof for the non-group example, which had none); [[formal-proofs-record]] (§3's cells, §5.2, §5A.4, §5A.5); [[formal-proofs-plan]] (§6, the lines of T5, T10 and T11); [[formal-proofs-implementation-plan]] (§6.2). **Added:** `lean/decls-batch-3.txt`. **Updated:** `out/lean/builds-batches-1-3.jsonl`, `axioms-batches-1-3.json` (now the check over all 58 names), `profile-batches-1-3.json`.

## [2026-10-01] build | Formal proofs integrated: both branches merged, 105 theorems, no `sorry`, standard axioms only

**The two proofs chats' work is one tree, and all of it is checked.** On the owner's word the branch of batches 1 to 3 (`origin/claude/atlas-proofs-batches-1-3-ztrott`, four commits) was merged into `atlas-0.1` as `c830510`. The only conflict was the end of this log, where both branches had appended: both sides are kept, the three entries of batches 4 to 6 first, then the three of batches 1 to 3 (which is why those sit just above this entry, out of clock order). The three formal-proofs pages merged by themselves.

**The three checks, with `ATLAS_LEAN_TAG` unset, on the laptop, on battery** ([[formal-proofs-record]] §7.11):

- `python scripts/lean_build.py --clean`: success, **137.9 s**, 27 files, 0 errors, **0 `sorry`**;
- `python scripts/lean_axioms.py`: **142 declarations, 0 failing**; nothing beyond `propext`, `Classical.choice`, `Quot.sound`;
- `python scripts/lean_build.py --profile`: 26 files, 447.2 s, **0 steps at or over ten seconds**.

**No theorem failed at the merge.** The eleven theorems of batches 4 to 6 that quoted T1, T4, T4c or T5 compiled against the proved statements with no change to any file. The project now holds **105 theorems the owner approved** (48 and 57) in 26 Lean files and four levels of imports.

**Honest notes.** The merged build took 137.9 s against 79.3 s twenty minutes earlier on the same 27 files; every first-level file was about twice as slow, changed or not, so it reads as the laptop's state on battery, **[AI Inference]**, not measured. Two linter remarks remain in files of batches 1 to 3 (an unused section variable, an unreferenced variable name), neither an error; they were left alone. **Three status lines outside either chat's block were rewritten at the integration**, because after the merge they still said "nothing is proved yet": the header lines of [[formal-proofs-record]], [[formal-proofs-plan]] and [[formal-proofs-implementation-plan]]. **Still not written:** the row for [[formal-proofs-implementation-plan]] in [[index]] and the W353 status in [[gap-worklist]]; both pages hold another chat's uncommitted edits. Nothing was pushed. **Changed:** the three formal-proofs pages, `out/lean/builds.jsonl`, `out/lean/axioms.json`, `out/lean/profile.json`.

## [2026-10-01] build | Proofs: the integrated tree checked a second time, and the profile's count corrected

[[formal-proofs-record]] §7.12; W353. The owner asked the set-up chat to integrate; the merge and the three checks were already committed by the other chat, so they were **run again independently** on the same tree, on battery.

- **Build:** every project file re-checked in 100.4 s, 27 files, 0 errors, 0 `sorry`, no Mathlib module compiled. The saved log, counted a second way, agrees.
- **Axioms:** 142 declarations, 0 failing; over all the answers the only axiom names are `propext`, `Quot.sound` and `Classical.choice`.
- **Profile:** 0 elaboration steps at or over ten seconds in 26 files (836.4 s on this run against 447.2 s on the first; nothing changed between them but the laptop's state).

**An instrument corrected.** `scripts/lean_build.py --profile` counted the profiler's `import took …` lines as slow steps. On this run 18 of 26 files loaded Mathlib in 10.2 to 15 s, and the script reported 18 violations of a rule that is about tactic calls. Imports are now recorded beside the steps and never among them, and this run's record was reclassified by the same function.

**Left as it is, for the owner:** two statements carry a hypothesis their proofs never use (`shrink_re`'s $0\le\alpha$; `average_of_equivariant`'s commuting of the action with scalars). Both are true as written. **Changed:** `scripts/lean_build.py`, [[formal-proofs-record]], `out/lean/builds.jsonl`, `out/lean/axioms.json`, `out/lean/profile.json`.

## [2026-10-01] build | Demo step 8: the learned case, evaluated once; T3's Lean names in R10

The owner's item 1.5, built to [[demo-learned-case-plan]]: one fixed wind-farm case in which a trained network steps every window, against a gate written before any data existed. **It failed one bar of six.**

**In order:**
- **Step 0:** a random network was priced on the laptop first. Width 12, 350,186 parameters, met the speed budget at $1.93\times$; widths 20 and 24 did not.
- **The registration:** the gate G1–G6 and the held-out split, committed in `a7a300a` before the first sample. The layout of `farm-12` is held out; the inflow cannot be, since the band fixes it.
- **The truth:** the full domain at twice the resolution, 40 macro-steps on the laptop.
- **The data and the training:** 110 layouts, 132,000 window samples, and 60 epochs on one rented RTX 4090 box. The box cost \$0.35 and was destroyed; no later charge appeared.
- **The evaluation, once,** on AC with no other Python process.

**The result** (`out/learned-case/gate-20261001-154054.json`):
- **The learned arm L is as accurate as the classical decomposition** (farm power $7.50\%$ from the truth against $7.53\%$), $1.52\times$ faster than it and $7.45\times$ faster than the full domain.
- **It is not as accurate as the full domain** ($0.51\%$). G2's registered bar was the worse of the two classical arms.
- **G5 fails:** the classical decomposition on a grid twice as coarse is closer to the truth in both measures at both horizons, in a third of L's time.
- A smoke test had shown the classical arms' errors before the evaluation. The gate was unchanged, and the record says so.

**In the page:** the first line under the wind farm's *Fast example* arrow (a menu item: the header has no width left), read-only, four arms against the truth, with the verdict and the caveat on its card. Walking it found a fault the tests had missed: the Run view had no colour for a learned arm and did not draw. The headline Speed card had named the coarse competitor's ratio; it now names the learned arm's.

**In the installer:** the weights and the truth are the only binary files its builder allows, by exact path. Verified on fresh clones on Windows (333 tests passed) and Linux (329 passed), where the self-test steps the learned case and prints its verdict; not on macOS.

**W348, the same day:** the proofs chat checked T3 and found its Schwarz part false as stated. R10's decertification now cites T3's Lean names, and for restricted additive Schwarz requires a convergent sweep (`eccf0a2`). The change is text only: the two compilers' syntax trees match with string constants blanked.

**Tests:** the workbench suite, 347 passed. The full suite, 2315 passed and none failed, every log written after the start (on battery, 942 s).

**Added:** `atlas/workbench/learned_gate.py`, `learned_net.py`, `learned_arms.py`, `learned_case.py`, `scripts/learned_*.py`, `scripts/learned_box_run.sh`, `tests/test_workbench_learned.py`, `out/learned-case/`. **Changed:** `atlas/compiler.py` (R10's message), `app.py`, `runview.py`, `runner.py`, `spec.py`, `families/windfarm.py`, the installer's templates, builder and verifier, four test files, the workbench README, [[demo-learned-case-plan]], [[demo-finish-plan]], [[showcase-gallery]], [[gap-worklist]] (W348, W349, W351), [[00-proposal-workstreams]], [[index]].

## [2026-10-02] build | Website step 1: the three reference sites read, and a proposal to the owner

**The website chat (W354), step 1 of nine.** The planning session could not reach the three reference sites, so its design notes came from search summaries. This step read them.

**How:** each site's home page, every stylesheet it links, and Antigravity's own script bundles, fetched with TLS verified from a cloud container. **Not rendered:** the container's headless browser refuses the sandbox proxy's certificate, and loosening the check was not allowed, so no screenshot was taken. Every value recorded is the site's own CSS or code.

**What they are built from:**
- **Antigravity:** Astro, GSAP (ScrollTrigger, ScrollSmoother, SplitText) and three.js; Google Sans Flex; a 72 px headline at 1440 px with $-2\%$ tracking; a 4 px spacing scale up to 180 px; a black pill; no `prefers-reduced-motion` rule in its CSS.
- **Googlebook:** CSS scroll timelines with no animation library; headlines that are short sentences with a full stop; **a band of four figures, each with a footnote marker**; 21 reduced-motion rules.
- **SimScale:** WordPress; Geist; pills; sections padded 64–112 px; logos and customer quotes as its proof.

**One open question settled:** Google Sans Flex is under the SIL Open Font License (Google's font repository, `ofl/googlesansflex`). The notes recommend Inter or Geist anyway, so the site does not read as a Google page.

**Next:** the proposal to the owner (the section list, the technology, where the source lives, changes to the outline) waits for a yes before step 2.

**Added:** [[website-design-notes]]. **Changed:** [[proposal-website-plan]] (a status line: §2 superseded by the notes), [[index]].

## [2026-10-02] build | Website step 2: the evidence register brought up to the records, and the literature read

**The owner's answers to step 1 (2026-10-02):** yes to every proposed change of the outline; Inter; the workbench's packages and leanblueprint may be installed; cleaned copies of the cited records may be published with the site; and from now on the chat may commit, push and merge.

**Added to [[website-evidence-and-citations]] §1.1, each from its record:**
- **A11, the eight Fast examples**: four meet both bars ($4.62$, $5.58$, $5.85$, $4.67\times$), four do not ($0.008$, $0.275$, $1.17$, $0.004\times$), each with its limit.
- **A12, the learned case**, all four arms: L matches the classical decomposition (farm power $7.50\%$ from the truth against $7.53\%$) and is $1.52\times$ faster, and **G5 fails**: the coarse classical arm is at $3.63\%$ in a third of L's time.
- **A13, the proofs**: 142 declarations, 0 `sorry`, three standard axioms, 105 approved theorems, a clean re-check in 100.4 s on battery, and the T3 finding.
- **A14, the architecture's measurements**, labelled as not training results: an untrained trunk's forward cost, and the Gram's squared error with synthetic perturbations.
- **A15, the installer**: Windows and Linux verified, macOS not.
- A1 is superseded on the site by the Fast farm; **A7 and A10 stay off** (no run record).

**The literature, read (§3.1):** nine rows verified with their own words, location and DOI: L2, L4, N1–N7. **Off the site:** L1 (786,000 cores is not in the paper as far as could be read; the thesis's 458,752 replaces it), L3 (GenEO's text unreachable), L5, L6 and N8–N12 (not attempted), and N2's "3% error" (not in the abstract). N7's author list corrected.

**"A first", searched again (§3.2):** no machine-checked theory of domain decomposition was found, but machine-checked proofs of related iterations exist (Jacobi in Rocq; Krasnoselskii–Mann in Lean 4). The site says *"we found no earlier"*, not *"a first"*.

**Changed:** [[website-evidence-and-citations]].

## [2026-10-02] build | Website steps 3 and 4: the claims pipeline, and the site's skeleton

**Step 3, the claims pipeline** (commit of its own): `scripts/site_claims.py` reads the run records and writes `site/data/claims.json`, and publishes each cited record to `site/records/` with the computer's name, process ids, local paths, log excerpts and the installer's scans removed. `site/data/literature.json` holds the nine verified published results. `tests/test_site_claims.py` fails if a claim differs from its record, if anything private or forbidden reaches `site/`, or if **any number on the site's pages** is not attributed to one of the two files. Three negative controls were run and caught: a number typed into the copy, a stamped number edited by hand, and a claim edited in the JSON.

**Step 4, the skeleton** (`site/`, plain HTML, CSS and a little JavaScript; no npm, no framework):
- `index.html`, every section of [[website-outline]] with the changes the owner accepted (its status line); `install.html` with three tabs, macOS marked not verified; `evidence.html`, every number with its record or quotation.
- `scripts/site_build.py` writes every number, the two charts (the Fast examples on a log scale against a $1\times$ line; the learned case's four arms, time against error), the literature cards, the proofs' dependency graph and the evidence tables into the pages, from the two files.
- **"How it works" is drawn from the real S-channel**: `scripts/site_export_schannel.py` builds the workbench's own example and solves it decomposed and whole; the export agrees with the record ($1.8\times10^{-9}$ of the span).
- **The blueprint is built** into `site/proofs/blueprint/` by `scripts/site_blueprint.py` (leanblueprint 0.0.20 on plasTeX 3.1, installed with the owner's yes), from a copy of its sources with the public addresses and two workarounds for a machine without TeX; `lean/` is not edited. The architecture page is copied unchanged into `site/architecture/`, and the Lean sources into `site/proofs/lean/` as readable pages.
- **A real screenshot of the workbench**, run in the cloud container with the workbench's pinned packages (installed with the owner's yes): the heat-conduction Fast example. **The wind farm's Fast example could not run there**: its solver is vendored only in the install branch, and this checkout looks for it on the laptop. The screenshot's caption says the speed in it is the cloud machine's.
- The chart colours were validated with the dataviz skill's checker (cyan $\#008cbf$ and orange $\#e0632f$ on white).

**Changed:** [[website-outline]] (status: the accepted changes).

## [2026-10-02] build | Website steps 5 to 9: the image brief, the links, the browser walk, the launch checklist

**Step 5, the images:** [[website-image-prompts]] holds the final prompts as one numbered list: four required (H1, 1a, 2a, 3a, each 21:9 and 4:5) and thirteen optional, each with its crop, one shared negative prompt, and the calm areas the headline and the drawn agent graphs need. The owner generates them; the site shows marked placeholders until then.

**Step 6, the honest parts**, each where [[website-outline]] puts them (with the accepted changes): the passage that everything in *Working today* is classical; *"On small problems the undivided solve is faster"*, above the four kinds that missed the bar, each card naming its limit; the learned case with all four arms, the coarse competitor beside it, and *"The coarse solver won, so the gate failed: 5 of six bars passed"*; *"Vision. Not built yet."*; *"Published by other researchers"* on every literature card, with a banner saying the numbers are not Atlas's and not comparable.

**Step 7, the links:** the architecture page is copied unchanged into `site/architecture/`; its MathJax is served by `cdn.jsdelivr.net` (HTTP 200 with TLS verified, from Python; the container's browser cannot fetch external sites, so the render itself is checked at launch). The blueprint is built into `site/proofs/blueprint/`, the Lean sources are in `site/proofs/lean/`, and the install commands are written from `site/data/site.json`. **One stale sentence found and not edited** (not this chat's file): the architecture page says its finite-dimensional theorems have not been machine-checked; the fix is proposed in [[website-launch-checklist]] §0.

**Step 8, the walk:** every page at 1440 × 900, 1090 × 620 and 390 px, and with reduced motion. No console error, no failed request, no missing anchor, no horizontal overflow (after two fixes: the charts scroll sideways on a phone instead of shrinking, and the Lean index wraps its file names), every relative link and asset returns 200. Accessibility pass (custom, as Lighthouse is not installed): one `h1` per page, no heading jumps, every image with alt text, every control named, text contrast at least 4.5:1 (the cross marks of the S1–S12 list, non-text, at 3.5:1). Page weight: Inter 73 kB, the screenshot 113 kB, the S-channel's data 116 kB.

**Step 9:** [[website-launch-checklist]], the owner's steps in order; none is done. W354 in [[gap-worklist]] is *built, awaiting the owner's images and launch*.

**Added:** [[website-image-prompts]], [[website-launch-checklist]]. **Changed:** [[proposal-website-plan]], [[vision-scenarios-and-image-prompts]] (status lines), [[00-proposal-workstreams]] (workstream 2), [[gap-worklist]] (W354), [[index]].

## [2026-10-02] build | Website launch: published from nonidino/Atlas; the architecture page's proof status brought up to date

**The owner made `nonidino/Atlas` public and enabled Pages**, and allowed the website chat to fix the architecture page.
- **The architecture page** (`proposal/architecture/index.html`): its four passages that said the finite-dimensional theorems "have not been machine-checked" now name the Lean theorems that check them (`Atlas.Piece.gram_posSemidef`, `Atlas.Superelement.hostSystem_existsUnique`, `Atlas.Superelement.energy_optimal`, `Atlas.interface_perturbation`, `Atlas.Piece.labelFree_eq_supervised_add_const`, `Atlas.cayley_bound`, `Atlas.contraction_contract`, `Atlas.two_level_contract`, and the headlines), with the two hypotheses the formal tier-0 statement makes explicit, T9a's $\ell^2$ defect, and T9b's whole-round hypothesis with its diverging example ([[formal-proofs-record]] §7.6, §7.11). Novelty row N6 reads "done here". Copied into the site.
- **Publishing:** `.github/workflows/pages.yml` uploads `site/` to Pages on each push to `atlas-0.1` that changes it; the owner sets Pages' source to *GitHub Actions*. `site/data/site.json` now points at `nonidino/Atlas` and `https://nonidino.github.io/Atlas/`; the blueprint was rebuilt with those addresses.
- **Images** are uploaded to `incoming/images/`.
- **Still open:** pushing the demo's branch `atlas-workbench` from the laptop; the owner's images. [[website-launch-checklist]] §1 records that the whole repository, not only the site, is now public.

**Changed:** [[website-launch-checklist]].

## [2026-10-02] build | Website: the owner's illustrations placed

**Fourteen images** arrived in `incoming/images/` (commit `7ada48a`): the four required in both crops (H1, 1a, 2a, 3a) and six optional in 16:9 (1c, 1d, 1e, 3c, 3d, M1).
- **Checked** against [[website-image-prompts]] §1: no readable text, logo, livery or insignia (the car carries a few tiny illegible marks that read as surface detail); the palette and the calm areas as briefed.
- **Compressed** to WebP with `scripts/site_images.js` (1.6 MB for all fourteen) into `site/images/`, each with a sidecar: prompt, negative prompt, alt text, date, source commit. **The generator's name and version were not given.** The originals were removed from `incoming/` and remain in git history.
- **Placed:** the hero (wide, phone crop under 700 px, loaded first); the three scenarios, each with its phone crop; 1c, 1d, 1e under the car, 3c, 3d under the base, M1 above the chart operator's panels; every one tagged *Illustration*.
- **The agent graphs moved onto the parts**: `scripts/site_build.py` now draws each scenario's graph from a table of positions in the image's own pixels, one set per crop, so the graph stays on the car's brakes or the base's microgrid at any width.
- **Walked again** at 1440, 1090 and 390 px: no console error, every link 200, no sideways scroll after one fix (the learned case's chart had widened its grid column on a phone). Claims test: 8 passed.

**Changed:** [[website-image-prompts]], [[website-launch-checklist]].

## [2026-10-03] build | Website: two halves and their join; the install runs from atlas-0.1

**The owner's restructure** ([[website-outline]], revision of 2026-10-03). The page is now framed as two halves and their join: neural operators are faster than classical solvers but struggle on complex geometry, decomposition is faster and simpler than a full-domain solve on complex geometry, and so learned experts on decomposed problems are much faster.
- **Sections:** *The power of domain decomposition* (the three fastest Fast examples; the wind farm's speed against its rotor count, a new chart from every rung of W346, [[website-evidence-and-citations]] A16; the demo; *What others have shown*). *The speed of learned experts* (published results, then the project's learned case in brief, with its failed gate). *A universally compatible neural operator: the chart operator* (why today's operators aren't built for the job, in brief; the neural-decomposition work; the three panels; *compatible by design*). *Mathematically proven.* *The full vision.* The nav follows.
- **Removed:** the four slower Fast examples' cards, the learned case's chart, the S1–S12 checklist, image M1, and the path section.
- **Literature verified** (§3.3): L7 and L8, overset grids for complex bodies (NASA records); N13, the Fourier neural operator's speed; N14 and N15, the geometry limitation, in the words of the N6 and N5 abstracts.
- **The owner's three emphases**, written into the outline and the page: modular, extendable experts (learned, classical or composite, one standard interface, any level of detail, meant for real engineering systems); the chart operator as a different formulation, with Lean-verified theory; formalization as part of the discovery.

**The install, fixed.** The site cloned `atlas-workbench`, a branch that was never pushed. The workbench now runs from the root of `atlas-0.1`: the launchers come from `atlas/workbench/bundle/`, the wind farm's solver sits in `vendor/` (copied from the public PoC branches, `vendor/README.md`), and there is a root `README.md`. A fresh clone's `./run.sh --check` on a cloud machine installed everything and ran all eight kinds, the learned case and the page. It exited 3 only because Gmsh lacked `libGLU`. `launched` is now true. The image generator is recorded as ChatGPT.

**Noted for the owner:** `poc1-windfarm-demo`, a branch of this now-public repository, carries Poseidon-T's weights (`vendor/hf-cache/.../model.safetensors`). They are not on the site and not on `atlas-0.1`.

**Checked:** claims test 8 passed; the browser walk at 1440, 1090 and 390 px, with reduced motion, showed no console error, every link 200 and no sideways scroll.

**Changed:** [[website-outline]], [[website-evidence-and-citations]], [[website-image-prompts]], [[website-launch-checklist]], [[gap-worklist]].

## [2026-10-03] build | Website: the three goals, and the owner's second pass

**Added:** a *Three goals* band after the hero. It shows the three goals on a dark field with animated seams that stop under reduced motion, and the Lean count drawn from its record.
**Removed or reworded,** as listed in [[website-outline]] (second revision of 2026-10-03):
- the illustration tags, the port-type buttons, the vision banner, the wildfire's *Today* paragraph;
- the proof cards and two of the proof counts;
- the fine print and the literature banners;
- the learned case's gate paragraph (its verdict stays on the evidence page);
- the first person, from the site's own pages.

**Fixed:** the install page's screenshot was squeezed on narrow screens, because `height` kept its attribute value while the width shrank. All images now keep their proportions (`img { height: auto }`).

**Checked:** claims test 8 passed. The browser walk at 1440, 1090 and 390 px, with reduced motion, found no console error, every link 200, no sideways scroll, and the accessibility pass clean.

**Changed:** [[website-outline]].

## [2026-10-03] build | Website: macOS verified

The owner reports the install verified on a Mac. The install page's macOS tab now says *verified*, and its *not verified* banner is gone. No record file came with the report.

**Changed:** [[website-launch-checklist]].

## [2026-10-10] note | The chart operator in general form: nonlinear physics, multi-field seams, and two single-field experts on one piece

The architecture chat, on the owner's questions: why so much of the design relies on the PDE being linear; a generalized form; fields that differ across a seam (the rocket's air against its skin); and two separately trained experts, temperature and stress, working in one frame. **A proposal for discussion, not a decision**; version 1 and the public paper are unchanged. No network was trained.

**The answer:** version 1's guarantee rests on exact traces and on the host evaluating the true physics, not on linearity. In the general form the network is a trial-space generator and the host projects the true physics onto that space. The guarantee becomes a class set by the physics' structure, from G0 (version 1's tier 0) to G3 (explicit hyperbolic steps). Across a seam only the typed ports of [[port-algebra-atlas-0.1]] must match. On one piece, two experts couple through a host-owned co-located bond, building on [[case-study-thermal-strain-atlas-0.1]]'s verdict that thermal strain is a bond and not a port.

**Checked** (`scripts/arch_general_form_checks.py`; classical solves plus perturbations standing in for networks):
- With advection, adjoint constraint modes keep the port error quadratic (slope 2.00 against Galerkin's 0.99 at Péclet 200). The exact piece's port matrix is itself indefinite there, so positivity was never available.
- For a thermoelastic steel bar the coupling strength is $\gamma^2=0.0108=\delta$. Over the product of two single-field trial spaces the coupled system kept every eigenvalue modulus in $[1,1.005]$ for every expert error tried, up to 100%, and the Galerkin error stayed within 1.003 of the best approximation.
- On a nonlinear convex piece, a learned subspace correction never did worse than one classical step from the same point. Network calls to $10^{-10}$ were 6, 7, 11, 25, 48 and 154 for proposal errors of 0 to 300%, against 923 classical steps. Proposals with no information were slower than the classical trajectory while still beating a classical step at every point, which corrected the claim to "never worse than the classical worst case".

**Added:** [[chart-operator-general-form]]; `scripts/arch_general_form_checks.py` and `out/arch/general_form_checks.txt`/`.json`.
**Changed:** [[chart-operator-architecture]] (a status line pointing to the discussion), [[index]].

## [2026-10-10] note | Chart-operator family catalog: the PDEs, the interactions a decomposition yields, unit-square speedups, and one architecture per family

The architecture chat, on the owner's request. The request asked for the PDEs the experts will cover; the interactions a full decomposition yields and how each is accounted for (same field across a seam, different fields across a seam, different fields on one domain); speedups specific to the unit square; and the chart-operator architectures, each with its chart information, constraint modes, ports and conservation, and coupling. **A design reference for discussion, not a decision.** It is published as the owner's artifact *Chart Operator Families* (https://claude.ai/artifact/6YFtDeR3cX96ifqn8pbjuk), and the vault digest is [[chart-operator-family-catalog]]. No network was trained.

**The organising result:** a chart operator is five slots: chart information, trial space, ports and conservation, host class G0–G3, backbone and time. Families are separated by trace space and class rather than by PDE. Twelve families cover about forty PDEs, and lumped components are port-Hamiltonian nodes, not chart operators. There are eight interaction kinds, each with one host-owned instrument, all under passivity and power-conserving interconnection.

**Checked** (`scripts/arch_family_checks.py`; classical solves standing in for networks; bilinear finite elements):
- **S2.** CG preconditioned by the plain square's fast sine solve needed 30–31 iterations to $10^{-6}$ and 52–55 to $10^{-10}$ at every grid from $32^2$ to $256^2$ ($\lvert\mu\rvert\le0.5$, contrast 10). The derived bounds are 69 and 113. Time was 0.53 to 0.09 of a sparse direct solve.
- **S7.** High port modes live in strips near their side. At amplitude tolerance $10^{-3}$, cropping the mode head to the strips covers 0.438 of the square steady (2.28×) and 0.178 at $\mathrm{Fo}=10^{-3}$ (5.63×), losing at most $5.5\,\mathrm{tol}^2$ of energy. At distortion $K=3$ the saving collapses to 1.27×.
- **S9.** Rounding is an interior error, so it enters the Gram quadratically. A bfloat16 correction over the exact principal part gave a Gram error of $3.2\times10^{-6}$; rounding the whole field gave $7.0\times10^{-5}$. A float32 Gram formed from exact modes gave $1.5\times10^{-7}$.
- **D.** In 2-D the flux (dual) problem is a diffusion problem on $\kappa^{-1}$ with the same Beltrami coefficient, so the same network bounds each port matrix from below. The Loewner chain held to $6.4\times10^{-16}$. For smooth port data the computable gap was 2.2–2.7× the true error at perturbations 0.01–0.3. For worst-case data the dual needs about twice the primal's ports: 63 ports give 105%, 127 give 13.3%, all nodes 13.0%.

**Recorded for others:**
- The rocket's $a$–$b$ edge is a co-located bond rather than a seam.
- Its $c$–$d$ edge declares `ADVEC`$(h_0)$ at what the port algebra's §3.2 would call a wall; this is a question for [[case-study-rocket-ascent-2d-atlas-0.1]].
- Four port-vocabulary gaps are candidates for [[port-algebra-atlas-0.1]]'s amendment procedure.

Neither page was edited.

**Added:** [[chart-operator-family-catalog]]; `scripts/arch_family_checks.py` and `out/arch/family_checks.txt`/`.json`.
**Changed:** [[index]].

## [2026-10-12] note | The public architecture document, version 2: the general formulation, explained from first principles

The architecture chat, on the owner's request. The owner asked for three things:
- the per-family training-time table on [[chart-operator-training-and-cost]] (its new §5.1);
- one comprehensive, readable and skimmable HTML page for the whole generalised chart operator, combining the family-catalog artifact with the background and structure of the published proposal. The owner judged that the published proposal gave too little explanation and intuition;
- that page to replace the old proposal on the website.

**The page.** `proposal/architecture/index.html` is version 2 of *Chart Operators* (Naunidh Singh, 12 October 2026). It contains:
- a five-minute tour;
- gentle background on domain decomposition, charts and neural networks;
- the record S1–S12;
- version 1's design and its machine-checked guarantee;
- the general form with the classes G0–G3;
- the five slots;
- the eight interactions, with a new figure for the three the owner named;
- the PDE catalog and the family cards;
- the unit-square speedups and the dual bracket;
- training with the per-family table, cost, gates, related work, limitations, and the roadmap with decisions F1–F10;
- the Expert Card, derivations, reproducibility and a glossary.

Every section opens with an *In short* summary, every formula is preceded by a plain-words picture, and proofs are folded away. The page has 16 sections, 4 appendices, 16 numbered statements and 113 references, all resolved by `scripts/arch_paper_build.py`, which now takes a page path. The Lean names cited include `Atlas.interconnection_passive` and the other passivity theorems, so the passivity principle is marked machine-checked. Headless check: 938 formulas typeset, no MathJax errors, no unresolved equation references, and no horizontal overflow at narrow width.

**On the website.** `site/architecture/index.html` is replaced by version 2, and version 1 is archived beside it as `paper-v1.html` with a banner saying it is superseded. On the home page one sentence is qualified: "For linear physics, whatever the network returns" now carries the note that beyond linear physics the guarantee is set by the structure of the physics. Merged into `atlas-0.1`, whose Pages workflow publishes `site/`.

**Added:** `proposal/architecture/index.html` (version 2), `proposal/architecture/paper-v1.html` (version 1, moved), `site/architecture/paper-v1.html`.
**Changed:** [[chart-operator-training-and-cost]] (§5.1), [[chart-operator-architecture]] (a status line), `site/architecture/index.html`, `site/index.html` (one sentence), `scripts/arch_paper_build.py`, [[index]].

## [2026-10-17] lint | Whole vault and repository: the link graph is clean, the status lines were not

**Why.** The owner asked for the whole wiki and the GitHub repository to be organised and cleaned up before Stage 0 starts. This entry is group A, the safe fixes. The structural decisions (groups B and C: the size of the log, the index and the worklist; root files; `out/`; branches) go to the owner one at a time.

**First, the checkout was behind.** Local `atlas-0.1` was 19 commits behind `origin/atlas-0.1`: the website, the README and chart operator version 2, all pushed from cloud sessions between 2026-10-02 and 2026-10-12. It was fast-forwarded (clean tree, no local commits) before anything was read, so this lint covers the published state.

**What the scans found.** `scripts/link_scan.py`: 291 files, 10,580 links, 6 dead, all 6 pinned. `scripts/vault_scan.py`: 0 problems. A one-off structural scan (orphans, index coverage, link convention) found no duplicate page names and no folder-path links. It found one orphan, `misc/prereqs-and-resources.md`, which the index linked by a markdown path that Obsidian's graph does not count. It also found two pages missing from the index, and two summaries missing from [[code-and-papers]].

**What the reading found: status lines that their own pages had outgrown.**
- [[chart-operator-architecture]] still said "none is machine-checked" (T24–T26 were checked on 2026-10-01) and "private until the website launches" (launched 2026-10-02).
- [[index]]'s row for [[formal-proofs-record]] still said "none is proved yet".
- [[case-study-wind-farm-wake-2d-atlas-0.1]] and [[spec-wind-farm-wake-atlas-0.1]] said "not built"; building began on 2026-08-20.
- [[case-study-rocket-ascent-2d-atlas-0.1]] said "nothing here has been implemented"; Phases 0 and 1 were built, and M2 stopped the plan.
- [[00-start-here]], [[index]], [[code-and-papers]] and the root `CLAUDE.md` all named the private `nonidino/physics-foundation-model` as the code home. The code has lived in this public repository, `nonidino/Atlas`, since 2026-08-27. `CLAUDE.md` also said "preliminary research; don't assume an implementation exists", and its folder tree had none of `Atlas 0.1/`, `Noether 1.0/`, `Noether 1.1/` or `misc/`.

Each stale line is corrected by a dated note beside it, not rewritten, so the page still shows what it said when it was written.

**Changed:** [[chart-operator-architecture]], [[case-study-wind-farm-wake-2d-atlas-0.1]], [[spec-wind-farm-wake-atlas-0.1]], [[case-study-rocket-ascent-2d-atlas-0.1]] (correction notes); [[index]] (the code home, the proofs row, a row for [[formal-proofs-implementation-plan]], [[prereqs-and-resources]] as a wiki link, the date); [[00-start-here]] (the code home, a section 7 on Noether and Atlas, the folder tree, a pointer to the reading list); [[prereqs-and-resources]] (a See Also); [[code-and-papers]] (a section for `nonidino/Atlas`, the build repository's role restated, rows for [[walrus-overview]] and [[pde-transformer-landing]]); `CLAUDE.md` (code home, status, repository folders, the folder tree, the two scans to run before a commit).

**Not changed, and why.** The 58 log headers that do not follow `## [date] kind | title` (sub-headings and compound kinds such as `ingest+note`), and two out-of-order dates, are left alone because the log is append-only. The two sources in `new/` (GLM 5.2, Q-Former) were ingested on 2026-06-30 into concept pages without summaries. They stay where they are, since `CLAUDE.md` says not to move `new/`, and are put to the owner.

## [2026-10-17] reorg | The log archived by month: 1.29 MB down to 74 KB, every entry kept verbatim

**Why.** At 1,294,522 bytes, 7,467 lines and 264 entries, the log could not be read in one pass, and every append rewrote a 1.3 MB file on OneDrive. The owner approved decision B1 of today's lint: archive by month.

**What moved.** Every entry dated before 2026-10 moved, unchanged and in file order, into one file per month under `wiki/log/`:

| file | entries | bytes of entries |
|---|---|---|
| [[log-2026-04]] | 8 | 12,427 |
| [[log-2026-05]] | 5 | 19,111 |
| [[log-2026-06]] | 15 | 37,355 |
| [[log-2026-07]] | 23 | 73,452 |
| [[log-2026-08]] | 73 | 405,955 |
| [[log-2026-09]] | 110 | 674,545 |

The 30 entries from 2026-10 onward stay in [[log]], whose head now links each archive. Each archive opens with a short header saying where it came from, and that new entries go to [[log]].

**How it was checked.** An entry starts at a `## [YYYY-MM-DD]` line outside a code fence; sub-headings inside an entry stay with it. The split script asserted that the months run in file order with no interleaving: the log's two out-of-order dates, 2026-04-12 before 2026-04-11 and 2026-08-30 before 2026-08-29, each fall within one month. It also asserted that the archived lines plus the kept lines rebuild the original file byte for byte. After writing, the files on disk were compared again against a copy of the original taken first: 7,461 entry lines, identical.

**What else changed.** `scripts/link_scan.py`: its two pinned dead links in old log entries (`spec-wind-farm-wake`, `positive-controls-need-a-horizon`) now point at [[log-2026-09]], where those entries now live. `CLAUDE.md`, [[00-start-here]] and [[index]]: the folder trees and the build-history pointer name `log/`. No page linked to a heading inside the log (`[[log#...]]`), and no script or test reads it, so nothing else needed repointing.

## [2026-10-17] reorg | The index slimmed from 384 KB to 78 KB, with every long description kept word for word in a digest

**Why.** `CLAUDE.md` says to read the index first, and at 383,589 bytes it cost about 95k tokens. 151 of its 291 rows had grown past 300 characters into abstracts, holding most of the file; the longest was 11,651 characters. The owner approved decision B2 of today's lint: shorten the rows, keep the long text verbatim elsewhere, and split the Atlas 0.1 table by subfolder.

**What changed.**
- **151 rows shortened** to at most about 300 characters, each saying what its page is and its main result, and marked ↗. Each short version was written from its row's own lead. Eight whose lead was cut off mid-claim were checked against the full text, and corrected where the first draft had guessed.
- **[[index-digest]] (new)** holds the 151 long descriptions word for word, under the index's own section headings, each with its title and page links. The 7 table-escaped pipes in them are kept as written.
- **The Atlas 0.1 table** is now one table per subfolder: the top level (1 row), `common/` (102), `case-study-wind-farm-wake/` with its `implementation/` (10), `case-study-rocket-ascent/` (3), and one each for the ground-effect, brake-thermal, wing-FSI and RBC folders. Rows keep their order within each table. The rocket's `implementation/` folder, the outcome and the proposal already had their own sections.
- **Two pages were indexed only from inside long descriptions**: [[wind-farm-implementation-log]] and [[vast-ai-training-runbook]]. Shortening dropped those mentions, so each now has its own row. The six log archives are linked from the index's header.

**How it was checked.** The build script asserted these before writing: the shortened set is exactly the rows over 300 characters; every short version is free of pipes, backslashes and tabs; the row count is unchanged; every row's page is unchanged; every long description appears verbatim in the digest; and every page the old index linked is linked by the new index or the digest. After writing: `link_scan.py` shows 0 new dead links and 0 stale pins, `vault_scan.py` shows 0 problems, and every page in the vault is linked from the index. The longest row is now 453 characters, title and links included.

**Also changed:** `CLAUDE.md` (the folder tree, and the ingest rule: a row of about 300 characters, longer notes to [[index-digest]]), [[00-start-here]] (the folder tree).

## [2026-10-17] reorg | The gap worklist split: Tiers 0–75 archived word for word, and a status board of all 348 items

**Why.** [[gap-worklist]] was 722,350 bytes in 2,664 lines, arranged by tier (0 to 89) rather than by status. 140 of its 348 W-items have rows in more than one tier, so finding what is still open meant reading all of it. The owner approved decision B3 of today's lint.

**What moved.** The sections from `# Tier 0` up to `## Tier 38` went to [[gap-worklist-tiers-0-37]] (346 KB); the sections from `## Tier 38` up to `# Tier 76` went to [[gap-worklist-tiers-38-75]] (274 KB). Both were moved unchanged and in order. [[gap-worklist]] keeps its header (status line, how-to-use note and legend), a note naming the archives, and Tiers 76 onward. The page is now 164 KB, of which the board is 57 KB.

**The status board.** It has one line per W-item:
- the **title**, taken from the item's first row (its first bold phrase, cut at a word boundary with math, code and links kept balanced);
- the **status** stated by its latest row, read from the table's `Status` column or from the first cell that opens with a status word;
- **where** that latest row now is, by page and tier.

Counts at generation: open 173, done 122, closed 28, in progress 10, met 3, refused 1, and 11 whose latest row states no status a script can read, which the board marks "see row". The board reports what the rows say, and some rows lag their work. W353, for example, the formal proofs, still reads "open" in its row although batches 1 to 6 were integrated on 2026-10-01 ([[formal-proofs-record]]). Its row was not edited here.

**How it was checked.** Before writing, the split script asserted that every tier heading falls in its own range, and that header plus the three parts reproduce the original lines. After writing, the files were compared against a copy taken first: the two archives' bodies equal the original's Tier 0–75 lines exactly, the kept tiers equal its Tier 76 onward exactly, and every header line is still present. `link_scan.py` reports 0 new dead links, `vault_scan.py` 0 problems, and no board line has an unbalanced `$`, backtick or link.

**Also changed:** [[index]] (a row for each archive, and the worklist's row), `CLAUDE.md` (the folder tree). No code reads the worklist: the five scripts and one test that name it do so in comments.

## [2026-10-17] correction | The two inbox sources do have summaries

Today's lint entry says the two sources in `new/` (GLM 5.2, Q-Former) "were ingested on 2026-06-30 into concept pages without summaries". **That is wrong.** Both have summaries: [[glm-index-share-attention]] and [[q-former-architecture]] in `summaries/transformers-theory/`. Both are also in [[code-and-papers]] and [[index]], and the concept pages [[index-share-sparse-attention]] and [[hierarchical-query-attention]] link to them. The lint had searched for the sources' file names and never opened `summaries/`. The ingest is complete, and only the two files still sit in `new/`.

**Found while checking, for decision B4.** Of the 36 summaries, 14 cite a `raw/` file as their source, and none of those 14 files exists: `raw/` holds only two files, the Poseidon paper and a multimodal-transformers survey, and no summary cites either by path. Git has no record of the other files, because the repository began on 2026-08-30, after they had gone. Obsidian's `.trash/` is empty. The other 22 summaries cite no `raw/` path.

## [2026-10-17] reorg | The source folders made local-only: the inbox emptied into raw/, and neither is published

**Why.** The repository is public, and `raw/` and `new/` held other people's text: the Poseidon paper in full, a survey of multimodal transformers, and two articles clipped from MindStudio and emergentmind.com. The owner approved decision B4 of today's lint.

**What changed.**
- The two inbox sources, GLM 5.2 (summary [[glm-index-share-attention]]) and Q-Former (summary [[q-former-architecture]]), were moved from `new/` to `raw/`, since their ingest finished on 2026-06-30. The `new/` folder itself stays where it is, now empty, as `CLAUDE.md` requires.
- `raw/` and `new/` are untracked (`git rm --cached`) and listed in `.gitignore`. All four files remain on disk and in OneDrive; only the repository stops carrying them.
- `CLAUDE.md` now says both folders are local only, that a finished ingest moves its source to `raw/`, and that a summary must cite its source's URL or arXiv id.

**Left alone, as the owner chose.** The 14 summaries whose `Source:` line names a `raw/` file that no longer exists keep that line, as the record of what each was written from. Git history still contains the four files. Removing them from history would mean rewriting it and force-pushing, which was not done.

## [2026-10-17] reorg | The repository root cleared: four plans to docs/plans/, two stale files removed

**Why.** Decision B5 of today's lint, the last of group B. The owner then asked for all of group B to be finished and committed together with group A.

**What changed.**
- `ROCKET-POC-PLAN.md` (PoC 4), `ATLAS-CLASSICAL-PLAN.md`, `COMPUTE-PLAN.md` and `POC3-RACELAB-REQUIREMENTS.md` (PoC 3) moved, with `git mv` so their history follows them, from the repository root to `docs/plans/`. Their names are unchanged. Every citation of them names the file without a path, in wiki pages, in log entries and in docstrings in `atlas/cases/racelab*.py`, `scripts/tier51_racelab_graph.py` and `tests/test_tier51_racelab_graph.py`. So no citation needed editing, and nothing reads them by path.
- `NEXT-SESSION-PROMPT.md` was deleted: a Tier 51 pickup prompt from 2026-09-12 that nothing cites, outrun by Tiers 52–89. Git history keeps it, last at `721facb`.
- `wf_test_after.log` was deleted: one line of pytest progress, untracked and already git-ignored.
- `README.md` ("What is where") and `CLAUDE.md` (the repository folders) list `docs/plans/`.

**Left alone.** `atlas-demo/` is a git-ignored local copy of the PoC 1 bundle, assembled 2026-09-03 with its own `.venv` and the CC-BY-NC Poseidon-T weights, and is not published. `incoming/images/` is the website's documented drop folder for images.

**Group B in total, all on 2026-10-17:**
- B1: the log archived by month.
- B2: the index slimmed to 78 KB, with [[index-digest]].
- B3: the gap worklist split, with its status board.
- B4: the source folders made local-only.
- B5: this entry.

Group C (remote branches, `out/`, the repository's GitHub description) is next.

## [2026-10-18] build | The chart operator's Stage 0: CO-D built, eleven arms trained, judged once

**What.** The owner's Stage 0 brief (W355): build, train and gate the first learned expert, CO-D, for 2-D linear diffusion on curved four-sided pieces, steady and backward Euler, with `THERM` ports and tier-0 energy-Gram coupling. The record is [[chart-operator-stage-0]].

**Built.** `atlas/chartop/` holds:
- the charts (Coons, Winslow, harmonic with a conformal correspondence, closed form), Proposition 3's certificate and the envelope;
- bilinear elements on the reference grid (D7), and the corner-hat and side-sine port basis (D3);
- the plain square's principal part by sine transforms;
- the host, with the full-domain reference, and the three benchmarks as conforming layouts;
- the generator, the U-Net and spectral networks with three output forms, the energy objectives and training;
- the registered gates, the dual bracket, $\kappa(T)$ by frozen coefficients, and the Expert Card.

There are 67 tests in `tests/test_chartop_*.py`, the scripts are `scripts/chartop_*.py`, and the records are in `out/chartop/`.

**Registered before training.** The owner's decisions D1–D9, with the gates and comparison bars in `atlas/chartop/gates.py` and its hash in `out/chartop/registered.txt`. E4's maximum-error bar was re-registered the same day (D9), before any training result had been read: the 16-mode classical superelement itself misses the first bar on the s-channel step.

**Trained.** Two vast.ai boxes trained the eleven registered arms in 8.6 GPU-hours: an RTX 3090 at \$0.195/h and an A100 at \$0.532/h. The whole stage cost \$4.06 of the \$15 ceiling. Both boxes were destroyed after their weights' sha256 had been checked.

**Judged once**, on 2,400 held-out pieces and the three benchmarks, on the laptop's CPU:
- E3, SE1, SE2, SE3 and DUAL **met**;
- E1, E4 and KT **not met**;
- E5: no arm faster at comparable accuracy;
- E6: S1–S12, seven met and four not (S5, S7, S9, S11);
- C1, C2, C4 and C7 ties; C3 and C6 lose; C5 chooses 0.3 M.

**The finding.**
- The modes did not learn in any arm. Validation $e_H$ stayed at the principal part's 0.337, whatever the objective, data, backbone, size or output form; the particular field did learn (test median $e_p$ 0.079).
- On the benchmarks, the trained correction breaks the exact sum of the corner-hat modes, which reproduces a uniform temperature. $e_H$ weights that sum at a fraction of a per cent, but at 300–400 K it dominates the coupled error. Lowering every temperature by 300 K cuts B's steady errors from 24–67% to 4.5–11% (post hoc).
- §8's guess, that the zero source of the steady problems was the cause, was refuted after the evaluation.

**Changed:** [[chart-operator-stage-0]] (new) · [[chart-operator-training-and-cost]] §5.2 · [[chart-operator-architecture]] (status) · [[outcome-c5-requirements-for-dd-native-experts]] (status) · [[gap-worklist]] Tier 90, W355–W360 · [[index]] · [[index-digest]].

## [2026-10-18] correction | The architecture page's erratum (W359): bilinear elements, not cell-centred finite volumes

**What.** The owner said yes to the erratum drafted for W359. The public architecture page, `proposal/architecture/index.html` and its byte-identical copy `site/architecture/index.html`, described the piece solver as cell-centred finite volumes. Stage 0 built bilinear finite elements on the reference grid's vertices (D7), because a two-point flux drops the off-diagonal entry of a sheared chart's tensor ([[chart-operator-stage-0]] §1).

**Changed on the page.**
- A dated note under the status line, the version line and the footer.
- The text in Section 2.1, in Section 4.3 (the unknowns, what the energy is a sum of, the dimension of the mode matrix), in Section 5.2, and in Theorem 6's setting and proof. The theorem's statement and bound are untouched.
- The last item of Section 10. Its prediction, that the dual bracket's floor vanishes on cell-centred finite volumes, was made for a discretisation that was not built, and was never checked. The page's own Table 21 had already measured the bracket on bilinear elements, where the floor does not vanish, so the item now says that and leaves the finite-volume prediction open.

**Found on the way.** The page never states E4's maximum-error bar (its Table 27 says "registered per family"), so D9 needed no change there. [[chart-operator-stage-0]] §13 and the W359 rows had said it did, and both are corrected. The cell-centred wording was also in Section 4.3, the proof of Theorem 6 and Section 5.2, beyond the three places the row named.

**Left alone.** `paper-v1.html`, the archived first draft, which also says finite volumes. The Lean docstrings, which still gloss the unknowns as "cell values" and "boundary faces" (the statements are abstract). The page's "No network has been trained", true on 12 October and not since Stage 0. No Stage 0 result went onto the website.

**Checked.** Every edit's old text occurs once in each copy, and the copies stay byte-identical. `scripts/arch_paper_build.py` leaves the corrected page unchanged, so its cross-reference labels are the build's own. `tests/test_site_claims.py` passes (8). Rendered in headless Edge, the note shows in both colour schemes and MathJax typesets 945 expressions against 938 before, with no errors. The local link check finds no new broken link. Not pushed when written: the live check follows the push.

**Changed:** [[gap-worklist]] W359 · [[chart-operator-stage-0]] §13.

## [2026-10-19] build | Stage 0's three operational defects closed (W360)

**What.** The three defects Stage 0 found in its own tooling ([[chart-operator-stage-0]] §12, item 6), fixed for Stage 1's runs. No gate, no weights and no Stage 0 record was touched; `atlas/chartop/gates.py` is byte-identical.

**The workers.** The generator ran 191 workers on the A100 box, whose quota was 23 CPUs (`cpu.max` `2304000 100000`), at 4.2 s per example against 2.0 s with 63 workers on a 64-core box. `scripts/chartop_cpus.py` (standard library only) now takes the smallest of the cgroup v2 quota, the cgroup v1 quota, the affinity mask and `os.cpu_count()`, rounding a fractional quota down. Windows has no cgroup files and no `sched_getaffinity`, and reads the CPU count alone. The runner's `WORKERS`, `chartop_data.py`, `chartop_price.py` and `chartop_evaluate.py` use it, the box payload ships it, and the data manifests, the price record and the box's `env.log` say which source bound.

**The cost run.** `chartop_e5.py` no longer computes or records the learned answer's energy error, gate E4's measure, unless `--with-gate-measures` is given, and its record says whether it was. The coarse classical solve's error stays: it belongs to the competitor, not to the weights.

**The evaluation.** `chartop_evaluate.py --resume` continues an interrupted evaluation from checkpoints (`scripts/chartop_registry.py`, `evaluation-<time>-checkpoints/`): one per test shard and port basis, benchmark problem with its dual bracket, chunk of dual-bracket test pieces, KT benchmark, and the rollout. It refuses unless the weights, the gates module, `--test-count`, `--dual-test` and the test split (by content hash) are the ones the evaluation began with. A finished evaluation, and weights evaluated before the checkpoints existed (Stage 0's eleven), are still refused with or without `--resume`. A resumed record lists which session made which unit. Resuming also needed three changes: a test split is drawn again unless it holds exactly the registered seeds in both port bases (any one shard used to count as the split), the record is written last and whole, and a second evaluation on the same day no longer overwrites the first one's registry entry.

**Checked.** `tests/test_chartop_ops.py`, 71 tests. Fourteen deliberately re-introduced defects each fail the test written for them. On untrained weights and an 8-example test split drawn by the generator, one evaluation was killed during a benchmark and resumed, and another ran straight through: the two records agree in every number outside the timings and the provenance, and the test-rows arrays are identical. That run checks the machinery and is not a gate.

**Left alone.** `atlas/chartop/gates.py`, every record in `out/chartop/`, and [[chart-operator-stage-0]] §12 item 6, which still reads as a to-do. The checkpoint folder is not git-ignored (a few MB for a full evaluation), so it is committed with the record it belongs to.

**Changed:** [[gap-worklist]] W360.

## [2026-10-19] lint | The published site after the W359 push: the erratum is live, and three old dead links found

**What.** The erratum commit `7717d6d` was pushed to `atlas-0.1` on 2026-10-18 at 23:36 (local time), on the owner's yes, and the Website workflow (`.github/workflows/pages.yml`) deployed it. The launch checklist's live checks ([[website-launch-checklist]] §1a, steps 4, 6 and 7) were then run against `nonidino.github.io/Atlas/`.

**Found.**
- **The deployment is the commit.** All 198 files under `site/` are served with status 200 and the bytes the commit holds (text compared after CRLF to LF).
- **The architecture page, live.** Rendered in headless Edge, the erratum note shows, MathJax typesets 945 expressions with no error, and the rendered page is identical to the local one. The console of the five pages checked (architecture, home, install, evidence, blueprint) shows no error.
- **Relative links.** 172 distinct targets on the 71 pages; all resolve but one. `evidence.html` links three times to the directory `proofs/lean/AtlasProofs`, which has no `index.html`, so each answers 404. They come from `scripts/site_claims.py`, whose claims `lean.files`, `lean.theorems` and `lean.lines` give that directory as their public link, and so from `site/data/claims.json`. They predate the erratum. The fix belongs in the generator (a link to `proofs/lean/index.html` would resolve), after which `claims.json` and the page are regenerated. Not done: it is public content, so the owner's call.
- **External links of the architecture page.** 73 distinct URLs: 56 answer, 13 publisher pages refuse scripted requests (403), and 4 publisher hosts did not answer from this machine. All 17 of those DOIs are registered (doi.org's handle API returns `responseCode` 1 for each), so none is broken.

**Not checked.** The external links of the other pages (among them the checklist's two literature links), and the pages at phone width.

**Changed:** nothing but this entry.

## [2026-10-19] build | The chart operator's kernel-exact and split output forms (W356, W357), checked untrained

**What.** The first two fixes Stage 0 asked for ([[chart-operator-stage-0]] §12, items 1–2), built and checked with untrained weights only: zero heads, or seeded random weights. Nothing was trained, registered or hash-pinned, and `atlas/chartop/gates.py` is unchanged. The record is [[chart-operator-kernel-exact-forms]].

**The identity.** For any piece, $Hc_1+s_0u_p(\mathbf 1)=\mathbf 1$: a uniform temperature is a fixed point of the step. Steady, the corner hats sum to one. On a short backward-Euler step the hats alone fall to 0.02 of one, and the particular field of the old temperature makes up the rest. Exact classical solves meet it to $7\times10^{-15}$ in both port bases.

**Built.**
- `atlas/chartop/forms.py` holds the kernel-exact form (`"kernel"`). It projects the hats onto the identity with a partition of unity, so the identity holds to rounding whatever the weights. The form predicts a level field and splits the source's level off to it.
- The split form (`"split"`): a mode network without the load, beside a particular-field network.
- The measure: $e_K$, the uniform state's residual, and the level-weighted $e_H(\theta)=(e_H^2+\theta^2e_K^2)^{1/2}$.
- The expert, the host's `Response` and the trainer carry the new forms, and `benchmarks.datum_shift` lowers every temperature, $T_{\text{old}}$ included. Stage 0's outputs, arm B's among them, are bit for bit unchanged.
- 46 new test cases.

**Measured, untrained** (`scripts/chartop_forms_controls.py` → `out/chartop/forms-controls.json`):
- **The datum drops out.** Lowering every temperature by 300 K changes the new forms' coupled error by at most $1.4\times10^{-10}$ on all six problems, whatever the weights. A random bubble's error falls 1.6–213 times.
- **The steps.** As initialised, their errors fall from the principal part's 60/52/32% to 0.27/0.69/0.39%. Lowered by 300 K, where the level no longer inflates the reference energy, they are 3.9/10.8/4.1%. The steady errors start where the principal part's are.
- **The measure sees B.** B's $e_H$ is within 0.5% of the principal part's on the steady problems, while its $e_H(\theta)$ at the pieces' own levels is 2.6–4.3 times its $e_H$.
- **What the network is asked.** The side modes ask what Stage 0's did, so W358 is untouched. The level field's quotient peaks next to a corner and grows only 1.8 times from $n=32$ to 128.

**Drafted, not registered.** The next registration ([[chart-operator-kernel-exact-forms]] §8): gates K, D and L with numerical bars, E4 met in both frames, comparisons CK1 and CK2, and four predictions. Whether to train now or after W358's demonstration is the owner's decision.

**Changed:** [[chart-operator-kernel-exact-forms]] (new) · [[chart-operator-stage-0]] §12 and See Also · [[gap-worklist]] W356, W357 · [[index]].

## [2026-10-19] correction | The three dead evidence.html links fixed, with a test that would have caught them (not pushed)

**What.** The evidence page's three Lean records (`lean.files`, `lean.theorems`, `lean.lines`) linked to the folder `proofs/lean/AtlasProofs`, which holds no `index.html` and so answers 404 on the live site (found by the live check in the entry above). On the owner's yes, `cite()` in `scripts/site_claims.py` now maps that source folder to `proofs/lean/index.html`, the Lean project's own index page. `site/data/claims.json` and `site/evidence.html` were regenerated, three links each and nothing else. `scripts/site_build.py` was not run whole, because it also copies the architecture page and the Lean sources, and another session was applying edits to the architecture page at the time.

**A test.** `tests/test_site_claims.py` gained a check that every relative link on the site's three pages names a file or a folder that holds an `index.html`. Against the unfixed site it fails on exactly `evidence.html: proofs/lean/AtlasProofs`; now all 9 tests pass. The local link check of the 71 pages finds 0 broken, against 3 before, and the evidence page, the Lean index and the home page show no console error in headless Edge.

**Not done.** Not pushed: the live site keeps the three dead links until the next push, and the live re-check follows it. The commit (`688e559`) sits on local `atlas-0.1` beside other sessions' unpushed Stage 0 commits, which a push would publish too.

**Changed:** `scripts/site_claims.py`, `site/data/claims.json`, `site/evidence.html` and `tests/test_site_claims.py` (commit `688e559`) · this entry.

## [2026-10-19] correction | The architecture page's second erratum (W361): the default port basis, and the SE3 bar

**What.** The owner said yes to the second erratum drafted for W361, both items. The public architecture page, `proposal/architecture/index.html` and its byte-identical copy `site/architecture/index.html`, was not consistent about the default port basis, and its Table 27 named a different quantity for SE3 from the one registered. Stage 0 built the conforming basis, four corner hats and 15 side sines per side, as its default and registered the per-side cosines as comparison C6 (D3, [[chart-operator-stage-0]] §1 and §4.2).

**Changed on the page.**
- A second dated note under the first, the version line ("errata of 18 October 2026 and 19 October 2026") and the footer.
- Remark 4 and the first item of Section 4.5. The page's own recommendation F2 (Table 30) and its Expert Card already named sines plus corner functions as the default, but Section 4.5 still said "16 cosine coefficients per side" and Remark 4 called the conforming basis an alternative. Remark 4 now defines the default basis and calls the cosines the alternative (C6). Section 4.5 names the 15 side amplitudes and four corner values, $4\cdot15+4=4m=64$ numbers per piece.
- The proof of Theorem 5, which said that $Q_i$ has orthonormal columns. That held for version 1's orthonormalised cosines and is false of corner hats plus side sines: the normalised Gram matrix of their columns has off-diagonal entries up to 0.55. The proof needs only independent columns, which is all Lean's `gram_posDef` assumes. The statements of Theorems 5 and 6 are untouched.
- Table 27's SE3 bar. It read "below the network's E1 error", a per-bin statistic on held-out pieces. The registered bar ([[chart-operator-stage-0]] §3, `atlas/chartop/gates.py`) is the 16-mode truncation error below the network term of Theorem 6's bound, on the same problem. The intent matched and the quantity did not.

**Found on the way.** The text that was wrong is version 1's. "16 cosine coefficients per side" is version 1's sentence (`fe2baf0`), and version 1's card said `port_basis: cosine`. Version 2 (`732cd6d`) changed the card to `sine-plus-corner`, added F2 and Remark 4's "prefer it", and reworded SE3 (version 1: "below the E1 error of the network"), but left Section 4.5, the proof of Theorem 5 and the Remark's "alternative" as they were. The probes behind Tables 5, 24 and 25 (`arch_superelement_gram.py`, `arch_coupling_cost.py`) took cosine modes, so the note says that their numbers are reported as measured there.

**Left alone.** Section 2.3, which defines port modes by the cosines the probes used (Tables 5, 24 and 25 and the example's 1,792 interface unknowns are measured on them). The CO-D card's "cosines, or the conforming sines plus corner functions", which claims no default. Table 28's C6 row, whose title is the registered gate's. F2, the Expert Card and Section 10, which already agree with D3. `paper-v1.html`. [[chart-operator-training-and-cost]] §6, which still carries version 2's SE3 wording as the plan. No Stage 0 result went onto the website: the note says that Stage 0 registers the cosines as C6 and gives no ratio and no verdict.

**Checked.** Every edit's old text occurs once in each copy, and the copies stay byte-identical. `scripts/arch_paper_build.py` leaves the corrected page unchanged, so its cross-reference labels are the build's own; a deliberately wrong label is repaired by it, so the check can fail. `tests/test_site_claims.py` passes (9 tests, on a clean export of `02eda67` with the two pages applied). Rendered in headless Edge, MathJax typesets 952 expressions against 945 before, with no errors; the notes, Remark 4, Section 4.5, the proof of Theorem 5 and Table 27 read correctly in both colour schemes. The local link check of the page finds no new broken link (395 links against 380, none broken). Not pushed when written: the live check follows the push.

**Changed:** [[gap-worklist]] W361 · [[chart-operator-stage-0]] §13.

## [2026-10-19] lint | The published site after the W361 push: the second erratum is live

**What.** The commit `e6ad0c0` was pushed to `atlas-0.1` on 2026-10-19 at 09:49 (local time), on the owner's yes, and the Website workflow deployed it (success). The launch checklist's live checks were run against `nonidino.github.io/Atlas/`.

**Found.**
- The live architecture page equals the committed one (text compared after CRLF to LF) and carries the second erratum.
- Rendered in headless Edge, MathJax typesets 952 expressions, the same as the local page, with no input error and no unresolved reference.
- The home, install, evidence, archived version 1, Lean index and blueprint pages all answer 200, and every relative target of the architecture page does too.

**Not checked.** The external links of the other pages, and the pages at phone width.

**Changed:** nothing but this entry.

## [2026-10-19] note | W358: the corner bubble and the strips, checked untrained

Four candidate fixes for the bubble form's corner growth, compared on diagnose's validation pieces at n = 32 to 256 with classical solves only (`scripts/chartop_corner_controls.py` → `out/chartop/corner-controls.json`): the corner bubble with strips for the sines j = 9-15 is the only one whose peak levels off, and is recommended. Written once in [[chart-operator-kernel-exact-forms]] §10; [[gap-worklist]] W358 is in progress.
