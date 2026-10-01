# Formal proofs — the record: what is checked, what it says, and what a build costs

**Type:** Concept page — **proof record** (folder: `Atlas 0.1/atlas-0.1-proposal/formal-proofs/`)
**Status:** started 2026-10-01. **The set-up is done and timed. The statements of steps 1, 2 and 3 are written in Lean and in the blueprint, and they typecheck. Nothing is proved yet: every theorem below still holds a `sorry`, and the owner has not yet approved the statements.** One statement of the plan turned out false as written (T3 for restricted additive Schwarz, §5.1).
**Hub:** [[00-proposal-workstreams]] · **Plan:** [[formal-proofs-plan]] · **Formalises results from:** [[master-error-bound]] · [[defect-correction-learned-operator]] · [[temporal-error-accumulation]] · [[composition-error-theory]] · [[atlas-and-standard-dd-theory]]
**Code:** `lean/` (the Lean project and its blueprint) · `scripts/lean_build.py` · `scripts/lean_axioms.py` · `scripts/lean_t3_counterexample.py` · `out/lean/`

---

## 0. How to read this page

Each theorem appears three times: **in plain words**, **as mathematics**, and **as a Lean 4 declaration**. A theorem is **checked** only when all three of these hold:

1. `lake build` succeeds with no `sorry` in the project;
2. `#print axioms` on the declaration shows nothing beyond Lean's three standard axioms, `propext`, `Classical.choice` and `Quot.sound` (`scripts/lean_axioms.py` asks the kernel, and its positive control is a run that must *see* a `sorry`);
3. the build was timed and the time is written here.

A machine check guarantees that the proof proves the statement. It does not guarantee that the statement is the one that was meant. So the plain words come first, and the owner reviews them before anything is proved.

**Status words used below:** *drafted* (the Lean statement exists and typechecks, the proof is a `sorry`) · *approved* (the owner has read and accepted the statement) · *checked* (the three conditions above) · *finding* (the statement was false, or needed a hypothesis; §5).

---

## 1. The toolchain, and what a build costs

| what | value |
|---|---|
| toolchain manager | elan 4.2.4, installed with the official Windows script |
| Lean | `leanprover/lean4:v4.34.1` |
| Mathlib | tag `v4.34.1`, pinned in `lean/lakefile.toml` and `lean/lake-manifest.json`; fetched prebuilt with `lake exe cache get` (8908 files), **never compiled here** |
| blueprint | `leanblueprint` 0.0.20 on plasTeX 3.1. No separate Graphviz install was needed: the dependency graph is laid out in the browser, and the one external Graphviz program the plugin calls by default (`tred`) is switched off by its `nonreducedgraph` option |
| sources | `lean/`, tracked by git |
| build folder | `C:\Users\Nauni\.cache\atlas-lean`, **outside OneDrive** |

**Why the build is not in `lean/.lake`.** The repository lies inside OneDrive. Lake's `.lake` folder holds the unpacked Mathlib cache: several gigabytes in tens of thousands of files. Built in place, OneDrive would upload all of it, and it holds freshly written files for a moment, which breaks a build that writes thousands. So `scripts/lean_build.py` mirrors the sources into the build folder and runs `lake` there. Anywhere else (continuous integration, a clone outside OneDrive) plain `lake build` in `lean/` works: nothing in the project depends on the script.

**Disk.** Measured folder by folder after the install: the Lean toolchain (`.elan`) **3.1 GB**; the build folder **7.5 GB in 144,362 files** (the Mathlib sources and its unpacked cache); Mathlib's packed cache (`.cache\mathlib`) **0.45 GB**. **11.1 GB in all.** Free space on `C:` fell from 384.6 GB to 371.4 GB over the same hour, 13.2 GB; the 2.1 GB between the two figures is not attributed here (two other chats were writing to the disk, and 144,362 small files each round up to a whole cluster). The file count is the reason the build is kept out of OneDrive.

### 1.1 Timings

Every `lake` run goes through `scripts/lean_build.py`, which appends one line to `out/lean/builds.jsonl`: the command, the wall time, the exit code, the number of jobs, the errors, the `sorry` warnings, and **whether the laptop was on battery**.

**Every time below was measured on battery, with the processor held at 1.4 GHz of its 3.8 GHz, except the last row, taken on mains.** Measured on the same 13 files: a first-level file costs 26 to 30 s on battery and 12 to 15 s on mains once the files are cached, about half. A time on battery and a time on mains are not comparable without that factor, and a first load of the day is four times slower again.

| run | what it did | wall time | note |
|---|---|---|---|
| `lake update` | downloaded the Lean toolchain, cloned Mathlib and its dependencies, fetched and unpacked the Mathlib cache | 698 s | once per machine |
| `lake exe cache get` | nothing left to fetch | 294 s | **not a clean time**: an over-sized exploration script of this chat's own was running beside it |
| first build | an empty file importing one Mathlib module (`Mathlib.Topology.MetricSpace.Contracting`, 1687 modules behind it) | 201.5 s | cold: the first read of 1687 freshly unpacked files |
| second build | the same file with one comment added | 54.3 s | warm |
| one file alone | `lake env lean` on that file, profiler on | 28 to 29 s | **10 s loading the imported modules, 15.8 s running their initialisers, under 1 s for everything else** |
| statements | the nine statement files of steps 1 and 2, each proof a `sorry` | 81.6 s for five new files and the root (four replayed) | 17 to 30 s per file |
| statements | the four statement files of step 3 added | 102.6 s for the four new files and the root (nine replayed) | 28 to 43 s per file; the slowest imports the calculus library |
| axiom check | `#print axioms` on 39 declarations, then on 58 | 75.0 s, then 32.4 s | the second run was faster though it checked more. **[AI Inference]:** the operating system had the files cached by then |
| **full re-check, on mains** (2026-10-01, 16:24) | every project file re-checked: 13 statement files and the root | **136.5 s cold** (the first load of the day: 36 to 64 s per file, eleven at once), then **34.0 s warm** (12 to 15 s per first-level file, 10 s per file one level up) | **the only rows on this page not taken on battery** |

**What the numbers say about the owner's concern.** On this machine a Lean file costs about **26 seconds before its first proof is looked at**: the price of loading Mathlib's compiled files and running their start-up code. It is paid once per file, and files that do not import each other are built at the same time. So the cost of a full build is set by **how many files stand in a chain**, not by how many there are, and the project keeps that chain short. Proof checking itself has cost nothing measurable so far, and the rule stands: a tactic call over ten seconds is replaced by an explicit step (`python scripts/lean_build.py --profile` lists every such call).

**Two things would shorten every time above, and neither is this chat's to do:** running on mains power; and a Windows Defender exclusion for the build folder and `C:\Users\Nauni\.elan`, since real-time scanning is on and a build opens thousands of files. **[AI Inference]** on the second: its size is not measured.

### 1.2 An instrument that was wrong, and its control

The first version of `scripts/lean_build.py` counted `sorry` warnings with a pattern that matched neither Lake's line format nor Lean's wording, and reported **0 sorries on a log that held 32**. It was found on the first build that should have shown some. The patterns are now anchored on the line's first word, `--reparse` re-reads a saved log, and the two record lines written before the fix were recounted from their logs. The claim "no `sorry`" does not rest on that count: it rests on `scripts/lean_axioms.py`, which asks the kernel, and whose positive control (`--expect-sorry`) saw all 32.

---

## 2. Mathlib's names, confirmed against the installed version

[[formal-proofs-plan]] quoted declaration names from memory. Each was checked against Mathlib `v4.34.1`.

| the plan's name | in Mathlib `v4.34.1` | correction |
|---|---|---|
| `ContractingWith` | `Mathlib/Topology/MetricSpace/Contracting.lean` | none |
| `ContractingWith.fixedPoint` | same file | none |
| `ContractingWith.dist_fixedPoint_le` | same file | none. It is T7's Banach form |
| `ContractingWith.apriori_dist_iterate_fixedPoint_le` | same file | none. There is also `aposteriori_dist_iterate_fixedPoint_le` |
| *(not quoted)* | `ContractingWith.dist_fixedPoint_fixedPoint_of_dist_le'` and `ContractingWith.fixedPoint_lipschitz_in_map` | **T1's distance bound is already a Mathlib lemma**, for a defect bounded everywhere. T1 here needs the defect only at the classical answer, which is a little more |
| the inner-product-space API | `InnerProductSpace`, `norm_add_sq_real`, `norm_sub_sq_real`, `real_inner_self_eq_norm_sq`, `real_inner_le_norm` | the inner product now takes its scalar field as an explicit first argument |
| `spectralRadius` | `Mathlib/Analysis/Normed/Algebra/Spectrum.lean`; Gelfand's formula is `spectrum.pow_nnnorm_pow_one_div_tendsto_nhds_spectralRadius` in `Mathlib/Analysis/Normed/Algebra/GelfandFormula.lean`, for complex Banach algebras | `spectrum.spectralRadius_le_nnnorm` is deprecated in favour of `spectralRadius_le_nnnorm` |
| Lax–Milgram | `IsCoercive.continuousLinearEquivOfBilin` in `Mathlib/Analysis/InnerProductSpace/LaxMilgram.lean`, for real Hilbert spaces | none |
| Krasnoselskii–Mann (T9′) | **not in Mathlib** (searched) | T9′ is proved from scratch, as the plan's Tier B assumed |
| Schur complements | `Matrix.fromBlocks` and its determinant lemmas in `Mathlib/LinearAlgebra/Matrix/SchurComplement.lean` | T3a is stated for linear maps, so it does not need them |

---

## 3. The record

One row per theorem. **Build time** is the file's own time in the full build that checked it; **axioms** is what `#print axioms` returned.

| # | statement, in one line | Lean name | file | status | build time | axioms |
|---|---|---|---|---|---|---|
| **T1** | a contraction that differs from a map by $\delta$ at that map's fixed point has a unique fixed point within $\delta/(1-\rho)$ of it, reached at rate $\rho$ | `Atlas.perturbed_contraction` | `AtlasProofs/PerturbedContraction.lean` | **checked** | 3.2 s, cloud container | `propext`, `Classical.choice`, `Quot.sound` |
| T1, uniform | the same with the defect bounded everywhere | `Atlas.perturbed_contraction_uniform` | same | **checked** | 3.2 s, cloud container | the same three |
| T1, sharp | $\delta/(1-\rho)$ is attained | `Atlas.perturbed_contraction_sharp` | same | **checked** | 3.2 s, cloud container | the same three |
| **T7** | error $\le$ residual $/(1-L)$, when the step contracts *this* state toward the fixed point | `Atlas.certificate` | `AtlasProofs/Certificate.lean` | **checked** | 2.8 s, cloud container | the same three |
| T7, Banach | the same for every state, under a global contraction | `Atlas.certificate_of_contraction` | same | **checked** | 2.8 s, cloud container | the same three |
| **T7, W208** | for a linear step the least certifying constant is $\lVert(I-A)^{-1}\rVert$ | `Atlas.certificate_constant_is_opNorm` | same | **checked** | 2.8 s, cloud container | the same three |
| T7, example | $\operatorname{diag}(0.6,\,0.95)$: the ratio is $2.5$ on one mode and $20$ on the other | `Atlas.certificate_two_mode_example` | same | **checked** | 2.8 s, cloud container | the same three |
| **T4a** | the exact error recursion | `Atlas.error_recursion` | `AtlasProofs/MasterBound.lean` | **checked** | 2.0 s, cloud container | the same three |
| **T4b** | the defect is $\tau+\sigma+\gamma$ | `Atlas.defect_split` | same | **checked** | 2.0 s, cloud container | the same three |
| **T4c** | discrete Gronwall | `Atlas.accumulation` | same | **checked** | 2.0 s, cloud container | the same three |
| **T4** | the master bound | `Atlas.master_bound` | same | **checked** | 2.0 s, cloud container | the same three |
| T4, three terms | the master bound with $\tau$, $\sigma$, $\gamma$ named | `Atlas.master_bound_three_terms` | same | **checked** | 2.0 s, cloud container | the same three |
| **T4, $L<1$** | bounded by $\delta/(1-L)$ for all time | `Atlas.master_bound_contractive` | same | **checked** | 2.0 s, cloud container | the same three |
| **T4, $L=1$** | at most $N\delta$ | `Atlas.master_bound_nonexpansive` | same | **checked** | 2.0 s, cloud container | the same three |
| **T4, $L>1$** | at most $\frac{L^N-1}{L-1}\delta$ | `Atlas.master_bound_expansive` | same | **checked** | 2.0 s, cloud container | the same three |
| T4, attained | the scalar recursion attains the geometric sum | `Atlas.master_bound_attained` | same | **checked** | 2.0 s, cloud container | the same three |
| T4e | $\lVert\lambda^\dagger-\lambda^\star\rVert\le\lVert\Lambda-\tilde\Lambda\rVert\,\lVert\lambda^\star\rVert/\beta$ | `Atlas.trace_perturbation` | `AtlasProofs/TransmissionBound.lean` | **checked** | 2.3 s, cloud container | the same three |
| T4e | $\sigma\le\frac{C_\mu}{\beta}\lVert\Lambda-\tilde\Lambda\rVert\,\lVert\lambda^\star\rVert$ | `Atlas.transmission_bound` | same | **checked** | 2.3 s, cloud container | the same three |
| **T3a** | the undivided block system and the Schur system have the same solutions | `Atlas.schur_iff` | `AtlasProofs/Schur.lean` | **checked** | 1.9 s, cloud container | the same three |
| **T3b** | a fixed point of Dirichlet–Neumann solves the undivided system | `Atlas.TwoPieces.dn_fixedPoint_solves` | `AtlasProofs/DirichletNeumann.lean` | **checked** | 2.0 s, cloud container | `propext`, `Quot.sound` |
| T3b, converse | the undivided solution is a fixed point | `Atlas.TwoPieces.dn_solution_isFixedPt` | same | **checked** | 2.0 s, cloud container | `propext`, `Quot.sound` |
| T3b, face form | the same for the finite-volume form `styles.dirichlet_neumann` runs | `Atlas.FacePair.solves_of_fixedPoint` | same | **checked** | 2.0 s, cloud container | the same three |
| T3c | the Schwarz sweep is $u\mapsto u+M(f-Au)$ | `Atlas.Schwarz.sweep_eq` | `AtlasProofs/Schwarz.lean` | **checked** | 2.1 s, cloud container | the same three |
| **T3c** | the undivided solution is a fixed point of the sweep | `Atlas.Schwarz.solution_isFixedPt` | same | **checked** | 2.1 s, cloud container | the same three |
| T3c | $u$ is a fixed point iff $M(f-Au)=0$ | `Atlas.Schwarz.isFixedPt_iff` | same | **checked** | 2.1 s, cloud container | the same three |
| **T3c** | if every window agrees with $u$, then $Au=f$ | `Atlas.Schwarz.solves_of_windows_agree` | same | **checked** | 2.1 s, cloud container | the same three |
| **T3c** | if $M$ is one-to-one, every fixed point solves $Au=f$ | `Atlas.Schwarz.fixedPt_solves_of_precond` | same | **checked** | 2.1 s, cloud container | the same three |
| T3c | the same hypothesis read off the sweep | `Atlas.Schwarz.fixedPt_solves` | same | **checked** | 2.1 s, cloud container | the same three |
| **T3c** | a convergent sweep has one fixed point, the solution | `Atlas.Schwarz.fixedPt_eq_solution` | `AtlasProofs/SchwarzConvergence.lean` | **checked** | 2.2 s, cloud container | the same three |
| **T3c** | a convergent sweep converges to the solution | `Atlas.Schwarz.tendsto_solution` | same | **checked** | 2.2 s, cloud container | the same three |
| **T3d** | error $\le$ update $/(1-\rho)$ | `Atlas.Schwarz.error_le_update` | same | **checked** | 2.2 s, cloud container | the same three |
| **T3c, finding** | a Schwarz sweep with a fixed point that is not the solution | `Atlas.Schwarz.exists_spurious_fixedPt` | `AtlasProofs/SchwarzCounterexample.lean` | **checked** | 7.9 s, cloud container | the same three |
| **T5, Theorem 1** | any limit of defect correction is a fixed point of the classical map | `Atlas.DefectCorrection.limit_isFixedPt` | `AtlasProofs/DefectCorrection.lean` | approved | — | — |
| T5, Theorem 1, inexact | the same when each step is solved only approximately, the leftover tending to zero | `Atlas.DefectCorrection.limit_isFixedPt_of_inexact` | same | approved | — | — |
| **T5, Corollary 1** | a constant cheap map makes the iteration the classical march | `Atlas.DefectCorrection.step_of_const` | same | approved | — | — |
| T5 | the inner march's fixed points are the successors | `Atlas.DefectCorrection.isStep_iff` | same | approved | — | — |
| T5, Corollary 1 | with a constant cheap map the inner march returns $\Phi(w)$ at every stage | `Atlas.DefectCorrection.innerMarch_of_const` | same | approved | — | — |
| T5, Corollary 2 | $J_{\Psi_\alpha}=\alpha I+(1-\alpha)J_\Psi$, as a derivative | `Atlas.DefectCorrection.shrink_hasFDerivAt` | `AtlasProofs/DefectCorrectionShrink.lean` | approved | — | — |
| T5, Corollary 2 | an eigenvalue $\mu$ becomes $\alpha+(1-\alpha)\mu$ | `Atlas.DefectCorrection.shrink_hasEigenvalue` | same | approved | — | — |
| **T5, Corollary 2** | and those are all the eigenvalues, for $\alpha\ne1$ | `Atlas.DefectCorrection.shrink_hasEigenvalue_iff` | same | approved | — | — |
| T5, Corollary 2 | the real part is at least $\alpha$ | `Atlas.DefectCorrection.shrink_re` | same | approved | — | — |
| **T10** | the group average of any map is exactly equivariant | `Atlas.average_equivariant` | `AtlasProofs/SymmetryAveraging.lean` | approved | — | — |
| T10 | averaging returns an equivariant map unchanged | `Atlas.average_of_equivariant` | same | approved | — | — |
| T10, example | averaging over $\{\mathrm{id},M_x,M_y\}$, not a group, is not equivariant | `Atlas.average_over_non_group_not_equivariant` | same | approved | — | — |
| T11 | the connection rule conserves power | `Atlas.port_rule_power` | `AtlasProofs/Passivity.lean` | approved | — | — |
| T11 | the junction: connected port powers sum to zero | `Atlas.junction_power_eq_zero` | same | approved | — | — |
| **T11** | passive agents joined by the rule make a passive system | `Atlas.interconnection_passive` | same | approved | — | — |
| **T11, error form** | incrementally passive agents make a non-expansive step | `Atlas.interconnection_nonexpansive` | same | approved | — | — |

---

## 4. The statements

### 4.1 Step 1 — T1, T7, T4

**T1, perturbed contraction.** *In plain words:* replace the classical coupling map by a learned one. If the learned map shrinks every distance by a factor $\rho<1$, and at the classical answer it differs from the classical map by at most $\delta$, then the learned iteration has exactly one answer, reaches it geometrically from any start, and that answer is within $\delta/(1-\rho)$ of the classical one. Nothing is assumed about training, and the classical map need not contract.

Let $(X,d)$ be a complete metric space and $T_c,T_l:X\to X$ with $T_c(a_c)=a_c$, $\;d(T_lx,T_ly)\le\rho\,d(x,y)$ for all $x,y$ with $0\le\rho<1$, and $d(T_la_c,T_ca_c)\le\delta$. Then there is a unique $a_l$ with $T_l(a_l)=a_l$, and

$$d\bigl(T_l^{\,k}x_0,\,a_l\bigr)\le\rho^{k}\,d(x_0,a_l)\quad\text{for all }x_0,\ k,\qquad\qquad d(a_l,a_c)\le\frac{\delta}{1-\rho}.$$

The plan asked for the defect bound at every point. **It is needed only at $a_c$**, and the theorem is stated that way, with the plan's form as a corollary. The constant cannot be improved: $T_c(x)=\rho x$ and $T_l(x)=\rho x+\delta$ on $\mathbb R$ attain it.

**T7, the a-posteriori certificate.** *In plain words:* a solver hands back a state $w$. How far one more classical step moves it, the residual, can be computed. How far it is from the settled state $w^\star$ cannot. If the step brings $w$ closer to $w^\star$ by a factor $L<1$, the error is at most the residual over $1-L$.

Let $\Phi(w^\star)=w^\star$ and $d(\Phi w,\Phi w^\star)\le L\,d(w,w^\star)$ with $L<1$. Then

$$d(w,w^\star)\le\frac{d(w,\Phi w)}{1-L}.$$

**W208's lesson is the hypothesis.** The contraction is required of the state being certified, not of some other state. For a linear step $\Phi(w)=Aw+b$ with $I-A$ invertible, this is exact:

$$\lVert w-w^\star\rVert\le\bigl\lVert(I-A)^{-1}\bigr\rVert\,\lVert\Phi(w)-w\rVert\ \text{ for every }w,\qquad\text{and no smaller constant does this for every }w.$$

So a constant read along one march is a lower bound on the constant needed, never an upper bound. The pinned two-mode example is checked too: for $\Phi=\operatorname{diag}(0.6,\,0.95)$ the ratio of error to residual is $2.5$ along one mode and $20$ along the other.

**T1's distance bound is T7 applied to the learned map at the classical answer**: the residual of $a_c$ under $T_l$ is at most $\delta$.

**T4, the master bound.** *In plain words:* the new error is the old error carried through one step, plus what one step does wrong from the true solution. With a coupling, that defect has three named parts. If one step magnifies a difference by at most $L$, the error after $N$ steps is at most $L^N$ times the initial error plus every defect magnified by the steps after it.

With $u^{n+1}=\Phi(u^n)$ and any sequence $u^{n,\star}$:

$$u^{n+1}-u^{n+1,\star}=\bigl[\Phi(u^n)-\Phi(u^{n,\star})\bigr]+\underbrace{\bigl[\Phi(u^{n,\star})-u^{n+1,\star}\bigr]}_{d^{n+1}},$$

$$d=\underbrace{\Phi_{\lambda^\star}-\mathcal S}_{\tau}+\underbrace{\Phi_{\lambda^\dagger}-\Phi_{\lambda^\star}}_{\sigma}+\underbrace{\Phi_{\lambda^{(k)}}-\Phi_{\lambda^\dagger}}_{\gamma},$$

and if $\lVert\Phi(u^n)-\Phi(u^{n,\star})\rVert\le L\lVert u^n-u^{n,\star}\rVert$ for every $n$, with $L\ge0$,

$$\lVert u^N-u^{N,\star}\rVert\le L^N\lVert u^0-u^{0,\star}\rVert+\sum_{n=1}^{N}L^{N-n}\lVert d^n\rVert .$$

With $\lVert d^n\rVert\le\delta$: at most $L^N\lVert e^0\rVert+\delta/(1-L)$ when $L<1$; at most $\lVert e^0\rVert+N\delta$ when $L=1$; at most $L^N\lVert e^0\rVert+\frac{L^N-1}{L-1}\delta$ when $L>1$. The stability hypothesis is needed only **along the two trajectories**, which is weaker than a global Lipschitz constant. The scalar recursion attains the geometric sum, so the three regimes are not artefacts of the proof.

**T4e, the transmission term (offered beyond the plan's list).** If $\Lambda\lambda^\star=\chi$, $\tilde\Lambda\lambda^\dagger=\chi$ and $\beta\lVert x\rVert\le\lVert\tilde\Lambda x\rVert$, then $\lVert\lambda^\dagger-\lambda^\star\rVert\le\lVert\Lambda-\tilde\Lambda\rVert\,\lVert\lambda^\star\rVert/\beta$, and with a step that is $C_\mu$-Lipschitz in its interface datum, $\sigma\le\frac{C_\mu}{\beta}\lVert\Lambda-\tilde\Lambda\rVert\,\lVert\lambda^\star\rVert$. This is the boxed bound of [[master-error-bound]] §4.

### 4.2 Step 2 — T3

**T3a, a direct Schur interface solve.** *In plain words:* split the unknowns into those inside the pieces and those on the interface. Solve a smaller system for the interface alone, then each piece for its interior. The result is the undivided solution, and every undivided solution arises this way.

With $A_{II}$ invertible:

$$\begin{cases}A_{II}u_I+A_{I\Gamma}u_\Gamma=f_I\\ A_{\Gamma I}u_I+A_{\Gamma\Gamma}u_\Gamma=f_\Gamma\end{cases}\iff\begin{cases}u_I=A_{II}^{-1}(f_I-A_{I\Gamma}u_\Gamma)\\ \bigl(A_{\Gamma\Gamma}-A_{\Gamma I}A_{II}^{-1}A_{I\Gamma}\bigr)u_\Gamma=f_\Gamma-A_{\Gamma I}A_{II}^{-1}f_I .\end{cases}$$

**T3b, Dirichlet–Neumann.** *In plain words:* two pieces meet along an interface. Piece 1 is solved with the interface values given; the flux it sends is handed to piece 2, which returns its own interface values; the interface values move toward those by a factor $\theta\ne0$. If a sweep returns the interface values it was given, the two pieces side by side solve the undivided problem. Convergence is not assumed. The converse holds too: the undivided solution is a fixed point.

Two forms are stated: the textbook block form, and the cell-centred finite-volume form that `atlas/workbench/styles.py` runs, where the interface carries face values. In the second, a fixed point makes the flow through every cut face equal to $h\,(u_P-u_Q)$ with $h=(r_D+r_N)^{-1}$, the two half-cells in series, which is the undivided grid's harmonic-mean face.

**T3c, restricted additive Schwarz.** *In plain words:* overlapping windows are each solved exactly with the current iterate held outside, and blended by weights that sum to one. Write $M=\sum_iE^\chi_iA_i^{-1}R_i$ for the blended sum of the window solves. Then the sweep is

$$G_f(u)=u+M\,(f-Au),$$

and:

- **the undivided solution is always a fixed point**;
- **if every window's own solution agrees with $u$ on its whole window, then $Au=f$**, with no further hypothesis;
- **if only the blend is unchanged**, $u$ is a fixed point exactly when $M(f-Au)=0$, so $Au=f$ follows **if and only if $M$ sends no non-zero residual to zero**;
- that holds **whenever the sweep converges from every start**: then the sweep has one fixed point, the undivided solution, and its iterates converge to it;
- **without it the claim is false** (§5.1).

**T3d, what stopping on the update leaves.** If the sweep with zero right-hand side shrinks a norm by $\rho<1$,

$$\lVert u-u^\star\rVert\le\frac{\lVert G_f(u)-u\rVert}{1-\rho}.$$

The tolerance bounds the update. The error is larger by $1/(1-\rho)$, which is $33$ for a sweep contracting at $0.97$.

### 4.3 Step 3 — T5, T10, T11

**T5, defect correction.** *In plain words:* $\Phi$ is the classical step and $\Psi$ any cheap map, a learned operator for instance. Defect correction defines the next iterate $w'$ by $w'-\Psi(w')=\Phi(w)-\Psi(w)$, so the cheap map enters only through a difference. **If the iterates settle anywhere, they settle on a fixed point of the classical map**, whatever $\Psi$ is. A wrong cheap map can cost classical calls; it cannot change what is returned.

**Theorem 1.** Let $w_{k+1}-\Psi(w_{k+1})=(w_k-\Psi(w_k))-(w_k-\Phi(w_k))$ for every $k$, and $w_k\to\bar w$. If $\Phi$ and $\Psi$ are continuous at $\bar w$, then

$$\Phi(\bar w)=\bar w .$$

Two points of precision against [[defect-correction-learned-operator]] §2.1. **Both maps must be continuous at the limit**: the page's statement names $G_\Psi$ and its proof also uses $G_\Phi$. And the statement is also drafted for **inexact inner solves**, which is what the code runs: if each step misses the defining equation by $\varepsilon_k$ and $\varepsilon_k\to0$, the conclusion is the same.

**Corollary 1.** If $\Psi$ is constant, the successor of $w$ is $\Phi(w)$, and the inner march $x^{(m+1)}=\Phi(w)+[\Psi(x^{(m)})-\Psi(w)]$ equals $\Phi(w)$ at every stage: the iteration is the classical march. (The page's "bit for bit" is a statement about floating-point arithmetic; the Lean statement is the exact one over real vector spaces.)

**Corollary 2.** For $\Psi_\alpha=(1-\alpha)\Psi$, the map $G_{\Psi_\alpha}=I-\Psi_\alpha$ has derivative $\alpha I+(1-\alpha)J_\Psi$ wherever $\Psi$ has derivative $D\Psi$ and $J_\Psi=I-D\Psi$. For $\alpha\ne1$,

$$\nu\text{ is an eigenvalue of }\alpha I+(1-\alpha)J\iff\nu=\alpha+(1-\alpha)\mu\text{ for an eigenvalue }\mu\text{ of }J,$$

and for $0\le\alpha\le1$ and $\operatorname{Re}\mu\ge0$, $\operatorname{Re}\bigl(\alpha+(1-\alpha)\mu\bigr)\ge\alpha$. So the shrunk map is no closer to singular than $\alpha$ on any mode where the original was not on the wrong side of zero.

**T10, symmetry averaging.** *In plain words:* call the expert once per element of the symmetry group, undo each transformation, and average. The result respects the symmetry exactly, whatever the expert is.

For a finite group $G$ acting on the inputs, and linearly on the outputs, and **any** map $E$,

$$\tilde E(u)=\frac1{\lvert G\rvert}\sum_{g\in G}g^{-1}E(g\,u)\qquad\Longrightarrow\qquad\tilde E(h\,u)=h\,\tilde E(u)\ \text{ for all }h\in G .$$

If $E$ was already equivariant, $\tilde E=E$. And for the set $\{\mathrm{id},M_x,M_y\}$ of [[symmetry-averaging-atlas-0.1]] §2.1, which is not a group, the same average is not equivariant: with $E(x,y)=(x+y,0)$ and $u=(0,1)$ it gives $(-\tfrac13,0)$ at $M_xu=u$, while $M_x$ of the average at $u$ is $(\tfrac13,0)$.

**T11, passive interconnection.** *In plain words:* a port carries an effort and a flow whose inner product is power. The port algebra joins two ports by equal efforts and opposite flows. Then what enters one port leaves its partner, the connected ports' powers cancel over any graph, and if every agent is passive the whole system is passive with respect to its open ports.

With $p(x)$ the power entering through port $x$, and $\pi$ pairing each connected port with its partner:

$$e_B=e_A,\ f_B=-f_A\ \Longrightarrow\ \langle e_B,f_B\rangle=-\langle e_A,f_A\rangle;\qquad p(\pi(x))=-p(x)\ \Longrightarrow\ \sum_xp(x)=0;$$

$$H_i'\le H_i+\sum_{x\text{ of }i}p(x)+\mathrm{ext}_i\ \text{ for every agent }i\qquad\Longrightarrow\qquad\sum_iH_i'\le\sum_iH_i+\sum_i\mathrm{ext}_i .$$

**The error form** is the theorem of [[master-error-bound]] §6.1: if every agent is incrementally passive, $\lVert u_i'-v_i'\rVert^2\le\lVert u_i-v_i\rVert^2+2\sum_{x\text{ of }i}\Delta p(x)$ for two runs with the same external forcing, then $\sum_i\lVert u_i'-v_i'\rVert^2\le\sum_i\lVert u_i-v_i\rVert^2$. That is $L\le1$ in the product norm, the hypothesis of T4's linear regime. **The statement is discrete in time** (one step's energy balance), which is what a code can check; the continuous-time form with derivatives of the storage is not drafted.

---

## 5. Findings

### 5.1 T3 for restricted additive Schwarz is false as the plan states it

**The plan's sentence:** "every fixed point of restricted additive Schwarz ... built from exact local solves, with a partition of unity, solves $Au=f$." **It needs a hypothesis: that the blended sum of the window solves, $M$, is one-to-one.** Without it there are fixed points that are not solutions.

**Example 1, three unknowns** (the one written for Lean). Windows $\{1,2\}$ and $\{2,3\}$; the shared unknown belongs wholly to the first window.

$$A=\begin{pmatrix}1&1&0\\1&0&1\\0&1&1\end{pmatrix},\qquad f=0,\qquad u=(1,-1,-1),\qquad Au=(0,0,-2).$$

$\det A=-2$ and both window matrices have determinant $-1$, so every solve is exact, and the only solution is $u=0$. The first window returns $(1,-1)$, which is $u$ there. The second returns $(1,-1)$ on $\{2,3\}$: it disagrees with $u$ on the shared unknown, and the blend discards exactly that value. So the sweep returns $u$. **Run through the workbench's own driver** (`atlas.workbench.styles.schwarz`, imported and not changed, by `scripts/lean_t3_counterexample.py`): `converged=True` after one sweep, with an update of exactly $0$ and a residual of $(0,0,2)$. Three controls behave the other way: the same $A$ started at the solution stays there; a positive definite $A$ with the same windows goes to the solution in 28 sweeps; the same $A$ without overlap is moved by the sweep.

**Example 2, positive definiteness does not rescue it.** A symmetric positive definite $6\times6$ rational matrix (leading principal minors $1,\ \tfrac{15}{16},\ \tfrac{11}{16},\ \tfrac{403}{768},\ \tfrac{77}{1536},\ \tfrac{71}{4096}$), windows $\{0,1,2\}$, $\{1,2,3,4\}$, $\{3,4,5\}$, each unknown owned by one window. **In exact rational arithmetic**, $\det M=0$, and $u=\tfrac1{71}(640,\,-1708,\,4800,\,1632,\,-5600,\,1400)$ is a fixed point of the sweep with $f=0$ although $Au=(3,12,6,-8,-6,0)$. The driver reports `converged=True` after one sweep there too. Two honest limits: the ownership is unusual (the first window keeps unknown 2 and gives away unknown 1), and **the sweep does not converge on this problem** (its error propagation has an eigenvalue of modulus $1.148$), so the example refutes the statement, not the practice.

**What is true, and is what the compiler's rule needs:**

1. agreement of every window with the blend implies $Au=f$ (unconditional);
2. **a sweep that converges from every start has exactly one fixed point, the undivided solution** (`Atlas.Schwarz.fixedPt_eq_solution`). Convergence is Frommer and Szyld's theorem for M-matrices (*SIAM J. Numer. Anal.* 39, 2001), which covers the workbench's conduction family. For elasticity, whose matrices are positive definite and not M-matrices, convergence is **measured**, not proved;
3. for **two** windows and a symmetric positive definite $A$, every fixed point is the solution. **[AI Inference]:** this is a paper proof made in this session (the two windows' corrections differ on the overlap by a vector the overlap's own positive definite block sends to zero), not yet formalised.

**For the demo chat, which owns the R10 change** ([[gap-worklist]] W348; `compiler._r10_scheme`): the rule's sentence "every fixed point of an iterated linear coupling built from exact local solves — restricted additive Schwarz, Dirichlet-Neumann — is the undivided discrete solution" is right for Dirichlet–Neumann and for a direct Schur solve, and for restricted additive Schwarz it should read "**the limit of a convergent sweep**". And "bounded by the tolerance" should read "bounded by the tolerance over $1-\rho$" (T3d): the driver stops on the **update**.

### 5.2 Smaller findings

- **T1's distance bound is already in Mathlib** (§2). T1 here is stated with the defect at one point only.
- **T1 and T7 are one inequality.** T1's bound is T7 at the classical answer.
- **T4's stability hypothesis is needed only along the two trajectories**, not globally.
- **T5's Theorem 1 needs both maps continuous at the limit.** [[defect-correction-learned-operator]] §2.1 states it with "$G_\Psi$ is continuous" and uses the continuity of $G_\Phi$ in its proof. The classical step is continuous in every use on that page, so nothing it concludes changes.
- **T10 needs nothing of the action on the inputs** beyond its being a group action; linearity is needed only on the outputs.

---

## 5A. Batches 1 to 3: the proofs

*Kept by the chat that proves batches 1 to 3.* It holds the build that checked each batch, the axiom check's result, the measured authoring rate that replaces the estimates of [[formal-proofs-implementation-plan]] §6.2, and any statement that had to change. **Every time in this section was measured in a cloud container**, not on the owner's laptop: a Linux container with 4 virtual processors and 15 GB of memory, no battery and so no power state. Its build times are not comparable with §1.1's laptop times without that fact. The record lines are in `out/lean/builds-batches-1-3.jsonl`, `axioms-batches-1-3.json` and `profile-batches-1-3.json` (`ATLAS_LEAN_TAG=batches-1-3`). Clock times are UTC, from the container's clock.

### 5A.1 The set-up in the cloud container

| step | what happened | wall time |
|---|---|---|
| branch | the session's branch fast-forwarded to `428e4aa` (21:03:18) | — |
| elan | 4.2.4, the official `elan-init.sh` | — |
| `lake update` | the toolchain `v4.34.1` from `releases.lean-lang.org`; Mathlib and its eight dependencies cloned from `github.com` at the manifest's revisions (the manifest did not change); the cache, 8908 files, from `cache.mathlib.org` | 153.6 s |
| `lake exe cache get` | nothing left to fetch | 12.2 s |
| first build | the 13 statement files: 2263 jobs, 48 `sorry` warnings, **no Mathlib module compiled** | 17.2 s |
| positive control | `lean_axioms.py --expect-sorry` on the 58 names: all 48 theorems fail on `sorryAx`, the 10 definitions pass: **seen** | 3.1 s |

**Disk:** 30 GB free before; afterwards the toolchain (`~/.elan`) holds 3.0 GB and the build folder 7.7 GB, and 19 GB are free.

### 5A.2 Batch 1: T1, T7, T4, T4e — checked

**What is checked:** all 18 declarations of `lean/decls-batch-1.txt`, exactly as stated; no statement changed. Two private helper lemmas were added in `MasterBound.lean` (`sum_pow_reflect`: the weights $L^{N-(i+1)}$, $i<N$, are $L^0,\dots,L^{N-1}$ reversed; `sum_defects_le`: with every defect at most $\delta$ the accumulated defects are at most $\delta\sum_{i<N}L^i$). Every proof follows the blueprint's paper proof.

| check | result |
|---|---|
| `lean_build.py` (incremental, 21:20:53) | success, 7.6 s; 30 `sorry` warnings, **none in a batch-1 file** (all 30 are in batches 2 and 3) |
| `lean_build.py --clean` (every project file re-checked, 21:23:53) | success, 13.0 s. Batch-1 files: `PerturbedContraction` 3.2 s, `Certificate` 2.8 s, `MasterBound` 2.0 s, `TransmissionBound` 2.3 s |
| `lean_axioms.py --names lean/decls-batch-1.txt` | **18 declarations, 0 failing**, each resting on `propext`, `Classical.choice` and `Quot.sound` only; 4.0 s |
| `lean_build.py --profile` | 13 files, 38.7 s in all (each file alone, one after another), **no step of ten seconds or more**; the batch-1 files take 3.0 to 4.3 s each |
| blueprint | `\leanok` in all 18 proof environments of `contraction.tex` and `master-bound.tex`. The two-mode example had no proof environment, and leanblueprint shows a node as proved only through its proof's `\leanok`, so a one-line proof (the arithmetic) was added. Built with plasTeX 3.1 and leanblueprint 0.0.20: no warning, and the dependency graph shows the 17 batch-1 nodes as fully proved (examples are not drawn in the graph) |

**The measured rate.** 18 declarations; **150 lines of proof text** (172 lines added in all, counting comments and the helpers' docstrings), against the plan's 370.

| window | time | declarations per hour | proof lines per hour |
|---|---|---|---|
| first proof edit (21:15:49) to the batch verified, the profile being the last check (21:22:00) | 6 min 11 s | 175 | about 1,450 |
| from the fast-forward (21:03:18) to the batch verified, including reading the plan, the record, the 13 files and the blueprint, the set-up's downloads, and paper drafts of all 48 proofs made while Mathlib downloaded | 18 min 42 s | 58 | about 480 |

The record-keeping after verification (this section, the blueprint, the plans, the log, the three scans, the commit) is in neither window; its time is recorded with batch 2. The build loop here is 3 to 4 s per file, against 12 to 15 s on mains on the laptop (§1.1). **[AI Inference]:** batch 1 was the easiest of the three: Mathlib already has Banach's theorem with its estimates, and the rest is algebra. The rate is a property of this chat on this machine; the other chat's rate on the laptop is its own to measure.

**Findings:** none. Every batch-1 statement is proved as written.

**Notes for anyone writing Lean against this Mathlib** (`v4.34.1`): `ContinuousLinearMap.mul_apply`, `one_apply` and `sub_apply` are deprecated, in favour of the general `mul_apply_eq_comp`, `one_apply_eq_self` and `sub_apply`; from `h : b ≤ c`, `add_le_add_left h a` proves `b + a ≤ c + a` and `add_le_add_right h a` proves `a + b ≤ a + c`, the reverse of older Mathlib (checked with `#check`), so `add_le_add le_rfl h` is the unambiguous form; and plasTeX finds the blueprint's `\input` files through `kpsewhich`, so it needs a TeX installation or a stand-in for it.

### 5A.3 Batch 2: T3 (T3a, T3b, T3c, T3d and the counterexample) — checked

**What is checked:** all 21 declarations of `lean/decls-batch-2.txt`, the 14 theorems and the 7 definitions they rest on; no statement changed. **T3's status line** in [[formal-proofs-plan]] §6 was committed alone first (`bd7ddfe`), before this batch's own commit, as the demo chat's compiler rule requires. Added, all private: `window_eq` in `Schwarz.lean` (a window's solution is $B_iR_i(f-Au)+R_iu$, since $B_i(R_iAE_i)=I$); `sweep_sub_solution` in `SchwarzConvergence.lean` ($G_f(u)-u^\star=G_0(u-u^\star)$ when $Au^\star=f$); and `threeUnknowns`, the counterexample's sweep, built with `Matrix.toLin'` from the matrices of §5.1 example 1. **Imports added:** `Mathlib.Tactic.Abel` in `Schur.lean`, `DirichletNeumann.lean` and `Schwarz.lean`, whose imports did not reach the `abel` tactic; and `AtlasProofs.Certificate` in `SchwarzConvergence.lean`, since T3d applies T7, the one import between project files the instructions allow. The import chain keeps its depth: `SchwarzConvergence` was already one level above `Schwarz`.

| check | result |
|---|---|
| `lean_build.py` (21:37) | success, 14.6 s; 16 `sorry` warnings, **none in a batch-2 file** (all 16 are in batch 3) |
| `lean_build.py --clean` (21:39) | success, 16.9 s. Batch-2 files: `Schur` 1.9 s, `DirichletNeumann` 2.0 s, `Schwarz` 2.1 s, `SchwarzConvergence` 2.2 s, `SchwarzCounterexample` 7.9 s |
| `lean_axioms.py --names lean/decls-batch-2.txt` | **21 declarations, 0 failing**: `TwoPieces`, its two definitions and its two theorems rest on `propext` and `Quot.sound` only, the rest on those and `Classical.choice`; 4.1 s |
| `lean_build.py --profile` | 13 files, 44.3 s in all, **no step of ten seconds or more**. The counterexample file takes 9.2 s alone; with the profiler's threshold lowered to 0.5 s, its largest steps are three `simp` calls of about 1 s each, the matrix arithmetic |
| blueprint | `\leanok` in the 14 proofs of `consistency.tex`, each Lean proof following its paper proof. Builds without a warning; the dependency graph shows 31 nodes fully proved |

**The counterexample, in the kernel.** With the windows and window inverses of §5.1, $A$ is invertible (its inverse is $\tfrac12\begin{pmatrix}1&1&-1\\1&-1&1\\-1&1&1\end{pmatrix}$), the sweep with $f=0$ returns $u=(1,-1,-1)$ unchanged, and $Au=(0,0,-2)\ne0$. §5.1's finding is now machine-checked.

**The measured rate.** 14 theorems; **135 lines of proof text** (169 lines added in all, counting comments, docstrings and the 4 import lines), against the plan's 360.

| window | time | theorems per hour | proof lines per hour |
|---|---|---|---|
| first proof edit (21:31:08) to the batch verified, the profile being the last check (21:38:30) | 7 min 22 s | 114 | about 1,100 |

**Batch 1's record-keeping**, promised in §5A.2: from batch 1 verified (21:22:00) to its commit pushed (21:31:00), 9 minutes, for the record, the blueprint and its build, the two plans, the log, the three scans and the commit.

**Findings:** none. Every batch-2 statement is proved as written.

---

## 6. How to reproduce

```
python scripts/lean_build.py update            # once: toolchain, Mathlib, its cache
python scripts/lean_build.py                   # mirror lean/ and build, timed
python scripts/lean_build.py --clean           # re-check every project file
python scripts/lean_build.py --profile         # each file alone; steps over 10 s
python scripts/lean_axioms.py                  # what each theorem rests on
python scripts/lean_t3_counterexample.py       # section 5.1, through the workbench's driver
```

The blueprint's pages are made by `plastex -c plastex.cfg web.tex` in `lean/blueprint/src`, and served locally by the `lean-blueprint` entry of `.claude/launch.json` (port 8353). **Checked in a browser 2026-10-01:** a theorem page typesets its mathematics, the dependency graph draws, and the console is clean. Not checked: the print version, and the links from a statement to its Lean source, which go live with the public repository.

---

## 7. Batches 4 to 6: statements, record and findings

*Kept by the chat that states and proves batches 4 to 6 (tier 0: T25, T24, T26; tier 2: T8, T9, the SNI remark, T9′, T22, T2; the headline statements). Nothing here yet.* It will hold its own table, one row per theorem as in §3, the statements in plain words and LaTeX as in §4, and its findings.

---

## See Also

- [[formal-proofs-implementation-plan]] — which statements are machine-checked, by what route, in what order
- [[formal-proofs-plan]] — the inventory T1–T23 and the order of work
- [[master-error-bound]] · [[temporal-error-accumulation]] — T4's source
- [[defect-correction-learned-operator]] — T5's and T7's source, and W208
- [[symmetry-averaging-atlas-0.1]] — T10's source
- [[port-algebra-atlas-0.1]] — T11's connection rule
- [[atlas-and-standard-dd-theory]] — the classical theory T3 belongs to
- [[demo-finish-plan]] — §5, the compiler rule that cites T3
- [[gap-worklist]] — W353 (this work), W348 (R10), W208 (the certificate's constant)
