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
| **T1** | a contraction that differs from a map by $\delta$ at that map's fixed point has a unique fixed point within $\delta/(1-\rho)$ of it, reached at rate $\rho$ | `Atlas.perturbed_contraction` | `AtlasProofs/PerturbedContraction.lean` | drafted | — | — |
| T1, uniform | the same with the defect bounded everywhere | `Atlas.perturbed_contraction_uniform` | same | drafted | — | — |
| T1, sharp | $\delta/(1-\rho)$ is attained | `Atlas.perturbed_contraction_sharp` | same | drafted | — | — |
| **T7** | error $\le$ residual $/(1-L)$, when the step contracts *this* state toward the fixed point | `Atlas.certificate` | `AtlasProofs/Certificate.lean` | drafted | — | — |
| T7, Banach | the same for every state, under a global contraction | `Atlas.certificate_of_contraction` | same | drafted | — | — |
| **T7, W208** | for a linear step the least certifying constant is $\lVert(I-A)^{-1}\rVert$ | `Atlas.certificate_constant_is_opNorm` | same | drafted | — | — |
| T7, example | $\operatorname{diag}(0.6,\,0.95)$: the ratio is $2.5$ on one mode and $20$ on the other | `Atlas.certificate_two_mode_example` | same | drafted | — | — |
| **T4a** | the exact error recursion | `Atlas.error_recursion` | `AtlasProofs/MasterBound.lean` | drafted | — | — |
| **T4b** | the defect is $\tau+\sigma+\gamma$ | `Atlas.defect_split` | same | drafted | — | — |
| **T4c** | discrete Gronwall | `Atlas.accumulation` | same | drafted | — | — |
| **T4** | the master bound | `Atlas.master_bound` | same | drafted | — | — |
| T4, three terms | the master bound with $\tau$, $\sigma$, $\gamma$ named | `Atlas.master_bound_three_terms` | same | drafted | — | — |
| **T4, $L<1$** | bounded by $\delta/(1-L)$ for all time | `Atlas.master_bound_contractive` | same | drafted | — | — |
| **T4, $L=1$** | at most $N\delta$ | `Atlas.master_bound_nonexpansive` | same | drafted | — | — |
| **T4, $L>1$** | at most $\frac{L^N-1}{L-1}\delta$ | `Atlas.master_bound_expansive` | same | drafted | — | — |
| T4, attained | the scalar recursion attains the geometric sum | `Atlas.master_bound_attained` | same | drafted | — | — |
| T4e | $\lVert\lambda^\dagger-\lambda^\star\rVert\le\lVert\Lambda-\tilde\Lambda\rVert\,\lVert\lambda^\star\rVert/\beta$ | `Atlas.trace_perturbation` | `AtlasProofs/TransmissionBound.lean` | drafted | — | — |
| T4e | $\sigma\le\frac{C_\mu}{\beta}\lVert\Lambda-\tilde\Lambda\rVert\,\lVert\lambda^\star\rVert$ | `Atlas.transmission_bound` | same | drafted | — | — |
| **T3a** | the undivided block system and the Schur system have the same solutions | `Atlas.schur_iff` | `AtlasProofs/Schur.lean` | drafted | — | — |
| **T3b** | a fixed point of Dirichlet–Neumann solves the undivided system | `Atlas.TwoPieces.dn_fixedPoint_solves` | `AtlasProofs/DirichletNeumann.lean` | drafted | — | — |
| T3b, converse | the undivided solution is a fixed point | `Atlas.TwoPieces.dn_solution_isFixedPt` | same | drafted | — | — |
| T3b, face form | the same for the finite-volume form `styles.dirichlet_neumann` runs | `Atlas.FacePair.solves_of_fixedPoint` | same | drafted | — | — |
| T3c | the Schwarz sweep is $u\mapsto u+M(f-Au)$ | `Atlas.Schwarz.sweep_eq` | `AtlasProofs/Schwarz.lean` | drafted | — | — |
| **T3c** | the undivided solution is a fixed point of the sweep | `Atlas.Schwarz.solution_isFixedPt` | same | drafted | — | — |
| T3c | $u$ is a fixed point iff $M(f-Au)=0$ | `Atlas.Schwarz.isFixedPt_iff` | same | drafted | — | — |
| **T3c** | if every window agrees with $u$, then $Au=f$ | `Atlas.Schwarz.solves_of_windows_agree` | same | drafted | — | — |
| **T3c** | if $M$ is one-to-one, every fixed point solves $Au=f$ | `Atlas.Schwarz.fixedPt_solves_of_precond` | same | drafted | — | — |
| T3c | the same hypothesis read off the sweep | `Atlas.Schwarz.fixedPt_solves` | same | drafted | — | — |
| **T3c** | a convergent sweep has one fixed point, the solution | `Atlas.Schwarz.fixedPt_eq_solution` | `AtlasProofs/SchwarzConvergence.lean` | drafted | — | — |
| **T3c** | a convergent sweep converges to the solution | `Atlas.Schwarz.tendsto_solution` | same | drafted | — | — |
| **T3d** | error $\le$ update $/(1-\rho)$ | `Atlas.Schwarz.error_le_update` | same | drafted | — | — |
| **T3c, finding** | a Schwarz sweep with a fixed point that is not the solution | `Atlas.Schwarz.exists_spurious_fixedPt` | `AtlasProofs/SchwarzCounterexample.lean` | drafted | — | — |
| **T5, Theorem 1** | any limit of defect correction is a fixed point of the classical map | `Atlas.DefectCorrection.limit_isFixedPt` | `AtlasProofs/DefectCorrection.lean` | drafted | — | — |
| T5, Theorem 1, inexact | the same when each step is solved only approximately, the leftover tending to zero | `Atlas.DefectCorrection.limit_isFixedPt_of_inexact` | same | drafted | — | — |
| **T5, Corollary 1** | a constant cheap map makes the iteration the classical march | `Atlas.DefectCorrection.step_of_const` | same | drafted | — | — |
| T5 | the inner march's fixed points are the successors | `Atlas.DefectCorrection.isStep_iff` | same | drafted | — | — |
| T5, Corollary 1 | with a constant cheap map the inner march returns $\Phi(w)$ at every stage | `Atlas.DefectCorrection.innerMarch_of_const` | same | drafted | — | — |
| T5, Corollary 2 | $J_{\Psi_\alpha}=\alpha I+(1-\alpha)J_\Psi$, as a derivative | `Atlas.DefectCorrection.shrink_hasFDerivAt` | `AtlasProofs/DefectCorrectionShrink.lean` | drafted | — | — |
| T5, Corollary 2 | an eigenvalue $\mu$ becomes $\alpha+(1-\alpha)\mu$ | `Atlas.DefectCorrection.shrink_hasEigenvalue` | same | drafted | — | — |
| **T5, Corollary 2** | and those are all the eigenvalues, for $\alpha\ne1$ | `Atlas.DefectCorrection.shrink_hasEigenvalue_iff` | same | drafted | — | — |
| T5, Corollary 2 | the real part is at least $\alpha$ | `Atlas.DefectCorrection.shrink_re` | same | drafted | — | — |
| **T10** | the group average of any map is exactly equivariant | `Atlas.average_equivariant` | `AtlasProofs/SymmetryAveraging.lean` | drafted | — | — |
| T10 | averaging returns an equivariant map unchanged | `Atlas.average_of_equivariant` | same | drafted | — | — |
| T10, example | averaging over $\{\mathrm{id},M_x,M_y\}$, not a group, is not equivariant | `Atlas.average_over_non_group_not_equivariant` | same | drafted | — | — |
| T11 | the connection rule conserves power | `Atlas.port_rule_power` | `AtlasProofs/Passivity.lean` | drafted | — | — |
| T11 | the junction: connected port powers sum to zero | `Atlas.junction_power_eq_zero` | same | drafted | — | — |
| **T11** | passive agents joined by the rule make a passive system | `Atlas.interconnection_passive` | same | drafted | — | — |
| **T11, error form** | incrementally passive agents make a non-expansive step | `Atlas.interconnection_nonexpansive` | same | drafted | — | — |

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

*Kept by the chat that proves batches 1 to 3. Nothing here yet.* It will hold: the build that checked each batch (time, power state, machine), the axiom check's result, the measured authoring rate that replaces the estimates of [[formal-proofs-implementation-plan]] §6.2, and any statement that had to change.

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

*Kept by the chat that states and proves batches 4 to 6 (tier 0: T25, T24, T26; tier 2: T8, T9, the SNI remark, T9′, T22, T2; the headline statements).* Status words as in §0. Builds of this chat are recorded in `out/lean/builds-batches-4-6.jsonl`, its axiom checks in `out/lean/axioms-batches-4-6.json`, and its declarations are listed in `lean/decls-batches-4-6.txt`.

**Where this stands (2026-10-01).** The owner approved the statements of batches 4 and 5 on 2026-10-01. **Batch 4 (tier 0) is checked**: 28 theorems in five Lean files, no `sorry`, each resting on Lean's three standard axioms only (§7.1 to §7.3). **Batch 5 (tier 2) is proved**: 14 theorems in four files with no `sorry` of their own; 12 are checked, and 2 (`contraction_contract`, `two_level_contract`) quote T1 and wait on the other chat's proof of it (§7.4 to §7.6). **Batch 6 is proved** (approved by the owner 2026-10-01): 15 theorems in four files with no `sorry` of their own; 6 are checked, and 9 quote theorems of batches 1 to 3 and wait on the other chat (§7.7 to §7.9). **All three batches: 57 theorems, 46 checked, 11 waiting on the merge, listed by name in §7.10.**

### 7.1 The record, batch 4

| # | statement, in one line | Lean name | file | status | build time | axioms |
|---|---|---|---|---|---|---|
| **T25 (i)** | the Gram port matrix is symmetric positive semidefinite, for any fields | `Atlas.Piece.gram_posSemidef` | `AtlasProofs/GramPort.lean` | checked | 23 s | `propext`, `Classical.choice`, `Quot.sound` |
| T25 (i) | positive definite when the energy is and the port patterns are independent | `Atlas.Piece.gram_posDef` | same | checked | 23 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T25 (ii)** | $\tilde\Lambda-\Lambda$ is the Gram matrix of the field errors in the $A_I$ inner product | `Atlas.Piece.gram_sub_gram` | same | checked | 23 s | `propext`, `Classical.choice`, `Quot.sound` |
| T25 (ii) | the same as a quadratic form: $x^\top(\tilde\Lambda-\Lambda)x=\lVert Ex\rVert_{A_I}^2$ | `Atlas.Piece.quadratic_gram_sub` | same | checked | 23 s | `propext`, `Classical.choice`, `Quot.sound` |
| T25 (ii) | $\tilde\Lambda-\Lambda\succeq0$ | `Atlas.Piece.gram_sub_gram_posSemidef` | same | checked | 23 s | `propext`, `Classical.choice`, `Quot.sound` |
| T25 (ii) | $\operatorname{tr}\tilde\Lambda-\operatorname{tr}\Lambda=\sum_k\lVert E_k\rVert_{A_I}^2$ | `Atlas.Piece.trace_gram_sub` | same | checked | 23 s | `propext`, `Classical.choice`, `Quot.sound` |
| T25 (ii) | $x^\top(\tilde\Lambda-\Lambda)x\le\alpha\,\varepsilon^2\lvert x\rvert^2$: quadratic in the field error | `Atlas.Piece.quadratic_gram_sub_le` | same | checked | 23 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T25 (iii)** | $\tilde{\mathsf S}-\mathsf S\succeq0$ | `Atlas.assemble_gram_sub_posSemidef` | same | checked | 23 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T25 (iii)** | every lower bound $\beta\lvert c\rvert^2\le c^\top\mathsf Sc$ holds for $\tilde{\mathsf S}$ | `Atlas.assemble_gram_coercive` | same | checked | 23 s | `propext`, `Classical.choice`, `Quot.sound` |
| T25 (iii) | $\mathsf S$ positive definite implies $\tilde{\mathsf S}$ positive definite | `Atlas.assemble_gram_posDef` | same | checked | 23 s | `propext`, `Classical.choice`, `Quot.sound` |
| T25 (iii) | then $\tilde{\mathsf S}c=\chi$ has exactly one solution | `Atlas.assemble_gram_existsUnique` | same | checked | 23 s | `propext`, `Classical.choice`, `Quot.sound` |
| T25, lemma | a minimiser over $u+U_0$ is stationary | `Atlas.QuadEnergy.stationary_of_isMin` | `AtlasProofs/EnergyOptimality.lean` | checked | 21 s | `propext`, `Classical.choice`, `Quot.sound` |
| T25, lemma | a stationary point is a minimiser | `Atlas.QuadEnergy.isMin_of_stationary` | same | checked | 21 s | `propext`, `Classical.choice`, `Quot.sound` |
| T25, lemma | $\mathcal E(w)=\mathcal E(u)+\tfrac12\lVert w-u\rVert_m^2$ | `Atlas.QuadEnergy.energy_eq_add` | same | checked | 21 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T25, the abstract lemma** | least energy over a subset is closest to the minimiser in the energy size | `Atlas.QuadEnergy.closest_iff_isMin` | same | checked | 21 s | `propext`, `Classical.choice`, `Quot.sound` |
| T25, lemma | the Galerkin system characterises the least-energy trial field | `Atlas.QuadEnergy.galerkin_iff_isMin` | same | checked | 21 s | `propext`, `Classical.choice`, `Quot.sound` |
| T25, lemma | the energy size obeys the triangle inequality | `Atlas.QuadEnergy.norm_add_le` | same | checked | 21 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T25** | the host's system $\tilde{\mathsf S}c=\tilde\chi$ is the Galerkin system of the learned trial set | `Atlas.Superelement.hostSystem_iff_isMin` | `AtlasProofs/Superelement.lean` | checked | 16 s | `propext`, `Classical.choice`, `Quot.sound` |
| T25 | the trial set lies in the admissible set | `Atlas.Superelement.field_admissible` | same | checked | 16 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T25, energy optimality** | the rebuilt field is the energy-closest trial field to the undivided solution | `Atlas.Superelement.energy_optimal` | same | checked | 16 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T25, the bound** | error $\le$ reference error $+$ energy of the field errors | `Atlas.Superelement.energy_error_bound` | same | checked | 16 s | `propext`, `Classical.choice`, `Quot.sound` |
| T25 | the undivided solution is unique when the energy size is a norm on $U_0$ | `Atlas.Superelement.solution_unique` | same | checked | 16 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T25** | $\tilde{\mathsf S}$ is positive definite for every network output | `Atlas.Superelement.hostMatrix_posDef` | same | checked | 16 s | `propext`, `Classical.choice`, `Quot.sound` |
| T25 | the host's system has exactly one solution | `Atlas.Superelement.hostSystem_existsUnique` | same | checked | 16 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T24** | $\lVert\tilde c-c\rVert\le(\lVert\mathsf S-\tilde{\mathsf S}\rVert\lVert c\rVert+\lVert\chi-\tilde\chi\rVert)/\beta$ | `Atlas.interface_perturbation` | `AtlasProofs/InterfacePerturbation.lean` | checked | 22 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T24** | the same with $\beta$ the exact problem's coercivity constant | `Atlas.interface_perturbation_of_dominates` | same | checked | 22 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T26** | $\mathcal L_{\text{en}}=\mathcal L_{\text{sup}}+\text{a constant}$ | `Atlas.Piece.labelFree_eq_supervised_add_const` | `AtlasProofs/LabelFree.lean` | checked | 16 s | `propext`, `Classical.choice`, `Quot.sound` |
| T26 | between two outputs the two objectives change by the same amount | `Atlas.Piece.labelFree_sub_eq_supervised_sub` | same | checked | 16 s | `propext`, `Classical.choice`, `Quot.sound` |

**The build that checked batch 4 (2026-10-01, 18:43, on battery).** `python scripts/lean_build.py --clean`: every project file re-checked, **56.0 s**, 23 files, no error; the five batch-4 files took 16 to 23 s each, and none gave a warning. The 62 `sorry` warnings of that build are all in other files: 48 in the thirteen files of batches 1 to 3, which the other chat is proving, and 14 in batch 5, not yet proved at that moment. **The axiom check** on this chat's 68 names (17 s): the 28 batch-4 theorems and the 26 definitions depend on `propext`, `Classical.choice` and `Quot.sound` only. **No batch-4 theorem uses a theorem of batches 1 to 3**, so none waits on the other chat. No `sorry`, no new axiom and no `maxHeartbeats` in the five files.

**The profile (on battery).** `python scripts/lean_build.py --profile`: 22 files, 549.7 s one after another. **No tactic or elaboration step reached ten seconds.** The script's count was 10 all the same, and it returned a failure: every one of the ten lines is `import took 10.1 s` to `12.5 s`, the time to load Mathlib's compiled files at the battery's clock speed, and five of the ten are in files of batches 1 to 3 that hold statements only. So the rule "a tactic call over ten seconds is replaced" is met, and the script's own verdict is not: it needs a run in which loading is faster to read zero. **The second run, half an hour later and still on battery, read zero** (§7.4): the same files loaded in under ten seconds each. So the count depends on the machine's state at the moment, and what it is meant to catch, a slow tactic, has not occurred.

**What proving added, and what it changed.** 22 helper lemmas and one helper definition (`Atlas.Superelement.modeField`); the five files grew from 633 lines to 1,080. A script (`stmt_check.py`, this chat's scratch) compared every approved declaration with the proved file: all 56 (theorems, definitions and structures) are word for word what the owner approved. Three were then **generalised** by one line each (`omit [Fintype …] in`), because Lean's linter reported an unused hypothesis: `gram_sub_gram` needs no finiteness of the index set, `field_admissible` none of the set of pieces, `solution_unique` none of the port and coordinate sets. Each still implies its approved form. One route changed: the tier-0 Galerkin statement is proved from a new lemma for any affine parametrisation (`Atlas.QuadEnergy.isMin_iff_of_affine`), not from the coordinate form `galerkin_iff_isMin`, which is proved and stands on its own.

**Measured authoring, batch 4.** From the build log's own clock: the statements' first build was at 17:08 and their last at 17:14; the first proof build at 18:19 and the clean build at 18:43. About 450 lines of proof and helper text in the 25 minutes between the last two, against the plan's estimate of 4.5 to 8 hours for the batch. **[AI Inference]:** the estimates of [[formal-proofs-implementation-plan]] §6.2 were too high by roughly a factor of ten for linear-algebra statements of this kind; one batch is thin evidence, and the from-scratch analysis proof of batch 6 (T9′) is the one that may not follow it.

**Builds of the statements (before the proofs, on mains).** The five new files, with every proof a `sorry`: 101.2 s for the first build (65 to 67 s per first-level file, three at once), 42.5 s for the rebuild of two changed files and the two that import them, 2.9 s with nothing changed. The axiom check on the 53 names: 18.0 s; it knows all 53, passes the 25 definitions and sees a `sorry` in each of the 28 theorems, which is its positive control.

**Proved before the owner's OK, and why.** Three one-line facts about the definitions themselves: the energy form's formula (`Atlas.Piece.M_apply`), its symmetry (`M_symm`) and its sign (`M_self_nonneg`). The definition of the global energy `Atlas.Superelement.quad` cannot be written without the last two. They are not claims of the proposal.

**A check made before asking for the OK.** `b4_numeric_check.py` (a scratch script of this chat, not in the repository) builds random pieces and evaluates the identities and inequalities below in floating point: T25 (ii) and its trace to $6\times10^{-14}$, T26 to $6\times10^{-14}$, and no violation of T25 (i), (iii), the Galerkin statement, the optimality or the bound in 200 random trials each. That is a guard against a wrong sign in a statement. It is not a proof.

### 7.2 The statements, batch 4 (tier 0)

**The setting.** A *piece* of the domain has cell values $u\in U$ and boundary-face values $\lambda\in F$. Its discrete energy is

$$\begin{pmatrix}u\\ \lambda\end{pmatrix}^{\!\top}\!M\begin{pmatrix}u\\ \lambda\end{pmatrix}=u^\top A_Iu-2\,u^\top B\lambda+\lambda^\top D\lambda\ \ge0,\qquad M=\begin{pmatrix}A_I&-B\\-B^\top&D\end{pmatrix},$$

with $A_I$ and $D$ symmetric. In Lean the three blocks are bilinear forms on real vector spaces of any dimension (`Atlas.Piece`), so nothing depends on a basis. A *port mode* is a pattern $q_k\in F$ of boundary values. Its *exact constraint mode* is the interior field $H_k$ with $A_IH_k=Bq_k$. For any interior fields $h_k$, the *Gram port matrix* is

$$\Lambda[h]_{kl}=\begin{pmatrix}h_k\\ q_k\end{pmatrix}^{\!\top}\!M\begin{pmatrix}h_l\\ q_l\end{pmatrix}.$$

The exact port matrix is $\Lambda=\Lambda[H]$. The host computes $\tilde\Lambda=\Lambda[\hat H]$ from the fields $\hat H_k$ a network returned.

**T25 (i) to (iii), the Gram port matrix.** *In plain words:* the host never asks the network for a port matrix. It asks for fields and measures their energy. Then, whatever the network returned, the port matrix is symmetric and never indefinite; it is stiffer than the exact one by exactly the energy of the field errors; and after assembly the learned interface matrix is at least as "stiff" as the exact one, so it is solvable whenever the classical one is.

For **any** fields $\hat H_k$, with $E_k=\hat H_k-H_k$:

$$\text{(i)}\quad\tilde\Lambda=\tilde\Lambda^\top,\qquad x^\top\tilde\Lambda x\ \ge0\ \text{ for all }x;$$

$$\text{(ii)}\quad\tilde\Lambda_{kl}-\Lambda_{kl}=E_k^\top A_IE_l,\qquad\text{so}\qquad x^\top(\tilde\Lambda-\Lambda)x=\Bigl\lVert\sum_kx_kE_k\Bigr\rVert_{A_I}^2\ge0,\qquad\operatorname{tr}\tilde\Lambda-\operatorname{tr}\Lambda=\sum_k\lVert E_k\rVert_{A_I}^2;$$

$$\text{(iii)}\quad\tilde{\mathsf S}=\sum_iR_i^\top\tilde\Lambda_iR_i,\quad\mathsf S=\sum_iR_i^\top\Lambda_iR_i:\qquad c^\top\tilde{\mathsf S}c\ \ge\ c^\top\mathsf Sc\ \text{ for all }c .$$

From (iii): if $\beta\lvert c\rvert^2\le c^\top\mathsf Sc$ for all $c$, then $\beta\lvert c\rvert^2\le c^\top\tilde{\mathsf S}c$ for all $c$ (the smallest eigenvalue does not decrease); if $\mathsf S$ is positive definite, so is $\tilde{\mathsf S}$, and $\tilde{\mathsf S}c=\chi$ has exactly one solution for every $\chi$. From (ii): if $u^\top A_Iu\le\alpha\,s(u)^2$ for a measure of size $s$, and $s\bigl(\sum_kx_kE_k\bigr)^2\le\varepsilon^2\lvert x\rvert^2$, then $x^\top(\tilde\Lambda-\Lambda)x\le\alpha\varepsilon^2\lvert x\rvert^2$, which with Euclidean norms is the document's $\lVert\tilde\Lambda-\Lambda\rVert_2\le\lVert A_I\rVert_2\lVert E\rVert_2^2$: a 1% field error gives a port-matrix error of order $10^{-4}$. And if $M$ is positive definite and the $q_k$ are linearly independent, $\tilde\Lambda$ is positive definite.

*Why (ii) holds:* $(\hat H_k,q_k)=(H_k,q_k)+(E_k,0)$, and the cross terms are $E_k^\top(A_IH_l-Bq_l)=0$ by the definition of $H_l$.

**T25, energy optimality: the abstract lemma.** *In plain words:* a symmetric linear problem is "find the admissible field of least energy". If you minimise the same energy over a smaller set of fields, what you get is the field of that set closest to the true solution, with distance measured by the energy itself. No constant appears.

Let $m$ be a symmetric bilinear form with $m(w,w)\ge0$ on a real vector space, $\ell$ linear, $\mathcal E(w)=\tfrac12m(w,w)-\ell(w)$ and $\lVert w\rVert_m=\sqrt{m(w,w)}$. Let $u$ have the least energy in the affine set $u+U_0$. Then $m(u,v)=\ell(v)$ for all $v\in U_0$ (and conversely), and for every $w\in u+U_0$

$$\mathcal E(w)=\mathcal E(u)+\tfrac12\lVert w-u\rVert_m^2 .$$

So for **any** set $W\subseteq u+U_0$ and $\tilde u\in W$:

$$\mathcal E(\tilde u)\le\mathcal E(w)\ \ \forall w\in W\qquad\Longleftrightarrow\qquad\lVert u-\tilde u\rVert_m\le\lVert u-w\rVert_m\ \ \forall w\in W .$$

For a trial set $w(c)=w_0+\sum_ac_at_a$ with finitely many coordinates, $w(c)$ has the least energy among the $w(c')$ exactly when $m(w(c),t_a)=\ell(t_a)$ for every $a$: the Galerkin system. The energy size obeys the triangle inequality. $m$ need not be definite, and no dimension need be finite.

**T25, energy optimality: the tier-0 solve.** *In plain words:* the undivided discrete solution $u$ is the admissible field of least total energy. The host, knowing only what the network returned, forms a matrix and a right-hand side, solves once, and rebuilds a field on every piece. That linear system says exactly "the rebuilt field has the least energy among everything that can be rebuilt from the network's modes". By the lemma, the rebuilt field is then the best approximation of $u$ those modes can give. The coupling adds no error of its own.

A global field gives every piece $i$ a pair $w_i=(w_{I,i},w_{F,i})$. It is *admissible* when its face values differ from the prescribed ones $g$ by a family in $\Gamma_0$, the families on which neighbouring pieces agree and which vanish on the prescribed boundary. $U_0$ is the space of fields with face values in $\Gamma_0$. With

$$\mathcal E(w)=\sum_i\Bigl(\tfrac12w_i^\top M_iw_i-f_i^\top w_{I,i}\Bigr),\qquad\lVert w\rVert_M^2=\sum_iw_i^\top M_iw_i,$$

the host forms, from fields $\hat H_{i,k}$ and $\hat u_{p,i}$ (with $\tilde V_i$ the pairs $(\hat H_{i,k},q_{i,k})$),

$$\tilde\Lambda_i=\tilde V_i^\top M_i\tilde V_i,\qquad\tilde b_i=\hat H_i^\top f_i-\tilde V_i^\top M_i\begin{pmatrix}\hat u_{p,i}\\0\end{pmatrix},\qquad\tilde{\mathsf S}=\sum_iR_i^\top\tilde\Lambda_iR_i,\qquad\tilde\chi=\sum_iR_i^\top\bigl(\tilde b_i-\tilde\Lambda_ir_i\bigr),$$

where $r_i$ holds piece $i$'s port coordinates fixed by the boundary data and $R_ic$ those read off the free coordinates $c$. The rebuilt field is

$$w(c)_i=\begin{pmatrix}\hat u_{p,i}\\0\end{pmatrix}+\tilde V_i\,(r_i+R_ic).$$

The document's three assumptions are named hypotheses:

1. **conforming shared sides** (`Conforming`): for every $c$, the face values $\bigl(Q_iR_ic\bigr)_i$ lie in $\Gamma_0$;
2. **boundary data in the port span** (`BoundaryDataInPortSpan`): $\bigl(Q_ir_i\bigr)_i-g\in\Gamma_0$;
3. **the energy size is a norm on $U_0$** (`EnergyNormIsNorm`): $w\in U_0$ and $\lVert w\rVert_M=0$ imply $w=0$.

Then, for **any** $\hat H$, $\hat u_p$:

- *(Galerkin; no assumption)* $\tilde{\mathsf S}c=\tilde\chi\iff\mathcal E(w(c))\le\mathcal E(w(c'))$ for all $c'$.
- *(energy optimality; assumptions 1 and 2)* every $w(c)$ is admissible, and if $u$ is the undivided solution and $\tilde{\mathsf S}c=\tilde\chi$,
$$\lVert u-w(c)\rVert_M\le\lVert u-w(c')\rVert_M\qquad\text{for all }c' .$$
- *(the bound; assumptions 1 and 2)* for any reference fields $H$, $u_p$ and any coordinates $c^\star$, with $w^{\text{ref}}$ rebuilt from the reference fields at $c^\star$,
$$\lVert u-w(c)\rVert_M\le\lVert u-w^{\text{ref}}\rVert_M+\Bigl(\sum_i\bigl\lVert(\hat u_{p,i}-u_{p,i})+E_i\,(r_i+R_ic^\star)\bigr\rVert_{A_{I,i}}^2\Bigr)^{1/2}.$$
With the exact modes and the classical superelement's coordinates, $w^{\text{ref}}=u_m$: truncation error plus the network's field errors.
- *(one answer; assumptions 1 and 3)* if moreover each piece's patterns $q_{i,k}$ are linearly independent and $R_ic=0$ for all $i$ only for $c=0$, then $\tilde{\mathsf S}$ is positive definite and $\tilde{\mathsf S}c=\tilde\chi$ has exactly one solution. Under assumption 3 the undivided solution is unique.

**Not covered,** as the document says: non-matching parametrisations of a shared side (the mortar case), which fail assumption 1.

**T24, perturbation of the interface solve.** *In plain words:* the exact interface system and the learned one differ in their matrix and in their load. Their answers differ by at most those two errors, divided by the smallest stretch $\beta$ of the learned matrix. And for the Gram construction $\beta$ can be taken from the classical problem, because of T25 (iii).

If $\mathsf Sc=\chi$, $\tilde{\mathsf S}\tilde c=\tilde\chi$ and $\beta\lVert x\rVert\le\lVert\tilde{\mathsf S}x\rVert$ for all $x$ with $\beta>0$,

$$\lVert\tilde c-c\rVert\le\frac{\lVert\mathsf S-\tilde{\mathsf S}\rVert\,\lVert c\rVert+\lVert\chi-\tilde\chi\rVert}{\beta}.$$

The same bound holds if instead $\beta\lVert x\rVert^2\le\langle\mathsf Sx,x\rangle\le\langle\tilde{\mathsf S}x,x\rangle$ for all $x$: the constant of the **exact** matrix, and domination, which is T25 (iii).

**T26, the label-free objective.** *In plain words:* training against exact fields needs a classical solve per example. Minimising the energy of the network's own fields needs none, and it is the same training problem: the two objectives differ by a number that the network cannot influence.

With $A_IH_k=Bq_k$ and $A_Iu_p=f$, for any $\hat H$, $\hat u_p$:

$$\underbrace{\operatorname{tr}\tilde\Lambda+\hat u_p^\top A_I\hat u_p-2f^\top\hat u_p}_{\mathcal L_{\text{en}}}=\underbrace{\sum_k\lVert\hat H_k-H_k\rVert_{A_I}^2+\lVert\hat u_p-u_p\rVert_{A_I}^2}_{\mathcal L_{\text{sup}}}+\underbrace{\operatorname{tr}\Lambda-u_p^\top A_Iu_p}_{\text{no network output in it}} .$$

So between any two outputs the two objectives change by the same amount: the same gradients and the same minimisers. The identity is exact over the reals; in floating point the two differ, which the architecture document registers as comparison C2.

### 7.3 Findings, batch 4

None of these changes a claim of the architecture document. Each is a place where the formal statement is more precise than the prose.

1. **Energy optimality uses two of the three assumptions.** The identity $\lVert u-\tilde u\rVert_M=\min_{\tilde W}\lVert u-w\rVert_M$ needs conforming sides and boundary data in the port span. It does **not** need the energy size to be a norm on $U_0$. That assumption is what makes the undivided solution and the interface solution *unique*, and it is used for exactly that (`solution_unique`, `hostMatrix_posDef`).
2. **The error bound is more general than the document's.** It holds for any reference fields and any coordinates $c^\star$: neither the exactness of $H$ and $u_p$ nor $\mathsf Sc^\star=\chi$ is used. The document's inequality is the case that gives the first term its name, truncation error.
3. **Two facts the document's set-up implies are hypotheses of the solvability statement:** each piece's port patterns are linearly independent (the document's "$Q_i$ has orthonormal columns"), and every free coordinate is read by at least one piece ($R_ic=0$ for all $i$ only for $c=0$; the document's "$R_i$ selects the coordinates of piece $i$"). The document's own route to solvability, "$\mathsf S$ positive definite implies $\tilde{\mathsf S}$ positive definite", is stated as it stands (`assemble_gram_posDef`) and needs neither.
4. **The fixed coordinates are written out.** The document says "with the contribution of the fixed coordinates moved to the right-hand side". The Lean statement fixes what that is: $\tilde\chi=\sum_iR_i^\top(\tilde b_i-\tilde\Lambda_ir_i)$.
5. **T25 (i) and (ii) need neither $A_I$ positive definite nor orthonormal port patterns.** The exact modes are given, with $A_IH_k=Bq_k$ as the hypothesis. $A_I\succ0$ is what makes them exist.
6. **T24 needs no symmetry.** $\beta$ is any lower bound $\beta\lVert x\rVert\le\lVert\tilde{\mathsf S}x\rVert$; for a symmetric positive definite matrix the best one is $\lambda_{\min}(\tilde{\mathsf S})$, the document's constant.
7. **The existence of the undivided solution is a hypothesis** (`IsSolution`), as in the document ("we assume that the problem is well posed"). **[AI Inference]:** under assumption 3 in finite dimensions it exists by the finite-dimensional Lax–Milgram argument; that is not stated in Lean and nothing here depends on it.

### 7.4 The record, batch 5

| # | statement, in one line | Lean name | file | status | build time | axioms |
|---|---|---|---|---|---|---|
| T8 | $\lVert e-Zf\rVert^2=\lVert e+Zf\rVert^2-4Z\langle f,e\rangle$ | `Atlas.wave_identity` | `AtlasProofs/Cayley.lean` | checked | 14 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T8** | the bound, for any effort $e$ and flow $f$ | `Atlas.wave_bound` | same | checked | 14 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T8** | the Cayley bound for a linear $\Lambda$ | `Atlas.cayley_bound` | same | checked | 14 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T8** | in finite dimensions $I+Z\Lambda$ is invertible and $\lVert\mathcal S\rVert^2\le1-\frac{4Zc}{(1+ZM)^2}$ | `Atlas.cayley_transform` | same | checked | 14 s | `propext`, `Classical.choice`, `Quot.sound` |
| T8, $c=0$ | a passive $\Lambda$ has $\lVert\mathcal S\rVert\le1$ | `Atlas.cayley_nonexpansive` | same | checked | 14 s | `propext`, `Classical.choice`, `Quot.sound` |
| T8, matrices | a positive semidefinite matrix has a non-expansive scattering matrix | `Atlas.cayley_matrix_nonexpansive` | same | checked | 14 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T8 at tier 0** | the Gram port matrix of any network output has a non-expansive scattering matrix, for every $Z>0$ | `Atlas.Piece.gram_cayley_nonexpansive` | same | checked | 14 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T9a** | $\rho$-Lipschitz experts and a non-expansive exchange give a $\rho$-Lipschitz sweep in the $\ell^2$ product | `Atlas.waveSweep_lipschitz` | `AtlasProofs/ContractionContract.lean` | checked | 14 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T9a** | the contract: one answer, rate $\rho$, within $\bigl(\sum_i\delta_i^2\bigr)^{1/2}/(1-\rho)$ of the classical one | `Atlas.contraction_contract` | same | **proved here; waits on T1** | 14 s | the three standard ones, and `sorryAx` through `Atlas.perturbed_contraction` only |
| T9b | a fixed point of the two-level iteration is a fixed point of the sweep | `Atlas.twoLevel_fixedPt_isFixedPt` | `AtlasProofs/TwoLevel.lean` | checked | 13 s | `propext`, `Classical.choice`, `Quot.sound` |
| T9b | the converse | `Atlas.fixedPt_isFixedPt_twoLevel` | same | checked | 13 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T9b** | the contract, from a contraction of the composite two-level map | `Atlas.two_level_contract` | same | **proved here; waits on T1** | 13 s | the three standard ones, and `sorryAx` through `Atlas.perturbed_contraction` only |
| **T9b, finding** | a sweep that contracts the fine space, with an exact coarse solve, whose two-level iterates diverge | `Atlas.two_level_fine_contraction_diverges` | same | checked | 13 s | `propext`, `Classical.choice`, `Quot.sound` |
| **the SNI remark** | a contraction, a map within $\tfrac12$ of it everywhere, and a two-cycle of that map | `Atlas.sni_counterexample` | `AtlasProofs/SniCounterexample.lean` | checked | 22 s | `propext`, `Classical.choice`, `Quot.sound` |

**The build that checked batch 5 (2026-10-01, 19:00, on battery).** `python scripts/lean_build.py --clean`: **51.6 s**, 23 files, no error; the four batch-5 files took 13 to 22 s each and gave no warning. The 48 `sorry` warnings left are all in the thirteen files of batches 1 to 3. **The axiom check** on this chat's 68 names (17 s): 66 depend on the three standard axioms only.

**The profile (on battery, after the batch-5 proofs).** `python scripts/lean_build.py --profile`: 22 files, 307.2 s one after another (9 to 17 s each), **0 steps at or over ten seconds**, exit 0. The run before it, for batch 4, took 549.7 s on the same files and listed ten `import` lines (§7.1): both runs report battery power, so the difference is the machine's load or clock speed at the time, which this chat did not measure.

**Two theorems wait on the other chat.** `Atlas.contraction_contract` (T9a) and `Atlas.two_level_contract` (T9b) quote T1, `Atlas.perturbed_contraction`, whose proof is still a `sorry` in this tree. Their own proofs are complete: `ContractionContract.lean` and `TwoLevel.lean` give no `sorry` warning in the build log, and the `sorryAx` the kernel reports for these two arrives through that one name. They become *checked* when the branches are merged and T1 is proved. Nothing else in batch 5 uses a theorem of batches 1 to 3.

**What proving added, and what it changed.** Two helper lemmas in `Cayley.lean` (`cayley_inverse`, `cayley_inverse_apply`); the four files grew from 328 lines to 586. All 15 approved declarations are word for word what the owner approved. No `sorry`, no new axiom, no `maxHeartbeats`.

**The first build of the proofs failed, and how.** Eleven errors, all in `Cayley.lean` and none about the mathematics: two `linarith` hint lists broken across lines at a column that ends the tactic block; the tactic `push_neg`, deprecated in this Mathlib; and a missing import for the instance that says the reals are a star-ordered ring, which "the identity matrix is positive definite" needs. `TwoLevel.lean`, `SniCounterexample.lean` and `ContractionContract.lean` compiled as first written.

**Builds of the statements (before the proofs, on mains).** The four new files with `sorry` proofs: 36.1 s (12 to 17 s per file, four at once, then the root). The axiom check on this chat's 68 names: 13.3 s; it knew all 68, passed the 26 definitions and saw a `sorry` in each of the 42 theorems.

### 7.5 The statements, batch 5 (tier 2)

**T8, wave variables and the Cayley transform.** *In plain words:* on a side, the boundary values $e$ and the flux $f$ into the piece are combined into an incoming wave $e+Zf$ and an outgoing wave $e-Zf$. The outgoing wave's squared size is the incoming one's minus $4Z$ times the power the piece absorbs. So a piece that never generates power does not amplify waves, and one that absorbs a definite share shrinks them by a definite factor.

In any real inner-product space, for all $e$, $f$ and $Z$:

$$\lVert e-Zf\rVert^2=\lVert e+Zf\rVert^2-4Z\,\langle f,e\rangle .$$

If $Z>0$, $c\ge0$, $\langle f,e\rangle\ge c\lVert e\rVert^2$ and $\lVert f\rVert\le M\lVert e\rVert$:

$$\lVert e-Zf\rVert^2\le\Bigl(1-\frac{4Zc}{(1+ZM)^2}\Bigr)\lVert e+Zf\rVert^2 .$$

This is stated for **any** pair $e,f$, so it also covers the difference of two states of a nonlinear piece. For a bounded linear $\Lambda$ with $\langle\Lambda e,e\rangle\ge c\lVert e\rVert^2$ and $\lVert\Lambda\rVert\le M$ it holds with $f=\Lambda e$. In finite dimensions (and not the zero space), $I+Z\Lambda$ is invertible and

$$\mathcal S=(I-Z\Lambda)(I+Z\Lambda)^{-1},\qquad\lVert\mathcal S\rVert^2\le1-\frac{4Zc}{(1+ZM)^2}.$$

With only $\langle\Lambda e,e\rangle\ge0$: $\lVert\mathcal S\rVert\le1$. **At tier 0:** for the Gram port matrix $\tilde\Lambda$ of *any* network output and any $Z>0$, $1+Z\tilde\Lambda$ is invertible and $\lvert\mathsf Sa\rvert^2\le\lvert a\rvert^2$ for every $a$, because $\tilde\Lambda$ is positive semidefinite by T25 (i). A tier-0 expert serves at tier 2 with no certificate.

**T9a, the contraction contract, one level.** *In plain words:* one sweep applies every expert's scattering map, hands each outgoing wave to the neighbouring side, and adds the sources. If every expert's map shrinks distances by a certified $\rho<1$ and the hand-over lengthens nothing, the whole sweep shrinks distances by $\rho$. Then there is one answer, the iteration reaches it geometrically from any start, and it lies within a stated distance of the classical answer.

Pieces $i$ with complete normed wave spaces $W_i$; the global wave vector is measured in the $\ell^2$ product, $d(a,a')^2=\sum_id(a_i,a'_i)^2$. The sweep is $T(a)=\Pi\bigl((S_i(a_i))_i\bigr)+g$. If $d(S_ix,S_iy)\le\rho\,d(x,y)$ for all $i$ and $d(\Pi x,\Pi y)\le d(x,y)$, then

$$d(Ta,Ta')\le\rho\,d(a,a') .$$

If moreover $\rho<1$, $a_c$ is a fixed point of the classical sweep (the same $\Pi$ and $g$, classical maps $S^c_i$), and $d\bigl(S_i(a_{c,i}),S^c_i(a_{c,i})\bigr)\le\delta_i$, then the learned sweep has a unique fixed point $a_l$, $d(T^ka_0,a_l)\le\rho^k\,d(a_0,a_l)$, and

$$d(a_l,a_c)\le\frac{\bigl(\sum_i\delta_i^2\bigr)^{1/2}}{1-\rho}.$$

The maps $S_i$ may be nonlinear. The defect is needed only at the classical answer.

**T9b, two levels.** *In plain words:* with a coarse space, one round is a sweep followed by a coarse solve, a correction inside the coarse space that leaves no coarse residual. What must be certified is a contraction factor of that **whole round**. Then the conclusion is T1's, and the answer reached is a fixed point of the plain sweep: the coarse solve changes how the answer is reached, not what it is.

Let $T$ be the sweep, $C$ the coarse solve, $P_0$ linear, with $P_0(Cb-b)=Cb-b$ and $P_0\bigl(Cb-T(Cb)\bigr)=0$ for all $b$. Then $C(Ta)=a\Rightarrow Ta=a$. If the learned round satisfies $d(C_lT_lx,C_lT_ly)\le\rho\,d(x,y)$ with $\rho<1$, and $d(C_lT_la_c,C_cT_ca_c)\le\delta$ at the classical two-level answer $a_c$, then there is a unique $a_l$ with $C_lT_la_l=a_l$; $T_la_l=a_l$; $d\bigl((C_lT_l)^ka_0,a_l\bigr)\le\rho^kd(a_0,a_l)$; and $d(a_l,a_c)\le\delta/(1-\rho)$.

**The example that forces this (the expected finding).** On $\mathbb R^2$, coarse space the first axis, fine space the second:

$$G=\begin{pmatrix}0&\tfrac12\\10&0\end{pmatrix},\qquad C(x,y)=\bigl(\tfrac12y,\ y\bigr).$$

$G$ contracts the fine space by $\tfrac12$. $C$ changes only the coarse component and leaves no coarse residual. $0$ is the only fixed point of $G$. And $(CG)^k(1,2)=(5^k,\ 2\cdot5^k)$.

**The SNI remark beside T1.** *In plain words:* T1 needs the **learned** map to contract. "The classical map contracts and the learned map is close to it" is not enough.

$T(u)=\tfrac12u$ is a contraction with factor $\tfrac12$ and fixed point $0$. $\tilde T(u)=\tfrac12(u-\tanh10u)$ satisfies $\lvert\tilde T(u)-T(u)\rvert\le\tfrac12$ for all $u$. There is an $x\in[0.3,\,0.34]$ with $\tilde T(x)=-x$ and $\tilde T(-x)=x$, so $\tilde T^k(x)=(-1)^kx$: the learned iteration alternates for ever. ($x$ solves $\tanh10x=3x$; numerically $0.332471$.)

### 7.6 Findings, batch 5

1. **T9b: contracting the fine space is not enough (the expected finding of [[formal-proofs-implementation-plan]] §2.3, now a Lean statement).** Version 0's hypothesis, "each learned map contracts on the fine space, and the coarse space is solved exactly", admits the example above. T9b is stated with a contraction of the composite round as its hypothesis. **For the architecture chat's card:** a tier-2 convergence claim needs a certified contraction factor of the composite two-level sweep. The sentence in the architecture document's §5.3, "the contraction hypothesis is needed only on the complement of the coarse space", is not supported by T9b as stated here. One honest limit of the example: its plain sweep $G$ does not converge either ($G^2=5I$), so it shows that a fine-space bound plus an exact coarse solve does not *produce* convergence, not that a coarse solve *destroys* it.
2. **T9a's distance is in the $\ell^2$ product.** If each of $n$ experts is within $\delta$ of its classical map at the classical answer, the bound is $\sqrt n\,\delta/(1-\rho)$, not $\delta/(1-\rho)$. The headline sentence of [[formal-proofs-plan]] §3 says "differs from the classical one by at most $\delta$": it is right if $\delta$ means the defect of the whole sweep, and it needs the $\sqrt n$ if $\delta$ is per expert.
3. **T9a's exchange need only be non-expansive.** An isometry, which the document assumes, is a special case.
4. **T8's norm bound needs a non-zero space.** On the zero space every $c$ satisfies the accretivity hypothesis and $1-4Zc/(1+ZM)^2$ can be negative, so `cayley_transform` carries `Nontrivial E`. The pointwise bounds need no such hypothesis.
5. **T8's bound does not use linearity.** It is stated for any effort and flow, which is the form a nonlinear expert's increments satisfy.

### 7.7 The record, batch 6

| # | statement, in one line | Lean name | file | status | build time | axioms |
|---|---|---|---|---|---|---|
| T9′ | for $\omega\ne0$ the averaged sweep has the fixed points of $T$ | `Atlas.averaged_fixedPt_iff` | `AtlasProofs/KrasnoselskiiMann.lean` | checked | 21 s | `propext`, `Classical.choice`, `Quot.sound` |
| T9′ | $\lVert(1-\omega)a+\omega b\rVert^2=(1-\omega)\lVert a\rVert^2+\omega\lVert b\rVert^2-\omega(1-\omega)\lVert a-b\rVert^2$ | `Atlas.norm_sq_convex_combination` | same | checked | 21 s | `propext`, `Classical.choice`, `Quot.sound` |
| T9′ | one averaged step: $\lVert x'-p\rVert^2\le\lVert x-p\rVert^2-\omega(1-\omega)\lVert x-Tx\rVert^2$ | `Atlas.averaged_step` | same | checked | 21 s | `propext`, `Classical.choice`, `Quot.sound` |
| T9′ | the residuals $\lVert x_n-Tx_n\rVert$ tend to zero | `Atlas.averaged_residual_tendsto_zero` | same | checked | 21 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T9′** | Krasnoselskii–Mann: the averaged iteration converges to a fixed point | `Atlas.krasnoselskii_mann` | same | checked | 21 s | `propext`, `Classical.choice`, `Quot.sound` |
| **T22, tier 2** | one step: $\sigma\le C_\mu\delta/(1-\rho)$ and $\gamma\le C_\mu\rho^k d(\lambda^{(0)},\lambda^\dagger)$ | `Atlas.tier2_step_defect` | `AtlasProofs/LearnedMasterBound.lean` | **proved here; waits on T1** | 12 s | the three standard ones, and `sorryAx` through `Atlas.perturbed_contraction` only |
| **T22, tier 2** | the master bound with those two terms added to every step | `Atlas.learned_master_bound_tier2` | same | **proved here; waits on T1 and T4** | 12 s | the three standard ones, and `sorryAx` through `Atlas.perturbed_contraction` and `Atlas.master_bound_three_terms` only |
| **T22, tier 0** | the master bound in the energy size, defects $\tau^n+g^n$, no incomplete-solve term | `Atlas.learned_master_bound_tier0` | same | **proved here; waits on T4c** | 12 s | the three standard ones, and `sorryAx` through `Atlas.accumulation` only |
| **T22, tier 0** | the same with $g^n$ supplied by T25's error bound | `Atlas.learned_master_bound_tier0_superelement` | same | **proved here; waits on T4c** | 12 s | the three standard ones, and `sorryAx` through `Atlas.accumulation` only |
| **T2** | a map within $\eta$ of the classical sweep on a ball contracts there at $\rho+\eta$; its answer is within $\eta/(1-\rho-\eta)$ | `Atlas.existence_conditional` | `AtlasProofs/Existence.lean` | **proved here; waits on T1** | 15 s | the three standard ones, and `sorryAx` through `Atlas.perturbed_contraction` only |
| T2 | the same with the closeness stated through a derivative | `Atlas.existence_conditional_c1` | same | **proved here; waits on T1** | 15 s | the three standard ones, and `sorryAx` through `Atlas.perturbed_contraction` only |
| **T2** | if a class of maps contains such a map, it contains one with those properties | `Atlas.existence_of_learned_coupling` | same | **proved here; waits on T1** | 15 s | the three standard ones, and `sorryAx` through `Atlas.perturbed_contraction` only |
| **headline, tier 0** | solvable for every output, and energy-optimal | `Atlas.tier0_headline` | `AtlasProofs/Headline.lean` | checked | 11 s | `propext`, `Classical.choice`, `Quot.sound` |
| **headline, tier 2** | certified experts: geometric convergence near the classical answer, and the master bound with that term | `Atlas.tier2_headline` | same | **proved here; waits on T1 and T4** | 11 s | the three standard ones, and `sorryAx` through `Atlas.perturbed_contraction` and `Atlas.master_bound_three_terms` only |
| **headline, defect correction** | two cheap maps, one answer | `Atlas.defect_correction_headline` | same | **proved here; waits on T5** | 11 s | the three standard ones, and `sorryAx` through `Atlas.DefectCorrection.limit_isFixedPt` only |

**The build that checked batch 6 (2026-10-01, on battery).** `python scripts/lean_build.py --clean`: **79.3 s**, 27 files, no error; the four batch-6 files took 11 to 21 s each and gave no warning. The 48 `sorry` warnings left are all in the thirteen files of batches 1 to 3. **The axiom check** on this chat's 84 names (16 s): 73 depend on the three standard axioms only; the other 11 are the theorems of §7.10.

**The profile (on battery, after the batch-6 proofs).** `python scripts/lean_build.py --profile`: 26 files, 434.2 s one after another, **0 steps at or over ten seconds**, exit 0.

**Which of these wait on the other chat, as the axiom check found them.** `tier2_step_defect`, `learned_master_bound_tier2`, the three statements of T2 and `tier2_headline` quote T1 (`Atlas.perturbed_contraction`). `learned_master_bound_tier2` and `tier2_headline` also quote T4 (`Atlas.master_bound_three_terms`); the two tier-0 statements of T22 quote T4c (`Atlas.accumulation`); `defect_correction_headline` quotes T5 (`Atlas.DefectCorrection.limit_isFixedPt`). That is nine of the fifteen, the nine predicted before the proofs were written. T9′ (five statements) and `tier0_headline` quote nothing of batches 1 to 3 and are checked.

**What proving added, and what it changed.** Three helper lemmas: `tendsto_sub_succ_of_antitone` (a decreasing non-negative sequence has differences tending to zero), `perturbed_contraction_on` (T1 on a closed set the map sends into itself, which T2 is an instance of) and, in batch 5's `ContractionContract.lean`, `waveSweep_defect` (the defect of the sweep is at most the experts' defects combined in $\ell^2$), which the tier-2 headline needs and which `contraction_contract`'s own proof now quotes. The four files grew from 432 lines to 676. All 20 approved declarations compared are word for word what the owner approved. Two statements of T2 were then **generalised** by one line each (an `omit` of the `NormedSpace` hypothesis): `existence_conditional` and `existence_of_learned_coupling` use only the norm and completeness, not the vector-space structure over the reals. No `sorry`, no new axiom, no `maxHeartbeats`.

**The first build of the proofs passed.** Fifteen proofs in four files, no error; two linter remarks, both fixed (a `haveI` that should be `have`, and the unused hypothesis above). T9′, the from-scratch analysis proof the plan flagged as the risk, is about 115 lines of proof in a 195-line file that builds in 21 s.

### 7.8 The statements, batch 6

**T9′, Krasnoselskii–Mann.** *In plain words:* T1 and T9 need a sweep that shortens every distance by a factor below one. A passive coupling gives less: a sweep that lengthens nothing (T8 with $c=0$, T11). Iterating such a sweep need not settle: a rotation of the plane lengthens nothing and never converges. Averaging each sweep with the current iterate does settle, on a fixed point of the sweep, whenever there is one. No rate is claimed.

Let $E$ be a finite-dimensional real inner-product space, $\lVert Tx-Ty\rVert\le\lVert x-y\rVert$ for all $x,y$, $T(p)=p$ for some $p$, and $0<\omega<1$. With

$$x_{n+1}=(1-\omega)\,x_n+\omega\,T(x_n),$$

for every $x_0$ there is a $q$ with $T(q)=q$ and $x_n\to q$. The three steps are stated separately, in any real inner-product space:

$$\lVert(1-\omega)a+\omega b\rVert^2=(1-\omega)\lVert a\rVert^2+\omega\lVert b\rVert^2-\omega(1-\omega)\lVert a-b\rVert^2;$$

$$\lVert x_{n+1}-p\rVert^2\le\lVert x_n-p\rVert^2-\omega(1-\omega)\lVert x_n-T(x_n)\rVert^2\quad\text{for every fixed point }p;\qquad\lVert x_n-T(x_n)\rVert\to0 .$$

Finite dimension is used once: a bounded sequence has a convergent subsequence.

**T22, the master bound with learned experts, once per tier.** *In plain words:* the master bound adds up what each step does wrong. T4 names three parts of a coupled step's defect: $\tau$ (the experts are wrong even with the right interface data), $\sigma$ (the interface problem is posed with the wrong operator) and $\gamma$ (the interface solve is stopped early). Learned experts change $\sigma$ and $\gamma$, and differently at each tier.

*Tier 2.* At a state $v$, the classical interface sweep $T_c$ has the answer $\lambda^\star$; the learned sweep $T_l$ is a $\rho$-contraction, $\rho<1$, within $\delta$ of $T_c$ at $\lambda^\star$; the step $\Phi(\lambda,v)$ is $C_\mu$-Lipschitz in $\lambda$; the host runs $k$ sweeps from $\lambda^{(0)}$. Then $T_l$ has a fixed point $\lambda^\dagger$ and

$$\underbrace{\lVert\Phi(\lambda^\dagger,v)-\Phi(\lambda^\star,v)\rVert}_{\sigma}\le\frac{C_\mu\,\delta}{1-\rho},\qquad\underbrace{\lVert\Phi(T_l^{\,k}\lambda^{(0)},v)-\Phi(\lambda^\dagger,v)\rVert}_{\gamma}\le C_\mu\,\rho^k\,d(\lambda^{(0)},\lambda^\dagger).$$

Over $N$ steps, if these hold along the true trajectory, the starting guess is within $D$ of the learned answer, and a step magnifies the difference of two trajectories by at most $L$:

$$\lVert u^N-u^{N,\star}\rVert\le L^N\lVert u^0-u^{0,\star}\rVert+\sum_{n=1}^{N}L^{N-n}\Bigl(\lVert\tau^n\rVert+\frac{C_\mu\,\delta}{1-\rho}+C_\mu\,\rho^k\,D\Bigr).$$

*Tier 0.* There is no iteration, so there is no $\gamma$. In the energy size: if $u^{n,\text{ex}}$ is the undivided discrete step from the true state and $\tilde u^n$ the tier-0 step from it, with $\lVert u^{n,\text{ex}}-u^{n+1,\star}\rVert_M\le\tau^n$ and Galerkin error $\lVert u^{n,\text{ex}}-\tilde u^n\rVert_M\le g^n$, then

$$\lVert u^N-u^{N,\star}\rVert_M\le L^N\lVert u^0-u^{0,\star}\rVert_M+\sum_{n=0}^{N-1}L^{N-1-n}\bigl(\tau^n+g^n\bigr),$$

and a second statement puts T25's bound in place of $g^n$: the error of any reference field plus the energy of the network's field errors, step by step.

**T2, existence, as a conditional theorem.** *In plain words:* "can a learned coupling do as well as the classical one?" If a network can match the classical interface map and its derivative to within $\eta$ near the classical answer, then yes: its iteration contracts at the classical rate plus $\eta$, and its answer is within $\eta/(1-\rho-\eta)$ of the classical one. **The "if" is assumed, not proved**: it is the universal approximation theorem in $C^1$, which the proposal cites. And nothing here says that a *trained* network meets it.

Let $B$ be the closed ball of radius $r$ around $a_c$ in a real Banach space, $T_c(a_c)=a_c$, $\lVert T_cx-T_cy\rVert\le\rho\lVert x-y\rVert$ on $B$. Let $\lVert T_la_c-T_ca_c\rVert\le\eta$ and $\lVert(T_l-T_c)x-(T_l-T_c)y\rVert\le\eta\lVert x-y\rVert$ on $B$ (true if $T_l-T_c$ has a derivative of norm at most $\eta$ on $B$). If $\rho+\eta<1$ and $\eta\le(1-\rho-\eta)\,r$, then $T_l$ has exactly one fixed point $a_l$ in $B$; iterates from any $x_0\in B$ stay in $B$ with $\lVert T_l^{\,k}x_0-a_l\rVert\le(\rho+\eta)^k\lVert x_0-a_l\rVert$; and

$$\lVert a_l-a_c\rVert\le\frac{\eta}{1-\rho-\eta}.$$

The third statement is the existence form: if a class of maps (the networks of an architecture) contains one within $\eta$ in that sense, it contains one with these properties.

**The headline statements.** Each is one theorem that only combines others.

- **Tier 0** (`tier0_headline`). *Whatever a learned superelement returns, the coupled system is solvable, and its answer is the best its modes can represent in the energy norm.* Under the document's three assumptions, independent port patterns and every free coordinate read by some piece: for any network output, $\tilde{\mathsf S}c=\tilde\chi$ has exactly one solution, and $\lVert u-w(c)\rVert_M\le\lVert u-w(c')\rVert_M$ for all $c'$.
- **Tier 2** (`tier2_headline`). *If every learned expert's scattering map is certified to contract by $\rho<1$ and differs from the classical one by at most $\delta$, the coupled iteration converges geometrically from any start to a point within $\delta/(1-\rho)$ of the classical answer; over $N$ steps the error obeys the master bound with that term added.* The interface sweep of each step is the wave sweep of T9a, and $\delta=\bigl(\sum_i\delta_i^2\bigr)^{1/2}$.
- **Defect correction** (`defect_correction_headline`). *Inside defect correction, a learned expert can change how many classical steps the answer takes. It cannot change the answer.* Two runs with different cheap maps $\Psi_1$, $\Psi_2$ that converge, to $a_1$ and $a_2$, give $\Phi(a_1)=a_1$ and $\Phi(a_2)=a_2$; if $\Phi$ has at most one fixed point, $a_1=a_2$.

### 7.9 Findings, batch 6 (before the proofs)

1. **The tier-2 headline's $\delta$ is the combined defect** $\bigl(\sum_i\delta_i^2\bigr)^{1/2}$, as in §7.6 item 2. The sentence is true as written only with that reading.
2. **The tier-2 headline has no coarse space in it.** It is the one-level contract (T9a) inside the master bound. With a coarse space the hypothesis must be the contraction of the composite round (T9b), and `learned_master_bound_tier2` applies to it unchanged, because it is stated for any interface sweep, not only for the wave sweep.
3. **T2 needs room in the ball**: $\eta\le(1-\rho-\eta)\,r$, so that the learned map sends the ball into itself. The plan's one-line statement did not say so. It is not a restriction as $\eta\to0$.
4. **T2's rate is $\rho+\eta$, and its hypothesis is closeness of values at one point and of increments on the ball.** Closeness of values everywhere on the ball is not needed.
5. **T22 at tier 0 measures in the energy size**, the norm in which T25's bound holds. Its stability constant $L$ is therefore the tier-0 step's Lipschitz constant in that size, a hypothesis that has to be established for the time scheme in use; nothing here proves it.
6. **The defect-correction headline needs the classical problem to have at most one answer** to conclude that two runs agree. Without it, each run still ends on *a* fixed point of the classical map, which is T5's own statement.

**After the proofs:** none of the six changed, and proving found nothing further. No statement of batch 6 turned out false or short of a hypothesis.

### 7.10 What waits on the merge with the other chat's branch

Eleven theorems of this chat are proved in their own files and quote a theorem of batches 1 to 3 whose proof is a `sorry` on this branch. The kernel reports `sorryAx` for each, through the name in the last column and through nothing else. They become *checked* when the other branch is merged and that theorem is proved; nothing in this chat's files has to change.

| theorem | file | quotes |
|---|---|---|
| `Atlas.contraction_contract` (T9a) | `ContractionContract.lean` | T1, `Atlas.perturbed_contraction` |
| `Atlas.two_level_contract` (T9b) | `TwoLevel.lean` | T1 |
| `Atlas.tier2_step_defect` (T22) | `LearnedMasterBound.lean` | T1 |
| `Atlas.learned_master_bound_tier2` (T22) | same | T1; T4, `Atlas.master_bound_three_terms` |
| `Atlas.learned_master_bound_tier0` (T22) | same | T4c, `Atlas.accumulation` |
| `Atlas.learned_master_bound_tier0_superelement` (T22) | same | T4c |
| `Atlas.existence_conditional` (T2) | `Existence.lean` | T1 |
| `Atlas.existence_conditional_c1` (T2) | same | T1 |
| `Atlas.existence_of_learned_coupling` (T2) | same | T1 |
| `Atlas.tier2_headline` | `Headline.lean` | T1, T4 |
| `Atlas.defect_correction_headline` | same | T5, `Atlas.DefectCorrection.limit_isFixedPt` |

**If the other chat changes the statement of T1, T4, T4c or T5**, these eleven are where it would show: the merged build would fail in the file named, and the fix is there. The statements quoted are those of commit `428e4aa`.

**The count.** Batches 4 to 6: 57 theorems the owner approved (28, 14, 15), 27 helper lemmas (22, 2, 3) beside the three facts of §7.1 proved before the first OK, and 27 definitions and structures listed in the blueprint. 46 theorems are checked on this branch; 11 wait. The whole project builds clean in 79.3 s on battery (27 files, four levels of imports), with 48 `sorry` left, all in the thirteen files of batches 1 to 3.

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
