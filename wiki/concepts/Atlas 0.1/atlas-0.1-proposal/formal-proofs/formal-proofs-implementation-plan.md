# Formal proofs — the implementation plan: what gets machine-checked, by what route, and how long it takes

**Type:** Concept page — **implementation plan** (folder: `Atlas 0.1/atlas-0.1-proposal/formal-proofs/`)
**Status:** written 2026-10-01 at the owner's request, after the set-up was timed and the statements of steps 1 to 3 were drafted. **Nothing is proved yet** *(superseded at the integration, 2026-10-01: batches 1 to 6 are proved, merged and checked, 105 theorems with no `sorry`; [[formal-proofs-record]] §7.11; batch 7 of §3.3 is not started)*. Every effort figure on this page is an **estimate**, marked **[AI Inference]**, and §6.2 says how each will be replaced by a measured one. Every build time is measured, on the machine and power state named beside it.
**Hub:** [[00-proposal-workstreams]] · **Inventory:** [[formal-proofs-plan]] · **Record:** [[formal-proofs-record]] · **Design the theorems serve:** [[chart-operator-architecture]] (version 1; when this page was written it was only on `origin/atlas-0.1` and was read from the remote, and it was merged into the local branch the same day, merge commit `eb75132`) · [[chart-operator-design-decisions]] · [[coupling-cost-and-complexity]]

---

## 0. The plan in one paragraph

**About seventy statements will be machine-checked in Lean 4 for the proposal, in six batches, and the whole project will re-check in about a minute on mains.** They are all metric-space arguments or finite-dimensional linear algebra, which is why they are cheap. They support three sentences the website can say: *a certified learned coupling converges and lands near the classical answer* (tier 2); *whatever a learned superelement returns, the coupled system is solvable and its answer is the best its modes allow* (tier 0, new with the architecture's version 1); and *inside defect correction a learned expert changes the cost, never the answer*. **Estimated authoring effort: 26 to 43 hours of this chat's working time, [AI Inference]**, to be replaced by a measured rate after the first batch. What is **not** machine-checked is listed with the reason in §5: the continuous theory, chart injectivity, universal approximation, convergence for M-matrices, and anything about a trained network's own numbers, which are certified numerically.

---

## 1. What "formalized" and "Lean-verified" mean on this page

| word | what exists | who vouches |
|---|---|---|
| **stated** | the plain-language statement, the mathematics in LaTeX (blueprint and [[formal-proofs-record]]), and a Lean declaration that typechecks with `sorry` for its proof | nobody yet |
| **approved** | the owner has read the plain statement and the mathematics and said yes | the owner, for *what is claimed* |
| **Lean-verified** | no `sorry` anywhere in the project; `#print axioms` on the declaration shows only `propext`, `Classical.choice`, `Quot.sound`; the build is timed | the Lean kernel, for *the proof* |
| **paper proof** | a complete proof in the vault or the proposal document, not formalised | a careful reader |
| **cited** | a published theorem with its source and hypotheses | the literature |
| **certified hypothesis** | a number a theorem needs about a specific expert or chart, computed by a tool outside Lean | the tool |

**The gate that does not move:** a statement is proved only after the owner approves it, batch by batch. A statement that turns out false or short of a hypothesis is recorded as a finding and reported, never weakened in silence. One such finding exists already (T3, §2.1) and one more is expected (T9, §2.3).

---

## 2. What changed since the inventory was written

### 2.1 T3 is false as written for restricted additive Schwarz

Recorded in [[formal-proofs-record]] §5.1. The plan's sentence needed the hypothesis that the sweep converges from every start. The Schur and Dirichlet–Neumann parts stand. **Consequence for this plan:** T3 is five families of statements instead of one, and T14 (convergence for M-matrices), which supplies the missing hypothesis for conduction, becomes more valuable than its "later" slot suggested (§4).

**A cheaper remedy than any theorem, for the code.** The workbench's Schwarz driver stops on the size of its update. If it also computed the true residual $\lVert f-Au\rVert$ once at exit (one product with the undivided matrix), a spurious fixed point would be caught by a number, on every run, for every matrix class. **[AI Inference]:** that is worth more to the compiler's rule than a convergence theorem that covers one matrix class. It is the demo chat's code and its decision.

### 2.2 The architecture's version 1: three coupling tiers

The architecture chat's confirmed design replaces "iterate wave variables everywhere" by three tiers. What that does to the theorems, from the design page's own note to this chat:

| tier | how it couples | what guarantees it | theorems |
|---|---|---|---|
| **0, superelement** | each expert returns constraint modes; the host forms the port matrix as an **energy Gram** and solves one interface system; **no iteration** | holds for **every** network output | **T25, T24, T26** (new) |
| **1, tangent** | Newton on the interface equation with tangent modes | local; **[AI Inference]** on the design page | none planned |
| **2, black box** | Robin (wave) exchange, iterated | conditional on a certified contraction factor | **T1, T8, T9, T9′, T22, T2**, statements unchanged, scope now tier 2 only |

- **T1, T8, T9, T9′ keep their statements.** T8 gains a use at tier 0: a Gram port matrix is positive semidefinite, so its Cayley transform is non-expansive for every impedance.
- **T22's incomplete-solve term $\rho^k$ exists only at tier 2.** A tier-0 step has no iteration; its per-step defect is the Galerkin error of T25. So T22 is stated twice, once per tier.
- **The SNI counterexample** becomes a remark beside T1, checked in Lean (§3.2): accuracy of the learned solve plus contraction of the *classical* map does not give convergence. The learned map $\tilde T(u)=\tfrac12\bigl(u-\tanh 10u\bigr)$ stays within $\tfrac12$ of the contraction $T(u)=\tfrac12u$ and has the two-cycle $\pm0.332471$.

### 2.3 An expected finding: T9's two-level form

Version 0 of the design stated T9 as: each learned map is $\rho$-Lipschitz **on the fine space** $V_0^\perp$, the coarse space $V_0$ is solved directly, and then the two-level iteration converges at rate $\rho$. **[AI Inference], from a two-dimensional example worked in this session and not yet checked in Lean:** that hypothesis is not enough. Take $V_0=\operatorname{span}(e_1)$, $V_0^\perp=\operatorname{span}(e_2)$ and the linear sweep $G=\begin{pmatrix}0&\rho\\10&0\end{pmatrix}$. $G$ restricted to the fine space has norm $\rho$, yet after each exact coarse solve the fine error is multiplied by $10\rho$, which exceeds one for $\rho=\tfrac12$. The fine-space bound says nothing about how the coarse component feeds the fine one.

**So T9 will be stated in two parts:**
- **T9a (one level):** each expert's scattering map $\rho$-Lipschitz on its own port space, the exchange an isometry: the sweep is a $\rho$-contraction on the product, and T1 applies. Fully specified, Tier A.
- **T9b (two levels):** the hypothesis is a certified contraction factor **of the composite two-level sweep**, with the example above as a checked remark showing why the fine-space bound alone does not do. This is also what tier 2's card entry should certify.

**What survives any accelerator.** Existence, uniqueness and the distance $\delta/(1-\rho)$ are properties of the fixed point, so they hold whether the host iterates by sweeps, by Anderson or by GMRES. Only the *rate* is a statement about plain sweeps. For a linear contraction, GMRES is no slower than the sweep in residual norm, by comparing with the polynomial $(1-z)^k$: a **paper proof**, not planned for Lean.

---

## 3. What will be machine-checked for the proposal

Effort is in hours of this chat's working time, tool time included, **[AI Inference]**. "Lines" are lines of Lean proof, an estimate of size. "Level" is the file's depth in the import chain, which sets the build time (§6.1).

### 3.1 Batches 1 to 3: stated, typechecked, awaiting the owner's OK (48 declarations)

Each row's declarations, files and statements are in [[formal-proofs-record]] §3 and §4.

| batch | theorems | Lean route | lines | effort | level |
|---|---|---|---|---|---|
| **1** | **T1** perturbed contraction (3 declarations) | Mathlib's `ContractingWith`: `fixedPoint`, `fixedPoint_unique`, `dist_fixedPoint_le`, `LipschitzWith.iterate`; the sharp example by direct computation on $\mathbb R$ | 70 | 0.5–1 | 1 |
| 1 | **T7** certificate (4) | triangle inequality; `ContinuousLinearMap.opNorm_le_bound` for the least constant; `Prod.norm_def` and `norm_num` for the two-mode example | 80 | 0.5–1 | 1 |
| 1 | **T4** master bound (9) and **T4e** (2) | `abel` for the two identities; induction with `Finset.sum_range_succ` for accumulation; `geom_sum_eq` for the regimes; operator-norm inequalities for T4e | 220 | 1.5–2.5 | 1 |
| **2** | **T3a** Schur (1), **T3b** Dirichlet–Neumann (3) | `map_sub`, `map_add` and rewriting; `smul_eq_zero` for $\theta\ne0$ | 90 | 0.75–1 | 1 |
| 2 | **T3c** Schwarz algebra (6) | `Finset.sum_add_distrib`, `map_sum`; the partition of unity as a hypothesis | 90 | 0.75–1 | 1 |
| 2 | **T3c, T3d** convergence (3) | an induction giving $G_f^{\,k}(u)-u^\star=G_0^{\,k}(u-u^\star)$; uniqueness of limits; T7 | 80 | 0.75–1 | 2 |
| 2 | **T3c** counterexample (1) | an explicit `Schwarz` structure over $\mathbb Q$ from $3\times3$ and $2\times2$ matrices; `Matrix.toLin'`, `fin_cases`, `norm_num` | 100 | 1–2 | 2 |
| **3** | **T5** defect correction (9) | `Filter.Tendsto` algebra with `ContinuousAt`; `HasFDerivAt.sub`, `.const_smul`; `Module.End.HasEigenvalue` | 180 | 1.5–2 | 1 |
| 3 | **T10** symmetry averaging (3) | `Equiv.sum_comp` with `Equiv.mulRight`; `Finset.smul_sum`; the non-group example by `norm_num` | 70 | 0.75–1 | 1 |
| 3 | **T11** passive interconnection (4) | `inner_neg_right`; `Equiv.sum_comp`; `Finset.sum_fiberwise` | 60 | 0.5–0.75 | 1 |
| | **batches 1 to 3** | | **about 1,040** | **9–13** | |

**T3's status line for the demo chat** is written into [[formal-proofs-plan]] §6 and committed alone the moment batch 2 passes the axiom check, as the chat's instructions require.

### 3.2 Batches 4 to 6: to be stated next, then approved, then proved

| batch | theorem | the statement, in one line | Lean route | lines | effort | level |
|---|---|---|---|---|---|---|
| **4** | **T25 (i)–(iii)**, the Gram port matrix | for *any* learned modes $\hat H_k$: the Gram $\tilde\Lambda$ is symmetric positive semidefinite; $\tilde\Lambda-\Lambda$ is the Gram of the mode errors in the $A_I$ inner product; the assembled $\tilde{\mathsf S}\succeq\mathsf S$ and inherits $\mathsf S$'s coercivity constant | modes as a finite family of vectors, the Gram as a real matrix; `Matrix.PosSemidef`; the cross terms vanish by $A_IH_k=Bq_k$ | 120 | 1.5–2.5 | 1 |
| 4 | **T25, energy optimality** (the document's Theorem 6) | the reconstruction is the energy-closest element of the learned trial set to the exact solution (Céa with constant one), and its error is at most the truncation error plus the energy of the field errors | first an abstract lemma (a minimiser of a quadratic energy over an affine set is the closest point in the energy seminorm), then the instantiation: the host's system $\tilde{\mathsf S}c=\tilde\chi$ is that set's Galerkin system | 180 | 2–4 | 2 |
| 4 | **T24**, perturbation of the interface solve | $\lVert\tilde c-c\rVert\le\bigl(\lVert\mathsf S-\tilde{\mathsf S}\rVert\lVert c\rVert+\lVert\chi-\tilde\chi\rVert\bigr)/\beta$ with $\beta$ the coercivity constant, which T25 (iii) carries over from $\mathsf S$ | T4e's proof with a perturbed right-hand side | 30 | 0.5 | 2 |
| 4 | **T26**, the label-free objective | $\operatorname{tr}\tilde\Lambda+\hat u_p^\top A_I\hat u_p-2f^\top\hat u_p$ equals the supervised energy-norm loss plus a constant that does not depend on the network | T25 (ii) on the diagonal, and completing the square with $A_Iu_p=f$ | 40 | 0.5–1 | 2 |
| **5** | **T8**, the Cayley bound | for $\langle\Lambda e,e\rangle\ge c\lVert e\rVert^2$, $\lVert\Lambda\rVert\le M$, $Z>0$: $\lVert(I-Z\Lambda)e\rVert^2\le\bigl(1-\tfrac{4Zc}{(1+ZM)^2}\bigr)\lVert(I+Z\Lambda)e\rVert^2$ in any real inner-product space; in finite dimensions $I+Z\Lambda$ is invertible and $S$ has that norm bound; with $c=0$, a Gram port matrix gives a non-expansive $S$ | `norm_add_sq_real`, `norm_sub_sq_real`; `LinearMap.injective_iff_surjective`; `opNorm_le_bound` | 120 | 1.5–3 | 1 |
| 5 | **T9a**, the contraction contract, one level | $\rho$-Lipschitz experts and an isometric exchange give a $\rho$-contraction on the product; with T1: one answer, geometric convergence, within $\delta/(1-\rho)$ of the classical one | the $\ell^2$ product norm (`PiLp 2`); T1 | 80 | 1–2 | 2 |
| 5 | **T9b**, two levels, and the remark of §2.3 | the same conclusion from a certified contraction of the composite sweep; the $2\times2$ example | T1; an explicit matrix | 80 | 1–3 | 2 |
| 5 | **the SNI remark** beside T1 | a contraction $T$, a map $\tilde T$ within $\tfrac12$ of it everywhere, and a two-cycle of $\tilde T$ | the intermediate value theorem for $\tanh 10x=3x$ on $[0.3,\,0.34]$, with $e^6>19$ | 60 | 1–1.5 | 2 |
| **6** | **T9′**, Krasnoselskii–Mann in finite dimensions | a non-expansive map with a fixed point: the averaged iteration converges to a fixed point | from scratch: the identity $\lVert(1-\omega)x+\omega y\rVert^2=(1-\omega)\lVert x\rVert^2+\omega\lVert y\rVert^2-\omega(1-\omega)\lVert x-y\rVert^2$; summable residuals; a convergent subsequence by compactness (`IsCompact.tendsto_subseq`); Fejér monotonicity | 200 | 3–5 | 1 |
| 6 | **T22**, the learned master bound, per tier | tier 2: $\sigma\le C_\mu\,\delta/(1-\rho)$ and $\gamma\le C_\mu\,\rho^k\lVert\lambda^{(0)}-\lambda^\dagger\rVert$ inside T4; tier 0: the defect is T25's Galerkin error and $\gamma=0$ | T4's three-term bound with T1 and T25 as the estimates | 90 | 1–2 | 3 |
| 6 | **T2**, existence, as a conditional theorem | **if** a network within $\eta$ of the classical scattering map in $C^1$ on a closed ball exists (the approximation theorem, **a hypothesis**), the learned iteration contracts at $\rho_{\text{cl}}+\eta$ on that ball and its answer is within $\eta/(1-\rho_{\text{cl}}-\eta)$ | T1 on a complete subset the map sends into itself | 70 | 1–2 | 3 |
| 6 | **the headline statements** | the three sentences of §7, each as one Lean theorem that quotes the others | composition only | 60 | 1 | 4 |
| | **batches 4 to 6** (about 26 declarations) | | | **about 1,130** | **15–27** | |

**Status of this section, kept by the chat that states and proves batches 4 to 6** (statements, record and findings: [[formal-proofs-record]] §7).

- **Batch 4: Lean-verified, 2026-10-01** (approved by the owner the same day). 28 theorems in five files, no `sorry`, three standard axioms only, clean build of the whole project 56.0 s on battery. The route above was followed, with one change of shape: the energy enters as three bilinear forms on real vector spaces (the blocks $A_I$, $B$, $D$), not as matrices, and only the port matrices are matrices. T24 is on level 1, not 2: it needs nothing from T4e's file.
- **Batch 5: proved, 2026-10-01** (approved by the owner the same day). 14 theorems in four files with no `sorry` of their own, clean build 51.6 s on battery; 12 are Lean-verified, and T9a's and T9b's contracts quote T1 and become verified when the other chat's proof of it is merged. T9b's expected finding is stated as `Atlas.two_level_fine_contraction_diverges`. T8's bound is stated for any effort and flow, with the linear and the finite-dimensional forms as consequences; its file is on level 2, because it carries the tier-0 corollary about the Gram port matrix.
- **Batch 6: proved, 2026-10-01** (approved by the owner the same day). 15 theorems in four files with no `sorry` of their own, clean build of the whole project 79.3 s on battery (27 files); 6 are Lean-verified (T9′ and the tier-0 headline) and 9 quote T1, T4, T4c or T5 and become verified at the merge. T22 is stated per tier as planned; T2 carries one hypothesis the table's line did not, room in the ball ($\eta\le(1-\rho-\eta)\,r$); the headline file is on level 4, the chain's limit.
- **Integrated, 2026-10-01.** The branch of batches 1 to 3 was merged (commit `c830510`) and the whole project re-checked on the laptop, on battery: clean build 137.9 s, 27 files, no `sorry`; 142 declarations, none failing the axiom check; no step of ten seconds in the profile. **Batches 4 to 6 are Lean-verified in full: 57 theorems.** Measured authoring for these batches, from the build log's clock: batch 4's proofs about 25 minutes from first proof build to clean build; the fifteen proofs of batch 6 compiled on the first build ([[formal-proofs-record]] §7.1, §7.7, §7.11).

**Why tier 0 goes first (batch 4).** It is the confirmed design's own guarantee, it is pure linear algebra with complete paper proofs in the proposal document, and the architecture document already says in print that its theorems "have not been machine-checked". Batch 5 and 6 are the tier-2 theory, whose statements still need the redesign of §2.3.

### 3.3 Batch 7: the blueprint, continuous integration, the record

| item | what | effort |
|---|---|---|
| blueprint | every statement's plain words, paper proof, Lean name and status; the dependency graph. **It renders today** (checked in a browser 2026-10-01: mathematics typeset, graph drawn, no console errors), served locally by the `lean-blueprint` entry of `.claude/launch.json` | 1 |
| `checkdecls` | the blueprint's own check that every Lean name it cites exists; one more Lake dependency | 0.5 |
| CI | `.github/workflows/lean.yml`: fetch the Mathlib cache, build, fail on any `sorry` (the axiom check, not a log search). **Not enabled without the owner's OK**: it spends Actions minutes on a private repository | 1 |
| record | one row per theorem with its build time and axioms | with each batch |

---

## 4. Lean-verifiable later, not needed by the proposal

| # | statement | Lean route and what Mathlib lacks | effort, **[AI Inference]** | worth it when |
|---|---|---|---|---|
| **T3e** | two windows and a symmetric positive definite $A$: every Schwarz fixed point is the solution | index-level argument on the overlap block; needs windows as subsets, not abstract maps | 2–3 | the compiler wants an unconditional rule for two-window cases |
| T3c, second example | the positive definite $6\times6$ counterexample, in Lean | rational $6\times6$ and $4\times4$ matrix arithmetic; positive definiteness by an explicit $LDL^\top$ | 2–3 | the "even positive definite" claim is to be quoted publicly |
| **T23** | step consistency: the step-doubling defect bounds the drift over $2^m$ substeps | telescoping over halvings; iterates of compositions | 2–3 | a flow-map expert exists (the design applies it only to those) |
| **T12** | Schwarz as alternating projections; $(P_2P_1)^n\to P_{V_1\cap V_2}$ in finite dimensions | orthogonal projections are in Mathlib; the convergence and its rate by the angle are not | 4–8 | starting a formal DD library |
| **T6** | defect correction's local rate (Stetter's Theorem 2; Ostrowski) | Fréchet derivative and Gelfand's formula are in Mathlib (the latter over $\mathbb C$); Ostrowski's theorem is not | 8–15 | the rate, not only consistency, is to be claimed |
| **T13** | additive Schwarz's condition number from a stable decomposition | the abstract Schwarz framework is absent; finite-dimensional version from scratch | 10–20 | a formal DD library |
| **T14** | restricted additive Schwarz converges for M-matrices (Frommer and Szyld) | M-matrix and weak-regular-splitting theory is absent | 15–30, **with real risk** | T3's hypothesis is to be discharged for the conduction family by a checked theorem rather than a citation |
| T16 | the pull-back is exact in discrete form | change of variables is in Mathlib; the discrete Piola statement needs a mesh formalism | 6–12 | the chart is to carry a checked conservation claim |
| the design's Proposition 2 | the corner bound $K\ge\max(\tan\tfrac\alpha2,\cot\tfrac\alpha2)$ | a $2\times2$ inequality; stating $K$ through singular values takes care | 2–4 | the chart rule is to cite a checked bound |
| T21 | port typing and compiler rules as types and lemmas | engineering, incremental | open-ended | after the proposal |
| T19 | certified Lipschitz bounds for a trained tier-2 expert | bound propagation, following TorchLean | weeks | a tier-2 expert exists; **tier 0 does not need it** |

---

## 5. Not machine-checked, and why

| # | statement | level | why not Lean, now |
|---|---|---|---|
| T17 | continuous Schwarz convergence in Sobolev spaces | cited (Lions 1988, 1990) | needs Sobolev spaces and trace theory; a research project |
| T15, and the design's Proposition 1 | chart injectivity (Rado–Kneser–Choquet, Tutte, Floater); equal-value cuts give modulus $Mm/n$ | cited | conformal-map and planar-topology theory Mathlib does not have |
| the design's Proposition 3 | the discrete injectivity certificate | paper proof in the proposal document | its last step is a degree argument (a local homeomorphism whose boundary winds once is a bijection); Mathlib has no degree theory. **The certificate itself is computed per chart**, level "certified hypothesis" |
| T20 | cross-points in optimized Schwarz | open in general | an open problem; the face-based tier 0 has none |
| the approximation theorem inside T2 | a network within $\eta$ in $C^1$ exists | cited (Hornik 1991; Kovachki et al.) | universal approximation in $C^1$ is not in Mathlib; T2 takes it as a hypothesis and says so |
| GMRES and Anderson at tier 2 | no slower than the sweep, for linear contractions | paper proof (§2.3) | formalising Krylov methods is out of scope |
| the mortar case | non-matching side parametrisations | **unanalysed**, on the design page too | T25's optimality does not cover it; nothing here claims it |
| tier 1 | Newton with tangent modes | **[AI Inference]** on the design page | no theorem is stated yet |
| any trained expert's numbers | a tier-2 contraction factor; a chart inside its envelope; the Gram identity's residual | **certified hypothesis** | these are measurements of one object, made by a tool; Lean checks what follows from them |

---

## 6. How long

### 6.1 Lean's verification time (measured)

`scripts/lean_build.py` times every run into `out/lean/builds.jsonl`. The project's own proofs have cost nothing measurable so far; a file's time is the price of loading Mathlib and running its start-up code.

| what | on battery, 1.4 GHz | on mains |
|---|---|---|
| one file, before its first proof | 26 to 30 s | 12 to 15 s warm; 36 to 64 s when eleven files load for the first time that day |
| a file one level up the chain | 21 s | 10 s |
| **full re-check of all 13 statement files** | 81.6 s and 102.6 s, in two parts | **34.0 s warm, 136.5 s cold** |
| the axiom check on 58 declarations | 32 to 75 s | not yet timed |

Files that do not import each other are checked at the same time, so the total is roughly **one first-level file plus ten seconds per further level**, not a sum over files.

**Projected for the finished project, [AI Inference] from the rows above:** about 26 files in at most four levels: **60 to 80 s warm on mains, three to four minutes cold or on battery.** The owner's five-minute target holds with margin on mains. **Two rules keep it there:** the import chain stays at four levels or fewer, and `python scripts/lean_build.py --profile` must list no elaboration step of ten seconds or more, a slow step being rewritten as explicit steps, never hidden by a larger `maxHeartbeats`.

**In continuous integration, [AI Inference]:** fetching Mathlib's cache on a fresh runner is the larger part, a few minutes; the build itself is the cold figure above.

### 6.2 Authoring effort (measured on batches 1 to 3, and the rest rescaled)

**Measured** by the chat that proved batches 1 to 3, **in a cloud container** (Linux, 4 virtual processors), not on the laptop ([[formal-proofs-record]] §5A). Each window runs from the batch's first proof edit to the batch verified (build, axiom check, profile). The guesses of 2026-10-01 stand beside the measurements.

| batch | lines of proof: guessed, **measured** | first proof edit to verified | theorems per hour: guessed, **measured** | proof lines per hour, **measured** (guessed: 100–150) | build seconds per file in a full re-check (the laptop on mains, warm: 12–15) |
|---|---|---|---|---|---|
| 1: T1, T7, T4, T4e (18 theorems) | 370, **150** | 6 min 11 s | 4–7, **175** | about 1,450 | 2.0–3.2 |
| 2: T3 (14 theorems) | 360, **135** | 7 min 22 s | 3–4, **114** | about 1,100 | 1.9–2.2; the counterexample's matrix arithmetic 7.9 |
| 3: T5, T10, T11 (16 theorems) | 310, **108** | 8 min 5 s | 4–6, **119** | about 800 | 1.5–2.8 |
| **1 to 3** (48 theorems) | about 1,040, **393** | **21 min 38 s** | 4–5, **133** | about 1,090 | the whole project, 13 files: 16.7 s |

Lines of proof exclude comments, docstrings and the import lines added. Counting the reading, the set-up and paper drafts of all 48 proofs made while Mathlib downloaded, batch 1 took 18 min 42 s (58 theorems and about 480 lines per hour). The record-keeping after each batch (the record, the blueprint and its build, the plans, the log, the three scans, the commit) took **9 minutes for batch 1 and 3.5 for batch 2**. The whole session, from the branch's fast-forward to the final axiom check over all 58 declarations, took **47 min 26 s**.

**The table, rescaled.** The old estimate stays beside each line. The rescaled figures are **[AI Inference]**: three batches measured, all of them statements whose proofs needed no theory that Mathlib lacks.

| scope | effort, hours of chat working time: the estimate of 2026-10-01 | measured, or rescaled from batches 1 to 3 (**[AI Inference]** where not measured) |
|---|---|---|
| batch 1 (stated) | 2.5–4.5 | **measured: 0.1 to proof, 0.3 with the reading and the set-up, and 0.15 of record-keeping** |
| batch 2 (stated) | 3.25–5 | **measured: 0.12 to proof and 0.06 of record-keeping** |
| batch 3 (stated) | 2.75–3.75 | **measured: 0.13 to proof** |
| **batches 1 to 3** | **9–13** | **measured: 0.36 to proof; 0.79 for the whole session to the final check** |
| batches 4 to 6 (to state, approve, prove) | 15–27 | 2–5, if the other chat's rate is like this one's; its own batch-4 measurement replaces this |
| batch 7 (blueprint, CI, record) | 2–3 | 0.5–1.5 |
| **the proposal's scope** | **26–43** | **about 4–8** |
| §4's later items, if all were done | 50–100, most of it T13, T14 and T6 | 10–40; the least certain line, since T13, T14 and T6 build theory Mathlib does not have |

**Why the guess was high, [AI Inference]:** it assumed a hundred to a hundred and fifty lines per hour and a fifteen-second build loop, and it over-counted the lines by a factor of about two and a half. Batch 1 needed no theory Mathlib lacks: Banach's theorem and its estimates are there, and the rest is algebra. Batch 2 kept a similar rate with explicit $3\times3$ rational matrices, and batch 3 with derivatives, eigenvalues and group averages. The rate fell a little from batch to batch, as more of the work went into finding the Mathlib import that held a lemma. Where a batch must build theory from scratch (T9′ in batch 6; T13, T14, T6 later), this rate should not be assumed.

**The calibration rule** (applied above). When batch 1 is Lean-verified, this table is rewritten from the measured rate (lines and declarations per hour, build seconds per file), the guess is kept beside it, and every later estimate is scaled. If batch 1 takes more than twice its estimate, the owner is told before batch 2 starts. **Batch 1 took far less than its estimate**, so batch 2 started without a stop.

**Elapsed time** depends on the owner's approvals, one round per batch. **[AI Inference]:** at the measured rate, authoring no longer sets the elapsed time; the approval rounds do. The earlier figure was four to eight working days for the proposal's scope if each round is answered within a day. With authoring this fast, the elapsed time is about as many working days as there are approval rounds left: batches 4, 5 and 6, and the CI workflow, so about four.

---

## 7. What the website will be able to say, and what backs each sentence

| sentence | backed by | status |
|---|---|---|
| **Tier 0.** *Whatever a learned superelement returns, the coupled system is solvable, and its answer is the best its modes can represent in the energy norm.* | T25 (i)–(iii) and energy optimality; T24 for the sensitivity | to be stated (batch 4) |
| **Tier 2.** *If every learned expert's scattering map is certified to contract by $\rho<1$ and differs from the classical one by at most $\delta$, the coupled iteration converges geometrically from any start to a point within $\delta/(1-\rho)$ of the classical answer; over $N$ steps the error obeys the master bound with that term added.* | T1, T8, T9a, T4, T22 | T1 and T4 stated; the rest batches 5 and 6 |
| **Defect correction.** *Inside defect correction, a learned expert can change how many classical steps the answer takes. It cannot change the answer.* | T5 | stated (batch 3) |
| **Decomposition.** *A direct interface solve, and Dirichlet–Neumann at convergence, return the undivided discrete solution; a convergent Schwarz sweep converges to it.* | T3 | stated (batch 2), with the finding |
| **The certificate.** *The error of a returned state is at most its residual over $1-L$, and the constant must be an operator norm.* | T7 | stated (batch 1) |
| **Symmetry and passivity.** *Averaging any expert over a finite symmetry group makes it exactly equivariant; passive experts joined by the port rule form a passive system.* | T10, T11 | stated (batch 3) |

Each sentence carries its scope on the page that shows it: finite-dimensional, linear where stated, exact local solves, and for tier 2 a certificate that must actually be computed for the expert.

---

## 8. Order of work and gates

1. **The owner's OK on batches 1 to 3** (the statements are in [[formal-proofs-record]] §4). Batch 2 first if only one can be read now: the demo chat is waiting on T3.
2. Prove batch 1; build, axiom check, profile; rewrite §6.2 from the measured rate; commit.
3. Prove batch 2; **write T3's status line and commit it alone**; then the batch's commit.
4. Prove batch 3; commit.
5. State batch 4 (tier 0); the owner's OK; prove; commit. Then batches 5 and 6 the same way.
6. Batch 7; the owner's OK before the CI workflow is enabled.

**Each batch ends the same way:** `lake build` timed; `scripts/lean_axioms.py` passing; `--profile` showing no step over ten seconds; the record's rows filled; the vault's three scans; one commit.

---

## 9. Risks

| risk | what would show it | what is done |
|---|---|---|
| a statement is false or short of a hypothesis | the proof does not close, or a counterexample is found on paper first | recorded as a finding and reported. One found (T3), one expected (T9b) |
| T25's hypotheses do not match the proposal document's | the instantiation of the abstract optimality lemma needs something the document does not state | the document lists them (conforming sides, boundary data in the port span, $\lVert\cdot\rVert_M$ a norm on $U_0$); each becomes a named hypothesis, and any addition is reported to the owner for the architecture document |
| T9′ takes longer than estimated | batch 6 overruns | it is the only from-scratch analysis proof; it can be moved after the headline without blocking it, since the headline uses T9a |
| the Schwarz counterexample is slow to check | `--profile` lists a step over ten seconds | the matrices are $3\times3$; computed entry by entry rather than by one large `simp` |
| the local branch and `origin/atlas-0.1` had diverged | the design's version 1 and its T24–T26 were not in the local tree | **merged 2026-10-01 at the owner's request** (`eb75132`): two conflicts, in the gap-worklist and the log, each resolved by keeping both sides' text |
| two chats prove different batches at the same time | their branches both touch the record, the status lines and the build log | each chat owns named files and named blocks; `ATLAS_LEAN_TAG` gives each its own record files; the full build and the full axiom check are run once, after the branches are merged |
| timings rot | a later build on another power state is compared with these | every recorded time carries its power state; the calibration rule of §6.2 |
| Mathlib moves | a later bump renames a lemma | the version is pinned by tag and manifest; a bump is its own commit with its own timed build |

---

## See Also

- [[formal-proofs-plan]] — the inventory T1 to T23, the four levels of "proved", and the status lines
- [[formal-proofs-record]] — the statements, the build times, the findings
- [[chart-operator-architecture]] — the design; its version 1 defines the tiers and T24 to T26
- [[master-error-bound]] · [[defect-correction-learned-operator]] · [[symmetry-averaging-atlas-0.1]] · [[port-algebra-atlas-0.1]] · [[atlas-and-standard-dd-theory]] — the vault's results being formalised
- [[gap-worklist]] — W353 (this work), W348 (the compiler rule that cites T3)
