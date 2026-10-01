# Formal proofs — what can be proved, what can be machine-checked in Lean, and in what order

**Type:** Concept page — **theorem inventory and formalisation plan** (folder: `Atlas 0.1/atlas-0.1-proposal/formal-proofs/`)
**Status:** written 2026-09-30. Every effort estimate is **[AI Inference]**. The statements marked *ours* are new combinations of known results, and are proved here only in sketch. **Updated 2026-10-01 by the proofs chat:** Lean 4.34.1 and Mathlib `v4.34.1` are installed and timed; the Mathlib names quoted below are confirmed against that version, with the corrections in [[formal-proofs-record]] §2; the statements of steps 1, 2 and 3 are written in Lean and typecheck, **none is proved yet**; and **T3's statement for restricted additive Schwarz is false as written below** ([[formal-proofs-record]] §5.1). §6 holds one status line per theorem.
**Hub:** [[00-proposal-workstreams]] · **Formalises results from:** [[master-error-bound]] · [[defect-correction-learned-operator]] · [[composition-error-theory]] · [[temporal-error-accumulation]] · [[symmetry-averaging-atlas-0.1]] · [[atlas-and-standard-dd-theory]] · [[chart-operator-architecture]] §4
**Prior formal work:** [[dd-neural-prior-art-2026]] §1.6

---

## 0. The answer, in one paragraph

**Most of what the owner listed can be proved, and a useful core of it can be machine-checked in weeks, not years**, because the results that carry the proposal are **finite-dimensional and metric**: fixed points, contractions, telescoping sums, inner-product inequalities. Mathlib has Banach's fixed-point theorem with its a-priori and a-posteriori bounds, inner-product spaces, orthogonal projections and Lax–Milgram. The headline is **one theorem the website can state in a sentence**: *if every learned expert passes a checkable Lipschitz bound, the coupled iteration converges from any start, and its answer is within a stated distance of the classical one.* Its second half, *a learned expert inside defect correction can change how fast the answer comes and never what it is*, is already a theorem in this vault. What should **not** be attempted now is the continuous PDE theory (Schwarz convergence in Sobolev spaces, trace theorems), chart injectivity (Rado–Kneser–Choquet, Tutte) and universal approximation. Those are cited, not formalised. **No machine-checked domain-decomposition theory was found anywhere** ([[dd-neural-prior-art-2026]] §1.6), so even the classical parts are a first.

---

## 1. Four levels of "proved"

| level | what it means | the trust it earns |
|---|---|---|
| **cited** | a published theorem, with its source and hypotheses | the literature's |
| **proved (paper)** | a proof written here, of a statement stitched from known results | a careful reader's |
| **machine-checked** | Lean 4 on Mathlib, **no `sorry`**, `#print axioms` showing only Lean's three standard axioms | the Lean kernel's, whoever wrote the proof, Claude included |
| **certified hypothesis** | a numerical fact a theorem needs (a network's Lipschitz bound, a map's contraction constant), established by validated numerics outside Lean | the certifying tool's; the theorem then applies to that network |

**[AI Inference] on the main risk:** a machine check guarantees that the proof proves the **statement**, not that the statement says what was meant. So every Lean statement gets a plain-language restatement on its blueprint page, and the owner reviews the statements. Their background covers every Tier A and Tier B statement below.

---

## 2. The inventory

Tiers of Lean effort: **A** days · **B** weeks · **C** months · **D** not now. Novelty: **known** (a textbook or paper result) · **ours** (a new statement from known parts) · **open** (an unsolved problem).

### 2.1 The owner's item 4.1 — convergence with neural-operator experts

| # | statement | novelty | Lean |
|---|---|---|---|
| **T1** | **Perturbed contraction.** On a complete metric space, let $T$ have a fixed point $a^\star$, let $\tilde T$ be a $\rho$-contraction, $\rho<1$, and let $d(\tilde T a,Ta)\le\delta$ for all $a$. Then $\tilde T$ has a unique fixed point $\tilde a^\star$, $d(\tilde T^k a_0,\tilde a^\star)\le\rho^k d(a_0,\tilde a^\star)$, and $d(\tilde a^\star,a^\star)\le\delta/(1-\rho)$ | known (Banach plus one line) | **A.** `ContractingWith.fixedPoint`, `ContractingWith.dist_fixedPoint_le`, `ContractingWith.apriori_dist_iterate_fixedPoint_le`. **Confirmed 2026-10-01; and the distance bound is itself a Mathlib lemma**, `ContractingWith.dist_fixedPoint_fixedPoint_of_dist_le'`. The defect is needed only at $a^\star$, not for all $a$ |
| **T8** | **Wave variables make a passive agent non-expansive.** For a linear $\Lambda$ on a real inner-product space with $\langle\Lambda e,e\rangle\ge0$ and $Z>0$, $S=(I-Z\Lambda)(I+Z\Lambda)^{-1}$ exists and $\lVert S\rVert\le1$. If moreover $\langle\Lambda e,e\rangle\ge c\lVert e\rVert^2$ and $\lVert\Lambda\rVert\le M$, then $\lVert S\rVert^2\le1-\dfrac{4Zc}{(1+ZM)^2}$. *Proof:* with $y=(I+Z\Lambda)e$, $\lVert(I-Z\Lambda)e\rVert^2=\lVert y\rVert^2-4Z\langle\Lambda e,e\rangle$, and $\lVert y\rVert\le(1+ZM)\lVert e\rVert$ | known (the Cayley transform of an accretive operator) | **B**, finite dimensions first |
| **T9** | **The contraction contract** ([[chart-operator-architecture]] §4.4). With $\Pi$ an isometry that swaps each side's outgoing wave to its neighbour, and each learned scattering map $\tilde S_i$ a $\rho$-contraction on the fine space with defect $\delta$ from $S_i$, the two-level wave-variable iteration converges geometrically to within $\delta/(1-\rho)$ of the classical decomposition's answer. A corollary of T1 once the coarse correction is folded into the map | **ours**, as a statement about arbitrary certified experts. The exact-solver case is classical: Lions 1990; Després 1991; Collino, Ghanemi & Joly 2000 | **B** |
| **T9′** | **Non-expansive is enough without a rate.** If $\operatorname{Lip}(\Pi\tilde S)\le1$ in finite dimensions and a fixed point exists, the averaged iteration $a\leftarrow(1-\omega)a+\omega\,\Pi\tilde S(a)$ converges to one (Krasnoselskii–Mann) | known | **B**. Not in Mathlib `v4.34.1` (searched 2026-10-01), so it is proved from scratch |
| **T5** | **Defect correction** ([[defect-correction-learned-operator]] §2): **Theorem 1**, any limit of the iteration is a fixed point of the classical map $\Phi$, whatever the learned $\Psi$; **Corollary 1**, a constant $\Psi$ makes it the classical march; **Corollary 2**, shrinking toward the null element shifts $J_\Psi$'s spectrum to $\alpha+(1-\alpha)\sigma(J_\Psi)$ | known (Stetter 1978); the corollaries are this vault's | **A** (Theorem 1 is a continuity argument; Corollary 2 uses the spectral mapping for affine maps) |
| **T6** | **Defect correction's local rate** (Stetter's Theorem 2): $e_{k+1}=(I-J_\Psi^{-1}J_\Phi)e_k+o(\lVert e_k\rVert)$, so the rate is the spectral radius of $I-J_\Psi^{-1}J_\Phi$ | known | **C**: Ostrowski's local-convergence theorem through Fréchet derivatives and the spectral radius; Mathlib has the pieces (`HasFDerivAt`, `spectralRadius`, Gelfand's formula) but, as far as known, not Ostrowski |

### 2.2 Item 4.2 — can neural operators work as well as classical decomposition?

| # | statement | novelty | Lean |
|---|---|---|---|
| **T2** | **Existence.** Discretise the interface. For any $\varepsilon>0$ and any compact set of inputs, there is a neural network $\tilde S$ with $\lVert\tilde S-S\rVert_{C^1(K)}<\eta$, so that $\operatorname{Lip}(\Pi\tilde S)\le\rho_{\text{cl}}+\eta<1$ and T1 applies. **The learned iteration then converges at nearly the classical rate, to within $\varepsilon$ of the classical answer.** Uses universal approximation in $C^1$ (Hornik 1991; for operators, Kovachki, Lanthaler & Mishra, *JMLR* 22, 2021; Kovachki et al., *JMLR* 24, 2023) | **ours** (a composition of known theorems) | **A as a conditional theorem**: the approximation theorem is stated as a hypothesis, and the rest is checked. **D** for the approximation theorem itself. Stone–Weierstrass is in Mathlib, so a $C^0$ version is closer than a $C^1$ one |
| — | **What theory cannot say:** that a *trained* network meets T2's bound (that is measured and certified, level 4), or that it is *cheaper* (measured: S7). **"As well as" is a theorem about existence and about limits; "faster" is an experiment** | — | — |
| **T5, again** | **A learned expert inside defect correction returns exactly the classical answer** (Theorem 1), so in that slot "as accurate as classical" is not approximate but exact, and only the cost is in question | known | **A** |

### 2.3 Item 4.3 — unproved or unformalised holes in domain-decomposition theory

| # | statement | novelty | Lean | why it matters here |
|---|---|---|---|---|
| **T3** | **Fixed-point consistency.** For a linear discrete problem $Au=f$, every fixed point of restricted additive Schwarz or Dirichlet–Neumann built from exact local solves, with a partition of unity, solves $Au=f$; so does a direct Schur solve of the interface. For RAS: Frommer & Szyld, *SIAM J. Numer. Anal.* 39, 2001. **Corrected 2026-10-01: the restricted-additive-Schwarz part is false as written.** It holds when the sweep converges from every start, or when every window agrees with the blend; a three-unknown counterexample and a positive definite one are in [[formal-proofs-record]] §5.1. The Dirichlet–Neumann and Schur parts stand without any further hypothesis | known | **A–B**, finite-dimensional linear algebra | **it settles W348**: the compiler can admit iterated one-physics couplings citing a checked theorem, and the demo's fast examples stop showing a red `refuse` ([[demo-finish-plan]] §5) |
| **T12** | **Schwarz as alternating projections.** Alternating Schwarz with exact solves is a product of orthogonal projections in the energy inner product (Lions 1988). In finite dimensions $(P_2P_1)^n\to P_{V_1\cap V_2}$ (von Neumann), with rate set by the angle between the subspaces. The sharp rate of successive subspace correction is the **Xu–Zikatanov identity** (*J. Amer. Math. Soc.* 15, 2002) | known | **B** in finite dimensions; **C** for Xu–Zikatanov in Hilbert space | no formalisation of either was found; the natural first chapter of a formal DD library |
| **T13** | **Additive Schwarz's condition number from a stable decomposition** (the abstract Schwarz theory; Toselli & Widlund, *Domain Decomposition Methods*, Springer 2005, Ch. 2): $\kappa(P_{\text{ad}})\le C_0^2\,\omega\,(\rho(\mathcal E)+1)$, with $C_0$ the decomposition's stability constant, $\omega$ the local solvers' bound and $\rho(\mathcal E)\le N_c$, the number of colours | known | **B** | why a coarse space makes iteration counts independent of the number of windows |
| **T14** | **RAS converges for M-matrices in weighted max norms** (Frommer & Szyld 2001) | known | **C**: M-matrix theory is thin in Mathlib | the workbench's style B is RAS |
| **T17** | **Continuous Schwarz convergence**: overlapping alternating Schwarz in $H^1_0$ (Lions 1988); non-overlapping Robin (Lions 1990) | known | **C–D**. It needs Sobolev spaces and traces. The 2026 EllipticPDE library (Sobolev spaces, Poincaré, Lax–Milgram weak solutions, Rellich–Kondrachov) makes the overlapping case **thinkable** as a research project, not a first step | the full-strength statement behind T12 |
| **T15** | **Charts never fold**: Rado–Kneser–Choquet in 2-D; Tutte's and Floater's discrete injectivity (*Math. Comp.* 72, 2003); **false in 3-D** (Laugesen 1996) | known | **D** | cited by [[chart-operator-architecture]] §2.2 |
| **T20** | **Cross-points** in discrete optimized Schwarz: when energy estimates survive them, and which non-local exchange restores them (Gander & Kwok 2013; Claeys & Parolin, *Numer. Math.* 151, 2022) | **open** in general | **D** | the chart operator's named open part (§4.6 there) |

### 2.4 Item 4.4 — error decomposition, accumulation and boundedness

| # | statement | novelty | Lean |
|---|---|---|---|
| **T4** | **The master bound** ([[master-error-bound]]): the exact recursion $e^{n+1}=[\Phi(u^n)-\Phi(u^{n,\star})]+d^{n+1}$; the three-term split $d=\tau+\sigma+\gamma$, a telescoping identity; and $\lVert e^N\rVert\le L^N\lVert e^0\rVert+\sum_nL^{N-n}\lVert d^n\rVert$. Its three regimes ([[temporal-error-accumulation]]): bounded by $\delta/(1-L)$ when $L<1$, linear $N\delta$ when $L=1$, exponential when $L>1$ | known form; the split is this vault's | **A**, a discrete Gronwall by induction |
| **T22** | **The learned version.** T4 with T9: the per-step defect of a coupled step with learned experts adds $C\,\delta/(1-\rho)$, and an incomplete solve after $k$ sweeps adds $\rho^k$ times the initial interface error. So the rollout error is bounded by the four named sources, each measurable | **ours** | **A–B** once T4 and T9 exist |
| **T7** | **The a-posteriori certificate**: $\lVert w-w^\star\rVert\le r/(1-L)$ for a contraction, with $r=\lVert\Phi(w)-w\rVert$. **The formal statement makes W208's lesson explicit**: $L$ must be an operator-norm bound, not a ratio read off one approach, which is exactly the hypothesis that failed at twelve windows ([[defect-correction-learned-operator]] §3) | known | **A** (`ContractingWith.dist_fixedPoint_le`) |
| **T23** | **Step consistency**: if $\sup\lVert E_h-E_{h/2}\circ E_{h/2}\rVert\le\epsilon$ at every step size $h\le\Delta t$, and every $E_h$ is $(1+\beta h)$-Lipschitz, then $\bigl\lVert(E_{\Delta t/2^m})^{2^m}-E_{\Delta t}\bigr\rVert\le(2^m-1)\,\epsilon\,e^{\beta\Delta t}$. The bound is by telescoping over the $m$ halvings; it says the step-doubling defect must shrink with the step for a learned stepper to stay consistent at the coupling's cadence (S2) | known form | **A** |

### 2.5 Item 4.5 — other useful results

| # | statement | novelty | Lean | why |
|---|---|---|---|---|
| **T10** | **Symmetry averaging** ([[symmetry-averaging-atlas-0.1]]): for a **finite group** $G$ acting linearly, $\tilde E(u)=\lvert G\rvert^{-1}\sum_g g^{-1}E(gu)$ is $G$-equivariant for **any** $E$; and averaging over a set that is not a group need not be | known | **A** | exact equivariance for any learned map, a clean website item |
| **T11** | **Power-conserving interconnection of passive agents is passive**: typed ports with $e\cdot f$ = power and an ideal junction ($\sum e_if_i=0$) | known (port-Hamiltonian theory) | **A** | the port algebra's central property ([[port-algebra-atlas-0.1]]) |
| **T16** | **The pull-back is exact** (discrete form): the chart's reference-grid fluxes carry the same power as the physical ones; the Piola map preserves a zero divergence | known | **B** (Mathlib has change of variables and the divergence theorem on boxes) | [[chart-operator-architecture]] §2.3 |
| **T19** | **Certified Lipschitz bounds for trained experts**: bound propagation on the trained network, then T9 applied to *that* network | tooling | **B**, using or following **TorchLean** (arXiv 2602.22631) | turns T9 from "if" into "this expert" |
| **T21** | **Port typing and compiler rules as types and lemmas**: a connection that is not a power pair fails to typecheck; each compiler rule that cites a mathematical fact cites a Lean lemma | ours (engineering) | **B**, incrementally | the compiler's "every decision cites a rule" becomes "every rule cites a checked proof" |

---

## 3. The headline, stated for the website

> **Theorem (Atlas, learned experts).** Let the coupled system exchange wave variables through power-conjugate ports, with a coarse problem solved directly. If every learned expert's scattering map is certified to contract the fine interface space by a factor $\rho<1$ and differs from the classical solver's by at most $\delta$, then the coupled iteration converges geometrically at rate $\rho$ from any starting guess, and its result lies within $\delta/(1-\rho)$ of the classical decomposition's. Over $N$ time steps the error is bounded by the master bound with this term added.
>
> **Theorem (defect correction).** Inside defect correction, a learned expert can change how many classical steps the answer takes. It cannot change the answer.

T1 + T8 + T9 + T4 + T22 for the first, T5 for the second. All are Tier A or B.

---

## 4. The Lean project

- **Where:** a `lean/` folder in this repository to start, moved to a public repository when the website links to it ([[proposal-website-plan]]).
- **Toolchain:** Lean 4 through `elan`, Mathlib through Lake, with `lake exe cache get` for prebuilt Mathlib. **This downloads several gigabytes, so the owner is asked first.** It works on Windows.
- **The blueprint:** `leanblueprint` (Massot's tool, used by the PFR and Fermat's Last Theorem projects) renders a web page per theorem: the plain-language statement, the paper proof, the Lean declaration, and a **dependency graph coloured by status**. The graph is itself the website's picture of "mathematically secure". It is a Python package, so again the owner is asked first.
- **Continuous integration:** a GitHub Action builds the Lean project and the blueprint on every push, and fails on any `sorry`.
- **Each blueprint page links back** to the vault page whose result it formalises, and the vault page links forward.
- **Decided (O5 and O9, the owner, 2026-09-30):** the install is approved; the project stays private while it is built and goes public when the website links to it.
- **Built 2026-10-01.** The sources are in `lean/`. **The build runs outside OneDrive**, in `C:\Users\Nauni\.cache\atlas-lean`, through `scripts/lean_build.py`: built in place, OneDrive would upload the unpacked Mathlib cache, several gigabytes in tens of thousands of files. The install took 13.2 GB of disk. `leanblueprint` needed no separate Graphviz install. Details: [[formal-proofs-record]] §1.

### 4.1 The owner's concern: "Lean can take a long time to verify"

**Where Lean's time actually goes, and how this project keeps it short.** Estimates, **[AI Inference]**, to be replaced by the proofs chat's own timings:

| where the time goes | how long it takes | what the project does |
|---|---|---|
| **compiling Mathlib itself** | many hours on a laptop, if built from source | **never done.** `lake exe cache get` downloads Mathlib already compiled: a one-time download of a few gigabytes, minutes on a good connection. If the cache for the pinned Mathlib version is missing, the chat stops and pins a version that has one, rather than building from source |
| **the first `import Mathlib`** in a session | tens of seconds, loading compiled files | **import only the Mathlib files each theorem needs**, never the whole library, so every file loads faster |
| **checking this project's own proofs** | seconds to minutes per file for statements of this size | files kept small, one topic each, so `lake build` re-checks only what changed |
| **slow tactics**: broad `simp`, `nlinarith`, `decide` on large terms, `aesop` | occasionally minutes for one line | a style rule: a tactic call over ten seconds is replaced by an explicit proof step, and `set_option maxHeartbeats` is never raised to hide it |
| **continuous integration** | the same cache, on GitHub's runners | the Mathlib cache is fetched, not built, on every run |

**Every `lake build` is timed and written to the proof record, so the owner can see it.** Target: a full build of the project, with Mathlib cached, **under five minutes** on the owner's laptop. If a file breaks that target, it is split or its slow step is rewritten.

**Measured 2026-10-01, on battery** ([[formal-proofs-record]] §1.1), in place of the table's estimates:

- the one-time download and unpacking took 698 s;
- **a file costs about 26 s before its first proof is looked at** (10 s loading Mathlib's compiled files, 16 s running their start-up code), where the table guessed "tens of seconds" for a whole session. That cost is per file, and files that do not import each other are built at the same time;
- so the full build's time is set by the **longest chain of files that import each other**, and the rule added to the table's is: **keep that chain short**;
- the nine statement files of steps 1 and 2 built in 81.6 s.

---

## 5. Order

| step | theorems | about | what it unlocks |
|---|---|---|---|
| 1 | set-up; **T1**, **T7**, **T4** | days | the certificate and the master bound, checked |
| 2 | **T3** | days to a week | **W348**, and so the demo's fast examples ([[demo-finish-plan]] §5) |
| 3 | **T5**, **T10**, **T11** | days | the defect-correction headline; exact equivariance; the port algebra's passivity |
| 4 | **T8**, **T9**, **T9′**, **T22**, T2 (conditional) | weeks | **the headline theorem of §3** |
| 5 | the blueprint and CI, public | days | the website's link |
| 6 | **T12** (finite dimensions), **T13**, **T16**, **T21** | weeks | the start of a formal DD library |
| 7 | T19 on the demo's learned expert ([[demo-learned-case-plan]]) | weeks, after that expert exists | "this expert provably converges" |
| later | T6, T12 (Xu–Zikatanov), T14, T17 | months | research-grade formalisation |

**[AI Inference]:** steps 1–5 are the proposal's needs. Steps 6 and later are a contribution to formal mathematics in their own right, which a funder in scientific machine learning may value, but they should not delay the proposal.

---

## 6. Status

One line per theorem, kept by the proofs chat. *drafted*: the Lean statement exists and typechecks, its proof is a `sorry`. *approved*: the owner has accepted the statement. *checked*: no `sorry`, only Lean's three standard axioms, build timed. Statements, files and times are in [[formal-proofs-record]].

- **Set-up: done, 2026-10-01.** Lean 4.34.1, Mathlib `v4.34.1` from its cache, `leanblueprint`; names confirmed.
- **T1: drafted**, `Atlas.perturbed_contraction`, `lean/AtlasProofs/PerturbedContraction.lean`, 2026-10-01. Awaiting the owner's OK.
- **T7: drafted**, `Atlas.certificate` and `Atlas.certificate_constant_is_opNorm`, `lean/AtlasProofs/Certificate.lean`, 2026-10-01. Awaiting the owner's OK.
- **T4: drafted**, `Atlas.master_bound` and its regimes, `lean/AtlasProofs/MasterBound.lean`, 2026-10-01. Awaiting the owner's OK.
- **T3: drafted, with a finding**, 2026-10-01. Direct Schur (`Atlas.schur_iff`, `lean/AtlasProofs/Schur.lean`) and Dirichlet–Neumann (`Atlas.TwoPieces.dn_fixedPoint_solves`, `lean/AtlasProofs/DirichletNeumann.lean`) stand as planned. Restricted additive Schwarz needs a hypothesis (`Atlas.Schwarz.fixedPt_eq_solution`, `lean/AtlasProofs/SchwarzConvergence.lean`), and the counterexample is `Atlas.Schwarz.exists_spurious_fixedPt`. Awaiting the owner's OK. **Not yet checked: the compiler must go on citing the classical source.**
- **T5: drafted**, `Atlas.DefectCorrection.limit_isFixedPt` (Theorem 1), `step_of_const` (Corollary 1), `shrink_hasEigenvalue_iff` (Corollary 2), `lean/AtlasProofs/DefectCorrection.lean` and `DefectCorrectionShrink.lean`, 2026-10-01. Awaiting the owner's OK.
- **T10: drafted**, `Atlas.average_equivariant`, `lean/AtlasProofs/SymmetryAveraging.lean`, 2026-10-01. Awaiting the owner's OK.
- **T11: drafted**, `Atlas.interconnection_passive` and `Atlas.interconnection_nonexpansive`, `lean/AtlasProofs/Passivity.lean`, 2026-10-01. Awaiting the owner's OK.
- **T8, T9, T9′, T22, T2** (step 4): not started.
- **T24–T26, the scoping of T1, T8, T9 and T22 to "tier 2", and the SNI counterexample beside T1:** announced by the owner on 2026-10-01 as just added by the architecture chat. **Their text was not in the vault when this line was written** (no "For the proofs chat" heading in [[chart-operator-architecture]] yet), so nothing has been drafted from them.

---

## See Also

- [[formal-proofs-record]] — what is checked, each statement in plain words and in Lean, the build times, and the findings
- [[master-error-bound]] · [[defect-correction-learned-operator]] · [[temporal-error-accumulation]] · [[composition-error-theory]] — the vault's results being formalised
- [[chart-operator-architecture]] — §4, where T8–T9 come from
- [[atlas-and-standard-dd-theory]] — the classical theory T3 and T12–T17 belong to
- [[dd-neural-prior-art-2026]] — §1.6, the formal work next door
- [[mathematical-compendium]] — the vault's general mathematical reference
- [[gap-worklist]] — W348, which T3 settles
