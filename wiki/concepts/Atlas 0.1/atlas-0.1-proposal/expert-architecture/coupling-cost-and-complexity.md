# Coupling cost and complexity — traditional Schwarz, the wave-variable family and the superelement, measured with exact local solves

**Type:** Concept page — **measured cost model for the coupling contract** (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** written 2026-10-01 in the architecture chat, from `scripts/arch_coupling_cost.py` (run 2026-09-30 in a cloud container; output `out/arch/coupling_cost.txt` and `.json`, commit `ded7017`). **Every count and ratio below is measured, with exact classical local solves standing in for the experts.** Every statement about a learned expert's own forward cost is an estimate, marked **[AI Inference]**, with its anchor. Wall times are the container's, so only ratios are quoted as results.
**Hub:** [[00-proposal-workstreams]] · **Used by:** [[chart-operator-design-decisions]] (the coupling stack, decision 5) · [[chart-operator-architecture]] · [[chart-operator-training-and-cost]]

---

## 0. The answer, in one paragraph

**In the currency that matters for learned experts — how many times every expert must be called — the superelement needs about two calls per piece and no iteration, where traditional Schwarz needs hundreds to thousands.** On a steady 8 × 8 grid of pieces the traditional scheme (Dirichlet transmission, overlap, plain iteration: what SNI, NEST and L-DDM use) takes 1,864 rounds of local solves; version 0's wave exchange (optimized Robin, no overlap) 1,124; the same with GMRES 83; two levels with a coarse space 70. Rounds grow like the number of pieces for the one-level fixed points, like its 0.3 power with Krylov, and barely at all with a coarse space. Exact substructuring needs no iteration but $4n$ local solves per piece to build each port matrix, and **is a reorganised direct solve, of the same order as one**; truncating each side to $m$ cosine modes costs $4m$ solves per piece and a relative error of about $10^{-2}$ at $m=8$ and $10^{-3}$ to $10^{-4}$ at $m=16$. **The honest result: on one machine at these sizes, a monolithic sparse direct solve beats every decomposed method**, including the best (0.036 of the traditional scheme's time against 0.070 for the best iterative coupling), as the vault's showcase record and Mao and Fan (2025) also found. So the superelement's case is not "faster than a direct solver on a fixed 2-D geometry"; it is **fewer expert calls by two to three orders of magnitude, where the classical work cannot be reused or scaled** — changing geometry or material, expensive or nonlinear local physics, and three dimensions.

---

## 1. What was measured, and why it is the best case for a learned expert

**Intuition.** A learned expert's cost is dominated by how often it is called. A coupling scheme that needs a thousand rounds calls every expert a thousand times; one that needs two calls it twice. Replacing the experts by **exact classical solves** removes the question of the expert's accuracy and leaves exactly that count, plus the cost of everything around the expert (assembly, coarse solves, interface factorisations). No learned expert can converge in fewer rounds than its exact counterpart under the same scheme, so these counts are a **floor**.

**The problem** is the one the charts produce in reference space ([[chart-operator-design-decisions]], decisions b1–b3): a grid of $P_x\times P_y$ square pieces, each $n\times n$ cells, cell-centred finite volumes on a unit grid with five points, a smooth log-normal coefficient $\kappa$ (face conductance the harmonic mean), zero Dirichlet data on the outer boundary and a smooth source:

$$-\nabla\!\cdot(\kappa\nabla u)+\frac{1}{\mathrm{Fo}_h}\,u=f,\qquad \mathrm{Fo}_h=\mathrm{Fo}\,n^2,\qquad \mathrm{Fo}=\frac{\kappa\,\Delta t}{L^2},$$

steady ($1/\mathrm{Fo}_h=0$) and implicit steps at Fourier numbers $\mathrm{Fo}=1,\ 0.1,\ 0.01$ per piece of side $L$.

**The currency.** One *round* is one local solve on every piece. Iterations are counted to a **true** relative residual $\lVert b-Ax\rVert/\lVert b\rVert<10^{-8}$.

| strategy | what it is | the vault's name |
|---|---|---|
| **RAS-Rich** | restricted additive Schwarz, Dirichlet transmission, overlap 2 cells, plain (Richardson) iteration | the **traditional** scheme of SNI, NEST, L-DDM ([[dd-neural-prior-art-2026]]) |
| RAS-GMRES | the same preconditioner inside right-preconditioned GMRES | — |
| **OSM-Rich** | non-overlapping optimized Schwarz (Lions' Robin exchange) on the interface faces: each piece solves with Robin data $p\lambda-q=g$ and hands its neighbour $p\lambda+q$ | **version 0's wave-variable exchange** ([[chart-operator-architecture]] §4) |
| OSM-AA | the same fixed point with Anderson acceleration, depth 5 | tier 2, black-box experts |
| OSM-GMRES | GMRES on the same affine interface fixed point | tier 2 with JVPs |
| ORAS-GMRES | optimized RAS with zero overlap as a GMRES preconditioner | — |
| **2L-GMRES** | that preconditioner plus an additive coarse correction on $\{1,x,y,xy\}$ per piece | tier 2 with a coarse space |
| **Schur** | exact direct substructuring, hybridised on the interface faces: each piece's port matrix $S_i=D_i-B_i^{\top}A_i^{-1}B_i$ (its Dirichlet-to-Neumann map) formed by one local solve per face, assembled, factorised, solved | the probed-DtN direct Schur of [[probed-dtn-coupling]] |
| **Port-$m$** | the same with each interface edge reduced to its first $m$ cosine modes, $S^{\text{red}}=Q^{\top}SQ$ | **the ideal superelement**: the matrix a learned expert would supply |

Two practical findings from building the probe, kept because they would bite a build:
- **Optimized RAS with zero overlap diverges as a plain iteration.** It equals the optimized Schwarz method only with algebraic overlap of at least one layer (St-Cyr, Gander & Thomas, *SIAM J. Sci. Comput.* 29, 2007). It is a valid GMRES preconditioner; the genuine non-overlapping Robin exchange (OSM) is what converges as a fixed point.
- **The zeroth-order optimum $p^\star=\kappa\sqrt{k_{\text{lo}}k_{\text{hi}}}$ was too stiff by a factor of two** on these pieces ($k_{\text{lo}}=\sqrt{(\pi/n)^2+s}$, $k_{\text{hi}}=\sqrt{\pi^2+s}$, $s=1/\mathrm{Fo}_h$). Tuned once on the smallest case to $p=p^\star/2$ (76 rounds against 117 at $p^\star$ on 2 × 2 pieces of $16^2$), and held fixed everywhere else.

---

## 2. Measured: rounds of local solves

| pieces | $n$ | Fo | RAS-Rich | RAS-GMRES | OSM-Rich | OSM-AA | OSM-GMRES | ORAS-GMRES | 2L-GMRES | **Schur** solves/piece | **Port-4** | **Port-8** | **Port-16** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 × 1 | 32 | steady | 43 | 16 | 63 | 34 | 32 | 33 | 33 | 50 | 8 | 14 | 26 |
| 8 × 1 | 32 | steady | 47 | 20 | 71 | 37 | 35 | 43 | 44 | 58 | 9 | 16 | 30 |
| 16 × 1 | 32 | steady | 66 | 25 | 159 | 68 | 47 | 52 | 53 | 62 | 10 | 17 | 32 |
| 2 × 2 | 32 | steady | 118 | 21 | 87 | 39 | 37 | 41 | 39 | 66 | 10 | 18 | 34 |
| 4 × 4 | 32 | steady | 433 | 42 | 296 | 94 | 53 | 94 | 58 | 98 | 14 | 26 | 50 |
| **8 × 8** | 32 | steady | **1,864** | 81 | 1,124 | 152 | **83** | 188 | **70** | 114 | 16 | 30 | 58 |
| 4 × 4 | 64 | steady | 1,013 | 65 | 425 | 115 | 64 | 135 | 83 | 194 | 14 | 26 | 50 |
| 4 × 4 | 32 | 1 | 288 | 39 | 193 | 70 | 60 | 87 | 57 | 98 | 14 | 26 | 50 |
| 4 × 4 | 32 | 0.1 | 81 | 26 | 135 | 60 | 50 | 57 | 48 | 98 | 14 | 26 | 50 |
| 4 × 4 | 32 | 0.01 | 22 | 13 | 51 | 30 | 34 | 33 | 33 | 98 | 14 | 26 | 50 |
| 8 × 8 | 32 | 1 | 441 | 58 | 283 | 88 | 61 | 138 | 65 | 114 | 16 | 30 | 58 |
| 8 × 8 | 32 | 0.1 | 84 | 29 | 107 | 54 | 47 | 69 | 55 | 114 | 16 | 30 | 58 |
| 8 × 8 | 32 | 0.01 | 27 | 15 | 69 | 36 | 40 | 37 | 37 | 114 | 16 | 30 | 58 |

The last four columns are **local solves per piece**, not rounds: the solves that form the port matrices (one per interface face for Schur, one per mode for Port-$m$), plus one for the particular solution and one to reconstruct. **A learned superelement would replace them by about two expert calls per piece** — one that returns its port data, one (or a linear combination) that reconstructs — **[AI Inference]**, the design of [[chart-operator-design-decisions]] decision 6, not measured.

**The port truncation error** (relative $L^2$ against the exact solution; interface unknowns in brackets for 8 × 8):

| modes per edge $m$ | 4 | 8 | 16 | all ($n=32$) |
|---|---|---|---|---|
| steady, all cases | 1.4–5.9 % | 0.22–1.05 % | **0.017–0.14 %** | $10^{-14}$ |
| 8 × 8 steady | 1.35 % (448) | 0.22 % (896) | 0.017 % (1,792) | (3,584) |

The error falls about tenfold per doubling of $m$. At $m=16$ it sits below the accuracy a learned expert is likely to reach, so it would not be the limiting term.

---

## 3. Measured: how the rounds scale

From the steady rows, the exponents of the rounds in the number of pieces $P$ (from 2 × 2 to 4 × 4 to 8 × 8) and in the cells per piece side $n$ (32 to 64 at 4 × 4):

| strategy | $\propto P^{a}$ | $\propto n^{b}$ | reading |
|---|---|---|---|
| RAS-Rich | $a=0.94,\ 1.05$ | $b=1.23$ | the one-level bound $\kappa\lesssim H^{-2}(1+H/\delta)$: the global low mode contracts by $1-O(1/D^2)$ per round, $D$ the diameter of the piece graph; a fixed overlap in cells shrinks against a growing piece |
| OSM-Rich | $a=0.88,\ 0.96$ | $b=0.52$ | the optimized Robin rate $1-O(h^{1/2})$ removes most of the $n$ growth (Gander, *SIAM J. Numer. Anal.* 44, 2006), not the $P$ growth |
| OSM-GMRES | $a=0.26,\ 0.32$ | $b=0.27$ | Krylov: a square root of the condition, still diameter-bound |
| RAS-GMRES | $a=0.50,\ 0.47$ | $b=0.63$ | the same |
| 2L-GMRES | $a=0.29,\ 0.14$ | $b=0.52$ | the coarse space carries the global modes: nearly flat in $P$ |
| Schur, Port-$m$ | — | — | no iteration; cost moves to forming and factorising the interface system |

**The diameter bound, said plainly.** Any scheme in which a piece hears only from its neighbours needs at least $D$ rounds to carry information across a domain $D$ pieces wide, for a steady (elliptic) problem. Krylov acceleration does not escape it; only a coarse space or a direct interface solve does. **Small time steps do escape it physically**: an implicit step reaches only a diffusion length $\sqrt{\kappa\Delta t}$, so at $\mathrm{Fo}=0.01$ every scheme needs 13–40 rounds whatever the number of pieces.

---

## 4. Asymptotic complexity, two dimensions

$P$ pieces in a $\sqrt P\times\sqrt P$ grid, $n\times n$ cells each, $N=Pn^2$ unknowns, $m$ port modes per side, overlap $\delta$ cells, $C_\ell$ one classical local solve, $C_E$ one expert call.

| method | rounds $K$ | cost of one solve |
|---|---|---|
| traditional one-level Schwarz | $O\!\left(P\,n/\delta\right)$ | $K\,P\,C_\ell$ |
| optimized Robin, one level | $O\!\left(P\,n^{1/2}\right)$ | $K\,P\,C_\ell$ |
| + Krylov or Anderson | $O\!\left(P^{1/2}\right)$ up to logarithms | $K\,P\,C_\ell$ plus orthogonalisation $O(K^2N)$ |
| + coarse space | $O(1)$ in $P$ | $K\,(P\,C_\ell+C_0)$, $C_0$ a coarse solve of size $O(P)$ |
| exact substructuring | 0 | $4n\,P\,C_\ell$ to form, then an interface system of $O(Pn)$ unknowns whose nested-dissection factorisation is $O\!\left((\sqrt P\,n)^3\right)=O\!\left(P^{3/2}n^3\right)$ — **the order of the monolithic direct solve $O(N^{3/2})$**: it *is* a nested dissection, reordered |
| **superelement, $m$ modes** | 0 | $\approx 2P\,C_E$ plus an interface factorisation $O\!\left(P^{3/2}m^3\right)$, which is $(n/m)^3$ cheaper than the exact one |

**With a fixed step and fixed geometry the port matrices are reused**: each further step costs about one or two expert calls per piece (the particular response and the reconstruction) and a back-substitution on the factorised interface, $O(Pm\log P)$-ish. The iterative schemes pay their rounds again every step.

**Three dimensions, [AI Inference].** A monolithic sparse direct solve costs $O(N^2)$ flops and $O(N^{4/3})$ memory in 3-D, which is what ends it at scale. The superelement's interface carries $m^2$ modes per face, and a direct factorisation of that system grows like $O(P^2m^6)$, so tier 0 in 3-D should be **preconditioned conjugate gradients on the assembled, symmetric positive definite port system** with a coarse space of the BDDC/FETI-DP kind, whose iteration counts grow only polylogarithmically. Untested here.

---

## 5. Measured: wall time, as ratios

Setup (every factorisation) plus solve, as a ratio to the traditional scheme's, with the monolithic sparse LU (factor and solve) as the classical reference. One local solve cost **0.10–0.20 ms at $n=32$ and 0.85 ms at $n=64$** in this container (Python included).

| pieces | $n$ | Fo | RAS-Rich (s) | RAS-GMRES | OSM-Rich | OSM-AA | OSM-GMRES | ORAS-GMRES | 2L-GMRES | Schur | Port-8 | **monolithic LU** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 × 4 | 32 | steady | 0.94 | 0.38 | 0.80 | 0.35 | 0.21 | 0.72 | 0.45 | 1.05 | 0.35 | **0.10** |
| 8 × 8 | 32 | steady | 22.3 | 0.13 | 0.65 | 0.12 | 0.070 | 0.39 | 0.16 | 0.18 | 0.12 | **0.036** |
| 4 × 4 | 64 | steady | 14.1 | 0.13 | 0.37 | 0.12 | 0.064 | 0.33 | 0.18 | 0.21 | 0.11 | **0.029** |
| 8 × 8 | 32 | 1 | 5.49 | 0.29 | 0.77 | 0.31 | 0.23 | 0.86 | 0.52 | 1.03 | 0.51 | **0.075** |
| 8 × 8 | 32 | 0.1 | 1.42 | 0.60 | 1.16 | 0.64 | 0.59 | 1.34 | 1.89 | 2.24 | 1.57 | **0.24** |
| 8 × 8 | 32 | 0.01 | 0.69 | 0.95 | 1.69 | 1.07 | 1.01 | 2.06 | 3.99 | 5.39 | 3.41 | **0.51** |

The full table, all 13 cases, is in `out/arch/coupling_cost.txt`. Three readings:
1. **The monolithic direct solve wins every row.** At 2-D sizes up to $6.5\times10^4$ unknowns a single sparse LU is faster than any decomposition on one machine. That is the vault's record at showcase sizes ([[demo-fast-examples-plan]]) and Mao and Fan's (their learned two-level solver is about twice as slow as PARDISO on 80 cores).
2. **For small steps the superelement's one-off cost loses**, 3.4 times the traditional scheme's at $\mathrm{Fo}=0.01$, because forming $4m$ responses per piece outweighs 15–40 cheap rounds. With the port matrices reused over many steps it falls to about two solves per piece per step. **Caveat:** the probe forms the reduced port matrices through dense basis slices, so its forming times are upper bounds; the solve counts of §2 are the intrinsic measure.
3. **For steady problems on many pieces** the best iterative couplings (GMRES, a coarse space) and the superelement are 6–15 times faster than the traditional scheme, and within a factor of 2–5 of the direct solve.

---

## 6. Projected for learned experts — estimates

**[AI Inference] throughout; none of this is measured.** A learned expert's call costs far more than a classical local solve with a stored factorisation:

| anchor | value | source |
|---|---|---|
| a classical local solve, $n=32$ / $64$, stored factorisation | 0.10–0.20 ms / 0.85 ms | **measured**, this probe |
| Poseidon-T, one agent, one step, on the owner's CPU | 86.1 ms | **measured** ([[outcome-c3-learned-speed-at-scale]] §2) |
| SM-FNO subdomain solver, $64^2$ | 10.2 M parameters, 1.43 GFLOP | SNAP-DDM (Mao et al., ICML 2024) |
| optimal learned subdomain preconditioner | about 1.5 M weights | Mao & Fan (2025) |
| a 1–5 M-parameter convolutional expert at $32^2$–$64^2$ | about 1–5 GFLOP: **10–100 ms on a laptop CPU, under 1 ms per piece batched on a GPU** | estimate from the above |

So an expert call is **50–500 times** a classical local solve on a CPU. For 8 × 8 pieces, steady:

| coupling | expert calls | estimated laptop-CPU time |
|---|---|---|
| traditional Schwarz | $1{,}864\times64\approx1.2\times10^5$ | 20–200 minutes |
| two-level + GMRES | $70\times64\approx4{,}500$ | 45 s – 7.5 min |
| **superelement** | $\approx2\times64=128$ | **1.3–13 s**; well under a second batched on a GPU |
| classical monolithic LU (measured) | — | 0.8 s in this container |

**Where the superelement earns its place, [AI Inference]:**
- **Geometry, material or parameters change on every call** (design loops, uncertainty sampling), so no factorisation or classical constraint mode can be reused. Recomputing a piece's classical port matrix costs a factorisation plus $4m$ solves; the learned one costs one forward call.
- **The local physics is expensive**: a flow window that needs a march, a nonlinear or reacting piece, where $C_\ell$ is seconds and $C_E$ milliseconds.
- **Three dimensions at scale**, where the monolithic direct solve stops being available at all (§4).

**Where it does not:** a fixed 2-D geometry with linear physics, where classical static condensation with stored constraint modes, or a monolithic direct solve, is cheaper than any network. This is S7 of [[outcome-c5-requirements-for-dd-native-experts]], and the proposal must say it.

---

## 7. What this page does not establish

1. **No network was run.** Every learned cost is an estimate from the anchors in §6.
2. **Two dimensions, one linear family**, square pieces in reference space with a smooth coefficient. The workbench's own charted pieces add anisotropy through $\mu$ ([[chart-operator-design-decisions]] b1), which the probe does not include.
3. **One machine.** The ranking of a direct solve against a decomposition changes on a cluster, where the decomposition's parallelism counts and a monolithic factorisation's memory does not fit.
4. **Cross-points do not arise** in the face-hybridised formulation used for OSM and Schur (interfaces carry face values, and four pieces meeting at a vertex share no face). The vertex-based formulations of [[atlas-and-standard-dd-theory]] §3 would meet them.
5. **The Robin coefficient was tuned once** (to half the textbook optimum). A per-mode impedance, the self-describing ports of [[chart-operator-design-decisions]], is not measured.

---

## 8. Reproduce

```
set PYTHONIOENCODING=utf-8
python scripts/arch_coupling_cost.py
```

About four minutes in the cloud container (numpy, scipy). Writes `out/arch/coupling_cost.txt` and `out/arch/coupling_cost.json`. Not a timing of the workbench; it imports nothing from `atlas/`.

---

## See Also

- [[chart-operator-design-decisions]] — the decisions this page feeds: the coupling stack, the tier-2 accelerator, the transient policy, the superelement's output format
- [[chart-operator-architecture]] — version 0's wave exchange and contraction contract (§4), which OSM-Rich measures
- [[probed-dtn-coupling]] — the probed Schur complement, measured at 44 calls against 594 on the wind farm's seam
- [[atlas-and-standard-dd-theory]] — the one-level bound, coarse spaces, and why they apply here
- [[chart-operator-training-and-cost]] — the cost model this page gives measured inputs to
- [[outcome-c5-requirements-for-dd-native-experts]] — S7, the bar a learned expert must clear
