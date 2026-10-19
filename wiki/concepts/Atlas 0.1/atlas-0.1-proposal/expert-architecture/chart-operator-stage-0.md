# Chart operator Stage 0 — the first learned expert (CO-D), built, registered and judged

**Type:** Concept page — **build record and pre-registration** (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** opened, built, trained and judged on 2026-10-18 (W355). The owner chose the gates and comparison bars of §3, and they were registered in code (`atlas/chartop/gates.py`, hash in `out/chartop/registered.txt`) **before any training run**. E4's maximum-error bar was re-registered the same day (D9), before any training result had been read. **The one evaluation (§9):** the coupling's guarantees held to rounding (E3, SE1–SE3, DUAL), and the network's accuracy did not (E1, E4 and KT not met). E5 found no arm faster at comparable accuracy (§10). Of S1–S12, seven are met and four are not (§11). The modes did not learn in any of the eleven arms. After the evaluation, the dominant failure on the benchmarks was traced to the hat modes' broken sum (§9.7).
**Hub:** [[chart-operator-architecture]] · **Plan it executes:** [[chart-operator-training-and-cost]] · **Decisions it extends:** [[chart-operator-design-decisions]] · **Family:** [[chart-operator-family-catalog]] (CO-D) · [[chart-operator-general-form]] (G0, and §2.4's frozen-coefficient route)

---

## 1. The owner's decisions for Stage 0 (2026-10-18)

| # | question | decided |
|---|---|---|
| D1 | where the code lives | `atlas/chartop/` in this repository; tests `tests/test_chartop_*.py`, scripts `scripts/chartop_*.py`, records `out/chartop/` |
| D2 | the numerical bars | as proposed: §3 |
| D3 | the default port basis (catalog F2) | corner hats plus side sines; per-side cosines are comparison C6 |
| D4 | precision (catalog F6) | a bfloat16 network on the GPU, float32 on the laptop's CPU, every Gram and every loss in float64 |
| D5 | comparison C7 (catalog F7) | registered |
| D6 | the rented-GPU ceiling | \$15 for the whole stage, each box at most \$0.70 an hour |
| D7 | the piece solver | **bilinear finite elements** on the reference grid's vertices, not two-point-flux finite volumes |
| D8 | the main run's objective | the label-free energy objective (T26); labels for validation, the held-out test and C2 only |
| D9 | E4's maximum-error bar (re-registered the same day, before any training result was seen) | the 16-mode classical superelement's own maximum error plus $0.01$ of the range, on every problem |

**Why D9.** As proposed under D2, E4's maximum-error bar was 1% of the range on every problem. With exact classical modes, the 16-mode superelement itself misses that on the s-channel step benchmark. The one-step boundary layer running from the hot end along the end pieces' insulated walls is steeper than 15 side sines can follow, so the maximum error is 2.81% of the range, though only 0.135% in the energy norm (`scripts/chartop_validate_host.py`, `out/chartop/host_validation.json`). No network could have met the bar there. This should have been checked when the bars were proposed. The registration was withdrawn in a commit stating the reason, and registered again with the owner's choice. At that point arm B had trained for about ten minutes and no validation or test number had been read.

**Why D7.** The pulled-back tensor of a shearing chart is $\hat\kappa\,J G^{-1}=\frac{\hat\kappa}{1-|\mu|^2}\begin{pmatrix}|1-\mu|^2&-2\operatorname{Im}\mu\\-2\operatorname{Im}\mu&|1+\mu|^2\end{pmatrix}$, and a two-point flux discards its off-diagonal entry. Winslow charts, which the chart rule uses at every convex corner that is not a right angle, shear. Bilinear elements carry the full tensor element by element. On the reference grid they are exactly the isoparametric bilinear elements of the physical mesh the chart draws, so Proposition 3's certificate is that mesh's validity condition. They are conforming, which is Theorem 2's setting. T24–T26 hold unchanged, since they assume only $M\succeq0$ and $A_I\succ0$.

---

## 2. What is being judged

**Intuition.** The network is asked one question per piece: given values on the four sides of the unit square, what is inside? It answers with 64 constraint modes and one particular field. The host measures their energy and couples the pieces. Theorem 1 makes the coupling safe whatever the answer, so the gates ask how *good* the answer is, piece by piece (E1) and coupled (E3, E4). They also ask what it costs (E5), and whether the 16 port modes per side, rather than the network, are the limit (SE3).

**Rigorously.** For one piece with energy matrix $M=\begin{pmatrix}A_I&-B\\-B^\top&D\end{pmatrix}$, port basis $Q$, exact modes $H=A_I^{-1}BQ$ and particular field $u_p=A_I^{-1}b_I$:
$$e_H=\Bigl(\frac{\sum_k\lVert\hat H_k-H_k\rVert_{A_I}^2}{\operatorname{tr}\Lambda}\Bigr)^{1/2}=\Bigl(\frac{\operatorname{tr}\tilde\Lambda-\operatorname{tr}\Lambda}{\operatorname{tr}\Lambda}\Bigr)^{1/2},\qquad e_p=\frac{\lVert\hat u_p-u_p\rVert_{A_I}}{\lVert u_p\rVert_{A_I}},$$
the second equality being T25 (ii). On a benchmark, $u$ is the full-domain solve in the same discretisation (every piece's grid merged into one mesh) and $\tilde u$ the coupled tier-0 field. Theorem 2 then reads
$$\lVert u-\tilde u\rVert_M\le\lVert u-u_m\rVert_M+\Bigl(\sum_i\lVert e_{p,i}+E_iR_ic^\star\rVert_{A_{I,i}}^2\Bigr)^{1/2},$$
with $u_m$ the classical superelement with the same 16 modes per side. The bracket is the **network term**.

---

## 3. The registered bars

Chosen by the owner on 2026-10-18 (D2), before any training run. `tests/test_chartop_gates.py` checks this table against `atlas/chartop/gates.py`.

| gate | registered bar |
|---|---|
| E1 | on the held-out test split, in every bin of $\sup\lvert\mu\rvert$ and input frequency with at least 30 examples: median $e_H\le0.01$ and 90th percentile $\le0.03$; median $e_p\le0.02$ and 90th percentile $\le0.05$ |
| E3 | on every benchmark problem, $\lVert u-\tilde u\rVert_M$ within Theorem 2's bound evaluated with the measured field errors, to a relative rounding of $10^{-9}$ |
| E4 | on every benchmark problem, energy-norm error $\le0.02$, and maximum error at most the 16-mode classical superelement's own maximum error plus $0.01$ of the solution's range (D9) |
| E5 | reported whatever it says; "faster" only where measured faster |
| SE1 | Gram identity residual $\le10^{-10}$ relative, and $\lambda_{\min}(\tilde{\mathsf S})\ge\lambda_{\min}(\mathsf S)(1-10^{-10})$, on every benchmark problem |
| SE3 | on every benchmark problem, the 16-mode truncation error $\lVert u-u_m\rVert_M$ below the network term |
| DUAL | on steady pieces, every difference of the chain $\tilde\Lambda_{\text{low}}\preceq\Lambda_{\text{low}}\preceq\Lambda\preceq\tilde\Lambda$ has smallest eigenvalue $\ge-10^{-10}\lvert\Lambda\rvert$; the gap against the true error reported |
| KT | $\kappa(T)$ on every steady benchmark: the frozen-coefficient iteration converges within 50 outer iterations and meets E4's bars against the classical frozen-coefficient solution, its floor being the same iteration with the 16-mode classical superelement |
| C1–C7 | ratio = the alternative arm's median $e_H$ over the default arm's (C6: their mean E4 energy errors); the alternative wins at a ratio $\le0.8$, loses at $\ge1.25$, otherwise a tie and the default stands |

**SE3, read correctly.** SE3 does not ask the network to beat the truncation floor. It asks the opposite: that 16 modes per side are enough, so that the network, not the port count, is the limiting term.

**The bins of E1.** $\sup\lvert\mu\rvert$ in $[0,0.15)$, $[0.15,0.30)$, $[0.30,0.45)$ and $[0.45,0.65]$. Input frequency is the correlation length $\ell$ of the material and source fields: low ($\ell=0.5$), mid ($0.25$), high ($0.125$).

---

## 4. What was built (`atlas/chartop/`)

### 4.1 The piece's energy: bilinear elements on the reference grid (D7)

**Intuition.** Every piece becomes the unit square. Its curvature, its stretching and its material become one tensor field on the square. The square is cut into $n\times n$ cells. The unknowns are the temperatures at the $(n+1)^2$ cell corners, and the energy of a field is computed exactly for bilinear interpolation between those corners.

**Rigorously.** With $\hat{\mathsf A}=\hat\kappa\,JG^{-1}$ constant on each element (its value at the element's centre), the reaction $\sigma=\tilde J/\mathrm{Fo}$ of a backward-Euler step ($\tilde J=J/\lvert\Omega\rvert$, $\mathrm{Fo}=\kappa_{\text{ref}}\Delta t/(\rho c\lvert\Omega\rvert)$), and a nodal field $u$,
$$u^\top Mu=\sum_e\int_e\bigl(\hat\nabla u\cdot\hat{\mathsf A}\hat\nabla u+\sigma u^2\bigr)d\hat x,\qquad M=\begin{pmatrix}A_I&-B\\-B^\top&D\end{pmatrix},$$
where $I$ are the $(n-1)^2$ interior nodes and the boundary block holds the $4n$ loop nodes. The load of a source $g$ is the mass matrix weighted by $\tilde J$ applied to $g$. The physical energy is $\kappa_{\text{ref}}$ times this one. On the reference grid these are exactly the isoparametric bilinear elements of the physical mesh the chart draws. `fe.py`; the closed-form element energy is checked against the assembled matrix to $3\times10^{-15}$ (`tests/test_chartop_core.py`).

### 4.2 Ports: corner hats and side sines (D3)

Four hats, each linear along the two sides meeting at its corner, and per side $\sin(j\pi t)$ for $j=1,\dots,15$: $4+4\cdot15=64=4m$ columns. Every trace is continuous round the loop. **The coefficients are vertex values and side amplitudes.** A corner shared by any number of pieces (a cross-point, such as insert-round's box corners, where three pieces meet) is one vertex unknown, and the host needs no special rule for it. The constant is the sum of the hats, so a uniform temperature is represented exactly. A neighbour traverses a shared side backwards, and $\sin(j\pi(1-t))=(-1)^{j+1}\sin(j\pi t)$ gives the sign. Comparison C6's cosines $\cos(j\pi t)$, $j=0,\dots,15$, give each corner node the average of its two sides' values. That arm is nonconforming at corners: it is the bilinear-element form of the document's Remark 4. `ports.py`.

### 4.3 The principal part by fast sine transforms

On the plain square, $A_I^0=K_1\otimes M_1+M_1\otimes K_1+s_0\,M_1\otimes M_1$, with $K_1=n\,\mathrm{tridiag}(-1,2,-1)$ and $M_1=\tfrac1{6n}\mathrm{tridiag}(1,4,1)$. The type-I sine transform diagonalises both 1-D matrices, with eigenvalues $k_p=2n(1-\cos\tfrac{p\pi}n)$ and $m_p=\tfrac{4+2\cos(p\pi/n)}{6n}$. So $H^{(0)}=(A_I^0)^{-1}B^0Q$ costs two transforms and a division. It matches a sparse LU to $10^{-14}$ (`principal.py`, tests). Inside the network it is computed in float32. Its rounding is then part of the network's output, an interior error like any other, so T25 still applies. This cut one training step from 83 to 63 ms on the RTX 3090.

### 4.4 Charts, the certificate and the envelope

`chart.py` makes four kinds of chart: the Coons patch; Winslow, the harmonic chart with the sides' parametrisations as data, solved in its forward form by Picard iteration from the patch; the harmonic chart with a conformal boundary correspondence, which *is* the conformal map of a piece of given modulus; and closed-form maps. `certificate.py` holds Proposition 3: every cell corner's bilinear Jacobian positive and the boundary polygon simple and positively oriented. It also holds the envelope declared on the card: $\sup\lvert\mu\rvert\le0.65$, $\max\log J-\min\log J\le\log8$, $\lVert\hat\nabla\log J\rVert_\infty+\lVert\hat\nabla\mu\rVert_\infty\le40$, $\lvert\log\hat\kappa\rvert\le2.3$, and $\mathrm{Fo}\in[10^{-2},10^{2}]$ or steady. A refusal names the quantity it violates.

**The chart rule, at work on the s-channel's end piece.** Its corners are $154.0^\circ$ and $105.2^\circ$ at the channel's straight end, and two right angles at the cut. The conformal chart was tried first. Its $\log J$ range of $4.61$ exceeds $\log8=2.08$, so it was refused, and Winslow was used: $\sup\lvert\mu\rvert=0.543$ on the grid, $\log J$ range $1.43$. Proposition 2 puts $\lvert\mu\rvert\ge(\tan77^\circ-1)/(\tan77^\circ+1)=0.625$ *at* the $154^\circ$ corner for any $C^1$ chart. The grid's samples stop half a cell away, so the envelope's $k_\mu=0.65$ leaves the corner itself room.

### 4.5 The host

`host.py`: the Gram $\tilde\Lambda_i=\tilde V_i^\top M_i\tilde V_i$ and the load $\tilde b_i=\tilde V_i^\top b_i-\tilde V_i^\top M_i(\hat u_{p,i},0)$, assembled over vertex and side unknowns. The Dirichlet sides' coefficients are fixed by projection. Then one Cholesky solve and the reconstruction $\tilde u_i=\hat u_{p,i}+\hat H_iR_ic$, with the THERM port type checked on every seam. A seam whose nodes do not coincide is a **mortar**: the second piece's side amplitudes are the $L^2$ projection, in physical arc length, of the owner's trace minus its own vertex interpolant. It is nonconforming and not covered by Theorem 2, but tested: 16 elements against 24 couple to within 0.008 K of the conforming answer on a 100 K span. The **full-domain reference** merges the pieces' coincident nodes into one mesh and solves it once.

### 4.6 The benchmarks as conforming layouts (`benchmarks.py`)

The general layout for forks and rings is not built, so each benchmark has a hand-made layout of four-sided pieces in which every seam is conforming:
- **bend-3**: 15 pieces, $3\times5$, cut at equal values of $\log r$ and $\theta$ and charted by the log-polar map.
- **s-channel**: the layout's harmonic coordinates are solved on a fine Coons mesh of the drawn channel, giving modulus $M=8.61$ and $E(u)E(v)=1.0005$. That gives 9 pieces cut at equal values of $u$: 7 conformal and 2 Winslow (above).
- **insert-round**: an O–H grid with every ring cut in three along its rays, to keep each piece's modulus near one (decision 2). That is 9 core rectangles, 12 disc-ring pieces (corners 135/105/90/75°), 12 plate-ring pieces (corners 90/45/75/105°) and 6 column pieces, 39 in all, copper inside the circle and steel outside.

Each comes **steady**, as the workbench defines it, and as **one backward-Euler step** from 300 K: $\Delta t$ = 50, 100 and 100 s, Fourier numbers 0.033–41.

### 4.7 The generator, the network and the objectives

- **Generator** (`data.py`). Random spline quadrilaterals, two thirds charted by Winslow and one third by the Coons patch, and one draw in five a random conformal map (a cubic or an exponential of $M\xi+i\eta$). Every chart is certified and kept only inside the envelope; a quarter of the draws were rejected. $\log\hat\kappa$ is a random field plus, in 40% of draws, a random material jump, with a distribution symmetric under negation (for the dual). Sources, Fourier numbers and $n\in\{32,64,128\}$ are drawn as registered.
- **Network** (`model.py`). A node-aligned U-Net: $2^k+1\to2^{k-1}+1$ by a stride-2 average, bilinear up with aligned corners, two $3\times3$ convolutions per level, GELU, FiLM on two scalars of Fo, and even reflection padding. The outputs are corrections times the bubble $\beta=16\xi(1-\xi)\eta(1-\eta)$, so traces are exact. Widths $(24,48,96,192)$ give 1.10 M parameters; 0.28 M and 3.05 M for C5; 1.06 M for C4's spectral operator.
- **Objectives**, in float64. The label-free $\mathcal L_{\text{en}}$ (T26) is normalised per example by the principal part's own energies, which do not depend on the weights. It equals the supervised loss plus a constant to $10^{-9}$ in the training code's own arithmetic, provided labels are solved from the stored float32 coefficients. If they are solved from the float64 ones, the identity holds only to $10^{-7}$; that discrepancy was found and fixed before training.
- **D4 augmentation.** Each symmetry acts on the inputs, on the tensor ($R\hat{\mathsf A}R^\top$) and on $\mu$ (a $90^\circ$ rotation sends $\mu\to-\mu$, a reflection $\mu\to\bar\mu$). For C2's supervised arm it also permutes the labels by the symmetry's signed action on the port basis, checked against classical solves of every transformed problem to $10^{-9}$.

---

## 5. Measured before training: the host with exact classical modes

`scripts/chartop_validate_host.py` → `out/chartop/host_validation.json`, $n=64$ per piece. **With every mode** ($m=n$: hats and all side sines) the host reproduces the full-domain solve to rounding. **With 16 modes per side** what remains is the port-truncation floor, which SE3 and E4 (D9) measure against.

| benchmark | problem | pieces | interface unknowns | all modes vs full domain, energy | 16 modes, energy | 16 modes, max over range |
|---|---|---|---|---|---|---|
| bend-3 | steady | 15 | 594 | $2.3\times10^{-12}$ | $2.3\times10^{-12}$ | $9.7\times10^{-13}$ |
| bend-3 | step | 15 | 594 | $1.3\times10^{-14}$ | $1.73\times10^{-4}$ | $4.2\times10^{-3}$ |
| s-channel | steady | 9 | 440 | $2.7\times10^{-12}$ | $9.25\times10^{-3}$ | $1.5\times10^{-3}$ |
| s-channel | step | 9 | 440 | $1.8\times10^{-14}$ | $1.35\times10^{-3}$ | $2.81\times10^{-2}$ |
| insert-round | steady | 39 | 1,338 | $4.6\times10^{-11}$ | $3.13\times10^{-4}$ | $4.7\times10^{-5}$ |
| insert-round | step | 39 | 1,338 | $1.6\times10^{-13}$ | $6.6\times10^{-5}$ | $4.9\times10^{-4}$ |

- **bend-3, steady**: the solution is linear in $\theta$, which the corner hats represent exactly, so 16 modes are as exact as all of them. The full-domain solve matches the analytic $T=400-100\,\theta/(\pi/2)$ to $9\times10^{-11}$ K.
- **s-channel, steady**: the full-domain solve matches $T=400-100u$ on the fine harmonic coordinate to 0.037 K. Its 16-mode floor, 0.93% in energy, comes from the end pieces' corners. The trace of $u$ along a wall leaving a $154^\circ$ corner is not smooth in arc length.
- **s-channel, step**: the floor of $2.81\%$ of the range is what D9 corrected. The one-step boundary layer along the insulated walls is steeper than 15 sines can follow.

---

## 6. Priced before the long runs (F9's discipline)

Untrained networks: speed does not depend on the weights. `scripts/chartop_price.py` → `out/chartop/price-54cd5124ef46-20261018-181032.json` and `-181424.json` (RTX 3090, before and after the float32 principal part) and `out/chartop/price-17c071b456c5-20261018-185004.json` (A100 SXM4).

| quantity | value |
|---|---|
| forward per piece, bfloat16, batch 16, RTX 3090: $n=32$ / $64$ / $128$ | 0.31 / 0.65 / 2.26 ms (1.1 M); 0.34 / 0.76 / 2.58 ms (3.05 M); 0.31 / 0.65 / 2.17 ms (0.28 M) |
| the same, batch 1 | 4.1–5.3 ms per call, whatever the size: launch-bound |
| forward per piece, A100 SXM4, 1.1 M, $n=64$, batch 16 / 1 | 0.37 / 4.5 ms |
| one training step of arm B, RTX 3090, $n=32$ / $64$ / $128$ (batch 32 / 32 / 8) | 27 / 63 / 60 ms (30 / 83 / 84 ms with the principal part in float64) |
| the same on the A100 | 29 / 46 / 43 ms |
| generator on the RTX 3090 box (EPYC 7502, 63 workers, one BLAS thread each) | 31.7 examples/s |
| one labelled example (factorisation plus 65 pairs of solves), $n=32$ / $64$ / $128$ | 0.027 / 0.079 / 0.43 s (the laptop, on battery) |

**What the pricing found.**
- A step is dominated by the float64 objective and the principal part's transforms, not by the network: 0.28 M and 3.05 M cost 59 and 75 ms at $n=64$.
- 63 generator workers each starting 64 OpenBLAS threads could not create their threads at all. One BLAS thread per worker fixed it.
- **Against [[chart-operator-training-and-cost]] §5.1's estimate** ("2–5 h on one GPU" for $10^5$ pieces and 100 passes), the measured mean step of 54.6 ms per 20 examples gives about 7.6 h on the RTX 3090.

---

## 7. Training

**The data.** Each box drew it from the registered seeds (`out/chartop/box/A/`, `B/`). Both drew exactly the same training set: 50,000 pieces, 16,523 rejected charts, generator source `b31081dc…`, the committed code.
- Coverage of the training set: $n=32$, 64, 128 at 30.2%, 49.8% and 20.0%. The four $\sup\lvert\mu\rvert$ bins of E1 at 8.9%, 20.1%, 33.3% and 37.7%. Spline quadrilaterals 80%, conformal maps 20%. The three frequencies a third each.
- The other sets: 12,000 near-conformal pieces for C1 (every one in the first $\mu$ bin), 6,000 labelled pieces for C2 (the first of the training seeds, drawn again with labels), and 600 labelled validation pieces.

**The boxes.** Box A, an RTX 3090 with a 64-core EPYC at \$0.195 an hour, trained B, B′ and C1. Box B, an A100 SXM4 (40 GB) with a quota of 23 CPUs at \$0.532 an hour, trained the other eight arms. Each box armed a self-stop deadline and was destroyed once its weights had been downloaded and their sha256 matched the box's.

**Every arm, as registered.** The records are in `out/chartop/runs/<arm>/` (`arm.json`, `log.jsonl`, `summary.json`); the weights `best.pt` stay local, and their sha256 is in the evaluation record. The checkpoint kept is the one with the lowest validation median $e_H$ (`train.py`). A pass is one visit to every training piece.

| arm | box | network | form | ports | objective | data (pieces) | steps | passes | wall-clock, s | validation median $e_H$: first / kept (step) | validation median $e_p$: kept / last |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B | A | U-Net 1.10 M | bubble | sine-hat | label-free | train (50,000) | 150,000 | 60 | 8,471 | 0.3372 / 0.33660 (140,000) | 0.094 / 0.089 |
| B′ | A | U-Net 1.10 M | bubble | sine-hat | label-free | train (50,000) | 37,500 | 15 | 2,122 | 0.3373 / 0.33667 (35,000) | 0.148 / 0.147 |
| C1 | A | U-Net 1.10 M | bubble | sine-hat | label-free | near-conformal (12,000) | 37,500 | 62 | 2,124 | 0.3373 / 0.33714 (37,500) | 0.212 / 0.212 |
| C2-sup | B | U-Net 1.10 M | bubble | sine-hat | supervised | labelled (6,000) | 37,500 | 124 | 3,759 | 0.3372 / 0.33664 (27,500) | 0.153 / 0.133 |
| C2-free | B | U-Net 1.10 M | bubble | sine-hat | label-free | labelled (6,000) | 37,500 | 124 | 1,910 | 0.3373 / 0.33673 (30,000) | 0.168 / 0.153 |
| C3 | B | U-Net 1.10 M | direct | sine-hat | label-free | train (50,000) | 37,500 | 15 | 1,903 | 1.4083 / 1.20403 (32,500) | 0.560 / 0.570 |
| C4 | B | spectral 1.06 M | bubble | sine-hat | label-free | train (50,000) | 37,500 | 15 | 2,060 | 0.3370 / 0.33704 (2,500) | 0.232 / 0.179 |
| C5-0.3M | B | U-Net 0.28 M | bubble | sine-hat | label-free | train (50,000) | 37,500 | 15 | 1,874 | 0.3372 / 0.33680 (22,500) | 0.234 / 0.188 |
| C5-3M | B | U-Net 3.05 M | bubble | sine-hat | label-free | train (50,000) | 37,500 | 15 | 2,046 | 0.3375 / 0.33716 (7,500) | 0.451 / 0.209 |
| C6 | B | U-Net 1.10 M | bubble | cosine | label-free | train (50,000) | 37,500 | 15 | 2,440 | 0.3354 / 0.33408 (25,000) | 0.195 / 0.161 |
| C7 | B | U-Net 1.10 M | fast-solve | sine-hat | label-free | train (50,000) | 37,500 | 15 | 2,335 | 0.3373 / 0.33659 (35,000) | 0.217 / 0.217 |

C6's $e_H$ is measured in its own port space (cosines), so it is not comparable with the others; C6 is judged on the benchmarks instead (§3).

B ran at 56.5 ms a step on the RTX 3090, 3.5% over the price. Every validation is logged in its `log.jsonl`.
- **The particular field learned.** B's validation median $e_p$ fell from 1 (a zero field) through 0.375 at step 5,000 to 0.089 at the end.
- **The modes did not, in any arm.** With the principal part and the sine-hat ports, every validation median $e_H$ of every arm lay between 0.3366 and 0.3382, whatever the objective, data, backbone, size or output form. The plain square's principal part alone gives 0.337. Without it (C3), $e_H$ started at 1.41 and reached 1.20.
- **So noise in $e_H$ chose the kept checkpoint.** C4 kept step 2,500, whose particular field (0.232) is worse than at its last validation (0.179). C5-3M kept step 7,500, at 0.451 against 0.209.
- The supervised arm C2-sup was as flat as its label-free twin, so the objective is not the cause.

---

## 8. What the output form asks of the network (diagnostics, before the evaluation)

`scripts/chartop_diagnose.py` → `out/chartop/diagnose-20261018-153945.json`, with classical solves on validation seeds only.

**Where $e_H$ lives.** $e_H^2$ weights every mode by its energy, and the energy of $\sin(j\pi t)$ grows with $j$. For the principal part, the median share of $\sum_k\lVert H^{(0)}_k-H_k\rVert^2_{A_I}$ is:
- the corner hats: 0.1%;
- sines $j=1$–3: 5.2%;
- sines $j=4$–8: 26.1%;
- sines $j=9$–15: 68.5%.

The high sines decay within one or two cells of their side. At $n=32$ the top one has barely two nodes per wavelength. **Learning every mode up to $j=8$ perfectly would still leave $e_H\approx0.337\sqrt{0.685}=0.28$.**

**What the bubble asks for.** The head must output $N=(H-H^{(0)})/\beta$, with $\beta=16\xi(1-\xi)\eta(1-\eta)$:
- Its root mean square is moderate: 0.06 for the hats, rising to about 1 for the high sines.
- **Its maximum sits at the node diagonally next to a corner, and grows with the resolution:** median 16 at $n=32$, 50 at $n=64$, 89 at $n=128$.
- The bubble vanishes like $\xi\eta$ at a corner. Under a shearing chart, though, the exact mode near the corner has the local behaviour of the pulled-back anisotropic operator, not the plain square's, so the quotient blows up as the grid resolves the corner.

**[AI Inference]:** a network with a smooth convolutional bias learns these grid-scale, corner-singular corrections last. This is consistent with the modes not moving in 150,000 steps, and with the overfit probes on four examples, which moved $e_H$ by a few per cent in 600 steps under every loss weighting tried.

**What E5 showed early, and a control.** The E5 timing script (§10) records, beside each time, the learned superelement's energy error on the benchmarks. That is E4's energy measure, so **E4 was seen before the one evaluation**: 24%, 42% and 67% on the three steady problems, and 10%, 7.1% and 7.4% on the steps. Nothing was changed in response.

The control with no trained weights, the principal part alone (`scripts/chartop_principal_control.py` → `out/chartop/principal-control.json`), gives:
- 0.017%, 5.3% and 5.6% on the steady problems;
- 60%, 52% and 32% on the steps, where the missing particular field dominates.

So **on the steady problems the trained modes are worse than their own starting point.** Those problems have no source, so the network's fifth input channel is identically zero. The generator always drew a unit-RMS source, so that input never appeared in training, and the network's modes (one forward pass for all 65 fields) depend on it. **[AI Inference]:** the physics says the modes do not depend on the source. The output design lets them, and the generator never tested the case. *(Tested after the evaluation, §9.7: the modes do depend on the source channel, but the zero source is not why they fail. The temperature level is.)*

---

## 9. The one evaluation

**How it ran.** `scripts/chartop_evaluate.py --runs out/chartop/runs --test-dir ~/.cache/chartop/test` ran once, from 20:38 to 21:24 on 2026-10-18. It used the laptop's CPU, with the networks in float32 on 8 threads (`gates.EVAL_PRECISION`), on battery. Before reading anything it recorded the eleven weights' sha256 in `out/chartop/evaluated.json`. The records are `out/chartop/evaluation-20261018-203833.json` and `.txt`, the per-piece rows in `-test-rows.npz`, and the console log in `-console.log`.

**The test split** was drawn on the laptop from the registered seeds 3,000,000–3,002,399, by the same generator source as every training set (`b31081dc…`). It holds 2,400 pieces (841 charts rejected): 743, 1,211 and 446 at $n=32$, 64 and 128, with exact labels in both port bases. The 6.7 GB are kept outside OneDrive, in `~/.cache/chartop/test`.

### 9.1 The verdicts

| gate | verdict | what decided it |
|---|---|---|
| E1 | **not met**, in all 12 cells | arm B's median $e_H$ is 0.12–0.15 at $\sup\lvert\mu\rvert<0.15$ and 0.44 at $[0.45,0.65]$, against 0.01; its median $e_p$ is 0.063–0.105, against 0.02 |
| E3 | **met** on all six problems | the coupled error never exceeds Theorem 2's bound evaluated with the measured field errors |
| E4 | **not met** on any problem, by any arm | arm B's energy error is 7.1–67% against 2%; the best arm on any problem reaches 3.9% |
| E5 | reported in §10 | no arm is faster at comparable accuracy |
| E6 | §11 | S1–S12 row by row |
| SE1 | **met** on all six | the Gram identity's residual is at most $1.9\times10^{-15}$, and $\lambda_{\min}(\tilde{\mathsf S})\ge\lambda_{\min}(\mathsf S)$ everywhere |
| SE2 | **met** | one network call per piece per linear solve |
| SE3 | **met** on all six | the network term exceeds the 16-mode truncation error by a factor of at least 46 |
| DUAL | **met** | the chain's worst smallest eigenvalue is $-7.4\times10^{-15}\lvert\Lambda\rvert$, which is rounding, on 63 benchmark pieces and 100 test pieces |
| KT | **not met** | the frozen-coefficient iteration converges in 5 outer iterations with learned modes, as with classical ones, but misses E4's bars |
| C1–C7 | §9.6 | ties, except C3 and C6, which lose; C5 chooses 0.3 M |

**In one sentence:** everything the coupling guarantees whatever the network returns held to rounding (E3, SE1, SE2, DUAL), and everything that asks the network to be accurate failed (E1, E4, KT).

### 9.2 The benchmarks, arm B

$n=64$ per piece; energy errors relative to the full-domain solve in the energy norm, maximum errors over the solution's range.

| benchmark | problem | pieces | 16-mode floor: energy / max | arm B: energy / max | E4's max bar | E3: error / bound | $\lambda_{\min}$: learned / exact |
|---|---|---|---|---|---|---|---|
| bend-3 | steady | 15 | $2.3\times10^{-12}$ / $9.7\times10^{-13}$ | 24.4% / 2.2% | 1.0% | 125.3 / 125.3 | 8.0527 / 8.0506 |
| bend-3 | step | 15 | $1.7\times10^{-4}$ / $4.2\times10^{-3}$ | 10.4% / 41.5% | 1.4% | 3,559 / 3,566 | 10.858 / 10.857 |
| s-channel | steady | 9 | $9.3\times10^{-3}$ / $1.5\times10^{-3}$ | 42.3% / 2.5% | 1.15% | 96.7 / 99.6 | 2.6145 / 2.5982 |
| s-channel | step | 9 | $1.35\times10^{-3}$ / $2.8\times10^{-2}$ | 7.1% / 27.1% | 3.8% | 1,535 / 1,577 | 9.8937 / 9.8853 |
| insert-round | steady | 39 | $3.1\times10^{-4}$ / $4.7\times10^{-5}$ | 66.8% / 5.2% | 1.0% | 388.6 / 395.3 | 4.1093 / 4.0571 |
| insert-round | step | 39 | $6.6\times10^{-5}$ / $4.9\times10^{-4}$ | 7.4% / 14.5% | 1.05% | 1,406 / 1,442 | 11.587 / 10.412 |

E3's errors and bounds are in the energy norm. Each bound is the classical floor plus the network term, and the network term is nearly all of it. No piece was refused: every benchmark piece lies inside the envelope.

**Every arm, E4's energy error in per cent** (C6's comparison reads this table):

| problem | B | B′ | C1 | C2-sup | C2-free | C3 | C4 | C5-0.3M | C5-3M | C6 | C7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bend-3, steady | 24 | 36 | 21 | 39 | 133 | 622 | 153 | 144 | 177 | 194 | 200 |
| bend-3, step | 10 | 11 | 13 | 10 | 13 | 25 | 14 | 11 | 14 | 12 | 3.9 |
| s-channel, steady | 42 | 60 | 57 | 66 | 241 | 604 | 264 | 247 | 302 | 330 | 356 |
| s-channel, step | 7.1 | 9.0 | 13 | 7.1 | 9.5 | 26 | 12 | 9.4 | 13 | 10 | 7.1 |
| insert-round, steady | 67 | 130 | 66 | 124 | 385 | 939 | 409 | 395 | 452 | 483 | 463 |
| insert-round, step | 7.4 | 8.2 | 8.2 | 7.4 | 7.8 | 51 | 5.8 | 6.2 | 8.1 | 8.6 | 7.5 |

The principal part alone, with no weights, gives 0.017%, 5.3% and 5.6% on the steady problems (§8). On those problems every arm is worse than its own untrained starting point, and seven of the eleven are worse than a zero field (over 100%) on all three. On the steps, which have a source, the arms agree to within a factor of four, C3 aside.

### 9.3 E1, cell by cell (arm B)

| $\sup\lvert\mu\rvert$ | frequency | pieces | $e_H$: median / 90th pct | $e_p$: median / 90th pct |
|---|---|---|---|---|
| $[0,0.15)$ | low | 76 | 0.131 / 0.209 | 0.063 / 0.118 |
| $[0,0.15)$ | mid | 69 | 0.124 / 0.198 | 0.065 / 0.127 |
| $[0,0.15)$ | high | 70 | 0.145 / 0.207 | 0.082 / 0.155 |
| $[0.15,0.30)$ | low | 163 | 0.263 / 0.372 | 0.074 / 0.145 |
| $[0.15,0.30)$ | mid | 147 | 0.267 / 0.381 | 0.063 / 0.142 |
| $[0.15,0.30)$ | high | 183 | 0.260 / 0.369 | 0.078 / 0.154 |
| $[0.30,0.45)$ | low | 274 | 0.325 / 0.449 | 0.063 / 0.146 |
| $[0.30,0.45)$ | mid | 248 | 0.326 / 0.458 | 0.073 / 0.144 |
| $[0.30,0.45)$ | high | 281 | 0.337 / 0.453 | 0.091 / 0.174 |
| $[0.45,0.65]$ | low | 298 | 0.437 / 0.623 | 0.076 / 0.148 |
| $[0.45,0.65]$ | mid | 287 | 0.442 / 0.657 | 0.085 / 0.175 |
| $[0.45,0.65]$ | high | 304 | 0.439 / 0.619 | 0.105 / 0.194 |

Every cell had at least 30 pieces, so all 12 were judged, and none met the bars (median $e_H\le0.01$, 90th percentile $\le0.03$; median $e_p\le0.02$, 90th percentile $\le0.05$).
- **$e_H$ grows with the distortion**, from 0.13 to 0.44. The principal part is the plain square's, exact only at $\mu=0$ with uniform material, and the trained correction does not change that: every arm with the principal part and sine-hat ports has a test median within 0.22% of B's (§9.6).
- **The particular field did learn**, to a median $e_p$ of 0.079 over the whole split, but its median is 3 to 5 times the bar in every cell.
- **By resolution:** median $e_H$ 0.318, 0.344 and 0.350 at $n=32$, 64 and 128; median $e_p$ 0.070, 0.071 and 0.140.
- **By Fourier number** (a breakdown of the same rows, made after the evaluation: `scripts/chartop_test_rows.py` → `out/chartop/test-rows-breakdown.json`): median $e_H$ 0.323–0.346 and $e_p$ 0.077–0.078 in every decade from 0.01 to 100, and 0.329 and 0.081 when steady. **The error does not grow as the step shrinks** (S2).

### 9.4 The dual bracket (DUAL)

**Met.** On the 63 steady benchmark pieces, with the coupled solution's traces, and on 100 steady test pieces, with smooth random traces, every difference of the chain $\tilde\Lambda_{\text{low}}\preceq\Lambda_{\text{low}}\preceq\Lambda\preceq\tilde\Lambda$ has smallest eigenvalue at least $-7.4\times10^{-15}\lvert\Lambda\rvert$. Arm B served as both experts, the dual one with $\log\hat\kappa$ negated.

**The gap is a certificate that needs no exact solve.** $c^\top(\tilde\Lambda-\tilde\Lambda_{\text{low}})c$ bounds the network's energy error $c^\top(\tilde\Lambda-\Lambda)c$ from above. Against the true error it was:
- on the benchmark pieces, a median of 1.002 times the true error and at most 1.61 times;
- on the test pieces, a median of 2.03 times and at most 7.9 times.

The exact bracket's own width $c^\top(\Lambda-\Lambda_{\text{low}})c$, the floor, is a median of $3.1\times10^{-5}$ of the energy on the benchmark pieces and $1.1\times10^{-3}$ on the test pieces.

### 9.5 Nonlinear conduction, $\kappa(T)$ (KT)

Steel $\beta=-3\times10^{-3}$, copper $-1\times10^{-3}$ per kelvin, frozen coefficients, tolerance $10^{-6}$:

| benchmark | outer iterations: classical / 16-mode / learned | learned against classical: energy / max | E4's max bar | the nonlinearity itself (max) | Kirchhoff against frozen, classical (max) |
|---|---|---|---|---|---|
| bend-3 | 5 / 5 / 5 | 22.9% / 2.06% | 1.0% | 4.4% | $1.4\times10^{-8}$ |
| s-channel | 5 / 5 / 5 | 41.0% / 2.39% | 1.18% | 4.4% | $1.1\times10^{-6}$ |
| insert-round | 5 / 5 / 5 | 69.1% / 5.36% | 1.0% | 4.7% | (two materials) |

**The route works and the expert fails it.** With learned modes the iteration converges in the same 5 outer iterations as the classical one, at one network call per piece per iteration (75, 45 and 195 calls). On the two single-material benchmarks the classical frozen-coefficient answer matches the Kirchhoff transform to $10^{-6}$ of the range, so the route itself is exact to its tolerance. The answer inherits the learned linear solves' error, which is E4's failure again.

### 9.6 The comparisons

| | question | ratio | verdict |
|---|---|---|---|
| C1 | general against near-conformal training, on the 215 test pieces with $\sup\lvert\mu\rvert<0.15$ | 1.0003 | tie: the general class costs nothing measurable there |
| C2 | supervised against label-free, equal compute | 0.9996 | tie |
| C3 | direct prediction against the principal part | 3.57 | **direct loses**: the principal part helps |
| C4 | spectral operator against U-Net | 1.0015 | tie: the U-Net stands |
| C5 | 0.3, 1 and 3 M: medians 0.3360, 0.3357, 0.3363 | — | **0.3 M**, the smallest within 1.25 of the best |
| C6 | cosines against sine-hats, mean E4 energy error | 4.07 | **cosines lose** |
| C7 | fast-solve form against the bubble | 0.9996 | tie |

**Reading them.** Every comparison of $e_H$ but C3 is a tie because no arm with the principal part moved its modes. The ties say that nothing separates these choices at this error level. They do not say the choices are equivalent once the modes learn. C6's loss is carried by the three steady problems, where every arm is worse than untrained. On the three steps alone the ratio is 1.09, a tie (a breakdown of the same record, after the evaluation).

### 9.7 After the evaluation: why the steady problems fail

`scripts/chartop_posthoc_zero_source.py` → `out/chartop/posthoc-zero-source.json`, with arm B's weights, run after the evaluation. None of it is a gate.

**Not the zero source.** §8 guessed that the sourceless problems fail because the network's source channel is identically zero there, a value training never drew. Setting that channel to the constant 1, inside the training distribution, makes B's coupled errors on the three steady problems **worse**: 54%, 100% and 174%, against 24%, 42% and 67%. So the modes do depend on the source channel, which the physics says they should not. But the zero source is not why they fail.

**The temperature level is.** The coupled answer is linear in the boundary data, and a uniform temperature has no energy on a steady piece. Every steady diffusion operator reproduces a uniform temperature exactly as the sum of the four corner-hat modes, and so does the plain square's principal part. B's correction does not.
- On bend-3's pieces, B's hat modes have a median relative energy error of 0.014, against the principal part's $4.7\times10^{-6}$. Its $e_H$ hardly differs: 0.0195 against 0.0194. The hats carry 0.2% of $e_H^2$, the side sines the rest.
- The benchmarks run at 300–400 K, and that level multiplies the defect. Lowering every temperature by 300 K changes neither the problem nor its energy norm; the reference energies agree to $10^{-9}$. It cuts B's errors to **4.5%, 9.1% and 10.6%**. The principal part's do not move.

| steady benchmark | B, as registered | B, source channel set to 1 | B, every temperature lowered by 300 K | principal part alone | principal part, lowered by 300 K |
|---|---|---|---|---|---|
| bend-3 | 24.4% | 53.5% | 4.5% | 0.017% | 0.0027% |
| s-channel | 42.3% | 100% | 9.1% | 5.3% | 5.3% |
| insert-round | 66.8% | 174% | 10.6% | 5.6% | 5.6% |

The principal part's small change on bend-3 is its float32 rounding times the temperature level.

**Piece by piece**, against the exact modes (medians over each problem's pieces):

| problem | pieces | $e_H$: B / principal part | hats' relative error: B / principal part | pieces where B's $e_H$ is the worse |
|---|---|---|---|---|
| bend-3, steady | 15 | 0.0195 / 0.0194 | 0.014 / $4.7\times10^{-6}$ | 15 |
| bend-3, step | 15 | 0.0207 / 0.0203 | 0.050 / 0.047 | 15 |
| s-channel, steady | 9 | 0.0306 / 0.0305 | 0.013 / $5.8\times10^{-5}$ | 4 |
| s-channel, step | 9 | 0.0328 / 0.0330 | 0.076 / 0.086 | 3 |
| insert-round, steady | 39 | 0.267 / 0.267 | 0.016 / 0.032 | 5 |
| insert-round, step | 39 | 0.267 / 0.267 | 0.034 / 0.036 | 5 |

On every problem B's $e_H$ is within 2% of the untrained principal part's. In the measure it was trained on, the correction is small. For the coupled answer it is in the wrong place. On insert-round the principal part's own hat modes are the worse, but their sum is still exact.

**[AI Inference]:** this is the measure's blind spot as much as the network's. $e_H$, and with it the training objective (T26), weights each mode by its energy, so the hats count for a fraction of a per cent. At a temperature level $T_0$, though, the coupled answer errs by about $T_0$ times the hats' failure to sum to one. An output form that keeps that sum exact whatever the weights, as the bubble keeps the traces exact, would remove the dominant term. On steady pieces that means hat corrections summing to zero. A host that couples temperatures relative to a datum would shrink it. After the shift B's errors still exceed the principal part's, so the correction also damages the other modes, but much less.

### 9.8 The rollout (S11)

The s-channel marched 20 backward-Euler steps from 300 K. The learned superelement kept its modes from the first call and predicted a new particular field each step, one call per piece per step (180 in all).

| step | energy error | max error over range | 16-mode classical floor, max |
|---|---|---|---|
| 1 | 7.1% | 27.1% | 2.8% |
| 5 | 10.3% | 48.9% | 1.2% |
| 10 | 14.8% | 75.0% | 0.84% |
| 20 | 25.5% | 132% | 0.61% |

**The error grows at every recorded step**, to 3.6 times its first-step energy error by step 20, while the classical superelement's floor falls. By step 20 the largest pointwise error exceeds the solution's whole range. E4 is not met at any step.

---

## 10. E5: the cost, reported whatever it says

`scripts/chartop_e5.py`, medians of 5. Two runs:
- the laptop on AC power, network in float32 on 8 threads, with the owner's `phone-remote` `server.py` processes running beside it: `out/chartop/e5-champ-cpu-20261018-173923.json`;
- the rented box, network in bfloat16 on the RTX 3090 and the host on its EPYC CPU: `out/chartop/e5-54cd5124ef46-cuda-20261018-225222.json`.

On the fixed benchmarks the pieces' matrices are assembled once and shared by all arms. In the design loop every arm pays its own assembly. The learned answers are arm B's.

**The laptop, seconds** (energy error of the learned answer in the last column):

| problem | monolithic, cold / warm | classical superelement, cold / warm | learned superelement, cold / warm (forward alone) | coarse classical ($n=32$), cold | learned, energy error |
|---|---|---|---|---|---|
| bend-3, steady | 1.80 / 0.013 | 0.63 / 0.066 | 0.33 / 0.24 (0.23) | 0.33 | 24% |
| bend-3, step | 1.84 / 0.018 | 0.71 / 0.11 | 0.40 / 0.23 (0.21) | 0.43 | 10% |
| s-channel, steady | 0.92 / 0.006 | 0.45 / 0.043 | 0.18 / 0.13 (0.13) | 0.24 | 42% |
| s-channel, step | 0.89 / 0.007 | 0.51 / 0.041 | 0.20 / 0.14 (0.14) | 0.23 | 7.1% |
| insert-round, steady | 6.68 / 0.054 | 2.37 / 0.19 | 1.45 / 0.68 (0.56) | 1.29 | 67% |
| insert-round, step | 7.17 / 0.043 | 2.12 / 0.26 | 1.42 / 0.66 (0.48) | 1.45 | 7.4% |

**The monolithic arm by phase** (`scripts/chartop_e5_mono_phases.py` → `out/chartop/e5-mono-phases-champ.json`). The reference code's node merge, a Python loop, is 0.6–2.8 s of the cold time. Without it, assembly, factorisation and one solve cost 0.77, 0.30 and 4.15 s on the three benchmarks (61,567, 37,375 and 159,871 unknowns).

**The design loop:** 8 variants of bend-3 with the inner radius moved from 34 to 48 cells, so no factorisation or mode survives between solves.

| | monolithic | classical superelement | learned superelement | learned, energy error |
|---|---|---|---|---|
| laptop, total s | 14.0 | 6.17 | 3.44 | 20–30% |
| GPU box, total s | 7.58 | 29.8 | 4.56 | 21–32% |

**What E5 says.**
1. **Geometry fixed, a new load each solve:** a kept monolithic factorisation wins by an order of magnitude (0.006–0.054 s against 0.13–0.68 s for the learned superelement). Stored classical modes also beat the network. This is the record's known result, measured again.
2. **Geometry new each solve:** the learned superelement is the fastest arm. On the laptop it is 1.5–2.6× faster than recomputing classical modes (1.8× over the design loop) and 4.5–5.4× faster than the cold monolithic solve, merge included; without the merge, 1.7–2.9×. On the GPU box its network takes 0.016–0.079 s for all pieces, and the host's float64 work, on that box's slower CPU, dominates.
3. **But its answers are 7–67% wrong in energy**, while the classical solve on a grid twice as coarse costs about the same as the learned one (0.23–1.45 s on the laptop) and is off by at most 1.9% of the range. **So nowhere is the learned expert faster at a comparable accuracy. S7 is not met**, and the coarse classical solver is again the competitor the learned arm loses to, as in [[demo-learned-case-plan]] §9.

---

## 11. E6: S1–S12, row by row

The rows of [[outcome-c5-requirements-for-dd-native-experts]] §2, each answered for arm B or named as not met.

| # | property | Stage 0's answer | verdict |
|---|---|---|---|
| S1 | accepts the subdomain's geometry | Any certified four-sided piece inside the envelope, through its chart. All 63 pieces of the three benchmarks were accepted. A piece outside the envelope is refused, naming the violated quantity (`tests/test_chartop_eval.py`). Forks, rings and pieces outside the envelope are not accepted | **met**, for four-sided pieces in the envelope |
| S2 | runs at the coupling's step | The Fourier number is an input over $[10^{-2},10^{2}]$ and steady, and the benchmarks' steps (0.033–41) were accepted. The error does not grow as the step shrinks: median $e_H$ 0.323–0.346 and $e_p$ 0.077–0.078 in every decade, and 0.329 and 0.081 when steady (§9.3) | **met** as the row asks; the error's level is E1's failure |
| S3 | takes interface data as an input | By construction: one constraint mode per port mode, so the neighbour's trace enters the host's interface coefficients exactly and linearly | **met**, by construction |
| S4 | a bounded, measurable domain of dependence | By construction: a piece's answer depends only on its own chart, material, source, step and port coefficients, with no halo, and the coupling is one interface solve at one call per piece (SE2) | **met**, by construction |
| S5 | the right slow-mode response | The modes *are* the response to interface data. On the test split their median $e_H$ is 0.336, no better than the untrained principal part's. On the benchmarks the response to a uniform temperature, the lowest interface mode, is what broke (§9.7) | **not met** |
| S6 | not closer to singular than the classical map on any mode | SE1: $\lambda_{\min}(\tilde{\mathsf S})\ge\lambda_{\min}(\mathsf S)$ on every benchmark problem, as T25 says it must be for any output | **met**, by theorem and as measured |
| S7 | cheaper than the cheapest classical alternative in the same slot | E5: the classical solve on a grid twice as coarse costs about the same as the learned one and is within 1.9% of the range, against 7–67% in energy (§10) | **not met** |
| S8 | a reference of its own class | The classical superelement with the same 16 port modes per side, at any resolution; E3 measures against it and holds | **met** |
| S9 | accurate enough to be worth the slot | E4: 7.1–67% in energy against 2%, and no arm meets it on any problem (§9.2) | **not met** |
| S10 | precision the probes can use | Every Gram, Schur complement and loss in float64, and TF32 disabled and asserted. With float32 networks, SE1's identity holds to $1.9\times10^{-15}$ and the dual chain to $7.4\times10^{-15}$ | **met**, for the host's probes |
| S11 | a stable free rollout | The s-channel's 20-step march: the energy error grows from 7.1% to 25.5%, and by step 20 the largest pointwise error exceeds the solution's range (§9.8) | **not met** |
| S12 | a licence that allows use | Trained from scratch on this repository's own generator, with no Poseidon or NeuberNet weights anywhere in the chain. The repository has no licence file, so the licence for anyone else is the owner's to choose | **met** as far as third parties go; the owner's licence is open |

S5a and S7a, Tier 89's additions, concern iterated couplings: a stable approach, and the cold march as the bar at a coupled seam. Tier 0 is one direct interface solve, so neither applies here.

**Seven of the twelve are met**, and five of those seven by construction or by theorem. **The four that ask for accuracy, S5, S7, S9 and S11, are not met.** S12 is met only as far as third parties go. The design answered the rows the frozen checkpoint failed for structural reasons (S1, S3, S4, S6, S8, S10). The network has not yet answered the rows that need it to be right.

---

## 12. What Stage 1 needs

Stage 1 is the next four entries of F1's build order ([[chart-operator-family-catalog]]): **CO-E** (plane elasticity), **the thermoelastic bond** on CS-9 ([[case-study-thermal-strain-atlas-0.1]]), **CO-T** (advection–diffusion) and **the one biharmonic network** (CO-P / CO-S). F1's second entry, nonlinear conduction through CO-D's coefficient channel, ran here as KT (§9).

**What Stage 0 measured that Stage 1 must change.** Each item applies to all four families.
1. **A uniform temperature must cost nothing.** Every steady diffusion operator reproduces a constant exactly as the sum of the corner-hat modes, and the plain square's principal part inherits that. The trained correction broke it. At the benchmarks' 300–400 K, the broken sum dominated the coupled error: on bend-3 it fell from 24% to 4.5% when every temperature was lowered by 300 K (§9.7). $e_H$ weights the hats at a fraction of a per cent, so neither the objective nor E1 could see it. **[AI Inference]:** the output form should reproduce the operator's kernel exactly whatever the weights, as the bubble does for traces, and the registered measure should weight that kernel as the coupled answer does. The kernel is the constant for CO-D and CO-T, the rigid motions for CO-E, and the affine fields for the plate.
2. **The modes must not see the load.** By T25's structure the exact modes depend on the geometry, the material and the step, never on the source. This network computes them in the same forward pass as the particular field, from all five channels, and they do depend on the source channel: setting it from 0 to 1 more than doubles B's coupled error on the steady problems (§9.7). **[AI Inference]:** a mode network with no load channel, beside a particular-field network that has one, would make that independence hold by construction. Every Stage 1 family has a load: a body force for CO-E, a source for CO-T, a transverse load for the plate.
3. **The output form must follow the corners.** The quotient the bubble form asks the head for peaks at the node diagonally next to a corner, and the peak grows with the resolution: 16, 50 and 89 at $n=32$, 64 and 128 (§8). **[AI Inference]:** a principal part that already carries the piece's own operator near its corners would leave the network a correction without that growth. Candidates are the frozen-coefficient principal part of [[chart-operator-general-form]], or a few classical smoothing sweeps on the true pulled-back operator. Plane elasticity has corner singularities of its own (Williams, *J. Appl. Mech.* 19, 1952), and so does the plate, so the same growth should be expected there.
4. **$e_H$ is dominated by local modes.** 68.5% of the principal part's $e_H^2$ sits in the side sines $j=9$–15, which decay within one or two cells of their side (§8). **[AI Inference]:** those modes could be computed classically, by a strip solve along each side at a cost linear in $n$, and only the low modes learned. E1's measure would then weight what the network is for.
5. **Seeds.** Every arm was trained once. Here no comparison came near a bar (§9.6), but Stage 1's comparisons need repeats per arm before a ratio near 0.8 or 1.25 can be read.
6. **Operations**, each found in this build:
   - The generator's worker count must follow the container's CPU quota, not `os.cpu_count()`. The A100 box reported 192 CPUs on a quota of 23 and ran 191 workers, at 4.2 s per example against 2.0 s with the 3090 box's 63.
   - A cost script must not record a gate's measure. E5's `learned_energy_rel` showed E4's energy errors before the evaluation (§8).
   - The evaluation records its weights as evaluated before it starts, so an interrupted run could not be resumed without overriding its own refusal. It should resume instead.
   - The kept checkpoint is the lowest validation $e_H$. When $e_H$ does not move, noise decides it (§7).

**Per family** ([[chart-operator-family-catalog]] §4):
- **CO-E**: vector $H^1$ with `MECH` ports, twice the outputs, and a Poisson-ratio input. G0 and T24–T26 hold unchanged. The principal part is not diagonalised by a sine transform, so it is precomputed per Poisson ratio.
- **The thermoelastic bond**: a host object with nothing to train. It carries CO-D's temperature into CO-E's thermal strain, so it inherits CO-D's error, which at Stage 0's accuracy is 7–67% in energy on the benchmarks (§9.2). Thermal strain is driven by the temperature level above a reference, the very quantity item 1 found broken. The bond needs a CO-D that meets E4 first.
- **CO-T**: guarantee class G0′, Petrov–Galerkin with adjoint modes from the same network by $\hat b\to-\hat b$. T25's positivity does not hold, so SE1's second clause has no analogue and the gate must be rewritten. At Péclet 20 and 200 the exact port matrix's symmetric part is indefinite (measured in the catalog), so the seams cannot exchange value and diffusive flux alone.
- **The biharmonic network**: value and slope ports, a stiffer operator, and the corner question of item 3 in a stronger form.

**The worklist rows** are [[gap-worklist]] W356 (item 1), W357 (item 2), W358 (items 3 and 4) and W360 (item 6). Items 1 and 2 are built, checked with untrained weights, and carry a draft registration in [[chart-operator-kernel-exact-forms]].

**Cost.** Arm B's 150,000 steps took 2.35 h on an RTX 3090: 60 passes over 50,000 pieces, about \$0.46 at \$0.195 an hour (§7). **[AI Inference]:** at that rate one Stage 1 family costs a few GPU-hours per arm. Comparisons with seed repeats multiply that by the number of arms, which still fits a ceiling like D6's.

---

## 13. What this page does not show

- **No accuracy that earns the slot.** S7 and S9 are not met (§10, §11). Nothing here says the learned expert is useful. It records what was built, how it was judged, and why it does not yet work.
- **One resolution on the benchmarks**: $n=64$ per piece. The test split covers 32, 64 and 128.
- **Hand-made layouts only.** Every benchmark is a conforming layout of four-sided pieces built by hand (§4.6). The general layout for forks and rings is not built, and no benchmark has a mortar seam.
- **Tier 0 only.** No scattering (tier-2) coupling ran, so E2 has no row and the card carries no contraction factor.
- **One seed per arm**, and every comparison arm at a quarter of B's steps.
- **One family, in 2-D**: CO-D with `THERM` ports.
- **The post-hoc numbers** (§9.7, and §9's breakdowns by Fourier number) were computed after the one evaluation, from its own record or with B's weights. None is a gate.
- **No Stage 0 result on the website or in `proposal/`.** The public architecture page (`proposal/architecture/index.html`, copied to `site/architecture/`) carries two dated errata. The first (2026-10-18, [[gap-worklist]] W359): its piece solver is bilinear finite elements, not cell-centred finite volumes (D7). The second (2026-10-19, W361): its default port basis is the corner functions plus side sines, with the cosines as comparison C6 (D3), and Table 27's SE3 bar names the network term of Theorem 6's bound, as registered. It never stated E4's maximum-error bar (its Table 27 says "registered per family"), so D9 needed no change there. The page still says "No network has been trained", which was true on 12 October and has not been since this stage.
- **The weights are not in the repository.** `out/chartop/runs/*/best.pt` are git-ignored; their sha256 are in the evaluation record and on the card.
- **No cost from the evaluation.** It ran on the laptop on battery, so the `elapsed_s` fields in its record are not costs. E5 (§10) is the cost measurement, made on AC power.

---

## See Also

- [[chart-operator-architecture]] — the design being built (version 1, the CO-D card)
- [[chart-operator-training-and-cost]] — the data, objectives, stages, gates and costs this page executes
- [[chart-operator-design-decisions]] — decisions 1–19, which D1–D8 extend
- [[chart-operator-family-catalog]] — CO-D, the dual certificate (§5), decisions F1–F9
- [[chart-operator-general-form]] — G0, and the frozen-coefficient route for $\kappa(T)$ (§2.4)
- [[outcome-c5-requirements-for-dd-native-experts]] — S1–S12, gate E6
- [[coupling-cost-and-complexity]] — the measured coupling costs E5 is read against
- [[formal-proofs-record]] — T24, T25, T26 in Lean
- [[gap-worklist]] — W355, and the rows Stage 1 inherits, W356–W360
- [[case-study-thermal-strain-atlas-0.1]] — CS-9, where Stage 1's thermoelastic bond is to be tested
- [[chart-operator-kernel-exact-forms]] — §12's items 1 and 2 built: the kernel-exact and split output forms, checked untrained
