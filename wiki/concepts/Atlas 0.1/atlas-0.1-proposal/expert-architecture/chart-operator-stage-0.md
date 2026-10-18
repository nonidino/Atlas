# Chart operator Stage 0 — the first learned expert (CO-D), built, registered and judged

**Type:** Concept page — **build record and pre-registration** (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** opened 2026-10-18 (W355). **Build in progress.** The owner chose the gates and comparison bars of §3, and they were registered in code (`atlas/chartop/gates.py`, hash in `out/chartop/registered.txt`) **before any training run**. E4's maximum-error bar was re-registered the same day (D9), before any training result had been read. The results sections are filled after the one evaluation, whatever it says.
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

**The main arm B.** 150,000 steps in 8,471 s on the RTX 3090 (56.5 ms a step, 3.5% over the price), with every validation logged in `out/chartop/runs/B/log.jsonl`.
- **The particular field learned:** the validation median $e_p$ fell from 1 (a zero field) through 0.375 at step 5,000 to 0.089 at the end.
- **The modes did not:** the validation median $e_H$ was 0.3372 at step 5,000 and 0.3366 at the best checkpoint (step 140,000). The plain square's principal part alone gives 0.337.
- The supervised arm C2-sup was just as flat (0.3372 to 0.3366 over its 37,500 steps), so the objective is not the cause.

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

So **on the steady problems the trained modes are worse than their own starting point.** Those problems have no source, so the network's fifth input channel is identically zero. The generator always drew a unit-RMS source, so that input never appeared in training, and the network's modes (one forward pass for all 65 fields) depend on it. **[AI Inference]:** the physics says the modes do not depend on the source. The output design lets them, and the generator never tested the case.

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
- [[gap-worklist]] — W355
