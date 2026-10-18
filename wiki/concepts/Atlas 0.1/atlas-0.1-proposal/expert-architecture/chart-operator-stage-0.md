# Chart operator Stage 0 — the first learned expert (CO-D), built, registered and judged

**Type:** Concept page — **build record and pre-registration** (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** opened 2026-10-18 (W355). **Build in progress.** The gates and comparison bars of §3 were chosen by the owner and registered in code (`atlas/chartop/gates.py`, hash in `out/chartop/registered.txt`) **before any training run**. Nothing has been trained when this section was written; the results sections are filled after the one evaluation, whatever it says.
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
| E4 | on every benchmark problem, energy-norm error $\le0.02$ and maximum error $\le0.01$ of the solution's range |
| E5 | reported whatever it says; "faster" only where measured faster |
| SE1 | Gram identity residual $\le10^{-10}$ relative, and $\lambda_{\min}(\tilde{\mathsf S})\ge\lambda_{\min}(\mathsf S)(1-10^{-10})$, on every benchmark problem |
| SE3 | on every benchmark problem, the 16-mode truncation error $\lVert u-u_m\rVert_M$ below the network term |
| DUAL | on steady pieces, every difference of the chain $\tilde\Lambda_{\text{low}}\preceq\Lambda_{\text{low}}\preceq\Lambda\preceq\tilde\Lambda$ has smallest eigenvalue $\ge-10^{-10}\lvert\Lambda\rvert$; the gap against the true error reported |
| KT | $\kappa(T)$ on every steady benchmark: the frozen-coefficient iteration converges within 50 outer iterations and meets E4's bars against the classical frozen-coefficient solution |
| C1–C7 | the challenger wins at a ratio $\le0.8$ of the reference's median $e_H$ (C6: of the mean E4 energy error), loses at $\ge1.25$, otherwise a tie and the default stands |

**SE3, read correctly.** SE3 does not ask the network to beat the truncation floor. It asks the opposite: that 16 modes per side are enough, so that the network, not the port count, is the limiting term.

**The bins of E1.** $\sup\lvert\mu\rvert$ in $[0,0.15)$, $[0.15,0.30)$, $[0.30,0.45)$ and $[0.45,0.65]$. Input frequency is the correlation length $\ell$ of the material and source fields: low ($\ell=0.5$), mid ($0.25$), high ($0.125$).

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
