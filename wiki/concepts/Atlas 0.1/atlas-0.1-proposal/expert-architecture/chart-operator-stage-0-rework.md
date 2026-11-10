# Chart operator — Stage 0 reworked: the two-level, coarse-exact expert (drafted)

**Type:** Concept page — **design draft** (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** drafted on 2026-11-09 from a design conversation with the owner, who then asked for it to be built end to end. **Phase 1 is built and tested** (§11): the coarse level, `atlas/chartop/coarse.py`, with Proposition A and Lemma B checked numerically. **Phase 2's script is written and smoke-tested** (§12); its predictions (§9) carry their readings there and are **not yet registered**, and nothing has been measured on the pieces Phase 2 reads. Nothing is trained or rented. The page sets out how Stage 0's CO-D expert should be formulated, given everything Stage 0 and its successors measured. It gives a phased workflow whose decisive phase needs no training.
**Hub:** [[chart-operator-architecture]] · **Follows:** [[chart-operator-stage-0]] and [[chart-operator-kernel-exact-forms]] · **Worklist:** [[gap-worklist]] W362 (open) · **Family:** [[chart-operator-family-catalog]] (CO-D)

---

## 0. In one paragraph

Stage 0's guarantees held, and every accuracy failure since has one of two causes:
- the network was asked for a **global** response (the gentle modes the coupled answer reads), which it cannot produce and which its loss barely weights;
- it was asked for a **singular** one (the corners), which a smooth output cannot follow.

The rework divides the work by scale:
- **A coarse classical correction, per piece, gives the global part.** One Galerkin solve on a $16\times16$ coarse grid of the piece's true operator.
- **The strips give the fast side layers**, as now.
- **The network gives only what is left**, and only through a map that leaves it no way to change the coarse part.

That map is one affine projection, $\mathcal C$. It does for the coarse scale what the bubble does for the traces and the kernel projection (W356) does for a uniform temperature: it holds the property exactly whatever the weights. The accuracy gate becomes E4 alone. Whether a network is needed at all for CO-D is measured first, untrained.

---

## 1. The problems, in plain words

| # | problem | where measured | status before this page |
|---|---|---|---|
| P1 | The trained correction broke the uniform-temperature sum; at 300–400 K it dominated the coupled error | [[chart-operator-stage-0]] §9.7 | fixed by W356's kernel projection $\Pi$ |
| P2 | The modes depended on the load channel | Stage 0 §9.7 | fixed by W357's split form |
| P3 | The target the head must output grows with $n$ at corners and where a material interface meets a side | Stage 0 §8; [[chart-operator-kernel-exact-forms]] §10, §11.5 | partly treated: the corner bubble and the strips level the peak off, except at interface–side points |
| P4 | The coupled error sits in the **gentle, global modes**: the hats' corner differences and the side sines $j=1$–3 | forms §11.1; §11.3 (with those columns exact, E4 is met on all three steady problems) | open |
| P5 | The network cannot produce global responses. It underfits four pieces (median $e_{H,L}$ 0.16 after 3,000 steps), and at $n=128$ a side node cannot see the opposite side | forms §11.6, §11.2 | open; a global path (B3) was a tie |
| P6 | **The loss and the accuracy gate disagree.** $e_H$ and the energy loss weight modes by energy, which the fast and singular parts dominate; the coupled answer weights them by the interface data, which the gentle modes dominate. Training lowered $e_H$ while raising the coupled error (bend-3: 3.94% against 0.00034% untrained) | forms §8.11 | open; reweighting the columns (B2) was a tie |
| P7 | E1's bar (median $e_H\le0.01$) lies below the form's own floor: the best smooth correction leaves 0.07–0.19 at $n=128$. The bar was set without that check, as E4's maximum was before D9 | forms §10 | open |
| P8 | The principal part is the plain square. Its error grows with the shear: B's median $e_H$ is 0.13 at $\sup\lvert\mu\rvert<0.15$ and 0.44 at $[0.45,0.65]$ | Stage 0 §9.3 | open |
| P9 | The price. A classical solve on a grid twice as coarse costs about what the network costs and is within 1.9% of the range. H2r is exact on R2's layouts and priced at 0.306 | Stage 0 §10; forms §11.19 | the bar any learned arm must beat |
| P10 | The particular field generalises poorly: $e_p$ 0.004–0.010 when overfitting four pieces, a median of 0.066 on the test split | forms §11.6, §8.11 | open |

Operations (W360) are fixed and not repeated here.

**The pattern.** Every guarantee in the design was structural, and every one held: traces by the bubble, the kernel by $\Pi$, the load by the split, stability by the Gram. Every accuracy property was left to training, and none was reached. The rework makes one more accuracy property structural, the coarse scale, and leaves training only what is local.

---

## 2. The idea: divide the work by scale

**Intuition.** A temperature field's broad shape (how a piece tilts between its corners, and the first few wiggles along a side) is smooth and depends on the whole piece. A coarse grid captures smooth, whole-piece behaviour almost exactly, and a $15\times15$ linear system is nearly free. A network is slow to learn exactly that. Thin layers along a side are local, and the strips already solve them classically. What is left between the two scales is local and of moderate size, which is the job a convolutional network is built for.

| part of a mode | who computes it | why |
|---|---|---|
| broad, global response (what E4 reads) | **a Galerkin coarse correction** on the piece's true operator | smooth and global; exact in its class, at a cost linear in the fine unknowns |
| fast side sines $j=9$–15 | the strips (W358's C3, built) | thin and local; already classical |
| the uniform temperature | $\Pi$ (W356, built) | exact by construction |
| what remains: local, mid-scale | **the network**, through $\mathcal C$ | local and bounded; the network cannot reach the coarse part |

**The principle.** Give the network only what is local, and give it no way to damage what is global. The bubble, $\Pi$ and the split already apply this to the traces, the kernel and the load. $\mathcal C$ applies it to the coarse scale.

---

## 3. The formulation

### 3.1 Notation

As in [[chart-operator-stage-0]] §2. One piece on the reference grid has $n\times n$ cells, energy matrix $M=\begin{pmatrix}A_I&-B\\-B^\top&D\end{pmatrix}$, port basis $Q$ (64 columns: four hats and the sines $j=1$–15 on each side) and exact modes $H=A_I^{-1}BQ$. The kernel coefficients are $c_1$ (with $Qc_1=1$ on the loop), the reaction is $s_0$, the level field is $u_1=A_I^{-1}b_1$, and the particular field is $u_p=A_I^{-1}b_I$. The plain square's modes are $H^{(0)}$ (sine transforms). $\beta_c$ is the corner bubble and $\Pi$ the kernel projection of forms §3.

### 3.2 The coarse space and the coarse correction

Let $n_c$ divide $n$ (both powers of two). $P\in\mathbb R^{(n-1)^2\times(n_c-1)^2}$ is bilinear interpolation from the coarse grid's interior nodes to the fine grid's interior nodes, with the coarse boundary held at zero. Its range $V_c$ consists of the fine interior fields that are bilinear on the coarse cells and zero on the loop. The **Galerkin coarse operator** of the piece is
$$A_c=P^\top A_IP\qquad\bigl((n_c-1)^2\text{ unknowns: }225\text{ at }n_c=16\bigr),$$
assembled from the piece's own operator, so it carries the piece's shear, stretch, material and reaction.

For the modes (right-hand side $BQ$), and likewise for the level field ($b_1$) and the particular field ($b_I$), the **coarse correction** of any interior fields $X$ is
$$\mathcal C(X)=X+P\,A_c^{-1}P^\top\bigl(BQ-A_IX\bigr).$$
It needs no exact solution, only a fine residual and a coarse solve. Writing $E=X-H$ for the input's error, $A_IH=BQ$ gives
$$\mathcal C(X)-H=\mathcal Q_c\,E,\qquad \mathcal Q_c=I-P\,A_c^{-1}P^\top A_I,$$
where $\mathcal Q_c$ is the $A_I$-orthogonal projector onto the $A_I$-orthogonal complement of $V_c$.

**Proposition A** (what $\mathcal C$ guarantees, for any input $X$):
1. **Galerkin orthogonality.** $P^\top\bigl(BQ-A_I\,\mathcal C(X)\bigr)=0$. The output's error has no component the coarse space can express, in the energy inner product.
2. **Never worse, column by column.** $\lVert H_k-\mathcal C(X)_k\rVert_{A_I}=\min_{v\in V_c}\lVert H_k-X_k-v\rVert_{A_I}\le\lVert H_k-X_k\rVert_{A_I}$, and $\lVert\mathcal Q_cE\rVert_{A_I}^2=\lVert E\rVert_{A_I}^2-\lVert(I-\mathcal Q_c)E\rVert_{A_I}^2$.
3. **Idempotent, and blind to coarse shifts.** $\mathcal C(\mathcal C(X))=\mathcal C(X)$, and $\mathcal C(X+Pv)=\mathcal C(X)$ for every coarse $v$.
4. **Exact at the top.** At $n_c=n$, $V_c$ is the whole interior space and $\mathcal C(X)=H$ for every $X$. This is the positive control.
5. **Traces untouched.** $P$ maps into the interior, so the traces stay those of $Q$ exactly, and Theorem 2, E3 and the Gram identity (SE1) apply unchanged.
6. **Affine and cheap to differentiate.** The gradient through $\mathcal C$ is $\mathcal Q_c^\top$ applied to the upstream gradient: one fine matrix–vector product and one coarse solve per column.

*Proof.* (1) $P^\top(BQ-A_I\mathcal C(X))=P^\top(BQ-A_IX)-A_cA_c^{-1}P^\top(BQ-A_IX)=0$. (2) The minimiser $v^\star$ of $\lVert E+v\rVert_{A_I}$ over $V_c$ solves $P^\top A_I(E+Pa)=0$, that is $a=-A_c^{-1}P^\top A_IE$, so $E+Pa=\mathcal Q_cE$; Pythagoras in the $A_I$ inner product gives the identity. (3) $\mathcal Q_c^2=\mathcal Q_c$ and $\mathcal Q_cP=0$. (4) $P=I$ gives $\mathcal Q_c=0$. (5) and (6) are immediate. $\square$

**Lemma B** ($\mathcal C$ keeps the uniform temperature exact). Let the exact fields satisfy the kernel identity $Hc_1+s_0u_1=1$. If the inputs satisfy it too, $Xc_1+s_0X_1=1$, then so do the outputs $\mathcal C(X)$ and $\mathcal C_1(X_1)$.
*Proof.* $\mathcal Q_c$ acts on every column through the same spatial operator, so it commutes with taking combinations of columns: $\mathcal C(X)c_1+s_0\mathcal C_1(X_1)-1=\mathcal Q_c\bigl(Ec_1+s_0E_1\bigr)=\mathcal Q_c\,0=0$. $\square$

**The one condition the lemma needs.** The exact fields must satisfy the identity in the discrete system. Steady ($s_0=0$), they do, because $M\mathbf 1=0$. For a step, the stored arrays round $\sigma$ and $\tilde J$ to float32 separately, so with $b_1=\text{mass}(\tilde J)\mathbf 1$ the identity holds only to $2\times10^{-10}$–$3\times10^{-8}$ (the corner controls' record). The rework therefore defines
$$b_1=\frac{(M\mathbf 1)_I}{s_0}\qquad(s_0>0),$$
the reaction's own row sums, which is $\text{mass}(\sigma/s_0)\mathbf 1$. The identity is then exact to float64 rounding. Nothing else reads $b_1$.

### 3.3 The expert

S's form (split, corner bubble, strips, $\Pi$; forms §8), followed by $\mathcal C$:
$$\hat H=\mathcal C\Bigl(\Pi\bigl(H^{(0)}+\beta_cN_\theta;\ \hat u_1\bigr)\Bigr),\qquad \hat u_p=\mathcal C_p\bigl(\bar g\,\hat u_1+\text{scale}\cdot\beta_cN_{p,\theta}\bigr),$$
with the strip columns replaced by strip solves before $\Pi$, as now, and $\hat u_1$ itself passed through $\mathcal C_1$.
- **Untrained ($N=0$)** this is a fully classical **two-level expert**: the plain square on the fine grid, the strips along the sides, and the piece's true operator on the coarse grid. Call it **TL0**. On a square of uniform material the residual $BQ-A_IH^{(0)}$ is round-off, so TL0 equals U to rounding there, and H2r's exactness on squares is kept.
- **Trained**, the network adds $\beta_cN_\theta$. By Proposition A it can change only $\mathcal Q_c$ of the error. It cannot move the coarse component that the coupled answer leans on. This is the failure trained S showed (forms §8.11).

### 3.4 The objective

**Unchanged: T26's label-free energy.** T26 rests on the Gram identity, which holds for any trace-exact fields, so $\mathcal C$'s output qualifies, and the exact modes stay the minimiser. Because the output's error is $\mathcal Q_cE$, the loss sees only the part of the error that is not coarse. That part is exactly what the network is asked for. So the misalignment of P6 is removed by construction, not by reweighting. B2's prior is not needed. D4 augmentation stays: nested uniform grids are symmetric, so $\mathcal C$ commutes with the square's eight symmetries.

### 3.5 The gates

| gate | in the rework |
|---|---|
| K, D, L, E3, SE1, SE2, DUAL | kept as in `gates_forms`. K by $\Pi$ and Lemma B; D from K; L from the split; E3, SE1 and DUAL from the traces (Proposition A, item 5) |
| **CX** (new) | coarse exactness, label-free and computable on every piece: $\lVert P^\top(BQ-A_I\hat H)\rVert/\lVert P^\top BQ\rVert\le10^{-10}$, and likewise for $\hat u_p$ |
| **E4** | **the only accuracy gate**, with its bars unchanged, in both frames |
| KT | kept; it follows E4 |
| E1 | **a diagnostic, not a gate**: $e_H$ reported by column group (hats, $j=1$–3, $j=4$–8, strips). **The rule for P7:** a piece-level bar may be registered only above the floor the form reaches with exact smooth corrections, measured untrained first |
| SE3 | **reported, not gated**. If $\mathcal C$ brings the network term below the 16-mode truncation floor, the ports, not the expert, are the next lever |
| E5 / price | against the exact classical solve, against the whole layout solved classically at $n/2$, and against H2r (0.306), under R2's price protocol |
| S11 | reported |

### 3.6 Cost (estimates, to be priced in Phase 2)

Per piece and call, at $n=128$, $n_c=16$:
- the fine residual $BQ-A_IX$ for 66 columns: about $66\times9\times127^2\approx10^7$ flops;
- the coarse matrix: once per piece, a sparse triple product linear in the fine unknowns;
- its Cholesky factorisation: about $225^3/3\approx4\times10^6$ flops;
- the solves: about $2\times225^2\times66\approx7\times10^6$ flops;
- the prolongation: linear.

**[AI Inference]:** that is well under a U-Net forward pass, and far under the classical hats (B1: 0.44–0.51 of an exact solve, dominated by the fine LU) or the strips (0.21–0.44). The coarse factorisation is reused across calls: modes once per piece, the particular field once per step. In training, $\mathcal C$ sits in the graph. The matrix–vector product uses the element stencil of `model.energy`, and the coarse solves use a batched float64 Cholesky.

---

## 4. How each problem is answered

| # | mechanism | status |
|---|---|---|
| P1 | $\Pi$ kept; Lemma B shows $\mathcal C$ preserves the identity, with $b_1$ from the operator's row sums | proven; K checked in Phase 1 |
| P2 | split form kept; $\mathcal C$ is linear in each field separately | proven; L checked in Phase 3 |
| P3 | corner bubble and strips kept; $\mathcal C$ does not treat corners or interface–side points | **open**: Phase 2 measures whether they limit E4 (§5, deferred items) |
| P4 | the coarse component of every mode's error is zero by construction; smooth global responses are mostly coarse | **to be measured** (Phase 2, predictions 3–4) |
| P5 | the network is no longer asked for global responses; by Proposition A it cannot change them | structural; reach no longer needed |
| P6 | the loss sees only $\mathcal Q_cE$; trained corrections cannot move the coarse component. **Limit:** the corner-local part of a hat mode is not coarse, so a network could still damage it | structural, with that limit |
| P7 | E1 demoted to a diagnostic; bars only above a measured floor | a rule of the registration |
| P8 | the shear, stretch and material are corrected at the coarse scale by the true operator; $H^{(0)}$ remains only as the fine-scale start | **to be measured** (Phase 2) |
| P9 | TL0 has no fine factorisation; it is priced against H2r and the coarse whole-layout solve | **to be measured** (Phase 2, then R2's protocol) |
| P10 | $\mathcal C_p$ removes the global component of $u_p$'s error. **[AI Inference]:** a local residual should generalise better than a global one | to be measured (Phase 3) |

---

## 5. What is deliberately not done, and why

| not done | why |
|---|---|
| attention, or a global path | reach is not the limiter: at $n=32$ the U-Net sees the whole piece and $e_{H,L}$ was still 0.25, and the global path B3 was a tie (CG). Under the rework the network's job is local, where a convolution's bias is right |
| wavelet operators | the strips already make the fast sines exact, and the coupled answer did not move (CW, a tie). Better high frequencies are not what E4 lacks. Kept for CO-T and the plate, where sharp features carry the answer |
| a new chart (conformal with its corner factor split off, or a log-polar corner blow-up) | it addresses P3, the corners. Deferred until Phase 2 shows the residual's floor sits at non-right corners |
| B1, classical hats | subsumed by $\mathcal C$, without the fine LU that made B1 cost half an exact solve |
| B2, the mode prior | not needed once the loss sees only $\mathcal Q_cE$ |
| a bigger network, more data, regularisation | the modes underfit; they do not overfit. C5 chose 0.28 M |
| any change to the host, Theorem 2 or the coupling | none is needed: the output is trace-exact fields, as before |

---

## 6. Long-term: why a coarse level, and where learning goes next

- **It stays inside the decomposition.** Each piece's coarse solve is its own, and the coupling is still the one interface solve.
- **It adds no new dependency.** The host already needs each piece's $M$ for the Gram, which is what makes a wrong network safe.
- **Its cost does not grow with the fine resolution:** about $15^2$ unknowns in 2-D and about $7^3$ in 3-D.
- **It is how scalable solvers handle the global part:** multigrid, domain decomposition's coarse spaces, multiscale finite elements (Hou & Wu, *J. Comput. Phys.* 134, 1997) and localized orthogonal decomposition (Målqvist & Peterseim, *Math. Comp.* 83, 2014).

**Where a bilinear coarse space fails:**
- features finer than a coarse cell (thin walls, small holes, high-contrast inclusions);
- advection-dominated transport (CO-T at high Péclet);
- oscillatory physics.

There the network should move **into the coarse space**: a learned prolongation $P_\theta$, the shape of each coarse function bent around the piece's features. Learned multigrid prolongation is studied, for example by Luz et al. (ICML 2020). **[AI Inference]:** Proposition A holds for any full-rank $P$. With a learned $P_\theta$, the output is still a Galerkin correction, never worse column by column than its input, and every guarantee above stands. Learning the coarse basis is therefore as safe as learning the fine residual. It plays to a network's strength (a local, structured shape) and avoids its demonstrated weakness (an accurate global response). It is the natural Stage 1 step wherever Phase 2's kind of control shows the bilinear coarse space failing.

---

## 7. What changes in the code

The R2 pin (`out/chartop/registered-squares.txt`) hashes `forms.py`, `expert.py`, `data.py`, `host.py`, the benchmarks and every earlier gate module, so **none of them is edited.** Everything new is a new module or a default-off option.

| file | new or changed | what |
|---|---|---|
| `atlas/chartop/coarse.py` | new | `prolongation(n, n_c)` (sparse, interior to interior); `CoarseSpace(A_I, n, n_c)` with $A_c$'s Cholesky factor and `correct(X, R)`; `level_rhs(M, s0)` giving $b_1=(M\mathbf 1)_I/s_0$; torch counterparts: the element-stencil product $A_IX$ from the batch's arrays, batched coarse Cholesky, and $\mathcal C$ in the graph |
| `atlas/chartop/twolevel.py` | new | `TwoLevelExpert`, a subclass of `forms.KernelExpert`: `super().forward`, then $\mathcal C$ on the modes, the level field and the particular field. `n_c=None` returns S's answer bit for bit |
| `atlas/chartop/gates_twolevel.py` | new | bars read from `gates_forms`; CX; E1 and SE3 as reports; E4 and KT; the price comparisons of §3.5 |
| `atlas/chartop/train.py` | changed (not pinned) | `Arm.coarse` (default `None`, bit for bit as before) and loading `TwoLevelExpert` |
| `scripts/chartop_twolevel_controls.py` | new | Phase 2, untrained |
| `scripts/chartop_twolevel_overfit.py` | new | Phase 3's probe, on B4's four pieces, importing `chartop_overfit_probe.pieces` |
| `tests/test_chartop_coarse.py`, `tests/test_chartop_twolevel.py` | new | Proposition A and Lemma B numerically; numpy against torch; the pins unchanged |

---

## 8. The implementation workflow

Each phase ends at a decision, and nothing past Phase 2 starts unless Phase 2 says it should.

### Phase 0 — Agreement (the owner; no code)

- Accept or amend this draft.
- Choose the coarse sizes to try, proposed as $n_c\in\{8,16,32\}$.
- Agree in principle to §3.5's gates: E4 the only accuracy gate, E1 and SE3 reported, CX added.
- **Done when** the owner accepts.

### Phase 1 — The coarse level, built and tested (no network, no training)

- **Build:** `coarse.py` and `tests/test_chartop_coarse.py`.
- **Done when:**
  - Proposition A holds numerically: optimality against a brute-force least-squares fit over $V_c$; idempotence; $\mathcal C(X+Pv)=\mathcal C(X)$; $n_c=n$ exact to $10^{-10}$; traces bit for bit.
  - Lemma B holds: K to rounding on step pieces, with $b_1$ from the row sums.
  - numpy and torch agree to $10^{-12}$.
  - The full suite and every pin pass unchanged.
- **Cost:** coding and the suite, on the laptop.

### Phase 2 — TL0 measured untrained: the decisive phase

- **Before running:** register the predictions of §9 in the script, with the owner's go.
- **Measure**, never on any test split:
  - $e_H$ by column group at $n_c\in\{8,16,32\}$ and $n\in\{32,64,128\}$, on diagnose's validation pieces (seeds 2,000,100 onward), on B4's four pieces and on W362's piece types (rectangles, ring pieces);
  - E4 in both frames, with A1's group split of what remains, on the six Stage 0 benchmarks;
  - the cost against the exact solve, on the laptop on AC, as a first reading.
- **Positive controls:** $n_c=n$ is exact; $n_c=$ `None` reproduces U bit for bit.
- **Decision (the owner):**
  - **2a. E4 met on all six.** TL0 is the classical CO-D expert to price against H2r under R2's protocol, on a rented box. Training for CO-D stops. The network work moves to Stage 1 (§6).
  - **2b. E4 missed, and what remains is local and outside $V_c$** (A1's split). Go to Phase 3.
  - **2c. E4 missed, and what remains sits at non-right corners or interface–side points.** Draft the deferred corner treatment (singular functions in the bubble, or the chart), not a network.

### Phase 3 — The network on the residual, overfit probe first

- **Build:** `twolevel.py`, with $\mathcal C$ in the graph, and its tests: CX, K, D and L; `n_c=None` equals S bit for bit; training, saving and loading.
- **Probe:** B4's protocol (four pieces, $n=128$, S's weights as the start, AdamW at $3\times10^{-4}$, 3,000 steps, deterministic). Report the fine residual's $e_H$ by group and the coupled error on the four pieces.
- **Done when** the network lowers the residual on its own pieces without raising any group's coarse error. CX makes that structural, and the probe checks it. **If it cannot fit four pieces, stop: no registration.** `scripts/chartop_smooth_target.py`, written on 2026-11-09 and stopped at step 0, can separate an optimisation limit from a target limit here.

### Phase 4 — Registration (the owner accepts; hash-pinned)

- **Arms:** TL (trained) and TL0 (its untrained floor), seeds 0 and 1, at the comparison budget, with S′ as the reference.
- **Gates:** §3.5.
- **Predictions:** written before training.
- **Price protocol:** dry-run first, per the pinned-conditions discipline.

### Phase 5 — Train and evaluate once

- Rented boxes at no more than \$0.70 an hour, each destroyed after its weights are verified.
- One evaluation on the laptop on AC.
- Every verdict reported whatever it says.

### Phase 6 — Write-up and carry-over

- The record goes on this page, with the [[gap-worklist]] row and the [[log]].
- Stage 1's CO-E takes $\mathcal C$ with a vector bilinear $P$.
- A learned $P_\theta$ (§6) comes only where a Phase-2-style control shows the bilinear coarse space failing.

---

## 9. Predictions to register in Phase 2 (drafted, not registered)

1. Proposition A and Lemma B hold to rounding on every piece: CX at most $10^{-12}$, no column's $e_H$ above U's, and $n_c=n$ exact.
2. bend-3, steady, meets E4 at every $n_c$, as U alone already does.
3. **[AI Inference]:** at $n_c=16$, the low group's $e_H$ (hats and $j=1$–3) is at most half of U's on the validation pieces.
4. **[AI Inference]:** s-channel and insert-round, steady, meet E4's energy bar untrained at $n_c=16$ or $32$. Their error lay in the hats and $j=1$–3 coming from $H^{(0)}$'s global response (forms §11.1), and those responses are smooth.
5. **[AI Inference]:** the steps' **maximum**-error bar is the most at risk, because the one-step boundary layer is thinner than a coarse cell. Refuted if all three steps meet it at $n_c=16$.
6. **[AI Inference]:** at $n=128$, $n_c=16$, the coarse level costs under 0.1 of an exact solve.

Predictions 3–6 can be refuted, and each is read on the registered measure. The readings the Phase 2 script applies are in §12.

---

## 10. What this page does not show

- **Phase 2 has not run.** The coarse level is built and tested (§11), but nothing here measures TL0 on the pieces Phase 2 reads. Every accuracy number cited comes from earlier records, and §9's predictions are not results.
- **No price against H2r.** TL0's cost is estimated, not measured.
- **Corners and interface–side points** are not treated by $\mathcal C$ (P3). Whether they matter is Phase 2's question.
- **The smooth-target control** (`scripts/chartop_smooth_target.py`, 2026-11-09) was stopped at step 0. Its step 0 reproduced B4's 0.3356. It has not been run further. The script is committed with this page, with that step-0 record (`out/chartop/smooth-target-exact.json` and `.log`), and no test covers it.
- **Nothing on the website or in `proposal/`.**

---

## 11. Phase 1: the coarse level, built and tested (2026-11-09)

**Built:** `atlas/chartop/coarse.py`, and `tests/test_chartop_coarse.py` with 69 tests. No pinned module was edited.

| code | what it does |
|---|---|
| `prolongation(n, n_c)` | $P=p\otimes p$, with the 1-D coarse hats $p_{iJ}=\max(0,1-\lvert i/n-J/n_c\rvert\,n_c)$ computed from integers, in `fe.grid`'s interior order |
| `residual(M, n, X, Q, rhs)` | $r=\text{rhs}-(Mu_{\text{full}})_I$, the residual from the full field: $BQ-A_IX$ for the modes, without forming $B$ |
| `CoarseSpace` | $A_c=P^\top A_IP$ and its Cholesky factor; `correct(X, r)` $=X+PA_c^{-1}P^\top r$. At $n_c=n$, the positive control, $A_c=A_I$ is kept sparse and factored by a sparse LU |
| `CoarseSpace.of_matrix(...).apply` | the same correction from the **restricted residual** $P^\top\text{rhs}-G_IX-G_G\,(\text{loop values})$, with $G=P^\top M_{I,:}$ split into its interior and loop columns, and $A_c=G_IP$. No fine residual and no full field are formed |
| `level_rhs(M, n, s0)` | $b_1=(M\mathbf 1)_I/s_0$ for $s_0>0$; a steady piece is refused |
| `coarse_exactness` | CX: $\lVert P^\top r_{\text{out}}\rVert/\lVert P^\top F\rVert$ over the columns; with no data, no ratio |
| `correct_fields`, `CoarseCorrected` | one piece's two-level answer from any answer: $\mathcal C(H)$; $\mathcal C_p(u_p)$ against the piece's load; $\mathcal C_1(u_1)$ on a step. The adapter wraps any expert for the host: around U it is TL0, and with no coarse size it returns U bit for bit |
| `apply_M`, `residual_torch`, `level_rhs_torch`, `coarse_matrix_torch`, `TorchCoarseSpace` | torch, batched, float64. $MU$ comes from `model.energy`'s element formulas, by slice additions (no `scatter_add`). $A_c$ is assembled by nine colourings of the coarse hats, then a batched Cholesky; $\mathcal C$ is differentiable |

**Two definitions settled while building.**
- **The particular field is corrected against the piece's own load $b_I$**, as §3.2 states. So $\hat u_p$ is coarse-exact, and exact at $n_c=n$.
  - The uniform state's particular field is then $\mathcal C_p(s_0u_1)$ against $s_0\,\text{mass}(\tilde J)\mathbf 1$. It equals $s_0\,\mathcal C_1(u_1)$ only to the rounding by which $\sigma$ differs from $s_0\tilde J$.
  - On a `PieceProblem` that is one float64 rounding, and K holds. On arrays stored in float32 it is about $10^{-8}$.
  - K was never measured that way on stored arrays: on test pieces the registered K read the level-field identity $\hat Hc_1+s_0\hat u_1-1$ (`chartop_evaluate_forms.e1_rows`), which Lemma B keeps exact.
- **A steady piece's level field is passed on uncorrected.** It is the unit source's response, which only a march reads, and the identity does not involve it when $s_0=0$.

**The tests**, on generator pieces from seeds no registration draws (9,000,100 onward: Fo 0.63, Fo 0.0117 and steady), at $n=16$ and 32, read from their stored arrays:

| test | what it shows | to |
|---|---|---|
| the prolongation | each column is the bilinear hat of its coarse node at `fe.grid`'s nodes, zero on the loop; every entry exact | bit for bit |
| the residual and `apply_M` | $BQ-A_IX$; `fe.assemble(...) @ U`; half the gradient of `model.energy` | $10^{-13}$; $10^{-12}$ |
| the restricted residual | equal to the fine residual restricted, the sparse top level included | $10^{-12}$ |
| Proposition A, item 1 | CX of the modes and of a field after $\mathcal C$ at $n_c=2$, 4 and 8; the input's CX was above $10^{-3}$ | $10^{-12}$ |
| item 2 | never worse, column by column; equal to a brute-force least-squares fit over $V_c$ (a dense Cholesky of $A_I$); Pythagoras | $10^{-10}$ |
| item 3 | idempotence; $\mathcal C(X+Pv)=\mathcal C(X)$ | $10^{-13}$; $10^{-12}$ |
| items 4 and 5 | at $n_c=n$, the exact modes, particular field and level field; traces $Q$ | $10^{-10}$; bit for bit |
| item 6 | the gradient through $\mathcal C$ is $\mathcal Q_c^\top G$ | $10^{-12}$ |
| Lemma B | the identity survives with $b_1$ from the row sums on steps, and the hats' sum on a steady piece | $10^{-13}$ |
| Lemma B's condition | with $b_1=\text{mass}(\tilde J)\mathbf 1$ the identity misses by $10^{-9}$–$10^{-6}$ on a short step's stored arrays, and holds on the same piece in float64 | as §3.2 states |
| numpy against torch | $A_c$, the residual, the corrections of the modes, of a field and of the level field | $10^{-12}$ |
| the symmetries | $\mathcal C$ commutes with all eight, in numpy and through `model.d4` | $10^{-12}$ |
| the adapter | with no coarse size, U bit for bit; at $n_c=n$, the classical solve; its answers coarse-exact, and kernel-exact through the host's uniform state | $10^{-10}$; $10^{-12}$ |

**Beyond the tests**, at $n=128$, on three never-drawn pieces with U's own answers (one-off, not in a record): after $\mathcal C$ the identity holds within $6\times10^{-14}$, CX within $8\times10^{-15}$ and the particular field's CX within $4\times10^{-14}$, at every $n_c$ from 4 to 32. Prediction 1's bars are $10^{-12}$. On stored arrays, the $\text{mass}(\tilde J)$ variant missed by $5\times10^{-11}$ at $s_0=0.017$, $4\times10^{-10}$ at $s_0\approx1$ and $3\times10^{-8}$ at $s_0=79$: the float32 rounding, growing with the reaction.

**The cost, timed before prediction 6 was registered.** One-off: two never-drawn pieces at $n=128$, on battery, so only the ratios are read. The coarse level at $n_c=16$, from the assembled matrix, took 0.085–0.104 of the exact solve with the fine residual, and 0.062–0.067 with the restricted one. The arithmetic was not the cost: the time went into slicing the whole matrix twice and allocating full fields. With the restricted form, $n_c=4$, 8 and 32 read 0.047–0.051, 0.051–0.063 and 0.18–0.19. None of this is prediction 6's measure, which is §12's, on AC.

**The suite and the pins** (on AC, 2026-11-09):
- The full suite ran on an export of the last commit with these files laid over it: 2,667 passed, 129 skipped, 14 failed and 3 errors.
- All 17 came from the export itself. `git archive` here rewrites line endings (`core.autocrlf=true`), so the squares pin's byte comparisons with its dry run failed; and nine tier tests read an untracked record, `out/tier0b/s0_state.npz`, which no export carries.
- All 17 pass in the working tree, which no other session had changed, and so do the 103 tests of every gate module and pin.

---

## 12. Phase 2: the script and the predictions' readings (drafted for registration, 2026-11-09)

**The script:** `scripts/chartop_twolevel_controls.py`. It has been smoke-tested (below) and not run on any piece it will measure. TL0 is U (`chartop_forms_diagnose.zero_heads`) followed by `coarse.correct_fields`. Records go to `out/chartop/twolevel/`, rewritten after every measurement, so a stopped run resumes.

| part | pieces | recorded |
|---|---|---|
| `modes` | the post-hoc validation pieces 2,000,100–2,000,163 at $n=32$, 64 and 128, labelled from their stored arrays; B4's four (the first four at 128); W362's types: rectangles of aspect 1 down to 0.5, a hole's and a round inclusion's rings, the held-out benchmarks' curved pieces | for U and TL0 at each $n_c\le n$: $e_H$ by group (hats, $j=1$–3, 4–8, 9–15) and for the low group (hats with $j=1$–3); $e_H$; $e_{H,L}$; $e_p$; $e_1$; $e_K$ and $e_{K,\infty}$; CX of the modes and of the particular field; the column-by-column check against U |
| `e4` | the six Stage 0 benchmarks at $n=64$, both frames | U, TL0 at $n_c=4$, 8, 16 and 32, and the two controls ($n_c=64$; no coarse size). E4, D and K by `gates_forms`' own judges; CX per piece; the network term split by mode group (A1's `split_term`) and by where it sits (below) |
| `cost` | the first 8 validation pieces at each $n$ | the coarse level from the assembled matrix, against `fe.Exact` (the LU, 64 mode columns, the particular field), assembly excluded from both, best of 5; U's forward and strips beside. Torch and the BLAS at 4 threads; the power sampled before and after every resolution; the part refuses to start on battery |

**Why a split by place.** Decision 2b against 2c (§8) turns on where TL0's remaining error sits: spread and local, or at non-right corners and interface–side points. A1's split says which modes carry it, not where.
- The regions, per element and in this order: each $n/8\times n/8$ corner block, right or not right by more than 2° (`chart.corner_angles`); within $n/8$ of a side and 2 cells of a material interface (the forms page's §11.5 definitions); at an interface inside; the side layers; the interior.
- The element energies of Theorem 2's per-piece term $w_i=\mathcal Q_c(e_{p,i}+E_ic^\star_i)$ add up exactly to the network term (to $10^{-15}$ in the smoke run), so the shares are exact.
- Below $10^{-9}\lVert u\rVert_M$ the network term is at the floor, and no share is formed.

**The predictions, with the readings `judge` applies** (§9's text unchanged):

| # | prediction | reading |
|---|---|---|
| 1 | Proposition A and Lemma B hold to rounding on every piece: CX at most $10^{-12}$, no column's $e_H$ above U's, and $n_c=n$ exact | For TL0 at every $n_c$, on every piece of `modes` and every benchmark piece in both frames: no column whose energy-norm error exceeds U's by more than $10^{-9}$ of U's plus $10^{-12}\sqrt{\Lambda_{kk}}$; and where $n_c<n$, CX of the modes $\le10^{-12}$, and of the particular field wherever the piece has a load. $e_K$ and $e_{K,\infty}\le10^{-12}$ for TL0 at every $n_c$: the level-field identity on every piece of `modes`, the uniform state through the host on every benchmark piece as registered. At $n_c=n$: $e_H\le10^{-10}$ on every $n=32$ validation piece, and each benchmark's energy error within $10^{-10}$ of the 16-mode floor in both frames |
| 2 | bend-3, steady, meets E4 at every $n_c$, as U alone already does | `judge_e4_both_frames` met for TL0 at $n_c=4$, 8, 16 and 32 |
| 3 | **[AI Inference]** at $n_c=16$, the low group's $e_H$ is at most half of U's on the validation pieces | at each $n$, the median over the validation pieces of TL0's low-group $e_H$ is at most 0.5 times U's median |
| 4 | **[AI Inference]** s-channel and insert-round, steady, meet E4's energy bar untrained at $n_c=16$ or 32 | for each, the energy error $\le0.02$ in both frames, at $n_c=16$ or at $n_c=32$ |
| 5 | **[AI Inference]** the steps' maximum-error bar is the most at risk; refuted if all three steps meet it at $n_c=16$ | at $n_c=16$, at least one step problem misses the maximum-error bar in at least one frame |
| 6 | **[AI Inference]** at $n=128$, $n_c=16$, the coarse level costs under 0.1 of an exact solve | the median over the cost part's $n=128$ pieces is below 0.1; judged only if every power sample read AC |

**Checked against each other and against the code.**
- **1 and 2 agree.** Per piece, TL0's term is $w_i^{\text{TL0}}=\mathcal Q_cw_i^{\text U}$, so its energy cannot exceed U's (Proposition A, item 2). Theorem 2 then bounds TL0's energy error by the floor plus a network term no larger than U's, and U's error on bend-3 steady is 0.0003%. The maximum-error bar is not bounded this way.
- **3 and 4 agree without implying each other.** **[AI Inference]:** the s-channel's error is 93% in $j=1$–3 (the forms page's §11.1), so halving the low group alone might leave it near 2.6%, above the bar; 4 also allows $n_c=32$.
- **4 and 5 read different problems and different bars**, and 5 is the only prediction of a miss.
- **6 is weak as drafted.** §11's one-off timing read 0.062–0.067 on battery, before registration. It can be kept as drafted, or tightened.
- `coarse.py`'s docstring and `forms.py`'s contradict none of them. On a square of one material U is exact to float32 (the forms page's §11.19), and $\mathcal C$ cannot make it worse.

**The smoke run** (`--smoke`): seeds 9,000,100–9,000,101 at $n=32$, the `benchmarks_squares` layout of seed 9,000,008 at $n=16$, a scratch directory, the controls only. Every part ran.
- At $n_c=n$: $e_H\le7\times10^{-16}$, and the coupled error within $7\times10^{-14}$ of the floor.
- With no coarse size: U bit for bit.
- CX within $2\times10^{-15}$; TL0's $e_{K,\infty}$ within $7\times10^{-15}$; both splits sum to the network term within $10^{-15}$.

**Its tests**, `tests/test_chartop_twolevel_controls.py` (30): `judge` on made-up records, where every prediction holds and then each clause is broken in turn and refutes it; each reading's bars found both in the script and in this section's table, and the script's prediction texts in §9 word for word; the guards (the registered run keeps its seeds and record, the cost part refuses battery, a record of another configuration is refused); the regions partition the elements; and a smoke run at $n=16$ that checks the controls and resumes without measuring again.

**Running it.** On the laptop on AC: `modes` by resolution and `e4` by problem in parallel processes, about 15 minutes, then `cost` alone, about 3 minutes. A rented box would need a new payload, since the box payload scripts are pinned, and the cost reading belongs on the laptop.

**Then the owner's decision:** 2a, 2b or 2c, as §8 states, read from E4 by $n_c$, A1's split and the split by place.

---

## See Also

- [[chart-operator-stage-0]] — what was built, judged once, and diagnosed
- [[chart-operator-kernel-exact-forms]] — W356–W358, the overfit probe (§11.6), A1's group split (§11.1) and R2's hybrid (§11.19)
- [[chart-operator-architecture]] — the design; Theorem 2, which the rework leaves unchanged
- [[chart-operator-training-and-cost]] — T26's label-free objective, kept
- [[chart-operator-general-form]] — the guarantee classes; the frozen-coefficient principal part (C1), the costlier alternative to $\mathcal C$
- [[chart-operator-family-catalog]] — CO-E, CO-T and the plate, where §6's learned coarse basis would come in
- [[coupling-cost-and-complexity]] — the measured costs E5 is read against
- [[dd-neural-prior-art-2026]] — neural domain decomposition prior art
- [[gap-worklist]] — W362
