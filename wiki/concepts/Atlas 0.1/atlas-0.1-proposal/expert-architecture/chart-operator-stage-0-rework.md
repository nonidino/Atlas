# Chart operator — Stage 0 reworked: the two-level, coarse-exact expert (built and judged once)

**Type:** Concept page — **design and build record**: drafted, built, trained and judged once (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** drafted on 2026-11-09 from a design conversation with the owner, who then asked for it to be built end to end. **Phase 1 is built and tested** (§11): the coarse level, `atlas/chartop/coarse.py`, with Proposition A and Lemma B checked numerically. **Phase 2's script is written and smoke-tested** (§12); its predictions (§9), with their readings there, were **registered on the owner's go** the same day, before Phase 2 ran. **Phase 2 has run** (§13). Untrained, TL0 meets E4 on 4 of the six benchmark problems at $n_c=16$ and on 5 at $n_c=32$; the sixth no 16-mode expert can meet. Where it misses, 85–90% of what remains sits at non-right corners. Five predictions held, and prediction 5 was refuted. The owner chose branch 2c: a corner treatment, an enlarged coarse space at the non-right corners, was drafted, accepted and built (§15, §16). Its untrained check ran with five registered predictions, and all held (§17). Its candidate B, the corner blocks' fine hats, meets E4 untrained on every problem a 16-mode expert can meet. The owner adopted it for Phase 4, and its training path is built. Phase 3's overfit probe ran on the laptop (§14): on its own pieces the network takes $\rho$ to 0.95 from S's weights. Phase 4's registration was accepted and, after its price protocol's dry run passed on a rented machine (\$0.32), hash-pinned (§18). **Phase 5 trained both runs** on a rented GPU box (\$1.08) **and ran the one evaluation** (§19), pricing it on a rented 5900X (\$0.35). Every structural gate was met. On the test pieces, the network removes two-thirds of TL0's error on the learned columns at the median, almost all of it in $j=4$–8. On the held-out problems, TL meets E4 on 4 of 6, as untrained TL0 does, and is worse than TL0 on 4 of 6: **E4 and NC are not met**. TL costs 1.05 of a classical solve and TL0 0.84. Predictions 1, 4 and 5 held; 2 and 3 were refuted. The coarse level carries the coupled gain. **Phase 6** wrote the rework up on the owner's go (§20, 2026-11-12), and [[gap-worklist]] W363 records it as done. No box is up. The page sets out how Stage 0's CO-D expert should be formulated, given everything Stage 0 and its successors measured. It gives a phased workflow whose decisive phase needs no training.
**Hub:** [[chart-operator-architecture]] · **Follows:** [[chart-operator-stage-0]] and [[chart-operator-kernel-exact-forms]] · **Worklist:** [[gap-worklist]] W363 (this rework, done) · W362 (open) · **Family:** [[chart-operator-family-catalog]] (CO-D)

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

## 9. Predictions for Phase 2 (registered 2026-11-09, before Phase 2 ran)

**Registered** by the owner on 2026-11-09, in chat ("Register as drafted (Recommended)"), as drafted here with the readings of §12. The one-off timing that makes prediction 6 weak (§11) was disclosed before the go. The registration is the commit that adds this paragraph; no part of Phase 2 had run on the pieces it measures.

1. Proposition A and Lemma B hold to rounding on every piece: CX at most $10^{-12}$, no column's $e_H$ above U's, and $n_c=n$ exact.
2. bend-3, steady, meets E4 at every $n_c$, as U alone already does.
3. **[AI Inference]:** at $n_c=16$, the low group's $e_H$ (hats and $j=1$–3) is at most half of U's on the validation pieces.
4. **[AI Inference]:** s-channel and insert-round, steady, meet E4's energy bar untrained at $n_c=16$ or $32$. Their error lay in the hats and $j=1$–3 coming from $H^{(0)}$'s global response (forms §11.1), and those responses are smooth.
5. **[AI Inference]:** the steps' **maximum**-error bar is the most at risk, because the one-step boundary layer is thinner than a coarse cell. Refuted if all three steps meet it at $n_c=16$.
6. **[AI Inference]:** at $n=128$, $n_c=16$, the coarse level costs under 0.1 of an exact solve.

Predictions 3–6 can be refuted, and each is read on the registered measure. The readings the Phase 2 script applies are in §12.

---

## 10. What this page does not show

*Revised 2026-11-12, at Phase 6.* The draft's first three items held until later sections answered them:
- Phase 2 is untrained. §19 trained TL, drew the test split and priced it under R2's protocol.
- No price against H2r. §19.3 priced TL0 and H2r.
- $\mathcal C$ does not treat corners. §17 treated the non-right ones.

What the page still does not show:
- **One budget, one start, one coarse size.** TL was trained twice, seeds 0 and 1, each for 37,500 steps from S's weights, at $n_c=16$ with B's patches. No other budget, start, objective or coarse size was trained. TL-s0's validation $\rho$ was still rising at the end: 0.644 at step 30,000, 0.661 at 37,500 (§19.1).
- **One run priced, on one machine type.** PR priced TL-s0, as registered, on a rented Ryzen 9 5900X.
- **TL0 at $n_c=32$ is not priced.** It meets E4 on 5 of the 6 held-out problems (§19.2). On the laptop, its coarse level without the patches cost 0.61 and 0.15 of an exact solve at $n=64$ and 128 (§13).
- **Stage 0's layouts, not W362's.** The held-out problems are Stage 0's three layouts with new materials and end profiles. No layout whose outline does not cut into squares was judged, so W362's registration is not answered.
- **Interface–side points are untested.** No benchmark piece has a material interface inside it (§13).
- **No loss that weights the modes as the coupled answer does** was trained with $\mathcal C$. TL's objective is T26's energy, relative to TL0's (§14).
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

## 12. Phase 2: the script and the predictions' readings (registered 2026-11-09)

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

## 13. Phase 2: TL0 measured untrained (2026-11-09)

**What ran.** `scripts/chartop_twolevel_controls.py` at the registration's commit (`84e1ab7`), on the laptop on AC.
- `modes` and `e4` ran in three processes in 375 s: 64 validation pieces and 88 W362 pieces at each of $n=32$, 64 and 128 (456 in all), and the six benchmark problems in both frames.
- `cost` ran alone afterwards. All 7 power samples read AC, and the only other Python process was the idle phone-remote server.
- Records: `out/chartop/twolevel/` (`modes-*.json`, `e4-*.json`, `cost-laptop.json`, `verdicts.json` and `.txt`). No weights were read and no test piece was drawn.

**The controls held.** With no coarse size, the adapter answered U's bits on every benchmark problem in both frames. At $n_c=64$, TL0 reproduced the 16-mode floor within $8\times10^{-14}$ on all six problems. At $n=32$, $n_c=32$ gave $e_H=0$ to the printed digits on every piece.

**The predictions, as registered:**

| # | verdict | measured |
|---|---|---|
| 1 | **held** | over 456 pieces and six problems: CX at most $2.3\times10^{-14}$ (benchmarks $7.6\times10^{-15}$); the particular field's at most $8.1\times10^{-14}$; $e_{K,\infty}$ at most $2.7\times10^{-13}$; no column worse than U's; $n_c=n$ exact |
| 2 | **held** | bend-3 steady met E4 at every $n_c$ |
| 3 | **held** | the low group's median over U's: 0.26, 0.28 and 0.28 at $n=32$, 64 and 128 |
| 4 | **held** | s-channel steady 1.41% at $n_c=32$ (2.15% at 16); insert-round steady 0.57% at 16 |
| 5 | **refuted** | at $n_c=16$ all three steps met the maximum-error bar: 0.42%, 3.45% and 0.051% of the range, against 1.42%, 3.81% and 1.05%. The energy bar lowered by 300 K was the one at risk |
| 6 | **held** (weak, §11) | median 0.050 at $n=128$, $n_c=16$ (0.034–0.088 over 8 pieces) |

**E4 on the benchmarks.** Energy error in %, as registered / lowered by 300 K (one number when steady, where the frames agree); ✓ met, ✗ missed. D and K were met by every variant on every problem.

| problem | 16-mode floor | U | TL0, $n_c=4$ | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| bend-3, steady | 0.0000 | 0.0003 ✓ | 0.0003 ✓ | 0.0003 ✓ | 0.0003 ✓ | 0.0003 ✓ |
| bend-3, step | 0.017 / 0.25 | 0.27 / 3.89 ✗ | 0.19 / 2.74 ✗ | 0.11 / 1.58 ✓ | 0.057 / 0.83 ✓ | 0.030 / 0.44 ✓ |
| s-channel, steady | 0.93 | 5.26 ✗ | 4.06 ✗ | 3.08 ✗ | 2.15 ✗ | 1.41 ✓ |
| s-channel, step | 0.14 / **2.13** | 0.68 / 10.75 ✗ | 0.54 / 8.50 ✗ | 0.42 / 6.56 ✗ | 0.29 / 4.59 ✗ | 0.19 / 3.07 ✗ |
| insert-round, steady | 0.031 | 5.58 ✗ | 2.18 ✗ | 1.14 ✓ | 0.57 ✓ | 0.26 ✓ |
| insert-round, step | 0.0066 / 0.070 | 0.39 / 4.14 ✗ | 0.16 / 1.63 ✓ | 0.081 / 0.86 ✓ | 0.041 / 0.43 ✓ | 0.019 / 0.20 ✓ |
| **met** | | 1 of 6 | 2 of 6 | 4 of 6 | 4 of 6 | **5 of 6** |

- **The sixth cannot be met on this basis.** As the forms page's §11.14 records, the s-channel step's floor lowered by 300 K is 2.134%, above E4's 2% energy bar, so no expert with 16 modes per side meets E4 there. TL0 at $n_c=32$ misses it only in that frame's energy (3.07%). As registered it errs by 0.19%, and its maximum error is 3.02% of the range against a bar of 3.81%.
- **So §8's branch 2a, read literally, cannot happen.** TL0 at $n_c=32$ meets every E4 bar that a 16-mode expert can meet; at $n_c=16$ it misses only the s-channel, steady by 0.15 points.
- **Maximum errors.** U missed the maximum bar on all three steps (2.60%, 5.96% and 2.33% of the range). TL0 met it from $n_c=8$ on bend-3 and $n_c=4$ on insert-round, and from $n_c=16$ on the s-channel. At $n_c=4$ the s-channel step's maximum error was worse than U's (6.12% against 5.96%): Proposition A bounds energies, not maxima.

**The error above the floor, exactly.** The 16-mode classical answer $u_m$ solves each piece exactly for its traces, and the coupled solve makes its error energy-orthogonal to every field whose traces lie in the port space. Every expert's coupled answer $\tilde u$ is such a field, so
$$\lVert u-\tilde u\rVert_M^2=\lVert u-u_m\rVert_M^2+\lVert u_m-\tilde u\rVert_M^2 ,$$
and the excess $\lVert u_m-\tilde u\rVert_M/\lVert u\rVert_M$ is what the expert itself gets wrong:

| problem | U | $n_c=4$ | 8 | 16 | 32 | ratio per doubling of $n_c$ |
|---|---|---|---|---|---|---|
| s-channel, steady | 5.18 | 3.95 | 2.94 | 1.94 | 1.06 | 0.74, 0.66, 0.55 |
| s-channel, step, lowered | 10.54 | 8.23 | 6.21 | 4.06 | 2.20 | 0.75, 0.65, 0.54 |
| bend-3, step, lowered | 3.88 | 2.72 | 1.56 | 0.80 | 0.36 | 0.57, 0.51, 0.45 |
| insert-round, steady | 5.58 | 2.18 | 1.14 | 0.57 | 0.26 | 0.52, 0.50, 0.45 |

To meet E4 on the s-channel steady problem, the excess must fall to 1.77%. At $n_c=16$ it is 1.94%.

**Where the excess sits.** A1's split by mode group, and the split by place (§12), for TL0 where it misses E4 and, beside it, where it meets it (the steps lowered by 300 K). Each row of A1's shares sums to one; the strips' share is under 0.01 everywhere. On a step the hats' level part is the level field's error: both $\hat H c_1+s_0\hat u_1$ and $Hc_1+s_0u_1$ equal one, so $(\hat H-H)c_1=-s_0(\hat u_1-u_1)$.

| problem | variant | $j=1$–3 | $j=4$–8 | hats' differences | hats' level | corner blocks, not right | corner blocks, right | side layers | interior |
|---|---|---|---|---|---|---|---|---|---|
| s-channel, steady | U | 0.93 | 0.07 | 0.006 | 0 | 0.35 | 0 | 0.49 | 0.16 |
| s-channel, steady | $n_c=16$ / 32 | 0.56 / 0.51 | 0.41 / 0.44 | 0.03 / 0.05 | 0 | **0.85 / 0.88** | 0 | 0.11 / 0.09 | 0.04 / 0.03 |
| s-channel, step | $n_c=16$ / 32 | 0.66 / 0.63 | 0.47 / 0.50 | 0.02 / 0.01 | −0.14 / −0.14 | **0.88 / 0.90** | 0 | 0.09 / 0.08 | 0.03 / 0.02 |
| bend-3, step | $n_c=16$ | −0.28 | 0.00 | 0.50 | 0.78 | 0 | 0.39 | 0.53 | 0.07 |
| insert-round, steady | $n_c=16$ | 0.04 | 0.06 | 0.90 | 0 | 0.04 | 0.06 | 0.49 | 0.41 |

- **On the s-channel the coarse level removes the side layers' error and leaves the corners.** U's excess sat half in the side layers and a third at the non-right corners. At $n_c=16$, 85–88% of it lies in the $n/8\times n/8$ blocks at the end pieces' non-right corners (154° and 105°), and the share grows with $n_c$. Each block is $1/64$ of its piece's reference square. No piece of these benchmarks has a material interface inside it, so the interface regions are empty.
- **Its excess falls more slowly with $n_c$** (0.74 → 0.55 per doubling, against 0.52–0.57 → 0.45 on bend-3 and insert-round), as a corner singularity would make it.
- **This is §8's branch 2c:** E4 missed where what remains sits at non-right corners.

**The modes** (validation pieces, medians; U → TL0 at $n_c=16$ → at 32, for $n=32$ / 64 / 128; at $n=32$, $n_c=32$ is the exact control):

| measure | U | $n_c=16$ | $n_c=32$ |
|---|---|---|---|
| low group (hats, $j=1$–3) | 0.34 / 0.36 / 0.36 | 0.088 / 0.100 / 0.102 | 0 / 0.048 / 0.052 |
| $j=4$–8 | 0.36 / 0.38 / 0.37 | 0.19 / 0.22 / 0.23 | 0 / 0.11 / 0.13 |
| $j=9$–15 (the strips) | 0.030 / 0.040 / 0.046 | 0.019 / 0.026 / 0.027 | 0 / 0.020 / 0.021 |
| $e_H$, 64 columns | 0.21 / 0.21 / 0.21 | 0.10 / 0.12 / 0.12 | 0 / 0.061 / 0.068 |
| $e_{H,L}$, 36 learned columns | 0.37 / 0.38 / 0.38 | 0.17 / 0.21 / 0.21 | 0 / 0.10 / 0.12 |
| $e_p$ | 0.69 / 0.69 / 0.70 | 0.11 / 0.12 / 0.12 | 0 / 0.056 / 0.064 |
| $e_1$, the level field (steps) | 0.28 / 0.28 / 0.28 | 0.047 / 0.062 / 0.059 | 0 / 0.029 / 0.029 |

- Per piece, the low group's error at $n_c=16$ is 0.25–0.28 of U's (median; 10th–90th percentiles 0.21–0.37).
- The sines $j=4$–8 fall least, to about 0.6 of U's at $n_c=16$. Their wavelengths, $1/4$–$1/8$ of a side, span two to four coarse cells.
- **B4's four pieces** ($n=128$, $e_{H,L}$): U 0.23, 0.72, 0.42, 0.39; TL0 at 16: 0.14, 0.39, 0.24, 0.20 (median 0.22); at 32: 0.08, 0.22, 0.13, 0.11 (median 0.12). Trained on these very pieces for 3,000 steps from S's weights, B4's network reached a median of 0.162. Untrained, TL0 at $n_c=32$ is lower.
- **W362's piece types** ($n=128$, $e_H$, U → 16 → 32): rectangles of aspect 0.9, 0.7 and 0.5: 0.041 → 0.021 → 0.012, 0.140 → 0.074 → 0.040, 0.280 → 0.149 → 0.083. A hole's rings 0.230 → 0.127 → 0.071; a round inclusion's 0.225 → 0.127 → 0.072; the held-out benchmarks' curved pieces 0.122 → 0.064 → 0.035; the square 0 (the control). W358's best smooth correction (the $16\times16$ cosine class) left 0.073 and 0.077 on the same rings (the forms page's §11.20), so untrained TL0 at $n_c=32$ reaches that class's level.

**The cost** (laptop on AC; medians over 8 pieces of the coarse level's time over the exact solve's):

| $n$ | exact solve | $n_c=4$ | 8 | 16 | 32 | U's forward and strips |
|---|---|---|---|---|---|---|
| 32 | 7.9 ms | 0.17 | 0.16 | 0.21 | — | 4.3 |
| 64 | 60 ms | 0.071 | 0.079 | 0.096 | 0.61 | 1.48 |
| 128 | 295 ms | 0.036 | 0.051 | 0.050 | 0.15 | 0.78 |

- At $n_c=32$ the dense Cholesky of 961 unknowns dominates.
- **[AI Inference]:** U's networks answer zeros here, so a classical TL0 needs only $H^{(0)}$ (two sine transforms), the strips and the coarse level. Its price would then be mostly the strips', which the forms page timed at 0.21–0.44 of an exact solve. None of this is R2's price protocol.

**What it means.**
- **Proposition A and Lemma B hold in practice** on every piece and problem, to $10^{-13}$.
- **Untrained, the coarse level does most of what training never did.** On the validation pieces it cuts the low group's error by a factor of 3.5–4 at $n_c=16$ and the particular field's by 6. On B4's pieces TL0 at $n_c=32$ beats 3,000 steps of overfitting.
- **E4 is met on 4 of 6 at $n_c=16$ and on 5 of 6 at $n_c=32$**; the sixth no 16-mode expert can meet.
- **What remains where E4 is missed sits at non-right corners**, and so does the s-channel's 16-mode floor (Stage 0's §5). Neither the coarse level nor the corner bubble treats a corner's singular exponent.
- **[AI Inference]:** a corner treatment (singular functions at each non-right corner in the bubble, or a chart that splits off the corner's factor) would address the s-channel directly. A network on the residual would be asked for exactly the corner-singular output that W358 found a smooth network cannot follow.

**The decision, the owner's** (§8): 2a, 2b or 2c. Phase 3, the network study, runs under every branch.

**Chosen on 2026-11-09: 2c** ("2c: corner treatment (Recommended)"). A corner treatment is to be drafted, not a network: singular functions at each non-right corner in the bubble, or a chart that splits off the corner's factor, with its own untrained check. Phase 3 runs beside it, to measure what a network adds to TL0.

---

## 14. Phase 3: the network on TL0's residual, built (2026-11-09)

**Built.** Nothing has been trained beyond the tests and one development check (below).

| file | what |
|---|---|
| `atlas/chartop/twolevel.py` (new) | `TwoLevelExpert`, a subclass of `forms.KernelExpert` that adds no parameter: S's form, then $\mathcal C$. Two paths for $\mathcal C$: torch from a batch's element arrays (differentiable), numpy from the host's operators. Also: `zero_heads_twin` (TL0 with stand-ins answering zeros, so no backbone runs); `call` (`forms.call` is pinned); the relative objective; the ledger (ρ with its guard, the coupled attribution, $n_c^\ast$, $\kappa_{\text{kill}}$, the dead-network stop) |
| `atlas/chartop/train.py` (changed; not pinned) | `Arm.coarse`, `init_from`, `head_init` and `drop_easy`, and the objective `"relative"`. Validation logs ρ, a dead network stops the run, and `load_expert` loads two-level checkpoints |
| `scripts/chartop_twolevel_overfit.py` (new) | the probe (below) |
| `scripts/chartop_box_payload.py`, `scripts/chartop_box_run.sh` (changed) | `--with-probe` ships the probe and what it imports; the runner's `probe` stage writes a done file per start |
| `tests/test_chartop_twolevel.py` (21), `tests/test_chartop_twolevel_overfit.py` (4) | below |

**The defaults are today's, bit for bit.** One-off, not a committed test: five default arms trained 4 steps with the committed `train.py` and with the new one. The arms were Stage 0's bubble form, S's form with and without the strips in the training steps, the supervised objective, and B2's prior. Every saved tensor (the weights, the optimiser, the scheduler, the best checkpoint) and every logged number was equal bit for bit; a different seed differed. All 514 chartop tests pass, the pins' among them.

**The anti-collapse methods.** Methods 1–3 are built and on for every two-level arm:
1. the relative objective, $L_i=\varepsilon_{\theta,i}/\varepsilon_{0,i}+\varepsilon_{p,\theta,i}/\varepsilon_{p,0,i}$;
2. zero heads at the start (the U-Net's own initialisation, and `head_init="zero"` after loading weights);
3. pieces with nothing to learn ($\varepsilon_0<10^{-8}\operatorname{tr}\Lambda$) left out and counted.

The ladder (residual-scaled outputs, arm R, the orthogonality penalty) is not built; it comes only if ρ stalls.

**What the tests show** (seeded random weights or zero heads):
- With no coarse size the expert is S bit for bit. With zero heads it is TL0 bit for bit, and Phase 2's numpy TL0 to $10^{-12}$.
- With random weights, CX and the kernel identity hold within $10^{-12}$, while the form's own answer before $\mathcal C$ had CX above $10^{-6}$. The numpy and torch paths agree to $10^{-12}$.
- Through the host, `gates_forms`' own judges find L, K and D met.
- The relative objective reads exactly 1 per term at TL0 and 0 at the exact fields. Its gradient equals the label-free gradient divided by TL0's error, to $10^{-10}$; over all 64 columns the gradient is the same, since the strips and $\mathcal C$ carry none. A square of one material, where TL0 is exact, is left out.
- The ledger's hand-built cases: ρ of 0, 1, −1 and the guard; the attribution and its floor; $n_c^\ast$ exact on $e\propto1/n_c$, with extrapolation flagged; $\kappa_{\text{kill}}=1$ on $V_c$ and 0 off it; the stop at 20%.
- `train.train` on a two-level arm starts at a loss of exactly 2, logs ρ, stops a dead network at the first validation past 20% of the steps, and its checkpoint loads as a two-level expert. Starting from given weights with zeroed heads starts at TL0; keeping their heads does not.

**A development check, not the probe**: two post-hoc validation pieces at $n=32$, $n_c=8$, from zero heads, 200 steps.
- At the registered learning rate, $3\times10^{-4}$, the median ρ rose from 0.005 at step 25 to 0.14 at step 200. At $3\times10^{-3}$ it reached 0.36.
- The shuffled-input control fell to −0.35 and −0.02: the network reads its own pieces.
- In the first 25 steps ρ stays near zero, and can dip below it.

**The probe** (`scripts/chartop_twolevel_overfit.py`). B4's protocol: the four pieces 2,000,100–2,000,103 at $n=128$, AdamW at $3\times10^{-4}$, 3,000 steps. It runs from zero heads and from S's weights, at $n_c=16$, on the relative objective. The ledger is written every 250 steps: ρ per piece and group, ρ of the particular field, $e_{H,L}$, $e_p$, $\kappa_{\text{kill}}$ and the shuffled-input control. The dead-network stop applies, and a repeat check measures how far a rerun of the first steps differs.
- **On the laptop** (on AC), a step takes 1.5 s at 8 threads and 2.3 s at 4: 77–113 minutes per start.
- **On CUDA it cannot be bit for bit**: the U-Net's bilinear upsampling uses atomics in its backward pass, so a rerun differs by rounding. On the CPU it is deterministic (tested).
- **Where it ran:** the owner declined the question about renting a box, so the probe ran on the laptop (below).

**The probe, run (2026-11-09 and 10).** On the laptop's CPU at 8 threads, the two starts in turn: zero heads from 23:53, then S's weights from 02:49 to 04:16.
- **Its timings are not timings.** The laptop slept in Modern Standby from 00:17 to 01:48 during the first start, and during the second the corner check (§17) and the test suite shared the CPU, so its seconds per step (3.47 and 1.72) measure nothing.
- **Its numbers are deterministic.** Each start's repeat check reran its first 20 steps: the weights were equal bit for bit.
- Records: `out/chartop/twolevel/probe-scratch.json` and `probe-S.json`.

**The ledger on the four pieces**, medians over the pieces. $\rho$ is over the 36 learned columns, then by group; $\rho_p$ is the particular field's; $e_{H,L}$ is measured as the forms page measures it. $\kappa_{\text{kill}}$ is the fraction of the network's raw output that $\mathcal C$ discards. The shuffled control feeds each piece the next piece's inputs.

From zero heads (step 0 is TL0):

| step | $\rho$ | p10–p90 | hats | $j=1$–3 | $j=4$–8 | $\rho_p$ | $e_{H,L}$ | $e_p$ | $\kappa_{\text{kill}}$ | shuffled $\rho$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.220 | 0.157 | — | 0 |
| 250 | 0.010 | 0.005–0.013 | 0.306 | 0.024 | 0.008 | 0.887 | 0.218 | 0.055 | 0.975 | −0.00 |
| 500 | 0.028 | 0.013–0.126 | 0.371 | 0.034 | 0.028 | 0.958 | 0.209 | 0.026 | 0.964 | −0.08 |
| 750 | 0.271 | 0.036–0.585 | 0.392 | 0.219 | 0.275 | 0.972 | 0.188 | 0.021 | 0.839 | −0.42 |
| 1,000 | 0.476 | 0.166–0.691 | 0.317 | 0.317 | 0.486 | 0.977 | 0.159 | 0.019 | 0.738 | −0.61 |
| 2,000 | 0.827 | 0.726–0.862 | −0.010 | 0.422 | 0.841 | 0.987 | 0.090 | 0.014 | 0.693 | −1.28 |
| 3,000 | **0.918** | 0.894–0.941 | 0.013 | 0.452 | 0.946 | 0.995 | **0.063** | 0.009 | 0.709 | −1.36 |

From S's weights, with their heads:

| step | $\rho$ | p10–p90 | hats | $j=1$–3 | $j=4$–8 | $\rho_p$ | $e_{H,L}$ | $e_p$ | $\kappa_{\text{kill}}$ | shuffled $\rho$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.123 | 0.055–0.250 | 0.009 | −0.071 | 0.135 | 0.804 | 0.197 | 0.056 | 0.604 | −0.36 |
| 250 | 0.731 | 0.680–0.829 | 0.023 | 0.186 | 0.762 | 0.986 | 0.114 | 0.015 | 0.611 | −1.19 |
| 1,000 | 0.876 | 0.859–0.925 | 0.020 | 0.397 | 0.902 | 0.995 | 0.078 | 0.008 | 0.628 | −1.29 |
| 2,000 | 0.929 | 0.912–0.957 | 0.019 | 0.512 | 0.948 | 0.998 | 0.061 | 0.006 | 0.642 | −1.34 |
| 3,000 | **0.950** | 0.932–0.968 | 0.047 | 0.537 | 0.968 | 0.999 | **0.053** | 0.004 | 0.649 | −1.36 |

**What it shows.**
- **On its own pieces the network adds a great deal beyond the coarse solve.** From S's weights, $\rho$ reaches 0.73 in 250 steps and 0.95 at 3,000, and $e_{H,L}$ falls from TL0's 0.20 to 0.053. That is a third of B4's 0.162 (3,000 steps from S's weights without the coarse level; the forms page's §11.6), and under half of untrained TL0 at $n_c=32$ (0.12, §13). From zero heads: 0.92 and 0.063.
- **It reads its own inputs.** The shuffled control's median falls to −1.36: fed the next piece's inputs, the network makes three of the four pieces much worse than TL0. On 2,000,101, a low-frequency step, it still helps (+0.60 and +0.63), so part of its correction there does not depend on the input. **[AI Inference]:** a shape the coarse level misses on every piece, such as a side layer's.
- **What it learns, by group.** The fast sines $j=4$–8 almost entirely (0.95–0.97); $j=1$–3 about half (0.45–0.54); the particular field nearly exactly ($\rho_p$ 0.995–0.999). The hats gain almost nothing by the end (0.01–0.05). From zero heads, the hats rose to 0.39 by step 750 and fell back as the sines took over. **[AI Inference]:** the relative objective sums the 36 columns against TL0's whole error, in which the hats' share is small, so giving them back costs little. NC reads the 36 together, so the hats are worth reporting separately (§18, prediction 4).
- **$\kappa_{\text{kill}}$.** From zero heads it starts at 0.97: almost all of the network's raw output lay in the coarse space, and $\mathcal C$ discarded it. It falls to 0.71. From S's weights it holds at 0.60–0.65. The network learns to put its output where $\mathcal C$ keeps it, though no term asks for that.
- **The start sets the speed, not the end.** S's weights reach in 250 steps what zero heads reach only at about step 1,700.
  - From zero heads, $\rho$ stayed under 0.05 until step 500 (0.028).
  - The dead-network stop reads at the first validation at or after 20% of the steps: step 750 here, where $\rho$ was 0.27.
  - **[AI Inference]:** had it read at step 600, it might have stopped a network that went on to 0.92. From S's weights the question does not arise.
- **The ladder (methods 4–6) is not needed:** $\rho$ did not stall from either start.
- **Phase 3 is done**, by the handoff's criterion: $\rho>0$ on its own pieces, with the controls failing as they should (shuffled inputs negative; zero heads exactly at TL0).
- **The winner is S's weights:** a higher $\rho$ and a lower $e_{H,L}$ at every step. §18 drafts it as TL's start.
- **One call's cost** at $n=128$, $n_c=16$, per piece, from S's start once the run had ended, the CPU otherwise idle: 0.21 s in all. Of that, the networks' forward took 0.09 s, the strips 0.06 s and the coarse level (the plain grid) 0.06 s. This is a reading, not R2's protocol.

---

## 15. Branch 2c: a corner treatment for the coarse level (accepted as drafted, 2026-11-09)

**Accepted** by the owner as drafted ("Accept as drafted"), with both candidates measured. Accepting starts the build and the untrained check. The predictions below are registered only on the owner's go, after they have been checked against each other and against the new code.

**What it must do.** On the s-channel, 85–90% of TL0's excess over the floor sits in the $n/8\times n/8$ blocks at the end pieces' two non-right corners, 154.0° and 105.2°, and TL0 at $n_c=16$ misses E4 there by 0.15 points (§13). A corner treatment should remove that part of the excess. It must keep every guarantee $\mathcal C$ gives, and it must not need a network.

**The principle: enlarge the coarse space, not the network's job.** Proposition A's proof uses only that $P$ has full column rank. So any enlarged space $V_e=\operatorname{range}P_e\supset V_c$, with $P_e=[\,P\mid S\,]$ and $S$ zero on the loop, gives a correction $\mathcal C_e$ with all of Proposition A:
- Galerkin orthogonality on $V_e$ (CX on the enlarged space);
- idempotence and blindness to shifts in $V_e$;
- traces untouched;
- **never worse than TL0, column by column.** It minimises the same error over a larger set: $\lVert H_k-\mathcal C_e(X)_k\rVert_A\le\lVert H_k-\mathcal C(X)_k\rVert_A$.

Lemma B holds unchanged, because every column still sees the same operator. On a piece without non-right corners, $S$ is empty and $\mathcal C_e=\mathcal C$ bit for bit.

**Two candidates**, both classical. "Non-right" means more than 2° from 90°, from the chart's corner angles, as §12's regions are.

| | A: corner singular functions | B: corner patches |
|---|---|---|
| what $S$ adds | at each non-right corner $c$, $K$ functions $s_{c,k}=\chi(r/R)\,\rho^{k\pi/\omega_c}\sin(k\pi\varphi/\omega_c)$, $k=1,\dots,K$ | at each non-right corner, the fine hats at every node of the corner block that is not a coarse node (a hierarchical complement, so $P_e$ keeps full rank) |
| where they come from | the corner's frozen tensor: $y=A_c^{-1/2}x$ maps the corner's quarter-plane to a wedge of angle $\omega_c$, $\cos\omega_c=-a_{12}/\sqrt{a_{11}a_{22}}$ in the corner's own frame (the forms page's §10); $(\rho,\varphi)$ are $y$'s polar coordinates from the first side, and $\chi$ a smooth cut-off, one inside $R/2$ and zero beyond $R$ | the block's own fine grid: an exact local solve of the piece's true operator, inside one Galerkin projection with the coarse grid |
| extra unknowns | $K$ per corner (2–3) | $(n/8)^2$ less four coarse nodes per corner: 60 at $n=64$, 252 at 128 (or a quarter of that with $n/16$ blocks) |
| what it can capture | the corner's leading singular behaviour, as the frozen tensor predicts it | whatever is local to the block: the singular exponent, the chart's variation, a material |

**Which is likely to matter.** The s-channel's corner exponents are mild: $\pi/\omega$ is 1.17 at 154° and 1.71 at 105°, so the exact modes are nearly $H^2$ there. But the Winslow chart varies fast near the 154° corner: Proposition 2 puts $\lvert\mu\rvert\ge0.625$ at the corner itself, against a supremum of 0.543 over the grid's samples, which stop half a cell from it (Stage 0's §4.4). **[AI Inference]:** the excess may come from the pulled-back coefficients' variation near the corner more than from the singular exponent. B treats both; A treats only the exponent.

**Deferred, as before:** singular functions in the bubble (a network's output form, so not this branch) and a chart that splits off the corner's factor (a change to the geometry, which the host reads).

**The untrained check**, drafted (classical solves and U's zero heads only):
- **Arms:** TL0 at $n_c=16$; A with $K=1$, 2, 3; B with $n/8$ and $n/16$ blocks; and, for the curve, A and B at $n_c=8$ and 32.
- **Where:** the six Stage 0 benchmarks in both frames (E4, the excess over the floor, the split by place, A1's split); Phase 2's validation pieces and W362's types ($e_H$ by group); the cost against `fe.Exact` on the laptop on AC.
- **Controls:** on pieces with only right corners (bend-3, the squares), each arm is TL0 bit for bit. At $n_c=n$, each is exact. Each arm is never worse than TL0, column by column.

**Predictions to register before the check runs** (drafted, not registered; each to be checked against the others and against the code's docstrings first):
1. The controls and Proposition A hold for both enlarged spaces, to rounding: CX at most $10^{-12}$ on $V_e$, no column worse than TL0's, TL0 bit for bit where no corner is non-right, $n_c=n$ exact.
2. **[AI Inference]** B with $n/8$ blocks brings the s-channel steady problem's excess below 1.77% at $n_c=16$: E4 met on 5 of 6 at $n_c=16$, all but the s-channel step lowered by 300 K, which no 16-mode expert can meet.
3. **[AI Inference]** A gains less than B on the s-channel: at $K=3$ its excess there stays above B's.
4. **[AI Inference]** With B, the non-right corner blocks hold less than half of the s-channel's remaining excess.
5. **[AI Inference]** B's coarse level at $n=128$, $n_c=16$, costs under 0.15 of an exact solve per piece with two non-right corners (TL0's was 0.050).

**What it would not do.** Lower the 16-mode floor, which also comes from these corners (Stage 0's §5): that is the ports' lever, not the expert's. Nor move the s-channel step lowered by 300 K under E4's bar: its floor is above it.

**The owner's decision:** accept the draft (and which candidates to measure), amend it, or stop.

---

## 16. Branch 2c: the build, the script and the predictions' readings (registered 2026-11-10)

**Registered** by the owner on 2026-11-10, in chat ("Register as drafted (Recommended)"): §15's five predictions as drafted, with the readings below, before any part ran on the pieces they read. The script carries the registration (`REGISTERED`). Its SHA-256 at registration is `cc5da98c24a86eef594ab878162df55c89b3635a1a73e5ccb9335cc31c6f2d82`, and `coarse.py`'s is `a17c6a7c70c25649cf847a82d2b98a17a390a0515068c69b1fb503a38cfce8e8`; every record stores both.

**Built.** Nothing had run on the pieces the check measures when it was registered.

| file | what |
|---|---|
| `atlas/chartop/coarse.py` (changed; not pinned) | `CornerTreatment` builds $S$ at a piece's non-right corners (`non_right_corners`): candidate A with `corner_singular_columns`, B with `corner_patch_columns`. `corner_wedge` gives each corner's frozen wedge. `CoarseSpace` and `correct_fields` take the added columns (`extra=`), and the host's adapter `CoarseCorrected` takes a treatment (`corners=`) |
| `scripts/chartop_twolevel_corners.py` (new) | the untrained check: parts `e4`, `modes` and `cost`, `--judge` and `--smoke`. Every registered part is refused until the predictions are registered |
| `tests/test_chartop_coarse_corners.py` (67), `tests/test_chartop_twolevel_corners.py` (36) | below |

**The defaults are today's, bit for bit.** One-off, not a committed test: the committed `coarse.py` (at `9a30938`) was loaded beside the new one and both were run on three pieces, at $n=16$ and 32 with $n_c$ from 2 to $n$. The host's adapter (with and without a coarse size), $A_c$ and the correction agreed bit for bit in all 150 comparisons. The 124 tests of `coarse.py`, the two-level expert, Phase 2's script and the probe pass. The enlarged space is numpy only: the torch path that training uses keeps the coarse grid alone, and extending it waits for the check.

**What the draft left open, settled here:**
- **A.** The frozen tensor is the corner element's, scaled to determinant one, which changes only the columns' scale.
  - $\varphi$ is measured from the side along which $\eta$ is fixed.
  - $\chi$ is the cubic smoothstep $s^2(3-2s)$, $s=2(1-r/R)$, between $R/2$ and $R=1/2$.
  - Each column is scaled to a largest absolute value of one.
  - On an affine chart the frozen wedge is the corner's angle to $10^{-9}$. On the drawn pieces it differs from the chart's corner angle by up to 5° (126.6° against 121.6°). The Winslow chart varies at the corner, which is §15's reason to expect B to gain more.
- **B.** The block is the interior nodes at offsets 1 to $w$ from the corner along both sides, $w=n/8$ or $n/16$: the page's $(n/8)^2$ nodes, less the coarse nodes among them. The hats on a block's inner edge reach one cell beyond it.
- **At $n_c=n$ nothing is added.** The coarse space is already the whole interior, and an added column would cost full rank. Columns passed there by hand are refused.

**What the tests show** (seeded pieces at $n=16$ from 9,000,100: three with every corner non-right, and one conformal piece with every corner right):
- **The corners.** Non-right means more than 2° from 90°: 88.1° and 91.9° are right, 87.9° and 92.1° are not. On a parallelogram, the frozen wedge is the corner's angle at all four corners. The first singular function is positive at every node inside the cut-off and zero beyond it, at all four corners, so $\varphi$ stays in the wedge whichever way the corner's frame turns.
- **The patches.** 60 and 252 columns per corner at $n=64$ and 128 with $n/8$ blocks, and 15 and 63 with $n/16$, as §15's table says. Each column is one fine hat in its corner's block; no coarse node is among them, and no node appears twice.
- **Proposition A on $V_e$**, for A with $K=1$ and 3 and for both B:
  - CX within $10^{-12}$ for the modes and the particular field;
  - never worse than TL0, column by column, and equal to a brute-force least-squares fit over $V_e$;
  - idempotence and blindness to shifts in $V_e$;
  - full column rank, traces untouched, and Lemma B.
- **The controls.** On the conformal piece every output is TL0's, bit for bit. At $n_c=n$ nothing is added and the answer is exact.
- **Through the host's adapter**, the answers are coarse-exact on each piece's own $V_e$, kernel-exact and never worse than TL0, and they are TL0's bits on the conformal piece.

**The smoke run** (`--smoke`, on seeds no registration draws): validation-style seeds 9,000,100–101 at $n=16$ and 32; the `benchmarks_squares` layout of seed 9,000,008 at $n=32$, which has 8 pieces of 25 with a non-right corner. Every part ran.
- The treatments' CX was at most $9.4\times10^{-15}$ and $e_{K,\infty}$ at most $2.4\times10^{-14}$; no column was worse than TL0's.
- Pieces with no non-right corner gave TL0's bits, and nothing was added at $n_c=n$.
- $e^2=\text{floor}^2+\text{excess}^2$ held to $1.7\times10^{-12}$.

**The predictions, with the readings `judge` applies** (§15's text unchanged):

| # | prediction | reading |
|---|---|---|
| 1 | Proposition A and the controls on both enlarged spaces | for every treatment (A with $K=1$, 2, 3; B with $n/8$ and $n/16$ blocks) at every $n_c$ measured, on every piece of `modes` and every benchmark piece in both frames: where $n_c<n$, CX on the arm's own space $\le10^{-12}$ for the modes, and for the particular field wherever the piece has a load; no column whose energy-norm error exceeds TL0's at the same $n_c$ by more than $10^{-9}$ of TL0's plus $10^{-12}\sqrt{\Lambda_{kk}}$; $e_K$ and $e_{K,\infty}\le10^{-12}$ (the level-field identity on every piece of `modes`, the uniform state on every benchmark piece in the registered frame). On every piece with no non-right corner, each treatment's answers are TL0's bit for bit, and on bend-3 each treatment's energy and maximum errors are TL0's. At $n_c=n$ no column is added, the answers are TL0's bit for bit, and $e_H\le10^{-10}$ on every $n=32$ validation piece |
| 2 | **[AI Inference]** B with $n/8$ blocks brings the s-channel steady excess below 1.77% at $n_c=16$, and E4 is met on 5 of 6 | for B with $n/8$ blocks at $n_c=16$: the s-channel steady problem's excess over the floor is below 0.0177 in both frames, and `judge_e4_both_frames` is met on each of the five problems other than the s-channel step |
| 3 | **[AI Inference]** A with $K=3$ gains less than B on the s-channel | at $n_c=16$, A with $K=3$ has a larger excess than B with $n/8$ blocks, on both s-channel problems, in both frames |
| 4 | **[AI Inference]** with B, the non-right corner blocks hold under half of the s-channel's remaining excess | for B with $n/8$ blocks at $n_c=16$, the split by place (§12's regions) puts less than 0.5 of the network term in the non-right corner blocks, on both s-channel problems, in both frames. A frame whose network term is under the split floor ($10^{-9}$ of the reference energy) has no share and counts as held |
| 5 | **[AI Inference]** B costs under 0.15 of an exact solve per piece with two non-right corners | the median, over the cost part's pieces with exactly two non-right corners, of B's coarse level ($n/8$ blocks, $n=128$, $n_c=16$, its columns' building included) over the exact solve is below 0.15; judged only if every power sample read AC |

**The pieces each reading touches.**
- **`e4`:** the six benchmark problems at $n=64$, in both frames. The pieces with a non-right corner are the s-channel's two end pieces (105.2° and 154.0°) and insert-round's 24 ring pieces (45° to 135°). Bend-3 has none, so it is the bit-for-bit control.
- **`modes`:** Phase 2's 456 pieces. Of the 64 validation pieces at $n=32$:
  - 42 have four non-right corners;
  - 10 have three;
  - 2 have two;
  - 10, the conformal maps, have none.
- **`cost`:** at $n=128$, the first 16 validation pieces and the s-channel's two end pieces. The pieces with exactly two non-right corners, which prediction 5 reads, are 2,000,104, 2,000,106 and the two end pieces (109.7° and 154.2° at this resolution).

**Checked against each other and against the code.**
- **1 is Proposition A on $V_e$.** `coarse.py`'s docstring states it, and the tests show it to rounding. It reads every treatment at every $n_c$, including on pieces with no non-right corner.
- **2 rests on the energy bar alone.** At $n_c=16$, TL0 missed E4 on the s-channel steady problem only by energy: 2.146% against 2%. Its maximum error was 0.19% of the range, against a bar of 1.15%. Since $\sqrt{0.93^2+1.77^2}=2.00$, the excess clause and the energy bar are one condition.
  - The other four problems met E4 at TL0-16, and B leaves bend-3 untouched (1).
  - B does change insert-round's ring pieces. Their network term can only fall (Proposition A), but their maximum error is not bounded that way. It stood at 0.020% and 0.051% of the range, against bars of 1.00% and 1.05%.
- **2 and 4 agree, and 4 is the stronger.** At TL0-16, 85–88% of the s-channel's network term sat in the non-right corner blocks (§13). **[AI Inference]:** if the rest stays as it is, the share falls under one half only if B removes more than about 82% of the corner part. The excess would then fall well below 1.77%. 2 can hold while 4 fails.
- **3 follows from neither.** A's functions reach beyond the block ($R=1/2$), while B's patch stops at its edge. If the corner's error extends past the block, A could gain more than B.
- **5 against §13's costs.** At $n=128$, TL0 at $n_c=32$, with 961 coarse unknowns, cost 0.15 of an exact solve. B with $n/8$ blocks at two corners has $225+2\times252=729$, so 5 says it costs less than that. The dense factor dominates both.
- §15's text, `coarse.py`'s docstring and `forms.py`'s contradict none of them.

**Running it**, on the laptop on AC once the probe has ended: `e4` by problem and `modes` by resolution in parallel processes, then `cost` alone. Records go to `out/chartop/twolevel/corners/`; Phase 2's are left as they are.

**The owner's decision** (register the five predictions with these readings, amend them, or stop): registered as drafted, above.

---

## 17. Branch 2c: the corner treatment measured untrained (2026-11-10)

**What ran.** `scripts/chartop_twolevel_corners.py` at the registration's commit (`b09f040`), on the laptop on AC, one process at a time because memory was short:
- `e4` first: the six problems in both frames, 13 minutes;
- then `modes`: n = 32 and 64, then 128, 15 minutes;
- then `cost`, alone, once Phase 3's probe had ended.

The probe shared the CPU with the first two parts. It reads none of the check's files, and the check's numbers are deterministic. Records: `out/chartop/twolevel/corners/` (`e4-*.json`, `modes-*.json`, `cost-laptop.json`, `verdicts.json` and `.txt`). No weights were read and no test piece was drawn.

**The predictions, as registered:**

| # | verdict | measured |
|---|---|---|
| 1 | **held** | over 456 pieces and six problems, both enlarged spaces at every $n_c$: CX on $V_e$ at most $7.5\times10^{-15}$ (the particular field's $8.5\times10^{-14}$); $e_K$ and $e_{K,\infty}$ at most $2.7\times10^{-13}$; no column worse than TL0's. Every piece with no non-right corner gave TL0's bits: 1,628 benchmark answers and 1,440 arm rows of `modes`. Bend-3's errors were TL0's. At $n_c=n$ nothing was added and the answers were TL0's bits; on the $n=32$ validation pieces $e_H$ was at most $9.5\times10^{-16}$ |
| 2 | **held** | B with $n/8$ blocks at $n_c=16$: the s-channel steady problem's excess 0.785% (TL0's 1.936%), its energy error 1.21%; E4 met on all five meetable problems |
| 3 | **held** | A with $K=3$ against B with $n/8$ blocks, at $n_c=16$: s-channel steady 1.815% against 0.785%; the step 0.241% against 0.093% as registered, and 3.79% against 1.46% lowered by 300 K |
| 4 | **held** | with B, the non-right corner blocks hold 0.023 (steady) and 0.034 (step) of the network term, where TL0 left 0.846 and 0.878 there |
| 5 | **held** | on the four cost pieces with exactly two non-right corners (validation 2,000,104 and 2,000,106 and the s-channel's two end pieces, at $n=128$): B's coarse level, with $n/8$ blocks at $n_c=16$ and its columns' building included, cost a median 0.114 of an exact solve (TL0's 0.045 on the same pieces); both power samples read AC, and no other process ran |

**E4 by arm.**
- TL0 meets 4 of 6 at $n_c=8$ and 16, and 5 at 32.
- A meets what TL0 meets.
- B with $n/16$ blocks meets 5 from $n_c=16$.
- B with $n/8$ blocks meets 5 at every $n_c$ from 8.

The sixth, the s-channel step lowered by 300 K, no 16-mode expert can meet (§13).

**The excess over the floor, in %** (TL0 at $n_c=4$ from Phase 2's records):

| problem | TL0, $n_c=4$ / 8 / 16 / 32 | A ($K=3$), 16 | B ($n/16$), 16 | B ($n/8$), 8 / 16 / 32 |
|---|---|---|---|---|
| s-channel, steady | 3.95 / 2.94 / 1.94 / 1.06 | 1.82 | 1.26 | 1.58 / **0.785** / 0.367 |
| s-channel, step, lowered | 8.23 / 6.21 / 4.06 / 2.20 | 3.79 | 2.62 | 2.97 / **1.46** / 0.69 |
| insert-round, steady | 2.18 / 1.14 / 0.569 / 0.257 | 0.557 | 0.567 | 1.12 / 0.560 / 0.252 |

- **The equivalent coarse size** (`twolevel.equivalent_nc`: TL0's excess, interpolated in log–log). On the s-channel, B with $n/8$ blocks at $n_c=16$ is worth TL0 at $n_c^\ast\approx45$ (steady) and 51 (step), extrapolated past 32. That is 465 unknowns per end piece where TL0 needs about 2,000. B with $n/16$ blocks is worth about 26 and A about 17. On insert-round, whose non-right corner blocks hold 4% of the excess, all three are worth about 16.
- **Where the s-channel's excess sits now.** With B ($n/8$), the non-right corner blocks hold 2–3% of the network term, the side layers 70–72% and the interior 24–28%. Under TL0 the corner blocks held 85–88%. A1's split moves toward the low sines: $j=1$–3 holds 0.69 of the steady problem's term.
- **A barely moves the corner share** (0.79 and 0.81 against TL0's 0.85 and 0.88), and its gain does not grow with $K$: $K=1$, 2, 3 give 1.819%, 1.818% and 1.815% on the steady problem.

**The modes** (validation pieces, medians at $n_c=16$; $n=32$ / 64 / 128):

| measure | TL0 | A ($K=3$) | B ($n/16$) | B ($n/8$) |
|---|---|---|---|---|
| $e_H$, 64 columns | 0.103 / 0.116 / 0.120 | 0.102 / 0.115 / 0.119 | 0.099 / 0.113 / 0.117 | 0.090 / 0.105 / 0.107 |
| $e_{H,L}$, 36 learned | 0.173 / 0.205 / 0.213 | 0.170 / 0.202 / 0.210 | 0.166 / 0.199 / 0.205 | 0.154 / 0.182 / 0.187 |
| low group | 0.088 / 0.100 / 0.103 | 0.086 / 0.098 / 0.100 | 0.086 / 0.098 / 0.101 | 0.081 / 0.093 / 0.096 |

- 54 of the 64 pieces have a non-right corner. On those pieces, B ($n/8$) brings each piece's $e_H$ to a median of 0.875–0.888 of TL0's (10th–90th percentiles 0.84–0.94).
- On W362's ring pieces (a hole's and a round inclusion's, one non-right corner each), B ($n/8$) lowers $e_H$ from 0.109–0.127 to 0.103–0.121.
- Pieces with only right corners (the rectangles, most of the held-out benchmarks' curved pieces) are TL0's to the bit, as prediction 1 requires.

**The cost** (laptop on AC, alone; $n=128$, $n_c=16$; the coarse level's time over the exact solve's, medians by the piece's non-right corners, the columns' building included):

| non-right corners | pieces | TL0 | A ($K=3$) | B ($n/16$) | B ($n/8$) |
|---|---|---|---|---|---|
| 2 | 4 | 0.045 | 0.093 | 0.068 | **0.114** |
| 3 | 3 | 0.050 | 0.120 | 0.078 | 0.167 |
| 4 | 10 | 0.049 | 0.158 | 0.076 | 0.250 |

- B's cost grows with its unknowns: $n/8$ blocks add 252 per corner at $n=128$. With four non-right corners, 1,233 coarse unknowns cost a quarter of an exact solve. B with $n/16$ blocks adds 63 per corner at this resolution and stays under 0.08.
- A's twelve columns cost more than their count suggests: each spreads over a fifth of the piece, which fills the restricted operator.
- Most generated pieces have three or four non-right corners (§16), so on generic pieces B with $n/8$ blocks costs 0.17–0.25 of an exact solve, against TL0's 0.05.

**What it means.**
- **B with $n/8$ blocks is the corner treatment.**
  - Untrained, at $n_c=16$, it meets E4 on every problem a 16-mode expert can meet.
  - Its extra unknowns are local: one block per non-right corner.
  - It treats what TL0 left at the s-channel's corners almost entirely.
- **A, the frozen-tensor singular functions, lowers the s-channel's excess by 6–7% and insert-round's by 2%.** **[AI Inference]:** the corner's error comes from the chart's variation near the corner more than from the singular exponent, as §15 guessed. B treats both.
- **On generic pieces the gain is smaller:** $e_H$ falls by 10–12% (each piece's ratio a median of 0.875–0.888). What remains there is the side layers' and the interior's, which the network is for.
- **The owner adopted B with $n/8$ blocks for Phase 4** on 2026-11-10, after the `e4` part ("Adopt B, n/8 (Recommended)"), together with the NC gate's bar as proposed and the comparison budget of 37,500 steps. The training path is built and committed (`279b7ff`), off by default and bit for bit:
  - `atlas/chartop/coarse_patches.py`: $\mathcal C_e$ in torch, assembled by probing;
  - `TwoLevelExpert(corners=8)` and `twolevel.host_expert`;
  - `train.Arm.corners`;
  - `chartop_data.py --corners`, each example's flags from its chart.

---

## 18. Phase 4: the registration (accepted as drafted and hash-pinned, 2026-11-10)

**Hash-pinned** with the owner's approval on 2026-11-10 ("Pin it (Recommended)"), after the price protocol's dry run passed (below), before any network of TL was trained and before any expert read the test split. `scripts/chartop_register_twolevel.py --pin` wrote `out/chartop/registered-twolevel.txt`; `check_pin` verifies all 31 files, and a test fails if any of them changes. It records:
- the scope by sha256: the gate module, `coarse.py` (unchanged since the corner check's registration), `coarse_patches.py`, `twolevel.py`, `train.py`, the training and data scripts, the price protocol with its payload and runner, and the training box's payload and runner;
- the files earlier registrations pinned, each agreeing with them;
- beside it: the classical record, the passing dry run and the arms' builders;
- by value: S's weights' sha256, the test split's content sha256 and the gates.

**Accepted** by the owner as drafted, in chat on 2026-11-10 ("Accept as drafted (Recommended)"): the arms, the start from S's weights at the probe's rate, the held-out problems, the gates and the five predictions below. The gate module carries the acceptance (`gates_twolevel.REGISTERED`) and the predictions word for word (`PREDICTIONS`, tested). The pin follows the price protocol's dry run, with the owner's approval.

**Settled by the owner already** (in chat):
- the formulation;
- $n_c\in\{8,16,32\}$, 16 the default;
- E4 the only accuracy gate, CX added, E1 and SE3 reported;
- branch 2c's B with $n/8$ blocks in TL and TL0;
- the NC bar as proposed;
- the comparison budget of 37,500 steps.

**The module:** `atlas/chartop/gates_twolevel.py`. `REGISTERED` records the acceptance and the pin (it was `None` while this section was a draft). Every bar it shares with the earlier registrations is read from `gates_forms` and `gates_prior`.

**The arms:**

| arm | what | training |
|---|---|---|
| TL | S's form (split, corner bubble, strips for $j=9$–15), then $\mathcal C_e$: $n_c=16$ with B's corner patches ($n/8$), on the relative objective with methods 1–3 | from S's weights with their heads (`S/best.pt`, sha256 `e17f881b…a35d`), Phase 3's probe's winner (§14); AdamW at $3\times10^{-4}$ (the probe's rate), S′'s schedule and batches, the strips skipped in training; 37,500 steps; training seeds 0 and 1 |
| TL0 | TL with zero heads, untrained: its own floor (16, with corners) and the curve at $n_c=4$, 8, 16 and 32 | none |
| S′ | reused, not trained again (`gates_prior.REUSED`) | none |

TL's start carries S's own 150,000 steps, so its total training is not S′'s. NC, the gate on what the network adds, compares TL with TL0 and does not depend on that history.

**What is judged, once.**
- TL's seed-0 run is chosen on the validation split.
- It is evaluated once on a **fresh test split**: seeds 6,000,000–6,002,399, 2,400 pieces as before. The 3,000,000 and 5,000,000 ranges have been evaluated. The split's content sha256 is recorded at registration, before any expert reads it.
- It is also evaluated on the **held-out benchmark problems** (`benchmarks`' variant `"held-out"`). Their classical 16-mode floors are all under 2% (1.24% at most, the s-channel step lowered by 300 K), so E4 can be met on all six.
- The six problems Phase 2 and the corner check read are reported beside, as seen.
- K, D and L are judged on every run.

**The gates:**

| gate | bar |
|---|---|
| E4 | the only accuracy gate: every held-out problem, both frames, the bars of `gates_forms` |
| CX | every test and held-out benchmark piece: CX on the arm's own space $\le10^{-10}$, the particular field likewise wherever there is a load |
| NC (new) | TL's seed-0 run on the test split against TL0 (16, corners): median $\rho\ge0.3$ and 10th percentile $\ge0$ over the pieces with something to learn. On no held-out problem, in either frame, is TL's energy error above TL0's by more than $10^{-9}$ of it |
| K, D, L, E3, SE1, SE2, DUAL, KT | as `gates_prior` worded them, on the held-out problems |
| PR | TL's price under R2's protocol, on a rented whole machine, reported against the classical solve, H2r (0.306) and TL0; "faster" only where measured so |
| reported | E1 and SE3; S11; the network ledger (ρ by group, $\rho_p$, the coupled attribution, $n_c^\ast$, $\kappa_{\text{kill}}$, the usage controls) |

**Predictions** (registered with the acceptance, before any network of TL was trained; checked against each other and against the code below):
1. CX, K, D and L hold on every run, by structure: Proposition A and Lemma B on $V_e$, and the split form.
2. **[AI Inference]** NC is met: TL's median test $\rho$ is at least 0.3.
3. **[AI Inference]** TL meets E4 on all six held-out problems.
4. **[AI Inference]** By group, $\rho$ is lowest on the hats: on the probe's pieces the hats gave back what they had gained.
5. **[AI Inference]** PR: TL is not faster than the classical solve. U's forward with its strips already cost 1.48 and 0.78 of an exact solve at $n=64$ and 128 (§13).

**Still to build when the owner accepted (the handoff's Phase 4; all of it built since, below):**
- `scripts/chartop_register_twolevel.py`, modelled on `chartop_register_squares.py`. It hash-pins the gate module, the coarse level and its torch path, the two-level expert, `train.py`'s options and the price protocol, and records the test split's content sha256 and the held-out floors. A test adds one byte to a pinned copy and requires `check_pin` to fail.
- **The price protocol:** R2's per-problem count, AC and sleep checks, four BLAS threads, best of 5, A/A controls. It is dry-run on control problems on a rented whole machine before pinning.
- **Then Phase 5, rented:**
  - data on a CPU box: Stage 0's training seeds again with `--corners`, the validation split labelled, the test split;
  - training on a GPU box at no more than \$0.70 an hour, after a step is priced;
  - the one evaluation: accuracy on the laptop on AC, the price on the rented whole machine.

  **[AI Inference]:** in all, a few dollars, inside the handoff's \$15 ceiling. Each rental is asked separately.

**The predictions, checked against each other and against the code:**
- **1 is structural.** Proposition A and Lemma B hold on $V_e$ for any full-rank $P_e$ (§15). The tests show them on both paths, and through the host, with random weights (§17's build). L holds because the corner columns read the piece's chart and tensors, never its load.
- **2 and 4 agree.** NC reads the 36 learned columns together, and on the probe's pieces the hats were a small share of TL0's error there. 4 can hold while 2 holds or fails.
- **3 is the riskiest.**
  - The held-out s-channel step lowered by 300 K has a floor of 1.24%, so E4 asks for an excess below 1.57%.
  - On the seen step, TL0 with B left 1.46%, untrained. The held-out materials differ, so that number does not carry over.
  - 3 leans on the network: under 2, TL removes at least 30% of TL0's error energy on the learned columns of the median test piece.
  - The maximum-error bar is not bounded by Proposition A.
- **3 and NC's benchmark clause agree.** The clause fails if TL is worse than TL0 on any held-out problem, and 3 needs TL at least as good as E4 there.
- **5 rests on §13's costs:** U's forward and strips cost 1.48 and 0.78 of an exact solve at $n=64$ and 128, and B's coarse level adds 0.11–0.25 (§17). PR is reported, not gated, so 5 cannot fail a gate.
- `gates_twolevel.py`'s docstring and `twolevel.py`'s contradict none of them.

**The owner's decision** (accept, amend or stop): accepted as drafted, above.

**Built after the acceptance** (2026-11-10), none of it run on a rented box yet:
- **The price protocol**, `scripts/chartop_price_twolevel.py`: R2's protocol with R2's own helpers (`gates_squares`), the arms in turn:
  - the classical expert;
  - TL through the host (`twolevel.host_expert`);
  - TL0, U's form without its networks behind the same coarse level;
  - H2r;
  - the classical expert again, for the A/A check on every problem;
  - the coupled solve.

  PR's problems are the held-out six and R2's sixteen, so TL and H2r are priced on the same problems. The dry run is on §11.17's twelve problems, with TL's networks at S's weights, checked by their sha256. Its payload is built from one commit (`chartop_price_twolevel_payload.py`, with `IDENTITY`), and its runner (`chartop_price_twolevel_box_run.sh`) installs the laptop's versions and checks the weights. A smoke dry run passed on the laptop in subprocesses: BLAS threads fixed and checked, AC and sleep sampled, per-problem counting, the A/A arm.
- **The registration script**, `scripts/chartop_register_twolevel.py`. Its classical record (`out/chartop/registration-twolevel.json`), measured with the classical solver and the generator only:
  - the held-out problems' floors (1.0008% and 1.2359% at most, as the earlier registration measured them), with no refusals;
  - the fresh test split's content sha256, `735dbc76dde1fd81…`, over 2,400 pieces (731 at $n=32$, 1,220 at 64, 449 at 128), on the laptop where the evaluation's accuracy runs.

  `--pin` refuses until the dry run has passed on the pricing CPU with the code it pins. A test adds one byte to a pinned copy and `check_pin` fails.
- **The training side.** `chartop_train.py` checks a `gates_twolevel` plan before anything trains: arm TL's configuration, seeds 0 and 1 once each, and S's weights by sha256. The box runner's data stage writes the corner flags with `DATA_CORNERS=1`.

**The price protocol's dry run passed** (2026-11-10). The owner approved the box, and when the first offer was taken, its fallback. It ran on a rented whole machine with the pricing CPU: an AMD Ryzen 9 5900X with 23 CPUs in its container's quota, \$0.172 an hour.
- **Six runs** on §11.17's twelve problems, at commit `345055c`. Every run held AC and sleep, with no other load. Every measurement counted, and the A/A check held within 0.004 of 1.
- **Geometric means of the medians**, against the classical solve: TL (with S's weights) 1.023, TL0 0.804, H2r 0.827.
- **A first launch** on `6eb5e8d` was stopped after its smoke run. The price runs load S's weights through `train.py`, which the light labels then changed.
- **The laptop slept** in Modern Standby from 15:24 to 19:05, on battery, while the dry run ran.
  - The box finished in 22 minutes and stopped itself at its deadline.
  - Its records were copied off the stopped instance through the host's rsync service, and rsync's checksums showed no file differing.
  - The instance was then destroyed.
- **Cost \$0.32.** Record: `out/chartop/dryrun-twolevel-20261110-201853.json`, with the box's logs beside it.
- With it, `chartop_register_twolevel.py` finds nothing that refuses the pin.

**The training side, built since** (not run on any box): TL's step priced on its own box (`chartop_price.py --twolevel`), its budget (`chartop_budget.py`), the runner's stage, and the plan `out/chartop/plan-twolevel.json`. The plan is generated from the gate module, and `chartop_train.py` checks it before anything trains.

**Then, with the owner's go, Phase 5's box** (§19): data with the corner flags and light labels, TL's step priced and budgeted on that box, then both runs.

---

## 19. Phase 5: TL trained, and the one evaluation (2026-11-10 to 11)

### 19.1 The training box

**Approved by the owner** in chat on 2026-11-10 ("3090, Turkiye, \$0.20/h (Recommended)"): one RTX 3090 in a container with a quota of 30 CPUs, at \$0.217 an hour with its storage (instance 60143950). The pinned runner (`chartop_box_run.sh`) ran its stages in order:
- **TL's step, priced on that box** (`chartop_price.py --twolevel`, untrained weights, for cost only). A step took 0.083 s at $n=32$ (batch 32), 0.210 s at 64 (batch 32) and 0.252 s at 128 (batch 8): a mean of 0.207 s over the training mix.
- **The budget** (`chartop_budget.py`):
  - each run 2.27 h with its 15 validations, the data 3.11 h, 0.4 h fixed;
  - this box: 8.05 h, \$1.75;
  - the registration's rule (both runs on one box like this one, plus the dry run's \$0.32): \$2.07 against the \$10 ceiling. Within, so training went on.
- **The data**, with `DATA_CORNERS=1 DATA_LIGHT=1`. The 50,000 training examples, with their corner flags and light labels, took 1,352 s on 29 workers; the generator rejected 16,523 charts. The 600 validation examples took 23 s.
- **Both runs**, one after the other.

| run | steps | wall time | validation chose | median $e_H$ | median $\rho$ | 10th percentile of $\rho$ |
|---|---|---|---|---|---|---|
| TL-s0 | 37,500 | 8,149 s | step 37,500 | 0.05855 | 0.661 | 0.314 |
| TL-s1 | 37,500 | 8,077 s | step 35,000 | 0.05854 | 0.659 | 0.313 |

$\rho$ here is the training's validation measure against TL0, over the 600 validation pieces; none had nothing to learn. Neither run stopped as a dead network. The validation's largest $e_K$ was about $10^{-15}$ at every check: round-off, as the split form requires.

**TL-s0's validation, by step:**

| step | 2,500 | 5,000 | 10,000 | 15,000 | 20,000 | 25,000 | 30,000 | 37,500 |
|---|---|---|---|---|---|---|---|---|
| median $\rho$ | 0.319 | 0.383 | 0.494 | 0.585 | 0.594 | 0.639 | 0.644 | 0.661 |
| 10th percentile | 0.123 | 0.127 | 0.201 | 0.233 | 0.287 | 0.293 | 0.314 | 0.314 |
| median $e_H$ | 0.0759 | 0.0727 | 0.0679 | 0.0635 | 0.0619 | 0.0601 | 0.0593 | 0.0586 |

- **The step price held.** The runs took 2.26 h and 2.24 h against 2.27 h priced. The data took 23 minutes against 3.11 h budgeted.
- **Collected, then destroyed.** Every file came off the box by tar over ssh, and its sha256 was compared with the box's own `sha256sum`. Then the instance was destroyed, and the listing showed none. The box ran about five hours: credit went from \$8.97 to \$7.89, **\$1.08**.
- **The laptop polled the box every five minutes** under a power request (system and execution required). It was on battery from 21:00 to 23:29. It still entered Modern Standby from 21:34:48 to 21:36:58, but no poll was missed: the one at 21:35:21 returned. The box's own deadline was the guard either way.
- **Records:** `out/chartop/runs/twolevel/` (commit `4a2a359`):
  - each run's `arm.json`, `log.jsonl` and `summary.json`;
  - `weights-sha256.txt`, each `best.pt`'s sha256 at download, which stage A checks (the weights are git-ignored);
  - `box/`: the box's logs, its budget, its plan (the repository's, with LF endings), its data manifests and its price record of TL's step.

### 19.2 The one evaluation, stage A: the accuracy (2026-11-11)

**What ran.** `scripts/chartop_evaluate_twolevel.py` at commit `4a2a359`, on the laptop on AC, with the owner's go ("Yes, run stage A"). It ran from 00:42 to 01:10 under its own power request, and the System log shows no power event in that window. Before any expert read anything:
- the pin (31 files) and the code were checked: `atlas/` and `scripts/` at the commit;
- each run's weights were checked against their sha256 at download, with TL's configuration, training options and seed; S's and S′'s weights were checked as the registration names them;
- the registry accepted TL's weights as never evaluated;
- **the fresh test split was drawn** (2,400 pieces: 731 at $n=32$, 1,220 at 64, 449 at 128, in 109 s). Its content sha256 matched the registered `735dbc76dde1fd81…`.

Records (commit `7dab3f6`): `out/chartop/evaluation-twolevel-20261111-004223-accuracy.json`, with its test rows, its checkpoint state and the console log.

**On the test split** (TL-s0 against TL0 at $n_c=16$ with B's corners, 2,400 pieces; the hats had nothing to learn on 40):

| measure | median | 10th percentile | 90th percentile |
|---|---|---|---|
| $\rho$, the 36 learned columns | **0.669** | 0.318 | 0.790 |
| $\rho$, the hats | 0.044 | −0.271 | 0.629 |
| $\rho$, $j=1$–3 | 0.049 | −0.010 | 0.126 |
| $\rho$, $j=4$–8 | 0.706 | 0.340 | 0.821 |
| $\rho_p$, the particular field | 0.939 | 0.755 | 0.987 |
| $\kappa_{\text{kill}}$ | 0.657 | 0.530 | 0.789 |
| control: each piece fed the next piece's inputs | −0.569 | −3.37 | 0.235 |
| control: frozen at S's weights, through $\mathcal C$ | 0.290 | 0.070 | 0.454 |

- **By resolution**, $\rho$ is 0.736 at $n=32$, 0.687 at 64 and **0.356 at 128**. By shear it is 0.46 below $\sup\lvert\mu\rvert=0.15$ and 0.67–0.69 above.
- **K** holds: $e_K$ at most $1.4\times10^{-15}$. **CX** is at most $2.2\times10^{-13}$, against $10^{-10}$.
- **E1** (reported, not gated) is not met. The median $e_H$ is 0.057 over all 64 columns and $e_{H,L}$ 0.095 over the learned ones, against E1's 0.01 (P7).

**On the held-out problems** ($n=64$; the energy error in %, as registered / lowered by 300 K where they differ; bold is the better of TL0 at 16 and TL-s0):

| problem | floor | U | TL0, $n_c=16$ | TL0, $n_c=32$ | TL-s0 | TL-s1 | E4, TL-s0 |
|---|---|---|---|---|---|---|---|
| bend-3, steady | 0.043 | 0.342 | **0.055** | 0.046 | 0.205 | 0.207 | met |
| bend-3, step | 0.006 / 0.053 | 0.221 / 1.97 | 0.034 / 0.304 | 0.016 / 0.145 | **0.031 / 0.276** | 0.033 / 0.294 | met |
| s-channel, steady | 1.001 | 32.2 | 4.76 | 2.32 | **4.60** | 4.60 | **missed**: energy above 2% |
| s-channel, step | 0.129 / 1.236 | 1.14 / 10.9 | **0.227 / 2.18** | 0.154 / 1.48 | 0.240 / 2.31 | 0.237 / 2.28 | **missed**, lowered: 2.31% |
| insert-round, steady | 0.043 | 7.76 | **0.801** | 0.362 | 0.906 | 0.899 | met |
| insert-round, step | 0.004 / 0.025 | 0.613 / 4.21 | **0.068 / 0.467** | 0.031 / 0.211 | 0.083 / 0.573 | 0.083 / 0.569 | met |

- **E4 counts:** TL-s0 and TL-s1 meet E4 on 4 of 6. TL0 meets it on 3, 4, 4 and 5 at $n_c=4$, 8, 16 and 32; U on 2; S′'s runs on 1 each.
- **Both misses are the energy clause.** TL-s0's max error is at most 0.63% of the range everywhere, against bars of 1.0–1.6%.
- **TL0's s-channel steady is not near its floor.** At $n_c=16$ it leaves 4.76% against a classical floor of 1.00%, and 2.32% even at $n_c=32$. The held-out variant changes the channel's material (copper) and puts $\cos\pi s$ and $\cos2\pi s$ profiles on the ends; the seen problem with the same pieces gives TL0 1.21%, as in §17. The registration checked that the held-out floors are under 2%, not that TL0 comes near them.

**The coupled attribution** (coarse first, then the network, as fractions of U's energy error above the floor; the same in both frames):

| problem | the coarse level | the network | the network, of what the coarse level left | $n_c^\ast$ |
|---|---|---|---|---|
| bend-3, steady | 0.960 | −0.501 | −12.5 | 2.5 (extrapolated) |
| bend-3, step | 0.869 | +0.015 | +0.11 | 17.5 |
| s-channel, steady | 0.879 | +0.005 | +0.04 | 16.5 |
| s-channel, step | 0.902 | −0.013 | −0.14 | 14.9 |
| insert-round, steady | 0.902 | −0.014 | −0.14 | 14.2 |
| insert-round, step | 0.894 | −0.025 | −0.24 | 13.1 |

No problem is at its floor. On the laptop, TL costs 0.037–0.051 s a piece and TL0 at $n_c^\ast$ 0.026–0.040 s. These are laptop seconds, not the price.

**The seen six, beside** (TL-s0 against TL0 at 16): TL-s0 meets E4 on 5 and so does TL0. Both miss the s-channel step lowered by 300 K. On the seen bend-3 steady TL0 is almost exact (0.0003%) and TL-s0 leaves 0.195%. On the held-out one TL-s0 goes from TL0's 0.055% to 0.205%, about as much again.

**The other gates, read from the record** (stage C's verdicts agree, §19.4):
- **Met on both runs**, on every held-out piece in both frames: K, D, L, SE2 and CX. CX is at most $2.7\times10^{-15}$, the particular field's $2.0\times10^{-14}$. For TL-s0, E3, SE1 and SE3 are also met.
- **L** holds on the first 100 test pieces for both runs.
- **DUAL** holds on 63 benchmark and 100 test pieces; the smallest relative eigenvalue, $-7.4\times10^{-15}$, is round-off.
- **KT:** 5 iterations on bend-3, the s-channel and insert-round, as many as the classical expert.
- **The rollout:** 20 steps on the s-channel, 180 network calls, the modes kept bit for bit at every later step.
- **NC is missed.** Its test clause holds (median 0.669 against 0.3; 10th percentile 0.318 against 0). Its benchmark clause does not: TL's energy error is above TL0's on 4 of 6 held-out problems.
- **E4 is missed:** 4 of 6.

**The predictions** are judged in §19.4.

**What it means.**
- **The rework's structure held everywhere it was measured.** That covers the coarse exactness, the kernel, the duality bracket and the iteration, and the rollout's modes held bit for bit.
- **The coarse level carries the rework's coupled gain.** It removes 87–96% of U's error above the floor on every held-out problem.
- **On the generated pieces, the network does what it was trained for.** It removes two-thirds of TL0's error on the learned columns at the median. The controls show the gain depends on each piece's own inputs (shuffled, −0.57), and that training added most of it (frozen at S's weights, 0.29).
- **But the gain sits in $j=4$–8.** The hats and $j=1$–3 get 0.04–0.05, and the hats are worse than TL0's on more than a tenth of the pieces.
- **[AI Inference]:** the coupled answer reads mostly the gentle groups. P4 found this, and §17 found $j=1$–3 holding 0.69 of the s-channel steady's network term under B. A network that improves $j=4$–8 and leaves the hats where they were, or worse, moves the coupled error little or the wrong way. $n_c^\ast$ of 13–17.5 says TL is worth about TL0's own coarse size.
- **[AI Inference]:** P6 survives $\mathcal C$. The coarse space takes the gentlest part exactly, but what lies between it and $j=4$–8 is still weighted by energy, the loss's way, not by interface data.
- **[AI Inference]:** the held-out s-channel steady's 4.76% under TL0 halves from $n_c=16$ to 32. So most of it lies just finer than the coarse grid, in the part $\mathcal C$ leaves to the network, and TL removes 0.16 points of it.

### 19.3 The one evaluation, stage B: the price (2026-11-11)

**The box.** The owner approved a Quebec machine in chat ("Yes, rent it"). That offer and its named fallback were taken before the create, so the owner chose again: "Argentina, \$0.168/h (Recommended)". That is the machine the price protocol's dry run passed on (§18): a whole AMD Ryzen 9 5900X, 23 CPUs in its container's quota, \$0.172 an hour with its storage (instance 60286902).
- **An interrupted create still rented it.** The box idled about 1.5 hours, with no self-stop, until the session resumed; the runner is what arms the self-stop.
- **The uploads were checked.** The payload of A's commit (`price-twolevel-4a2a359.tar.gz`, sha256 `6f837a8b…`), S's weights and TL-s0's weights were each checked by sha256 on the box. The driver, `chartop_evaluate_twolevel_box.sh`, then ran the pinned runner's install and smoke run, armed a 2-hour self-stop, and ran stage B.
- **Three price runs took 21 minutes.** Every measurement counted, and each of PR's 22 problems had its three. The A/A check held within 0.0027 of 1, and the BLAS pools held in every run.
- **Collected, then destroyed.** Every file came off by tar over ssh, and its sha256 was compared with the box's. Credit went from \$7.88 to \$7.53, **\$0.35**, about \$0.26 of it the idle hours.

**The price** (geometric means of the per-problem medians, as a fraction of the classical solve's cost; lower is cheaper):

| arm | all 22 | the held-out six | R2's sixteen | per-problem medians |
|---|---|---|---|---|
| TL (TL-s0) | **1.050** | 1.061 | 1.045 | 1.018–1.104 |
| TL0, $n_c=16$ | 0.836 | 0.822 | 0.842 | 0.776–0.898 |
| H2r | 0.418 | 0.978 | 0.304 (0.306 registered) | 0.227–1.002 |

- **TL is dearer than the classical solve on every problem**, by 2–10%. PR's record finds it faster than none of the classical solve, H2r and TL0.
- **The network adds about a fifth of a classical solve** to TL0's cost (1.050 against 0.836).
- **H2r reproduces its registered price on R2's layouts** (0.304 against 0.306). On the held-out six it costs about what the classical solve does.

### 19.4 The one evaluation, stage C: the verdicts (2026-11-11)

`--judge` wrote `out/chartop/evaluation-twolevel-20261111-004223.json` and `.txt`, once.

| gate | verdict |
|---|---|
| K, D, L, E3, SE1, SE2, CX, DUAL, KT | met |
| E4 | **not met**: 4 of 6. It misses the s-channel steady, and the s-channel step lowered by 300 K |
| NC | **not met**. The test clause holds (median 0.669, 10th percentile 0.318). TL is worse than TL0 on 8 of the 12 problem-frames |
| PR | measured: TL 1.050, TL0 0.836, H2r 0.418 |
| reported | E1 not met. SE3 met on every problem. S11, the 20-step rollout on the seen s-channel step, meets E4 at steps 1, 5, 10 and 20. The ledger: §19.2 |

**The predictions:**

| # | prediction | verdict | measured |
|---|---|---|---|
| 1 | CX, K, D and L hold on every run | **held** | on both runs, every piece and frame |
| 2 | NC is met | **refuted** | the test clause holds; TL is worse than TL0 on 4 of 6 held-out problems |
| 3 | TL meets E4 on all six held-out problems | **refuted** | 4 of 6 |
| 4 | $\rho$ is lowest on the hats | **held** | 0.044, against 0.049 for $j=1$–3 and 0.706 for $j=4$–8 |
| 5 | TL is not faster than the classical solve | **held** | 1.050 |

**What the one evaluation says.**
- **The guarantee side of the rework is sound.** Every structural gate was met on the test split, on the held-out problems in both frames and in the rollout. With $\mathcal C$ and B's corner patches, the two-level form is coarse-exact to round-off whatever the weights.
- **The coarse level is what pays.** Untrained, TL0 at $n_c=16$ removes 87–96% of U's coupled error above the floor. It meets E4 on as many held-out problems as the trained TL, at 0.84 of a classical solve.
- **The trained network adds no coupled accuracy and costs about a fifth of a solve.** It is worse than TL0 on 4 of the 6 problems.
- **[AI Inference]:** for CO-D, on this evidence and at this training budget, the network is not needed. The rework asked this first (§0). A network would earn its place only by reaching the hats and $j=1$–3, which its loss does not weight (P6).

### 19.5 Next

**Phase 6**, the write-up and W363 in the gap-worklist, was done on the owner's go: §20.

---

## 20. Phase 6: the write-up and the carry-over (2026-11-12)

**Written** on the owner's go in chat on 2026-11-12 ("yes" to starting Phase 6). The handoff asked for sections on Phase 2, the probe, the registration and the evaluation, with the network's ledger. They are §13 (with the corner treatment's §15–§17), §14, §18 and §19. This section gathers what they found, says plainly what the network added, sets §1's problems against the record and states what carries over. The worklist row is [[gap-worklist]] W363.

**In one paragraph.** The rework made the coarse scale structural, and it worked:
- Untrained, the coarse level with B's corner patches removes 87–96% of U's coupled error above the floor on Stage 0's held-out problems.
- Every guarantee held to round-off.

The network trained on top of it adds nothing to the coupled answer:
- It learns the mid sines ($j=4$–8), which the coupled answer reads little.
- It barely moves the hats and $j=1$–3: a median $\rho$ of 0.04–0.05.
- It costs about a fifth of a classical solve.

**[AI Inference]:** for CO-D, on this record and at this budget, no network is needed. The question §0 put first, untrained, is answered.

### 20.1 The arms on the held-out problems

| arm | what it is | trained | E4, held-out six | price, all 22 problems (R2's protocol) |
|---|---|---|---|---|
| U | S's form with its networks answering zeros: the plain square and the strips | no | 2 of 6 | not priced |
| S′ | S's form trained at the comparison budget in the forms registration, reused | yes | 1 of 6, each run | not priced |
| TL0 | U, then $\mathcal C_e$: the coarse level with B's corner patches ($n/8$) | no | 3, 4, **4** and 5 of 6 at $n_c=4$, 8, **16** and 32 | **0.836** at $n_c=16$; the others not priced |
| **TL** | TL0 at $n_c=16$ with S's networks, trained 37,500 steps from S's weights | yes | **4 of 6**, both runs | **1.050** |
| H2r | R2's hybrid: reuse on squares, the classical solve elsewhere | no | not judged here | 0.418 |

### 20.2 What the network added, plainly

**Nothing to the coupled answer, at about a fifth of a classical solve.**
- **On the generated test pieces it works as trained.** It removes a median 0.669 of TL0's error energy on the 36 learned columns (10th percentile 0.318), and 0.939 on the particular field.
- **The controls show the gain is learned and specific to each piece.**
  - Fed the next piece's inputs, the median falls to −0.569.
  - At S's weights through the same $\mathcal C_e$, it is 0.290.
  - $\mathcal C_e$ discards a median 0.657 of the energy of the network's raw output ($\kappa_{\text{kill}}$).
- **But the gain lands where the coupled answer reads little.**
  - By group: $j=4$–8 0.706, $j=1$–3 0.049, the hats 0.044. The hats are worse than TL0's on more than a tenth of the pieces.
  - By resolution, $\rho$ falls from 0.736 at $n=32$ to 0.356 at 128.
- **On the held-out problems:**
  - The coarse level removes 0.87–0.96 of U's error above the floor.
  - The network removes between −0.50 and +0.015 of it.
  - TL is worse than TL0 on 4 of the 6.
- **$n_c^\ast$**, the coarse size at which TL0 alone would match TL, is 13.1–17.5 on five problems and 2.5 (extrapolated) on bend-3, steady. TL is worth about TL0's own $n_c=16$.
- **The price:** TL costs 1.050 of a classical solve and TL0 0.836. The network adds about 0.21 of a solve and buys no coupled accuracy.

### 20.3 §1's problems against the record

| # | problem (§1) | what the rework did | where it stands |
|---|---|---|---|
| P1 | the uniform-temperature sum broken | $\Pi$ kept; Lemma B carries it through $\mathcal C$ and $\mathcal C_e$ | **held**: $e_K$ at most $1.4\times10^{-15}$ on the test split, and K met (§19.2) |
| P2 | the modes saw the load | the split form kept | **held**: L met on both runs (§19.2) |
| P3 | the corners | B's fine-hat patches at non-right corners, classical (§15–§17) | **treated at non-right corners**: their share of the s-channel's excess fell from 0.85–0.88 to 0.02–0.03 (§17). Interface–side points are untested: no benchmark piece has an interface inside it |
| P4 | the coupled error sits in the gentle modes | the coarse component made exact, whatever the weights | **mostly answered by the coarse level**: 87–96% of U's error above the floor, untrained (§19.2). What is left lies just finer than $n_c=16$: the held-out s-channel steady's 4.76% halves at 32 |
| P5 | a network cannot produce global responses | it is no longer asked for them | **structural for the coarse part.** Just finer than the coarse grid the network still barely reaches the hats and $j=1$–3 ($\rho$ 0.04–0.05) |
| P6 | the loss and the gate disagree | §3.4 expected the loss, seeing only $\mathcal Q_cE$, to agree with the gate by construction | **not removed.** **[AI Inference]:** inside $\mathcal Q_cE$ the loss still weights by energy. Trained, $j=4$–8 improved while the coupled error rose on 4 of 6 problems (§19.2) |
| P7 | E1's bar lies below the form's floor | E1 reported, not gated | **a rule kept**: the median $e_{H,L}$ is 0.095 against E1's 0.01 (§19.2) |
| P8 | the principal part's error grows with the shear | the true operator at the coarse scale | **not split by shear for TL0** on this page. The network's $\rho$ is 0.46 below $\sup\lvert\mu\rvert=0.15$ and 0.67–0.69 above (§19.2) |
| P9 | the price | TL0 has no fine factorisation | **TL 1.050, TL0 0.836, H2r 0.418** (§19.3). TL is dearer than the classical solve on every problem |
| P10 | the particular field generalised poorly | $\mathcal C_p$ on the particular field | **answered**: untrained, $e_p$ fell from 0.69–0.70 to 0.11–0.12 (§13); trained, $\rho_p$ is 0.939 on the test split (§19.2) |

### 20.4 What carries over

§8's Phase 6 named two carry-overs before any result. Against the record:
- **"Stage 1's CO-E takes $\mathcal C$ with a vector bilinear $P$."** $\mathcal C$'s guarantees held wherever they were measured:
  - Proposition A and Lemma B on $V_e$;
  - CX at most $2.2\times10^{-13}$ on 2,400 test pieces;
  - the rollout's modes bit for bit.

  **[AI Inference]:** what carries over is the coarse level and Phase 2's order of work, not the network on top. Measure the two-level expert untrained first. Train on its residual only where a control shows a remainder that reaches the coupled answer. Then judge the trained part against simply raising $n_c$, which $n_c^\ast$ measures, at equal cost.
- **"A learned $P_\theta$ (§6) only where a Phase-2-style control shows the bilinear coarse space failing."** In CO-D it did not fail in §6's sense.
  - What TL0 left fell by about half per doubling of $n_c$ (0.45–0.55 from 16 to 32, §13), as a resolution limit does.
  - The one place it fell more slowly, the s-channel's non-right corners, B's classical patches treated (§17).

  **[AI Inference]:** CO-D gave no case for $P_\theta$. §6's candidates remain untested: thin walls, small holes, high-contrast inclusions, CO-T at high Péclet, and oscillatory physics.
- **Built and tested, for any later registration:**
  - the corner patches (`coarse.CornerTreatment`, `coarse_patches.py`);
  - the ledger (`twolevel.rho`, `attribution`, `equivalent_nc`, `kappa_kill`);
  - the relative objective with methods 1–3;
  - the price protocol with its A/A check and fixed BLAS threads (`chartop_price_twolevel.py`).

### 20.5 What it changes

- **The CO-D expert stays H2r**, with nothing learned (W362's row). It is exact on R2's layouts, priced here at 0.304 against 0.306 registered.
- **[AI Inference]:** off the squares, TL0 is the cheaper classical candidate.
  - On the held-out six it costs 0.822 of a classical solve, where H2r costs 0.978.
  - But it misses E4 on two of them: the s-channel steady, and the s-channel step lowered by 300 K.
  - Using it would need its own registration, with its coarse size chosen first.
- **W362 stays open.** The rework was judged on Stage 0's layouts with new materials, not on layouts whose outlines do not cut into squares. Its network added nothing coupled there.
- **[AI Inference]:** a next learned idea for CO-D would have to change what the loss weights, toward the hats and $j=1$–3 as the coupled answer reads them (P6). B2's mode prior did that without the coarse level and tied (CP, forms §11.10). Under $\mathcal C$ it has not been tried.

### 20.6 Cost and records

- **The rework's boxes cost \$1.75** of the registration's \$10 ceiling:
  - the price protocol's dry run, \$0.32;
  - training, \$1.08;
  - the price, \$0.35.

  Each box was destroyed after its files had been checked against the box's own copies.
- **Commits:**
  - the training records, `4a2a359`;
  - the evaluation's stage A, `7dab3f6`;
  - its stages B and C, `fd607bd`;
  - §19, `99258ed` and `eb0707d`;
  - this section and W363, the commit that adds them.

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
- [[gap-worklist]] — W363, this rework (done); W362, the pieces that are not squares (open)
