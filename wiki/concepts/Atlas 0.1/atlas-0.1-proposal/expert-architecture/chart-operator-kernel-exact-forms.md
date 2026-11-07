# Chart operator — the kernel-exact and split output forms (W356, W357), built and checked untrained

**Type:** Concept page — **build record and draft registration** (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** built and checked on 2026-10-18, written up on 2026-10-19. §1–7 and §10 use **untrained weights only**: zero heads, or seeded random weights. §8 is the next registration: accepted by the owner and hash-pinned on 2026-10-19, its bars in `atlas/chartop/gates_forms.py` and pinned in `out/chartop/registered-forms.txt`. Its five runs were trained and evaluated once the same day (§8.9–8.11): K, D, L, E3, SE1–SE3 and DUAL are met, E1, E4 and KT are not, and CW is a tie. `atlas/chartop/gates.py`, Stage 0's registration, is unchanged.
**Hub:** [[chart-operator-architecture]] · **Follows:** [[chart-operator-stage-0]] §9.7 and §12, items 1–2 · **Worklist:** [[gap-worklist]] W356, W357; W358 in §10 · **Family:** [[chart-operator-family-catalog]] (CO-D)

---

## 1. What Stage 0 left

Stage 0's bubble form, $\hat H=H^{(0)}+\beta N$, keeps every trace exact whatever the weights, and nothing else. Two defects were found after the one evaluation ([[chart-operator-stage-0]] §9.7):

- **The hats' sum (W356).** Every steady diffusion operator reproduces a uniform temperature exactly as the sum of the four corner-hat modes, and so does the plain square's principal part. Arm B's correction broke the sum: a uniform temperature came back up to $5.3\times10^{-3}$ of its level away on bend-3's pieces (§7.3). The hats carry a fraction of a per cent of $e_H^2$ (0.1–0.2%), so neither the training objective nor gate E1 could see it. The coupled answer multiplies the defect by the temperature level, and at 300–400 K it dominated: B's steady errors of 24%, 42% and 67% fell to 4.5%, 9.1% and 10.6% when every temperature was lowered by 300 K.
- **The modes see the load (W357).** One forward pass returned the modes and the particular field from all five input channels. Setting the source channel from 0 to 1 more than doubled B's steady errors, though the exact modes do not depend on the source.

---

## 2. The identity: a uniform temperature is a fixed point of the step

**Intuition.** Put the same temperature $T_0$ on every side of a piece, and start the step from $T_0$ everywhere. Nothing happens, and the answer is $T_0$. For a steady piece this says that the four corner hats, each the response to one corner's value, add up to one. For a backward-Euler step it says more. The hats alone add up to less than one, because the heat capacity holds the interior near its old temperature rather than the boundary's. The particular field of the old temperature makes up the difference exactly.

**Rigorously.** A piece's energy is $M=K+\operatorname{mass}(\sigma)$ with $K\mathbf 1=0$ and $\sigma=\tilde J s_0$, where $s_0=1/\mathrm{Fo}$, zero when steady. Let $c_1$ be the port coefficients of the constant, $Qc_1=\mathbf 1$: the four hats of the sine-hat basis, or the four $j=0$ cosines of the cosine basis. Let $u_p(\mathbf 1)=A_I^{-1}\bigl(\operatorname{mass}(\tilde J)\mathbf 1\bigr)_I$ be the particular field of the unit uniform source. The interior rows of $M\mathbf 1=s_0\operatorname{mass}(\tilde J)\mathbf 1$ read $A_I\mathbf 1_I-B\mathbf 1_G=s_0\bigl(\operatorname{mass}(\tilde J)\mathbf 1\bigr)_I$, so

$$Hc_1+s_0\,u_p(\mathbf 1)=\mathbf 1_I .$$

Steady, this is $Hc_1=\mathbf 1_I$. The plain square satisfies the same identity with $H^{(0)}$ and $u_1^{(0)}=(A_I^0)^{-1}(h^2\mathbf 1)$.

**Checked** on exact classical solves of four generator pieces ($n=32$), in both port bases, steady and at $\mathrm{Fo}=0.01$, $1$ and $100$. The residual is at most $6.7\times10^{-15}$. The hats alone miss one by up to $0.96$–$0.98$ at $\mathrm{Fo}=0.01$, $0.06$–$0.08$ at $\mathrm{Fo}=1$ and $10^{-3}$ at $\mathrm{Fo}=100$. One of the four pieces is kept as `tests/test_chartop_model.py::test_exact_modes_make_the_uniform_state_a_fixed_point`.

**Why the fixed point is the backward-Euler analogue of the steady constraint.**
1. A step has no kernel: $M\succ0$ once $\sigma>0$. What the coupled answer needs from the constant is that the datum drops out. Lowering every temperature by $T_0$, the boundary data and $T_{\text{old}}$ together, lowers the exact step's answer by exactly $T_0$, because $K\mathbf 1=0$. A trial space keeps this if it contains the uniform state at every level, and the identity above is exactly that condition.
2. The steady constraint, imposed on a step, would force $\hat Hc_1=\mathbf 1$ where the exact modes have $Hc_1$ as low as $0.02$: a bound on accuracy that no training could remove.
3. Leaving steps unconstrained leaves the level in the network's particular field. Stage 0's principal part has no particular field, so on every step it misses $T_{\text{old}}\,s_0u_p(\mathbf 1)$. That is nearly all of its 60%, 52% and 32% on the three steps: lowered by 300 K, where no particular field is needed, its absolute error is 0.45–1.3% as large (§7.1).

---

## 3. The kernel-exact form (`"kernel"`, W356)

**Intuition.** Let the network say what it likes about the modes, then correct the one combination it must not get wrong. The combination's defect is measured at every node and taken off the hats, each hat taking its share of the constant near its own corner. The side modes are never touched. The source's level is handled the same way: the network never sees it, and its response comes from the same level field the hats were corrected against.

**Rigorously.** One forward pass returns $4m$ mode corrections $N_k$, a particular correction $N_p$ and a **level-field** correction $N_1$. With the plain square's $H^{(0)}$ and $u_1^{(0)}$ at the piece's $s_0$, and the bubble $\beta=16\xi(1-\xi)\eta(1-\eta)$,

$$\hat H^{\text{raw}}=H^{(0)}+\beta N,\qquad \hat u_1=u_1^{(0)}+a(s_0)\,\beta N_1,\qquad a(s_0)=\max\lvert u_1^{(0)}\rvert,$$

so $\hat u_1$ is the form's answer for $u_p(\mathbf 1)$, scaled by the plain square's own level: $0.074$ steady, about $1/s_0$ for short steps. Then, in float64,

$$r=\hat H^{\text{raw}}c_1+s_0\,\hat u_1-\mathbf 1,\qquad \hat H_k=\hat H^{\text{raw}}_k-W_k\,r,\qquad W_k=\frac{c_{1,k}\,H^{(0),\text{steady}}_k}{\sum_j c_{1,j}^2\,H^{(0),\text{steady}}_j}.$$

$W$ is the plain square's steady modes of the kernel columns, a partition of unity on the interior nodes: $\sum_k c_{1,k}W_k=1$. For the sine-hat basis it is the bilinear corner functions $(1-\xi)(1-\eta)$, $\xi(1-\eta)$, $\xi\eta$ and $(1-\xi)\eta$, to $4\times10^{-15}$. So

$$\hat Hc_1+s_0\,\hat u_1=\mathbf 1\quad\text{to float64 rounding, whatever the weights.}$$

The source is split into its level and the rest, $g=\bar g+g'$, with $\bar g=\sum_e\tilde J_e\,g_e/\sum_e\tilde J_e$ the load-weighted mean ($g_e$ the mean of element $e$'s four nodal values). Then

$$\hat u_p=\bar g\,\hat u_1+\operatorname{rms}(g')\,\beta N_p ,$$

and the network sees $g'$ alone, as $Jg'/\operatorname{rms}(Jg')$: version 1's $\hat f$ with the level removed.

**What follows, whatever the weights.**
- **Traces exact**, as before: the form returns interior values only.
- **The uniform state is a fixed point.** With data $T_0$ and source $s_0T_0$, $\hat u_p+\hat H(T_0c_1)=T_0(s_0\hat u_1+\hat Hc_1)=T_0$.
- **The datum drops out.** Lowering every temperature by $T_0$ subtracts $s_0T_0$ from $g$, which leaves the network's input unchanged. It also subtracts $T_0c_1$ from the interface coefficients and $s_0T_0\hat u_1$ from $\hat u_p$, so the coupled answer moves by exactly $T_0$, steady or not.
- **T26 holds unchanged.** The exact fields satisfy the identity, so they are a fixed point of the projection, and the label-free objective's minimiser is the same.
- **The square's symmetries commute with the projection.** Kernel columns go to kernel columns with sign $+1$, and $W$ moves with them, so D4 augmentation stays exact (tested for all eight symmetries in both bases).
- **Precision.** $H^{(0)}$ and $u_1^{(0)}$ are computed in float32 inside the network, as in Stage 0. The projection runs in float64, so the identity holds to float64 rounding whatever the float32 or bfloat16 rounding was. As initialised, this alone removes the principal part's own level defect: on bend-3 the coupled error falls from $1.7\times10^{-4}$ to $3.4\times10^{-6}$ (§7.2).

**Choices made, and why.**
- **A level field, rather than $u_p(\mathbf 1)=(\mathbf 1-\hat Hc_1)/s_0$.** At $\mathrm{Fo}=100$, $\mathbf 1-Hc_1\approx10^{-3}$, and deriving the level field from the hats would divide their error by $s_0=0.01$: a uniform heating would inherit a hundredfold amplification. The other direction multiplies $\hat u_1$'s error by $s_0$, and $\hat u_1$ is itself of size $\mathrm{Fo}$ when $s_0$ is large. So the error stays relative in both limits.
- **A partition of unity, rather than an equal quarter to each hat.** Both are exact. The partition puts a defect near corner $c$ onto hat $c$, where that mode lives, instead of onto all four.
- **The steady source's level goes through $\hat u_1$ too.** Steady, $\hat u_1$ is the response to a unit source. The particular network then sees only sources of zero mean, steady or not.
- **Not a host datum.** A host that couples temperatures relative to a datum would shrink the defect, as §9.7's inference said, but the answer would then depend on the datum chosen. The form makes the answer independent of it.

---

## 4. The split form (`"split"`, W357)

- **The mode network** sees $(\log J,\ \operatorname{Re}\mu,\ \operatorname{Im}\mu,\ \log\hat\kappa)$, and Fo through FiLM. It returns the $4m$ mode corrections and the level correction, every field that does not depend on the load.
- **The particular-field network** sees the same four channels and the level-free source, and returns $N_p$.
- The projection and the source split are §3's. The modes and the level field are then the same for every load, bit for bit, as the exact modes are by T25's structure.
- **A march** that keeps the first call's modes keeps a level field consistent with them (the rollout, S11). For the single-network kernel form, whose level field follows the load's shape, `LearnedExpert.respond(modes_from=...)` pairs the kept level field with the kept modes. Tested: unpaired, the uniform state misses by $5\times10^{-3}$ with random weights of scale $0.1$; paired, by less than $10^{-12}$.
- **Size.** Both networks at Stage 0's 1 M widths come to 2.20 M parameters; with a 0.3 M particular network, 1.38 M. The kernel form at 1 M has 1,100,778, B's 1,100,753 plus the level row. The forward cost of two networks has not been priced (§9).

---

## 5. The measure that weights the kernel as the coupled answer does

**Intuition.** $e_H$ asks how wrong the modes are on average over random boundary data, each mode weighted by its energy. On a steady piece a uniform temperature has no energy, so it gets no weight at all. Real data carry a level, often hundreds of kelvin against variations of tens, and the coupled answer multiplies the modes' error on the uniform state by that level. So measure the uniform state directly, and weight it by the level.

**Rigorously.** For any expert, one call on a piece's uniform state at unit level (data $1$, source $s_0$) gives its residual

$$r_K=\hat u_p(s_0\mathbf 1)+\hat Hc_1-\mathbf 1\qquad(\hat Hc_1-\mathbf 1\ \text{when steady}),$$

and

$$e_K=\frac{\lVert r_K\rVert_{A_I}}{\sqrt{\operatorname{tr}\Lambda}},\qquad e_{K,\infty}=\max\lvert r_K\rvert .$$

$e_K$ is on $e_H$'s scale, and $e_{K,\infty}$ is the pointwise error of a unit uniform state. Theorem 2's network term for one piece is $\lVert e_p+E\,Rc^\star\rVert_{A_I}$. For traces $c=\theta c_1+\xi$, with $\xi$ unit and uncorrelated and the source $\theta s_0$, its level part is exactly $\theta r_K$ by the identity. So its mean square is $e_H^2\operatorname{tr}\Lambda+\theta^2\lVert r_K\rVert^2_{A_I}$, and over the energy $\xi$ carries,

$$e_H(\theta)=\bigl(e_H^2+\theta^2e_K^2\bigr)^{1/2}:$$

the level adds $\theta e_K$ in quadrature. A step's uniform state has energy of its own, $\theta^2s_0$, which is left out of the denominator, so that a large level cannot make an error look small. For a kernel-exact form $e_K=0$, so $e_H(\theta)=e_H$ at every level.

**The level of a benchmark piece**, $\theta$, is the mean of its four vertex values in the classical coupled solution over the root mean square of the rest of its coefficients. The steady medians are 140 (bend-3), 252 (s-channel) and 1,831 (insert-round). On the steps the variation sits near the hot wall, and $\theta$ reaches $10^4$–$10^7$.

**In code.** `forms.kernel_residual`, `kernel_measures` and `level_weighted_e_H` (numpy, any expert, through the host's interface); `forms.kernel_error_torch` for the training code. `train.validate` logs $e_K$ for the new forms.

---

## 6. What was built

| file | what changed |
|---|---|
| `atlas/chartop/forms.py` (new) | the identity's pieces: `kernel_coefficients`, the partition of unity (`KernelSquare`), `level_principal`, `project`, `residual`, `split_source`; `KernelExpert` with the forms `"kernel"` and `"split"`; `make` and `call` for the trainer; the measure |
| `atlas/chartop/model.py` | `UNet` and `SpectralNet` take an input-channel count, default 5 |
| `atlas/chartop/expert.py` | `LearnedExpert` passes each piece's element weights and source to the new forms, keeps their level fields, and pairs them with kept modes in a march |
| `atlas/chartop/host.py` | `Response.level`, optional |
| `atlas/chartop/train.py` | builds, calls, validates ($e_K$ logged) and loads the new forms; `Arm.p_size`. Stage 0's call path is unchanged |
| `atlas/chartop/benchmarks.py` | `datum_shift(b, dT)`: the data and $T_{\text{old}}$ moved together |
| `scripts/chartop_forms_controls.py` (new) | §7's controls → `out/chartop/forms-controls.json` |

**Stage 0 still loads and answers as it did.** Before and after the edit to `model.py`, one-off: arm B's trained weights, and every Stage 0 form and backbone with seeded random weights, at $n=32$ and $64$. All 14 outputs are bit-for-bit equal.

**The tests**, all with random or zero weights, none trained:

| test (`tests/test_chartop_model.py` unless named) | what it shows |
|---|---|
| `test_exact_modes_make_the_uniform_state_a_fixed_point` | the identity for exact solves (§2) |
| `test_the_partition_of_unity` | $W$ sums to one and is zero off the kernel columns; for sine-hat, the bilinear corner functions |
| `test_kernel_exact_whatever_the_weights` | both forms, both bases, weight scales $0.02$ and $0.3$ (raw outputs up to $10^8$), steady and $s_0=3$ and $100$: the identity's residual is at most $10^{-14}(1+\lvert r\rvert)$, $r$ the raw residual |
| `test_the_projection_moves_only_the_kernel_columns` | the side modes are the network's own $H^{(0)}+\beta N$ bit for bit, and the hats move by $-Wr$. A Stage 0 bubble's weights copied in give its side modes to float32 rounding, and the bubble alone breaks the sum |
| `test_zero_heads_give_the_principal_part` | as initialised: $H^{(0)}$ to float32 rounding, $\hat u_p=\bar g\,u_1^{(0)}$, the identity exact |
| `test_a_uniform_source_shift_moves_only_the_level_term` | $g\to g+a$: modes and level field unchanged to $10^{-10}$, and $\hat u_p$ moves by $a\hat u_1$ |
| `test_split_source_is_homogeneous_and_level_free` | the split's homogeneity, its level invariance, and a uniform source being all level |
| `test_the_split_forms_modes_never_see_the_load` | modes and level field bit for bit equal for four loads and any fifth input channel; the single-network form's modes do change |
| `test_the_projection_commutes_with_the_squares_symmetries` | eight symmetries, two bases |
| `test_the_kernel_forms_are_trainable` | the label-free objective's gradient reaches the mode rows, the level row and the particular field (one backward pass, no optimiser step) |
| `test_a_kernel_expert_saves_and_loads` | through `train.load_expert` |
| `test_the_uniform_state_measure_through_the_expert` | $e_K$ and $e_{K,\infty}$ below $10^{-12}$ for exact solves and both forms; a random Stage 0 bubble misses by more than $10^{-3}$ |
| `test_a_march_keeps_the_level_field_with_the_modes` | §4 |
| `tests/test_chartop_eval.py::test_the_datum_drops_out_of_the_kernel_forms` | bend-3 at $n=16$, steady and step, lowered by 300 K: both forms' coupled errors unchanged to $10^{-9}$, a random bubble's changed by over 1%; a uniform 350 K state reproduced to $10^{-9}$ |

The chartop tests (`tests/test_chartop_*.py`) pass: 184 in all, of which 46 are these, 67 are Stage 0's and 71 are W360's. Stage 0's registration test is unchanged.

---

## 7. The controls, untrained

`scripts/chartop_forms_controls.py` → `out/chartop/forms-controls.json`, run on the laptop on 2026-10-18 in 452 s. The registered benchmarks at $n=64$, 16 sine-hat modes per side. The controls of `scripts/chartop_posthoc_zero_source.py` and `scripts/chartop_diagnose.py` were re-run on the new forms, with the same definitions imported from those scripts.

**The experts.**
- **The principal part**: Stage 0's zero head.
- **The new forms as initialised**: kernel and split with zero heads. They answer identically.
- **Stage 0's bubble with seeded random weights** at two scales, $0.002$ and $0.02$, that bracket arm B. On the steady benchmarks this bubble errs by 3.8–14% and 47–154%, B by 24–67%. Its largest hat-sum defect is $9.4\times10^{-4}$ and $1.2\times10^{-2}$, B's $5.3$–$6.1\times10^{-3}$.
- **The kernel form with that bubble's weights**, its level row zero. On a sourceless piece the projection is then the only difference.
- **The split form with random weights** at the same scales.

**Nothing here is a trained network's accuracy.** The section shows what the forms guarantee, and where they start.

### 7.1 The datum shift, on all six problems

The absolute error $\lVert u-\tilde u\rVert_M$, with every temperature lowered by 300 K, over the same error as registered. On a step the reference energy contains the level ($\sigma\!\int T^2$) and moves with the shift, so the absolute error is what can stay the same.

| problem | principal part | bubble, random $0.002$ | bubble, random $0.02$ | kernel and split forms, six variants: largest $\lvert\text{ratio}-1\rvert$ | largest change of max error, of the range |
|---|---|---|---|---|---|
| bend-3, steady | 0.164 | 0.126 | 0.134 | $1.4\times10^{-10}$ | $8\times10^{-13}$ |
| bend-3, step | 0.0045 | 0.0047 | 0.011 | $1.5\times10^{-13}$ | $1\times10^{-14}$ |
| s-channel, steady | 1.0000005 | 0.61 | 0.15 | $7.6\times10^{-13}$ | $4\times10^{-14}$ |
| s-channel, step | 0.013 | 0.014 | 0.026 | $1.0\times10^{-13}$ | $1\times10^{-14}$ |
| insert-round, steady | 0.99995 | 0.42 | 0.15 | $2.0\times10^{-12}$ | $1.5\times10^{-11}$ |
| insert-round, step | 0.012 | 0.013 | 0.020 | $5.2\times10^{-12}$ | $3\times10^{-13}$ |

- **The new forms are blind to the datum** on every problem, to $1.4\times10^{-10}$, whatever the weights.
- **The bubble is not.** Its error falls by 1.6–8 times with the shift on the steady problems and by 38–213 times on the steps, the pattern B showed on the steady three.
- The principal part's changes on bend-3 steady are its float32 rounding times the level. Its changes on the steps are its missing particular field.
- The exact 16-mode superelement's ratio is one to $10^{-11}$, except on bend-3 steady, where its error is rounding ($2.3\times10^{-12}$) and no ratio is read.

### 7.2 The coupled answer, as registered and lowered by 300 K

Energy errors in per cent; for the steps, as registered and then lowered by 300 K, and for the steady problems one number where the two agree.

| problem | 16-mode floor | principal part | arm B, trained (Stage 0) | **new forms, as initialised** | kernel form, bubble's weights $0.02$ | split, random $0.02$ | bubble, random $0.02$ |
|---|---|---|---|---|---|---|---|
| bend-3, steady | $2\times10^{-10}$ | 0.017 / 0.0027 | 24.4 / 4.5 | **0.00034** | 2.28 | 2.59 | 47.4 / 6.3 |
| s-channel, steady | 0.93 | 5.28 | 42.3 / 9.1 | **5.28** | 5.96 | 7.83 | 83.2 / 12.7 |
| insert-round, steady | 0.031 | 5.58 | 66.8 / 10.6 | **5.58** | 5.66 | 8.40 | 154 / 22.9 |
| bend-3, step | 0.017 / 0.25 | 60.0 / 3.9 | 10.4 / not run | **0.27 / 3.9** | 0.31 / 4.5 | 0.40 / 5.8 | 26.2 / 4.1 |
| s-channel, step | 0.14 / 2.1 | 52.0 / 10.8 | 7.1 / not run | **0.69 / 10.8** | 0.76 / 11.9 | 0.63 / 10.0 | 27.5 / 11.3 |
| insert-round, step | 0.0066 / 0.070 | 32.3 / 4.1 | 7.4 / not run | **0.39 / 4.1** | 0.48 / 5.0 | 0.44 / 4.6 | 20.3 / 4.2 |

**Maximum errors** of the new forms as initialised, over the range, in both frames: 0.0000045%, 0.40% and 0.93% on the steady bend-3, s-channel and insert-round, and 2.6%, 6.0% and 2.3% on their steps. E4's maximum-error bars on the same six are 1.0%, 1.15% and 1.0%, then 1.42%, 3.81% and 1.05%.

- **As initialised, the steps' error falls 76–224 times against the principal part's** in the registered measure, to below arm B's trained 7–10%: 0.27%, 0.69% and 0.39%. It is the fixed point at work: the level field supplies the particular field of $T_{\text{old}}=300$ K, which Stage 0's principal part lacked.
- **That measure flatters a step.** Its reference energy includes the capacity term at 300 K. Lowered by 300 K, the same absolute error is 3.9%, 10.8% and 4.1%: the plain square's modes at each piece's own Fourier number, with no particular field left to supply. B was not run in that frame (§9).
- **On the steady problems the new forms start where the principal part does**: 5.3% and 5.6% on s-channel and insert-round, its modes' own error. On bend-3 the error falls from $1.7\times10^{-4}$ to $3.4\times10^{-6}$, because the float32 rounding of the level is gone.
- **With random weights, the projection alone removes most of a bubble's error.** At scale $0.02$ the kernel form with the bubble's weights errs by 2.3%, 6.0% and 5.7% on the steady problems against the bubble's 47%, 83% and 154%. What remains above the untrained form's error, 2.3, 0.7 and 0.1 percentage points, is the random correction's damage to what the projection leaves free: the hats' differences and the side modes. B showed the same: after the shift its errors still exceeded the principal part's.
- **None of this meets E4.** As initialised, the new forms would pass both of E4's bars on bend-3 steady only. They pass the energy bar on the steps in the registered measure but not their maximum-error bars, and neither bar lowered by 300 K. They are a starting point, not a result, and nothing here was judged.

**The modes under a load (W357's control, steady problems).** The modes are taken from a call whose source is a unit-RMS random field, and the particular field is set to zero.
- The split form's coupled errors are bit for bit the registered ones, at both weight scales.
- The single-network forms, the bubble and the kernel form, change by up to $9.1\times10^{-5}$ relative. At random initialisation the modes barely depend on the source, so this control cannot reproduce B's learned dependence, which doubled its error. The structural guarantee is the test.

### 7.3 The modes, piece by piece

| problem | principal part: $e_H$ / $e_K$ / $e_{K,\infty}$ | B, trained (its own form): $e_H$ / $e_K$ / $e_{K,\infty}$ / $e_H(\theta)$ | bubble, random $0.02$: $e_H$ / $e_H(\theta)$ | kernel form, bubble's weights $0.02$: $e_H$ / $e_H(\theta)$ |
|---|---|---|---|---|
| bend-3, steady | 0.0194 / $3.4\times10^{-7}$ / $6.6\times10^{-7}$ | 0.0195 / $4.9\times10^{-4}$ / $5.3\times10^{-3}$ / **0.071** | 0.0236 / 0.146 | 0.0236 / 0.0236 |
| s-channel, steady | 0.0306 / $3.4\times10^{-7}$ / $6.6\times10^{-7}$ | 0.0306 / $5.2\times10^{-4}$ / $6.0\times10^{-3}$ / **0.133** | 0.0334 / 0.269 | 0.0334 / 0.0334 |
| insert-round, steady | 0.267 / $3.4\times10^{-7}$ / $6.6\times10^{-7}$ | 0.267 / $4.1\times10^{-4}$ / $6.1\times10^{-3}$ / **0.688** | 0.267 / 1.92 | 0.267 / 0.267 |
| bend-3, step | 0.0203 / 0.085 / 0.80 | 0.0208 / 0.013 / 0.14 / — | 0.0270 / — | 0.0270 / — |
| s-channel, step | 0.0330 / 0.072 / 0.59 | 0.0328 / 0.0091 / 0.099 / — | 0.0376 / — | 0.0376 / — |
| insert-round, step | 0.267 / $4.6\times10^{-4}$ / 0.33 | 0.267 / 0.0014 / 0.039 / — | 0.267 / — | 0.267 / — |

Medians over each problem's pieces; $e_{K,\infty}$ is the largest over them.

- **Every variant of the new forms** has $e_K\le4.3\times10^{-16}$ and $e_{K,\infty}\le4.4\times10^{-16}$ on every piece of every problem. The exact classical modes reach $1.5\times10^{-13}$.
- **The measure sees what $e_H$ could not.** B's $e_H$ is within 0.5% of the principal part's on every steady problem, and its $e_H(\theta)$ at the pieces' own levels is 3.7, 4.3 and 2.6 times its $e_H$. On a 400 K level, B's uniform state misses by up to 2.4 K, from the hats' sum alone.
- **On the steps**, $\theta$ is $10^4$–$10^7$, so $e_H(\theta)$ for B and the principal part is dominated by $\theta e_K$ and runs to the thousands. Only $e_K$ is quoted there. The principal part's $e_K$ is its missing particular field.
- **The projection changes $e_H$ only through the hats**, 0.1% of $e_H^2$. The kernel form with a bubble's weights has the bubble's $e_H$ to four figures on every problem.

### 7.4 What the form asks of the network

The second part of `scripts/chartop_diagnose.py`, on validation seeds $2{,}000{,}100$–$2{,}000{,}107$ at $n=32$, 64 and 128 (medians over the eight pieces). $D=(H-H^{(0)})/\beta$ is Stage 0's quotient. Once the projection carries the hats' sum, any $D_c+W_c\tau$ will do, so the hats' smallest quotient is $D_c-W_c\sum_jW_jD_j/\sum_jW_j^2$. The level field's quotient is $N_1=(u_p(\mathbf 1)-u_1^{(0)})/(a\beta)$.

| $n$ | side modes: largest $\lvert D\rvert$ | hats, RMS: Stage 0 / projected | hats, largest: Stage 0 / projected | level field: RMS | level field: largest | its distance from a corner |
|---|---|---|---|---|---|---|
| 32 | 16.4 | 0.075 / 0.067 | 0.55 / 0.48 | 0.33 | 1.37 | 0.044 |
| 64 | 50.2 | 0.077 / 0.068 | 0.66 / 0.68 | 0.34 | 1.90 | 0.022 |
| 128 | 89.2 | 0.078 / 0.068 | 0.93 / 0.95 | 0.35 | 2.44 | 0.011 |

- **The side modes ask exactly what Stage 0's did**, since the projection does not touch them: W358's corner growth (16, 50, 89) is unchanged.
- **The hats ask 11–12% less in RMS**, and about as much at their peak.
- **The level field asks a moderate quotient**, RMS 0.33–0.35. It peaks at the node diagonally next to a corner ($\sqrt2\,h$), like the side modes, but grows only 1.8 times from $n=32$ to 128, where theirs grows 5.4 times.
- **The shares of $e_H^2$** for the new forms as initialised are Stage 0's: hats 0.1%, sines $j=1$–3 5.2%, $j=4$–8 26.1%, $j=9$–15 68.5% (24 pieces at $n=64$).
- **As initialised, on the same 24 validation pieces** (7 steady, 17 transient): Stage 0's principal part has a median $e_K$ of $4.3\times10^{-4}$. Its $e_{K,\infty}$ is $6.6\times10^{-7}$ on the steady pieces (float32) and up to $0.96$ on the transient ones (no particular field). The new forms reach $2.5\times10^{-16}$ and $3.3\times10^{-16}$.

**[AI Inference]:** the level field is a smooth, low-frequency field with a mild corner peak. It should be among the easiest outputs to learn, and it carries the whole uniform-source response, steady and transient. W358's corner growth still governs the side modes, which carry 68.5% of $e_H^2$. These forms do not change that, so E1's bar of $0.01$ is unlikely to be met by them alone.

---

## 8. The next registration: accepted by the owner and hash-pinned (2026-10-19)

*This replaces the draft of commit `f848f3d`, which covered W356 and W357 alone (arms S, S′, K′ and B″, comparisons CK1 and CK2).*

**Accepted.** The owner accepted this section as written, in chat on 2026-10-19: "I accept the draft registration in chart-operator-kernel-exact-forms section 8 as written." It was drafted in commit `0f1b4ae`, and its bars and arms have not changed since.
- `atlas/chartop/gates_forms.py` carries them. Every bar shared with Stage 0 is read from Stage 0's module, which is unchanged ([[chart-operator-stage-0]] §3).
- `tests/test_chartop_gates_forms.py` checks this section against that module.
- **Hash-pinned** on the owner's OK the same day ("Yes, hash-pin gates_forms.py now, the way Stage 0's registration was pinned"). `scripts/chartop_register_forms.py` wrote `out/chartop/registered-forms.txt` with the sha256 of `gates_forms.py` (`94f29f27…`) and of `gates.py`, whose bars it reads (`b26cf4e5…`, Stage 0's pinned hash). The test fails if either file changes.
- Nothing has been trained or rented.

The owner's choices, quoted:
- output form for the side modes: "Bubble + strips". This is W358 as in §10: `forms.corner_bubble`, with strip solves for the sines $j=9$–15 (commits `9ed90e0` and `6594c8a`).
- particular-field network: "1 M (as drafted)";
- registration shape: "Main + W358 control", that is, one main arm and one comparison that isolates W358.

### 8.1 What would be judged

Family CO-D. Everything else is as Stage 0:
- the generator, the data seeds and the splits;
- the three held-out benchmarks at $n=64$, and the envelope;
- the label-free objective, with D4 augmentation;
- sine-hat ports, 16 modes per side;
- the checkpoint kept by validation median $e_H$ (`train.py`);
- E1 judged in float32 on the laptop's CPU (D4).

| arm | networks | form | budget | training seeds |
|---|---|---|---|---|
| **S** (main) | two U-Nets: modes 1 M, particular 1 M (2,198,990 parameters) | split, corner bubble, strips for $j=9$–15 | Stage 0's main, 150,000 steps | 1 |
| S′ | as S | as S | 37,500 steps | 2 |
| **W** (W358's control) | two U-Nets: modes 1 M, particular 1 M (2,199,690 parameters) | split, Stage 0's square bubble, no strips | 37,500 steps | 2 |

- **A training seed** is `train.Arm.seed`: the weights' initialisation and the batch order. S uses seed 0; S′ and W use 0 and 1. Every arm uses the data of the registered seeds.
- **W differs from S′ in W358 alone**, the bubble and the strips together. The registration shape asks for one comparison, so the two are not separated.
- **Both arms start from zero heads** (`model.py`), where the bubble does not enter. At step 0 they differ only in the 28 strip columns, and W's step 0 is exactly §7's "new forms, as initialised".
- **W358's own pass condition was met by no candidate** (§10): a peak that stops growing over $n=32$–128. The corner bubble with strips levels off only from 128 to 256. This draft trains the owner's choice, and it would not close W358.

### 8.2 The gates

| gate | drafted bar | untrained, before training |
|---|---|---|
| **K** (the uniform state) | as the old draft: on every held-out test piece and every benchmark piece, steady and step, $e_K\le10^{-12}$ and $e_{K,\infty}\le10^{-12}$ | old form $\le4.3\times10^{-16}$ and $\le4.4\times10^{-16}$ (§7.3). New form: to be re-measured. The tests of `6594c8a` hold the fixed point to $10^{-14}$ with random weights at $n=32$ |
| **D** (the datum) | as the old draft: on every benchmark problem, steady and step, with every temperature lowered by 300 K, the data and $T_{\text{old}}$ together: $\bigl\lvert\lVert u-\tilde u\rVert_{M,\text{lowered}}/\lVert u-\tilde u\rVert_M-1\bigr\rvert\le10^{-8}$ wherever both errors exceed $10^{-9}\lVert u\rVert_M$, and the maximum error over the range within $10^{-9}$ of the range | old form $\le1.4\times10^{-10}$ and $\le1.5\times10^{-11}$. New form: to be re-measured |
| **L** (the load) | as the old draft, on both arms: the modes and the level field bit for bit the same with the piece's own source, no source and a unit-RMS random source, on every benchmark piece and the first 100 test pieces | old form bit for bit. New form: to be re-measured. The strips read the piece's operator ($a_{11},a_{12},a_{22},\sigma$), never the load |
| **E1** | Stage 0's bins and bars, judged on S: median $e_{H,L}\le0.01$ and 90th percentile $\le0.03$ on the 36 columns the network learns (below), and $e_p$'s bars unchanged. Reported beside it, not judged: $e_H$ over all 64 columns, and $e_K$ | — |
| E3, SE1–SE3, DUAL, KT | Stage 0's bars, unchanged | hold for any outputs with exact traces (T25) |
| **E4** | Stage 0's bars (energy $\le0.02$; maximum $\le$ the 16-mode floor $+\,0.01$ of the range), **met in both frames**: as registered, and with every temperature lowered by 300 K, where a step's reference energy no longer contains the level | W's start (§7.2): steps 0.27/0.69/0.39% registered and 3.9/10.8/4.1% lowered; steady 0.00034/5.3/5.6%. S's start: to be measured with K, D and L |
| **E5** | priced on the box before any long run: the forward pass and one training step at $n=32$, 64 and 128 (batches 32/32/8), including the strips and the assembly of the operator they need; then reported whatever it says | the strips alone, on the laptop on battery: 0.27 of an exact solve at $n=64$, 0.21 at $n=128$ |
| S11 | the rollout reported, with the level field and the strip columns formed once, with the modes | — |

**E1 on the learned columns.** Let $L$ be the 36 columns the network returns: the four hats, and the sines $j=1$–8 of each side. Then
$$e_{H,L}=\Bigl(\frac{\sum_{k\in L}\lVert\hat H_k-H_k\rVert_{A_I}^2}{\sum_{k\in L}\Lambda_{kk}}\Bigr)^{1/2}=\Bigl(\frac{\sum_{k\in L}(\tilde\Lambda_{kk}-\Lambda_{kk})}{\sum_{k\in L}\Lambda_{kk}}\Bigr)^{1/2},$$
the second form being the diagonal of the Gram identity that SE1 checks. The denominator is the learned columns' own energy, not $\operatorname{tr}\Lambda$:
- On the plain square, the strip columns carry 62–70% of $\operatorname{tr}\Lambda$, at $n=32$–128, steady and with $s_0$ up to 100. This was computed for this draft from `model.plain_square`, and it is not in a record.
- Dividing by $\operatorname{tr}\Lambda$ would therefore lower the measure by a factor of 0.55–0.62, and make the bar 1.6–1.8 times easier: the permissive choice.

**Before training.** K, D, L and E4 as initialised are re-measured on S's form, with `scripts/chartop_forms_controls.py` given the bubble and the strips as options. The same run gives the untrained comparison CW₀ (§8.3).

### 8.3 The comparison

Stage 0's bars apply. The ratio is the alternative arm's over the default's: the alternative wins at $\le0.8$ and loses at $\ge1.25$; otherwise it is a tie, and the default stands. The default is S′, the owner's form, so a tie keeps W358. With two seeds, the pairings are seed 0 with seed 0 and seed 1 with seed 1. A verdict stands only if both pairings give it; otherwise it is reported as unresolved.

| | question | ratio |
|---|---|---|
| **CW** (W358) | do the corner bubble and the strips help? | mean E4 energy error over the six problems, the steps lowered by 300 K: W over S′; median $e_H$ over all 64 columns beside it |

- **CW₀**, the same ratio at step 0 (zero heads), is reported beside CW and not judged. At step 0 only the strips differ, so CW₀ is their classical share. CW against CW₀ is what training made of the rest.
- **The measure is E4's in the lowered frame**, as in the old draft. $e_H$ weights every port mode by its energy, while the coupled answer weights each by the interfaces' own coefficient. The frame keeps the level out of a step's reference energy.

### 8.4 Predictions, registered beside the gates so that they can be refuted

**[AI Inference]**, each checked against the others and against `forms.py`'s docstring:
1. **K, D and L hold on both arms.** The projection comes last and moves only the kernel columns, and the strip columns are not among them. The strips read the operator, never the load. A failure would be a defect in the code.
2. **E1 is probably not met.**
   - Suppose each output were the energy-best member of the 8 × 8 or the 16 × 16 smooth class. Then $e_H$ over all 64 columns would be 0.124 or 0.071 at $n=128$ (0.109 for 8 × 8 at $n=64$; §10), against a bar of 0.01.
   - Those figures include the strips' own error, and the record does not give the learned columns' share. Only the learned columns change with the class, so the 8 × 8 class leaves at least $(0.124^2-0.071^2)^{1/2}\approx0.10$ of $\operatorname{tr}\Lambda$ in them. On their own energy that is roughly 0.17–0.18. The bound is approximate: the figures are medians over eight pieces, and the share is the plain square's. The 16 × 16 class is not bounded this way.
   - $e_p$ has its own bar. B's validation median was 0.089 after 150,000 steps, against a bar of 0.02. S's particular network is also 1 M, and it sees what B saw, without the level.
   - In Stage 0 no arm's modes moved from the principal part. If E1 is met, the corner bubble and the strips made the modes learnable.
3. **CW is at least 1: W does no better than S′.** A tie, or a loss for W, is expected. A win for W ($\le0.8$) would refute this.
   - **Why S′ starts ahead.** At step 0 the arms differ only in the strip columns. On every piece, each strip column is at least as close to the exact mode in energy as the principal part's. It minimises the energy over fields with the same traces that equal $H^{(0)}$ beyond the strip, a set that contains $H^{(0)}$, and the energy error is the energy excess (T25 (ii), column by column).
   - **Why W should stay near its start.** In Stage 0 no arm's modes left the principal part. W's start is a mean of 4.9% on CW's measure (§7.2). The corner bubble bounds the target S′ learns, which should make learning easier for S′.
   - **The size is the uncertain part**, so a tie is as likely as a loss for W. The coupled answer weights the side sines by the interfaces' coefficients, which are small at $j\ge9$ for smooth fields. A basis that is closer mode by mode does not guarantee a closer coupled answer.
   - **Checked against 2.** This prediction needs no learning. If the modes do learn, which would refute 2 in the favourable direction, CW should rise above CW₀.

No verdict is predicted for E4, E5 or S11.

### 8.5 Cost and box limits

**The ceiling: \$10 for the whole registration, each box at most \$0.70 an hour.** Stage 0's ceiling (D6) was \$15, and it spent \$4.06. This draft has 300,000 training steps: 150,000 for S and four runs of 37,500 for S′ and W. Of these, 225,000 run with the strips.

**[AI Inference], an extrapolation until E5 has been priced on the box:**
- **The networks.** B's step took 56.5 ms on the RTX 3090. It was dominated by the float64 objective and the transforms, not by the network: 0.28 M and 3.05 M cost 59 and 75 ms at $n=64$. Two 1 M networks are taken at no more than 1.2 times one, so 300,000 steps take about 5.7 h. At Stage 0's all-in rate (\$4.06 for 525,000 steps, pricing, data and E5 included), that is \$2.3–2.8.
- **The strips as built.** They run serially, on the CPU, in the main process, at every forward pass, training steps included (`KernelExpert._strip_columns`).
  - The inputs: the strips' battery prices (0.27 of an exact solve at $n=64$, 0.21 at $n=128$; $n=32$ taken at $n=64$'s ratio), and Stage 0's labelled examples (0.027, 0.079 and 0.43 s on battery). Stage 0's batches (32/32/8) and step mix give the weights: 19%, 31% and 50% of steps, from its piece coverage of 30/50/20%.
  - The result: about 0.62 s a step on the laptop on battery, against a GPU step of 0.055 s.
  - A box core is taken as one to 4.4 times faster than the laptop on battery. (4.4 is how much slower this laptop has run on battery than on AC.) That gives 0.14–0.62 s a step, plus the operator's assembly, which nothing has priced. Over the 225,000 steps, that is **9–39 h**.

| box (Stage 0's) | rate | box time, with about 2 h for data, pricing and E5 | cost |
|---|---|---|---|
| RTX 3090 with a 64-core EPYC (box A) | \$0.195/h | 16–46 h | \$3.2–9.0 |
| A100 SXM4 (box B) | \$0.532/h | 16–46 h | \$8.7–25 |

- **So the strips, not the networks, are expected to set the cost.** The draft fits the ceiling only on a cheap box with fast cores. S alone would take 9–29 h of wall-clock. Two boxes in parallel would halve the wall-clock, not the cost.
- **The strip columns do not depend on the weights.** $H^{(0)}$ is the plain square's at the piece's $s_0$, and the operator is the piece's own. The projection does not move them, and the label-free objective sums the columns' energies. So they add a constant to the loss and nothing to its gradient. Computing them only at validation would remove their cost from training without changing a gradient. That is a code change this draft does not make, and the owner could choose it after E5.
- **If E5's price puts the estimate over \$10, no long run starts**, and the decision returns to the owner.

**The owner's decision (2026-10-19): accepted as written.** The yes starts four steps, in order, each asking the owner first wherever the rules require it:
1. **Done:** a new gates module carrying these bars (`atlas/chartop/gates_forms.py`), with a test against this section (`tests/test_chartop_gates_forms.py`).
2. **Done:** the untrained re-measurement (§8.6).
3. **Done:** E5 on a rented box, on the owner's OK (§8.7).
4. **Done:** the five runs, trained on two rented boxes and destroyed (§8.9), and evaluated once on the owner's go (§8.10, §8.11). K, D, L, E3, SE1–SE3 and DUAL are met; E1, E4 and KT are not; CW is a tie.

The module was hash-pinned on the owner's OK, on 2026-10-19, before any network in these forms was trained.

### 8.6 The untrained re-measurement (after the acceptance)

`scripts/chartop_forms_controls.py --bubble corner --strips` → `out/chartop/forms-controls-corner-strips.json`, run on the laptop on AC power on 2026-10-19 in 361 s. It measures S's form beside W's, with untrained weights only: zero heads, and seeded random weights at scales 0.002 and 0.02 for each form.
- **Where:** the registered benchmarks at $n=64$, and diagnose's 24 validation pieces $2{,}000{,}100$–$2{,}000{,}123$ (7 steady, 17 transient).
- **Not here:** no trained network, and no test piece. The test split is drawn only at the one evaluation, so the test-piece clauses of K and L wait for it.
- **Not re-run:** the script's default run. Its record, `forms-controls.json`, is unchanged.
- **The verdicts below** are `gates_forms`' own judges, applied to untrained forms. None is a verdict on a trained network.

| gate | W's form (Stage 0's bubble, no strips) | S's form (corner bubble, strips) | bar |
|---|---|---|---|
| **K**, benchmark pieces, every weight set | $e_K\le3.6\times10^{-16}$, $e_{K,\infty}\le4.4\times10^{-16}$ | $e_K\le4.7\times10^{-16}$, $e_{K,\infty}\le4.4\times10^{-16}$; on the validation pieces at zero heads $\le3.4\times10^{-16}$ and $\le3.3\times10^{-16}$ | $10^{-12}$: met |
| **D**, the six problems, every weight set | ratio within $9.6\times10^{-11}$ of one; maxima within $1.5\times10^{-11}$ of the range | within $2.0\times10^{-10}$; maxima within $1.4\times10^{-11}$ | $10^{-8}$ and $10^{-9}$: met, all six problems above the floor |
| **L**, all 126 benchmark pieces, every weight set | bit for bit | bit for bit | met. The pieces' own source is non-zero on the steps only |
| **E4** as initialised | met on bend-3 steady only | met on bend-3 steady only | not met, as §7.2 |

**The coupled answer at step 0** (zero heads), energy errors in per cent; for the steps, as registered and then lowered by 300 K:

| problem | 16-mode floor | W's step 0 | S's step 0 |
|---|---|---|---|
| bend-3, steady | $2.3\times10^{-10}$ | 0.00034 | 0.00034 |
| s-channel, steady | 0.93 | 5.282 | 5.258 |
| insert-round, steady | 0.031 | 5.577 | 5.577 |
| bend-3, step | 0.017 / 0.25 | 0.268 / 3.893 | 0.268 / 3.892 |
| s-channel, step | 0.14 / 2.13 | 0.685 / 10.80 | 0.682 / 10.75 |
| insert-round, step | 0.0066 / 0.070 | 0.393 / 4.138 | 0.393 / 4.138 |
| **CW's measure** (the mean, steps lowered) | — | 4.948 | 4.937 |

**CW₀ = 4.948 / 4.937 = 1.0023**, a tie by Stage 0's bars.

**The modes at step 0.**
- On the benchmark pieces, the strips cut the median $e_H$ over all 64 columns by 35–45% on every problem. Steady: 0.0194, 0.0306 and 0.267 become 0.0110, 0.0172 and 0.146. Steps: 0.0203, 0.0330 and 0.267 become 0.0129, 0.0216 and 0.146.
- On the validation pieces (medians), $e_H$ falls from the principal part's 0.397 to 0.230. $e_{H,L}$ is 0.402 for both, since the learned columns at zero heads are the principal part's.
- The shares of $e_H^2$ at S's step 0: hats 0.4%, sines $j=1$–3 15.2%, $j=4$–8 78.4%, $j=9$–15 3.2%. In Stage 0, $j=9$–15 held 68.5%.

**What it shows.**
- **Prediction 1 holds before training.** K, D and L hold on S's form as on W's, at zero heads and with random weights.
- **The strips cut the modes' error, but barely move the coupled answer.** At step 0, the benchmarks' coupled error sits in the modes the network learns, not in $j=9$–15.
- **[AI Inference]:** prediction 3 stands as registered: CW is at least 1. CW₀ shows that the strips alone give a tie, so a loss for W ($\ge1.25$) needs S′'s learned columns to improve where W's do not. That is the question the corner bubble puts to training. Unless the modes learn, a tie is the likelier outcome.
- **[AI Inference]:** for E1, $e_{H,L}$ starts at 0.402 against a bar of 0.01, so the network must reduce the learned columns' error about 40 times. At step 0, about 94% of $e_H^2$ already sits in the columns it learns (the medians of the shares).
- **Unchanged by this:** the bars, arms and predictions of §8.1–8.5, as accepted.

### 8.7 E5, priced on the box

On the owner's OK ("rent one box at most \$0.70 an hour, preferably an RTX 3090-class box with a fast many-core CPU"):
- **The script:** `scripts/chartop_price.py --forms`, the `price-forms` stage of `scripts/chartop_box_run.sh`.
- **The record:** `out/chartop/price-forms-a061116c3e9e-20261019-201554.json`, with the box's logs in `out/chartop/box/E5-forms/logs/`. Every file's sha256 was checked against the box's list before the box was destroyed.
- **The box:** one RTX 3090 with a Ryzen 9 5950X (32 threads, a CPU quota of 30), at \$0.272 an hour. It was rented from 16:12 to 16:17 local time with a 40-minute self-stop deadline, then destroyed. The credit went from \$2.693 to \$2.671, so the box cost \$0.022.
- **The weights:** untrained random weights, on the same examples as Stage 0's step pricing. Stage 0's arm B is timed in the same run as the reference.

**One training step, in seconds** (the forward pass, the objective, backward, clipping and AdamW, as `train.train` takes it):

| $n$ (batch) | B, Stage 0 (reference) | W | S | of S's: strips / assembly |
|---|---|---|---|---|
| 32 (32) | 0.0203 | 0.0266 | 0.106 | 0.056 / 0.020 |
| 64 (32) | 0.0627 | 0.0760 | 0.314 | 0.153 / 0.047 |
| 128 (8) | 0.0604 | 0.0710 | 0.346 | 0.187 / 0.041 |
| **mean over Stage 0's mix** (19%, 31%, 50% of steps) | 0.0535 | 0.0641 | **0.290** | 0.151 / 0.039 |
| over B | 1 | 1.20 | **5.42** | |

**The forward pass**, per batch call: B 5.0, 11.1 and 9.2 ms at $n=32$, 64 and 128; W 8.9, 17.6 and 14.0 ms; S 88, 254 and 282 ms. Peak GPU memory is at most 1.7 GiB.

**What the pricing found.**
- **The reference reproduces Stage 0.** B's step takes 62.7 and 60.4 ms at $n=64$ and 128, against box A's 62.8 and 60.4 (§6 of [[chart-operator-stage-0]]).
- **W costs 1.2 times B:** the two networks.
- **S costs 5.4 times B.** The strips (0.151 s) and their assembly (0.039 s) are 65% of its mean step. The draft's extrapolation for the strips, 0.14–0.62 s a step, was right at its low end.
- **The projected training time** at these steps: S 12.1 h, S′ 6.0 h (two seeds), W 1.3 h (two seeds), 19.5 h in all.
- **The generator** drew 24.7 examples a second with 29 workers, against box A's 31.7 with 63. The 50,000 training pieces would take about 0.6 h on each box that trains.

**The cost, an extrapolation from these steps.** Validation during training and the labelled validation draws are not priced.

| layout | box time | cost at \$0.272/h | wall-clock |
|---|---|---|---|
| one box, all five runs | about 20.5 h (19.5 training, 0.6 data, ~0.4 setup and downloads) | about \$5.6 | about 20.5 h |
| two boxes: S on one, S′ and W on the other | about 21.5 h (13.1 and 8.4) | about \$5.8 | about 13 h |

- **Within the ceiling of \$10**, and within the draft's range for this box class (\$3.2–9.0, §8.5).
- **The credit, \$2.67, does not cover it.** The account needs a top-up before the runs.
- **[AI Inference]:** §8.5's option would remove 0.19 s of S's 0.29 s step: computing the strip columns only at validation, since they add nothing to the gradient. That is about 12 of S's and S′'s 18 training hours, roughly \$3.2 at this rate, and S's box would finish in about 5 h instead of 13. It is a code change, and the owner's choice. Its claim that the weights are unchanged would first need a test showing them equal bit for bit with the strips on and off in training.

### 8.8 The strips only at validation: built, tested, not yet used

The owner chose to add the option before any run: "First add the option to compute the strip columns only at validation, with a test showing the weights bit for bit equal with the strips on and off in training." No bar, arm or prediction of §8.1–8.5 changes, and `gates_forms.py` is unchanged (its pin test passes).

**What was built.**
- `train.Arm.strips_in_training`, True by default, so the registered path is unchanged. With it False, the training steps call `forms.call(expert, batch, strip_solve=False)`, and `KernelExpert.forward` leaves the 28 strip columns at the plain square's $H^{(0)}$. Neither the strips nor the operator assembly they need runs.
- Validation (`train.validate`), the host's calls (`LearnedExpert`) and the evaluation still solve the strips. The kept checkpoint is therefore chosen exactly as before.
- The logged training loss changes with the option: it then counts the plain square's strip columns instead of the strips.

**Why the weights cannot change.** The strip columns depend on the piece's operator and $H^{(0)}$, never on the weights. The W356 projection reads only the kernel columns, which are not strip columns. The objective sums the columns' energies, normalised by the principal part's. So the strip columns add a constant to each example's loss, and nothing to its gradient.

**The test** (`tests/test_chartop_model.py`, `test_the_strips_only_at_validation_train_the_same_weights`). It trains S's form twice through `train.train`: split, corner bubble, strips, 0.3 M networks, four steps on four training pieces, with two validations on two labelled pieces at $n=32$, on the CPU.
- **The result:** the parameters, the AdamW state and the scheduler state are equal bit for bit, and so are both validations' records.
- **Control 1:** every logged training loss is strictly lower with the strips. Each strip column minimises the energy over a set that contains $H^{(0)}$'s column, so the strips ran in one run and not in the other.
- **Control 2:** a different training seed changes the weights, so the comparison can fail.
- **Control 3:** an expert without strips answers the same with the flag on or off. With strips, the learned columns and the particular field are equal bit for bit, while the strip columns differ.
- The 218 chartop tests pass.

**Not shown:** the GPU. `train.py` does not ask for deterministic GPU kernels, so two GPU runs with the same setting are not guaranteed to agree in the last bits. The equality is shown where the computation is deterministic, and the argument above is what carries it to the GPU.

**The cost with the option**, an extrapolation from §8.7's priced steps until the training box prices it. `chartop_price.py --forms` now times this step too, as the row "S, strips only at validation".
- **The bracket for S's step.** The lower end is W's step, 0.064 s. The upper end is S's priced step without its strips and assembly, 0.100 s, which keeps the transfers that the strip call made. On the laptop's CPU at $n=32$, the skipped step was 0.81 times S's and 0.88 times W's, since S's mode network has a smaller head.
- **The hours:** S 2.7–4.2 h, S′ 1.3–2.1 h, W 1.3 h; training 5.4–7.6 h, against 19.5 h with the strips.
- **Validation** now matters relative to the steps, and stays small: about 10 s each with the strips (from §8.7's forward passes), about 0.2 h over the five runs.

| layout | box time | cost at \$0.272/h | wall-clock |
|---|---|---|---|
| one box, all five runs | about 6.6–8.8 h | about \$1.8–2.4 | 6.6–8.8 h |
| two boxes: S on one, S′ and W on the other | about 7.6–9.8 h (3.8–5.3 and 3.8–4.5) | about \$2.1–2.7 | 3.8–5.3 h |

Either layout is under half the cost with the strips solved in training (\$5.6–5.8). The upper end is close to the \$2.67 credit, so the owner's top-up comes first.

### 8.9 The five runs: trained, not yet evaluated

**The owner's go:** "I have topped up. Run the five registered runs with strips_in_training=False for S and S', on two RTX 3090-class boxes (at most \$0.70/h each): S on one, S' and W on the other. Price the skipped step first on each box and stop if the projection exceeds \$10. Destroy both when done."

**The plan, binding before the first run:** `out/chartop/plan-forms.json` (commit `874c767`).
- It holds S, Sp-s0, Sp-s1, W-s0 and W-s1, with Stage 0's batches (32/32/8), learning rate, weight decay, augmentation and validation cadence.
- `strips_in_training` is false for S and S′.
- Each run has a wall-clock guard of about 1.5–1.7 times its upper projection. None was reached.
- `scripts/chartop_train.py` checked the whole plan against the pinned `gates_forms.ARMS` before anything trained.

**The boxes.**

| box | machine | rate | rented, local time | runs |
|---|---|---|---|---|
| 1 | RTX 3090, EPYC 7532 (a quota of 30 CPUs, 125 GB), vast.ai 56786550, offer 44730076 | \$0.342/h | 16:35–20:19 | S |
| 2 | RTX 3090, EPYC 7352 (a quota of 15 CPUs, 944 GB), vast.ai 56786592, offer 56775475 | \$0.238/h | 16:36–20:11 | Sp-s0, Sp-s1, W-s0, W-s1 |

- **The rates include the 60 GB disk.** The plan named offer 44730076 at the offer's \$0.327 and another offer, 29548834, for box 2.
- **That offer's host refused the contract** (`success: false`). The stopped contract was destroyed at once, and the next suitable offer was taken. The plan's runs, data and gate did not change.
- **Each box was destroyed** once its runs had been downloaded and every file's sha256 matched the box's own list.

**Each box priced itself first, then the gate decided** (`scripts/chartop_budget.py`, `out/chartop/box/forms-{1,2}/logs/budget.log`):
- **Box 1:** S's step with the strips skipped took 0.0764 s and W's 0.0766 s. The whole registration at its prices came to 8.76 h and \$3.03.
- **Box 2:** the whole registration came to 8.50 h and \$2.05.
- Both were within \$10, so both trained.
- The price records are `out/chartop/price-forms-a1bca8838637-20261019-204154.json` (box 1) and `-abed3637b057-20261019-204233.json` (box 2).

**The data.** Each box drew Stage 0's training set from the registered seeds. Each drew 50,000 pieces with 16,523 rejected charts, Stage 0's own count, in 342 s and 685 s. Each also drew the 600 labelled validation pieces. The manifests are in `out/chartop/box/forms-{1,2}/`.

**The runs** (`out/chartop/runs/forms/<run>/`: `arm.json`, `log.jsonl`, `summary.json`). The weights `best.pt` stay local, and their sha256 is in `out/chartop/runs/forms/weights-sha256.txt`. The kept checkpoint is the one with the lowest validation median $e_H$, as in Stage 0. The validation measure is $e_H$ over all 64 columns, not E1's $e_{H,L}$.

| run | steps | hours | ms a step | validation median $e_H$: first (step) → kept (step) | 90th percentile, kept | validation median $e_p$: first → kept → last | $e_K$, largest |
|---|---|---|---|---|---|---|---|
| **S** | 150,000 | 3.47 | 83.2 | 0.1943 (5,000) → **0.1518** (140,000) | 0.234 | 0.301 → 0.073 → 0.071 | $1.1\times10^{-15}$ |
| S′, seed 0 | 37,500 | 0.83 | 79.4 | 0.1947 (2,500) → 0.1934 (20,000) | 0.291 | 0.349 → 0.239 → 0.194 | $1.1\times10^{-15}$ |
| S′, seed 1 | 37,500 | 0.83 | 79.4 | 0.1947 (2,500) → 0.1922 (35,000) | 0.289 | 0.323 → 0.132 → 0.132 | $1.1\times10^{-15}$ |
| W, seed 0 | 37,500 | 0.79 | 75.3 | 0.3372 (2,500) → 0.3365 (30,000) | 0.503 | 0.328 → 0.134 → 0.127 | $7.3\times10^{-16}$ |
| W, seed 1 | 37,500 | 0.79 | 75.9 | 0.3373 (2,500) → 0.3362 (35,000) | 0.503 | 0.336 → 0.123 → 0.123 | $7.2\times10^{-16}$ |

**What the validation shows. None of it is a gate.**
- **The strips' classical share shows from the first validation.** S's form starts at about 0.195 against W's 0.337, Stage 0's flat level.
- **W's modes did not move**, as no Stage 0 arm's did: 0.3372 → 0.3365.
- **S′'s barely moved in 37,500 steps:** 0.1947 → 0.192–0.193.
- **S's did, over 150,000:** 0.1943 → 0.1518, falling still at 145,000, with S's validation median $e_p$ reaching 0.073. Stage 0's arm B ended at 0.3366 and 0.089.
- **The kernel-exact form held throughout training:** $e_K\le1.1\times10^{-15}$ at every validation of every run.
- **[AI Inference]:** S′ and S share the form and differ in budget, so the modes' learning appears only late in a long schedule. With its learning rate decaying over 150,000 steps, S had about four times as many steps at a high rate as S′.
- **[AI Inference], against the predictions of §8.4:** E1's bar is 0.01 on $e_{H,L}$, and S's validation $e_H$ over all 64 columns is 0.15. Prediction 2 (E1 probably not met) is not contradicted by anything seen here.
- **Not yet known:** CW, which is judged on E4's coupled answer, not on $e_H$. S′'s and W's step-0 coupled answers were a tie (CW₀ = 1.0023), and their validation $e_H$ differs by the strips' classical share. How that carries to the trained coupled answer is the one evaluation's to say.

**What was not done.** The one evaluation, which judges E1, E3, E4, SE1–SE3, DUAL, KT, S11 and CW on the held-out test split and the benchmarks, has not been run, and the test split has not been drawn. Its script for `gates_forms` is not written. Both need the owner's go.

**The cost.** The two boxes cost about \$2.13 by their rates and rented hours. The credit went from \$12.671 to \$10.528, \$2.14 for the two boxes, so the registration has spent \$2.17 in all, E5's box included, against its ceiling of \$10. The instance list was empty after each destroy and once the credit had settled.

### 8.10 The one evaluation's script: written and tested, not run

**The owner's instruction:** "Write the one evaluation's script for gates_forms (E1 on e_H,L, E3, E4 in both frames, SE1-SE3, DUAL, KT, S11, CW with both pairings), tested on small cases. Do not draw the test split or evaluate the weights until I approve." Neither has been done. `out/chartop/data/` does not exist, and there is no evaluation record.

**What `scripts/chartop_evaluate_forms.py` does**, every bar from `gates_forms`:
1. **It checks the weights before anything runs.** Each run's `best.pt` must have the sha256 recorded at download (`weights-sha256.txt`) and its registered arm's configuration (form, networks, ports, bubble, strips).
2. **It opens the registry.** That is Stage 0's (`chartop_registry.py`), so these weights are evaluated once. An interrupted evaluation resumes from its checkpoints, and a finished one is refused.
3. **The test split** is Stage 0's own: the same registered seeds and both bases' labels. It is drawn only if it is not already whole on disk, and its sha256 is taken.
4. **E1 for S** uses $e_{H,L}$ on the 36 learned columns against their own energy, each $\Lambda_{kk}$ being the energy of the exact mode $k$. The script records the largest gap between their sum and the labels' $\operatorname{tr}\Lambda$. Beside E1 go $e_H$ over all 64 columns, and K on every test piece. The network runs in float32 on the CPU, and every call solves the strips.
5. **L for S** on the first 100 test pieces.
6. **The six benchmark problems**, each as registered and lowered by 300 K, for all five runs:
   - E4 in both frames, D, and K and L on every piece;
   - $e_H$ and $e_{H,L}$ against the exact modes on every piece;
   - for S, E3, SE1, SE2 and SE3, and DUAL on the steady pieces.
7. **CW:** W over S′, seed with seed, on `cw_measure`. Beside it go each run's median $e_H$ over the benchmark pieces, and CW₀ (1.0023, §8.6).
8. **DUAL** for S on 100 steady test pieces, **KT** for S, and **S11**, S's rollout. These are Stage 0's functions.
9. **E5** is named by its price records, not re-priced.
10. **The record** is `out/chartop/evaluation-forms-<time>.json` and `.txt`, written whatever it says.

**Two readings this script takes.**
- **The comparison arms run on the benchmarks only.** `gates_forms`' docstring says so. So CW's side-by-side median $e_H$ is taken over the benchmark pieces; Stage 0 took its comparison arms' medians on the test split.
- **K, D and L are judged for all five runs**, and K and L for S on the test pieces too, because the bars say "both arms".

**The tests** (`tests/test_chartop_evaluate_forms.py`). They use throwaway weights and a stand-in test split drawn from validation seeds, with the registered test base patched for the test, so no registered test piece is drawn.
- **E1's rows** agree with an independent numpy computation from each piece's exact modes, to $2\times10^{-3}$ relative, the gap being the stored arrays' float32. $e_K$ is below $10^{-12}$.
- **One benchmark problem in both frames** (bend-3 at $n=32$, steady and step): the kernel-exact split forms meet K, D, L and SE2 whatever the weights, E3, SE1 and SE3 are judged for the main arm only, and the steady problem carries the dual bracket.
- **End to end:** the plan's five runs pass the weights' checks, E1 runs for S alone, and CW pairs W-s0 with Sp-s0 and W-s1 with Sp-s1. The gates are assembled and the record written. The same weights are refused a second time, with or without `--resume`, as are a tampered file and a configuration that is not its arm's.
- **Two mutations are caught.** E1 against $\operatorname{tr}\Lambda$ and CW the wrong way round each make their test fail, and the script was restored byte for byte.
- The 223 chartop tests pass.

**The run, when the owner approves.** On the laptop on AC power: `python scripts/chartop_evaluate_forms.py`. It first draws the 2,400 labelled test pieces (Stage 0 drew them; they are not on disk now), then evaluates. **[AI Inference]:** about an hour, by Stage 0's per-piece costs and the strips' prices; not measured. It took 20 minutes (§8.11).

### 8.11 The one evaluation

**The owner's go:** "run the one evaluation now on the laptop on AC power: draw the test split, then python scripts/chartop_evaluate_forms.py, and report every verdict whatever it says."
- **The run:** on the laptop, 20:59–21:19 on 2026-10-19, once.
- **The power:** the AC line read online before it started. It went offline briefly at about 21:10, and was back by 21:10:31. Nothing was interrupted.
- **The test split** was drawn first: Stage 0's 2,400 registered seeds, both bases labelled, 841 rejected charts, in 225 s. Its content sha256 is `3fcd59fa…`.
- **The records:** `out/chartop/evaluation-forms-20261019-205923.json` and `.txt`, with the test rows in `-test-rows.npz`, the console log, the registry entries in `evaluated.json`, and the provenance in `-checkpoints/state.json`.

**The verdicts**, every one whatever it says:

| gate | verdict | what decided it |
|---|---|---|
| **K** | **met** | every run and piece: $e_K\le1.1\times10^{-15}$, $e_{K,\infty}\le4.4\times10^{-16}$ (2,400 test pieces for S, and every benchmark piece of every run) |
| **D** | **met** | every run and problem: the ratio within $3.4\times10^{-12}$ of one, the maxima within $1.4\times10^{-11}$ of the range |
| **L** | **met** | bit for bit on every benchmark piece of every run, and on S's first 100 test pieces |
| **E1** | **not met** | not met in any of the 12 bins (median $e_{H,L}$ 0.110–0.351, against 0.01; median $e_p$ 0.050–0.082, against 0.02) |
| **E3** | met | all six problems: the coupled error within Theorem 2's bound |
| **E4** | **not met** | not met on any of the six problems, in either frame (table below) |
| **SE1** | met | all six problems |
| **SE2** | met | one call per piece per solve, both frames |
| **SE3** | met | all six problems |
| **DUAL** | met | the worst smallest eigenvalue is $-7.4\times10^{-15}$ relative, over 63 benchmark pieces and 100 steady test pieces |
| **KT** | **not met** | each of the three steady benchmarks converged in 5 outer iterations, like the classical solution, but erred by 3.9%, 6.6% and 7.9% in energy |
| S11 | reported | below |
| **CW** | **tie**, both pairings | seed 0: 1.040; seed 1: 0.990 |
| E5 | reported | priced before training (§8.7, §8.9), not re-run |

**E1, beside the gate.** Over all 2,400 test pieces:
- $e_{H,L}$ has median 0.267 and 90th percentile 0.420;
- $e_H$ over all 64 columns has median 0.152 and 90th percentile 0.244;
- $e_p$ has median 0.066 and 90th percentile 0.132.

$e_{H,L}$ grows with $\sup\lvert\mu\rvert$: about 0.11–0.14 in the first bin, 0.33–0.35 in the last. By resolution it is 0.250, 0.268 and 0.294 at $n=32$, 64 and 128. The sum of the exact modes' energies matches the labels' $\operatorname{tr}\Lambda$ to $1.1\times10^{-13}$.

**E4, arm S**, as registered and lowered by 300 K. The 16-mode floors are as in §8.6.

| problem | energy error, % | maximum error, % of the range | the maximum's bar | E4 |
|---|---|---|---|---|
| bend-3, steady | 3.94 / 3.94 | 0.22 | 1.00 | not met (energy) |
| bend-3, step | 0.21 / 3.08 | 1.66 | 1.42 | not met (maximum; energy lowered) |
| s-channel, steady | 6.56 / 6.56 | 0.35 | 1.15 | not met (energy) |
| s-channel, step | 0.69 / 10.81 | 5.08 | 3.81 | not met (maximum; energy lowered) |
| insert-round, steady | 7.84 / 7.84 | 1.08 | 1.005 | not met (both) |
| insert-round, step | 0.36 / 3.80 | 1.89 | 1.05 | not met (maximum; energy lowered) |

No run met E4 on every problem. Their energy errors in per cent, steady and then the steps lowered:

| run | bend-3 | s-channel | insert-round | bend-3 step | s-channel step | insert-round step | CW's measure |
|---|---|---|---|---|---|---|---|
| S | 3.94 | 6.56 | 7.84 | 3.08 | 10.81 | 3.80 | 6.01 |
| S′, seed 0 | 1.29 | 5.49 | 4.37 | 4.16 | 7.26 | 3.12 | 4.28 |
| S′, seed 1 | 2.65 | 5.91 | 7.88 | 1.91 | 6.19 | 2.74 | 4.55 |
| W, seed 0 | 1.39 | 5.21 | 6.84 | 2.73 | 8.12 | 2.43 | 4.45 |
| W, seed 1 | 2.01 | 5.83 | 6.61 | 2.34 | 8.02 | 2.21 | 4.50 |
| as initialised (§8.6) | 0.00034 | 5.28 / 5.26 | 5.58 | 3.89 | 10.80 / 10.75 | 4.14 | 4.95 / 4.94 |

The last row gives W's form, then S's form, where they differ.

**CW** = W over S′ = 4.45 / 4.28 = 1.040 for seed 0, and 4.50 / 4.55 = 0.990 for seed 1. Both are ties, so S′, the owner's form, stands.
- **Beside it, the median $e_H$ over the benchmark pieces:** S 0.077, S′ 0.077 (both seeds), W 0.137 (both seeds).
- **CW₀** was 1.0023 (§8.6).

**S11, the rollout.** The s-channel was marched 20 steps, with the modes formed once: 180 network calls over 9 pieces, one per piece per step. S's energy error is 0.69%, 0.44%, 0.33% and 0.25% at steps 1, 5, 10 and 20. Its maximum is 5.08%, 3.63%, 2.78% and 2.14% of the range, against the 16-mode floor's 2.81%, 1.19%, 0.84% and 0.61%. E4's bars are not met at any of them.

**The predictions of §8.4, against the result.**
1. **K, D and L hold: held**, for every run, and with the strips solved in every call.
2. **E1 is probably not met: not met**, in every bin. The learned columns' error, 0.267 at the median, is about 27 times the bar. §8.4 bounded the 8 × 8 smooth class at roughly 0.17–0.18 at $n=128$; the trained S reaches 0.294 there.
3. **CW is at least 1, a tie or a loss for W: a tie in both pairings**, so this is not refuted. Its stronger wording, "at least 1", holds for seed 0 (1.040) and fails for seed 1 (0.990, W 1% better). The predicted refutation, a win for W at 0.8 or below, did not happen.

**What the evaluation shows.**
- **The structure holds after training.** K, D, L, E3, SE1–SE3 and DUAL are all met. The kernel-exact split form with the corner bubble and the strips keeps the uniform state, the datum and the load out of its errors, bit for bit or to rounding.
- **The accuracy does not.** E1, E4 and KT are not met.
- **Neither the strips nor the bubble changes the coupled answer measurably:** CW is a tie. The strips nearly halve the benchmarks' $e_H$ (0.137 against 0.077), as they did untrained.
- **[AI Inference]:** S trained four times as long as S′, and its validation $e_H$ fell from 0.194 to 0.152 (§8.9). Yet on all three steady benchmarks its coupled answer is worse than the untrained form's, with energy errors of 3.9%, 6.6% and 7.8% against 0.00034%, 5.3% and 5.6%. On bend-3 and the s-channel it is also worse than every other run's, and on insert-round than all but S′ seed 1's (7.88%).
  - The learned corrections lower the energy-weighted mode error that training and validation measure, but move the modes in directions that the benchmarks' interface data weight heavily.
  - This is §8.3's point that $e_H$ and the coupled answer weight the modes differently, now seen with a trained network.
- **[AI Inference]:** CW₀, the trained CW and the benchmarks' $e_H$ agree that the side sines $j=9$–15 are not what limits the coupled answer here. The limit is in the learned low modes, and in a training objective that does not see the coupled answer.

**Not done here.** No comparison arm was evaluated on the test split (§8.10). Nothing is on the website or in `proposal/`.

---

## 9. What this page does not show

- **No trained network in either new form.** Every number in §7 is from zero heads or random weights, and none is a gate.
- **Arm B was not passed through the new forms**, so that the next registration's bars are drafted before any trained number in them has been seen. B's weights ran only in their own form, as a check of the measure (§7.3).
- **B on the steps lowered by 300 K** was not run, so B's step errors in that frame are not known.
- **Not priced**: the split form's forward pass and training step.
- **W358 is not in these forms as measured here**: the side modes ask what Stage 0's did (§7.4). §10 measures the fixes, untrained.
- **The random-weight controls cannot reproduce a learned dependence on the load** (§7.2). The guarantee is structural, and its test is bitwise.
- **The cosine basis** is covered by the forms and their tests, but no benchmark ran with it.
- **One resolution on the benchmarks**, $n=64$; the diagnostics ran at 32, 64 and 128.
- **Nothing on the website or in `proposal/`.**

---

## 10. W358: a bubble that follows the corners, and the high sines by strips (checked untrained)

**Intuition.** A conformal chart keeps angles, so it can carry the reference square's right-angled corners only onto right-angled corners. A piece whose physical corner is not a right angle needs a chart that shears there ($\operatorname{Im}\mu\neq0$). Near such a corner the exact modes behave like the physical corner's own harmonic functions, which vanish like $r^{\pi/\alpha}$ at a corner of angle $\alpha$. Stage 0's bubble vanishes like $r^2$ there. At an obtuse corner the quotient the head must output therefore behaves like $r^{\pi/\alpha-2}$: unbounded, and the grid sees more of it at every refinement. A bubble that vanishes only like the distance to the boundary, at the corners as along the sides, leaves a bounded quotient at every convex corner, whatever its angle.

**Rigorously.** Freeze the pulled-back tensor $A$ at a corner and change variables by $y=A^{-1/2}x$. The operator becomes the Laplacian and the reference quarter-plane a wedge of angle $\omega$, with

$$\cos\omega=\frac{-a_{12}}{\sqrt{a_{11}a_{22}}}=\frac{2\operatorname{Im}\mu}{\lvert1-\mu\rvert\,\lvert1+\mu\rvert}$$

in the corner's own frame. $\omega$ is the physical corner angle: on seed 2,000,101 the corner elements give 55.7°, 109.1°, 97.8° and 74.2° against the generator's 54.9°, 109.0°, 98.0° and 73.4°. A real $\mu$, the stretched conformal maps, leaves $\omega=\pi/2$. Near the corner the exact mode's difference from the plain square's vanishes on both sides and behaves like $c\,r^{\pi/\omega}$ (Grisvard, *Elliptic Problems in Nonsmooth Domains*, 1985). At a right angle with varying coefficients it behaves like $r^2\log r$. Divided by $\beta\sim16\,\xi\eta$, the peak at the node diagonally next to the corner grows like $n^{2-\pi/\omega}$ once $\omega>\pi/2$. The **corner bubble** is

$$\beta_c=\frac{4\,\xi(1-\xi)\,\eta(1-\eta)}{r_0\,r_1\,r_2\,r_3},$$

with $r_c$ the distance to corner $c$. It is one at the centre and zero on the loop. On every interior node, at every resolution, it lies between 2 and 3.28 times the distance to the boundary; near a corner it is $\xi\eta/r$ up to a factor. Every convex corner has $\pi/\omega>1$, so $(H-H^{(0)})/\beta_c$ stays bounded with no angle to estimate. It is one fixed field for every piece, invariant under the square's eight symmetries bit for bit, so the traces, the projection of §3 and D4 augmentation are untouched. Bubbles built from distance functions are the standard way to impose Dirichlet data exactly in physics-informed networks (Sukumar & Srivastava, *CMAME* 389, 2022). The product of the four sides' distances vanishes to second order at the corners, which is Stage 0's bubble.

**The controls.** `scripts/chartop_corner_controls.py` → `out/chartop/corner-controls.json`, run on the laptop on 2026-10-19 in 657 s, on battery at 1.4 GHz. It uses diagnose's validation pieces $2{,}000{,}100$–$2{,}000{,}107$ at $n=32$, 64, 128 and 256, with classical solves only: no network and no training. Per candidate it measures:
- **The target**: the quotient $N^\star=(H-P)/\beta$ the head must output for the form to return the exact modes, over the columns it learns ($P$ the candidate's classical part). Peak and RMS, medians over the eight pieces. Sweeps do not change the target.
- **$e_H$ if the network learned only what is smooth**: each output is the energy-best member of $\beta\cdot\operatorname{span}\{\cos p\pi\xi\cos q\pi\eta:\ p,q<K\}$, which is the minimiser the label-free objective would reach in that class (T26). It then goes through the sweeps and the projection of §3, so the uniform state stays a fixed point. $K=8$, and $K=16$ at $n=128$.
- **The cost per piece**: the classical work the candidate adds to the network's call, over the exact classical solve of the same piece (one sparse LU, 64 mode solves and the level field). Matrix assembly is excluded throughout, and the ratios come from one run.

The candidates:
- **C1**: the piece's own tensor and reaction frozen at each corner element, solved exactly on the whole square, the four blended by a smooth partition of unity that is one near its corner.
- **C2**: damped Jacobi sweeps (weight 2/3) on the true operator, after the output and before the projection.
- **C3**: the side sines $j=9$–15 by a strip solve: the true operator on the nodes within $1/8$ of their side, with the plain square's modes as data beyond. The network learns the other 36 columns only.
- **C4**: the angle bubble ($r^{\min(\pi/\omega,2)}$ at each corner, $\omega$ from the corner element) and the corner bubble $\beta_c$.

**Positive control:** Stage 0's bubble reproduces diagnose's record (16.42, 50.24, 89.24) to $10^{-13}$.

| candidate | peak of $N^\star$: $n=32$ / 64 / 128 / 256 | RMS ($n=128$) | $e_H$, $N=0$ ($n=128$) | $e_H$, smooth $N$, $K=8$: 32 / 64 / 128 / 256 | $K=16$ ($n=128$) | cost / exact ($n=128$) |
|---|---|---|---|---|---|---|
| Stage 0's bubble | 16.4 / 50.2 / 89.2 / 169.6 | 0.971 | 0.385 | 0.265 / 0.302 / 0.335 / 0.363 | 0.187 | 0.20 |
| C1 frozen principal part | 4.5 / 9.7 / 10.2 / 14.4 | 0.164 | 0.149 | 0.108 / 0.128 / 0.123 / 0.125 | 0.079 | 3.87 |
| C2 sweeps ×4 | as Stage 0 | 0.971 | 0.326 | 0.051 / 0.159 / 0.270 / 0.341 | 0.121 | 0.40 |
| C2 sweeps ×16 | as Stage 0 | 0.971 | 0.242 | 0.004 / 0.052 / 0.174 / 0.293 | 0.049 | 1.12 |
| C3 strips, $j=9$–15 | 13.4 / 26.7 / 39.4 / 84.6 | 0.624 | 0.219 | 0.099 / 0.109 / 0.121 / 0.134 | 0.072 | 0.41 |
| C4 angle bubble | 11.7 / 28.0 / 58.9 / 84.1 | 0.736 | 0.385 | 0.264 / 0.301 / 0.334 / 0.363 | 0.184 | 0.21 |
| **C4 corner bubble** | 4.2 / 9.1 / 14.4 / 19.0 | 0.396 | 0.385 | 0.263 / 0.300 / 0.333 / 0.362 | 0.187 | 0.20 |
| C4 + C1 | 2.8 / 5.4 / 8.6 / 12.2 | 0.143 | 0.149 | 0.107 / 0.128 / 0.123 / 0.126 | 0.081 | 3.87 |
| C4 + C2 ×16 | 4.2 / 9.1 / 14.4 / 19.0 | 0.396 | 0.242 | 0.004 / 0.051 / 0.173 / 0.291 | 0.049 | 1.12 |
| **C4 + C3** | 4.2 / 6.8 / 9.4 / 10.5 | 0.316 | 0.219 | 0.099 / 0.109 / 0.124 / 0.136 | 0.071 | 0.41 |
| C4 + C3 + C2 ×4 | 4.2 / 6.8 / 9.4 / 10.5 | 0.316 | 0.205 | 0.029 / 0.069 / 0.106 / 0.130 | 0.054 | 0.61 |

**What the table shows.**
- **Stage 0's peak keeps growing**: 1.9 times from $n=128$ to 256, and on one piece 7.4 times (56.9 to 418.5).
- **The corner bubble divides the peak by 4–9**, and the level field's by 2–3.4 (0.70, 0.73, 0.75, 1.07 against 1.37, 1.90, 2.44, 3.65). Its growth per doubling falls from 2.2 to 1.6 to 1.3. It leaves the smooth $e_H$ as it was (0.333 against 0.335): it changes what the head must say near the corners, not how much of the answer is smooth.
- **The angle bubble does much worse** (58.9 at $n=128$). It needs an angle from one element, and it leaves the right-angled corners' $r^2\log r$ at full strength.
- **C1 is the best classical part**: $e_H$ 0.149 with no network, against 0.385. But it costs 3.9 exact solves, and its peak still grows (10.2 to 14.4). It has the singular exponent right, but not the coefficient, which depends on the whole piece.
- **C2 fades with the resolution.** Sixteen sweeps leave a smooth $e_H$ of 0.004 at $n=32$ and 0.29 at $n=256$, because a sweep's reach is fixed in cells while the high sines' layers are fixed in the reference square.
- **C3 removes the high sines' error at every resolution**, at a fifth of the exact solve on top of the principal part. With the corner bubble it is **the only candidate whose peak levels off**: 9.4 to 10.5 from $n=128$ to 256, 1.12 times. Per piece the growth is 1.06–1.32, the largest on 2,000,106, whose material interface meets a side.
- **The pass condition, a peak that stops growing over $n=32$–128, is met by no candidate.** C4 + C3 grows 1.62 and then 1.38 times per doubling there, and 1.12 after that.

**Where the remaining growth comes from** (one-off probes on three of the pieces out to $n=512$, not in the record):
- **The corner singularity.** At obtuse and near-right corners, Stage 0's quotient grows at every resolution in every mode group. On a 120° parallelogram (a test, below) Stage 0's quotient grows fivefold from $n=16$ to 64, and the corner bubble's by under a half, with shrinking increments.
- **The modes' own layers.** Under the corner bubble the peak moves to the first interior row along a side. There it climbs, with shrinking increments, while the layers of the higher sines (width about $1/(j\pi)$) are resolved. The $j=15$ mode of 2,000,101 gives 18.6, 25.5 and 30.3 at $n=128$, 256 and 512, up by 6.9 and then 4.9, and its first-row profile converges. C3 takes the worst of these away.
- **A material interface meeting a side.** This is a singular point of the exact solution in its own right, and the form leaves it alone. On 2,000,106 the peak sits on the first column, beside the point where a jump of 0.94 in $\log\hat\kappa$ meets the side $\xi=0$. Its sines $j=1$–3 level off there (8.8, 10.4, 10.5 at $n=128$, 256, 512) while $j=9$–15 still climb.

**Recommendation.** Register **C4 + C3**: the corner bubble on every output (built and tested, below), with the side sines $j=9$–15 computed by strip solves. It is the only combination whose target levels off. It leaves a smooth $e_H$ of 0.124 ($K=8$) or 0.071 ($K=16$) at $n=128$, against Stage 0's 0.335 and 0.187, for 0.41 of the exact solve's CPU time. Four sweeps on top (C2 ×4) are optional: they lower the smooth $e_H$ to 0.106 for another 0.2 of the exact solve. Not C1, which costs more than the solve it would replace, and not C2 alone, which fades with $n$. **The strips are now built into the expert** (commit 6594c8a, untrained, off by default): the 36 learned columns come from the network, the 28 strip columns from strip solves on the piece's operator (passed in the call, never a network input), and the W356 projection stays last; they are priced at 0.27 of an exact solve at $n=64$ and 0.21 at $n=128$ (on battery). The owner chose, in a side session, the output form "Bubble + strips (Recommended)" and a particular-field network of "1 M (as drafted)"; no registration has been chosen. **[AI Inference]:** neither smooth class reaches E1's bar of 0.01. The error that remains is the low modes' anisotropy, which is what the network is for, so E1 now measures the network rather than the form. Whether a trained U-Net beats a 16 × 16 cosine class is a training question. The strips' fifth of an exact solve also counts against E5, whose network cost has not been priced.

**Built** (untrained; nothing registered or hash-pinned; `gates.py` unchanged):
- `forms.corner_bubble`, and `KernelExpert(..., bubble="corner")` for both forms. `"square"`, Stage 0's bubble, stays the default and answers bit for bit as before. `train.Arm.bubble` and `train.load_expert` carry the choice. A configuration written before W358 loads as `"square"`, and Stage 0's own forms refuse `"corner"`.
- Five tests in `tests/test_chartop_model.py`, with random weights and classical solves only:
  - the bubble's bounds against the distance to the boundary at $n=16$, 64 and 256, and its bitwise symmetry;
  - the 120° parallelogram above;
  - both forms kernel-exact under the corner bubble at weight scale 0.3, with the side modes equal to $H^{(0)}+\beta_cN$ bit for bit;
  - the square bubble unchanged;
  - training, saving and loading.

**Not shown here:** a trained network in either bubble; the particular field's quotient; any resolution above 256 in the record; any treatment of the interface meeting a side.

---

## 11. After the one evaluation: where the coupled error is, three options, a draft (post hoc, 2026-11-07)

**Nothing here is a gate.** The test split was not drawn or read again. Everything was measured post hoc on the steady benchmarks and on validation seeds from 2,000,100 onward, as Stage 0's §9.7 did. The weights were checked against `weights-sha256.txt` before they were read. U is S's form (split, corner bubble, strips) with zero heads. Scripts: `scripts/chartop_forms_diagnose.py` (parts `a1`, `a2`, `a3`, `b1`, `i13`) and `scripts/chartop_overfit_probe.py`. Records: `out/chartop/forms-diagnose-<part>.json`, and for the overfit probe `out/chartop/forms-overfit-<arm>.json` (500 steps) and `forms-overfit-<arm>-3000.json` (3,000 steps).

**Intuition.** The coupled answer asks a piece's expert for two things above all: how the piece's temperature tilts between its corners (the hats' corner differences) and the three lowest sines on each side. Both are global responses of the whole piece. Three things work against them:
- the objective weighted all 64 columns alike;
- at $n=128$ the U-Net sees less than half of the piece;
- the quotient it must output keeps growing with $n$, fastest where a material interface meets a side.

### 11.1 Where the coupled error is (A1)

**Rigorously.** Per piece, Theorem 2's term is $w_i=e_{p,i}+E_i\,c^*_i$ with $E_i=\tilde H_i-H_i$ and $c^*_i=R_ic^*$, the classical coupled coefficients. Split $E_ic^*_i$ by columns: the hats' corner-difference part $E_{\mathrm{hats}}(c^*_{\mathrm{hats}}-\bar c\,\mathbf 1)$, their level part $\bar c\,E_{\mathrm{hats}}\mathbf 1$, and the side sines $j=1$–3, 4–8 and 9–15 (the strips, on and off their strip nodes). Then $w_i=\sum_g w_{i,g}$ exactly. Each group's share is

$$s_g=\frac{\sum_i\kappa_i\langle w_{i,g},\,w_i\rangle_{A_I}}{\sum_i\kappa_i\lVert w_i\rVert^2_{A_I}},\qquad \sum_g s_g=1,$$

which can be negative when a group cancels another.

| steady problem | run | energy error | hats, corner differences | $j=1$–3 | $j=4$–8 | $j=9$–15 | hats, level |
|---|---|---|---|---|---|---|---|
| bend-3 | U | 0.0003% | 1.000 | 0.000 | 0.000 | 0.000 | $8\times10^{-10}$ |
| bend-3 | S | 3.94% | 1.000 | 0.000 | 0.000 | 0.000 | $8\times10^{-13}$ |
| s-channel | U | 5.26% | 0.006 | **0.929** | 0.071 | −0.006 | $2\times10^{-12}$ |
| s-channel | S | 6.56% | **0.449** | **0.528** | 0.026 | −0.003 | $3\times10^{-13}$ |
| insert-round | U | 5.58% | **1.159** | −0.163 | 0.004 | 0.000 | $-9\times10^{-13}$ |
| insert-round | S | 7.84% | **1.030** | −0.030 | 0.000 | 0.000 | $-1\times10^{-12}$ |

**Controls.**
- The groups' vector sum reproduces `network_term` to at most $6\times10^{-16}$ relative.
- S's energy errors reproduce the evaluation's (3.9379, 6.559, 7.8385%).
- The steady benchmarks have no source (largest $\lvert g\rvert=0$), so $e_p=0$ exactly.
- Re-solving U's strips at width 1/8 gives U bit for bit. At widths 1/4 and 1/2 the s-channel's error goes 5.257 → 5.276 → 5.284%, insert-round's stays 5.577% and bend-3's 0.000%.

**Verdict (issue 6).**
- **s-channel:** U's 5% is in the low side sines $j=1$–3 (share 0.93; that group's own norm is 0.98 of the term). U takes these columns from the plain square's $H^{(0)}$ unchanged, so the principal part is the cause.
- **The strips are not the cause:** their share is under 1% on every problem, and widening them from 1/8 to 1/2 changes nothing.
- **insert-round:** U's error *is* in the hats' corner differences (share 1.16, with $j=1$–3 cancelling 0.16). These are the plain square's hats, which do not know the round insert. "It isn't in the hats" is refuted there.

### 11.2 The reach (A2)

From `model.py`, a 3 × 3 convolution at level $k$ adds $2^k$ nodes of reach, a stride-2 average adds $2^k$ going down to level $k+1$, and an aligned bilinear upsampling adds at most $2^k$ coming back to level $k$:

$$R=(2+1+4+2+8+4+16)+(4+8+2+4+1+2)=58\ \text{nodes}.$$

The support of one output node's gradient (random weights, $n=256$, output nodes at each residue mod 8) measures 53–58 nodes, depending on the node's position. The effective field below is the radius, in nodes (Chebyshev), that holds 90% and 99% of $\sum_c\sum_{\mathrm{ch}}\lvert\partial y_c/\partial x_{\mathrm{ch}}\rvert$ for S's trained mode network. It was measured on 4 validation pieces, at the centre node and at the node next to the middle of side 0.

| $n$ | $R/n$ | a side node sees the opposite side | centre: 90%, 99% (of $n$) | side node: 90%, 99% |
|---|---|---|---|---|
| 32 | 1.81 | yes | 0.47, 0.50 | 0.66, 0.95 |
| 64 | 0.91 | no | 0.47, 0.50 | 0.45, 0.64 |
| 128 | 0.45 | no | 0.32, 0.40 | 0.25, 0.34 |

**Verdict (issue 2): confirmed at $n=128$, and refuted as the cause of E1's level.**
- At $n=128$ a node next to a side holds 99% of its gradient within a third of the piece, and cannot see the opposite side at all.
- At $n=32$ the field covers the piece 1.8 times over, yet E1's median there was 0.250.
- So the reach can explain the rise with $n$ (0.250 → 0.268 → 0.294), but not the level, 25 times the bar.

### 11.3 S with its hats replaced (A3)

S's responses were re-coupled with chosen columns replaced, steady, in both frames. E4 is judged by `gates_forms.judge_e4_both_frames`. "Recovered" is $(e_S-e)/(e_S-e_{\mathrm{floor}})$.

| variant | bend-3 | s-channel | insert-round |
|---|---|---|---|
| S | 3.9379% ✗ | 6.5590% ✗ | 7.8385% ✗ |
| S + exact hats | 0.0000% ✓ (100%) | 4.4401% ✗ (37.6%) | 2.2978% ✗ (71.0%) |
| S + U's projected hats | 0.0003% ✓ (100%) | 4.5136% ✗ (36.3%) | 6.3436% ✗ (19.1%) |
| S + exact hats and $j=1$–3 | 0.0000% ✓ | 1.5350% ✓ (89.2%) | 0.1950% ✓ (97.9%) |
| S + every learned column exact | 0.0000% ✓ | 0.9289% ✓ | 0.0409% ✓ |
| U | 0.0003% ✓ | 5.2575% ✗ | 5.5767% ✗ |
| U + exact hats | 0.0000% ✓ | 5.1717% ✗ | 2.9127% ✗ |
| classical floor | $2.3\times10^{-10}$% | 0.925% | 0.031% |

**Verdict.**
- A hat fix recovers all of bend-3, whichever hats are used.
- It recovers 36–38% of the s-channel, and on insert-round 19% (U's hats) or 71% (exact hats). It meets E4 on bend-3 only.
- Exact hats together with exact $j=1$–3 meet E4 on all three problems.
- **The fix must reach the low side sines as well as the hats.**

### 11.4 Three options, built untrained (B1–B3)

Each is off by default. Each is tested in `tests/test_chartop_forms_options.py`: 34 tests, random weights, and two-step training runs.

**Bit for bit when off.**
- The 223 earlier chartop tests pass unchanged.
- A3 was re-run after all three options were built: every number is identical to the record written before them.

**B1, `KernelExpert(..., hats=)`** (`"learned"` by default).
- **The options.** `"principal"` takes the kernel columns from the plain square's $H^{(0)}$, with no learned correction. `"classical"` solves them exactly, $A_IH_k=-M_{IG}Q_k$: one sparse LU and four right-hand sides, from the `ops` the strips already read.
- **What the network still does.** Its hat rows keep their shape but are not read, so S's weights load. The projection comes last, as always. Steady, the classical hats stay exact. In a step the projection moves them by $s_0W(u_1-u_p(1))$, the level field's own error.
- **Tests:** the default unchanged; K to rounding with either option; the head's hat rows unread; L bit for bit; D through the host on bend-3 (lowering by 300 K lowers the answer by 300 K to $10^{-9}$ of the shift); the refusals; save and load.
- **The price** is shown as a fraction of the exact classical solve (`fe.Exact`: the LU, 64 mode columns and the particular field). Each is the median of 4 validation pieces, best of 3, with torch on 4 threads. The exact solve itself took 7, 49 and 209 ms.

| $n$ | classical hats (4 columns) | hats and $j=1$–3 (16) | strips (28) | S's forward, no strips |
|---|---|---|---|---|
| 32 | 0.51 | 0.60 | 0.44 | 3.97 |
| 64 | 0.44 | 0.47 | 0.26 | 1.19 |
| 128 | 0.46 | 0.57 | 0.21 | 0.57 |

- **Verdict (issues 4, 5, 7): B1 is a bound, not an expert.**
  - The classical hats cost about half a classical solve, because the LU dominates. With S's forward and the strips, the expert would cost 1.2 times (at 128) to 5 times (at 32) the solve it replaces.
  - `"principal"` costs nothing more, but A3 shows it recovers only bend-3.

**B2, `train.Arm(mode_prior="generator-field")`** (default `None`).
- **What changes.** The objective weights the modes' errors by a covariance $\Sigma$ of a piece's traces:

$$\operatorname{tr}(F^{\mathsf T}\tilde\Lambda F)=\operatorname{tr}(\Sigma\tilde\Lambda)=\operatorname{tr}(\Sigma\Lambda^*)+\operatorname{tr}(\Sigma E^{\mathsf T}A_IE),\qquad \Sigma=FF^{\mathsf T},$$

  so T26's argument holds for any $\Sigma$, and the exact modes stay the minimiser. $Z_H$ becomes the principal part's $\operatorname{tr}(\Sigma\Lambda_0)$.
- **The prior.** $\Sigma=Q^+CQ^{+\mathsf T}$, the covariance of the least-squares coefficients of a trace drawn as the training generator draws its sources: a level $c_0\sim U(-1,1)$ plus $s_g\sim U(0,1.5)$ times a unit Gaussian-kernel field,

$$C(x,x')=\tfrac13+0.75\cdot\tfrac13\sum_{\ell\in\{0.125,\,0.25,\,0.5\}}e^{-\lvert x-x'\rvert^2/2\ell^2}.$$

  It is exact, so nothing is sampled, and nothing is fitted to the benchmarks.
- **What it weights, at $n=64$.** Each hat corner difference about 0.73. Side sines: $j=1$ 0.63, $j=2$ 0.19, $j=3$ 0.086, $j=4$ 0.042, $j=8$ 0.0013, $j=15$ $2.6\times10^{-6}$. This is the order A1 measured, though the prior never saw A1.
- **The floor.** The learned block's eigenvalues run from 2.28 to $1.1\times10^{-3}$, and each block is floored at $10^{-3}$ of its largest. There are no terms between the learned and the weight-free columns, so the strips' errors cannot pull the learned ones.
- **Tests:** $F=I$ gives the loss, its normaliser and every gradient bit for bit as the default; T26's constant is the same for two networks; $P\Sigma P^{\mathsf T}=\Sigma$ for all eight symmetries, so D4 augmentation keeps the weighting; and the prior changes the weights two training steps reach.
- Validation and $e_H$ stay unweighted.

**B3, the `"unet-spectral"` backbone.**
- **What it adds.** The U-Net gets a global path at its coarsest level, $x+C_{\mathrm{out}}(\operatorname{gelu}(S(C_{\mathrm{in}}x)))$. Here $S$ is `SpectralLayer(cosine=True)`: real weights per channel pair and cosine mode $(p,q)$, $p,q<8$, on 32 channels, mirrored so that the output is a cosine series of the square as the input is. $C_{\mathrm{out}}$ starts at zero.
- **Its size.** About 79k parameters per network. Its parameters are drawn after the others, so S's weights load into it as the same function, bit for bit.
- **Tests:** off, the U-Net and the spectral layer are as before, bit for bit; on, symmetry $k$ of the input, with the symmetry's action on the weights (kernels moved, cosine weights transposed), gives symmetry $k$ of the output to $10^{-5}$, for all eight; K holds to rounding for both forms; at $n=128$ the centre node reads an input node at a corner, 64 nodes away, where the plain U-Net's gradient is exactly zero.

### 11.5 Issues 1 and 3, as measured

`i13` uses 8 validation pieces, the same seeds drawn at $n=32$–256, so it is the same pieces refined. It takes $N^*=(H-H^{(0)})/\beta_c$ over S's 36 learned columns, finds its peak, and classifies where the peak sits:
- within $n/8$ of a corner;
- within $n/8$ of a side and 2 cells of a material interface, where $\lvert\Delta\log\kappa\rvert\ge0.25$ between neighbouring elements;
- elsewhere ("regular", away from both).

| $n$ | median peak | regular | at a corner | where an interface meets a side (4 of 8 pieces) |
|---|---|---|---|---|
| 32 | 4.17 | 3.81 | 3.55 | 3.63 |
| 64 | 6.80 | 6.29 | 5.32 | 5.73 |
| 128 | 9.37 | 7.36 | 6.70 | 8.52 |
| 256 | 10.63 | 8.59 | 9.01 | **11.43** |

The median peaks reproduce the earlier record (4.2, 6.8, 9.4, 10.5).

**Growth exponents from 128 to 256** ($\log_2$ of the ratio):

| where | exponent |
|---|---|
| the whole piece | 0.18 |
| an interface meeting a side | 0.19–0.52 per piece (median 0.37), with increments that do not shrink (+2.1, +2.8, +2.9) |
| corners | mostly 0.15–0.28; 0.74 on one piece, whose largest corner is 115° (which of its corners peaks is not recorded) |
| the regular part | 0.10–0.27 (one piece 0.79) |

The largest corner angles of the 8 pieces run from 79° to 128°.

**Verdict.**
- **Issue 3 is confirmed:** the target is still growing at $n=256$.
- **Issue 1 is confirmed as its fastest site:** where a material interface meets a side, the growth does not level off. The corner bubble does not treat such points.
- **How common they are:** 4 of the 8 validation pieces have one. The generator draws an interface in 40% of pieces.
- **Not proven:** the classification is coarse, since 2 cells is a tight neighbourhood. So the regular part's growth may partly be the singular points' slow decay.

**Proposed treatments (not built):**
- input channels for the interface, its signed distance and contrast, and one for each corner's angle;
- a classical patch solve at each point where an interface meets a side, as the strips are, with the network's answer as data outside the patch.

**[AI Inference]:** Near a point where an interface meets a Dirichlet side at an angle other than 90°, the exact modes behave like $r^{\alpha}$ with $\alpha<1$ fixed by the contrast. The quotient then grows like $r^{\alpha-1}$, which a bubble that vanishes like the distance cannot absorb. That fits the exponents above, but it was not derived here.

### 11.6 The overfit probe (B4)

**The set-up.**
- **The pieces:** four validation pieces at $n=128$, seeds 2,000,100–2,000,103. Three are steps ($Fo$ = 0.92, 0.23 and 11.2) and one is steady; all four are spline quads.
- **The training:** S's form and the registered label-free objective, with no augmentation. The batch is the four pieces. AdamW at a constant $3\times10^{-4}$, strips skipped in training.
- **The start:** both arms start from S's weights. The B3 arm starts as S bit for bit, since its $C_{\mathrm{out}}$ is zero.
- **The measure:** $e_{H,L}$ as E1 measures it, on the same four pieces.
- **The runs.** First 500 steps an arm (`forms-overfit-<arm>.json`). Then, with the owner's go, 3,000 steps an arm (`forms-overfit-<arm>-3000.json`), 10:57–12:35 on the laptop. The full run reproduces the short one at every step both measured: the probe is deterministic.
- **The conditions:** on AC power, with no sleep, standby or power-source event in the window (System log, Ids 1, 42, 105, 187, 506 and 507), at 0.92–1.46 s a step.

| median $e_{H,L}$, step | 0 | 250 | 500 | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 |
|---|---|---|---|---|---|---|---|---|
| U-Net | 0.3356 | 0.2513 | 0.2322 | 0.2053 | 0.1843 | 0.1751 | 0.1688 | **0.1616** |
| U-Net + the path (B3) | 0.3356 | 0.2578 | 0.2371 | 0.2039 | 0.1869 | 0.1750 | 0.1673 | **0.1577** |

**Beside it.**
- At step 3,000 the four pieces stand at 0.074, 0.199, 0.133 and 0.190 for the U-Net, and 0.074, 0.185, 0.131 and 0.187 with the path.
- The particular field does fit: $e_p$ falls from 0.017–0.029 at step 500 to 0.004–0.010, below E1's bar of 0.02.
- The hats hold 1.4–3.1% of the learned columns' error, the unweighted objective's own weighting.
- The path's trajectory is not monotone (0.1750 → 0.1765 → 0.1673 at steps 2,000–2,500).

**Verdict.**
- **Neither arm reaches 0.01 while fitting the very pieces it is measured on.** The medians are 0.162 and 0.158 at 3,000 steps, and even the best piece, at 0.074, is seven times the bar.
- **The short run's extrapolation was too pessimistic.** Over the last 1,000 steps the median falls 2.0% (U-Net) and 2.6% (path) every 250 steps. That is a local power-law exponent of $-0.20$ and $-0.26$, against $-0.10$ over steps 100–500.
- **How far 0.01 is:**
  - kept up as a steady exponential decay, 27,000–35,000 more steps, about 7–10 h an arm on this laptop;
  - on the power law, out of any budget.
- **The path gives no measurable gain.** It ends 2.4% lower, after trailing until step 1,000, and the arms crossed twice.
- **On this evidence the modes, not the data, are the limit at any registration's budget.** The registration's 37,500 steps must fit thousands of pieces, not four.
- **Not shown:** whether an exponential tail would carry on for 30,000 steps, or another learning rate. A run that long would settle it.

**[AI Inference]:** §11.5's quotient is still growing at the grid scale. A smooth network output times a bubble cannot follow that, which would put a floor under $e_{H,L}$ whatever the data. The particular field, which has no such singular target, fits in the same steps. That is consistent with the reading, but nothing here separates it from an optimisation limit.

### 11.7 The next registration: accepted and hash-pinned (2026-11-07), not yet trained

**Status.**
- The owner accepted this draft in chat on 2026-11-07, and settled its two open choices the same day (quoted below).
- It is written as `atlas/chartop/gates_prior.py`, which reads every bar from `gates_forms`. Twelve tests in `tests/test_chartop_gates_prior.py` check it against this section.
- **Hash-pinned with the owner's approval the same day ("Pin it").**
  - `out/chartop/registered-prior.txt` holds the sha256 of `gates_prior`, `gates_forms`, `gates` and `benchmarks` (which defines the held-out problems), and of the classical record.
  - The `gates_forms` and `gates` hashes are the ones the earlier registration pinned.
  - The treatments are recorded by value: the prior's kernel and floor, and the spectral path's 8 modes on 32 channels.
  - The pin test fails when it should: one byte added to `benchmarks.py` failed it, and the original bytes passed again.
- **E5 priced on a rented box** (below), with the owner's go ("Rent and price"); the box was destroyed.
- **Planned and trained** with the owner's go ("Plan and train"): `out/chartop/plan-prior.json` (commit 4ceab31), and the four runs on two rented boxes (§11.8). Not yet evaluated.
- **The one evaluation's script** is written and tested, not run (§11.9).

**What it would decide.** Whether weighting the modes as the coupled answer does (B2), with and without a global path (B3), brings S's form within E4, and what that costs E1.

**Not registered: B1.**
- The classical hats are a bound (§11.3), and they cost about half a classical solve (§11.4).
- `"principal"` recovers only bend-3.

**Arms.** All use S's form (split, corner bubble, strips for $j=9$–15, learned hats) at the comparison budget: 37,500 steps, strips skipped in training, seeds 0 and 1, as S′.

| arm | backbone | objective | new runs |
|---|---|---|---|
| P | U-Net | `mode_prior="generator-field"` | 2 |
| PG | `"unet-spectral"` | `mode_prior="generator-field"` | 2 |
| S′ (trained, reused) | U-Net | unweighted | 0 |

**Gates.**
- `gates_forms`' bars, unchanged and read from it: E1; E4 in both frames; K, D, L, E3, SE1–SE3, DUAL and KT. S11 is reported.
- As `gates_forms` judged its main arm, the seed-0 run of each arm is judged on every gate, and K, D and L on every run.
- **CP**, P against S′, seed with seed, on CW's measure over the six held-out problems; a verdict stands only if both pairings give it. The same ratio on the six seen problems is reported beside it.
- **CG**, PG against P, the same way.

**Settled by the owner (2026-11-07).**
- *The test split:* "5,000,000 (Recommended)". The split at 3,000,000 has been evaluated once. E1 is judged on 2,400 pieces at seeds 5,000,000–5,002,399, drawn only at the one evaluation.
- *The benchmarks:* "New held-out problems (Recommended)". §11.1 and §11.3 read all three steady problems post hoc, so the benchmark gates (K, D, L, E3, E4, SE1, SE3) are judged on new problems, and the old six are reported beside as seen. KT's nonlinear problems and S11's rollout, which §11 never read, stay as they were.

**The held-out problems** (`benchmarks.build(..., variant="held-out")`; off by default). They use the same layouts, pieces and charts, with new materials and new data on the ends:
- **Materials:** bend-3 in aluminium (not steel); the s-channel in copper (not steel); insert-round with its contrast reversed, a steel disc in an aluminium plate (not copper in steel).
- **Data:** the hot end at $400+20\cos\pi s$ and the cold end at $300+10\cos2\pi s$, with $s$ running from 0 to 1 along each end. The walls stay insulated, the step starts from 300 K, and the time steps are unchanged.
- **Measured with the classical solver only** (`scripts/chartop_register_prior.py`, `out/chartop/registration-prior-heldout.json`). No learned expert, trained or not, has read them.
- **The controls:** the six seen problems' floors equal the one evaluation's bit for bit, so the defaults did not move; there are no envelope refusals; and every Fourier number lies in the generator's range, 0.01–100.

| held-out problem | classical floor, steady | step | per-piece $Fo$ |
|---|---|---|---|
| bend-3 | 0.043% | 0.006% | 0.24–0.80 |
| s-channel | 1.00% (seen: 0.93%) | 0.13% | 0.67–1.13 |
| insert-round | 0.043% | 0.004% | 1.2–4.2 |

**The first design was dropped.** It had $300+10\sin\pi s$ on the cold end. That data meets the insulated walls with a slope they cannot carry, and gave the s-channel's steady problem a classical floor of 2.03%, above E4's 2% bar: no expert could have met E4 there. The registered profiles both have zero slope at the walls. The change was made on the classical floor alone, a property of the problem and the port basis.

**Predictions** (refutable; checked against each other: 1 and 3 together say that S′ also fails E4 on the held-out s-channel steady).
1. P's steady energy errors are below the S′ run of the same seed on all three held-out steady problems.
2. P meets E4 on the held-out bend-3 steady problem.
3. Neither P nor PG meets E4 on the held-out s-channel steady problem. On the seen one, exact hats alone left 4.44% (§11.3), so the low sines would have to be nearly exact as well.
4. Neither meets E1; the median $e_{H,L}$ stays above 0.1. B2 trades unweighted accuracy for coupled accuracy, and the target still grows with $n$ (§11.5).
5. PG's median $e_{H,L}$ at $n=128$ is no higher than at $n=32$, while P's rises with $n$, as S's did (0.250 → 0.294).
6. K, D and L are met, by structure.

**Cost.**
- S′ took 0.83 h a run (79.4 ms a step) on an RTX 3090 box (§8.9).
- The prior adds one $64\times64$ recombination a step. The path adds about 79k parameters and an 8-mode FFT on at most $17\times17$ nodes.
- Budget 1.0–1.5 h a run: four runs, about 4–6 box-hours on two boxes. At the boxes' last rates (\$0.24 and \$0.34/h) that is about \$1–3, within the \$0.70/h cap. (E5, below, measured 0.82 h a run.)
- The one evaluation: about 20 minutes on the laptop, as before.

**E5, priced on the box (2026-11-07).**
- **The box:** one RTX 3090 on an AMD EPYC 7552 (vast.ai instance 59610940, offer 27741141), at \$0.176/h with 20 GB of disk.
- **The run:** `scripts/chartop_price.py --prior` (the runner's stage `price-prior`), untrained random weights, bfloat16 autocast as decision D4 has it. S′'s form, as S′ was trained, is the reference timed in the same run. Record: `out/chartop/price-prior-2c136ab53e4f-20261107-185847.json`, hash-checked against the box before it was destroyed.
- **The box's life:** 13:50–13:59, credit \$10.528 → \$10.508.

| registered size | S′'s step | P's step | PG's step | forward (strips and assembly included), every arm |
|---|---|---|---|---|
| $n=32$, batch 32 | 52.4 ms | 60.0 ms | 56.7 ms | 0.13 s |
| $n=64$, batch 32 | 78.0 ms | 85.3 ms | 86.8 ms | 0.39–0.40 s |
| $n=128$, batch 8 | 73.1 ms | 80.6 ms | 82.8 ms | 0.47–0.48 s |
| Stage 0's mix | **70.7 ms** | **78.2 ms (×1.106)** | **79.1 ms (×1.119)** | |

**What the price says.**
- **The prior** costs about 11% a step: the float64 recombination of the 64 mode fields.
- **The path** costs about 1% more; it adds 158k parameters to the 2.2 M.
- **The reference matches S′'s own training:** 70.7 ms here against S′'s 79.4 ms (§8.9), so this box ran about 11% faster.
- **Training:** P 1.63 h and PG 1.65 h for their 75,000 steps each, 3.3 h in all.
- **Cost:** about \$0.6–1.1 at \$0.18–0.34/h, within the \$10 ceiling. The memory peaks at 1.7 GiB.
- **The data:** the generator ran 24 examples a second on the box's 14 workers.

**The first pricing run failed, and was fixed before the second.**
- **What happened:** PG's path died under bfloat16 autocast. Autocast lowered the cosine layer's real mixing (a matmul) to bfloat16, and `torch.complex` takes no bfloat16.
- **Why the tests missed it:** they all ran without autocast.
- **The fix:** `SpectralLayer`'s cosine path now computes its spectrum in float32 with autocast off. With autocast already off, that changes no operation, so the CPU answers are bit for bit what they were.
- **The test:** a new test runs the path under the CPU's bfloat16 autocast. It failed with the box's error before the fix, and passes after.
- **What did not change:** the registered treatment, 8 modes on 32 channels, by value.

**E1 is out of reach for these arms.** B4's full probe (§11.6) is done: neither arm fits four pieces below 0.16 in 3,000 steps. So E1's bar is out of this form's reach at a registration's budget, and the plan should say so before anything trains. The arms above are aimed at E4, not E1; prediction 4 stands.

### 11.8 The four runs: trained, not yet evaluated (2026-11-07)

**The owner's go:** "Plan and train", answering "Write the plan and train P and PG on rented boxes (about 3.3 GPU-hours, about \$1)?".

**The plan** (`out/chartop/plan-prior.json`, binding from commit 4ceab31):
- S′'s batches, learning rate, validation cadence, data counts and 120-minute guards, run for run, with the strips skipped in training.
- `scripts/chartop_train.py` refuses the plan unless every run is a registered arm and seed with its registered configuration.
- On each box, `scripts/chartop_budget.py` costs the whole registration from that box's own price record and trains only within \$10.

| box | machine | rate | rented, local time | runs |
|---|---|---|---|---|
| 1 | RTX 3090, EPYC 7C13 (32 effective CPUs, 63 GB), vast.ai 59614685, offer 37955906 | \$0.234/h | 14:27–16:17 | P-s0, P-s1 |
| 2 | RTX 3090, Xeon E5-2680 v4 (28 effective CPUs, 63 GB), vast.ai 59614692, offer 51465380 | \$0.246/h | 14:27–16:57 | PG-s0, PG-s1 |

**Each box priced itself first and passed the gate.**
- **Box 1:** P ×1.095 and PG ×1.127 over S′'s 68.2 ms. The whole registration projected 5.34 h and \$1.27.
- **Box 2:** P ×1.099 and PG ×1.154 over 77.9 ms; 6.49 h and \$1.62.
- **The upload, at the first try, arrived empty:** a trailing `&` put the whole `&&` chain in the background, and a background job reads /dev/null. Nothing ran. It was redone in two calls, with the payload's sha256 checked on both boxes.

**The runs** (`out/chartop/runs/prior/<run>/`: `arm.json`, `log.jsonl`, `summary.json`). The weights `best.pt` stay local; their sha256 are in `out/chartop/runs/prior/weights-sha256.txt`. Every file was hash-checked against its box before the box was destroyed. The kept checkpoint is the one with the lowest validation median $e_H$, which is unweighted: not what P and PG were trained on.

| run | steps | hours | ms a step | validation median $e_H$: first (step) → kept (step) | 90th percentile, kept | validation median $e_p$: first → kept → last | $e_K$, largest |
|---|---|---|---|---|---|---|---|
| P, seed 0 | 37,500 | 0.84 | 80.3 | 0.1941 (2,500) → 0.1932 (20,000) | 0.291 | 0.355 → 0.228 → 0.192 | $1.1\times10^{-15}$ |
| P, seed 1 | 37,500 | 0.84 | 80.2 | 0.1944 (2,500) → 0.1883 (37,500) | 0.281 | 0.353 → 0.115 → 0.115 | $1.1\times10^{-15}$ |
| PG, seed 0 | 37,500 | 1.07 | 103.2 | 0.1939 (2,500) → 0.1915 (37,500) | 0.287 | 0.300 → 0.137 → 0.137 | $1.1\times10^{-15}$ |
| PG, seed 1 | 37,500 | 0.98 | 93.8 | 0.1937 (2,500) → 0.1918 (27,500) | 0.287 | 0.295 → 0.138 → 0.130 | $1.1\times10^{-15}$ |
| S′, seed 0 (§8.9, beside) | 37,500 | 0.83 | 79.4 | 0.1947 (2,500) → 0.1934 (20,000) | 0.291 | 0.349 → 0.239 → 0.194 | $1.1\times10^{-15}$ |

**Beside it.**
- **Cost:** credit \$10.506 → \$9.459, so \$1.05 for training. With E5's box, \$1.07 for the registration so far.
- **No box is up.**

**Not shown here:** any gate. The runs are not evaluated, and neither the fresh test split nor the held-out problems has been read by any of them.

**[AI Inference]:** P's unweighted validation median is S′'s to the third digit at seed 0 (0.1932 against 0.1934). That fits a prior that moves accuracy among the columns without changing the unweighted total much. Whether it moved accuracy to where the coupled answer needs it is E4's question, at the one evaluation.

### 11.9 The one evaluation's script: written and tested, not run (2026-11-07)

**The owner's go:** "Write it", answering "Write the one evaluation's script for gates_prior (tested, not run)?".

**`scripts/chartop_evaluate_prior.py`, in order:**
1. **The pin:** every file `registered-prior.txt` pins must hash as recorded, or it refuses.
2. **The weights:**
   - every run's `best.pt` must hash as downloaded;
   - its configuration must be its arm's: form, backbone, networks, ports, bubble, strips, hats;
   - its `arm.json` must show the arm's training: the prior, the strips skipped, the steps, the seed.
3. **S′:** S′'s two runs must hash as `gates_prior.REUSED` names them.
4. **The registry:** it refuses the four runs if they were evaluated before. S′, evaluated once already, is reused by name and recorded beside; it is not registered again.
5. **The fresh test split:** 2,400 pieces at seeds 5,000,000–5,002,399, with sine-hat labels. `chartop_data.py` gained the split `test-prior`, whose seed base it reads from the pinned `gates_prior`. The split is drawn unless it is whole on disk, and its sha256 is taken.
6. **E1 and K** for P-s0 and PG-s0; **L** for all four runs on the first 100 test pieces.
7. **The six held-out problems,** both frames, for the four runs and S′:
   - E4, D, K and L on every piece, and every piece's $e_H$ and $e_{H,L}$;
   - E3, SE1, SE3 and the dual bracket for the seed-0 runs.
8. **The seen six** beside, E4's energy error only.
9. **CP** (P over S′) and **CG** (PG over P), seed with seed, on the held-out problems, with the seen problems' ratios beside.
10. **DUAL** on 100 steady test pieces; **KT** and **S11** on `gates_forms`' problems. Then it writes `evaluation-prior-<time>.json` and `.txt`, whatever they say.

The parts the two registrations share come from `chartop_evaluate_forms` unchanged: E1's rows, gate L's three calls, the per-piece measures, the dual bracket's test records.

**Tested** (`tests/test_chartop_evaluate_prior.py`, two tests, throwaway weights; validation seeds stand in for the fresh split, so no registered test piece is drawn):
- **bend-3's held-out problem in both frames at $n=32$:**
  - K, D, L and SE2 met for every run, whatever the weights;
  - E3, SE1, SE3 and the bracket for the seed-0 runs only;
  - the variant really applied: the held-out floor is above $10^{-6}$, the seen one below;
  - the seen problem carrying E4's energy error alone.
- **The whole evaluation end to end,** with stand-ins for the benchmarks, KT and the rollout. Its pairings come out as designed: CP 0.5 in both pairings, "alternative wins"; CG 1, a tie.
- **Refusals:** a second evaluation; a tampered weights file; a configuration that is not its arm's; reused weights the registration does not name; and a changed pin.
- **The suites:** all 273 chartop tests pass.

**Not run.** No trained weight has been read, and neither the fresh split nor the held-out problems has been drawn or read by a learned expert. The run is about an hour on the laptop, on AC: the draw, two E1 passes, 12 problems for six runs, KT and the rollouts.

---

## See Also

- [[chart-operator-stage-0]] — the record this follows: §9.7 found both defects, §12 asked for these fixes
- [[chart-operator-architecture]] — the design: constraint modes, exact traces, the energy Gram
- [[chart-operator-training-and-cost]] — the objective (T26) the forms leave unchanged
- [[chart-operator-general-form]] — G0, and the kernels of the other families (rigid motions, affine fields)
- [[chart-operator-family-catalog]] — CO-D, and the families Stage 1 builds next
- [[formal-proofs-record]] — T25 and T26, which hold for these forms as for any outputs with exact traces
- [[gap-worklist]] — W356 and W357; W358, the corner growth, and its fixes compared in §10
