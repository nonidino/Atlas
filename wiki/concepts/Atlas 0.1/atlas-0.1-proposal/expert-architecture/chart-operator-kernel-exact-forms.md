# Chart operator — the kernel-exact and split output forms (W356, W357), built and checked untrained

**Type:** Concept page — **build record and draft registration** (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** built and checked on 2026-10-18, written up on 2026-10-19. **Untrained weights only**: zero heads, or seeded random weights. Nothing was trained, registered or hash-pinned. §8 is a draft registration for the owner to accept, change or refuse. `atlas/chartop/gates.py`, Stage 0's registration, is unchanged.
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

## 8. A draft of the next registration (for the owner; not registered)

Nothing in this section is registered, hash-pinned or acted on. The owner may accept it, change it or refuse it in chat. It extends Stage 0's registration ([[chart-operator-stage-0]] §3) and leaves `gates.py` as it is. A new module would carry the new bars.

### 8.1 What would be judged

Family CO-D again, with Stage 0's generator, seeds, splits, benchmarks and envelope, the label-free objective and D4 augmentation.

| arm | network | form | budget | seeds |
|---|---|---|---|---|
| **S** (main) | two U-Nets: modes 1 M, particular 1 M (2.20 M) | split | Stage 0's main, 150,000 steps | 1 |
| S′ | as S | split | comparison, 37,500 steps | 2 |
| K′ | one U-Net, 1 M | kernel | 37,500 steps | 2 |
| B″ | one U-Net, 1 M | Stage 0's bubble, retrained on this code | 37,500 steps | 2 |

The particular network's size is the owner's choice: 1 M as drafted, or 0.3 M (1.38 M in all). Stage 0's C5 chose 0.3 M at an error level where nothing had learned, which settles little.

### 8.2 The gates

| gate | drafted bar | measured untrained |
|---|---|---|
| **K** (the uniform state) | on every held-out test piece and every benchmark piece, steady and step: $e_K\le10^{-12}$ and $e_{K,\infty}\le10^{-12}$ | $\le4.3\times10^{-16}$ and $\le4.4\times10^{-16}$ |
| **D** (the datum) | on every benchmark problem, steady and step, with every temperature lowered by 300 K, the data and $T_{\text{old}}$ together: $\bigl\lvert\lVert u-\tilde u\rVert_{M,\text{lowered}}/\lVert u-\tilde u\rVert_M-1\bigr\rvert\le10^{-8}$ wherever both errors exceed $10^{-9}\lVert u\rVert_M$, and the maximum error over the range within $10^{-9}$ of the range | $\le1.4\times10^{-10}$ and $\le1.5\times10^{-11}$ |
| **L** (the load) | the modes and the level field bit for bit the same with the piece's own source, no source and a unit-RMS random source, on every benchmark piece and the first 100 test pieces. K′ and B″ report their change instead | bit for bit (split) |
| E1 | Stage 0's bins and bars, unchanged; $e_K$ and the low-mode $e_H$ (hats and sines $j\le8$) reported beside $e_H$ | — |
| E3, SE1–SE3, DUAL, KT | Stage 0's bars, unchanged | hold for any outputs (T25) |
| **E4** | Stage 0's bars (energy $\le0.02$; maximum $\le$ the 16-mode floor $+\,0.01$ of the range), **met in both frames**: as registered, and with every temperature lowered by 300 K, where a step's reference energy no longer contains the level | as initialised: steps 0.27/0.69/0.39% registered and 3.9/10.8/4.1% lowered; steady 0.00034/5.3/5.6% |
| E5 | reported, the two-network forward and one training step priced on the box before any long run | not priced |
| S11 | the rollout reported, the level field kept with the modes | — |

### 8.3 The comparisons

Each ratio is the alternative arm's over the default's, with Stage 0's bars: the alternative wins at $\le0.8$ and loses at $\ge1.25$; otherwise a tie, and the default stands. With two seeds, a verdict stands only if both pairings give it; otherwise it is reported as unresolved.

| | question | ratio |
|---|---|---|
| **CK1** (W357) | does a load-blind mode network help? | mean E4 energy error over the six problems, the steps lowered by 300 K: K′ over S′; median $e_H$ beside it |
| **CK2** (W356) | does the projection help a single network? | the same measure, B″ over K′ |

The two isolate one change each: B″ to K′ is the projection, and K′ to S′ the split. The measure is E4's, because $e_H$ is blind to the effect being compared (§7.3).

### 8.4 Predictions, registered beside the gates so that they can be refuted

**[AI Inference]**, each checked against the others:
1. K, D and L hold. They are structural, so a failure would be a defect in the code.
2. CK2 favours the projection, B″ over K′ at least 1.25: B's steady errors fell 5.4, 4.6 and 6.3 times when the level was removed.
3. E1 is not met, since 68.5% of $e_H^2$ sits in the side sines $j=9$–15 and these forms leave them as Stage 0 did (§7.4).
4. Lowered by 300 K, S's step errors stay within 1.25 times the untrained form's (3.9%, 10.8%, 4.1%). There is no source left in that frame, so only the modes count, and in Stage 0 the modes did not learn. If this prediction fails in the favourable direction, the modes have learned.

### 8.5 Cost, and a decision for the owner

Stage 0 spent \$4.06 on 525,000 training steps over eleven arms. This draft has 150,000 main steps and six comparison runs of 37,500, 375,000 in all, so of the same order. **[AI Inference]:** about \$3–5 at Stage 0's rates if the two networks cost under twice one network. Stage 0's step was dominated by the float64 objective and the sine transforms, not by the network. The draft would take a ceiling like D6's, \$10, each box at most \$0.70 an hour.

**The owner's decision: train these forms now, or after W358.** **[AI Inference]:** training now measures W356's and W357's effects in isolation (CK1, CK2) at a few dollars. But E1, the gate on the network's accuracy, is expected to fail again (prediction 3), because the side modes' corner growth is untouched. W358's definition of done asks for its fix to be shown on validation pieces before any training. If the owner prefers one run that could pass E1, the order is W358's demonstration first, then one registration carrying both.

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

**Recommendation.** Register **C4 + C3**: the corner bubble on every output (built and tested, below), with the side sines $j=9$–15 computed by strip solves. It is the only combination whose target levels off. It leaves a smooth $e_H$ of 0.124 ($K=8$) or 0.071 ($K=16$) at $n=128$, against Stage 0's 0.335 and 0.187, for 0.41 of the exact solve's CPU time. Four sweeps on top (C2 ×4) are optional: they lower the smooth $e_H$ to 0.106 for another 0.2 of the exact solve. Not C1, which costs more than the solve it would replace, and not C2 alone, which fades with $n$. **The strips are measured here but not built into the expert.** The expert would need the piece's assembled operator near its sides in its call; the host has it. **[AI Inference]:** neither smooth class reaches E1's bar of 0.01. The error that remains is the low modes' anisotropy, which is what the network is for, so E1 now measures the network rather than the form. Whether a trained U-Net beats a 16 × 16 cosine class is a training question. The strips' fifth of an exact solve also counts against E5, whose network cost has not been priced.

**Built** (untrained; nothing registered or hash-pinned; `gates.py` unchanged):
- `forms.corner_bubble`, and `KernelExpert(..., bubble="corner")` for both forms. `"square"`, Stage 0's bubble, stays the default and answers bit for bit as before. `train.Arm.bubble` and `train.load_expert` carry the choice. A configuration written before W358 loads as `"square"`, and Stage 0's own forms refuse `"corner"`.
- Five tests in `tests/test_chartop_model.py`, with random weights and classical solves only:
  - the bubble's bounds against the distance to the boundary at $n=16$, 64 and 256, and its bitwise symmetry;
  - the 120° parallelogram above;
  - both forms kernel-exact under the corner bubble at weight scale 0.3, with the side modes equal to $H^{(0)}+\beta_cN$ bit for bit;
  - the square bubble unchanged;
  - training, saving and loading.

**Not shown here:** a trained network in either bubble; the strips inside the expert; the particular field's quotient; any resolution above 256 in the record; any treatment of the interface meeting a side.

---

## See Also

- [[chart-operator-stage-0]] — the record this follows: §9.7 found both defects, §12 asked for these fixes
- [[chart-operator-architecture]] — the design: constraint modes, exact traces, the energy Gram
- [[chart-operator-training-and-cost]] — the objective (T26) the forms leave unchanged
- [[chart-operator-general-form]] — G0, and the kernels of the other families (rigid motions, affine fields)
- [[chart-operator-family-catalog]] — CO-D, and the families Stage 1 builds next
- [[formal-proofs-record]] — T25 and T26, which hold for these forms as for any outputs with exact traces
- [[gap-worklist]] — W356 and W357; W358, the corner growth, and its fixes compared in §10
