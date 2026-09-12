# The Corrupted Checkpoint, Swept — Tier 48's cell was a draw, and what the corruption actually moves

**Type:** Concept page — **a sweep with its controls, and a mechanism measured where one was inferred** (folder: `Atlas 0.1/common/`)
**Status:** built 2026-09-12, Tier 50. `scripts/w205_corruption_sweep.py`, `out/w205/w205.json`, `tests/test_tier50_corruption_sweep.py`. Worklist row **W205** (closed), **W204** and **W208** (annotated). Registered off the ladder as **CS-S5**.
**Related:** [[defect-correction-learned-operator]] · [[learned-contribution-kill-tests]] · [[substitution-campaign-checkpoint]] · [[case-study-scaling-ladder-atlas-0.1]] · [[gap-worklist]] · [[case-study-ladder-to-f1]] · [[f1-pathmap-and-end-goal]] · [[composition-error-theory]] · [[master-error-bound]]

---

# 0. The result, in one paragraph

**Tier 48's headline was the best of twelve draws, and the mechanism underneath it is the opposite of the one that page guessed.** [[defect-correction-learned-operator]] §8.2 measured a checkpoint carrying Gaussian noise at $3\%$ of every weight tensor's rms needing $71$ classical calls inside defect correction against the clean checkpoint's $99$, at six windows, **on one seed at one magnitude**, and left the inversion unexplained (W205). Swept over **twelve directions** at that same magnitude the counts run $\mathbf{71}$ to $\mathbf{140}$ about a clean checkpoint at $99$, so the pre-registered verdict is **NOISE**: $71$ is the joint *minimum* of the sample and the $28$-call saving Tier 48 reported is an extreme value rather than an effect size. **At twelve windows it is worse than a draw**: four directions give $138$, $221$, $261$, $309$ about a clean checkpoint at $222$, so the *mean* corrupted copy is **worse** than the clean one and Tier 48's $138$ is again the minimum — which reverses that tier's *"the margin widens"* reading of its two rungs (§6.1). **But the sweep also found a smaller effect the clause could not see**: ten of those twelve directions do beat the clean checkpoint (sign test $p = 0.019$), the median saves $13.5$ calls where Tier 48 claimed $28$, and the response to the corruption's *magnitude* is **non-monotone** — a little weight noise helps ($93.5$, $88.0$, $89.5$ mean calls at $\sigma = 0.003$, $0.01$, $0.03$ against $99$) and a lot hurts badly ($117.0$ at $\sigma = 0.1$, $130.2$ at $\sigma = 0.3$). **The mechanism is then measured rather than inferred.** Each map's derivative is probed directly along band-limited perturbations at the settled state, Theorem 2's per-band eigenvalue $\lambda = |\phi-\psi|/|1-\psi|$ is assembled from it, and over twenty copies it orders the classical-call counts at a rank correlation of $\mathbf{+0.83}$ against accuracy's $+0.69$ — so **Jacobian fidelity** decides these arms and accuracy does not, which is [[defect-correction-learned-operator]] §2.2's reading, confirmed. The slow band is where it is decided, and three numbers say why: the classical map's slow-mode response is $\phi = 0.4780$, the clean column's is $0.5578$ — **preserving what the monolith damps**, exactly the **[AI Inference]** of that page's §5, now measured on the derivative — and the $\alpha^\star = 0.5$ shrink halves it to $0.2789$, **overshooting to well below the classical map**. The corruption raises it back toward $\phi$, which is why a little helps and a lot hurts. **So what the corruption buys is compensation for an overshooting shrink**, the *dissipation* candidate named in [[gap-worklist]]'s W205 row is refuted in sign — what helps is damping **less** — and the cheapest thing this page opens is not a better checkpoint but a better $\alpha$: matching the two would want $\alpha \approx 0.14$, not $0.5$.

---

# 1. What the row asked, and why one cell could not answer it

W205, as [[gap-worklist]] Tier 48 wrote it:

> **A corrupted checkpoint is the better cheap operator, out of sample.** … One seed, one corruption level, two out-of-sample rungs. Done when the saving is attributed: a corruption sweep (several seeds and magnitudes) against a property of the column that is not its accuracy — its dissipation on the slow modes is the candidate — or the ordering is shown to be noise.

The row names its own weakness. A single draw of a random perturbation supports a claim about *that draw*; the sentence it was used for — *"whatever the shrunk column contributes to this iteration's rate, it is not what the checkpoint learned"* — is a claim about the *distribution*. This page measures the distribution.

**Intuitively.** Someone rolled one die, got a six, and concluded the die was loaded. The way to check is to roll it twelve times. It came up six once, and also came up one.

---

# 2. The instrument, and the control that gates it

Everything reuses `scripts/w202_kill_tests.py` unchanged — its `Maps`, its seven arms, its `DEC_SETTINGS` ($k_{\max}=25$, $m_{\max}=40$, inner tolerance $0.1$, stall on two non-decreasing residuals, fallback), its stopping residual and its cached reference marches. The only addition is `Maps.set_corruption(sigma, seed)`, which re-draws `Pw`'s noise on a **deepcopy** so the clean checkpoint is never touched and two cells differ by those two numbers and nothing else. **Both defaults are Tier 48's**, which is what makes the first stage a control rather than a formality:

| control | expected | measured |
|---|---|---|
| $\sigma = 0.03$, seed $20260911$, six windows | $71$ | $\mathbf{71}$ |
| the clean checkpoint at the same settings | $99$ | $\mathbf{99}$ |

**Nothing swept until that passed**, and `scripts/w205_corruption_sweep.py` refuses to continue if it does not: a sweep whose default cell does not reproduce is measuring its own scaffolding. `out/w202/w202.json` is untouched, so every Tier 48 test still asserts of it and still passes.

## 2.1 One correction to the reading, made before any cell was read

The first version of the verdict rule asked whether *every* magnitude's spread cleared the clean checkpoint. That scores the $\sigma \to 0$ **continuity control** — a copy corrupted a tenth as hard, which *should* behave like the clean checkpoint — as evidence of noise, which is backwards: it is the control working. The rule now decides at **Tier 48's own magnitude**, $\sigma = 0.03$, and treats the other magnitudes as controls on it. Recorded here rather than quietly, because a reading repaired after the numbers are in is not a pre-registered reading. Measured, the control behaves: $\sigma = 0.003$ sits $5.5$ calls from the clean checkpoint against the decision cell's $9.5$.

## 2.2 A design property that decides what the grid can conclude

`torch.Generator().manual_seed(s)` fixes the *sequence* of draws, and the corruption is `randn(shape) * sigma * rms`. **So a seed is a direction in weight space and $\sigma$ is its radius**: one row of the grid is one ray sampled at five radii, not five independent draws. It shows in the data — seed $20260913$ is the worst of its row at $\sigma = 0.003$, $0.01$ **and** $0.03$, because it is the same direction scaled.

That makes the grid strong for *"does the magnitude matter along a ray"* and weak for *"is one cell a draw"*, which needs many directions at one magnitude. The decision magnitude was therefore extended from four directions to **twelve**.

---

# 3. The grid

Six windows, $\alpha^\star = 0.5$, everything else Tier 48's. Classical calls to the same certified residual:

| $\sigma$ | directions | classical calls | mean | spread |
|---|---|---|---|---|
| $0.003$ | 4 | $87, 88, 91, 108$ | $93.5$ | $21$ |
| $0.01$ | 4 | $75, 79, 80, 118$ | $\mathbf{88.0}$ | $43$ |
| $\mathbf{0.03}$ *(Tier 48's)* | **12** | $\mathbf{71}, 71, 72, 79, 79, 84, 87, 91, 91, 91, 118, \mathbf{140}$ | $89.5$ | $\mathbf{69}$ |
| $0.1$ | 4 | $95, 115, 126, 132$ | $117.0$ | $37$ |
| $0.3$ | 4 | $115, 126, 134, 146$ | $130.2$ | $31$ |

against, from [[defect-correction-learned-operator]] §8.2.1 on the same rung: the clean checkpoint $99$, the composition layer with the checkpoint replaced by the identity $112$, the detuned checkpoint $113$, the cold classical march $141$.

## 3.1 The pre-registered verdict

**NOISE.** At the decision magnitude the across-direction spread $71$–$140$ covers the clean checkpoint's $99$, so which side of it a single draw lands on is luck. **Tier 48 drew the good one**: $71$ is the joint minimum of twelve, and the same magnitude produced $140$ — one call short of not iterating at all.

The corrupted family at one magnitude spans from *the best arm Tier 48 measured* to *the cold march*. **The corruption's effect is larger than every other effect that tier measured, and its sign is set by the draw.**

## 3.2 And a real effect underneath it, reported beside the verdict rather than instead of it

Two readings of the same twelve numbers, because they disagree and both are true:

| reading | value | verdict |
|---|---|---|
| directions beating the clean checkpoint | $10$ of $12$, sign test $p = 0.019$ | **significant** |
| mean against the clean checkpoint | $89.5$ against $99$, $s = 20.5$, $\mathrm{sem} = 5.9$, $t_{11} = 1.61$ | **not significant** |
| median | $85.5$ against $99$ | a $13.5$-call saving |

They disagree because the distribution has a **heavy upper tail**: ten copies cluster between $71$ and $91$ and two sit at $118$ and $140$, which pull the mean up without moving the count. So the honest statement is *the typical direction saves about half what Tier 48 reported, and one direction in six is much worse than not corrupting at all*.

## 3.3 The magnitude response is non-monotone, and that is the finding the mechanism has to explain

$$93.5 \;\to\; 88.0 \;\to\; 89.5 \;\to\; 117.0 \;\to\; 130.2 \qquad (\sigma = 0.003,\,0.01,\,0.03,\,0.1,\,0.3)$$

A little weight noise helps; a lot hurts. The minimum sits near $\sigma \approx 0.01$. And at $\sigma = 0.3$ **no direction beats the clean checkpoint** and the spread ($31$) is far below the gap — so the checkpoint's learned content *does* carry something, reliably, and destroying it is reliably bad. **The signal exists; Tier 48 read it in the wrong place.**

---

# 4. The mechanism, measured

## 4.1 Two candidates, and they are not the same candidate

There are **two** statements in the record about what the slow modes are doing, and they are not identical — which is worth separating before measuring, because the probe adjudicates between them:

| where | the candidate |
|---|---|
| [[defect-correction-learned-operator]] §2.2 and §5 | **Jacobian fidelity** on the slow modes: the learned column *holds each window's incoming mean and transports periodically* where the monolith *relaxes through its ring and its viscosity* — so the column **preserves** what the monolith damps. Marked **[AI Inference]**, read off the error structure of stalled iterates rather than off assembled Jacobians |
| [[gap-worklist]]'s **W205** row | **dissipation** on the slow modes: *"a property of the column that is not its accuracy — its dissipation on the slow modes is the candidate"* |

They point opposite ways. "Preserves too much" says the fix is to **damp more**; "dissipation is the property doing the work" says the same. §4.4 measures which, and the answer is neither quite: the column preserves too much **before** the shrink and too little **after** it.

Theorem 2 is about a derivative, so the derivative is what this page probes:

$$D\Psi\,d \;\approx\; \frac{\Psi(w^\star + \varepsilon d) - \Psi(w^\star)}{\varepsilon},\qquad \psi_{\text{band}} \;=\; \frac{\langle d,\,D\Psi\,d\rangle}{\langle d,\,d\rangle},$$

with $d$ a unit-rms random field whose energy sits in one band of $|k|$, averaged over three draws, **paired across maps** (the same fields for every map), at $\varepsilon = 10^{-4}$. The Rayleigh quotient is used rather than the amplification $\lVert D\Psi d\rVert$ because an amplification has no sign and therefore cannot tell a map that *preserves* a mode from one that *inverts* it — which is exactly the distinction Theorem 2's denominator turns on.

Three bands: **slow** $|k|\le 2$ (the band §5 names), **mid** $3\le|k|\le8$, **fast** $12\le|k|\le32$.

## 4.2 The classical map amplifies the slow band, which is why the rate lives there

| map | slow | mid | fast |
|---|---|---|---|
| $\Phi$, the classical monolith | $\psi = 0.4780$, amplification $\mathbf{1.2585}$ | $0.3696$ | $-0.0063$ |
| $P$, the clean checkpoint | $\psi = 0.5578$, amplification $1.0354$ | $0.3586$ | $-0.0056$ |
| $Z$, the checkpoint replaced by the identity | $\psi = 0.5624$ | $0.3903$ | $-0.0315$ |

The classical map **grows** low-mode perturbations over a macro-step — transient growth in an open advective domain, not an instability of the settled state — so the slow band is where $|\phi|$ is largest and where the iteration's rate is decided. It is decided there for every arm measured.

## 4.3 Theorem 2's eigenvalue, assembled and compared

With the shrink applied as the iteration applies it ($\Psi_\alpha = (1-\alpha)\Psi$, so $\psi_\alpha = (1-\alpha)\psi$ at $\alpha^\star = 0.5$),

$$\lambda_{\text{band}} \;=\; \left|\frac{\phi - \psi_\alpha}{1 - \psi_\alpha}\right|.$$

At seed $20260911$, slow band, against the measured classical calls:

| map | $\psi_{\text{slow}}$ | $\lambda_{\text{slow}}$ | classical calls |
|---|---|---|---|
| clean checkpoint | $0.5578$ | $0.2761$ | $99$ |
| $\sigma = 0.003$ | $0.5623$ | $0.2739$ | $87$ |
| $\sigma = 0.01$ | $0.5690$ | $0.2704$ | $79$ |
| $\sigma = 0.03$ | $0.5760$ | $\mathbf{0.2668}$ | $\mathbf{71}$ |
| $\sigma = 0.1$ | $0.5836$ | $0.2629$ | $95$ |
| $\sigma = 0.3$ | $0.3818$ | $\mathbf{0.3548}$ | $115$ |

**Over all twenty probed copies**, Spearman rank correlation against classical calls:

| predictor | $\rho$ |
|---|---|
| $\max_{\text{band}} \lvert\lambda\rvert$ — **Theorem 2's own rate** | $\mathbf{+0.83}$ |
| slow-band $\lvert 1-\psi_\alpha\rvert$ | $+0.83$ |
| **accuracy** — one-step distance from the classical map | $+0.69$ |
| mid-band amplification | $-0.84$ |
| slow-band amplification | $-0.79$ |
| fast-band amplification | $-0.48$ |

$\max|\lambda|$ is attained in the slow band for every copy, which is why it and the slow-band term give the same number.

## 4.4 What the corruption actually moves — and the shrink overshoots

Three numbers settle it, and the order matters:

$$\phi_{\text{slow}} = 0.4780,\qquad \psi_{\text{slow}}(\text{clean, unshrunk}) = 0.5578,\qquad \psi_\alpha = (1-\alpha^\star)\,\psi = 0.2789 .$$

- **Before the shrink the column preserves too much.** $0.5578 > 0.4780$: the clean checkpoint's slow-mode response sits *above* the classical map's, which is exactly [[defect-correction-learned-operator]] §5's diagnosis — *it holds each window's incoming mean where the monolith relaxes it* — now measured on the derivative rather than inferred from stalled iterates. **That inference is confirmed.**
- **After the shrink it preserves too little.** $\alpha^\star = 0.5$ halves it to $0.2789$, which is *below* $\phi_{\text{slow}}$ by more than the original excess was above it. **The correction overshoots**, and nothing in Tier 48 could see that, because $\alpha^\star$ was chosen on $N=2$ as the only value at which the checkpoint converged without falling back — a stability criterion, not a fidelity one.
- **The corruption partially undoes the overshoot.** $\psi_{\text{slow}}$ rises monotonically over the helpful range, $0.5578 \to 0.5623 \to 0.5690 \to 0.5760 \to 0.5836$, carrying $\psi_\alpha$ back toward $\phi_{\text{slow}}$; the numerator $|\phi-\psi_\alpha|$ falls faster than the denominator, so $\lambda$ falls. At $\sigma = 0.3$ the response collapses to $0.3818$, $\psi_\alpha$ drops to $0.1909$, and $\lambda$ jumps to $0.3548$.

$$\boxed{\;\text{The corruption helps by partly undoing an OVERSHOOTING shrink, and hurts once it overshoots the other way.}\;}$$

**So the two candidates of §4.1 are both half right, and the worklist row's is the one refuted in sign.** *Dissipation on the slow modes* would mean driving $\psi$ **down**; what helps at every magnitude that helps is driving it **up**. §2.2's and §5's reading — *Jacobian fidelity*, and a column that preserves what the monolith damps — is confirmed of the **unshrunk** column and now carries a number: $\rho = +0.83$ against accuracy's $+0.69$.

**[AI Inference]:** $\alpha$ and $\sigma$ are then two knobs on one quantity, and $\alpha^\star = 0.5$ is what puts the shrunk column below the optimum in the first place. The well-posed experiment is not to corrupt weights at all but to **choose $\alpha$ per rung so that $(1-\alpha)\,\psi_{\text{slow}} \approx \phi_{\text{slow}}$**, which here would want $\alpha \approx 1 - 0.4780/0.5578 = 0.143$ rather than $0.5$, and costs three forward calls instead of a grid of marches. Not measured here, and it is the cheapest live descendant of this page.

---

# 5. What this does to Tier 48's record

**Nothing in `out/w202/w202.json` moves, and no assertion in `tests/test_tier48_learned_contribution.py` changes.** Those tests pin what that tier measured, and it measured it correctly. Two *docstrings* are annotated, because they generalise from one cell:

| clause | Tier 48 | after the sweep |
|---|---|---|
| **G6 · loud**, second half: $P_w$ does not need fewer classical calls than $P$ | **fails** at six windows ($71 < 99$) and at twelve ($138 < 222$) | **a one-draw Bernoulli trial on a quantity whose across-direction spread is $69$ calls.** Ten of twelve directions fail it and two pass it. The clause is not *failed*; it is **not measurable as posed** |
| *"a corrupted checkpoint is the better cheap operator"* | stated of both out-of-sample rungs | **not supported as a general claim.** The typical direction saves about half as much, and one in six is worse than the clean checkpoint |
| *"whatever the shrunk column contributes to this iteration's rate, it is not what the checkpoint learned"* | inferred from the inversion | **still supported, by a different and stronger argument** (§5.1) |

## 5.1 The sentence that survives, and the number that carries it

The clean checkpoint needs $99$ classical calls; the same composition layer with the checkpoint replaced by the identity needs $112$. **So the learned content is worth $13$ classical calls.** A $3\%$ perturbation of that same checkpoint's weights moves the count across a range of $\mathbf{69}$.

$$\frac{\text{what perturbing the weights moves}}{\text{what training the weights bought}} \;\approx\; \frac{69}{13} \;\approx\; 5.3 .$$

**The checkpoint's learned contribution in this slot is roughly a fifth of its own weight-noise sensitivity.** That is a sharper statement than the inversion was, it does not depend on any single draw, and it is the one that bears on [[f1-pathmap-and-end-goal]] §3.3's standing sentence — *the foundation-model half has no learned expert admitted with a nonzero contribution*. The contribution here is nonzero and it is smaller than the noise floor of the object that carries it.

**One caveat, named.** $Z$ replaces the checkpoint with the **identity**, not with a randomly initialised network of the same architecture. So $13$ calls is *what this checkpoint buys over doing nothing in that slot*, and it is **not** *what training bought over random initialisation*, which no arm here measures.

---

# 6. The second rung, where it is worse than a draw

Twelve windows, four directions at the decision magnitude, everything else unchanged. The reproduction control passes first: $\sigma = 0.03$ at the original seed returns $\mathbf{138}$ and the clean checkpoint $\mathbf{222}$, both Tier 48's numbers to the call.

| | classical calls |
|---|---|
| the four directions | $\mathbf{138},\; 221,\; 261,\; \mathbf{309}$ |
| mean | $\mathbf{232.2}$ |
| median | $241$ |
| spread | $171$ |
| the clean checkpoint | $222$ |
| the cold classical march | $267$ |

**The verdict is NOISE again, and the sign of the effect has flipped.** Tier 48's $138$ is once more the sample *minimum*; but here the mean, $232.2$, is **above** the clean checkpoint's $222$ — so at twelve windows the typical corrupted copy is **worse** than the clean one, not better. Only one direction of four beats it, and one ($309$) is worse than **not iterating at all**.

## 6.1 What that does to "the margin widens"

[[defect-correction-learned-operator]] §8.2.2 and [[gap-worklist]]'s W205 row both read the two rungs as a trend: $28$ classical calls of margin at six windows, $84$ at twelve, *"and the margin widens"*. Measured over directions rather than over one draw each, the trend **reverses**:

| rung | clean $-$ Tier 48's draw | clean $-$ the mean over directions |
|---|---|---|
| $N=6$ | $+28$ | $+9.5$ |
| $N=12$ | $+84$ | $\mathbf{-10.2}$ |

The widening margin was two draws from the good tail of two distributions whose centres move the other way. **A trend built from one sample per point is not a trend**, and this is the cheapest possible demonstration of it.

## 6.2 And the spread grows with the rung

$69$ calls at six windows on a mean of $89.5$; $171$ at twelve on a mean of $232.2$ — $77\%$ and $74\%$ of the mean respectively. **The corruption's effect scales with the problem**, so a single draw becomes *less* informative as the graph grows, not more, which is the opposite of the direction an out-of-sample rung is usually added for.

---

# 7. What this tier did NOT do, named

- **No expert was trained, and none was fine-tuned.** The corruption is applied to a deepcopy of a frozen checkpoint.
- **The Jacobians were still not assembled.** What is measured is the derivative's action along three band-limited directions, three draws each — a projection, not a spectrum. $\lambda$ is therefore a *band-averaged* estimate of a quantity whose true value is a maximum over individual modes, and the $\rho = +0.83$ is a rank correlation over twenty copies and not a proof of causation.
- **Only one shrink was used.** $\alpha^\star = 0.5$ is Tier 48's, fixed on $N=2$, and §4.4's inference that $\alpha$ is the better knob is untested.
- **The $N=12$ grid is four directions at one magnitude**, not twelve at five: at $300$–$500$ s an arm, the twelve-direction sample the first rung got was not affordable at the second.
- **W204 is untouched.** The entry condition is still a cost bar the checkpoint does not clear, and nothing here changes the $2\times$-coarse classical solver's $38.1$ classical-equivalents at twelve windows.
- **W208 is untouched.** $\Theta$ still comes from one march.
- **Nothing was downloaded** — Poseidon-T from the local cache with the hub offline — **no machine was rented, NeuberNet was not loaded.**

## See Also

- [[defect-correction-learned-operator]] §2.2 (the reading that survives), §5 (the **[AI Inference]** this page refutes), §7 (the gate), §8.2 (the cell that was a draw).
- [[learned-contribution-kill-tests]] — the fourteen formulations this slot came out of.
- [[substitution-campaign-checkpoint]] — the chain W205 was a descendant of.
- [[case-study-scaling-ladder-atlas-0.1]] — the rungs, the geometry and the classical monolith.
- [[gap-worklist]] Tier 50 — W205 closed, and what it opened.
- [[f1-pathmap-and-end-goal]] §3.3 — the sentence §5.1 sharpens.
