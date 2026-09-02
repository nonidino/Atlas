# Seam Placement — can an algorithm choose the cut, and does the answer survive a rollout?

**Type:** Concept page — **case-study summary** (folder: `Atlas 0.1/common/`)
**Status:** opened 2026-09-02. The tenth real case study, CS-9★, and the first exercise of the one end-goal capability with no prior art. The short, readable companion to [[tier0-measurements]] §21, which is the full measurement record. Everything here is quoted from `out/w112/w112.json` and `out/w112/horizon.json`; nothing is estimated.
**Code:** `atlas/cases/seam_placement.py` (the decompositions, the scoring, the search), `scripts/w112_seam_placement.py` (the driver), `scripts/w112_horizon.py` (stage G, the march), `scripts/w112_summary.py` (the reader), `tests/test_tier25_seam_placement.py`
**Related:** [[tier0-measurements]] · [[gap-worklist]] · [[interface-transfer-theory]] · [[poc1-retrospective-and-hybrid-roadmap]] · [[case-study-scaling-ladder-atlas-0.1]] · [[case-study-thermal-strain-atlas-0.1]] · [[case-study-ladder-to-f1]] · [[f1-pathmap-and-end-goal]] · [[generalization-requirements]] · [[master-error-bound]] · [[composition-error-theory]]

---

## 1. The question, and the correction it starts from

[[f1-pathmap-and-end-goal]] §1 names four end-goal capabilities. [[poc1-retrospective-and-hybrid-roadmap]] §4 checks the ladder against them and finds three on the existing schedule and one on no rung anywhere: **automatic, adaptive seam placement**. [[prior-art-and-novelty-atlas-0.1]]'s verdict table finds prior art for every other mechanism in this framework and none for *where to place the cut, decided by an algorithm rather than by the person writing the tiling code*. Every case study in this vault so far has chosen its windows the same way: a person typing `N_COL` and `N_ROW`.

**W112** is the row, and it asks for one small thing: run a search over candidate decompositions, scored under a stated cost budget, and compare the cut the search picks against a hand-chosen tiling this vault has already measured.

### 1.1 The premise W112 states is not correct, and the record that corrects it is in this vault

W112 and [[poc1-retrospective-and-hybrid-roadmap]] §4 both say that [[interface-transfer-theory]] §9's score

$$\mathcal Q(\Gamma) \;=\; \frac{1}{\beta}\cdot\frac{\lVert \tilde\Lambda^M-\operatorname{diag}\tilde\Lambda^M\rVert}{\lVert \tilde\Lambda^M\rVert}$$

*"has never been used to choose a cut"* and has never had a falsification. **That is not true.** [[tier0-measurements]] §10.2 — 2026-08-28, four days after $\mathcal Q$ was proposed — scanned twelve cut placements of `tests/strip_model.py` and found $\mathcal Q$ ranks against the measured composed defect at $\mathbf{-0.853}$, that following it costs $1.407\times$ the best available cut, that it has a $\mathbf{361\times}$ orbit under an admissible re-declaration of the same interface space, and that it is constant to $8\times10^{-11}$ where the truth spreads $2.29\times$. `probe.cut_score`'s docstring has read **RETIRED** since that day, and `compiler._cut_policy` replaced it with **L2/C2**: minimize the $\chi$-weighted restriction defect.

So this case study does not re-run a falsification that has already happened. What §10.2 did was move **one cut of a fixed two-strip decomposition** of a linear model problem. What has still never happened — and is what W112's own *"done when"* clause actually asks for — is all four of these at once:

1. a search over **decompositions** (how many windows, where, how much overlap), not a scan of one cut of a fixed one;
2. under a **stated cost budget**, which is what makes a placement question well-posed at all;
3. on the **real geometry** of a measured case study, with a real expert and a developed flow;
4. against the **hand-chosen tiling** that geometry shipped with.

## 2. How it is built

A **candidate decomposition** of a fixed $n_y\times n_x$ domain is a set of interior cut positions per axis plus a halo: window $(i,j)$ spans $[x_{i-1}-h,\ x_i+h)\times[y_{j-1}-h,\ y_j+h)$, clipped to the domain. Adjacent windows overlap by $2h$; the partition of unity is `ArrayTiling`'s own rule — $\min(w_y,w_x)^2$ over a ramp at artificial faces only, normalized.

Two properties make that the right parameterization. **CS-7's hand-chosen tilings are candidates in it** — `ArrayTiling(n_col=3, n_row=2)` is exactly `x_cuts=(120,232), y_cuts=(120,), halo=8` — and the boxes, the weights and the composed defect are asserted identical to `wake_array`'s **to the bit**. And **the cuts move independently**, so the search can put a seam where a person would not.

The windows are `scaling_ladder.RectangularNS` with the elliptic part removed, against a no-projection monolith of the same class, over **one exchange interval** — the setup [[tier0-measurements]] §10.1 uses, so $D_i=\mathcal E_iR_i-R_i\mathcal E$ contains the restriction and nothing else. The rotors are not agents: the search scores the *fluid* decomposition, and the disks enter through the developed state it is scored on, which is the decision a placement algorithm actually faces.

**Eight criteria are scored on every candidate**, and the separation between them is [[tier0-measurements]] §10.2's finding carried onto real geometry:

| criterion | what it reads | what it costs |
|---|---|---|
| $\mathcal Q$, `__max` and `__sum` | the probed seam operator | $m+1$ solves per block per seam |
| $\lVert S-\operatorname{diag}S\rVert$ un-normalized, `__max` and `__sum` | the same operator, without $\mathcal Q$'s two divisions | the same probe |
| $\lVert\sum_i\chi_i\lvert D_i\rvert\rVert$ — **L2/C2**, the derived one | the local solves **and a monolith** | a monolith a frozen expert does not have (W95) |
| $\hat{\mathcal Q}$, the reference-free surrogate | neighbours' disagreement on the overlap | **nothing** beyond the local solves already done |
| `cut_shear`, `__mean` and `__max` — **new here** | $\lvert\partial_n(u,v)\rvert$ along the cut lines, **from the state** | two array gradients and a slice |

$\mathcal Q$ is computed by the framework's own `probe.cut_score` on an $S$ assembled by the framework's own rule; `tests/test_tier25_seam_placement.py` asserts the block this file probes equals `probe.probe_block`'s to $<10^{-12}$ relative. The score being graded is §9's $\mathcal Q$ and not a paraphrase of it.

### 2.1 An upper-bound budget is not a placement question, and the first run proved it

Under `n_windows <= 6` every criterion returned the same **two-window** decomposition at $0.085\times$ the hand-chosen defect — because fewer seams is less defect, always. That is a true statement about decomposition cost and a worthless one about placement.

So the gate's budget is an **equality**: $n_{\text{windows}}=6$, the hand-chosen tiling's own count, with the halo held at its own width. *Can an algorithm place six windows better than a person placed six windows?* The count and the halo are then swept on their own axes, one variable at a time. The degenerate pool is kept and reported, because a budget's shape is part of the criterion.

## 3. The answer, and it has two halves that point opposite ways

### 3.1 At one exchange interval the search wins, decisively

| geometry | hand-chosen | best found | ratio | where the hand-chosen ranks |
|---|---|---|---|---|
| **N=6** (CS-7's rung, $352\times240$) | $2.126\times10^{-6}$ | $\mathbf{5.707\times10^{-7}}$ | $\mathbf{0.2685}$ | **138th of 178** |
| **N=12** (CS-7's rung, $464\times352$) | $2.727\times10^{-6}$ | $\mathbf{8.316\times10^{-7}}$ | $\mathbf{0.305}$ | **48th of 51** |
| **SOLO** (one rotor, same domain) | $7.861\times10^{-7}$ | $\mathbf{2.319\times10^{-7}}$ | $\mathbf{0.2949}$ | 30th of 54 |

Three criteria — $\hat{\mathcal Q}$, L2/C2's $\mathcal Q^\star$, and `cut_shear__mean` — pick **exactly** that best candidate on N=6. $\mathcal Q$ picks one $1.336\times$ worse; $\mathcal Q$ aggregated by **sum** picks one $3.706\times$ worse, at $0.9949\times$ the hand-chosen defect, which is to say **it does not beat the person at all**.

**And the mechanism is structural rather than a lucky search.** `wake_array.Rotor` lives on an x-*adjacency* — the plane between window columns $i$ and $i+1$ — so a disk sits at the midpoint of an overlap **by construction**. The hand-chosen x-cuts are at cells $120$ and $232$; the rotor planes are at cells $120$ and $232$. **Every x-seam of CS-6 and CS-7 passes through a rotor disc.** The winner moves them to $53$ and $171$, and the mean shear along the cut lines falls from $0.333$ to $0.061$:

| | x-cuts | y-cut | `cut_shear__mean` | one-interval defect |
|---|---|---|---|---|
| hand-chosen | $120,\ 232$ — **the two rotor planes** | $120$ | $0.3333$ | $2.126\times10^{-6}$ |
| the search's answer | $53,\ 171$ | $168$ | $\mathbf{0.0611}$ | $\mathbf{5.707\times10^{-7}}$ |

That is [[generalization-requirements]] G5's slogan — *never along a shear layer, wake centreline or reaction front* — vindicated as a number, and the tiling every wind-farm case study in this vault has used violates it structurally. No page had noticed, because no case study had ever moved a cut.

### 3.2 Marched, the same cut is worse — and the ordering changes sign twice

`scripts/w112_horizon.py` marches three decompositions against the monolith for $240$ exchange intervals ($10.2$ macro-steps), reporting the ratio at every step.

| interval | hand-chosen | the search's answer | ratio |
|---|---|---|---|
| $1$ | $2.126\times10^{-6}$ | $5.707\times10^{-7}$ | $\mathbf{0.2685}$ |
| $8$ | $1.299\times10^{-5}$ | $1.228\times10^{-5}$ | $0.9455$ |
| $10$ | — | — | **crosses $1.0$** |
| $32$ | $2.606\times10^{-5}$ | $3.479\times10^{-5}$ | $1.3352$ |
| $90$ | — | — | $\mathbf{1.5909}$ — the worst |
| $240$ | $8.909\times10^{-5}$ | $8.053\times10^{-5}$ | $0.9039$ |

> **The cut every good criterion in this study picked is worse than the hand-chosen tiling for intervals $10$ through $230$ of $240$, by up to $59\%$.** The ratio crosses $1.0$ twice. A decomposition selected on the eight-interval defect instead holds an advantage longer ($0.35\to0.60$ out to interval $16$) and then crosses too, at interval $95$, peaking at $1.45\times$.

This is not a stability artifact: all three columns hold $u_{\max}$ at $1.658$ against the monolith's $1.658$ and carry an *identical* divergence ($0.2133$ at interval $240$), so the divergence is the state's and not the cut's.

**The honest verdict on the gate, stated as both halves.** The gate as W112 poses it is **passed** — an automatic search finds a decomposition at $0.27\times$ the hand-chosen tiling's defect, well inside any stated factor. And **the gate as posed is the wrong gate**: the quantity every criterion here predicts is the *one-exchange-interval* composed defect, which is exactly what L2/C2 bounds as a theorem, and it is **not monotonically related to what a rollout accumulates**. Following the best-ranking criterion in this study produces a decomposition that is, over most of a ten-macro-step rollout, materially worse than the one a person chose by hand.

## 4. What this says about $\mathcal Q$ specifically

**Three things, and the first is not what §10.2 would have predicted.**

**$\mathcal Q$ does not rank backwards on placement.** Over the N=6 placement pool it ranks $+0.404$ (`__max`) and $+0.726$ (`__sum`); on N=12, $+0.810$ and $+0.873$; on SOLO, $+0.759$ and $+0.681$. §10.2's $-0.853$ was measured on a strip model where the viscosity varied across the cut direction, and it does not carry over to *where a seam goes* in a real wake field. **$\mathcal Q$ is a weak positive predictor of placement, not an inverted one.**

**$\mathcal Q$ ranks exactly backwards on the overlap.** The halo sweep holds the placement fixed and moves only $h$:

| halo (overlap) | one-interval defect | $\mathcal Q_{\max}$ | $\beta_{\min}$ | $\hat{\mathcal Q}$ |
|---|---|---|---|---|
| $4$ ($8$ cells) | $5.434\times10^{-6}$ | $0.04066$ | $0.2371$ | $5.875\times10^{-4}$ |
| $8$ ($16$ cells) | $2.126\times10^{-6}$ | $0.07015$ | $0.2342$ | $3.802\times10^{-4}$ |
| $16$ ($32$ cells) | $1.267\times10^{-6}$ | $0.09600$ | $0.2323$ | $2.702\times10^{-4}$ |
| $24$ ($48$ cells) | $\mathbf{1.088\times10^{-6}}$ | $0.10685$ | $0.2311$ | $\mathbf{2.489\times10^{-4}}$ |

$\mathcal Q$ and both its factors rank $\mathbf{-1.000}$ and choose the narrowest overlap, which is the worst by $4.993\times$. $\hat{\mathcal Q}$ ranks $+1.000$ and chooses the widest. **So §10.2's sign inversion does reproduce — on an axis §10.2 never varied**, and it reproduces because a wider overlap makes the operator *less* diagonal while making the scheme *more* accurate.

**Its basis dependence is worse on real seams than the docstring's worry suggested.** Over $120$ seams of real decompositions of a developed wake field, the orbit of $\mathcal Q$ under $S\mapsto U^\top SU$ for orthogonal $U$ — an admissible re-declaration of the *same* interface space, which leaves the scheme bit-identical:

| | value |
|---|---|
| orbit ratio, min / median / max | $1.237\times$ / $\mathbf{5.652\times}$ / $\mathbf{85.40\times}$ |
| fraction of seams with orbit $>10\times$ | $\mathbf{42.5\%}$ |
| $\mathcal Q$ in the symmetric part's eigenbasis, as a fraction of $\mathcal Q$ declared | median $0.1771$, min $\mathbf{0.0118}$ |
| $\beta$ invariance over all $120$ | $2.859\times10^{-15}$ |

§10.2 measured $361\times$ on one seam of a linear strip model, and it was fair to wonder whether that was a property of the model. It is not: **on two-fifths of real seams $\mathcal Q$ can be moved by more than an order of magnitude without changing the scheme at all.**

### 4.1 Where $\mathcal Q$ and the truth disagree, in full

W112 asks for every geometry where $\mathcal Q$ and the measured defect disagree in ranking. Two readings, because they are different failures — the **sign** (the criterion systematically prefers worse cuts) and the **argmin** (it correlates positively and still picks badly, which is what a search acts on):

| geometry | pools $\times$ criteria | sign disagreements | argmin disagreements |
|---|---|---|---|
| **N=6** | $32$ | $\mathbf{4}$ — all four operator criteria, all in the HALO pool, all at exactly $-1.000$ | $15$ |
| **N=12** | $32$ | $0$ | $14$ |
| **SOLO** | $40$ | $0$ | $9$ |

Every row is in `out/w112/w112.json` under `disagreements`. The pattern is that **no criterion here disagrees in sign about placement and none of them gets the argmin right everywhere**, and the four that invert do so on the one axis that is not placement.

## 5. The controls

Four, each of which would have invalidated the run.

- **ZERO.** The one-window candidate has no interface: $\chi\equiv1$, cutting and assembling are the identity, and the composed defect is **exactly `0.0`**, not to a tolerance. [[tier0-measurements]] §8's lesson and CS-7's $N=1$ rung.
- **REPRODUCE.** `from_array_tiling(ArrayTiling())` gives boxes identical to `wake_array`'s, weights differing by **exactly $0.0$**, a partition-of-unity residual of $1.11\times10^{-16}$, and CS-7's published $11$ overlapping pairs. A search graded against a baseline it cannot express is graded against nothing.
- **NOISE.** The pipeline is deterministic, so a repeat is bitwise. The floor that bounds a *ranking* is the probe's own: halving the finite-difference step is an equally admissible reading of the same operator and moves $\mathcal Q$ by $\mathbf{1.404\times10^{-4}}$ at worst, $5.07\times10^{-5}$ in the mean — four orders below the spreads being ranked.
- **KNOWN.** The one-rotor state has a single wake along row $53$. Swept across the whole domain, a two-window y-cut's defect is $5.397\times10^{-7}$ nearest the wake and $1.674\times10^{-8}$ farthest from it — $\mathbf{32.25\times}$ — and **every criterion, $\mathcal Q$ included, picks row $216$**, the quietest available. No criterion places a seam on the wake.

Also on the record: **the L2/C2 identity closes** on every decomposition this file can build ($4.08\times10^{-12}$ relative on the hand-chosen tiling), the cellwise bound is violated by $2.22\times10^{-16}$, the $\chi$-weighted form is tight to $0.989$ and the max form is $179\times$ loose — [[tier0-measurements]] §10.1's $0.998$ and $0.0046$, reproduced on a different expert at a different size.

## 6. CS-9's shared domain — the two families want opposite cuts

[[case-study-thermal-strain-atlas-0.1]] splits one PDE on one domain into a conduction agent and an elasticity agent, and finds the thermal-strain coupling is a **bond and not a port** because the interface between the two families has **co-dimension zero**. So the family axis offers no $\Gamma$ to place, and **that is CS-9's own result restated rather than a limit of this file**. What remains placeable is a *spatial* cut of the shell — and because the domain is shared, that cut cuts both families.

The shell is heated by a Gaussian streak at $z=L_Z/2$ of width $W_\text{STREAK}=0.03$ m. All $785$ segmentations at one, two and three segments were scored exhaustively.

- **ZERO passes in both families.** One segment gives a conduction defect of $2.917\times10^{-11}$ and an elasticity defect of **exactly $0$**.
- **KNOWN passes, and sharply.** Distance from the streak versus the conduction defect ranks $\mathbf{-0.9988}$: cutting through the streak costs $4.642\times10^{-6}$ against $5.935\times10^{-9}$ far from it, a factor of $\mathbf{782}$.
- **And elasticity ranks $\mathbf{+0.9991}$ against the same distance.** Quasi-statics is a global elliptic solve with no local feature; its defect is *lowest* where conduction's is highest.

| | conduction's argmin | elasticity's argmin |
|---|---|---|
| cut | $z=3$ — far from the streak | $z=19,29$ — straddling it |
| cost to the *other* family | $3.60\times$ | $\mathbf{795.4\times}$ |

Over all $785$ candidates the two families' defects rank at $\mathbf{-0.0259}$ — essentially uncorrelated.

> **One $\Gamma$, two physics, opposite preferences, and a factor of $795$ between following one and following the other.** No scalarization of *one seam operator* can express that decision, in any basis, because the disagreement is not in the operator: it is between two operators that share a domain. This is the sharpest thing this case study found and it is the one that generalizes past wind farms.

## 7. What it cannot say

- **One expert family, one Reynolds number, two domains.** Everything in §3 is `reference.WindowNS` on a wake field.
- **The marched columns are not incompressible.** Both are `exposed` agents with no projection anywhere — CS-7 §5's fourth row, which is **W105**: stable, and not solving the equations. A production rollout applies one global projection after the blend, and **whether the placement advantage survives *that* is unmeasured**.
- **The search is not exhaustive and its net is on the record.** N=6 scored $193$ candidates of an infinite space; N=12's pool was restricted to the hand-chosen shape and its transpose. A search that did not look at $2\times6$ cannot report that $2\times6$ is not better.
- **`cut_shear` picks well and ranks badly.** It finds the exact best candidate on three of five pools at essentially zero cost, and its rank correlation is only $0.18$–$0.31$ on N=6 and N=12 — a good argmin and a poor ordering. It is also **blind to the overlap by construction**: the cut lines do not move when $h$ does, so it is constant on that axis and `spearman` correctly reports no ordering rather than a number.
- **Nothing here is wired into the compiler.** L2/C2 remains the rule; this is a measurement about criteria, and no verdict changed.

---

## See Also

- [[tier0-measurements]] — §21 is the full record; §10.1 and §10.2 are the derivation and the first falsification this page corrects the reading of
- [[interface-transfer-theory]] — §9's $\mathcal Q$, which this page tests on the axis it was proposed for and on one it was not
- [[gap-worklist]] — Tier 23: W112 closed, W123–W126 opened; W16/G5, W58 and W105 reframed
- [[poc1-retrospective-and-hybrid-roadmap]] — §4 proposes this case study and §4.1 states the three sequencing reasons; its premise is corrected in §1.1 above
- [[case-study-scaling-ladder-atlas-0.1]] — CS-7, whose $N=6$ and $N=12$ tilings are this study's baseline and whose §5 arrangement it marches
- [[case-study-thermal-strain-atlas-0.1]] — CS-9, whose co-dimension-zero finding is why §6 is a spatial question rather than a family one
- [[generalization-requirements]] — G5, whose slogan §3.1 measures for the first time
- [[master-error-bound]] — §4.1's overlapping branch, whose $\sigma$ has no $\beta$ in it, which is half of why $\mathcal Q$ inverts on the halo axis
- [[composition-error-theory]] — the assembly term the one-interval defect *is*
- [[case-study-ladder-to-f1]] — the ladder this is inserted into, between CS-9 and CS-10
- [[f1-pathmap-and-end-goal]] — the fourth end-goal capability, now represented once
- [[prior-art-and-novelty-atlas-0.1]] — the verdict table that finds no prior art for this one
