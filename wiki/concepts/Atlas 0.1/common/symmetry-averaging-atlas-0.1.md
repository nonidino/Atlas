# Symmetry Averaging — the composition layer supplying an invariance the expert lacks

**Type:** Concept page (folder: `Atlas 0.1/common/`)
**Related:** [[conservation-as-constraint-atlas-0.1]] · [[open-problems-atlas-0.1]] · [[expert-library-atlas-0.1]] · [[port-algebra-atlas-0.1]] · [[agent-definition-atlas-0.1]] · [[wind-farm-implementation-log]] · [[impl-wind-farm-guide]] · [[prior-art-and-novelty-atlas-0.1]] · [[f1-pathmap-and-end-goal]]
**Code:** `src/atlas/invariants/symmetry.py` · `CoupleConfig.symmetry` · `tests/atlas/invariants/test_symmetry.py`

> **The one-line version.** A frozen expert that does not respect a symmetry can be *made* to respect it, exactly, by averaging it over the symmetry group — no training, no access to weights, at a cost of one forward pass per group element.

---

## 1. The intuitive picture

Suppose you have a black box that predicts how a fluid evolves, and you have checked that the situation you are giving it is perfectly left–right symmetric. The answer must be symmetric too. Nothing in the physics distinguishes up from down.

The box does not know that. Ask it, and it returns something slightly lopsided — and if you feed the lopsided answer back in, the lopsidedness grows.

You cannot retrain the box. But you can ask it **twice**: once as posed, once with the picture flipped. Then flip the second answer back and average the two. Whatever the box's private preference for one side was, it appears with a plus sign in one answer and a minus sign in the other, and the average has none of it. The result is symmetric — not approximately, *identically*, because you built it out of a quantity and its own mirror image.

That is the whole idea. The interesting part is that it is a **framework** operation: the expert is untouched and unaware, and the guarantee is created entirely by how the expert is *called*. It is the clearest case so far of the composition layer giving the system a property that no individual frozen expert possesses.

---

## 2. The rigorous statement

Let $G$ be a finite group acting on the field space, and let $E$ be any operator (the frozen expert). Define the **Reynolds average**, or group average:

$$\tilde E(\mathbf u) \;=\; \frac{1}{|G|}\sum_{g\in G} g^{-1}\,E(g\,\mathbf u)$$

**Claim.** $\tilde E$ is exactly $G$-equivariant: $\tilde E(h\,\mathbf u) = h\,\tilde E(\mathbf u)$ for every $h\in G$.

**Proof.**

$$\tilde E(h\mathbf u) \;=\; \frac{1}{|G|}\sum_{g\in G} g^{-1}E(gh\,\mathbf u)$$

Substitute $g' = gh$. As $g$ ranges over $G$, so does $g'$ — this is precisely the statement that $G$ is **closed under composition**, and it is where the group axioms earn their place. Then $g^{-1} = h g'^{-1}$, so

$$\tilde E(h\mathbf u) \;=\; \frac{1}{|G|}\sum_{g'\in G} h\,g'^{-1}E(g'\mathbf u) \;=\; h\,\frac{1}{|G|}\sum_{g'\in G} g'^{-1}E(g'\mathbf u) \;=\; h\,\tilde E(\mathbf u) \qquad\blacksquare$$

Note what the proof does **not** require: nothing about $E$ at all. $E$ may be nonlinear, learned, discontinuous, or actively hostile to the symmetry. Equivariance of $\tilde E$ is a consequence of the sum's structure alone.

### 2.1 Why closure is load-bearing rather than decorative

If $G$ is merely a *set* of operations, the substitution $g'=gh$ does not permute the sum, and the conclusion fails. The result of averaging over a non-group is a **smoothed** operator that is not equivariant, and — the dangerous part — nothing in its output announces the difference. It looks like the real thing.

Concretely: $\{e, M_x, M_y\}$ is not a group, because $M_x\circ M_y$ is the point reflection, which is absent. Averaging over it produces a plausible-looking field with no guarantee attached.

The implementation therefore **verifies closure numerically at construction** — compose every pair on a random field, require the result to match some element to machine precision — rather than trusting the caller to have declared a group. This also catches the two errors a symbolic check would miss: an operation whose action on fields and on the frame velocity disagree, and one that reindexes correctly while getting a component's sign wrong. Both appear as a composition matching nothing.

### 2.2 The group elements are not permutations of samples

A reflection acts on a *vector* field, so it re-signs the component normal to the mirror:

$$\mathcal M_y:\quad u(x,y)\mapsto u(x,-y),\qquad v(x,y)\mapsto -v(x,-y)$$

An implementation that reindexes without re-signing produces a field with the right values in the right places that is the reflection of nothing — and it still passes an involution test, since applying it twice restores the original either way. The same transformation must also be applied to every *spatially constant* vector that travels with the field: the Galilean frame velocity and any uniform forcing. Because the frame is subtracted before the forward pass and added back after, a frame left pointing the old way cancels inside the expert and reappears in the output — invisible in the transformed input, present in the result.

### 2.3 What the average preserves, and what it cannot repair

Averaging is **linear**, so every linear property of the expert's output survives it:

| property | survives? | why |
|---|---|---|
| divergence-free output (C1) | yes | a family of divergence-free fields averages to one |
| the field's mean (W0's mean-flow handling) | yes | a family with a common mean averages to that mean |
| any linear conservation statement | yes | same argument |
| **accuracy** | **no** | the average is equivariant, not correct |

The last row is the honest limit. $\tilde E$ commits the same errors $E$ does; it merely commits them symmetrically. Group averaging repairs a *symmetry* defect and nothing else.

**[AI Inference]:** there is a weak accuracy argument in the other direction — if the expert's asymmetry is an uncorrelated per-call error, averaging $|G|$ evaluations reduces its variance by $\sqrt{|G|}$, the ordinary ensembling effect. This has not been measured and should not be claimed. What *is* measured is that the wind-farm checkpoint's asymmetry is deterministic and reproducible, which is the regime where ensembling buys least.

---

## 3. Why this belongs to Atlas rather than to a case study

[[conservation-as-constraint-atlas-0.1]]'s enforce-or-measure rule says a declared invariant is either enforced to a stated tolerance or measured and reported as unenforced — never quietly assumed. Symmetry averaging is the first **enforcement** mechanism in Atlas that is:

1. **exact**, not tolerance-bounded;
2. **general**, depending on no property of the expert;
3. **training-free**, using only forward calls; and
4. **cheap and predictable**, costing exactly $|G|$ evaluations.

Contrast with the momentum projection of **OP-1**, which is none of these — it needs a correction direction, and the one the wind-farm spec chose lies in the null space of the constraint it is meant to enforce. A group average cannot be degenerate in that way: it does not choose a direction, it sums over a group.

**[AI Inference]:** that contrast is the reusable lesson, and it suggests a design preference for the whole invariant layer — *prefer enforcement by averaging over a declared group to enforcement by projection along a chosen direction, wherever the invariant admits both forms*. Averaging is unconditionally well-posed; projection needs a conditioning argument that OP-1 shows is easy to get wrong and easy to fail to notice.

### 3.1 The Noether connection

A continuous symmetry and a conserved quantity are two views of one object. Atlas's conservation laws and its symmetries therefore belong in one layer, which is why `atlas/invariants/` holds this and is where OP-1's momentum projection should eventually move.

The correspondence is **not** exact here, and the difference is worth stating precisely: Noether's theorem applies to continuous symmetries of an action, whereas $\{e,\mathcal M_y\}$ is a *discrete* group and yields no conserved current. So enforcing the mirror does not by itself enforce a conservation law.

**[AI Inference]:** the structural claim that survives is weaker but still useful — a declared symmetry and a declared conserved quantity are both *constraints the composed system must satisfy and no single expert does*, and both should be subject to the same enforce-or-measure contract and reported in the same ledger. Whether a continuous group (translation, rotation) averaged this way would enforce the corresponding conserved quantity is an open and testable question, and would be a much stronger result than the discrete case.

---

## 4. Measured — wind farm, gate W7

The occasion for building this is **OP-6**: the frozen checkpoint ([[expert-library-atlas-0.1]], Poseidon-T) is not mirror-equivariant. One call on a *uniform inflow* — the most trivial mirror-symmetric field there is — returns $\max|u-\mathcal M u| = 2.42\times10^{-2}$, which is $2.4\times10^4$ times gate W7's $10^{-6}$ threshold, with no graph, ports, agents or tiling in the loop.

### 4.1 The operator, in isolation

| symmetric input, one call | raw expert | symmetrized |
|---|---|---|
| uniform inflow $u\equiv1$ | $3.59\times10^{-2}$ | $\mathbf{0}$ |
| wake deficit | $3.58\times10^{-2}$ | $\mathbf{0}$ |
| symmetric jet | $3.46\times10^{-2}$ | $\mathbf{0}$ |
| general equivariance $\|E(\mathcal M\mathbf u)-\mathcal M E(\mathbf u)\|$ | $3.73\times10^{-2}$ | $\mathbf{0}$ |
| accumulated over 8 calls | $1.02\times10^{-1}$ | $\mathbf{0}$ |

**Exactly zero, not machine precision.** On a lattice whose cell centres are symmetric about the axis — which the wind-farm domain satisfies as $\max|y_c + y_c^{\text{rev}}| = 0$ — the reflection *is* the array reversal, so the average is $\tfrac12(a + \operatorname{rev}(a))$, and floating-point addition is commutative. There is no residue to be small.

The mean is preserved to $10^{-15}$, confirming §2.3.

### 4.2 The coupled system

Per-window equivariance is enough for the *global* field only if the tiling is mirror-paired, which was verified first: $124$ tiles, $y$-centres $\{0,\pm0.7,\pm1.5,\pm2.1,\pm2.5,\pm3.5\}$, **zero mirror-unpaired tiles** in both the eight-agent and monolithic builds, every agent box symmetric or paired.

| gate W7, mirror residual (threshold $10^{-6}$) | $t=5$ | $t=10$ |
|---|---|---|
| baseline | $2.77\times10^{-2}$, **growing** | $7.42\times10^{-2}$ |
| **symmetrized** | $\mathbf{2.90\times10^{-7}}$, **flat from $t\approx1$** | *(measured; see log)* |

Roughly $10^5\times$ smaller, under the gate, and — the part that matters more than the ratio — **it plateaus instead of growing**. The baseline's defect was an accumulating per-call bias; removing the bias removes the accumulation.

### 4.3 The residual is not zero, and the reason is float32

The coupled system reads $\sim3\times10^{-7}$ rather than $0$. Instrumenting one macro-step, the asymmetry appears at the **first scatter** — not in the pressure solve, which slightly *reduces* it ($7.57\times10^{-7}\to5.66\times10^{-7}$), and not at the boundary condition.

Cause, measured directly: the expert runs in float32 ($\varepsilon = 1.19\times10^{-7}$) and its output depends on a window's **position within the batch**. The same window evaluated alone, at batch size 8, and at batch size 40 differs by $9.6\times10^{-7}$ to $1.5\times10^{-6}$ — about $11\varepsilon$ — while an identical batch repeated is bitwise identical. Mirror-paired tiles occupy different batch slots, so their results differ by a few float32 ulps, and that is the floor.

**This should be read as a caveat on the gate, not on the construction.** The construction is exact; the floor is arithmetic. But W7's $10^{-6}$ threshold sits only a factor of $\sim2$ above the raw lattice asymmetry of $5.7\times10^{-7}$, so the gate is being graded close to the precision of the expert it grades. Getting further down requires batch-invariant kernels or a float64 expert, neither of which is a composition-layer question.

### 4.4 Cost

Measured $40$ s/step against $24$ s/step, a factor of $\mathbf{1.7}$ on the full macro-step — the expert calls double exactly, and the rest of the step (pressure, assembly, fixed point) does not, so the system-level cost is below the operator-level $|G|=2$.

---

## 5. What this does and does not change

**Does:** W7 moves from *failing and unwinnable* to *passing*. One of four failing wind-farm gates is closed, and closed by framework work rather than by a better expert.

**Does not:** the physics is untouched. **OP-5** — the windowed decomposition carrying $66\%$ of the error — is exactly where it was, and W8, W9 and W10 fail for the reasons they did before. Symmetry averaging makes the system symmetric; it does not make it right.

**[AI Inference]:** the more durable result is the pattern rather than the gate. *An invariance the frozen expert provably lacks can be created by the composition layer, exactly, without training* is a direct positive answer to a piece of what [[f1-pathmap-and-end-goal]] calls F2 — and it is a claim the gates were not designed to elicit, since they ask whether the composed system matches truth rather than whether composition can *add* a guarantee. If the wind-farm case study reports one framework capability, this is the strongest candidate.

---

## See Also

- [[conservation-as-constraint-atlas-0.1]] — the enforce-or-measure contract this is the first exact enforcement under
- [[open-problems-atlas-0.1]] — OP-6, the defect this answers; OP-1, the projection it contrasts with
- [[expert-library-atlas-0.1]] — equivariance as an expert-selection criterion
- [[wind-farm-implementation-log]] — the measurements, in the context they were taken
- [[prior-art-and-novelty-atlas-0.1]] — where this sits against equivariant-network literature
