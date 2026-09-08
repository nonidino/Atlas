# The interaction horizon — replacing a 0/1 envelope with the sensitivity it stands for

**Type:** Concept page — measured result, **version-independent** (folder: `Atlas 0.1/common/`; no Atlas revision suffix, for the same reason as [[probed-dtn-coupling]])
**Status:** measured 2026-09-07, Tier 33. Artifact `out/w153/w153.json`, driver `scripts/w153_influence_envelope.py`, suite `tests/test_tier33_interaction_horizon.py`. **One measurement and its controls**, in the shape [[control-observability]] used: no optimiser, no `compiler.py` change, no `Accelerator` member. Closes **W60** and **W150**; opens **W153**–**W156**.
**Related:** [[master-error-bound]] · [[control-observability]] · [[probed-dtn-coupling]] · [[atlas-and-standard-dd-theory]] · [[gap-worklist]] · [[tier0-measurements]] · [[poc2-demo-and-novelty]] · [[composition-error-theory]] · [[schwarz-iteration-atlas-0.1]] · [[port-algebra-atlas-0.1]] · [[expert-library-atlas-0.1]] · [[interface-transfer-theory]]

> **The one-line version.** $\Pi$ is not a quantity — it is a **0/1 envelope standing in for a sensitivity nobody had measured**, and at one exchange per macro-step it reads $\Pi = 1.0000$ for *every* agent in this vault, which is the certificate being unavailable rather than a fact about any of them. Replace the indicator with the thing it stands for and the same three agents separate cleanly: $\Pi_w = \mathbf{0.186}$ for the local solver, $\mathbf{0.496}$ for the same solver carrying its own elliptic part, $\mathbf{0.790}$ for Poseidon-T. **The bound is non-vacuous for a frozen neural operator for the first time.** The instrument reduces exactly — the local agent's influence is *bitwise zero* at $b = 20$ cells, its declared $\rho s$, to the cell — and $C_\mu$ re-fitted against $\Pi_w$ on the sixteen configurations it was originally fitted on comes back **tighter**, spread $3.71\times$ against the indicator's $5.18\times$. **And the rank-one derivation this was built on is false**: the non-decaying part of an embedded elliptic solve has effective rank $3.44$, not $1$, so pinning one scalar buys $4.7\%$ and not a collapse.

---

# 0. The confound Tier 32 left open, settled first

[[control-observability]] measured that minimising a jump objective drives $\sigma$ **up** by $6.6\times$ to $21\times$. The obvious objection is that the venue was unwinnable: if that was measured on an R10-compliant column at the split-step cadence, the baseline is $\sigma = 3.73\times10^{-8}$ and nothing could have beaten it.

**Both figures came from the exposed (R10-admissible) column — and its venue was $\mathbf{14.8\times}$ *looser* than the scheme it names, not tighter.**

| column | $\sigma_{\text{lagged}}$ | against the published value | solved / lagged |
|---|---|---|---|
| exposed, declared pair | $5.5365\times10^{-7}$ | $14.8\times$ the split-step $3.7334\times10^{-8}$ | $6.58\times$ |
| exposed, full ring | $5.5365\times10^{-7}$ | $14.8\times$ | $21.14\times$ |
| embedded | $2.2640\times10^{-5}$ | $\mathbf{1.005\times}$ the as-built $2.2532\times10^{-5}$ | $5.19\times$ |

The exposed column ran at one exchange per macro-step rather than per sub-step (**R10b**), which inflates its lag baseline by an order of magnitude. **So virtual control was given $14.8\times$ more room than the real scheme leaves it, and still lost by $6.6$–$21\times$.** The confound is real, it is now named, and it cuts the other way: the negative result is *understated* by its venue, not manufactured by it.

The embedded row is a free validation of the $\sigma$ instrument — it reproduces the vault's published as-built $\sigma$ to $0.5\%$ without being told it.

---

# 1. $\Pi$ is an envelope, not a quantity

[[master-error-bound]] §4.1 bounds the assembled error cell by cell:

$$\bigl\lvert\text{error}(j)\bigr\rvert \;\le\; \sum_i \chi_{ij}\,R_i(j)\,\lVert\delta\lambda\rVert, \qquad R_i(j) \;:=\; \Bigl\lVert \frac{\partial\,\mathcal E_i(j)}{\partial\,g_i} \Bigr\rVert$$

and then substitutes $R_i(j) \le C_\mu\,\mathbf 1[\,b_i(j)\le d_i\,]$, giving $\Pi = \max_j\sum_i\chi_{ij}\mathbf 1[\cdot]$. **That indicator is tight for a local explicit agent and catastrophically loose for influence that is dense but decaying** — which is the only kind a learned operator has. Replace it with the thing it stands for:

$$\boxed{\;\Pi_w \;:=\; \max_j\;\sum_i \chi_{ij}\,\widehat R_i(j),\qquad \widehat R_i := R_i/\max_j R_i(j)\;}$$

$\Pi_w\le\Pi$ always; it reduces to $\Pi$ **exactly** under a hard cutoff, so no past verdict can move silently; and it keeps its teeth, because influence that genuinely does not decay gives $\widehat R\approx1$ and $\Pi_w\to1$ with the refusal standing and now measured.

> **This is not a new bound.** It is the same bound with the indicator replaced by the sensitivity the indicator was standing in for, and it has to be quoted that way or §2's sixteen-configuration control is not a control.

$\Pi_w$ is computed **cell-wise** on the assembled grid, from the same $\chi$ in the same call as $\Pi$, so the comparison is a swap of one factor rather than two measurements. The profiles reported below are a *diagnostic* on top of it and never enter the number.

---

# 2. Gate A — $C_\mu$ re-derived against $\Pi_w$, on the rows it was fitted on

$C_\mu = 1.2$ was fitted **with** the indicator, over the configurations of [[master-error-bound]] §4.1.1 — halo $1$ to $61$, five partitions of unity, plus a Reynolds and macro-step sweep, with $\sigma$ spanning $4.28\times10^{4}$. Swapping the envelope changes what $C_\mu$ means, so it is re-derived on those same rows.

**$\sigma$ is re-measured with the same harness rather than transcribed, and it reproduces `out/l6/w49_sigma.json` to $0.00\times10^{0}$ on all ten recorded rows** — bit-identical, which turns the record from an unchecked input into a positive control on the harness.

| | $\min$ | $\max$ | **spread** |
|---|---|---|---|
| $C_\mu$ with the indicator $\Pi$ | $0.2275$ | $1.1782$ | $5.18\times$ |
| $C_\mu$ with $\Pi_w$ | $0.8024$ | $2.9754$ | $\mathbf{3.71\times}$ |

$\Pi_w/\Pi \in [0.1683,\ 1.0000]$ over the sweep, and at **halo 1** — where the overlap is narrower than the domain of dependence, so nothing decays inside it — the ratio is $\mathbf{1.000000}$. **The refinement returns the indicator exactly where the indicator is right**, which is the property that lets it be adopted without re-auditing any past verdict.

At the sampled maximum the bound is $2.24\times$ tighter ($1.1782\,\Pi$ against $2.9754\,\Pi_w = 0.5266\,\Pi$ at the working partition). **A constant that moves $3.71\times$ while the quantity it relates moves $4.28\times10^{4}$ is behaving more like a constant than one that moves $5.18\times$**, and that — not the factor of two — is what the gate was asking.

**Gate A passes.**

---

# 3. Gate B — the three-way decay control

The influence Jacobian is assembled over the *whole window*, one column per control mode, and $\widehat R_i(j)$ is its row norm at each cell. $d_{\text{eff}}(\theta) := \min\{r : \widehat R(r)\le\theta\}$ is the **interaction horizon**.

| | $\widehat R$ at $b=0$ | $b=5$ | $b=10$ | $b=20$ | $b=40$ | $b=60$ | $d_{\text{eff}}(10^{-3})$ | $d_{\text{eff}}(0)$ |
|---|---|---|---|---|---|---|---|---|
| `SpectralNS`, periodic | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ | $\mathbf 0$ |
| `WindowNS`, **exposed** | $1.000$ | $2.75\times10^{-2}$ | $3.06\times10^{-5}$ | $\mathbf 0$ | $0$ | $0$ | $8$ | $\mathbf{20}$ |
| `WindowNS`, **embedded** | $0.729$ | $0.536$ | $0.386$ | $0.261$ | $0.167$ | $0.122$ | — | — |
| **Poseidon-T** | $1.000$ | $0.702$ | $0.626$ | $0.562$ | $0.573$ | $0.611$ | — | — |

**The instrument reduces to the old rule exactly.** The periodic agent's influence is bitwise zero over all $13\,924$ far-field cells — the same positive control that returns $\Xi = 0$, now asked of the envelope. And the local explicit agent's influence is **bitwise zero at $b = 20$**, which is $\rho s = 2\times10$, its declared domain of dependence, **to the cell**. Not "approximately $20$": the far-field block has Frobenius norm exactly $0.0$. That is the single most important assertion in the tier and it holds in the strongest available form.

**The embedded column does not decay**, at any threshold tried down to $10^{-10}$, which is R10's *"flat in distance from the cut"* measured rather than derived. **R10 is right**, and the same solver with the elliptic part in the composition layer is the control that says so.

---

# 4. The rank-one claim, and why it is false

The derivation this tier was built on says: on a strip of width $L$ a boundary perturbation's $k$-th mode decays like $e^{-\pi k r/L}$, so every mode decays except $k=0$, and **the non-decaying part of an embedded elliptic solve is rank one — the gauge mode**.

It is not. Measured on the far field the local agent cannot reach at all ($b>20$, where the exposed column is bitwise zero, so the region is unambiguous):

| | effective rank | $\sigma_1/\sigma_2$ | energy in mode 1 | $\lvert\langle v_1,\text{const}\rangle\rvert$ | constant column's share of far energy |
|---|---|---|---|---|---|
| `WindowNS` embedded | $\mathbf{3.44}$ | $1.054$ | $0.382$ | $0.057$ | $52.9\%$ |
| `WindowNS` embedded, full ring | $3.45$ | $1.054$ | $0.382$ | $0.038$ | $52.7\%$ |
| **Poseidon-T** | $\mathbf{26.09}$ | $1.008$ | $0.054$ | $0.021$ | $7.1\%$ |

*(effective rank $= 1/\sum_k p_k^2$ with $p_k = \sigma_k^2/\sum\sigma^2$ — a rank that needs no threshold.)*

**The derivation is half right and the half that fails is the operative half.** The constant mode *is* the single largest contributor — $52.9\%$ of the far-field energy in one column of thirty-two — but it is not the whole of it, and $14$ of $32$ modes sit within $10\times$ of the largest far-field response.

Tested in the derivation's own terms — per mode, so that no basis mixing can mislead it — the far-to-near ratio on one face splits into **two families with different decay laws**:

| family | $k=1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ | $8$ |
|---|---|---|---|---|---|---|---|---|
| **even about the face midpoint** (cosine) | $0.168$ | $0.027$ | $0.0057$ | $0.0017$ | $0.0009$ | $0.0004$ | $0.0003$ | $0.0003$ |
| **odd about the face midpoint** (sine) | $0.385$ | $0.182$ | $0.115$ | $0.082$ | $0.064$ | $0.053$ | $0.045$ | — |

The cosine family decays like $e^{-ck}$ — that is the strip argument, working. The sine family decays like $0.385/k$, **algebraically**, and the strip argument does not predict it.

**[AI Inference]:** the strip derivation assumes one perturbed edge on a domain that is otherwise uniform in the transverse direction. This window has *two* artificial faces meeting at a corner and two real boundaries, so the transverse problem is not translation-invariant and the modes do not decouple into pure exponentials. Untested, and cheap to falsify: the same measurement on a window with a single artificial face should collapse the algebraic family. What is *measured*, and does not depend on the explanation, is that the non-decaying part is not rank one.

**The consequence is practical.** Pinning one scalar per governing family per exchange — the minimal constraint that annihilates a harmonic gauge mode, and emphatically **not R12**, which imposed a whole second elliptic answer and failed by over-constraining — takes $\Pi_w$ from $0.4963$ to $0.4728$. **It buys $4.7\%$**, and the gap to the exposed column goes from $2.67\times$ to $2.54\times$. A repair, not a collapse.

---

# 5. Gate C — Poseidon-T's horizon, and W60 closes

**Poseidon-T has no finite interaction horizon.** $\widehat R$ is $0.562$ at $b=20$ and $0.611$ at $b=60$; $d_{\text{eff}}(\theta)$ is undefined at every $\theta$ from $10^{-1}$ down to $10^{-10}$. The far-field effective rank is $\mathbf{26.09}$ of $32$ control modes, with normalised singular values $1.000/0.992/0.986/0.962/0.948/0.939$ — a spectrum that is essentially flat — and **all $32$ modes sit within $10\times$ of the largest far-field response**. Stable across three probe states ($26.69$, $26.40$, $26.09$) separated by $10.6\%$.

**One qualification, because the two statistics say different things.** The profile above is the *maximum* over each shell, which is what a bound built on a max needs. The **mean** over the same shells falls from $0.593$ at $b=0$ to $0.067$ at $b=20$ and stays there — a factor of $9$. So Poseidon-T's influence is **sparse but unbounded in reach**: most far cells barely respond, and some respond as strongly as the near field. That is a receptive-field structure rather than a diffusive one, it is why $\Pi_w = 0.790$ rather than $\approx 1$, and it is the shape a tail charge (**W155**) would have to be written against. $\Pi_w$ itself is computed cell-wise and never touches either statistic.

W60 asked whether `elliptic_subsolve` is declarable for a learned operator, and it stayed open for many tiers because `poseidon.elliptic_signature` measures **non-normality** and was being read as measuring **globality** — two things that come apart for a self-adjoint operator. A decay profile measures globality directly and cannot return inconclusive.

> **W60 closes, and the answer is that the field is the wrong question.** The measurement says Poseidon-T's response is **global**: no horizon at any threshold, so `EllipticSubsolve.NONE` is contradicted and the refusal stands, now measured instead of assumed. But the *structure* of that globality is not elliptic. An embedded elliptic solve is gauge-dominated — effective rank $3.44$, $52.9\%$ of the far field in the constant mode. Poseidon-T is not — effective rank $26.09$, $7.1\%$ in the constant mode. **The two agents are both `global` and they are global by different mechanisms, and `EllipticSubsolve` has one symbol for both.** What the rules actually consume is *reach*, and reach is now directly measurable; what the field *names* is mechanism, which is what nobody could declare. **W154** carries the separation.

Three regimes, separated by one instrument, in one table:

| agent | horizon $d_{\text{eff}}(0)$ | far-field effective rank | constant-mode share | $\Pi_w$ |
|---|---|---|---|---|
| local explicit (`WindowNS` exposed) | $\mathbf{20 = \rho s}$, exactly | $0$ (bitwise) | — | $\mathbf{0.186}$ |
| embedded elliptic (`WindowNS` as-built) | unbounded | $3.44$ | $52.9\%$ | $0.496$ |
| learned operator (**Poseidon-T**) | unbounded | $26.09$ | $7.1\%$ | $\mathbf{0.790}$ |

**And the certificate is where the tier's value is.** At one exchange per macro-step, $d = \rho s = 20$ cells against a $21$-cell overlap, so $\Pi = \mathbf{1.0000}$ for **all three** — the indicator cannot tell them apart at all, which is exactly the "certificate becoming unavailable" [[control-observability]] §1 named. $\Pi_w$ ranks them $0.186 / 0.496 / 0.790$, in the order the physics says. **The bound is non-vacuous for a frozen neural operator for the first time in this vault** — and $0.790$ is a $1.27\times$ tightening, which is a real number and a small one.

---

# 6. W150 closes, as a negative result rather than a blocked row

[[control-observability]] left `Accelerator = VIRTUAL_CONTROL` open, blocked on "an objective whose minimiser is the truth". It is now closed, because there is a reason the minimiser is not the truth and the reason is structural.

Write each agent as exact-plus-defect, $\mathcal E_i = \mathcal S_i + e_i$, and linearise the jump about the true datum $g^\star$:

$$\text{jump}(g) \;=\; T\,(g-g^\star) \;+\; (e_i - e_j) \;+\; \text{h.o.t.}$$

The exact parts cancel at $g^\star$ by construction; **the defects do not**. So

$$\arg\min_g \lVert\text{jump}\rVert^2 \;=\; g^\star \;-\; T^{+}(e_i-e_j)$$

The optimiser does not find the true datum. It finds **the datum at which the two agents' errors cancel on the overlap** — and making two biased models agree means moving both further from the truth in a correlated direction. $J$ measures *consistency*; $\sigma$ measures *correctness*; for inexact agents those have different minimisers and $\lVert\Delta_e\rVert$ is the distance between them.

Three things this predicts, all of them already measured:

- **Every regulariser fails the same way.** Tikhonov, truncated SVD and step-length all shrink toward $g_{\text{lag}}$, not toward $g^\star$, so the whole regularisation path is a path back to the baseline. [[control-observability]] §5 measured exactly that across ten decades of $\eta$, six retained ranks and seven step lengths.
- **A richer control space makes it worse, not better.** More room to absorb $\Delta_e$. **Predicted before it was looked up, and confirmed**: widening from the declared conjugate pair to the full ring takes the controllable fraction from $0.36$ to $0.95$ and the pathology from $6.58\times$ to $\mathbf{21.14\times}$ — $3.21\times$ worse — with the overshoot doubling from $17.7\times$ to $36.8\times$.
- **The scheme comparison has a sign.** $\sigma_{\text{lag}} \le C_\mu \Pi \lVert\delta\lambda_{\text{lag}}\rVert$ against $\sigma_{\text{ctrl}} \lesssim C_\mu \lVert\Delta_e\rVert/\beta_{\text{ctrl}}$: **virtual control wins iff the agents are accurate relative to the lag.** And $\tau \gg \varepsilon$ is this project's founding premise about frozen operators.

> **So virtual control is structurally wrong for exactly the class Atlas is built around.** That is a transferable negative result about output-matching coupling of biased surrogates — it applies to any scheme whose objective is agreement between inexact models — and not a row waiting for effort. **W150 closes `refused`, with the reason recorded.**

---

# 7. What this page may not claim

1. **$\Pi_w$ does not make neural operators composable.** It sharpens an envelope. $\tau$ is untouched, $\Sigma$ still has no $\tau$ axis, and nothing here moves an agent's own infidelity.
2. **A finite horizon would not certify Poseidon-T, and it does not have one anyway.** $\Pi_w = 0.790$ makes the bound non-vacuous; the seam stays `admit-uncertified` with $\tau$ unmeasured and unmeasurable — there is no same-class monolith and there cannot be one.
3. **$\Pi_w$ is not a new bound.** It is the same bound with the indicator replaced by the sensitivity the indicator was standing in for, and the sixteen-configuration re-fit is only a control because that is true.
4. **R10 is right.** The claim under test was that its non-decaying part is rank one, which is narrower and sharper than the rule assumes — and Gate B killed it. The rule's own reasoning about harmonic error is untouched and is now measured.
5. **Nothing here is faster.** An influence Jacobian is $m+1$ marches per window per exchange interval — the same price as a probe, paid again — and this tier measures an envelope rather than accelerating anything.

---

# 8. Scope

One seam family, one governing family, two classical agents and one checkpoint, one tiling per agent. $\Pi_w$ is measured at one exchange per macro-step for the cross-agent table, because that is Poseidon-T's only cadence, and at one sub-step for Gate A, because that is the cadence the recorded $\sigma$ rows were measured at — the two are labelled everywhere and never mixed. The horizon is a property of a state: the three probe states here are on one spin-up of one configuration, separated by $10.6\%$, at cell Reynolds numbers $4.24$–$4.83$ against a declared envelope of $8$. Nothing establishes that effective rank $26$ travels to another checkpoint, and the algebraic mode family is explained by an **[AI Inference]** that has not been tested.
