# The Substitution-Campaign Checkpoint — standing on it, after CS-12

**Type:** Concept page — **declaration**, made against a measurement (folder: `Atlas 0.1/common/`)
**Status:** declared 2026-09-10, Tier 40. `scripts/w177_campaign_census.py`, `scripts/w174_refusal_validation.py`, `out/w177/w177.json`, `out/w174/w174.json`, `tests/test_tier40_epsilon_halo.py`. Worklist rows **W174**, **W175**, **W177**.
**Related:** [[case-study-ladder-to-f1]] · [[f1-pathmap-and-end-goal]] · [[poc2-novelty-audit]] · [[epsilon-halo-measurement]] · [[gap-worklist]] · [[expert-donor-survey]] · [[case-study-neural-interface-atlas-0.1]] · [[prior-art-and-novelty-atlas-0.1]]

---

# 0. Why this page exists, and what it is not

[[case-study-ladder-to-f1]] §7 names a stopping rule the pathmap does not have, and names when to stand on it:

> If the substitution campaign returns `refuse` or `blind` at **every** seam of **every** case study, the composed model cannot be certified with learned experts at all, and the honest outcome is the framework rather than the foundation model. … it is worth naming a checkpoint at which it is declared rather than approached asymptotically: **after CS-12**.

CS-12 is built. CS-13, CS-14 and CS-S1 are built on top of it. The checkpoint arrived and nobody had stood on it, which is the failure mode the rule was written to prevent — *approached asymptotically* rather than declared.

**This is a declaration, not a case study.** It runs no new physics. What it does is **count**, because the figure the programme has been reasoning with turned out to be quoted without its condition.

---

# 1. The declaration, up front

> **The stopping rule does not fire, and the reason it does not fire is not encouraging in the way it sounds.**
>
> The campaign's own instrument returns an informative, probe-resolvable **`admit`** at three of the eight agent-sides of the graph that actually holds the live checkpoint — on an interval of declared risk tolerance $0.2\%$ to $1.1\%$ of $\beta$ wide. So *"refuse or blind at every seam"* is **false**, measured, and the honest outcome is **not** the framework-only one on §7's own terms.
>
> **And what those admits admit is measured too, and it is not a faithful expert.** The checkpoint's boundary response is $0.8\%$ to $9\%$ of the classical incumbent's, and $\lVert\Delta\rVert/\lVert S_i\rVert$ is $0.95$–$1.01$ on **every** side — so swapping it in is, to within $5\%$, the same as **deleting the block**. That is W76's null replacement, measured rather than supposed. The seam stays invertible because the block contributes almost nothing, and the certificate cannot tell a faithful replacement from an absent one when that is true.
>
> **And nothing in the framework can locate that interval.** $\beta_{\min}$ derives from $\varepsilon_{\text{tol}}=\min(\tau,\sigma)$, and both are *structurally unmeasurable* for a checkpoint fixed at one resolution — there is no same-class monolith to be a reference pair. The window exists and there is no way to know you are in it.
>
> So what stands between this programme and an admitted learned expert at these seams is **not `L2/R10` and not the receptive field**. It is a missing reference pair.

Against the three branches the checkpoint had to choose between:

| branch | verdict |
|---|---|
| **(a) a foundation model with a constrained expert class** | **the live route, unattempted.** Not refuted by anything measured. [[expert-donor-survey]]'s fourth finding names three bounded-receptive-field donors and **none has been probed** |
| **(b) a foundation model pending a theory branch** | **closed.** The theory branch the ladder named is the $\varepsilon$-halo, and [[epsilon-halo-measurement]] measured it as not derivable — two checkpoints, both directions, with a positive control. No other branch is named |
| **(c) a coupling framework plus classical verification** | **what has been demonstrated**, and it remains true whatever happens to (a). [[f1-pathmap-and-end-goal]] §1.2's deployment story is unchanged and is the only one with a certified artifact behind it |

**The programme is (c) as demonstrated, (a) as the one live route, (b) closed.** Picking a single letter would be the false precision this page exists to avoid.

---

# 2. The census — because the number in circulation is quoted without its condition

**"Nine of nine seams have refused Poseidon-T on R10."** That sentence appears in [[gap-worklist]] W166, in [[log]]'s Tier 33 entry, and in both demo READMEs. Compiled rather than recalled:

| graph | fluid expert, as described | verdict | seams red / amber / green |
|---|---|---|---|
| `front_wing`, fixed shape | `WindowNS`, the incumbent | **`admit`** | $0$ / $0$ / $9$ |
| `front_wing`, fixed shape | Poseidon-T, as `poseidon_capabilities` declares it | **`admit-uncertified`** | $0$ / $0$ / $9$ |
| `front_wing`, fixed shape | Poseidon-T, `elliptic_subsolve=embedded` | `refuse` (`L2/R10`) | $0$ / $0$ / $9$ |
| `front_wing`, riding | `WindowNS`, the incumbent | `refuse` (`L2/InterfaceMotion`) | $0$ / $0$ / $9$ |
| `front_wing`, riding | Poseidon-T, as declared | `refuse` (`L2/InterfaceMotion`) | $0$ / $0$ / $9$ |
| `poseidon-t-2x2` (**the live weights**) | default (`elliptic_subsolve=none`) | **`admit-uncertified`** | $0$ / $4$ / $0$ |
| `poseidon-t-2x2` (**the live weights**) | `elliptic_subsolve=unknown` | **`admit-uncertified`** | $0$ / $4$ / $0$ |
| `poseidon-t-2x2` (**the live weights**) | `elliptic_subsolve=embedded` | `refuse` (`L2/R10`) | $0$ / $4$ / $0$ |

**Three corrections, and each of them matters to the stopping rule.**

1. **The nine is conditional on a declaration the vault itself records as undeclarable.** It is `front_wing` compiled with `elliptic_subsolve=EMBEDDED`, and the demo's own panel says so in terms — *"DECLARED, not measured. W60 says this field is undeclarable for a learned operator."* Under the record `poseidon_capabilities` actually builds, the same graph is `admit-uncertified` with **zero refusals**.
2. **`L2/R10` is a graph-level rule, not a per-seam one.** It emits **one** decision, whose subject is the six fluid **agents** — `F00, F10, F20, F01, F11, F21` — and no seam id appears as the subject of any refusal. The demo panel paints that one decision across all nine tiles so it can be seen, which is a defensible way to show a graph-level refusal on a seam display. **One rule firing once is not nine independent verdicts**, and reading it as nine is what made the campaign look more decided than it is.
3. **The riding refusal is not about the learned expert at all.** `L2/InterfaceMotion` refuses the *classical* incumbent on the same graph. Quoting a riding-shape refusal as a substitution result attributes to the checkpoint a refusal it shares with the solver it would replace.

**Compiled honestly, no seam of any graph refuses a learned expert. Every one is `admit-uncertified`.** And [[case-study-ladder-to-f1]] §7 says, in the paragraph immediately after the stopping rule: *"`admit-uncertified` is not failure."*

---

# 3. The campaign's own instrument, which is not the compiler

§5's campaign is four steps and step 3 is **`certify_substitution` at every seam at a declared $\beta_{\min}$**, step 4 **records `refuse` / `admit` / `blind` per seam**. A compile verdict is a different question and had been standing in for this one.

Run properly: the incumbent block is `window_ns.WindowAgent` with its elliptic part exposed — the arrangement `L2/R10` admits — and the replacement is `PoseidonAgent` on the same window at the same state, both reduced to the declared $16$-mode Fourier interface space. $\beta=\sigma_{\min}$ of the assembled two-sided block; $\lVert\Delta\rVert$ the swap; $\lVert S_i\rVert$ the block being replaced.

$\beta_{\min}$ is passed as **`None` on purpose**: W81 derives it from $\varepsilon_{\text{tol}}$, this graph has no $\varepsilon_{\text{tol}}$, and supplying a number would be importing another expert's constant to make a verdict appear. The certificate then reports the two thresholds that *would* make it a verdict:

$$\text{ADMIT} \iff \underbrace{\beta-\lVert S_i\rVert}_{\texttt{visible\_above}} \;\le\; \beta_{\min} \;<\; \underbrace{\beta-\lVert\Delta\rVert}_{\texttt{fails\_above}}$$

Below `visible_above` a pass carries no information — no replacement could have failed, which is `blind`. At or above `fails_above` the swap is refused.

| seam | side | $\lVert\Delta\rVert$ | $\beta$ | $\lVert\Delta\rVert/\beta$ | `visible_above` | `fails_above` | window | verdict function |
|---|---|---|---|---|---|---|---|---|
| sx0 | P00 | $0.3900$ | $0.3486$ | $\mathbf{1.119}$ | $-0.0440$ | $-0.0415$ | — | **`refuse` at every $\beta_{\min}\ge0$** |
| sx0 | P10 | $0.0710$ | $0.3486$ | $0.204$ | $0.2736$ | $0.2775$ | $\mathbf{0.0040}$ | blind → **`admit`** → refuse |
| sx1 | P01 | $0.3859$ | $0.3769$ | $\mathbf{1.024}$ | $-0.0116$ | $-0.0091$ | — | **`refuse` at every $\beta_{\min}\ge0$** |
| sx1 | P11 | $0.0725$ | $0.3769$ | $0.192$ | $0.3020$ | $0.3044$ | $\mathbf{0.0024}$ | blind → **`admit`** → refuse |
| sy0 | P00 | $0.1970$ | $0.2728$ | $0.722$ | $0.0783$ | $0.0758$ | $-0.0026$ | blind → refuse, **no admit anywhere** |
| sy0 | P01 | $0.1885$ | $0.2728$ | $0.691$ | $0.0837$ | $0.0843$ | $\mathbf{0.00064}$ | blind → **`admit`** → refuse |
| sy1 | P10 | $0.1909$ | $0.2887$ | $0.661$ | $0.0996$ | $0.0978$ | $-0.0018$ | blind → refuse, **no admit anywhere** |
| sy1 | P11 | $0.2035$ | $0.2887$ | $0.705$ | $0.0865$ | $0.0852$ | $-0.0013$ | blind → refuse, **no admit anywhere** |

**Three of eight sides admit informatively, on windows $0.2\%$–$1.1\%$ of $\beta$ wide.** Two refuse at every tolerance, saturated with $\lVert\Delta\rVert/\beta$ above $1$ — which is [[case-study-reuse-probe-atlas-0.1]]'s W109 reading arriving on a second, unrelated graph. Three are blind then refusing with no admit at any tolerance.

## 3.1 And the two columns that say what those admits are worth

W76 gives the bound the certificate's blind spot rests on: *"the largest move a swap of agent $i$ can make is bounded by that agent's own block — a replacement that ignores its boundary data entirely removes $S_i$ and nothing else, giving $\lVert\Delta\rVert = \lVert S_i\rVert$."* That is a **hypothetical** in the docstring. Measured here, on all eight sides:

| | sx0 P00 | sx0 P10 | sx1 P01 | sx1 P11 | sy0 P00 | sy0 P01 | sy1 P10 | sy1 P11 |
|---|---|---|---|---|---|---|---|---|
| $\lVert\Delta\rVert/\lVert S_i\rVert$ | $0.994$ | $0.947$ | $0.993$ | $0.969$ | $1.013$ | $0.997$ | $1.009$ | $1.007$ |
| $\lVert S_{\text{new}}\rVert/\lVert S_{\text{old}}\rVert$ | $0.0087$ | $0.0900$ | $0.0094$ | $0.0752$ | $0.0501$ | $0.0204$ | $0.0275$ | $0.0300$ |

**The checkpoint's boundary response is $0.8\%$ to $9\%$ of the classical incumbent's, and swapping it in is — to within $5\%$ — the same as deleting the block.** $\lVert\Delta\rVert/\lVert S_i\rVert$ sits at $0.95$–$1.01$ on every side, which is W76's null replacement measured rather than supposed.

This does not make the three admits wrong; the certificate asks *does the swap preserve $\beta$*, and it does, for exactly the reason the numbers give — **the block being replaced contributes so little that even removing it leaves the seam problem invertible with margin.** But it settles what is being admitted. It is not that Poseidon reproduces the classical Steklov–Poincaré operator well enough to substitute for it. It is that **at this seam Poseidon barely responds to its boundary data at all**, and the certificate is blind to the difference between a faithful replacement and an absent one whenever the block is small against $\beta$.

The same fact is in [[epsilon-halo-measurement]] §4.1's first column from the other side: the checkpoint's face operator has $\lVert S\rVert_2 = 0.0048$ against `WindowNS`'s $0.393$, a factor of $82$. A one-shot map at a coarse lead time does not propagate a ring perturbation — which is [[noether-1.0-rbc]]'s **smoothed-near-copy** signature arriving as a seam measurement instead of a rollout one.

> **And it explains why the windows are narrow.** An informative `admit` needs $\lVert\Delta\rVert < \beta-\beta_{\min} \le \lVert S_i\rVert$, so the band's width is exactly $\lVert S_i\rVert - \lVert\Delta\rVert$. With $\lVert\Delta\rVert/\lVert S_i\rVert \approx 1$ that width is nearly zero **by construction**, not by bad luck in the tolerance. The knife edge is the null-replacement bound pressing against the failure bound.

## 3.2 The control that had to be run, and it came back the other way

The windows are $6\times10^{-4}$ to $4\times10^{-3}$ wide. The probe that produces them is a finite difference at $\varepsilon=10^{-4}$, and [[epsilon-halo-measurement]] §5.3 measured this checkpoint's differencing floor rising as $1/\varepsilon$ — so the obvious reading is that the window is inside the instrument's error bar and the `admit` is noise straddling a threshold.

**It is not.** Repeat the entire construction at $\varepsilon=10^{-3}$ and compare where the endpoints land:

| seam·side | window width | endpoint shift over $10\times$ in $\varepsilon$ | shift / width | resolvable |
|---|---|---|---|---|
| sx0 · P10 | $3.96\times10^{-3}$ | $2.13\times10^{-4}$ | $5.4\%$ | **yes** |
| sx1 · P11 | $2.35\times10^{-3}$ | $1.95\times10^{-4}$ | $8.3\%$ | **yes** |
| sy0 · P01 | $6.45\times10^{-4}$ | $2.22\times10^{-4}$ | $34\%$ | **yes, and only just** |
| the other five | $\le 0$ | $3.6\times10^{-5}$–$2.2\times10^{-4}$ | — | no window to resolve |

The Fourier probe is far steadier than the delta probe of [[epsilon-halo-measurement]] §5.3, and for a reason worth writing down: a Fourier mode spreads the same $\varepsilon$ over all $128$ cells, so the *response* is a global sum where a delta poke's is a local difference. **The same checkpoint is noise-dominated at $10^{-4}$ under one probe and stable to $5\%$ of a $1\%$ window under the other.** That is a fact about the probe basis, not about the weights, and it is the reason a control that looked like a formality was worth the $78$ seconds.

## 3.3 What the three admits are and are not

They **are**: the first informative `admit` from **`certify_substitution`** for a live pretrained neural operator replacing a classical solver at a real seam, with the alternative — a blind pass — separated from it by a measurement rather than assumed away.

> **The word "first" is checked and it is narrower than it sounds.** Every `verdict` key in every artifact under `out/` was swept. Poseidon-T does carry six `admit` rows already, in `out/w33/w33.json` — and they are **`run_conformance`** field checks (schema, nondim, `bc_channel`), a different instrument answering *is the record internally consistent*, whose own overall verdict for that expert is `admit-uncertified`. CS-8's `out/w106/w106.json` has **no** `admit` in $60$ certificate readings, which is W109's *refuse at all $55$ cells and every admissible tolerance* seen from the artifact side. So: no prior artifact records a substitution certificate admitting a learned expert, and *that* is the claim.

They are **not** a certified expert. The window is located by $\beta_{\min}$; $\beta_{\min}$ comes from $\varepsilon_{\text{tol}}=\min(\tau,\sigma)$; and on this graph both are unmeasurable rather than unmeasured, because the checkpoint is fixed at $128\times128$ and has **no same-class monolith at any resolution** to be a reference pair. A window $1\%$ of $\beta$ wide, with no instrument that points at it, is not something a design can be certified on.

> **This is the finding that should change what gets worked on.** Every response to the campaign for nine tiers has attacked the **receptive field** — a bigger halo, an $\varepsilon$-halo, moving the operator to the interface. At the certificate, the receptive field is not what binds. What binds is $\tau$ and $\sigma$, and **W95 already said so** — *"no referent from a checkpoint alone"* — and was closed *by classical-first* rather than answered. Classical-first routed around it for the ladder and left it standing for the campaign.

---

# 4. Does the checkpoint clear the audit's third clause?

[[poc2-novelty-audit]] §4's sharpest open point, restated by Tier 39 and left standing:

> Nobody has yet exhibited **a seam the compiler refused that would in fact have diverged**, or **a seam it admitted that held**. Until one of those exists, "it judges correctly" is a statement about internal consistency, not about the world.

**Partly, and now measurably — and the answer is two-sided.**

`L2/R10/halo` is the one rule in the package with a numeric threshold on one declared integer: the overlap must cover $\rho s$, which is $20$ for `WindowNS`. CS-S1 built a sweep that varies exactly that integer and marches the coupling to its fixed point at each value. So the confusion table can be filled in — compile at each halo, march at each halo, cross-tabulate.

**CS-S1 swept two of the four cells and they differ in *two* variables at once**: its *"as-built"* is (embedded elliptic, **no** global projection) and its *"split-step"* is (exposed elliptic, **with** projection). A bracket measured across a two-variable change cannot attribute the divergence to either. The two missing cells were run here.

| arrangement | halo $4$–$18$ | halo $20$–$48$ | rule's verdict | tally |
|---|---|---|---|---|
| **embedded, no projection** (CS-S1's *as-built*) | contraction $1.565 \to 1.006$, **diverges at all five** | $0.981 \to 0.701$, converges at all six | refuse below $20$, admit at and above | **$11$ of $11$ correct** |
| **embedded, with projection** *(new)* | $0.984 \to 0.883$, **converges at all five** | $0.868\to0.663$, converges | same | $5$ over-fires, $6$ earned |
| exposed, with projection (CS-S1's *split-step*) | $0.089\to0.011$, converges | $0.011$, converges | same | $5$ over-fires, $6$ earned |
| **exposed, no projection** *(new)* | $0.195\to0.009$, converges | $0.009$, converges | same | $5$ over-fires, $6$ earned |

**The clause is cleared in one cell and not in the other three.** In (embedded, unprojected) the rule refuses at every halo where the coupling diverges and admits at every halo where it converges — $11$ of $11$, with the declared reach $20$ landing inside the measured bracket $(18, 20]$. **That is a refusal checked against a coupling that actually fails, and it is right.** In the other three the *same* refusal costs a coupling that converges at every halo from $4$ up.

**And the compiler cannot tell which cell it is in.** `project` is a parameter of the *exchange*, not of `neural_interface.build`; the graph is constructed identically either way, so all four cells compile to the same declarations and the same verdict. Confirmed by asking the constructor rather than arguing it: `project` is not in `build`'s signature.

## 4.1 Which makes a Tier 39 message over-attribute — W175

`_halo_bounds` branches on `elliptic_subsolve` $\in$ {`embedded`, `unknown`} and, on that branch, tells the reader the halo bounds *"ACCURACY **and** CONVERGENCE"*, closing with *"The same graph with the elliptic part EXPOSED converges at every halo from 4 up, which is R10's own repair showing up in this rule's threshold."*

Every factual claim in that message is true as stated — CS-S1's *as-built* **is** (embedded, unprojected). The **attribution** is over-broad by exactly one variable: restoring the global projection, with the elliptic part still embedded, also converges at every halo from $4$ up. So R10's repair is *a* repair and the message presents it as *the* one, and the branch will emit "accuracy and convergence" for an embedded-and-projected graph where convergence is not at stake.

**The declaration that would carry it already exists and is filled in wrongly.** `GlobalField.produced_by` has a documented meaning for the all-agents case — *"a global operation over the whole state, owned by the composition layer — a split-step pressure solve is the vault's four instances of this"* — and `neural_interface.build` declares `GlobalField("pressure", produced_by=all four agents)` **unconditionally**, including in the two cells where the composition layer does not project. This is not a missing schema. It is a mis-declaration plus a rule that does not read the declaration it would need. Opened as **W175**, not fixed here: a fifth rule narrowed in a session that already narrowed none is the pattern Tier 39 spent itself repairing.

## 4.2 What the clause still lacks

No **`admit`** has been checked against a coupling that held **and could have failed**. The (embedded, unprojected) column supplies the refusal half at $11$ of $11$; its admit half is real but weak evidence, because the coupling converges in three of the four cells regardless of the rule. The audit's *"a seam it admitted that held"* is satisfied only in the arrangement where an admission was at risk, and that is one column of one graph.

---

# 5. What the checkpoint changes, and what it does not

**Changes:**

- §7's stopping rule is **evaluated and does not fire.** It should not be re-approached asymptotically; it has been stood on and the answer is recorded here.
- The *"nine of nine"* figure is retired in favour of the census in §2. Three pages and two READMEs quote it; **W177** carries the correction.
- The campaign's binding constraint is re-identified: at the **compile** it is `L2/R10`'s halo, and at the **certificate** it is the absent reference pair. Those are different problems and only the first has had nine tiers of attention.
- One refusal is validated against a coupling that actually fails, $11$ of $11$, and the same refusal is shown to over-fire in three arrangements the compiler cannot distinguish (**W175**).

**Does not change:**

- **`admit-uncertified` is still not failure**, and six real case studies still land there with their unissued certificates named. [[case-study-ladder-to-f1]] §7's second paragraph stands untouched.
- **The deployment story is unchanged.** [[f1-pathmap-and-end-goal]] §1.2: search wide and cheap with the composed model, verify the shortlist with the classical stack. Nothing here makes that weaker and nothing here makes it unnecessary.
- **[[epsilon-halo-measurement]]'s negative is not softened by any of this.** Three of eight certificate-sides admitting is a statement about $\lVert\Delta\rVert$ against $\beta$ on the declared $16$-mode interface space. It says nothing about the halo, and the halo is separately measured as unbounded in both directions on two checkpoints.
- **`L2/R10` still decertifies every graph holding this checkpoint**, and it is right to: the halo is *undecidable*, which is a different thing from *inadequate*, and the compiler says so rather than assuming either.

---

# 6. What (a) would take, stated so it can be scheduled

Branch (a) — the vision on a constrained expert class — is the only live route, and it is not a hope: [[expert-donor-survey]]'s fourth cross-cutting finding already did the reading and found the locality verdict *"close to unanimous"*, with three exceptions:

| donor | what bounds it | what probing it costs |
|---|---|---|
| **MACE** | $r_{\text{cut}}\times$ layers, a hard finite radius | the existence proof the ladder's constrained-expert rung was waiting on, from outside continuum fluids |
| **DeepFlame**'s chemistry integrator | **pointwise** ($0$-D) — a trivially certifiable halo | splits an expert [[training-and-bootstrap-atlas-0.1]] had called from-scratch |
| **NeuberNet** | boundary displacement *is* the input; a boundary-driven patch | the one donor in the survey that takes boundary data as a declared argument |

`scripts/w173_epsilon_halo.py` runs against any object with a `boundary_response`, so the measurement that returned the negative here is the same one that would return the positive there, unchanged. **[AI Inference]:** on the survey's own reading all three should read compact and MACE should read $r_{\text{cut}}\times L$ exactly, the way `windowns-split-step` reads $13$–$20$ against a declared $20$. That is a prediction with a number in it and it has not been checked.

**And it does not lift the §3 constraint.** A bounded-receptive-field donor passes the halo rule; it still needs $\tau$ and $\sigma$ to be issued a certificate, and a fixed-resolution checkpoint still has no monolith. MACE and NeuberNet are not resolution-fixed the way scOT is, which is the reason to expect that branch to differ — and is itself unchecked.

---

# 7. The one-paragraph version

The stopping rule was written for a campaign that came back empty at every seam. It did not: at three of eight sides the certificate admits, informatively and resolvably, on a window of risk tolerance about one percent of $\beta$ wide that nothing in the framework can find. The $\varepsilon$-halo that was the named route past the receptive field is measured and is not derivable. The receptive field, meanwhile, turns out not to be what binds at the certificate — the missing reference pair is, and it has been open since W95 and was routed around rather than answered. So: **a coupling framework with a certified artifact, one validated refusal, a closed theory branch, an unattempted expert class, a constant nobody can measure standing between the campaign and its first real verdict — and three admissions of an expert that, at these seams, is within $5\%$ of not being there.** That is the state of it, and it is neither the vision nor the floor.

---

## See Also

- [[case-study-ladder-to-f1]] — §5's campaign and §7's stopping rule, the two things this page stands on
- [[epsilon-halo-measurement]] — branch (b), measured and closed
- [[poc2-novelty-audit]] — §4's third clause, partly cleared in §4 above
- [[f1-pathmap-and-end-goal]] — §1.2's deployment story, which (c) is
- [[expert-donor-survey]] — its three bounded donors (finding 4) and Open question 3, branch (a)'s subjects
- [[case-study-reuse-probe-atlas-0.1]] — W107 and W109, the saturated-ratio class §3 meets again
- [[gap-worklist]] — W174, W175, W177
