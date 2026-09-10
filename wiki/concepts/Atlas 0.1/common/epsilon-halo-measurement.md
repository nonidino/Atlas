# The $\varepsilon$-Halo, Measured — and it is not derivable for this expert class

**Type:** Concept page — **measured result**, negative (folder: `Atlas 0.1/common/`)
**Status:** measured 2026-09-10, Tier 40. `scripts/w173_epsilon_halo.py`, `scripts/w173_transverse_reach.py`, `scripts/w173_tables.py`, `out/w173/w173.json`, `out/w173/transverse.json`, `tests/test_tier40_epsilon_halo.py`. Worklist row **W173**.
**Related:** [[case-study-ladder-to-f1]] · [[gap-worklist]] · [[master-error-bound]] · [[expert-donor-survey]] · [[case-study-neural-interface-atlas-0.1]] · [[control-observability]] · [[substitution-campaign-checkpoint]] · [[probed-dtn-coupling]] · [[composition-error-theory]]

---

# 0. The one-sentence result

**The $\varepsilon$-halo does not exist for the scOT checkpoints, and the reason is not that the tail is large — it is that the tail stops shrinking.** Band-truncating the seam operator at radius $r$ leaves $27\%$ to $75\%$ of its spectral norm behind, and that fraction is flat from $r\approx16$ out to $r=64$ on a $128$-cell seam. Measured on **two** checkpoints, three faces, two base states and three amplitudes; with a **compact** classical operator on the same seam reading exactly zero past $r=13$, so the instrument can see a bounded halo when one is in front of it.

That is [[case-study-ladder-to-f1]] §5's **third** outcome — *"none pass and no rule is derivable"* — and §5 says to report it as a result rather than as a delay.

---

# 1. What was proposed, and why it was worth attacking

[[case-study-ladder-to-f1]] §5 marks one route through `L2/R10` as **[AI Inference]**, unbuilt, and says it is the one worth attacking first:

> `support_reach` currently asks whether the response is *nonzero* past the declared radius; the physically meaningful question is whether it is *larger than the defect the composition already tolerates*. An $\varepsilon$-halo — the radius beyond which the response falls below $\varepsilon_{\text{tol}}$, with the truncated tail carried as an explicit term in $\sigma$ — would be a bounded halo for an operator whose support is formally global.

The proposal is good and the reasoning behind it is sound. `L2/R10`'s halo rule asks a **binary** question — is the response nonzero past the declared radius — and a binary question about a smooth operator throws away everything except its support. Every real bound in [[master-error-bound]] is a magnitude, not a support, so replacing a support test by a magnitude test is the right *shape* of repair.

It has one empirical premise, and §5 names it: **whether the tail is summable is a question about the architecture, and it is measurable with the probe that exists.** This page answers it.

---

# 2. What has to be measured, and why one poke is not it

`probe.support_reach` pokes **one** seam cell and counts how many cells move. That is one *column* of the seam response, and it is the right instrument for *"is the support compact"* — it is what W93 used to read $64$ cells against a declared $2$.

A halo **declaration** is a different object. It says *cells further than $r$ apart on this seam do not talk*, which is a statement about the whole operator, so the quantity a rule would consume is the **band truncation** of the dense seam response:

$$S_{ij} \;=\; \frac{\bigl[R(\text{port},\, \lambda_0 + a\,e_j)\bigr]_i - \bigl[R(\text{port},\, \lambda_0)\bigr]_i}{a}, \qquad (S_r)_{ij} = S_{ij}\,\mathbf 1\bigl[\lvert i-j\rvert \le r\bigr]$$

$$E(r) \;:=\; \bigl\lVert S - S_r \bigr\rVert$$

built column by column in $n+1$ solves. Three reasons this and not the single poke:

1. **The seam is not translation-invariant.** It has corners and a varying base state, so a kernel that decays from the middle column need not decay from an edge one. Measuring every column is what the declaration would be about.
2. **$E(r)$ is exactly $\lVert\Lambda - \tilde\Lambda\rVert$** in [[master-error-bound]] §4's factorization. An $\varepsilon$-halo at radius $r$ is not a free approximation: it adds $C_\mu E(r)\lVert\lambda^\star\rVert/\beta$ to $\sigma$. That is the *"explicit term"* the proposal requires, and stating it as an operator norm is what makes it a term rather than a hope.
3. **An operator norm and a per-cell magnitude are different tests**, and §7 shows they give opposite answers here.

**Cost.** $129$ forward passes per configuration: $25$ s for Poseidon-T, $215$ s for Poseidon-B, $2$–$7$ s for the classical solvers. The whole sweep is $1035$ s. This measurement was affordable for the whole of the programme's life and nobody had run it.

---

# 3. The subjects — three, because one graph is an anecdote

| # | subject | global by | declared reach | why it is here |
|---|---|---|---|---|
| 1 | **Poseidon-T**, frozen scOT, $20.8$M | **architecture** | $2$ ($\rho\times s$) | the primary; W93's own subject |
| 2 | **Poseidon-B**, frozen scOT, $157.7$M | **architecture** | $2$ | **the second donor.** Same architecture family, `depths` $[8,8,8,8]$ against T's $[4,4,4,4]$, $7.6\times$ the parameters, its own weights. A tail measured on one checkpoint is a fact about that checkpoint |
| 3 | **WindowNS as-built**, classical, embedded pressure solve | **physics** | $20$ | a globally-supported operator whose globality is *not* an architecture choice — an elliptic solve inverts a sparse matrix whose inverse is dense |
| 4 | **WindowNS split-step**, the same solver, elliptic part exposed | *nothing* | $20$ | **the positive control.** If the instrument cannot read this one as compact, no negative it returns on the others means anything |

All four sit on the **same seam of the same geometry at the same state** — `poseidon.PoseidonTiling`, $235^2$ domain, $128$-cell windows, $21$-cell halo, the developed wake at $t=5$ that every Tier 0 number was taken at. Nothing about the geometry, the state or the port differs between rows.

---

# 4. The measurement

## 4.1 Along the seam: the truncation plateaus

$E(r)/\lVert S\rVert_2$, the fraction of the operator's spectral norm left outside a band of half-width $r$:

| subject | $r{=}2$ | $r{=}8$ | $r{=}20$ | $r{=}64$ | slope over $r=48\!\to\!64$ |
|---|---|---|---|---|---|
| poseidon-T P00 xhi | $0.855$ | $0.441$ | $0.297$ | $\mathbf{0.273}$ | $-1.3\times10^{-4}$ /cell |
| poseidon-T P00 yhi | $0.978$ | $0.755$ | $0.753$ | $\mathbf{0.753}$ | $-1.3\times10^{-5}$ /cell |
| poseidon-T P11 xlo | $0.719$ | $0.311$ | $0.304$ | $\mathbf{0.298}$ | $-8.5\times10^{-5}$ /cell |
| **poseidon-B** P00 xhi | $0.653$ | $0.374$ | $0.329$ | $\mathbf{0.316}$ | $-1.2\times10^{-4}$ /cell |
| **poseidon-B** P00 yhi | $0.902$ | $0.569$ | $0.561$ | $\mathbf{0.559}$ | $-3.9\times10^{-5}$ /cell |
| **poseidon-B** P11 xlo | $0.957$ | $0.846$ | $0.660$ | $\mathbf{0.517}$ | $-3.4\times10^{-4}$ /cell |
| windowns-as-built C00 | $0.201$ | $0.080$ | $0.047$ | $0.0118$ | $-7.0\times10^{-4}$ /cell |
| windowns-as-built C11 | $0.138$ | $0.042$ | $0.020$ | $0.0048$ | $-2.8\times10^{-4}$ /cell |
| **windowns-split-step** C00 | $0.0015$ | $\mathbf 0$ | $\mathbf 0$ | $\mathbf 0$ | $0$ |
| **windowns-split-step** C11 | $0.0027$ | $\mathbf 0$ | $\mathbf 0$ | $\mathbf 0$ | $0$ |

> **Corrected 2026-09-10 (Tier 41).** *Exact zero at $r=13$* in the next paragraph is one radius early. $E(13)/\lVert S\rVert_2$ is $1.4\times10^{-14}$ on C00 and $2.2\times10^{-14}$ on C11, and the first exact zero is $E(14)$ — in `out/w173/w173.json`, and again in CS-S2's same-session re-run of the control ([[case-study-bounded-donor-atlas-0.1]]), to the digit. §0's and §5.1's *exactly zero past $r=13$* are right as written. A test that read *past 13* as *from 13* failed on the re-run and was corrected, not loosened.

**Read the shape, not the level.** The classical local operator reaches *exact* zero at $r=13$ and stays there. The classical global one keeps falling — it loses a factor of $4$ between $r=20$ and $r=64$. The two checkpoints **stop falling**: between $r=16$ and $r=64$ Poseidon-T's `yhi` face moves from $0.7534$ to $0.7525$, which is $0.1\%$ of itself over $48$ cells.

**$r^\star$, the smallest radius reaching a target.** $n=128$, so $r=127$ means *keep the whole operator* and is arithmetic rather than decay; it is marked $(127)$ and is not an answer:

| subject | $<10^{-1}$ | $<10^{-2}$ | $<10^{-3}$ | $<10^{-4}$ | $<10^{-6}$ |
|---|---|---|---|---|---|
| poseidon-T (all 3 configs) | $(127)$ | $(127)$ | $(127)$ | $(127)$ | $(127)$ |
| poseidon-B (all 3 configs) | $(127)$ | $(127)$ | $(127)$ | $(127)$ | $(127)$ |
| windowns-as-built C00 | $6$ | $70$ | $120$ | $(127)$ | $(127)$ |
| windowns-as-built C11 | $3$ | $46$ | $110$ | $(127)$ | $(127)$ |
| windowns-split-step C00 | $0$ | $2$ | $3$ | $4$ | $6$ |
| windowns-split-step C11 | $1$ | $2$ | $3$ | $4$ | $6$ |

**Six of six learned configurations reach no target at any nontrivial radius, including $10^{-1}$.** Not *"the halo is large"* — *there is no halo*, at a tolerance of ten percent, on a seam whose whole length is $128$ cells.

Extrapolating each terminal slope — quoted as an extrapolation, which is what it is — the radius at which the truncation would reach $10^{-2}$:

| poseidon-T P00 xhi | poseidon-T P00 yhi | poseidon-T P11 xlo | poseidon-B P00 xhi | poseidon-B P00 yhi | poseidon-B P11 xlo |
|---|---|---|---|---|---|
| $2.0\times10^{3}$ | $5.7\times10^{4}$ | $3.5\times10^{3}$ | $2.7\times10^{3}$ | $1.4\times10^{4}$ | $1.5\times10^{3}$ |

against a seam of $128$ cells and a window the checkpoint cannot leave.

## 4.2 The most generous reading available, and it does not rescue it

The checkpoint was trained on the **periodic unit square**, so it treats the window's two ends as neighbours — and W98 already found that costing a wake array a manufactured deficit $3.5\,D$ upstream. Under *circular* distance a wrap-around coupling is counted as **near** instead of far, which is the most favourable reading the operator can be given:

| subject | $<10^{-1}$ | $<10^{-2}$ | trivial radius |
|---|---|---|---|
| poseidon-T P00 xhi | $51$ | $63$ | $64$ |
| poseidon-T P00 yhi | $20$ | $62$ | $64$ |
| poseidon-B P00 xhi | $50$ | $63$ | $64$ |
| windowns-split-step C00 | $0$ | $2$ | $64$ |

Circular distance on $128$ cells saturates at $64$, so $r=64$ is again *keep everything*. **A meaningful part of the far field is the periodic image** — the `yhi` face goes from $0.753$ linear to $0.095$ circular at $r=20$ — and it does not help, twice over: the wrap is still not below $10^{-2}$ at any nontrivial radius, and a periodic image is not a tail to be truncated in the first place. It is a **modelling error the tiling introduced**, since a tile's seam is not periodic in the composed problem.

## 4.3 Across the seam: R10's own quantity, and it agrees

§4.1 measures the operator **along** the seam, because that is what `support_reach` measures and what W93 compared against the declaration. But the number `L2/R10/halo` compares the overlap against is transverse — the rule's own sentence is *"an artificial boundary's influence travels `stencil_radius` cells per internal sub-step"*, which is a distance **into** the window. The two could differ: an operator that mixes along the seam through attention and only diffuses across it would be global one way and compact the other, and would have a derivable $\varepsilon$-halo. Nothing in this vault had measured it on a learned expert.

Poke one ring cell, step once, and profile $\lVert\delta(u,v)\rVert$ against depth $d$ from the poked face. $d^\star$ is the depth at which the response falls to a given fraction of its value **at the face**:

| subject | nonzero reach | $d^\star(10^{-1})$ | $d^\star(10^{-2})$ | $d^\star(10^{-4})$ | fitted $\ell$ |
|---|---|---|---|---|---|
| poseidon-T P00 xhi, $j_0=32,64,96$ | $127$, $127$, $127$ | $16$, $19$, $60$ | **never** | **never** | — |
| poseidon-T P11 xlo | $127$ | $29$ | **never** | **never** | — |
| **poseidon-B** P00 xhi, $j_0=32,64,96$ | $127$, $127$, $127$ | $16$, $16$, $16$ | **never** | **never** | — |
| windowns-as-built C00, $j_0=32,64,96$ | $126$ | $7$, $7$, $7$ | $59$, $55$, $59$ | $(127)$ | $21.6$, $20.7$, $21.5$ |
| **windowns-split-step** C00, $j_0=32,64,96$ | $17$, $15$, $16$ | $3$, $2$, $3$ | $5$, $4$, $4$ | $8$, $7$, $7$ | $0.470$, $0.464$, $0.466$ |

**Both directions give the same answer.** Neither checkpoint's response ever falls to $1\%$ of its face value at any depth into the window — there is no $d$ at which the stale datum has become negligible, so there is nothing for an overlap to outrun. The classical local solver's transverse reach is $15$–$20$ cells against a **declared** $\rho s=20$: the declaration the halo rule reads is *correct* for the agent it was derived for, and correct with no margin to spare, which is the strongest evidence available that the rule's arithmetic is right and its subject is wrong.

## 4.4 And even granted a radius, $\sigma$ gains nothing above $16$

[[master-error-bound]] §4.1's overlapping branch is $\sigma \le C_\mu \Pi \lVert\delta\lambda\rVert$, where $\Pi$ is the weight the assembly gives to cells a stale artificial-boundary datum can have reached. $\Pi$ is a function of the reach $d$, and on this tiling:

| $d$ | $0$ | $1$ | $2$ | $4$ | $8$ | $\mathbf{16}$ | $20$ | $32$ | $64$ | $127$ |
|---|---|---|---|---|---|---|---|---|---|---|
| $\Pi$ | $0.0000$ | $0.0116$ | $0.0954$ | $0.3648$ | $0.7250$ | $\mathbf{1.0000}$ | $1.0000$ | $1.0000$ | $1.0000$ | $1.0000$ |

$\Pi=1$ for **every** halo of $16$ cells or more, and the bound degrades to $\sigma\le C_\mu\lVert\delta\lambda\rVert$ — which [[master-error-bound]] §4.1 states is right and says out loud is uninformative. This is [[control-observability]]'s identity arriving as a *finite* statement rather than a limit: a globally receptive agent has $\Pi=1$ exactly, pinned by the partition-of-unity identity, and it turns out **the pinning does not need globality — $16$ cells is enough**.

So an $\varepsilon$-halo would have to be **simultaneously** below $16$ cells to buy anything in $\sigma$ and large enough to make the truncation small. At $r=16$ the two checkpoints' truncation is $0.32$ to $0.85$ of the operator. **The two requirements point in opposite directions with a gap of orders**, and this is a second, independent kill that does not depend on the decay measurement at all.

## 4.5 The third obstruction: the comparand does not exist

The proposal compares the tail against $\varepsilon_{\text{tol}}=\min(\tau,\sigma)$, *"which already exists in the compiler"*. It exists as a **rule**. On the graph the rule is about it has no value:

```
poseidon-t-2x2      verdict admit-uncertified
                    measured = None
                    unmeasured: L (W1), beta (W2), sigma (W3), tau (W3), C_mu (W3), p_decline (W36)
                    scheme.eps_tol = None
  L5/eps_tol        admit-uncertified -- "tau and sigma are unmeasured (W3),
                    so the rule cannot be applied and no tolerance is set"
```

And they are not merely unmeasured. `poseidon.build`'s own docstring records why: *"`tau`, `sigma` and `C_mu` are **not measurable at all** for this expert"* — the checkpoint is fixed at $128\times128$, so there is **no same-class monolith at any resolution** and no reference pair for the defect to be measured against. $\varepsilon_{\text{tol}}$ on this graph is not a number nobody has got round to; it is structurally unavailable until the checkpoint becomes resolution-free.

**This is why §4.1 and §4.2 are quoted relative to $\lVert S\rVert$.** A dimensionless truncation needs no constant borrowed from another expert, and a tail that is a *constant fraction of its own operator* is not below any tolerance a composition could declare, because it is the same size as the thing it is a perturbation of.

---

# 5. The controls

**Every one of these can fail, and the reason each is here is that without it a reading above means less than it appears to.**

## 5.1 The instrument reads a compact operator as compact

`windowns-split-step` is the same `WindowNS` solver with its elliptic part handed to the composition layer. On the same seam, same state, same instrument: $E(r)$ is **exactly zero** in float64 past $r=13$, the fitted along-seam length is $\ell=0.397$ cells at $R^2=0.995$ and $0.999$ on two different base states — agreeing to three digits — and the transverse fit is $\ell=0.464$–$0.470$ across three poke positions. The negative in §4 is not the instrument returning "global" for everything.

## 5.2 The second donor, and it is the claim's whole scope

The ladder's claim is about an **expert class**. Poseidon-B is $7.6\times$ Poseidon-T's parameters and twice its depth, and its truncation plateaus at $0.32$, $0.56$ and $0.52$ against T's $0.27$, $0.75$ and $0.30$ — the **same** behaviour, not a similar number. This is the reading that licenses saying *scOT-family windowed-attention operators* rather than *this checkpoint*.

**Scope, stated rather than implied.** Two checkpoints of **one** architecture family is not a survey. [[expert-donor-survey]]'s fourth cross-cutting finding already reached the same verdict by reading rather than measuring — *"the locality verdict is near-unanimous: of ~20 entries, three are bounded, and all three are local operators or finite-radius message passing, never transformers or spectral mixing"* — and names **MACE** ($r_{\text{cut}}\times$ layers) as the locality existence proof from outside continuum fluids. **And that survey's own Open question 3 asks for exactly this measurement** — *"nobody has run the same delta-poke on it that produced W93's 64. Until that measurement exists, 'bounded' is a derivation"* — so the instrument built here is the one that question was waiting for, pointed at the wrong half of the survey. This page measures the negative half and does not measure the positive half. See §8.

> **The positive half, attempted 2026-09-10 (Tier 41, [[case-study-bounded-donor-atlas-0.1]]).** The same two functions, unchanged, on NeuberNet — the one bounded donor in the survey with a continuum boundary port — beside linear elasticity solved on its own disc, with split-step `WindowNS` re-run as the positive control in the same session and reading exactly zero from $r=14$ again, to the digit. It did not come back compact: along its $29$-sensor ring the band truncation stays at $0.69$ of the operator outside $r=4$ where the elastic physics keeps $0.14$, and into the disc its response does not decay where the physics' does. So the negative now stands on a second architecture family — a NOMAD DeepONet beside scOT — measured on a very different seam, and the positive half is unmeasured on the two donors whose locality is a property of the operator rather than of its domain, MACE and DeepFlame.

## 5.3 Is $S$ an operator, or the instrument's own floor?

A linear response satisfies $S(a)=S(a/2)$ exactly, so the disagreement between two amplitudes is nonlinearity (growing with $a$) plus cancellation (growing as $1/a$). Relative disagreement against $a=10^{-2}$:

| $a$ | $1$ | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ | $10^{-5}$ |
|---|---|---|---|---|---|---|
| **poseidon-T** | $0.534$ | $0.073$ | $0$ | $0.310$ | $\mathbf{2.68}$ | $\mathbf{28.7}$ |
| windowns-as-built | $0.209$ | $0.021$ | $0$ | $0.0021$ | $0.0023$ | $0.0024$ |
| windowns-split-step | $0.104$ | $0.010$ | $0$ | $0.0010$ | $0.0011$ | $0.0011$ |

A clean **V** on the checkpoint and none on either classical column. The checkpoint runs in float32; the classical solvers in float64. So:

- $a=10^{-2}$ is at the bottom of the V and the operator measured there is an operator.
- **$a=10^{-4}$ is $268\%$ wrong and is the instrument's floor, not the architecture.** That row is kept in the tables and is *not* evidence: its $r^\star(10^{-1})=120$, its terminal slope $-5.2\times10^{-3}$ and its fitted $\ell=59$ cells are all readings of float32 cancellation. Reported rather than deleted because a table that quietly drops its worst row is the failure this vault refuses.
- The same V is why W93's original *"nonzero at every amplitude from $1$ to $10^{-3}$"* is sound and why pushing it to $10^{-5}$ would not have been.

**[AI Inference]:** this puts a floor under any future $\varepsilon$-halo independent of the architecture question. A tail can only be certified where it can be *measured*, and for a float32 checkpoint differencing at amplitude $a$ the measurable floor is about $\epsilon_{32}\lVert f_0\rVert/a$ — so $\varepsilon_{\text{tol}}$ below the expert's own reproducibility floor is undecidable whatever the decay looks like. Not built, and it would matter only if the decay question were ever answered the other way.

## 5.4 Two base states and two faces, per subject

Every subject was measured at two tiles cut from different parts of the same developed wake, and Poseidon at two faces of one tile. The `yhi` face plateaus three times higher than `xhi` ($0.75$ vs $0.27$) on T and twice as high on B, so the *level* is a property of the face and the *shape* is not.

---

# 6. What the far field is made of

A truncation that plateaus can plateau two ways — many small modes that never decay, or a few coherent global ones. The singular values of the off-band block at $r=20$ separate them:

| subject | share of leading mode | share of top four | rank to $99\%$ |
|---|---|---|---|
| poseidon-T P00 xhi | $0.227$ | $0.694$ | $33$ |
| poseidon-T P00 yhi | $0.667$ | $0.947$ | $19$ |
| poseidon-B P00 yhi | $0.757$ | $0.949$ | $17$ |
| windowns-as-built C00 | $0.571$ | $0.815$ | $63$ |
| windowns-split-step C00 | — | — | $0$ (the block is zero) |

**The checkpoints' far field is lower-rank than the classical one's** — four modes carry $69$–$95\%$ of it, in $17$–$33$ effective dimensions against the classical solver's $57$–$63$. So the plateau is not noise; it is a handful of **coherent global modes**, which is what a windowed-attention stack with four levels of patch merging is built to produce.

> **[AI Inference], and it is the one direction this negative leaves open.** An operator that is *banded plus low-rank* is not bounded by a halo and is not unbounded either — it is a different declaration, with $\lVert S - S_r - U_kV_k^{\!\top}\rVert$ in place of $E(r)$, and the rank-$k$ correction entering $\sigma$ the way the tail would have. Nothing here derives that, one graph is not enough to write a rule against ([[gap-worklist]]'s own standing rule), and the honest status is: **the measurement that would decide it is the same one this page ran, at a second architecture family.** Recorded as W176, not built.

---

# 7. The sharpest secondary finding: the proposal's own words name the wrong test

The proposal says *"the radius beyond which the **response** falls below $\varepsilon_{\text{tol}}$"*. Read literally that is a **per-cell** test, and per cell the checkpoints look like they pass:

| subject | mean $\lvert S_{ij}\rvert$ at $d{=}0$ | at $d{=}4$ | at $d{=}20$ | at $d{=}31$ | $\sum_{d>20}$ / peak |
|---|---|---|---|---|---|
| poseidon-T P00 xhi | $1.8\times10^{-4}$ | $3.4\times10^{-4}$ | $2.3\times10^{-5}$ | $1.1\times10^{-5}$ | $\mathbf{1.61}$ |
| poseidon-T P11 xlo | $3.4\times10^{-4}$ | $3.3\times10^{-4}$ | $2.1\times10^{-5}$ | $1.8\times10^{-5}$ | $\mathbf{1.41}$ |
| poseidon-B P00 xhi | $3.7\times10^{-4}$ | $2.8\times10^{-4}$ | $1.7\times10^{-5}$ | $1.3\times10^{-5}$ | $\mathbf{1.55}$ |
| poseidon-B P11 xlo | $1.2\times10^{-4}$ | $1.8\times10^{-4}$ | $1.8\times10^{-5}$ | $1.9\times10^{-5}$ | $\mathbf{3.50}$ |
| windowns-as-built C00 | $1.1\times10^{-1}$ | $7.6\times10^{-3}$ | $3.4\times10^{-4}$ | $6.4\times10^{-5}$ | $0.043$ |
| windowns-split-step C00 | $3.2\times10^{-1}$ | $3.0\times10^{-5}$ | $0$ | $0$ | $0$ |

At $d=20$ the entries are down by a factor of $15$ from the peak. **Summed over the $107$ cells beyond, the dropped tail is $1.4$ to $3.5$ times the peak entry** — for the classical global operator it is $0.04$, and for the local one exactly $0$.

The arithmetic is elementary and that is the point: a per-cell threshold on an $n$-cell seam admits $n\varepsilon_{\text{tol}}$ of error, and on a $128$-cell seam that is two orders of margin thrown away silently. **An $\varepsilon$-halo written to the proposal's own wording would have looked derivable at $r\approx20$ and would have been wrong**, and the compiler would have carried a bounded halo for an operator that has none.

This is the vault's own recurring class — Tier 24's *a verified number can be verified against the wrong question* — reaching a rule that had not been written yet, which is the cheapest place it has ever been caught.

---

# 8. Verdict against §5's three outcomes

| §5's outcome | what it would mean | measured |
|---|---|---|
| tail summable and below $\varepsilon_{\text{tol}}$ at a finite radius | the vision on a constrained expert class | **not for scOT.** Six of six learned configurations reach no target at any nontrivial radius, down to $10^{-1}$ |
| tail summable but only past the whole domain | the bound exists and buys nothing | **this is the classical embedded-elliptic solver's answer**, not the checkpoints': $r^\star(10^{-2})=46$–$70$, $r^\star(10^{-3})=110$–$120$ on a $128$-cell seam. Real, derivable, and worth nothing at the widths a tiling can afford |
| tail not summable | **R10 stands and there is no $\varepsilon$-halo** | **this is the checkpoints' answer.** The truncation is flat to $10^{-5}$ per cell at $27$–$75\%$ of the operator norm; $\Pi=1$ above $16$ cells regardless; and the transverse profile never reaches $1\%$ of its face value at any depth |

> **What "summable" can and cannot mean here, stated once.** On a finite seam every sum is finite, so *summable* in the literal sense is vacuous and is not what §5 is asking. The question a halo rule actually needs is whether $E(r)\to0$ **fast enough that a radius exists at which the truncation is below what the composition tolerates** — which is why every figure above is a decay *rate* and an $r^\star$, not a convergence claim. Read that way the three subjects separate cleanly: exponential with $\ell=0.397$ cells and exact zero at $r=13$; power law with exponent $-1.75$ to $-1.84$ and $r^\star(10^{-2})=46$–$70$; and a curve that stops decreasing, whose terminal slope implies $10^{3}$–$10^{4}$ cells on a seam of $128$. The middle one is summable and useless; the last is the one this page is about.

> **Corrected 2026-09-10 (Tier 41).** *Exact zero at $r=13$* above is one radius early: the first exact zero is $E(14)$, and $E(13)/\lVert S\rVert_2 = 1.4\times10^{-14}$ — see the note at the head of §4.1.

**`L2/R10` stands, unchanged, and no rule was written.** [[case-study-ladder-to-f1]] §5's instruction was to measure before writing one, and the measurement says do not write it.

**What is *not* claimed.** That neural operators cannot be composed — they compose, and [[case-study-scaling-ladder-atlas-0.1]] measured the frozen column marching stably to $110$ macro-steps while the classical one went unstable. That R10's refusal is *earned* — see [[substitution-campaign-checkpoint]] §3, which measures that separately and gets a two-sided answer. That every learned architecture is global — [[expert-donor-survey]] names three bounded ones and this page measured none of them. And that a *different* certificate is impossible: §6 leaves banded-plus-low-rank open, and [[case-study-neural-interface-atlas-0.1]] already moved the expert somewhere its globality is not a liability.

---

# 9. The cheapest thing this changes

`probe.support_reach`'s docstring already says the right thing — *"a reach that is large but not saturating should be read rather than thresholded"* — and now there is a quantity to read it with. **$E(r)$ is $12$ lines and $n+1$ solves**, it is the exact object [[master-error-bound]] §4 needs, and it separates the three subjects on the first try where the binary support test calls two of the three *global* and stops.

That is worth recording even though the answer came back negative: the reason this question stayed open for nine tiers is that the only instrument pointed at it returned a boolean.

---

## See Also

- [[case-study-ladder-to-f1]] — §5's substitution campaign, whose middle row this closes
- [[substitution-campaign-checkpoint]] — §7's stopping rule, stood on against this result
- [[master-error-bound]] — §4 and §4.1, the two $\sigma$ branches the tail would have entered
- [[control-observability]] — $\Pi=1$ for a globally receptive agent, as an identity; §4.4 is the finite version
- [[expert-donor-survey]] — its locality verdict (finding 4) and Open question 3, which asks for this measurement on a bounded donor
- [[case-study-neural-interface-atlas-0.1]] — the other response to R10: move the operator to the interface
- [[gap-worklist]] — W173, W176
