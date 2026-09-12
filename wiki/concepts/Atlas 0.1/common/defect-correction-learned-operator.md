# Defect Correction with a Learned Operator — a slot where a checkpoint changes the rate and never the answer

**Type:** Concept page — **theory, with a pre-registered gate** (folder: `Atlas 0.1/common/`)
**Status:** written 2026-09-11 and measured out of sample 2026-09-12, Tier 48. §1–§7 were written after the in-sample kill tests of [[learned-contribution-kill-tests]] at $N=2$ and **before** any out-of-sample run; §8 records both, in that order. `atlas/defect_correction.py`, `scripts/w202_kill_tests.py`, `tests/test_tier48_defect_correction.py`, `tests/test_tier48_learned_contribution.py`, `out/w202/w202.json`. Registered off the ladder as **CS-S3**. Worklist rows from **W202**.
**Related:** [[learned-contribution-kill-tests]] · [[substitution-campaign-checkpoint]] · [[case-study-neural-interface-atlas-0.1]] · [[master-error-bound]] · [[composition-error-theory]] · [[case-study-scaling-ladder-atlas-0.1]] · [[interaction-horizon]] · [[control-observability]] · [[f1-pathmap-and-end-goal]] · [[case-study-ladder-to-f1]] · [[gap-worklist]] · [[mathematical-compendium]]

---

# 0. The one-paragraph version

Every route this vault has tried asked a learned expert to be **part of the answer** — an agent at a seam, a starting guess — and so owed it a certificate it cannot be issued: a checkpoint fixed at one resolution has no same-class reference pair, hence no $\tau$, no $\sigma$, no $\varepsilon_{\text{tol}}$ and no $\beta_{\min}$ (W95). This page puts it somewhere else: inside Stetter's **defect correction** as the approximate operator, with the classical composed macro-step as the target. There, four things hold, and three of them are theorems. **Whatever the learned map is, any limit of the iteration is the classical settled state** — its errors cost classical calls, never correctness. **Replacing it by a constant makes the iteration the classical march, bit for bit**, so W76's null replacement is an identity measured in the same currency as everything else. **What it changes is the rate**, through its Jacobian, so CS-S1's one-sweep ceiling on a predictor does not bind here. And **the returned state is certified** by its own classical residual times a constant of the classical map. Measured in sample at $N=2$ on `wake_array`'s geometry: Poseidon-T's composed column, shrunk halfway toward the null element, finds the certified state in $25$ classical calls where the cold classical march needs $47$; the same column with its output detuned needs $32$; and the composition layer with the checkpoint replaced by the identity falls back after $14$ outer iterations, with $35$ classical calls in all. **So the saving is the checkpoint's, and it degrades as the checkpoint does.** But it spends $100$ checkpoint calls to do it, and a classical solver on a grid twice as coarse does the same job in $22$ classical calls and $90$ cheap ones. §7 states, before the out-of-sample rungs ran, what they must show for this to count as a meaningful contribution — and predicts that they will not show it. **They did not.** At six windows the checkpoint needs $99$ classical calls where the gate asked for at most $90$, costs $314$ classical-equivalents against the coarse solver's $52$, and is beaten by a copy of itself carrying Gaussian noise on every weight tensor. At twelve it is **fifth of seven arms by cost**, behind that corrupted copy and behind the composition layer with no checkpoint in it at all — and there **the certificate itself fails**, two arms of seven returning a state outside their own bound, because the constant $\Theta$ was read off one approach to the settled state and used on the rest. **What the tier establishes is the slot, its entry condition, and where the guarantee's own hypothesis breaks — not a contribution.**

---

# 1. The slot, and why it is not a starting guess

[[case-study-neural-interface-atlas-0.1]] priced a learned expert used as the **first iterate** of a convergent coupling iteration and found a ceiling of one sweep: a linearly convergent iteration needs $m=\ln(d_0/\epsilon)/\ln(1/\rho)$ sweeps, **a predictor moves $d_0$, and only the operator moves $\rho$**. At a contraction of $0.011$ per sweep, no start can buy more than one sweep.

[[learned-contribution-kill-tests]] §4.3 measured the same question one level up — the checkpoint's own settled wake used as the start of the classical march to the classical settled state, where the physical relaxation is slow and a log-gain argument would promise many steps. **It buys nothing**: $75$ classical steps to $10^{-4}$ of the cold distance from the checkpoint's start against $68$ from the freestream, because relaxation in an open advective domain is limited by **transit**, not by a contraction factor, and a start is flushed out in one transit whatever its size.

Both negatives are about the same thing: **a start is not part of the iteration operator.** The slot on this page is one where the learned map is.

---

# 2. The iteration

Let $\Phi$ be the classical composed macro-step on a declared problem — boundary data fixed — and $w^\star$ a settled state, $\Phi(w^\star)=w^\star$. Let $\Psi$ be any other map on the same state space: the learned composed column, the composition layer alone, a coarse classical solver. Write $G_\Phi=I-\Phi$ and $G_\Psi=I-\Psi$.

**Defect correction** (Stetter, *Numer. Math.* 29, 1978; Böhmer, Hemker & Stetter, *Computing Suppl.* 5, 1984) with $G_\Psi$ as the approximate operator and $G_\Phi$ as the target:

$$G_\Psi(w_{k+1}) \;=\; G_\Psi(w_k)\;-\;G_\Phi(w_k),$$

solved for $w_{k+1}$ by the cheap fixed-point march

$$x^{(0)}=\Phi(w_k),\qquad x^{(m+1)} \;=\; \Phi(w_k)\;+\;\bigl[\Psi(x^{(m)})-\Psi(w_k)\bigr].$$

One classical call per outer iteration, any number of cheap calls. The cheap map enters **only through a difference**, so a bias it carries cancels before it is added.

## 2.1 Consistency, whatever $\Psi$ is

**Theorem 1.** If $G_\Psi$ is continuous and $w_k\to\bar w$, then $\Phi(\bar w)=\bar w$.

*Proof.* $G_\Phi(w_k)=G_\Psi(w_k)-G_\Psi(w_{k+1})\to 0$, and $G_\Phi$ is continuous. $\square$

**Nothing about the accuracy of $\Psi$ enters.** A wrong cheap map can cost classical calls; it cannot change what is returned. This is the property CS-S1 measured for a predictor — a pure-noise start converges to the same fixed point — carried into a slot where the cheap map also changes the rate.

## 2.2 The rate

**Theorem 2** (local rate; Stetter 1978). If $\Phi$ and $\Psi$ are $C^1$ near $w^\star$ and $J_\Psi=I-D\Psi(w^\star)$ is invertible, with $J_\Phi=I-D\Phi(w^\star)$,

$$e_{k+1} \;=\; \bigl(I-J_\Psi^{-1}J_\Phi\bigr)\,e_k \;+\; o(\lVert e_k\rVert),$$

so the iteration converges linearly iff $q_\Psi:=\rho\bigl(I-J_\Psi^{-1}J_\Phi\bigr)<1$ — every eigenvalue of $J_\Psi^{-1}J_\Phi$ inside the unit disc centred at $1$.

Two readings carry the rest of the page:

- **The cheap map must not be closer to singular than the classical one on any mode.** Where $\Phi$ relaxes a mode and $\Psi$ preserves it — $J_\Psi\approx0$ there and $J_\Phi$ not — the corresponding eigenvalue of $J_\Psi^{-1}J_\Phi$ is large and $q_\Psi>1$.
- **What it measures is Jacobian fidelity, not accuracy.** A cheap map with a large constant bias and the right derivative converges fast; an unbiased one with the wrong slow modes diverges.

## 2.3 The null element, exactly

**Corollary 1.** If $\Psi$ is constant, the inner march returns $\Phi(w_k)$ after one cheap call, bit for bit, so $w_{k+1}=\Phi(w_k)$: **the iteration is the classical march.**

So the null replacement W76 was written about has an algebraic meaning in this slot, and every other cheap map is measured in the same currency against it: classical calls to the same certified residual. `tests/test_tier48_defect_correction.py` asserts the bitwise identity.

**Corollary 2** (shrinking toward the null element). For $\alpha\in[0,1]$ let $\Psi_\alpha=(1-\alpha)\Psi$. Then $J_{\Psi_\alpha}=\alpha I+(1-\alpha)J_\Psi$, whose eigenvalues have real part at least $\alpha$ wherever those of $J_\Psi$ have non-negative real part; $\alpha=1$ is Corollary 1 and $\alpha=0$ is $\Psi$ itself. A cheap map that is too close to singular on some mode is regularised on exactly that mode, and whatever it still contributes is measured against $\alpha=1$.

---

# 3. What is certified

The returned state carries its classical residual $r=\lVert\Phi(w)-w\rVert$, computed and never estimated. Two forms of the bound:

$$\lVert w-w^\star\rVert \;\le\; \frac{r}{1-L}\qquad(\text{Banach, if }\Phi\text{ contracts with constant }L<1\text{ on a ball holding }w\text{ and }w^\star),$$

$$\lVert w-w^\star\rVert \;\lesssim\; \Theta\,r,\qquad \Theta=\bigl\lVert\bigl(I-D\Phi(w^\star)\bigr)^{-1}\bigr\rVert\qquad(\text{linearised}).$$

$\Theta$ is a property of the **classical** map in its regime. It is read off a converged classical march as the largest error-to-residual ratio over the **middle half** of it (`theta_from_march`, whose window leaves out the transient at one end and the reference's own round-off floor at the other), re-measured on every rung, and the bound is reported beside the measured error. At $N=2$: $\Theta=3.55$ over $93$ steps of that window, against a smallest ratio of $1.99$, so the certificate at the stopping residual is $3.24\times10^{-4}$, and **every one of the twelve in-sample arms returns a state inside it** — the checkpoint's at $2.95\times10^{-4}$, $1.10\times$ inside.

**This reading of $\Theta$ is an estimate along one approach to $w^\star$, and that is the weak joint.** The true constant is an operator norm, a supremum over every direction; a march samples the one direction it took. The estimate is used on arms that approach from elsewhere, and §8.2.2 measures it **under-bounding two of them at twelve windows** — so the certificate below is reported per arm against its own returned residual, and the constant each arm would have needed is reported beside it (**W208**).

## 3.1 The hypothesis that is easy to break: the fixed point must be isolated

A map that carries boundary data as **state** has a family of fixed points, and $\Theta=\infty$ along it. The classical monolith in `wake_array`'s march is such a map: `WindowNS.step_batch` with `bc0=None` holds every ring at its input value and the freestream band resets only the inlet and the two laterals, so **the outflow column is whatever the starting state put there, forever**. Measured ([[learned-contribution-kill-tests]] §4.2): the classical march from the checkpoint's settled state, with the outflow column left as state, ends $5.9\times10^{-2}$ from the reference settled state — $29\%$ of the cold distance — **with a one-step residual of $1.6\times10^{-16}$**, while the same start with the column pinned to the freestream reaches $1.8\times10^{-5}$. It did not fail to converge; it converged exactly, to another member of the family, and a certificate built on the residual would have called it settled and been right about the wrong problem. **A residual certificate is a statement about a declared problem, and on this graph the problem had not been declared.** Every number on this page pins the outflow column; from the freestream that changes the reference trajectory by nothing.

---

# 4. The entry condition

For geometric convergence on both sides — a classical march contracting at $q_\Phi$ per call, defect correction at $q_\Psi$ per outer iteration with $\bar m$ inner calls each at relative cost $c_\Psi$ — the ratio of cold cost to corrected cost for the same residual reduction is

$$\mathcal X \;=\; \frac{\ln(1/q_\Psi)}{\ln(1/q_\Phi)\,\bigl(1+c_\Psi(1+\bar m)\bigr)} .$$

$\mathcal X>1$ is when the cheap map pays for itself. The measured version, which does not assume geometric convergence, is the ratio of classical calls on the cold march to $\phi_{\text{calls}}+c_\Psi\,\psi_{\text{calls}}$ on the corrected one, at the same stopping residual (`break_even`).

**Against the classical competitor.** The same solver on a grid coarsened by $2$ in each of $d$ dimensions has an explicit step costing $c_C\approx2^{-(d+1)}$ of the fine one — a quarter of the cells, and half the advective sub-steps. A learned map contributes **beyond classical coarsening** only if its measured ratio exceeds the coarse solver's at the same stopping residual. **[AI Inference]:** in two dimensions on this CPU the bar is harder than the arithmetic suggests — measured, the coarse solver costs $0.197$, $0.132$ and $0.0320$ of a classical step at two, six and twelve windows, below $2^{-3}$ at the top because the fine solve leaves cache and the coarse one does not. In three dimensions the arithmetic gives $2^{-4}$, while a learned forward on an accelerator has been measured in this vault at $1/19.5$ of its desktop cost ([[poc1a-frozen-expert-results]] §8), so the bar is not obviously out of reach there. Nothing on this page measures three dimensions or an accelerator.

---

# 5. What it dissolves, and what it moves

**Dissolved.**

- **W95, for the answer.** Nothing certifies the learned map, so nothing needs its reference pair. The referent is the classical map, which exists and is the verification authority [[f1-pathmap-and-end-goal]] §1.2 keeps.
- **CS-S1's ceiling, in this slot.** The cheap map is in the iteration operator (Theorem 2).
- **W76 as a blind spot.** The null replacement is the classical march itself (Corollary 1), and the checkpoint's contribution is read against it and against the composition layer with the checkpoint removed.
- **CS-8's per-state certificate, for the answer.** The residual is computed at the returned state. $\Theta$ remains a per-regime constant — of the classical map.

**Moved.**

- **To Jacobian fidelity on the slow modes.** The composed learned column holds each window's incoming mean and transports periodically on a padded domain; the monolith relaxes both through its ring and its viscosity. **[AI Inference]**, read off the error structure of the stalled iterates rather than off assembled Jacobians ([[learned-contribution-kill-tests]] §4.4): where the unshrunk checkpoint stalls, $72\%$ of its error is in the lowest modes and $23\%$ is constant per window, against $6$–$15\%$ and $\le10^{-4}$ for arms still converging; where the composition layer alone stalls, only $2\%$ is in the lowest modes, because that map has no viscosity and what it cannot relax is everything the monolith damps. **Two failures, one theorem**: Corollary 2 bounds the eigenvalues either kind of column puts near zero, which is why shrinking repairs both.

  > **2026-09-12, Tier 50: this inference is now measured, and it holds — with a correction about the repair.** [[corrupted-checkpoint-and-jacobian-fidelity]] probes each map's derivative directly along band-limited perturbations at $w^\star$ instead of reading stalled iterates. On the slow band the classical map's Rayleigh quotient is $\phi = 0.4780$ and the clean column's is $\mathbf{0.5578}$ — **it does preserve what the monolith damps**, as this bullet says. What the bullet does not say is that **the repair overshoots**: $\alpha^\star = 0.5$ takes the column to $0.2789$, below $\phi$ by $0.199$ where the original excess above it was only $0.080$. Theorem 2's assembled per-band eigenvalue orders twenty cheap maps by classical calls at $\rho = +0.83$ against accuracy's $+0.69$, so §2.2's *"Jacobian fidelity, not accuracy"* is confirmed as the thing that decides an arm. **[AI Inference]:** matching $(1-\alpha)\psi_{\text{slow}}$ to $\phi_{\text{slow}}$ would want $\alpha \approx 0.14$ here, not $0.5$, and costs three forward calls to check.
- **To boundary data as problem data** (§3.1).
- **To cost against classical coarsening** (§4), which is where the prediction in §7 says it fails.

**Hypotheses, each marked.**

| # | hypothesis | status |
|---|---|---|
| H1 | $w^\star$ is isolated and attracting on the declared problem | **checkable here** — the reference march converges, and §3.1's control breaks it on purpose |
| H2 | $J_\Psi$ is invertible along the path | **checkable at cost** — a violation shows as a rising residual, which the stall rule catches |
| H3 | $\Theta$ read off the cold march bounds the ratio along any other approach to $w^\star$ | **checked here, and it FAILS** — every arm's returned error is compared with its own $\Theta r$: inside on all twelve arms at two windows and all seven at six, **outside on two of seven at twelve**, where the worst arm needs $3.81$ against the cold march's $3.35$ (§8.2.2, **W208**) |
| H4 | the per-call cost ratios are properties of this host and this batch size | **checkable here, and they do not transfer** — [[atlas-proof-of-concept-1]] §10's standing warning; every ratio is measured in the same process as the classical call it divides |

---

# 6. Where this sits against the vault's other theory

- **[[master-error-bound]]** bounds a trajectory by $\tau$, $\sigma$, $\gamma$ and $L$. This page bounds a **settled state**, and none of the learned column's $\tau$, $\sigma$ or $\gamma$ enters it: they are properties of the column, and the column is not what is returned. The classical map's composition defect is inside $\Phi$, so the certified object is the classical composed answer, not the physics — the same scope as every classical column in the vault.
- **[[interaction-horizon]] §6** closed virtual control because its minimiser is *where two biased agents agree*, not the truth. Defect correction's limit is the truth of $\Phi$ by Theorem 1, and the structural difference is the **classical defect** in the iteration: agreement between the cheap map and itself is never the objective.
- **[[composition-error-theory]] §4.4** found the stack's differentiability unspent. Nothing here spends it; the checkpoint is used forward only.
- **Parareal** (Lions, Maday & Turinici 2001; Gander & Vandewalle, *SISC* 29, 2007) is the time-parallel analogue with a learned coarse propagator, and its total work is never below the serial classical march — [[learned-contribution-kill-tests]] records why it was not chosen.
- **The closest published pattern** is HINTS (Zhang, Kahana, Kopaničáková, Turkel, Ranade, Pathak & Karniadakis, *Nature Machine Intelligence* 6, 2024): a neural operator alternated with a classical relaxation, converging to the classical solution and exploiting the network's low-frequency bias against the relaxation's high-frequency one. HINTS trains its operator for the purpose; this page asks what a frozen checkpoint trained for something else can do in the same kind of slot.

---

# 7. The gate, pre-registered

Written after the $N=2$ runs and before any $N=6$ or $N=12$ run. **Fixed on $N=2$ and not re-tuned**: the iteration's settings (`DEC_SETTINGS`: at most $25$ outer and $40$ inner iterations, inner tolerance $0.1$ of the current residual, stall on two consecutive non-decreasing residuals, fallback to the classical march); the stopping residual rule (the cold march's residual at the step its error first reaches $10^{-3}$ of the cold distance); and **$\alpha^\star=0.5$**, the only value on the grid $\{0,0.5,0.8\}$ at which the checkpoint converged without falling back.

**Arms on every out-of-sample rung:** the cold classical march; $C$ at $\alpha=0$ and $\alpha^\star$ (the $2\times$-coarse classical solver); $Z$ at $\alpha^\star$ (the composition layer with the checkpoint replaced by the identity); $P$ at $\alpha^\star$ and at $\alpha=0$ (Poseidon-T's composed column); $P_c$ at $\alpha^\star$ (output detuned by $0.8$); $P_w$ at $\alpha^\star$ (Gaussian noise at $3\%$ of each weight tensor's rms, one fixed seed).

| clause | measured as | passes if |
|---|---|---|
| **G1 · non-null** | classical calls of $P$ at $\alpha^\star$ | $\le 0.8\times\min(\text{cold},\;Z\text{ at }\alpha^\star)$ |
| **G2 · non-vacuous** | returned error against $\Theta r$, $\Theta$ re-measured on the rung | every converged arm inside, and $\Theta r/\text{error}\le10$ for $P$ at $\alpha^\star$ |
| **G3 · out-of-sample** | G1, G2, G4, G5 and G6 on $N=6$ and $N=12$ | all hold on both, with nothing re-tuned |
| **G4 · marched** | the classical march continued from $P$'s answer, $50$ steps on $N=6$ and $30$ on $N=12$ | never further than $2\Theta r$ from the answer |
| **G5 · accounted** | $\phi_{\text{calls}}+c_P\,\psi_{\text{calls}}$, with $c_P$ measured in the same process on that rung | $\le\text{cold}/1.5$ **and** $\le$ the better of the two $C$ arms' cost |
| **G6 · loud** | $P_w$ at $\alpha^\star$ | its answer is inside $\Theta r$ whether it converged or fell back, **and** it does not need fewer classical calls than $P$ |

**The prediction, recorded with the gate.** G2 and G4 pass. G1 is plausible. **G5 fails on both rungs, on its second clause at least**: at $N=2$ the checkpoint needed $25$ classical and $100$ cheap calls where the coarse classical solver needed $22$ and $90$, and a coarse call is about an eighth of a classical one where a checkpoint call is not less than a quarter on any rung CS-7 timed. If that is what happens, **the learned expert does not contribute meaningfully in this slot on this host**, and the reason is §4's bar rather than §3's certificate.

---

# 8. Measured — CS-S3

## 8.1 In sample: $N=2$, where $\alpha^\star$ was chosen

`out/w202/w202.json`, `dec.N2`. Stopping residual $9.14\times10^{-5}$; the cold classical march reaches it in $47$ calls, at an error of $1.86\times10^{-4}$; certificate $\Theta r=3.24\times10^{-4}$.

| cheap map | $\alpha$ | status | classical calls | cheap calls | returned error | inside $\Theta r$ |
|---|---|---|---|---|---|---|
| *(cold march)* | — | converged | $47$ | $0$ | $1.86\times10^{-4}$ | yes |
| $C$, $2\times$ coarse classical | $0$ | converged | $21$ | $132$ | $8.48\times10^{-5}$ | yes |
| $C$ | $0.5$ | converged | $22$ | $90$ | $1.42\times10^{-4}$ | yes |
| $C$ | $0.8$ | cap, fallback | $36$ | $75$ | $2.60\times10^{-4}$ | yes |
| $Z$, no checkpoint | $0$ | residual **rising**; stall at $k=2$, fallback | $50$ | $41$ | $1.86\times10^{-4}$ | yes |
| $Z$ | $0.5$ | **stalled**, fallback at $k=14$ | $35$ | $68$ | $2.59\times10^{-4}$ | yes |
| $Z$ | $0.8$ | cap, fallback | $38$ | $75$ | $3.03\times10^{-4}$ | yes |
| $P$, Poseidon-T | $0$ | residual **rising**; stall at $k=2$, fallback | $50$ | $40$ | $1.86\times10^{-4}$ | yes |
| $P$ | $\mathbf{0.5}$ | **converged** | $\mathbf{25}$ | $\mathbf{100}$ | $2.95\times10^{-4}$ | yes, $1.10\times$ |
| $P$ | $0.8$ | cap, fallback | $40$ | $75$ | $2.18\times10^{-4}$ | yes |
| $P_c$, detuned | $0$ | residual rising; stall at $k=4$, fallback | $48$ | $40$ | $1.93\times10^{-4}$ | yes |
| $P_c$ | $0.5$ | cap, fallback | $32$ | $92$ | $2.54\times10^{-4}$ | yes |
| $P_c$ | $0.8$ | cap, fallback | $42$ | $75$ | $1.95\times10^{-4}$ | yes |

Read at $\alpha^\star$, the classical calls order as the checkpoint's integrity does: **$P$ $25$ < $P_c$ $32$ < $Z$ $35$ < cold $47$.** And **every arm, including every one whose residual rose, returns a state inside the certificate** — those because the stall rule handed them to the classical march, which is Theorem 1 made operational.

## 8.2 Out of sample

Nothing was re-tuned: the settings, the stopping rule, $\alpha^\star=0.5$ and the arm list are §7's, fixed on $N=2$. $\Theta$ is re-measured on each rung, as §3 requires. The detuned and the corrupted checkpoint are the same network and are charged at the checkpoint's own per-call ratio; the ratios are [[learned-contribution-kill-tests]] §2.3's, measured in the same process as the classical call they divide.

### 8.2.1 Six windows

Cold march: $141$ classical calls to $r=9.08\times10^{-5}$. $\Theta=3.671$, so the certificate is $3.33\times10^{-4}$. Per-call costs: $P$ $2.008$, $Z$ $0.149$, $C$ $0.132$ of one classical step.

| cheap map | $\alpha$ | stopped by | classical calls | cheap calls | cost, in classical calls | cold / cost | returned error |
|---|---|---|---|---|---|---|---|
| *(cold march)* | — | converged | $141$ | $0$ | $141$ | $1.00$ | $2.19\times10^{-4}$ |
| $C$ | $0$ | **converged** | $24$ | $211$ | $\mathbf{51.9}$ | $\mathbf{2.71}$ | $8.75\times10^{-5}$ |
| $C$ | $0.5$ | the cap | $79$ | $121$ | $95.0$ | $1.48$ | $2.14\times10^{-4}$ |
| $Z$ | $0.5$ | **stall at $k=13$** | $112$ | $65$ | $121.7$ | $1.16$ | $2.45\times10^{-4}$ |
| $P$ | $\mathbf{0.5}$ | the cap | $99$ | $107$ | $313.8$ | $\mathbf{0.45}$ | $2.79\times10^{-4}$ |
| $P$ | $0$ | **stall at $k=2$** | $144$ | $71$ | $286.5$ | $0.49$ | $2.19\times10^{-4}$ |
| $P_c$ detuned | $0.5$ | the cap | $113$ | $100$ | $313.8$ | $0.45$ | $2.58\times10^{-4}$ |
| $P_w$ **corrupted** | $0.5$ | the cap | $\mathbf{71}$ | $125$ | $322.0$ | $0.44$ | $2.99\times10^{-4}$ |

| clause | at six windows |
|---|---|
| **G1 · non-null** | **fails**: $99$ classical calls against $0.8\times\min(141,112)=89.6$ |
| **G2 · non-vacuous** | **passes**: every arm inside its own $\Theta r$ — the worst needs a constant of $3.34$ against the cold march's $\Theta=3.67$ — and the checkpoint's bound is $1.14\times$ its error |
| **G4 · marched** | **passes**: $50$ further classical steps move the checkpoint's answer at most $5.11\times10^{-4}$ against $2\Theta r=6.37\times10^{-4}$, and the coarse solver's $3.31\times10^{-4}$ against $6.67\times10^{-4}$ |
| **G5 · accounted** | **fails, twice**: $313.8$ against $141/1.5=94$, and against the coarse classical solver's $51.9$ |
| **G6 · loud** | **first half holds, second half fails**: the corrupted checkpoint's answer is inside the certificate, and it needed $71$ classical calls against the clean checkpoint's $99$ |

**The second half of G6 is the finding, and it is the one that decides the tier.** In sample the classical calls ordered as the checkpoint's integrity did — clean $25$, detuned $32$, no checkpoint $35$, cold $47$ — which read as a contribution that degrades with the checkpoint. Out of sample **a checkpoint with Gaussian noise on every weight tensor is the better cheap operator**, by $28$ classical calls. Whatever the shrunk column contributes to this iteration's rate, it is not what the checkpoint learned.

> **2026-09-12, Tier 50 (W205): this cell is one draw, and it is the best of twelve.** [[corrupted-checkpoint-and-jacobian-fidelity]] swept the corruption over twelve directions at this same magnitude: the counts run $\mathbf{71}$ to $\mathbf{140}$ about the clean checkpoint's $99$, so the $71$ above is the sample **minimum** and the $28$-call margin is an extreme value rather than an effect size. The clause is therefore **not measurable as posed** rather than failed — a one-draw Bernoulli trial on a quantity whose across-direction spread is $69$ calls. Ten of twelve directions do beat the clean checkpoint (sign test $p = 0.019$) and the median saves $13.5$ calls, so the *direction* of this paragraph survives and its *size* does not. **The last sentence stands, on a better argument**: the learned content is worth $13$ classical calls over the identity ($99$ against $Z$'s $112$), and perturbing the same weights by $3\%$ moves the count by $69$ — about five times more.

And the mechanism reproduces: where the unshrunk checkpoint stalls at $k=2$, $59.7\%$ of its error is in the lowest modes and $31.4\%$ is constant over each window, against $3$–$10\%$ and $\le0.2\%$ for the arms the cap stopped.

### 8.2.2 Twelve windows, and the constant the certificate is built on

Cold march: $267$ classical calls to $r=9.87\times10^{-5}$, at an error of $3.07\times10^{-4}$ (the reference records its error every fifth step, so this is the sample at step $265$). $\Theta=3.345$, read off the middle half of that march, so the bound at the stopping residual is $3.30\times10^{-4}$. Per-call costs: $P$ $0.296$, $Z$ $0.0365$, $C$ $0.0320$ of one classical step — **the checkpoint has crossed below a classical step at this size**, and the coarse classical solver is thirty times below it.

Each arm is judged against **its own** returned residual times $\Theta$, which is what §3 certifies; a fallback march can overshoot well below the stopping threshold, and then the bound it carries is the tighter one it actually earned.

| cheap map | $\alpha$ | stopped by | classical calls | cheap calls | cost, in classical calls | cold / cost | returned error | inside its own $\Theta r$ |
|---|---|---|---|---|---|---|---|---|
| *(cold march)* | — | converged | $267$ | $0$ | $267$ | $1.00$ | $3.07\times10^{-4}$ | — |
| $C$ | $0$ | the cap | $27$ | $347$ | $\mathbf{38.1}$ | $\mathbf{7.00}$ | $6.37\times10^{-5}$ | **no** — needs $3.45$ |
| $C$ | $0.5$ | the cap | $189$ | $124$ | $193.0$ | $1.38$ | $3.14\times10^{-4}$ | yes |
| $Z$ | $0.5$ | **stall at $k=13$** | $240$ | $65$ | $242.4$ | $1.10$ | $2.79\times10^{-4}$ | yes |
| $P$ | $\mathbf{0.5}$ | the cap | $222$ | $123$ | $258.4$ | $1.03$ | $2.93\times10^{-4}$ | yes |
| $P$ | $0$ | **stall at $k=2$** | $270$ | $82$ | $294.3$ | $0.91$ | $3.09\times10^{-4}$ | yes |
| $P_c$ detuned | $0.5$ | the cap | $237$ | $100$ | $266.6$ | $1.00$ | $3.05\times10^{-4}$ | yes |
| $P_w$ **corrupted** | $0.5$ | the cap | $\mathbf{138}$ | $125$ | $\mathbf{175.0}$ | $\mathbf{1.53}$ | $3.56\times10^{-4}$ | **no** — needs $3.81$ |

**Ranked by cost, the clean checkpoint is fifth of seven**: the coarse classical solver at $\alpha=0$ ($38.1$), the **corrupted** checkpoint ($175.0$), the coarse solver at $\alpha^\star$ ($193.0$), the composition layer with no checkpoint at all ($242.4$), then the checkpoint ($258.4$), the detuned checkpoint ($266.6$), and the unshrunk checkpoint ($294.3$) — which costs **more than not iterating at all**.

| clause | at twelve windows |
|---|---|
| **G1 · non-null** | **fails**: $222$ classical calls against $0.8\times\min(267,240)=192$, a margin of $1.16$ |
| **G2 · non-vacuous** | **fails**: two of seven arms return a state *outside* their own $\Theta r$ — the coarse solver at $\alpha=0$ by $3\%$ and the corrupted checkpoint by $14\%$. The checkpoint's own bound is $1.07\times$ its error |
| **G4 · marched** | **passes**: $30$ further classical steps move the checkpoint's answer at most $5.08\times10^{-4}$ against $2\Theta r=6.25\times10^{-4}$, and the coarse solver's $5.46\times10^{-4}$ against $6.34\times10^{-4}$ |
| **G5 · accounted** | **fails, twice**: $258.4$ against $267/1.5=178$, and against the coarse classical solver's $38.1$ |
| **G6 · loud** | **fails on both halves**: the corrupted checkpoint's answer is *outside* the certificate, **and** it needs $138$ classical calls against the clean checkpoint's $222$ |

**H3 is the hypothesis that broke, and it broke here rather than in the theorem.** $\Theta$ is read off one approach to $w^\star$ — the cold march's — and is then used on every other. The constant each arm actually needs, $\lVert w-w^\star\rVert/\lVert\Phi(w)-w\rVert$ at its returned state, is at most $3.45$ at two windows and $3.34$ at six, inside the $3.55$ and $3.67$ the cold march reported. At twelve it reaches $\mathbf{3.81}$ against a cold march reporting $\mathbf{3.35}$ — the estimate falls as the graph grows while the requirement rises, so the cold march's direction is becoming *less* representative of the others, not more. **Theorem 1 is untouched**: it fixes the limit of the iteration, and no arm's limit moved. What fails is the finite-residual certificate built on an estimated $\Theta$, and with it the claim that the guarantee is indifferent to the cheap map — the arm that breaks the bound worst is the corrupted checkpoint (**W208**).

**And it is the estimator, not the fluid.** `tests/test_tier48_defect_correction.py` pins the same failure in two dimensions with no physics in it: for $\Phi=\operatorname{diag}(0.6,\,0.95)$ the true constant is $20$ along the slow eigenvector and $2.5$ along the fast one; a march started almost entirely in the fast mode keeps the fast mode dominant across its whole middle half, so `theta_from_march` reads $2.5$ — and a state sitting on the slow eigenvector then lands outside the certificate that reading issues, by exactly $(1-0.6)/(1-0.95)$. The same test shows the reading bounding every iterate of the window it came from and **failing on the iterates just past it**, which is the drift above at $10^{-7}$ instead of $14\%$.

## 8.3 The verdict

**Pre-registered, and failed.** G1, G2, G5 and G6 all fail out of sample; G4 passes on both out-of-sample rungs; G3 asks for all of them at $N=6$ **and** $N=12$ and therefore fails. In sample, at the rung where $\alpha^\star$ was chosen, G1 and G2 passed and G5 already failed.

| clause | $N=2$ *(in sample)* | $N=6$ | $N=12$ |
|---|---|---|---|
| **G1 · non-null** | passes, $0.89$ | fails, $1.10$ | fails, $1.16$ |
| **G2 · non-vacuous** | passes | passes | **fails**, two arms outside |
| **G4 · marched** | not measured | passes | passes |
| **G5 · accounted** | fails | fails | fails |
| **G6 · loud** | no $P_w$ arm | fails, second half | **fails, both halves** |

**The gate was not moved to reach a positive, and the negative is the result.** The prediction recorded in §7 said G2 and G4 would pass, G1 was plausible and G5 would fail; what happened is worse than the prediction on two clauses — G1 fails out of sample, and G2 fails at the largest rung — and the reading of §7's last sentence stands unchanged: **the learned expert does not contribute meaningfully in this slot on this host.** The slot itself is real, and the map that pays in it is a $2\times$-coarse classical solver.

---

# 9. What this page does not claim

- **Not that a learned expert contributes.** §7 says what would show it and predicts it will not be shown on this host.
- **Not a trajectory certificate.** The certified object is a settled state. A transient, and any objective read off one before it settles — CS-10's horizon-dependent sensitivity — is outside it.
- **Not that the classical composed answer is the physics.** $\Phi$ carries its own composition defect; the certificate is about $\Phi$'s fixed point, as every classical column's is.
- **Not that the mechanism in §5 is established.** It is marked **[AI Inference]** and §8 carries the measurement that would contradict it.
- **Not that the certificate of §3 held.** It held at two and six windows and **failed at twelve**, on two arms of seven. The theorem it rests on is intact — Theorem 1 fixes the *limit* — but $\Theta$ is an estimate read off one approach to $w^\star$, and §8.2.2 measures it under-bounding two others. Anything downstream that wants this guarantee needs $\Theta$ from an operator norm rather than from a march (**W208**).
- **Not that the guarantee is indifferent to the cheap map.** That is true of the limit and was measured false of the finite-residual bound: the arm that breaks it worst is the corrupted checkpoint.
