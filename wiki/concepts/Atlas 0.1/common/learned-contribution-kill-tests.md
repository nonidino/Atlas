# What a Learned Expert Can Contribute — fourteen formulations, and the kill tests that chose between them

**Type:** Concept page — **divergence and kill record** (folder: `Atlas 0.1/common/`)
**Status:** measured 2026-09-11 and 2026-09-12, Tier 48, phases 1 and 2. `scripts/w202_kill_tests.py`, `out/w202/w202.json`, `tests/test_tier48_learned_contribution.py`. The formulation it converges on, with its theory and its pre-registered gate: [[defect-correction-learned-operator]]. Worklist rows from **W202**.
**Related:** [[defect-correction-learned-operator]] · [[substitution-campaign-checkpoint]] · [[case-study-neural-interface-atlas-0.1]] · [[case-study-bounded-donor-atlas-0.1]] · [[case-study-learned-pair-atlas-0.1]] · [[epsilon-halo-measurement]] · [[master-error-bound]] · [[composition-error-theory]] · [[control-observability]] · [[interaction-horizon]] · [[case-study-scaling-ladder-atlas-0.1]] · [[expert-donor-survey]] · [[f1-pathmap-and-end-goal]] · [[gap-worklist]]

---

# 0. The result, in one paragraph

After forty-seven tiers no learned expert had been admitted with a meaningful contribution, and [[substitution-campaign-checkpoint]] traced why to one chain: a checkpoint fixed at one resolution has no same-class reference pair, so no $\tau$, no $\sigma$, no $\varepsilon_{\text{tol}}$, no $\beta_{\min}$ (W95). This page writes down fourteen formulations that each drop one of the assumptions behind that chain — what is certified, what kind of guarantee, what the referent is, what role the learned model plays — and kills them cheaply, on `wake_array`'s geometry at two windows, in about five minutes of compute. **Seven die**, each for a stated and mostly measured reason; **two are not chosen** — killed on what they cost rather than on their guarantee; **three are left standing and unmeasured**, because the test that would decide them is not cheap; **one survives as a detector and contributes nothing on its own**; and **one is converged on** — the checkpoint as the approximate operator in Stetter's defect correction, where it can change the rate of a classical computation and cannot change its answer. Two findings came out of the kill tests that are about no candidate: **the classical monolith's settled state was not isolated**, because its outflow ring is carried as state; and **CS-S1's cost ratio of $61$ was an arrangement**, not a property of the checkpoint — on CS-7's own timing the checkpoint runs from $4.9\times$ a classical window's cost at one window to $0.25\times$ at twenty-four.

---

# 1. The knot, and what each formulation drops

The chain [[substitution-campaign-checkpoint]] §3.3 names, with the assumption underneath each link:

| link | the assumption it rests on |
|---|---|
| no same-class reference pair | **the referent must be a same-class monolith** |
| so no $\tau$ and no $\sigma$ | **what is certified is the expert** |
| so no $\varepsilon_{\text{tol}}$ | **the guarantee is deterministic and a priori** |
| so no $\beta_{\min}$ | **the learned model is an agent at a seam** |

Every candidate in §3 drops at least one of the four, and §3's table says which.

---

# 2. Two facts that changed before the first candidate was written

## 2.1 The cost ratio is an arrangement

CS-S1 priced a Poseidon-T forward at $61$ subdomain solves. That is one thread, a batch of four, against a split-step `WindowNS` window at a macro-step of $0.05$ with ten sub-steps — the cheapest classical window in the vault. **CS-7 timed the same checkpoint in the wake array's arrangement** — eight threads, one batched forward per macro-step, against the as-built window at twenty-one sub-steps (`out/w100/w100.json`, `timing`):

| windows | `WindowNS`, ms per agent per macro-step | Poseidon-T | checkpoint over classical |
|---|---|---|---|
| $1$ | $81.5$ | $399.7$ | $4.90$ |
| $2$ | $88.8$ | $270.1$ | $3.04$ |
| $6$ | $125.5$ | $154.4$ | $1.23$ |
| $12$ | $329.1$ | $109.8$ | $\mathbf{0.334}$ |
| $24$ | $340.8$ | $86.1$ | $\mathbf{0.253}$ |

**From twelve windows the checkpoint is cheaper per agent-step than the classical window it would replace**, on this laptop, in one process. The ratio is a property of the batch and the host and is quoted as that.

## 2.2 The learned column is not the least accurate cheap column available

CS-7's long march, the state against the classical monolith after $120$ macro-steps from the freestream, as the rms of the velocity difference (`long_march`, `defect_at` over the cell count):

| windows | Poseidon-T's composed column | the classical composed column, exposed + projected | wall for $120$ steps: checkpoint / classical column / monolith |
|---|---|---|---|
| $2$ | $0.0822$ | $0.0891$ | $34$ / $19$ / $21$ s |
| $6$ | $0.0959$ | $0.0937$ | $64$ / $78$ / $76$ s |
| $12$ | $0.121$ | $0.0830$ | $112$ / $419$ / $438$ s |

The R10-compliant classical column is **no closer to its own monolith than the checkpoint's column is at two and six windows**, and at twelve the checkpoint is $1.5\times$ further off at a quarter of the wall time. And the checkpoint's error **saturates** — at six windows $0.074$, $0.093$, $0.097$, $0.096$, $0.096$, $0.096$ at steps $20$ to $120$ — so it is a bias, not a drift.

**[AI Inference]:** together these reopen formulations in which the learned column's *output* replaces classical work — which CS-S1's $61$ had closed without the arrangement it was measured in.

## 2.3 The ledger this tier measures, in its own process

Every cost clause below and in [[defect-correction-learned-operator]] §7 is charged against this, not against §2.1: the same five maps on the same developed state, interleaved, the minimum over three repeats of the mean over three calls, as a ratio to one classical monolith macro-step in the same process (`out/w202/w202.json`, `cost`).

| windows | monolith $\Phi$ | Poseidon column | composition layer, no checkpoint | $2\times$-coarse classical | classical composed column |
|---|---|---|---|---|---|
| $2$ | $45.7$ ms | $3.43$ | $0.110$ | $0.197$ | $0.932$ |
| $6$ | $157.3$ ms | $2.01$ | $0.149$ | $0.132$ | $1.31$ |
| $12$ | $1791.5$ ms | $\mathbf{0.296}$ | $0.0365$ | $\mathbf{0.0320}$ | $1.02$ |

Two readings, and the second is the one that decides §6's choice. **The checkpoint crosses below a classical step between six and twelve windows** — §2.1's crossing, re-measured against the *monolith* rather than against a window. And **the coarse classical solver is cheaper still on every rung**: $17.4\times$ cheaper than the checkpoint at two windows, $15.2\times$ at six, $9.2\times$ at twelve — more than a cell count would suggest, because the monolith's own cost per cell rises with the grid on this machine while the coarse solve stays in cache. **The gap narrows as the graph grows**, which is the one trend measured here that runs the learned side's way; at twelve windows it is still an order of magnitude.

---

# 3. The fourteen candidates

| # | role of the learned model | kind of guarantee | drops | kill test | verdict |
|---|---|---|---|---|---|
| **F1** | agent, certified against a converged classical referent of another class | deterministic, a priori | same-class referent | reading: W177 | **died** — it would certify the null replacement |
| **F2** | coarse propagator in parareal | exact convergence, finite termination | agent | reading and cost | **not chosen** — no work saved |
| **F3** | low-fidelity model in a multifidelity Monte Carlo estimator | unbiased, statistical | deterministic; expert certified | not run | **alive, unrun** |
| **F4** | screen for a design search | conformal, finite-sample | expert certified; deterministic | theory | **died** |
| **F5** | agent certified on its training subspace | deterministic, on a subspace | expert certified whole | reading: CS-S2 | **died** |
| **F6** | tripwire: lead-time self-consistency | referent-free, **one-sided** | referent | measured §4.6 | **survives as a detector**, contributes nothing |
| **F7** | any producer; the **answer** is certified against the PDE | a posteriori, relative energy | referent; expert certified | measured §4.5 | **died** |
| **F8** | preconditioner / transmission operator from the checkpoint's JVP | Krylov exactness | agent | reading: W177 | **died** |
| **F9** | agent, lightly fine-tuned on classical windows | — | frozen | priced §4.7 | **unmeasured, affordable** |
| **F10** | a whole native-domain precursor behind a one-way port | same-domain referent | the domain is a cut | reading | **not run** |
| **F11** | proposal generator in trust-region model management | convergence to a classical critical point | agent | reading | **not chosen** |
| **F12** | QoI surrogate calibrated per regime | statistical, exchangeable | deterministic | reading and §2.2 | **died on "meaningful"** |
| **F13** | **start** of the classical march to the settled state | Banach residual | agent | measured §4.3 | **died** |
| **F14** | **approximate operator in defect correction** | exact consistency; classical residual certificate | expert certified; agent; referent | measured §4.4 | **converged on** |

**Unconventional**, in the sense of appearing on no page of this vault before: F6's use of the checkpoint's own time-conditioning as a proof of failure; F7's relative-energy certificate on the assembled answer; F10's native-domain precursor; and F14's reading of a frozen checkpoint as an approximate operator, with a shrink toward the null element.

## 3.1 Each candidate — claim, hypotheses, argument, and where it lives or dies

**F1 · the borrowed-class referent.** *Claim*: $\tau_i:=\lVert\mathcal E_{\text{learned}}-\mathcal S_{\text{classical}}\rVert$ against a converged classical expert of another class defines $\varepsilon_{\text{tol}}$, hence $\beta_{\min}$, hence a verdict. *Hypotheses*: the referent is converged (checkable here); the probe state is the deployment state (assumed — CS-8 measured the thresholds moving $155\%$ with it). *Dissolves* W95's "no pair"; *moves* it to which referent at which state. *Dies on* [[substitution-campaign-checkpoint]] §3.1: $\lVert\Delta\rVert/\lVert S_i\rVert=0.95$–$1.01$ at every agent-side, so a defined $\beta_{\min}$ would issue a verdict about an expert within $5\%$ of absent. W76, with a number.

**F2 · parareal with a learned coarse propagator.** *Claim*: the iterates converge to the classical serial trajectory, the first $k$ time slices exact after $k$ iterations whatever the coarse propagator is (Lions, Maday & Turinici 2001; Gander & Vandewalle, *SISC* 29, 2007). The propagator changes the rate through $\operatorname{Lip}(F-G)$, not the start, so CS-S1's ceiling does not bind. *Cost model*: $K$ iterations cost $KN$ fine and $(K+1)N$ coarse slices, never less than the $N$ fine slices of the serial march. *Not chosen*: it buys parallel depth and no work, and a design search spends its processors across designs before it spends them across time; it also carries parareal's documented non-monotone convergence for transport, which is what a wake is.

**F3 · multifidelity Monte Carlo.** *Claim*: an unbiased estimate of $\mathbb E[J_C]$ over an uncertain input, with the learned column as the low-fidelity model and variance reduced at the optimal allocation by $\bigl(\sqrt{1-\rho^2}+\sqrt{\rho^2 c}\bigr)^2$ (Peherstorfer, Willcox & Gunzburger, *SISC* 38, 2016). *Dissolves* W95 entirely: validity never depends on the learned model. *Moves* it to the correlation $\rho$ and the cost ratio $c$ — and its W76 control is an analytic wake model that costs nothing. *Not run*: the decisive test is sixteen or more paired rollouts at twelve windows or more, about an hour, and a deterministic design ranking is not an expectation.

**F4 · a conformal shortlist.** *Claim*: $P(\text{best design}\in\text{shortlist})\ge1-\alpha$, finite-sample and distribution-free (Fannjiang, Bates, Angelopoulos, Listgarten & Jordan, *PNAS* 119, 2022, for the feedback shift a design loop induces). *Dies in theory*: the event *the best survives* needs coverage over the whole pool simultaneously, which distribution-free needs about $(1-\alpha)M$ calibration evaluations of $M$ — no saving — and an optimiser's proposals are exchangeable with a random calibration set only if the design density is known.

**F5 · certified on a subspace.** *Claim*: along the far-field directions NeuberNet was trained on, its operator matches the classical one to $8$–$12\%$; decline elsewhere. *Dies on* [[case-study-bounded-donor-atlas-0.1]]: two directions of $87$; the classical elastic referent is cheaper than the donor; the plastic branch, where the donor would pay, has no referent here; and the donor is local-only with a seam that is a fiction (W184).

**F6 · lead-time self-consistency, a proof of failure.** *Claim*: Poseidon-T takes its lead time as an input, and the exact flow is a semigroup, so with $P$ the checkpoint and $S$ the classical map,

$$\delta_{\text{SC}} := \lVert P(u,2\Delta t)-P(P(u,\Delta t),\Delta t)\rVert \quad\Longrightarrow\quad \delta_{\text{SC}}-\lVert S(u,2\Delta t)-S(S(u,\Delta t),\Delta t)\rVert \;\le\; e_{2\Delta t}+e_{\Delta t,\Delta t}$$

by the triangle inequality. A large $\delta_{\text{SC}}$ proves the checkpoint wrong by at least that much, with no referent. It is step-doubling error estimation (Hairer, Nørsett & Wanner) applied to a learned operator's time input. *Hypothesis*: the classical scheme's own step-doubling gap is small (checkable here). *Dissolves* half of W69's complaint about `validity` — failure becomes provable at runtime. *Moves* to power: **the identity map is exactly self-consistent and wrong by the flow's whole change**, so a null replacement is invisible to it.

**F7 · the relative-energy certificate on the answer.** *Claim*: for two-dimensional incompressible Navier–Stokes, with $e=u_{\text{rec}}-v$ the distance from a smooth divergence-free reconstruction of the computed trajectory to the true solution and $R$ the reconstruction's residual, the nonlinear term $(e\cdot\nabla e,e)$ vanishes and

$$\frac{\mathrm d}{\mathrm dt}\lVert e\rVert \;\le\; s_\star\lVert e\rVert+\lVert R\rVert,\qquad s_\star=\max_x\bigl(-\lambda_{\min}(\mathrm{sym}\,\nabla u_{\text{rec}})\bigr),\qquad \lVert e(T)\rVert\le\int_0^T e^{\int_s^T s_\star}\lVert R(s)\rVert\,\mathrm ds$$

— the relative-entropy argument (Dafermos 1979), and for learned surrogates Hillebrecht & Unger (IJCNN 2022). It certifies the assembled answer whatever produced it, so W95, R10's halo and CS-8's state dependence all dissolve. *Moves* to two things: the reconstruction must resolve time at the learned model's lead spacing, and $e^{s_\star T}$ must stay finite over the horizon that matters.

**F8 · a learned preconditioner.** *Dies on* W177: the checkpoint's block is $0.8$–$9\%$ of the classical one, so as a preconditioner it is the zero operator.

**F9 · light adaptation.** *Claim*: fine-tuning on classical windows restores the boundary response and the one-step fidelity the frozen checkpoint lacks. *Unmeasured*, because it needs a training run; *priced* in §4.7, and inside this tier's own limit.

**F10 · a native-domain precursor.** *Claim*: W95 binds because a learned expert's domain is a cut of the problem; on its whole native domain — a periodic box — a classical referent of the same domain exists, and a one-way port carries its output into a classical region. *Moves* to chaos: a turbulent precursor's trajectory claim expires at $T_{\text{pred}}$, and [[master-error-bound]] §9 says the deliverable must then be statistical, a theory branch this vault does not have. *Not run.*

**F11 · trust-region model management** (Alexandrov, Dennis, Lewis & Torczon 1998). *Claim*: iterates converge to a critical point of the classical objective while the learned model proposes the steps. *Not chosen*: CS-10 already has the classical adjoint exact at $4.3$–$6.7$ forward evaluations, so a learned gradient has nothing to save unless it is both cheaper and aligned, and measuring alignment is a design-family study.

**F12 · per-regime calibration.** *Claim*: $P(\lvert J_L-J_C\rvert\le\hat q)\ge1-\alpha$ for states exchangeable with the calibration set. *Dies on "meaningful"*: the calibrated error is the checkpoint's model error, which §2.2 puts at $0.08$–$0.12$ in rms velocity — a third to a half of the wake's own fluctuation — and an optimiser breaks the exchangeability the guarantee needs.

**F13 · a learned start for the settled state.** *Claim*: a classical march contracting at $q$ per step saves $\ln(d_{\text{cold}}/d_{\text{learned}})/\ln(1/q)$ steps, which near $q=1$ is many — so CS-S1's one-sweep ceiling, derived at a contraction of $0.011$, would not bind. *Moves* to the relaxation's shape. *Measured* in §4.3.

**F14 · defect correction with the checkpoint as the approximate operator.** Stated, proved and gated on [[defect-correction-learned-operator]]. *Dissolves* W95 for the answer, CS-S1's ceiling and W76's blind spot. *Moves* to the checkpoint's Jacobian on the slow modes, to boundary data as problem data, and to cost against classical coarsening. *Measured* in §4.4.

---

# 4. The kill tests

On `wake_array`'s geometry at two windows — one rotor, $128\times240$ cells, $\mathrm{Re}_D=255$ — with the classical monolith as the referent and its outflow ring pinned (§4.2). All in one process per stage, about five minutes in all.

## 4.1 The control every number leans on

The Poseidon map in the driver is a rewrite of `w100_scaling_ladder.composed_step` with the time step as a parameter. Compared at the native step on a developed state: **bitwise identical**. The classical map against the long march's own monolith advance: **bitwise identical**. And the checkpoint moves the state $1.75\times10^{-2}$ in rms beyond the composition layer with the checkpoint replaced by the identity, so it is doing something. A failure here would have stopped every stage below it.

## 4.2 The referent — and a finding about no candidate

The monolith settles in $186$ steps to a one-step residual of $8.9\times10^{-12}$, bitwise identical on replay. Its error-to-residual ratio over the middle half of that march — the window `theta_from_march` uses, which excludes the transient and the round-off floor — is at most $\Theta=3.55$ and at least $1.99$, over $93$ steps.

**And its settled state was not unique.** `WindowNS.step_batch` with `bc0=None` holds every ring at its input value, and the freestream band resets only the inlet and the two laterals, so the outflow column is whatever the starting state put there, forever. The classical march from the checkpoint's settled state:

| outflow column | distance to the reference settled state | its own one-step residual there |
|---|---|---|
| left as state | $5.87\times10^{-2}$ after $320$ steps — $29\%$ of the cold distance | $\mathbf{1.6\times10^{-16}}$ |
| pinned to the freestream | $1.80\times10^{-5}$, inside $10^{-4}$ of the cold distance in $75$ steps | $6.3\times10^{-6}$ |

**The unpinned march did not fail to converge; it converged exactly, somewhere else.** Its residual is at machine zero while it sits a third of the cold distance from the reference — the checkpoint's settled state leaves the outflow column $0.203$ away from the freestream, and the classical map then holds that column forever. **The classical map's fixed points are a family parameterised by the outflow column, and $\Theta$ is infinite along it.** From the freestream the column is freestream at every step, so every reference trajectory this vault has published is unaffected; but any statement about *the* settled state — and every residual certificate — is about a declared problem, and on this graph the boundary data had not been declared as problem data. Every number below pins the column.

## 4.3 F13 — a learned start buys nothing

Classical calls from four starts to a distance of $10^{-1}$, $10^{-2}$, $10^{-3}$ and $10^{-4}$ of the cold distance, $0.199$:

| start | its distance | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ |
|---|---|---|---|---|---|
| the freestream (cold) | $0.199$ | $25$ | $31$ | $46$ | $68$ |
| **the checkpoint's settled state** | $0.0778$ | $15$ | $41$ | $52$ | $\mathbf{75}$ |
| the null expert's settled state | $0.0960$ | $13$ | $26$ | $45$ | $61$ |
| smooth noise at the checkpoint's distance | $0.0712$ | $9$ | $42$ | $55$ | $80$ |

**A start $2.6\times$ closer costs seven more classical steps**, and the composition layer without the checkpoint gives a better start than the checkpoint does. The ordering at loose tolerance is the ordering of the starts' distances; at tight tolerance it is not, because relaxation in an open advective domain is limited by **transit** — a start is flushed out in about one crossing of the domain whatever its size — and the checkpoint's error sits partly in structures that leave more slowly than the freestream's. F13 dies, and CS-S1's ceiling turns out to hold one level up for a different reason: not a fast contraction but no contraction to exploit.

## 4.4 F14 — defect correction

The full table is [[defect-correction-learned-operator]] §8.1. Its shape, at a stopping residual of $9.14\times10^{-5}$ where the cold march needs $47$ classical calls:

- **The unshrunk checkpoint stalls.** Its residual rises at the first two outer iterations; the stall rule hands it to the classical march, and the answer is the classical one.
- **Shrunk halfway to the null element it converges in $25$ classical calls** — and the detuned checkpoint needs $32$, the composition layer without the checkpoint falls back with $35$, and the cold march needs $47$. In sample, **the saving is the checkpoint's and degrades with it** — and **that reading does not survive the next rung**: at six windows a checkpoint with Gaussian noise on every weight tensor needs $71$ classical calls against the clean one's $99$ ([[defect-correction-learned-operator]] §8.2.1, **W205**).
- **A classical solver on a grid twice as coarse converges at every shrink**, in $21$ or $22$ classical calls.
- **Every arm returns a state inside $\Theta r=3.24\times10^{-4}$.**

**What the stalled iterates' errors are made of**, against the arms still converging when the iteration cap stopped them:

| arm, where it stopped | its error | share in the lowest modes, $\lvert k_x\rvert,\lvert k_y\rvert\le2$ | share uniform in $x$ | share constant per window |
|---|---|---|---|---|
| checkpoint, unshrunk, stalled at $k=2$ | $7.99\times10^{-2}$ | $\mathbf{0.724}$ | $0.431$ | $0.232$ |
| detuned checkpoint, unshrunk, stalled at $k=4$ | $1.03\times10^{-1}$ | $\mathbf{0.784}$ | $0.632$ | $0.503$ |
| composition layer alone, unshrunk, stalled at $k=2$ | $1.74\times10^{-1}$ | $0.021$ | $0.221$ | $0.009$ |
| composition layer alone, halfway, stalled at $k=14$ | $1.37\times10^{-2}$ | $0.059$ | $0.118$ | $0.000$ |
| the four arms at the cap with shrink $0.8$ | $1.6$–$3.8\times10^{-3}$ | $0.06$–$0.15$ | $0.06$–$0.08$ | $\le10^{-4}$ |

**Two different failures.** Where the checkpoint's column stalls, its error is in the slowest, largest structures — three quarters in the lowest modes, a quarter to a half constant over each window — which is what a column that holds each window's incoming mean and transports on a periodic padded domain would get wrong against a monolith that relaxes both through its boundary and its viscosity. Where the composition layer alone stalls, the error is **not** in the lowest modes: that map has no viscosity at all, and what it cannot relax is everything the monolith damps. **[AI Inference]**, and the two mechanisms are read off error structure rather than off the Jacobians, which were not assembled.

## 4.5 F7 — valid per step, and dead over a transit

$s_\star$ from the strain field: $3.43$ at step $5$ of the cold march, $3.34$ at step $20$, and $3.34$ at the settled state. The reconstruction is the cubic Hermite interpolant of two macro-steps and the scheme's own time derivative — the divergence-free part of the momentum right-hand side, with the ring and the band held.

| step | the flow's one-step change | certificate on **exact** classical data (its floor) | checkpoint: certificate / true error | persistence: certificate / true error |
|---|---|---|---|---|
| $5$ | $2.42\times10^{-2}$ | $3.63\times10^{-3}$ | $5.25\times10^{-2}$ / $2.05\times10^{-2}$ — $2.56\times$ | $3.58\times10^{-2}$ / $2.42\times10^{-2}$ — $1.48\times$ |
| $20$ | $1.47\times10^{-2}$ | $\mathbf{4.05\times10^{-2}}$ | $7.12\times10^{-2}$ / $3.23\times10^{-2}$ — $2.20\times$ | $4.53\times10^{-2}$ / $1.47\times10^{-2}$ — $3.08\times$ |

**Valid on every row and within $3.1\times$ of the truth for one step** — so a residual certificate on the assembled answer is not vacuous per step, for a learned column or anything else. It dies twice anyway. **Its floor at step $20$ exceeds the checkpoint's own error**, because a cubic across a macro-step of $0.2$ at a Courant number of $6.4$ does not resolve the transport, so it cannot tell a checkpoint step from an exact one there. And **over one transit of the domain, $7.5$ time units, its growth factor is $e^{s_\star T}=10^{10.9}$** at the settled state's $s_\star=3.34$.

> **The instrument was wrong first, and the correction is recorded.** The first run took the time derivative from one tiny monolith step, which projects the *state*; a reconstructed intermediate state is not exactly solenoidal, so the "derivative" carried divergence over step size. It read a floor of $3.1$ on exact data against $3.0$ for persistence — the instrument could not tell the two apart. Replaced by the projected right-hand side, which projects the *rate*.

## 4.6 F6 — a real lower bound, blind to the null replacement

Disks off, from the settled state, one step at $2\Delta t$ against two at $\Delta t$:

| | value |
|---|---|
| $\delta_{\text{SC}}$ for the checkpoint | $2.04\times10^{-2}$ |
| the classical scheme's own step-doubling gap | $1.60\times10^{-3}$ |
| the checkpoint's true errors: one step / two steps / the double step | $3.89$, $4.84$ and $4.78$, each $\times10^{-2}$ |
| the bound $\delta_{\text{SC}}-\text{gap}\le e_{2\Delta t}+e_{\Delta t,\Delta t}$ | **holds**, at $0.195$ of the right-hand side |
| the identity map: $\delta_{\text{SC}}$ / true error at the double step | $0$ / $4.33\times10^{-2}$ |

**A referent-free proof that the checkpoint is wrong by at least about $9\times10^{-3}$ per step**, within $5\times$ of the truth — and **exactly silent on a map that does nothing.** A tripwire with that blind spot cannot be the guarantee; it can sit beside one.

## 4.7 F9 — the price of adaptation

One forward of Poseidon-T on a batch of two $128\times128$ windows: $0.173$ s. Forward and backward through all $20.8$M parameters: $0.630$ s. **Two thousand iterations: $0.35$ h**, before an optimiser step or a data pipeline. Light adaptation on classical windows generated here fits inside this tier's two-hour limit; it was not run, because F14's theory says what a Jacobian must do and nothing here says fine-tuning on one-step errors would teach it that.

---

# 5. The dead, and why

| # | died of |
|---|---|
| F1 | the admit it would issue is the null replacement ($\lVert\Delta\rVert/\lVert S_i\rVert=0.95$–$1.01$) |
| F4 | *the best survives* needs simultaneous coverage, which costs as many classical evaluations as it saves |
| F5 | two faithful directions of $87$, and no referent where the donor would pay |
| F7 | a floor above the checkpoint's error at step $20$, and a growth factor of $10^{11}$ over one transit |
| F8 | a block $0.8$–$9\%$ of the classical one is a zero preconditioner |
| F12 | a calibrated error a third to a half of the wake is not a design signal |
| F13 | transit-limited relaxation: a start $2.6\times$ closer costs seven more classical steps |
| F2, F11 | not killed — **not chosen**: no work saved, and an exact adjoint already in hand |

---

# 6. What survived, and the choice

- **F3** is the one formulation whose guarantee is untouched by everything measured here, and the one not run. It stays on the worklist with its decisive test priced.
- **F6** is a working one-sided detector and belongs beside any guarantee, not in place of one.
- **F9** is affordable and was not attempted.
- **F14 is converged on**, for three reasons in order: its answer is certified by the classical map whatever the checkpoint does (Theorem 1 there), which is the only guarantee here that survives both a biased checkpoint and a transport-dominated flow; its null replacement is an identity rather than a supposition, so *non-null* is a subtraction in one currency; and it was the only candidate in which the checkpoint measurably did something the composition layer alone did not.
- **And the third of those reasons did not survive its own out-of-sample runs.** [[defect-correction-learned-operator]] §8.2 records the checkpoint beaten by the composition layer alone at six windows and at twelve, and beaten by a corrupted copy of itself on both. The first reason — Theorem 1, the limit — still holds exactly. The *second* half of it, the finite-residual certificate that converts a residual into an error bound, **failed at twelve windows** on two arms of seven, because its constant is estimated from one approach to the settled state (**W208**). So what this page's choice bought is a real slot with a real entry condition, an exact null element, and a guarantee whose hypothesis is now measured rather than assumed.

---

# 7. What this page does not claim

- **Not that the dead are dead everywhere.** F7 is valid per step; F13 would pay in a closed or diffusion-dominated domain; F2 pays wherever processors outnumber designs. Each died *here*, for the reason in §5.
- **Not a survey of learned experts.** One checkpoint family, one geometry, one Reynolds number, two windows.
- **Not that the mechanisms in §4.4 are established.** Error structure, not assembled Jacobians.
- **Not that F14 contributes.** [[defect-correction-learned-operator]] §7 says what would show it.
