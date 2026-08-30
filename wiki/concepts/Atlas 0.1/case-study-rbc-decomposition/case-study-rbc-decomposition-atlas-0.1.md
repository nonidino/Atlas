# Atlas 0.1: Case Study — Domain-Decomposed 2D Rayleigh–Bénard (Zero-New-Data Interface Test)

**Type:** Core Concept — Alternate First Instantiation / Plan of Action
**Status:** Proposed, not built. **Superseded as the primary substitute by [[case-study-wind-farm-wake-2d-atlas-0.1]]**, which meets the same no-new-data / no-new-experts constraints on a *real engineering system* rather than a synthetic cut. This page remains live as the **single-expert control**: the degenerate case (one governing family, one expert, monolithic gold standard) that isolates communication-layer error more cleanly than any engineering scenario can, and is worth running as a Phase-B control there. Written 2026-08-19 as a substitute first target for [[case-study-rocket-ascent-2d-atlas-0.1]], which is blocked at M2 on ~2,400 CPU-core-hours of corpus generation ([[impl-atlas-0.1-compute-and-training-budget]]) plus three from-scratch experts ([[impl-atlas-0.1-phase2-experts]]).
**Related Concepts:** [[00-atlas-0.1-overview]], [[port-algebra-atlas-0.1]], [[edge-generation-atlas-0.1]], [[conservation-as-constraint-atlas-0.1]], [[expert-library-atlas-0.1]], [[agent-definition-atlas-0.1]], [[unet-hierarchy-atlas-0.1]], [[impl-atlas-0.1-phase1-scaffold]], [[noether-1.0-rbc]], [[physics-simulation-datasets]], [[case-study-rocket-ascent-2d-atlas-0.1]]

---

# 0. The constraint, stated exactly

The requirement is a case study that:

1. **Generates no new training data.**
2. **Trains no new experts.**
3. Still exercises the thing Atlas exists to test: **multiple agents communicating through typed edges while staying physically plausible.**
4. Is designed to be **extended by later case studies that add agents through the same communication framework** — so the communication layer, not the physics coverage, is the deliverable.

Requirements 1 and 2 together rule out every scenario in [[pfm-purpose-and-direction]] — moon base, wildfire, battery, reentry, wind farm — for the same reason the rocket is blocked: each needs at least one governing family with no trained expert, and interface supervision that [[physics-simulation-datasets]] §4 argues is unobtainable from public corpora.

**The resolution is to stop looking for a new scenario and instead cut an existing one.** Take a system for which a trained expert and a generated corpus *already exist locally*, and partition its single domain into several agents. The agents are subdomains; the typed edges are the internal cuts; the monolithic trajectory the corpus already contains is the exact ground truth for the composed multi-agent rollout.

---

# 1. Recommendation

> **Decompose 2D Rayleigh–Bénard convection into a stack of three agents — hot boundary layer, convective bulk, cold boundary layer — coupled by two horizontal typed edges (heat + stress), with the already-trained Noether-1.0-RBC checkpoint frozen and instantiated once per agent as a single weight-shared expert.**

Everything it needs already exists:

| Requirement | What supplies it | Cost |
|---|---|---|
| Trained expert | **[[noether-1.0-rbc]]** — 5.50 M params, trained, Nu error ≤4.8% single-step, rollout stable ≥600 steps over $\mathrm{Ra}\in[3\!\times\!10^4,3\!\times\!10^5]$ | already trained; **frozen**, zero training runs |
| Interior training data | not needed — nothing is trained | zero |
| Evaluation ground truth | `rbc_train.npz`, `rbc_ra_sweep.npz` ([[noether-1.0-rbc]] §3), already generated | zero |
| More ground truth on demand | `src/noether/data/rbc_solver.py`, the self-contained IMEX Boussinesq DNS, validated against $\mathrm{Nu}\propto\mathrm{Ra}^{0.29}$ | minutes of CPU, and **optional** |
| **Interface flux fields** | **derived** from the stored whole-domain state on a horizontal cut — post-processing, not simulation (§2) | zero |
| Scaffold | [[impl-atlas-0.1-phase1-scaffold]], already built and green on M0/M3 | zero |

The three expert slots hold the **same frozen weights**. There is exactly one expert in the library, used three times. That is what "no new experts" means here, and it is also the right scientific choice: with the physics held identical across agents, **any error at the seams is attributable to the communication layer and to nothing else.**

---

# 2. Why the interface-data gap does not apply here

[[physics-simulation-datasets]] §4 concludes that interface flux fields "genuinely cannot be found," because no public corpus stores

$$\left.\left(\rho u_n,\; p,\; q_n,\; \boldsymbol\sigma\!\cdot\!\mathbf n\right)\right|_{\Gamma_{ij}}$$

on a *declared* boundary between two named subdomains. That conclusion is correct, and this case study does not contradict it — it sidesteps it.

**The precise statement is narrower than the page's phrasing.** Interface data is unobtainable when the two sides of $\Gamma_{ij}$ come from **different solvers**, because no single stored field spans the cut. It is *trivially* obtainable when the cut is **internal to one monolithic field**: given whole-domain state $(u_x,u_y,T)$ on a grid, the flux through any horizontal line $y=y_\Gamma$ is a finite-difference evaluation of data already on disk,

$$q_n(x,t)\;=\;u_y(x,y_\Gamma,t)\,T(x,y_\Gamma,t)\;-\;\partial_y T(x,y_\Gamma,t),\qquad
\sigma_{ny}\;=\;-p+2\Pr\,\partial_y u_y,\qquad
\phi_m\;=\;u_y(x,y_\Gamma,t),$$

with $p$ recoverable from the vorticity–streamfunction formulation the corpus already stores. **No solver is run. The labels are a `numpy` expression over an existing `.npz`.**

**[AI Inference]:** this is a general and reusable escape hatch, not an RBC accident. *Any* monolithic PDE corpus can be converted into a labelled multi-agent interface corpus by declaring cuts through it. What that buys is supervision for the **homogeneous** interface problem — same equation on both sides — which is exactly the sub-problem a communication framework must solve first. What it cannot buy is the **heterogeneous** interface (combustion↔structure), where §4's argument stands untouched. The correct reading of §4 is therefore *"heterogeneous interface data is unobtainable"*, and this page proposes spending the free half of that distinction before paying for the expensive half.

---

# 3. The agent graph

## 3.1 Geometry and the partition

Domain $[0,\Gamma]\times[0,1]$, $\Gamma=2$, grid $128\times64$, periodic in $x$, no-slip and Dirichlet-$T$ at $y=0$ (hot) and $y=1$ (cold) — identical to [[noether-1.0-rbc]] §1, unchanged.

Three agents, cut at two horizontal planes:

$$
\alpha:\ \text{hot boundary layer},\ y\in[0,\,y_1]\qquad
\beta:\ \text{convective bulk},\ y\in[y_1,\,y_2]\qquad
\gamma:\ \text{cold boundary layer},\ y\in[y_2,\,1]
$$

with $y_1,\,1-y_2 \approx \delta_T(\mathrm{Ra})$, the thermal boundary-layer thickness $\delta_T\simeq \tfrac12\mathrm{Nu}^{-1}$. At $\mathrm{Ra}=10^5$, $\mathrm{Nu}\approx4.5$, so $\delta_T\approx0.11$ — about 7 of 64 rows. Round to the grid and hold the cut fixed per episode.

**This partition is not arbitrary, and that matters.** It follows [[agent-definition-atlas-0.1]]'s granularity heuristic on a real physical boundary: the boundary layers are **conduction-dominated** ($\mathrm{Pe}_{\text{local}}\lesssim1$, near-parabolic), the bulk is **advection-dominated** (plumes, near-hyperbolic). The three agents therefore occupy genuinely different local regimes while sharing one governing equation — the exact configuration [[expert-library-atlas-0.1]] says should be *one expert conditioned continuously*, not three experts. So the case study also tests that design invariant, for free.

## 3.2 The typed edge list

$$
\alpha\!-\!\beta:\ \texttt{MECH},\ \texttt{THERM},\ \texttt{ADVEC}(h_0)\qquad
\beta\!-\!\gamma:\ \texttt{MECH},\ \texttt{THERM},\ \texttt{ADVEC}(h_0)
$$

*Relabelled 2026-08-19 to [[port-algebra-atlas-0.1]]'s closed port set; previously `heat, stress, conservation`.* `MECH` carries the traction–velocity pair $(\sigma_{ny},v)$ — the old `stress`; `THERM` carries $(T,\,q_n/T)$ with the **conductive** part $q_n=-\partial_yT$; `ADVEC` carries the mass flux $u_y$ with enthalpy as its passenger — which is where the **advective** part $u_yT$ of the old `heat` label actually lives. The old single `heat` label conflated the two transport mechanisms, and separating them is what makes the Nusselt gate of §4 well-posed: $\mathrm{Nu}$ is precisely the sum of the `ADVEC` enthalpy passenger and the `THERM` conductive flux. `conservation` is gone as a type — every port conserves by construction.

**Because this is Boussinesq, `MECH` and `THERM` are not independent** (buoyancy couples them), which the old three-label list obscured. The global power residual $\mathcal R(t)$ must close across both together.

Per [[expert-library-atlas-0.1]]'s cut rule, `heat` and `stress` here are outputs of **the same governing system** (Boussinesq couples them through the buoyancy term), so they are two channels of one expert — the same verdict the rule gives for the rocket's $a\!-\!b$, reached independently. Good sign for the rule.

## 3.3 What is deliberately *not* in scope

- **No global non-edge field.** Buoyancy is internal to the Boussinesq system, so unlike the rocket's gravity ([[global-fields-and-topology-atlas-0.1]]) there is no external uniform field to inject. One fewer mechanism under test.
- **No topology mutation, no staging, no composability.** Same non-goals as the rocket case study.
- **No expert routing.** One expert. The MLP gate is an identity map and should be asserted as such.
- **No new physics families.** By construction.

---

# 4. The four grading signals, all free

This is the case study's real advantage over the rocket: it has a **monolithic gold standard**, which the rocket never had at any phase.

**(G1) Trajectory error against the monolith.** Run the frozen expert once on the *undivided* domain — that is exactly Noether-1.0-RBC in its normal operating mode — and again as three coupled agents from the same initial condition. Report per-field RMSE between the two, and both against DNS. **The multi-agent rollout should be no worse than the monolithic one.** Any gap is communication-layer error, cleanly isolated. No other Atlas test has this property.

**(G2) Nusselt height-invariance — a conservation gate needing no labels.** In statistically steady RBC the horizontally-averaged vertical heat flux is independent of height:

$$\mathrm{Nu}(y)\;=\;\big\langle u_yT\big\rangle_x-\partial_y\big\langle T\big\rangle_x\;=\;\text{const in }y.$$

So $\mathrm{Nu}$ measured on the $\beta$ side of $y_1$ must equal $\mathrm{Nu}$ on the $\alpha$ side. **Define the flux residual $r_\Gamma = |\mathrm{Nu}_\alpha(y_1)-\mathrm{Nu}_\beta(y_1)|/\mathrm{Nu}$** and make it the acceptance gate for [[conservation-as-constraint-atlas-0.1]]. This is a first-principles physical identity, not a fitted label — it costs nothing, and it is precisely the "physically plausible" criterion the case study exists to establish.

**(G3) Zero net volume flux.** Incompressibility over any horizontal line gives $\int_0^\Gamma u_y(x,y_\Gamma)\,dx=0$ exactly. The composed rollout must preserve it. [[noether-1.0-rbc]]'s stream-function decoder head enforces $\nabla\!\cdot\!\mathbf u=0$ *by construction* **within** an agent; whether that survives composition across an interface is an open question and a sharp one. This is the single most informative measurement in the plan — a hard architectural constraint (encoding-spectrum level 5) meeting a soft interface.

**(G4) Long-horizon stability.** [[noether-1.0-rbc]] reports stable divergence-free rollout ≥600 steps monolithically. The composed system must match that horizon, or the seams are injecting energy.

---

# 5. The one real risk, and the design response

**The frozen expert has never seen an interface boundary.** It was trained on a full domain whose $y$-boundaries are no-slip walls with Dirichlet $T$. Agent $\beta$'s top and bottom are now neither — they are open surfaces with prescribed incoming flux. Applied naively, the expert is out of distribution on exactly the tokens the case study is about.

Two responses, both zero-cost; choosing between them is Phase B's job:

- **(R1) Overlapping (Schwarz) agents with halo tokens.** Agents overlap by $h$ rows; the edge layer supplies each agent's halo tokens from its neighbour's interior. The expert then always sees a locally complete stencil that *looks* like the interior of a full domain — no OOD boundary condition at all. This is classical additive-Schwarz domain decomposition wearing Atlas's clothes, which makes the framework's convergence behaviour interpretable against 40 years of DD literature.
- **(R2) Non-overlapping agents with flux-BC tokens.** The declared interface carries $(q_n,\sigma_{ny})$ as boundary-condition tokens, per [[edge-generation-atlas-0.1]] Mechanism A as already built. Truer to Atlas's declared-edge thesis, and the mechanism the rocket needs — but it is the OOD path.

**Run R1 first as the positive control, then R2 as the real test.** If R2 matches R1 on G1–G4, the declared-edge contract is validated. If only R1 works, the finding is that halo exchange is a *required* part of the framework — itself a first-class result, and one that would change [[edge-generation-atlas-0.1]].

**Second, smaller risk:** the outer walls stay walls, so $\alpha$ and $\gamma$ each retain one in-distribution boundary. Only $\beta$ is doubly-open. Expect $\beta$ to fail first; instrument for it.

---

# 6. Phased plan

**Phase A — Interface extraction harness (no model).**
Write `cut_interface(npz, y_cut) → (q_n, σ_ny, φ_m)(x,t)`. Verify G2 and G3 hold on **DNS data itself**, at several cut heights, before any model is involved. If the ground truth violates height-invariance beyond tolerance, either the tolerance is wrong or the run is not statistically steady — find out here, not in Phase E. *Exit gate: $r_\Gamma < 2\%$ and $|\int u_y\,dx| < 10^{-6}$ on DNS.*

**Phase B — Single-agent OOD probe (teacher-forced interface).**
Run the frozen expert on agent $\beta$ alone with interface conditions **supplied from DNS at every step**, under R1 and under R2. This is the go/no-go for the whole case study and involves no coupling at all. *Exit gate: teacher-forced $\beta$ tracks DNS to within the monolithic model's own single-step error, under at least one of R1/R2.*

**Phase C — Wire three agents into the existing scaffold.**
Reuse [[impl-atlas-0.1-phase1-scaffold]] verbatim; swap the identity placeholder experts for the frozen Noether-1.0-RBC forward. Two declared edges, MLP gate asserted identity, 2-level hierarchy. **This also resolves the scaffold's open action item 4** ([[atlas-0.1-implementation-log]], 2026-08-09: whether experts must interleave with message-passing layers) — with a real expert in the slot, intra-agent mixing becomes measurable instead of hypothetical.

**Phase D — Conservation constraint.**
Wire flux-matching at both edges per [[conservation-as-constraint-atlas-0.1]]. Measure $r_\Gamma$ with the constraint **off** and **on**; the delta is the constraint's entire justification, and it has never been measured anywhere in this project.

**Phase E — Coupled rollout vs. the monolith.**
Full G1–G4 sweep across $\mathrm{Ra}\in\{3\!\times\!10^4,10^5,3\!\times\!10^5\}$ in-distribution, plus $\mathrm{Ra}=10^6$ from `rbc_ra_sweep.npz` as the OOD stress case. Report multi-agent vs. monolithic vs. DNS as three curves, always.

**Phase F — Generality of the framework (the actual deliverable).**
This is what makes it a *generalist* communication framework rather than one wiring diagram, and every axis below is free:
- **Vary agent count:** $N=2,3,5,8$ horizontal slabs. Composition error should grow sub-linearly in the number of interfaces; if it grows super-linearly the framework does not scale, and that is the headline finding.
- **Vary cut placement:** cuts *on* the boundary-layer edge vs. cuts *through* a plume — physics-aligned vs. physics-blind partitions, held against G1.
- **2D partition:** a $2\times2$ tiling introduces vertical interfaces and, critically, **corner tokens shared by three agents** — the first genuine test of edge *composition* rather than a chain.
- **Heterogeneous resolution:** boundary-layer agents at finer token density than the bulk, exercising cross-resolution edges with the same weights.
- **Multi-rate stepping:** boundary layers at $\Delta t/4$ against the bulk's $\Delta t$, exercising the subcycling machinery the rocket needs, on a system where the answer is known.
- **Heterogeneous conditioning:** agents at different $\mathrm{Ra}$ in the conditioning vector — the cheapest available proxy for "different regimes talking."

**Phase G — Freeze the regression suite and hand off.**
Phases A–F become the frozen interface-layer regression suite that every later case study must still pass when it adds an agent. Per [[impl-atlas-0.1-phase5-expansion]]'s reuse levels this is R1-grade reuse: the communication layer ships unchanged, and the next scenario only adds experts.

---

# 7. What this proves, and what it does not

**Proves:** that declared typed edges, halo/flux exchange, flux-matching conservation, multi-rate and multi-resolution coupling, and $N$-agent composition produce a stable, conservative, physically plausible rollout — graded against a monolithic gold standard, at zero data and zero training cost.

**Does not prove:** anything about heterogeneous expert routing, cross-family coupling, the MoE gate, transfer between donor experts, or the heterogeneous interface-data problem of [[physics-simulation-datasets]] §4. Those remain the rocket's job.

**The honest framing to carry into any writeup:** this is a *communication-layer* validation with the physics held constant, deliberately. It is the control experiment the rocket case study skipped — and running it first means that when the rocket's coupled rollout eventually misbehaves, the interface layer is already exonerated.

---

# 8. Alternatives considered

| Alternative | Why not first |
|---|---|
| **2D incompressible NS** from Noether 1.1's P1/P2 corpus, subdomain-split | Also zero-cost and a valid backup — but it has **no height-invariant flux identity**, so G2 disappears and physical plausibility loses its label-free gate. Periodic-everywhere also means no in-distribution boundary survives the cut. Keep as the second decomposition target inside Phase F. |
| **Poseidon frozen on PDEgym Euler**, subdomain-split | Free weights and free data, but adds the donor-adapter unknown ([[incremental-transfer-roadmap]] Stage 1) on top of the interface unknown — two simultaneous unknowns, violating [[00-atlas-0.1-overview]] invariant 7. |
| **Rocket, with experts bootstrapped from Poseidon and no new data** | Does not clear the bar: [[impl-atlas-0.1-phase2-experts]]'s gates (the choked throat above all) are ungradeable without generated data, and `reacting_flow` has no donor at all. |
| **Wind farm** ([[impl-atlas-0.1-phase5-expansion]]) | Needs a new electromagnetics expert. Fails requirement 2 outright. |
| **1D Burgers** ([[noether-1.0]]), split | Cheapest of all and a reasonable smoke test — but 1D interfaces are *points*, so there is no interface **field** to exchange, which is the entire mechanism under test. Use as a Phase-A unit test, not the case study. |

---

## See Also

- [[case-study-rocket-ascent-2d-atlas-0.1]] — the blocked target this substitutes for; unchanged, and still the eventual goal
- [[noether-1.0-rbc]] — the frozen expert and the existing corpora
- [[physics-simulation-datasets]] — §4's interface-data argument, narrowed by §2 above
- [[conservation-as-constraint-atlas-0.1]] — the constraint that G2/G3 finally make measurable
- [[edge-generation-atlas-0.1]] — Mechanism A, and what R1-vs-R2 would change about it
- [[impl-atlas-0.1-phase1-scaffold]] — the built scaffold this reuses verbatim
- [[impl-atlas-0.1-compute-and-training-budget]] — the cost wall that motivated this page
- [[00-atlas-0.1-overview]] — invariant 7 (minimize simultaneous unknowns), of which this page is an application
