# Atlas: Prior Art, Novelty, and the Scaling Argument

**Type:** Core Concept — Research Positioning (folder: `Atlas 0.1/common/`)
**Status:** Assessment, written 2026-08-19 against a direct challenge: *is this actually novel, what has been done before, what problem does it solve, and does it scale as experts are added?* Prior-art claims below were checked against sources rather than recalled; links are in §9.
**Related Concepts:** [[00-atlas-0.1-overview]], [[pfm-purpose-and-direction]], [[conservation-as-constraint-atlas-0.1]], [[edge-generation-atlas-0.1]], [[expert-library-atlas-0.1]], [[spec-wind-farm-wake-atlas-0.1]], [[physics-foundation-models]], [[regime-moe-architecture]], [[backbone-1.1]]

---

# 0. Verdict up front

| Component of the Atlas thesis | Novel? | Who did it first |
|---|---|---|
| Split a domain into subdomains coupled at interfaces | **No** | Schwarz alternating method, 1870; the entire domain-decomposition field |
| Couple heterogeneous solvers through declared interfaces with mapping, multi-rate stepping, and implicit iteration | **No** | Co-simulation / FMI; **preCICE** is a mature open-source implementation |
| Enforce **flux continuity at subdomain interfaces** as the conservation mechanism | **No** | **cPINN** (Jagtap, Kharazmi & Karniadakis, 2020) does exactly this; XPINN generalizes it to space–time |
| Compose a *library* of pretrained neural blocks instead of one big pretrain | **No, as of Jan 2026** | **CompNO** — a library of "Foundation Blocks," each an operator-specialized FNO |
| A universal, domain-independent interface vocabulary that makes composition energy-consistent | **No** | Bond graphs (Paynter, 1959); port-Hamiltonian systems theory; power-bond co-simulation |
| One model spanning many PDE families | **No** | Poseidon, Walrus, MPP, GPhyT |
| **Coupling *frozen, independently-pretrained, heterogeneous* neural experts and asking whether partitioned-coupling stability survives** | **Yes — open** | §2.1 |
| **Composition (not pretraining breadth) as the scaling axis of a physics foundation model, with error growth in interface count as the measured quantity** | **Yes — open** | §2.2 |
| **Field experts and lumped/algebraic experts as first-class peers in one differentiable graph** | **Yes, in this combination** | §2.3 |

**The short version: almost every *mechanism* in Atlas has been done. The *combination*, and the specific empirical question it makes askable, has not.** That is a weaker novelty claim than "new architecture," and a stronger research claim, because it is falsifiable. §7 says exactly what would falsify it.

---

# 1. What problem is actually being solved

**Not** "we cannot simulate a coupled multiphysics system." We can. Coupled aero–thermal–structural analysis of a race car, an aircraft, or a turbine is routine industrial practice, and preCICE, ANSYS System Coupling, and Simulink/Modelica do it today with conservation at the interfaces and multi-rate time stepping.

The problem is **the cost and the character of a single coupled evaluation**:

| | Classical coupled multiphysics | Composed neural experts |
|---|---|---|
| Wall-clock per coupled evaluation | hours to days | milliseconds to seconds |
| Differentiable end-to-end w.r.t. design parameters | no (per-solver adjoints exist; the *coupling* generally breaks the chain) | **yes, by construction** |
| Adding a new physical subsystem | new solver + new coupling adapter | new expert + declared ports |
| Trustworthy for certification | yes | **no** |

Two consequences follow, and they are the actual value proposition:

1. **Design search changes regime.** At hours per evaluation, engineering design is a human-in-the-loop process exploring tens to hundreds of configurations by design-of-experiments. At seconds per evaluation *with gradients*, it becomes an optimization problem over thousands to millions. This is a 4–6 order-of-magnitude change in the unit economics of search, not a speed improvement.
2. **Differentiability is the enabling property, not speed.** A fast but non-differentiable surrogate still leaves you doing derivative-free search. Gradients through the *whole coupled stack* — aero through thermal through structure — are what make the autonomous-design vision tractable at all.

**The honest scope statement:** Atlas does not aim to beat a CFD solver on accuracy, and should never claim to. It aims to be fast and differentiable enough to *search*, with the classical stack retained to *verify* the shortlist. That is how surrogates are legitimately used in industry, and it is the only deployment story that survives the certification problem.

---

# 2. What is genuinely novel

## 2.1 Partitioned-coupling stability theory does not cover neural subsolvers

This is the sharpest defensible gap, and it is not a gap in either parent field — it falls between them.

Co-simulation has a real, mature stability theory: Jacobi vs. Gauss–Seidel coupling schemes, explicit vs. implicit coupling, waveform relaxation, quasi-Newton acceleration of the interface fixed point (preCICE's IQN-ILS), and **energy-residual analysis via power bonds**, in which the residual energy injected at each coupling step is computed directly from the exchanged variables and used for adaptive step-size control. Every one of those results rests on assumptions about the subsolvers:

- each is a **consistent** discretization with a known order of accuracy;
- error is **controllable by reducing the step size**;
- the input–output map is **Lipschitz**, and usually monotone in the coupling variable;
- stability is analyzable because the subsolver's amplification factor is known.

**A frozen neural surrogate satisfies none of these.** It has no consistency order, no CFL condition, no Lipschitz bound you can state, and — the structural difference that matters most — **its error cannot be reduced by shrinking the macro step, because it was trained at a native $\Delta t$ and shrinking below that takes it out of distribution.** In classical co-simulation, the universal remedy for coupling instability is a smaller communication step. *That remedy is unavailable here.* Whatever replaces it does not currently exist.

**[AI Inference]:** this predicts a specific and testable failure mode — coupling iterations that diverge not because the physics is stiff but because the expert's response to a boundary perturbation is non-monotone. [[spec-wind-farm-wake-atlas-0.1]] §8.3 makes fixed-point non-convergence an explicit gate for exactly this reason, and it is the most informative single measurement in the case study.

## 2.2 Composition as the scaling axis, with a measurable scaling law

Every physics foundation model in the catalogue scales the same way: **pretrain one model on more physics**. Poseidon, Walrus, MPP and GPhyT differ in breadth and architecture, not in that strategy. CompNO (January 2026) is the closest thing to a compositional alternative, and the distinction is precise and worth stating carefully:

- **CompNO composes operators *functionally*, within one domain.** Its Foundation Blocks are specialized to differential operators — convection, diffusion — and are assembled through adaptation blocks into an evolution operator for a target PDE. The composition axis is **operator-splitting**: block ∘ block on the same field.
- **Atlas composes subsystems *spatially and physically*, across interfaces.** Its experts occupy disjoint regions and exchange conserved fluxes. The composition axis is **partitioning**.

These are orthogonal, and both could be true at once. But they answer different questions, and the question Atlas asks has no published answer:

> **Does composition error grow sub-linearly in the number of interfaces?**

If yes, a library of experts is a foundation model that scales by *addition* — each new expert extends coverage without retraining the others. If no, composition is a dead end regardless of how good the experts are, and pretraining breadth wins. **Nobody has measured this**, because measuring it requires a library of independently-pretrained heterogeneous experts and a coupling framework to compose them in, and no one has assembled both. That is the empirical contribution on offer.

## 2.3 Field experts and lumped experts as first-class peers

This is the claim that most directly supports the F1 vision, and it is the one a monolithic PDE foundation model **structurally cannot** make.

A real machine is not a field problem. An F1 car is a **system of coupled field and lumped subsystems**: aerodynamic fields, structural fields, and thermal fields, but also a motor torque map, a tyre model, a gearbox, control logic, an energy-recovery state machine. A PDE foundation model — Poseidon, Walrus, any of them — represents fields on grids. It has no representation for a torque curve, and no amount of additional pretraining gives it one, because the object is not a field.

Atlas's structure admits both. The precedent is already established twice over: the rocket's `rigid_body` expert is a closed-form ODE integrator, and the wind farm's `disk` expert is algebraic with zero parameters, exchanging momentum with a learned PDE expert across a declared interface ([[spec-wind-farm-wake-atlas-0.1]] §7.3). **Heterogeneity of mathematical type — PDE, ODE, algebraic map, lookup table, learned surrogate — is native to a composition framework and foreign to a monolithic one.**

This is exactly what co-simulation has always done, of course. **The novelty is not field+system coupling; it is field+system coupling where the field parts are learned, fast, and differentiable, so the whole assembly can be put inside an optimization loop.**

---

# 3. What is *not* novel, stated plainly

Overclaiming here would be the fastest way to waste the project's credibility. So:

- **"Conservation imposed at declared interfaces" is cPINN's 2020 contribution**, not Atlas's. cPINN decomposes the domain and enforces flux continuity in strong form along subdomain interfaces; XPINN generalizes to arbitrary space–time decomposition. [[conservation-as-constraint-atlas-0.1]] should cite this and position against it rather than presenting the idea as new. The genuine difference: **cPINN trains all subdomain networks jointly against a PDE residual, so the interface condition is a training loss on networks that grow up together. Atlas imposes it at inference on frozen experts that never met.** That is a real distinction, and it is also precisely why Atlas's version might not work — which is the point of testing it.
- **Typed edges are preCICE's coupling data.** preCICE already provides participants, meshes, named coupling data, mapping between non-matching meshes, serial/parallel explicit/implicit coupling, quasi-Newton interface acceleration, and — directly relevant — a **conservative vs. consistent mapping distinction** for whether a mapped quantity should preserve its integral or its pointwise values. Atlas has been reinventing a subset of this. It should instead **borrow**: §8 lists what.
- **The agent/chart metaphor is presentational.** It is a good name and an honest description, but a manifold atlas is not a mechanism, and no result follows from the metaphor.
- **Regime-based expert routing** is not new either; MoE conditioned on physical regime appears in the neural-PDE literature.

---

# 4. The scaling problem, and the fix

The user's question — *how does this expand when many more experts are added?* — has a specific failure mode that the current design walks into.

## 4.1 The $O(K^2)$ interface-contract problem

Atlas's edge types today are `heat`, `pressure`, `stress`, `fluid`, `momentum`, `mass`, `shear`, `conservation`. **This list mixes categories.** Some entries are quantities (mass, momentum), some are phenomena (heat, shear), one is a medium (fluid), and one is a constraint (conservation). It grew by accretion, one case study at a time — `shear` was added by the wind farm, and a battery case study would add `charge` and `species`, an EM case study would add `field`, and so on.

The consequence at scale: **if the interface contract is defined per expert-pair, $K$ expert families need up to $K(K+1)/2$ contracts.** At $K=4$ that is 10 and manageable. At the $K\approx15$–$20$ an F1 car needs, it is 120–210 hand-built adapters, and the framework collapses under its own integration burden long before the physics fails. *This, not accuracy, is what actually kills the vision.*

## 4.2 The fix: make the interface a power bond

Bond-graph theory (Paynter, 1959) and its modern descendant, port-Hamiltonian systems theory, solved this problem for classical modelling. The insight: **every physical domain exchanges energy through a conjugate pair of an *effort* and a *flow*, whose product is power.**

| Domain | Effort $e$ | Flow $f$ | $e\cdot f$ |
|---|---|---|---|
| Mechanical (translation) | force $F$ | velocity $v$ | power |
| Mechanical (rotation) | torque $\tau$ | angular velocity $\omega$ | power |
| Hydraulic / fluid | pressure $p$ | volumetric flow $Q$ | power |
| Electrical | voltage $V$ | current $I$ | power |
| Thermal | temperature $T$ | entropy flow $\dot S$ | power |
| Chemical | chemical potential $\mu$ | molar flow $\dot n$ | power |

Three properties follow, and together they are the scaling argument:

1. **The contract becomes per-quantity, not per-pair.** An expert declares which **ports** it exposes, in effort–flow pairs. Any two experts sharing a port type can be connected. **$O(K)$ port declarations replace $O(K^2)$ pairwise adapters.** Adding the twentieth expert costs the same as adding the fourth.
2. **Energy conservation across the whole graph becomes one checkable residual.** The co-simulation literature already does this: because the exchanged variables *are* a power bond, the residual energy injected by the coupling is computable from nothing but the coupling values, so you can observe directly if and where energy conservation is violated. That converts [[conservation-as-constraint-atlas-0.1]] from a per-edge special case into **a single global diagnostic that works for every edge in every future case study.**
3. **It connects to structure the project already has.** [[backbone-1.1]]'s core is symmetric + skew-symmetric — a port-Hamiltonian structure. Composition of port-Hamiltonian systems is itself port-Hamiltonian, and passivity composes. The theory for "interconnect these subsystems without creating energy" is not something Atlas needs to invent.

**[AI Inference], and the main recommendation of this page:** re-express Atlas's edge types as effort–flow pairs before a third case study adds more ad-hoc labels. Concretely, the wind farm's `momentum` edge is the mechanical pair $(F, v)$; `mass` is the hydraulic flow $Q$ with pressure $p$ as its conjugate effort; the rocket's `heat` is $(T,\dot S)$; `stress` is $(F,v)$ on a surface. The relabelling costs one editing pass now and is close to impossible once ten case studies depend on the current vocabulary. **This is the single highest-leverage change available to the design.**

## 4.3 What still does not scale automatically

Honesty requires listing what the power-bond fix does *not* solve:

- **Nondimensionalization mismatch.** Two experts trained by different groups on different reference scales will not agree at a port without a renormalization adapter. This is per-expert engineering, unavoidable, and roughly linear in $K$ — tolerable, but real.
- **Error accumulation.** §2.2's open question is untouched by the port vocabulary. A clean interface contract does not make composition accurate.
- **Regime coverage.** A port tells you what a coupling *is*; it does not tell you whether the expert on the other side was ever trained near that operating point. Abstention (Gap 11, [[pfm-purpose-and-direction]]) remains unsolved and becomes *more* pressing as $K$ grows, because with 20 experts the probability that at least one is out of distribution approaches 1.

---

# 5. The F1 vision, assessed

Taking the stated long-term goal seriously — full multiphysics of a race car, then agents autonomously improving the design.

**What is already possible today.** Coupled aero–thermal–structural simulation of a race car is existing industrial practice. Formula 1 aerodynamic development is additionally subject to regulated limits on CFD and wind-tunnel usage, which makes the compute constraint on design search unusually literal in this specific domain: teams are explicitly rationed. *(Whether a surrogate would fall inside or outside such a rule is a regulatory question this page does not attempt to answer, and the argument does not depend on it — every engineering organization is compute-limited in practice.)*

**What Atlas would change.** Not whether the car can be simulated, but how many designs can be evaluated and whether gradients are available. That is the whole claim, and it is enough.

**What Atlas would not change.** Verification. A neural multiphysics stack cannot be certified, and no result in this project will change that. The deployment story is: **search wide and cheap with the composed model; verify the shortlist with the classical stack.** Any framing that implies replacement rather than pre-screening is an overclaim and should be corrected wherever it appears.

**The autonomous-design layer is a separate research program.** Agents proposing and evaluating design modifications is design optimization and search — a different literature, different failure modes, different evaluation. It *depends* on the differentiable physics stack, which is a good reason to build the stack first, but merging the two claims weakens both. Keep them separate in every writeup.

**The strongest form of the F1 argument** is §2.3's: a car is a system of coupled field *and* lumped subsystems, monolithic PDE foundation models can only represent the field half, and composition is the only route that admits both. That argument does not depend on speed, on differentiability, or on beating anyone's accuracy — which makes it the one to lead with.

---

# 6. Where this leaves the two active case studies

Neither changes, but what each is *for* sharpens:

- **[[spec-wind-farm-wake-atlas-0.1]]** is the minimal instance of §2.3 — one learned field expert and one algebraic lumped expert exchanging a conserved quantity, with gate W4 measuring whether they agree. It is a field+system coupling test in miniature, which is a better description of its value than "a wake study."
- **[[case-study-rbc-decomposition-atlas-0.1]]** is the minimal instance of §2.1 — can frozen experts be coupled stably at all, with the physics held constant so nothing else can be blamed.
- Neither yet touches §2.2, which needs a **third** case study reusing experts from the first two. That is where the foundation-model claim actually gets tested, and it should be scheduled explicitly rather than left as "expansion."

---

# 7. What would falsify the thesis

Stated in advance, so the project cannot quietly move the goalposts:

1. **Composition error grows super-linearly in interface count.** Then the framework does not scale, and no expert library rescues it. *Measured by [[case-study-wind-farm-wake-2d-atlas-0.1]] Phase F's $N=2,3,5,8$ sweep — this is the single most important experiment in the plan.*
2. **Frozen experts cannot be coupled stably without joint fine-tuning.** Then it is not a modular library; it is a monolith with extra steps, and Walrus's strategy is simply better. *Measured by the fixed-point convergence gate.*
3. **Per-expert adapter engineering exceeds the cost of training a monolithic model on the coupled system.** Then composition loses on economics even if it works. *Measured by tracking integration hours per added expert, starting now — this number should be in the build log from the first expert, not reconstructed later.*
4. **A monolithic foundation model reaches the required physics coverage first.** Poseidon and Walrus are improving quickly. The defence is §2.3 — lumped subsystems — and if that defence ever stops being true, the composition thesis is in real trouble.

---

# 8. Concrete changes this analysis recommends

1. ~~**Re-express edge types as effort–flow pairs**~~ — **done 2026-08-19**, see [[port-algebra-atlas-0.1]]. The migration turned out to be a *correction*, not a rename: four of seven rocket edges were under-specified, and two `ROT` ports appeared in the wind farm that the old vocabulary could not express.
2. **Cite and position against cPINN/XPINN in [[conservation-as-constraint-atlas-0.1]]**, with the frozen-vs-jointly-trained distinction made explicit. The page currently reads as if the idea is new.
3. **Adopt preCICE's conservative-vs-consistent mapping distinction** explicitly for every edge. Whether a mapped interface quantity preserves its *integral* or its *pointwise values* is a decision Atlas is currently making implicitly, per edge, by accident.
4. **Replace the plain fixed-point relaxation of [[spec-wind-farm-wake-atlas-0.1]] §8.3 with quasi-Newton interface acceleration (IQN-ILS)** if plain relaxation needs more than ~6 iterations. This is solved technology; there is no reason to rediscover it.
5. **Add a global power-residual diagnostic** computed from coupling values alone, reported every macro-step, in every case study. One number that says whether the composition is creating or destroying energy.
6. **Track integration hours per added expert** from the first expert onward (§7.3).
7. **Schedule the third case study explicitly** as the reuse test (§6), rather than leaving it as open-ended expansion.

---

# 9. Sources checked

- preCICE — partitioned multi-physics coupling library: [precice.org](https://precice.org/), [github.com/precice/precice](https://github.com/precice/precice), [preCICE v2 paper](https://open-research-europe.ec.europa.eu/articles/2-51)
- cPINN — conservative PINNs, flux continuity at subdomain interfaces: [Jagtap et al., CMAME 2020](https://ui.adsabs.harvard.edu/abs/2020CMAME.36513028J/abstract), [code](https://github.com/AmeyaJagtap/Conservative_PINNs)
- CompNO — compositional neural operators, Foundation Blocks library: [arXiv:2601.07384](https://arxiv.org/abs/2601.07384), [MDPI](https://www.mdpi.com/2076-3417/16/2/972); multi-dimensional extension [arXiv:2605.11691](https://arxiv.org/abs/2605.11691)
- Power bonds and energy conservation in co-simulation: [arXiv:1602.06434](https://arxiv.org/pdf/1602.06434), [arXiv:1606.05168](https://arxiv.org/abs/1606.05168), [Co-Simulation: A Survey (ACM CSUR)](https://dl.acm.org/doi/abs/10.1145/3179993)
- Partitioned port-Hamiltonian coupling: [arXiv:2603.16424](https://arxiv.org/pdf/2603.16424)
- Multiple Physics Pretraining: [arXiv:2310.02994](https://arxiv.org/pdf/2310.02994); Towards a Physics Foundation Model: [arXiv:2509.13805](https://arxiv.org/pdf/2509.13805)

---

## See Also

- [[composition-error-theory]] — §2's named gap developed into a bound, and the place where this page's end-to-end differentiability claim would finally be spent (adjoint error localization)
- [[pfm-purpose-and-direction]] — the vision this page stress-tests
- [[conservation-as-constraint-atlas-0.1]] — needs the cPINN citation and positioning from §3
- [[edge-generation-atlas-0.1]] — where the effort–flow relabelling of §4.2 would land
- [[expert-library-atlas-0.1]] — the closed-form-expert precedent §2.3 rests on
- [[spec-wind-farm-wake-atlas-0.1]] — §8's recommendations 3, 4 and 5 apply to it directly
- [[backbone-1.1]] — the port-Hamiltonian structure §4.2 connects to
- [[physics-foundation-models]], [[regime-moe-architecture]] — the monolithic-scaling alternative
