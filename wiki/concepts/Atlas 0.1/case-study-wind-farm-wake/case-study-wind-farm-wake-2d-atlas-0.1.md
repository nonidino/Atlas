# Atlas 0.1: Case Study — 2D Wind-Farm Wake Interaction (Engineering Scenario, No New Experts or Data)

**Type:** Core Concept — Alternate First Instantiation / Plan of Action
**Status:** Proposed, not built. Written 2026-08-19 as the **buildable substitute** for [[case-study-rocket-ascent-2d-atlas-0.1]], which is blocked on ~2,400 CPU-core-hours of corpus generation ([[impl-atlas-0.1-compute-and-training-budget]]) and three from-scratch experts, all three of them blocked by combustion ([[impl-atlas-0.1-phase2-experts]]).
**Related Concepts:** [[00-atlas-0.1-overview]], [[port-algebra-atlas-0.1]], [[f1-pathmap-and-end-goal]], [[expert-library-atlas-0.1]], [[edge-generation-atlas-0.1]], [[conservation-as-constraint-atlas-0.1]], [[global-fields-and-topology-atlas-0.1]], [[agent-definition-atlas-0.1]], [[impl-atlas-0.1-phase1-scaffold]], [[impl-atlas-0.1-phase5-expansion]], [[pfm-purpose-and-direction]], [[physics-simulation-datasets]], [[case-study-rbc-decomposition-atlas-0.1]]

---

# 0. The screen

Two hard constraints: **no new training data, no new experts.** A third, softer but load-bearing: the scenario must be a **real engineering system**, because the point of Atlas is a multi-agent framework that later case studies extend by adding agents — and agents that correspond to real components are what make that extension meaningful.

What is actually available to build with:

| Expert family | Trained weights available? | Source |
|---|---|---|
| Incompressible NS / turbulent shear | **Yes, twice over** | Noether 1.1 P2 checkpoint (2D incompressible NS, trained through the push-forward curriculum); [[poseidon-pde-foundation-model]]; [[walrus-paper]] |
| Compressible Euler / gas dynamics | **Yes** | Poseidon (PDEgym: Riemann, Kelvin–Helmholtz, Gaussian bumps), Walrus |
| Buoyant convection (Boussinesq) | **Yes** | [[noether-1.0-rbc]], 5.50 M params, trained, $\mathrm{Ra}\in[3\!\times\!10^4,3\!\times\!10^5]$ |
| Closed-form mechanics (ODE/algebraic) | **N/A — no training needed** | the precedent set by the rocket's `rigid_body` expert ([[expert-library-atlas-0.1]]) |
| Combustion / reacting flow | **No** | the rocket's blocker |
| Structural / thermoelastic | **No** | [[physics-simulation-datasets]] §3.4 — genuinely thin, no usable public corpus |
| Electromagnetics | **No** | nothing in the catalogue |
| Electrochemistry, rarefied/Knudsen | **No** | nothing |

So a buildable engineering scenario is one that decomposes **entirely into fluid families plus closed-form mechanics.** That is a narrow window, but it is not empty, and the screen below finds exactly one scenario that fits without contortion.

## 0.1 Why the named alternatives fail, and what would unblock each

| Scenario | Blocker | What would unblock it |
|---|---|---|
| **Wildfire** ([[pfm-purpose-and-direction]] ex. 2) | Combustion **and** radiative transport — the rocket's blocker plus a second one. Also Gap 9 (radiative reach is not adjacency) and Gap 11 (ignition is safety-critical). | A combustion expert; BLASTNet pretraining ([[physics-simulation-datasets]] §3.2) is the cheapest route |
| **Compartment / tunnel fire smoke** | Genuinely closer — fire-safety practice already replaces combustion with a *prescribed* heat-release curve, so no combustion expert is needed, and the buoyant plume is Boussinesq. **But** a fire plume runs at $\mathrm{Ra}\sim10^{10}$ against the RBC expert's trained $\le3\!\times\!10^5$ — five orders out of distribution, which is not a coupling test, it is an extrapolation test wearing a coupling test's clothes. | RBC expert retrained or fine-tuned at high $\mathrm{Ra}$; the turbulent-generative path [[noether-1.0-rbc]] §4.7 left unbuilt |
| **Aircraft (2D airfoil)** | AirfRANS is real public data and Poseidon has a demonstrated airfoil task (SE-AF) — but both are **steady**, which removes the rollout, and the rollout is what a coupling framework has to prove. Aeroelasticity, the interesting multi-agent version, needs the missing structural expert. | A structural expert; or accept a steady-state-only test, which forfeits the main claim |
| **Car (external aero + thermal)** | DrivAerStar/CarBench are 3D and industrial; the radiator/conjugate-heat version needs solid conduction, which has no expert | A structural/conduction expert |
| **EV battery thermal runaway** ([[pfm-purpose-and-direction]] ex. 3) | Electrochemistry + particle/many-body. Two missing families. | Dynami-CAL structural distillation ([[incremental-transfer-roadmap]] Stage 3) plus an electrochemistry expert |
| **Atmospheric reentry** (ex. 4) | Rarefied/continuum crossover within one agent — the hardest target in the whole roadmap by the roadmap's own assessment | Stage 4 shared trunk |
| **Offshore wind farm** as specced ([[impl-atlas-0.1-phase5-expansion]]) | Needs `electromagnetic` (generator) **and** `wave_structure` (foundation, sea surface). Two new families. | see below — **this is the one that survives a scope edit** |

## 0.2 The edit that makes one of them buildable

The wind farm's two missing experts are both attached to parts of the machine that **are not the physics under test**. The generator converts torque to current; the foundation resists waves. Neither participates in the wake interaction that is the scenario's actual multi-agent content.

Delete them. Take the farm **onshore**, cut a **horizontal plan-view slice at hub height**, and replace the rotor with a **closed-form actuator disk** — which is not a concession but standard practice in wind-farm aerodynamics, and is precisely the move [[expert-library-atlas-0.1]] already sanctioned when it made the rocket's `rigid_body` expert closed-form rather than learned.

What remains is a real, industrially important engineering problem — **wake-induced power loss and wake steering in a turbine array** — with exactly two experts, one of which is already trained and one of which has no parameters at all.

---

# 1. Recommendation

> **A 2D hub-height plan view of two (then $N$) wind turbines in line, decomposed into seven agents: inflow, two actuator-disk rotors, near wake, far wake, exit wake, and the bypass corridor beside the wakes — coupled by typed momentum/mass edges, using one frozen incompressible-NS expert instantiated five times and a zero-parameter actuator-disk expert instantiated twice.**

| Requirement | What supplies it | Cost |
|---|---|---|
| Fluid expert | Noether 1.1 P2 checkpoint (2D incompressible NS), **frozen**; [[poseidon-pde-foundation-model]] as the fallback donor if the local checkpoint is not at P2 | already trained |
| Rotor expert | Actuator-disk momentum theory — algebraic, **zero parameters** | none |
| Training data | none — nothing is trained | zero |
| Physical-plausibility gates | **closed-form theory**: Betz limit, actuator-disk momentum balance, Jensen and Bastankhah–Porté-Agel analytical wake models | zero |
| Reference trajectories | monolithic run of the same frozen expert; optionally a handful of 2D incompressible NS runs with disk forcing | zero, then minutes–hours |
| Scaffold | [[impl-atlas-0.1-phase1-scaffold]], built and green on M0/M3 | zero |

**Two experts, one library entry each, neither trained by this case study.** The heterogeneity that matters — a PDE expert and an algebraic expert exchanging a conserved quantity across a declared interface — is present from day one, and it is a scaled-down copy of the rocket's hardest coupling (`rigid_body` integrating the net force every flow expert produces).

---

# 2. Physics

## 2.1 The fluid agents

2D incompressible Navier–Stokes in the hub-height plane, with the turbines entering as body-force sinks:

$$\partial_t\mathbf u+(\mathbf u\cdot\nabla)\mathbf u=-\nabla p+\nu\nabla^2\mathbf u+\mathbf f_{\text{disk}},\qquad \nabla\cdot\mathbf u=0.$$

The plan view is deliberate and it is the reason the OOD risk is tolerable: at hub height there is **no ground**, so the dominant out-of-distribution feature of an atmospheric flow — a wall-bounded shear layer — is absent from the slice. What remains is a free shear flow with an inflow and an outflow, much closer to the homogeneous 2D turbulence the fluid expert was trained on than any vertical section would be.

## 2.2 The rotor expert, in full

One-dimensional actuator-disk momentum theory. With freestream $U_\infty$, axial induction factor $a$, disk area $A$ (a width in 2D):

$$U_{\text{disk}}=U_\infty(1-a),\qquad U_{\text{wake}}=U_\infty(1-2a),$$
$$T=2\rho A U_\infty^2\,a(1-a),\qquad C_T=4a(1-a),\qquad C_P=4a(1-a)^2 .$$

$C_P$ is maximized at $a=1/3$, giving the **Betz limit** $C_P^{\max}=16/27\approx0.593$. The theory is valid for $a\lesssim0.4$; beyond that the momentum balance breaks down and an empirical $C_T$ correction is used, which is a boundary the case study should stay inside rather than model.

Every quantity above is algebraic. The expert has no weights, no training, and no data. It sits at **level 5 of the physics-encoding spectrum** — a hard architectural constraint — exactly where [[expert-library-atlas-0.1]] placed `rigid_body`.

## 2.3 Why this is a real interface, not a formality

The disk expert *needs* $U_\infty$ — a quantity the fluid expert owns — and the fluid expert *needs* $\mathbf f_{\text{disk}}$, a quantity the disk expert owns. Neither can step without the other. That is a genuine bidirectional typed edge with a conserved quantity (momentum) crossing it, and both sides compute the same quantity by different means, so **the two can be checked against each other numerically** (§4, G2). Very few coupling tests have that property.

---

# 3. The agent graph

## 3.1 Agents

$$
I:\ \text{inflow corridor}\qquad R_1:\ \text{rotor 1 (disk)}\qquad N:\ \text{near wake, }0\text{–}3D\qquad F:\ \text{far wake, }3\text{–}7D
$$
$$
R_2:\ \text{rotor 2 (disk)}\qquad W:\ \text{exit wake}\qquad B:\ \text{bypass corridor}
$$

Seven agents — the same count as the rocket, by coincidence rather than design, which makes the scaffold's existing token and edge budgets a reasonable starting point.

The near/far split is not cosmetic. The near wake is dominated by **shear-layer roll-up and vortex breakdown**; the far wake is **self-similar, Gaussian in deficit, and dominated by turbulent entrainment**. Two different local regimes, one governing family — the configuration [[expert-library-atlas-0.1]] says is *one expert conditioned continuously*, not two experts. As in the RBC study, this tests that invariant for free.

$B$, the bypass corridor, is the physically essential agent and the one a naive decomposition omits: **wake recovery is entrainment of momentum from the bypass flow**, so a wake agent with no bypass neighbour cannot recover at all and will drift to a permanent deficit. If the framework gets this wrong the failure is unmissable, which makes it a good test.

## 3.2 Typed edges — ports

Fifteen edges, typed with [[port-algebra-atlas-0.1]]'s closed port set (relabelled 2026-08-19 from `momentum`/`mass`/`shear`/`conservation`):

$$
I\!-\!R_1,\; R_1\!-\!N,\; I\!-\!N,\; I\!-\!B^\pm,\; N\!-\!F,\; F\!-\!R_2,\; R_2\!-\!W,\; F\!-\!W:\quad \texttt{MECH},\ \texttt{ADVEC}
$$
$$
N\!-\!B^\pm,\; F\!-\!B^\pm,\; W\!-\!B^\pm:\quad \texttt{MECH}\ \text{(tangential)}\qquad R_1,R_2\ \text{shafts}:\quad \texttt{ROT}\ \text{(unconnected)}
$$

Three things the relabelling changed, none of them cosmetic:

- **`shear` was not a separate quantity.** It is the tangential component of the same traction $\mathbf t=\boldsymbol\sigma\!\cdot\!\mathbf n$ the streamwise edges carry normally — one `MECH` port either way.
- **`conservation` stopped being a type.** Every port conserves by construction. What distinguishes the four disk-face ports is not a label but the structural fact that **one side is closed-form** — an identity the algebraic expert satisfies exactly — which is what makes their residual *enforceable* rather than merely measurable, and gives [[conservation-as-constraint-atlas-0.1]]'s machinery its first test against a side that cannot be wrong.
- **Two `ROT` ports appeared that the old vocabulary could not express** — the shaft power each rotor extracts. Under the old scheme this was an absence; as an *unconnected port* it is a measurable power flow, and rung 7 of [[f1-pathmap-and-end-goal]] connects it to a generator rather than integrating one. **This is the mechanism by which the case study grows toward the full-vehicle target: each new subsystem is a port connection, not an integration project.**

The full edge table with interface curves is in [[spec-wind-farm-wake-atlas-0.1]] §5.

## 3.3 Deliberate non-scope

- **No generator, no tower, no foundation, no sea.** These are what would need new experts; removing them is the entire point of the scope edit.
- **No stratification.** A convective or stable atmospheric boundary layer would bring in the Boussinesq expert and is the single most attractive *extension* (§6, Phase G) — but adding it now would introduce a second unknown alongside the interface layer, violating [[00-atlas-0.1-overview]] invariant 7.
- **No fatigue, no decade timescales.** [[impl-atlas-0.1-phase5-expansion]] §2.2's fast/slow split is a genuinely new mechanism; it stays deferred.
- **No yaw / wake steering initially.** It arrives in Phase F as a generality axis, because it is the cleanest way to make an edge's *existence* state-dependent.
- **No 3D.** Same tractability reasoning as everywhere else in this project.

---

# 4. Grading — five signals, four of them free

**(G1) Decomposed vs. monolithic.** Run the frozen fluid expert once over the undivided domain with both disk forcings applied directly, and again as the seven-agent graph. Any difference is communication-layer error with nothing else mixed in. Free, and available before any external reference exists.

**(G2) Cross-expert thrust agreement — the interface contract, made numerical.** The disk expert asserts $T=2\rho AU_\infty^2a(1-a)$. The fluid expert independently produces a momentum-deficit flux across the same interface, $\oint \rho u_n\mathbf u\,ds$ evaluated just upstream and just downstream of the disk. **These two numbers are computed by different experts from different physics and must agree.** Define $r_T=|T_{\text{disk}}-\Delta\dot p_{\text{fluid}}|/T_{\text{disk}}$ as the primary acceptance gate. This is the single most valuable measurement in the plan and it needs no data at all.

**(G3) Betz and induction consistency.** $C_P\le16/27$ must hold, and the fluid expert's velocity at the disk must satisfy $U_{\text{disk}}/U_\infty\approx1-a$ to within a stated tolerance. A composed system that reports a super-Betz turbine has failed in a way no error norm would have flagged as unphysical — which is exactly what "physically plausible" is supposed to mean.

**(G4) Analytical wake models as a corridor.** The far wake must (i) collapse to a self-similar Gaussian deficit and (ii) sit inside the corridor spanned by the **Jensen/Park** linear-expansion model and the **Bastankhah–Porté-Agel** Gaussian model. These are closed-form engineering correlations — free, and the actual tools the industry validates against.

**(G5) Array efficiency.** $P_2/P_1$ for the fully waked downstream turbine, against published values (roughly $0.5$–$0.65$ at $7D$ spacing in a full wake). **Read this one with a discount**: 2D wakes recover differently from 3D — no vortex stretching, an inverse rather than forward cascade — so the number will be off, and it is reported as a sanity band, not a prediction. Stating that up front is the difference between a validation and an overclaim.

---

# 5. Risks, honestly

**(1) The fluid expert is out of distribution on inflow/outflow boundaries.** It was trained on periodic 2D turbulence. An inflow corridor and an outflow are not periodic. This is the same risk the RBC study identified, and the same two responses apply: **overlapping (Schwarz) agents with halo tokens** as the positive control, versus **non-overlapping declared flux-BC tokens** ([[edge-generation-atlas-0.1]] Mechanism A as built) as the real test. Run halo first; if only halo works, that is a finding about the framework, not a failure of the case study.

**(2) 2D turbulence is not 3D turbulence.** The inverse energy cascade means 2D wakes persist longer and recover by a different mechanism. Mitigation: grade G1–G3 (which are internal consistency, cascade-independent) as the primary gates, and treat G4/G5 as corridors. Do not let a 2D wake-recovery rate become a claim.

**(3) The disk expert can mask coupling error.** Because the algebraic side is exact, a coupling bug can show up entirely as fluid-side error and be misread as expert inaccuracy. Mitigation: G2 is a two-sided residual, and Phase B teacher-forces each side in turn.

**(4) Steady vs. unsteady.** If the composed system settles to a steady wake it has not tested temporal coupling. Force unsteadiness with a time-varying inflow (a gust ramp), which costs nothing and is itself a realistic engineering load case.

---

# 6. Phased plan

**Phase A — Verify the two ingredients (no new code).**
Confirm the Noether 1.1 checkpoint exists at P2 and loads; if not, fall back to Poseidon behind the Stage-1 adapter of [[incremental-transfer-roadmap]] and record the substitution. Implement actuator-disk theory as ~40 lines and unit-test it against the Betz limit and the $C_T(a)$ curve. *Exit gate: frozen expert runs a stable unforced 2D rollout; disk expert reproduces $C_P^{\max}=16/27$ at $a=1/3$ to machine precision.*

**Phase B — Single-interface probe, teacher-forced.**
One turbine, two agents ($I$ and $N$ across $R_1$). Feed each side the other's ground-truth quantity in turn. This is the go/no-go, and it involves no graph at all. *Exit gate: $r_T<5\%$ under at least one of halo/flux-BC.*

**Phase C — Wire the seven-agent graph.**
Reuse [[impl-atlas-0.1-phase1-scaffold]] verbatim; two expert slots instead of identity placeholders; MLP gate is a fixed two-way map and should be asserted as such. **Resolves the scaffold's open action item 4** ([[atlas-0.1-implementation-log]], 2026-08-09: interleave experts with message-passing layers, or add an intra-agent mixer?) — with real experts in the slots this becomes measurable.

**Phase D — Conservation constraint at the disk edges.**
Wire flux-matching per [[conservation-as-constraint-atlas-0.1]]. Report $r_T$ with the constraint off and on; that delta is the constraint's entire justification and has never been measured anywhere in this project.

**Phase E — Full coupled rollout.**
G1–G5 across turbine spacings $\{4D,7D,10D\}$ and induction factors $a\in\{0.2,1/3\}$, with a gust-ramp inflow. Always report decomposed vs. monolithic vs. analytical corridor as three curves.

**Phase F — Generality of the communication framework (the deliverable).**
Each axis is free and each maps onto a real engineering question:
- **Agent count:** $N=2,3,5,8$ turbines in a row. Composition error should grow sub-linearly in interface count; super-linear growth is the headline finding and would say the framework does not scale.
- **Staggered layout:** turbines offset laterally, so a wake only *partially* covers a downstream rotor — the first test of a **partial** interface.
- **Yaw / wake steering:** deflect the wake so the $F\!-\!R_2$ edge weakens or vanishes. This makes an edge's *existence* state-dependent, which is [[impl-atlas-0.1-phase5-expansion]]'s long-range direction-dependent edge in its simplest possible form, and the stated precondition for RL edge instantiation ([[edge-generation-atlas-0.1]] Mechanism B).
- **Heterogeneous resolution:** near-wake agents at finer token density than the bypass, same weights.
- **Multi-rate stepping:** rotor disks updated every substep, far wake every fourth — exercising the subcycling machinery the rocket needs, on a system with an analytic answer.

**Phase G — Extension path, and the handoff.**
Freeze Phases A–F as the interface-layer regression suite every later case study must still pass. Then the scenario grows by **adding agents to a framework that already works**, in this order, each step adding exactly one new family:
1. **Stratified inflow** → adds the already-trained Boussinesq expert ([[noether-1.0-rbc]]) as a thermal agent. **Still zero new experts.** Atmospheric stability is the dominant control on real wake recovery, so this is the highest-value next agent and it is free.
2. **Tower and blades** → structural expert. First genuinely new family; needs the corpus [[physics-simulation-datasets]] §3.4 says does not exist.
3. **Generator** → electromagnetics, restoring [[impl-atlas-0.1-phase5-expansion]]'s original scope.
4. **Fatigue accumulation** → the fast/slow split, a new mechanism rather than a new expert.

Step 1 is the important one for the user's stated goal: it is a **second case study that adds an agent through the same communication framework, with no new training**, and it is reachable immediately after this one.

---

# 7. What this proves, and what it does not

**Proves:** that declared typed edges, halo/flux exchange, flux-matching conservation, multi-rate and multi-resolution coupling, and $N$-agent composition produce a stable, momentum-conserving, physically plausible rollout of a **real engineering system** — with a learned PDE expert and a closed-form algebraic expert agreeing across an interface, at zero data and zero training cost.

**Does not prove:** anything about combustion, structural, or electromagnetic coupling; anything about 3D wake physics; anything about expert *generalization* (one scenario, per [[transfer-learning-fine-tuning]]'s diversity principle, is evidence the architecture is wired correctly and nothing more); and nothing bankable about array power, given the 2D discount in G5.

---

## See Also

- [[case-study-rocket-ascent-2d-atlas-0.1]] — the blocked target this substitutes for; unchanged, and still the eventual goal
- [[impl-atlas-0.1-phase5-expansion]] — the original offshore wind-farm spec this reduces; §2.1's agent table shows exactly which two experts the scope edit removes
- [[case-study-rbc-decomposition-atlas-0.1]] — the synthetic, single-expert version of the same interface test; useful as a Phase-B control, but not an engineering system
- [[expert-library-atlas-0.1]] — the closed-form-expert precedent (`rigid_body`) that the actuator disk follows
- [[conservation-as-constraint-atlas-0.1]] — the constraint that G2 finally makes measurable, against a side that cannot be wrong
- [[edge-generation-atlas-0.1]] — Mechanism A vs. halo exchange; Mechanism B unlocked by Phase F's yaw axis
- [[physics-simulation-datasets]] — the expert/data availability screen §0 rests on
- [[noether-1.0-rbc]] — the Boussinesq expert that makes Phase G step 1 free
- [[00-atlas-0.1-overview]] — invariant 7 (minimize simultaneous unknowns), which §3.3's non-scope applies
