# Generalization Requirements — the gap list, cut by which goal each gap blocks

**Type:** Concept page — framework gap analysis, **version-independent** (folder: `Atlas 0.1/common/`)
**Status:** written 2026-08-26; **restructured the same day** around two distinct goals, in answer to *"list all the gaps preventing (a) accurate composition given accurate agents, and (b) a generalized architecture where experts plug in instead of every case study being a rework."*
**Related:** [[master-error-bound]] · [[general-coupling-scheme]] · [[probed-dtn-coupling]] · [[composition-error-theory]] · [[temporal-error-accumulation]] · [[port-algebra-atlas-0.1]] · [[expert-library-atlas-0.1]] · [[open-problems-atlas-0.1]] · [[f1-pathmap-and-end-goal]] · [[global-fields-and-topology-atlas-0.1]] · [[regime-moe-architecture]]

> **The one-line version.** Twenty gaps. **They are two different lists with only four items in common**, which is the most useful thing on this page: *composing accurately* and *composing generically* are close to independent problems, and the framework's history has been spending on neither — the largest Goal-A item is that **three of the four constants in the bound have never been measured**, and the largest Goal-B item is a **single line in the port algebra's connection rule** that requires interface geometries to coincide.

**Numbering note.** G-numbers are stable identifiers in order of identification, *not* priority — other pages cite them. Priority is §D.

---

# 0. What "sufficiently accurate agents" actually buys

It sets $\tau\approx0$ in

$$\lVert e^N\rVert \;\le\; L^N\lVert e^0\rVert + \sum_n L^{\,N-n}\bigl(\tau^n+\sigma^n+\gamma^n\bigr)$$

That retires OP-3, OP-5's checkpoint share and most of OP-6 — genuinely a lot. It leaves **$\sigma$, $\gamma$, $L$ and the assembly operator untouched**, and it does not make the bound *apply* where it currently does not. **Goal A is therefore almost entirely about the three constants nobody has measured**, not about the agents.

One sharpening the premise itself needs: across a fluid–structure seam there is no single global operator whose restrictions the agents approximate, so *"an accurate agent"* is undefined there until **G2** is settled.

---

# A. Gaps blocking accurate composition (given accurate agents)

## A-tier: the bound's own constants

| # | Gap | Why it blocks accuracy | Status |
|---|---|---|---|
| **G20** | **Three of four constants are unmeasured and unreported: $\sigma$, $\beta$, $L$** (and the passivity defect $\pi$) | You cannot control what you do not measure. [[master-error-bound]] §7: the framework reports $\gamma$ — *the smallest term* — and nothing else | theory now exists; **zero measurements** |
| **G21** | **Transmission infidelity $\sigma$ is structurally large**: the rung is a one-cell Dirichlet ring, and $\lVert\Lambda-\tilde\Lambda\rVert$ has never been computed | $\sigma\le\frac{C_\mu}{\beta}\lVert\Lambda-\tilde\Lambda\rVert\lVert\lambda^\star\rVert$ is the term perfect agents do **not** touch, and iteration cannot reduce it | construction exists ([[probed-dtn-coupling]]), unbuilt |
| **G22** | **No expert declares a storage function**, so incremental passivity — the only *structural* route to $L\le1$ — is unavailable | Without it $L$ is an empirical hope. [[port-algebra-atlas-0.1]] §2 asserts passivity composes and then never uses it | identified, unbuilt |

## B-tier: silent correctness bugs — no error, no failed gate

| # | Gap | Why it blocks accuracy | Status |
|---|---|---|---|
| **G11** | **Conservative-vs-consistent interface mapping is chosen implicitly, per edge, by accident.** An `ADVEC` flow must map conservatively (integral preserved); a `THERM` effort consistently (pointwise values preserved) | Getting it backwards is a classic partitioned-coupling bug that silently destroys conservation or accuracy | **already flagged** in [[port-algebra-atlas-0.1]] §9.3, explicitly unfixed |
| **G1** | **Cross-points** — where 3+ subdomains meet, transmission conditions are not independent; the difficulty FETI-DP/BDDC exist for | Overlapping+PoU avoids it, so 124 tiles meeting four-at-a-corner have been fine. **[[probed-dtn-coupling]] introduces it** by going non-overlapping, and does not mention it. A mishandled cross-point degrades $\beta$ rather than raising an error | **zero coverage** |
| **G4** | **Multirate**: conservation must be enforced on the **time-integrated** flux; the stability limit is set by *coupling stiffness*, not either agent's step; interface interpolation order caps the scheme's order | Subcycling is **forced** by differing `native_dt`. Pointwise-in-time flux matching across different clocks is not conservative *and the residual looks fine* | mechanism specced (rocket), **theory absent** |
| **G3** | **Field↔lumped reduction/prolongation must be adjoint** w.r.t. the port pairing: $\langle e,\mathcal P_\Gamma f_0\rangle_\Gamma=\langle\mathcal R_\Gamma e,f_0\rangle_0$ | Otherwise the interface leaks power, which breaks the Dirac interconnection — **so G3 silently disables G22's passivity theorem.** The actuator disk is an instance built without the condition stated | **zero coverage** |
| **G18** | **Nondimensionalization is declared but not translated.** The `nondim` block exists; automatic conversion at a port does not | Two experts on different reference scales *will not agree at a port*. Called out as real, unavoidable, $O(K)$ | declared; **mechanism unbuilt** |

## C-tier: terms perfect agents do not remove

| # | Gap | Why it blocks accuracy | Status |
|---|---|---|---|
| **G12** | **Assembly / partition-of-unity consistency.** OP-5 charges **19% of total error to the assembly rule** — that share survives perfect agents and perfect windows | No stated condition that the blend be at least as accurate as the local solves it blends | measured as a share, **never diagnosed** |
| **G13** | **Interface temporal resolution is floored at the expert's native $\Delta t$.** $\lambda(t)$ is represented at the exchange cadence — a component of $\sigma$ in *time*, not space | Convergence does not remove it; a frozen expert cannot refine below its native step. Higher-order-in-time interface representation (waveform relaxation) is the only route | the $\Delta t$ constraint is recorded; **its status as an error term is not** |
| **G2** | **Heterogeneous physics**: no single global operator, so $\tau$ needs redefining per agent, and the optimized-Robin coefficient has **no symbol** to derive from | Makes the premise of this whole section ill-defined at multiphysics seams | **zero coverage** |
| **G6** | **Chaotic regime**: past $T_{\text{pred}}\approx\lambda_1^{-1}\ln(\delta_{\text{tol}}/\delta)$ the trajectory bound is vacuous, and **no construction changes that** | Perfect agents do not help — the divergence is the physics. Most of the intended domain. Needs a second claim class (invariant measure, spectra, structure functions) | **zero mentions** of shadowing, ergodicity or long-time statistics |
| **G7** | **No runtime a posteriori estimate.** Every bound is a priori and needs a reference to evaluate | You cannot tell whether a given composition is accurate *while running it*. Adjoint/DWR is the route | unbuilt; differentiability now identified unspent **three times** |

---

# B. Gaps blocking a generalized plug-in architecture

| # | Gap | Why it blocks generality | Status |
|---|---|---|---|
| **G14** | **The connection rule requires interface geometries to coincide.** *"Two agents may share an edge for port $P$ iff both declare $P$ and their interface geometries coincide."* | **This is the single most restrictive line in the framework for plug-in use.** Independently built experts will have different discretizations, resolutions and interface parameterizations. Non-conforming transfer — mortar projection with a measured inf-sup constant — is what removes it | **zero coverage** of non-conforming interfaces or mesh mismatch |
| **G23** | **The capability record is incomplete.** Present: `ports`, `conditioning`, `native_dt`, `nondim`. Absent: `bc_channel`, `bc_time_varying`, `differentiable`, `storage`, `equivariances`, `validity`, `regime_law` | Nothing can be *derived* from a declaration that does not say what the expert accepts. [[general-coupling-scheme]] §2 lists the full record | spec exists (this session), **unimplemented** |
| **G24** | **No coupling compiler.** $\Sigma=(\mathcal D,\tilde\Lambda,\mathcal O,\mathcal K,\mathcal C,W,\varepsilon_{\text{tol}})$ is hand-designed per case | This *is* the "rework every case study" problem, stated exactly. Nine admissibility rules are written; none are enforced in code | spec exists, **unimplemented** |
| **G5** | **No decomposition policy** — who decides where to cut an arbitrary system | The granularity heuristic is *physical* (regime boundaries) and says nothing about whether the resulting interface is well conditioned. **[AI Inference]:** the criterion falls out of the $\sigma$ bound — cut where the exact DtN is closest to local, i.e. where coupling is weakest, never along a shear layer, wake centreline or reaction front — and both factors come from a probe, so cut quality is **measurable before any rollout** | criterion identified; unimplemented |
| **G15** | **No expert↔subdomain routing from a library.** Given a subdomain and its regime, which expert? | [[regime-moe-architecture]] covers gating within a model; *selection from a library of independently trained experts* is a different problem and is unaddressed | **not addressed** |
| **G16** | **Port vocabulary completeness is unproven.** Five types are closed *for the cases considered* | `ADVEC`'s variable passenger list is already flagged as *where case-study-specific growth will reappear*. Radiation, phase change, contact/friction and EM field coupling are unexercised | closure claimed for scope; **completeness open**, species lists flagged |
| **G17** | **Moving / deforming interfaces.** FSI, free surfaces, combustion fronts, contact | The port geometry is static. Any $\Gamma$ that moves changes the interface operator every step and invalidates a cached $S$ | **zero coverage** |
| **G9** | **Topology mutation.** Staging, docking, contact make/break | The bound assumes a fixed graph. Reconciles as a **restart with $e^0\neq0$** — so $L^N\lVert e^0\rVert$ reactivates at every mutation — but conservation *across* the event has no rule | mechanism named, **no rule** |
| **G8** | **Abstention and validity domains.** An expert must be able to **decline** | With $K$ experts, P(at least one OOD) $\to1$, and it gets *worse* as the vocabulary gets cleaner — a clean contract makes more connections **legal** without making them **valid** | **already flagged** in [[port-algebra-atlas-0.1]] §8; unbuilt |
| **G10** | **No general verification protocol.** W0–W11 are wind-farm gates | **[AI Inference]:** the generalizable core is three — *exactness on an invariant state*, *agreement with a classical reference at the **matched** regime*, and *insensitivity to graph refinement* — and the wind farm's history is that each caught something the others did not | case-specific |
| **G19** | **Reference availability.** Validating a new case needs a classical reference at the matched regime | For arbitrary systems one may not exist, or may cost more than the surrogate. Without it G10 cannot be run and no accuracy claim is checkable | **not addressed** |
| **G12′** | **Differentiability is claimed but not implemented end-to-end** | Named as a headline property against preCICE, and now identified as unspent for the adjoint (G7), the probe JVP, and runtime estimation | claimed; unverified |

---

> **Status update, 2026-08-27.** Several *"zero coverage"* entries above are stale. [[interface-transfer-theory]] writes **G11, G3, G14, G4, G13, G18, G1, G7 and G5** as rules, and two of the entries were mis-framed rather than merely uncovered: **G11 and G3 are the same condition** — the two halves of one adjoint pair, filed in different tiers because one was stated about quantities and the other about operators — and **G14 is a restriction of the connection rule, not of the coupling construction**, since probing never touches either agent's discretization. Read those rows as *what was missing before the rules were written*; the constructions now exist and the **measurements still do not**, which is the part the status column was right about. [[plug-in-composition-theorems]] additionally **promotes G22 to both lists** (passivity is the only property that survives an expert swap without re-certification), **reframes G19** as a cost curve on the architecture's one unverifiable field, **gives G15 a criterion**, and **adds G25–G28**.

---

# C. The four that block both

| # | why it is on both lists |
|---|---|
| **G3** field↔lumped adjointness | *accuracy*: leaks power, disables the passivity theorem. *generality*: F1's "first-class peers" claim is unbuildable without it |
| **G8** abstention | *accuracy*: every bound is vacuous outside validity. *generality*: a library of pluggable experts without validity predicates is unsafe by construction |
| **G18** nondimensionalization | *accuracy*: experts silently disagree at a port. *generality*: it is the $O(K)$ per-expert cost that plug-in is supposed to remove |
| **G7 / G12′** differentiability + runtime estimation | *accuracy*: the only route to an error bar with no reference. *generality*: the compiler needs probe derivatives to choose a scheme |

---

# D. Priority

**Blocking in the strict sense — the framework produces wrong answers with no signal:**

1. **G11** interface mapping chosen by accident — *already identified, unfixed, and cheap*
2. **G1** cross-points — required the moment a probed-Schur solve meets a non-chain graph
3. **G4** multirate conservation — the rocket needs it and the mechanism is already specced
4. **G3** field↔lumped adjointness — silently disables G22

**Highest value per hour, Goal A:**

5. **G20** — measure $L$ (one paired rollout), then assemble one $\tilde\Lambda$ for $\beta$, $\kappa$ and $\sigma$. *Nothing else on the list can be prioritized rationally until these three numbers exist.*
6. **G12** assembly rule — a 19% share that is already measured and has never been diagnosed

**Highest value per hour, Goal B:**

7. **G23 + G24** — capability record and compiler. Together they *are* the plug-in architecture; both specs are written
8. **G14** non-conforming interfaces — the one line that most restricts who can plug in

**Then:** G6, G5, G13, G8, G15, G2, G16, G7, G9, G10, G17, G19, G18.

> **Two observations worth more than the ordering.** First, **the two lists barely overlap** — four items out of twenty. Accurate composition and generic composition are close to independent programmes, and progress on one should not be expected to buy the other. Second, **every strictly-blocking gap is a gap about silence.** A mishandled cross-point degrades $\beta$; a backwards mapping conserves the wrong integral; non-conservative subcycling yields a residual that is computed correctly and means nothing. None raises an error or fails a gate. That is the identical failure mode as the interface residual that was **identically zero, bitwise, while the answer was $2\times$ wrong** — and it is the mode this vault has been caught by more often than any other.

---

# E. What "generalized" concretely means

Reached when all of these hold and none requires a new case study:

1. A new system is specified as **a graph plus capability records**, and $\Sigma$ is *compiled* (G23, G24).
2. The compiler **refuses inadmissible schemes citing a rule** — as the `schwarz > 1` guard already does for one rule out of nine.
3. Every run emits $(\tau,\sigma,\gamma,L,\beta,\kappa,\pi,\mathcal R,T_{\text{pred}})$ (G20).
4. Interfaces connect **without geometric coincidence** (G14), with mapping conservative-or-consistent **by declaration** (G11).
5. Cuts are **placed** by G5's criterion and their quality reported before the rollout.
6. Agents **decline** outside validity (G8) rather than returning a number.
7. Claims are typed: **trajectory** inside $T_{\text{pred}}$, **statistical** outside, never silently mixed (G6).

**[AI Inference]:** item 3 changes the most for the least effort and is nearly free — every quantity is a by-product of machinery the coupled solve already builds. The framework's recurring failure has not been computing the wrong thing; it has been **reporting the term that was easy to compute.**

---

## See Also

- [[gap-worklist]] — **these gaps as tracked, actionable items**, with a definition of done for each and an order that puts measurement first. Start there to do the work; stay here for the reasoning
- [[master-error-bound]] — the four terms §A is organized around; §9 is G6
- [[general-coupling-scheme]] — §2's capability record is G23, §4's compiler is G24, R9 is G4's rule, §5 is G1
- [[probed-dtn-coupling]] — introduces G1; supplies G5's measurable cut criterion and G20's assembly
- [[port-algebra-atlas-0.1]] — §9.3 is G11, §8 flags G8 and G18, §5's connection rule is G14, `ADVEC`'s passenger list is G16
- [[composition-error-theory]] — why $\gamma=0$ is not accuracy, which is why G20 ranks above G-anything-about-agreement
- [[temporal-error-accumulation]] — G13's exchange-cadence floor and G6's predictability horizon
- [[f1-pathmap-and-end-goal]] — F1's field+lumped claim, whose missing condition is G3
- [[open-problems-atlas-0.1]] — what the premise discharges (OP-3, OP-5, OP-6) and what it leaves (OP-1, OP-2)
- [[expert-library-atlas-0.1]] — where capability records and validity predicates live
- [[regime-moe-architecture]] — gating within a model; G15 is the different problem of selecting *between library experts*
- [[global-fields-and-topology-atlas-0.1]] — the graph mutation G9 must reconcile with a fixed-graph bound
- [[theory-closure-audit]] — **the same gaps re-cut by *kind of work***: 4 closed, 9 importable from the classical literature, 6 genuinely open. Read it before deciding whether any of these needs research or only a recorded decision
- [[interface-transfer-theory]] — writes G11, G3, G14, G4, G13, G18, G1, G7 and G5 as rules. **G11 and G3 are shown to be the same condition** — the two halves of one adjoint pair — and G14, *"the single most restrictive line in the framework for plug-in use"*, is argued to be a restriction of the connection rule rather than of the construction
- [[plug-in-composition-theorems]] — **adds G25 (closure), G26 (substitution), G27 (conformance) and G28 (abstention under composition)**, none of which this page's twenty-four cover. It also promotes **G22** to both lists, since passivity is the only property that survives an expert swap without re-certification, and reframes **G19** as a cost curve on the architecture's single unverifiable field rather than an independent gap
- [[end-to-end-architecture-spec]] — **the architecture §E describes the finished state of.** Every one of §E's seven conditions is a layer there; G15 routing is flagged as a layer the spec does **not** cover; and §E item 7 (claims typed trajectory-vs-statistical) is decided rather than listed
