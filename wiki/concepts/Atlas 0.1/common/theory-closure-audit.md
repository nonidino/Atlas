# Theory Closure Audit — what is proved, what is importable, what is genuinely open

**Type:** Concept page — framework audit, **version-independent** (folder: `Atlas 0.1/common/`)
**Status:** written 2026-08-27, in answer to the direct question *"are all of the issues we are currently facing solved from a theoretical standpoint?"* asked before committing to an implementation push.
**Verdict:** **No — and the useful part of the answer is that the remaining holes fall into two very different classes, only one of which is research.**
**Related:** [[master-error-bound]] · [[generalization-requirements]] · [[gap-worklist]] · [[general-coupling-scheme]] · [[probed-dtn-coupling]] · [[composition-error-theory]] · [[temporal-error-accumulation]] · [[port-algebra-atlas-0.1]] · [[open-problems-atlas-0.1]]

> **The one-line version.** The **composition error theory is closed** — the master bound is a theorem, not a hope, and every constant in it is measurable. But it is a theorem **on an envelope of seven hypotheses**, and *three of those seven are unverified even for the wind farm*, while five of the framework's twenty-four gaps have **no answer anywhere** — not in the vault and not off the shelf. The most important of these is that **past $T_{\text{pred}}$ the framework has no notion of what it is even claiming**, and most of the intended domain is past $T_{\text{pred}}$.

---

# 1. The three classes

Sorting the gaps by *what kind of work closes them* is more actionable than sorting by priority, because the three classes have different costs, different risks, and different owners.

| class | meaning | count | risk of "back to the drawing board" |
|---|---|---|---|
| **CLOSED** | proved in this vault; the remaining work is measurement or code | 4 | none |
| **IMPORTABLE** | solved in the classical DD/coupling literature; absent here, but no research required — only adoption and a decision | 9 | low — but each needs *a decision recorded*, and an undecided default is how the vault has been bitten before |
| **OPEN** | no off-the-shelf answer, or the answer requires a modelling commitment Atlas has not made | 6 | **this is the drawing board** |

Five gaps are engineering only (`G23`, `G24`, `G12′`, `G15`, `G19`) and are not theory at all.

---

# 2. CLOSED — what this vault has actually proved

These are settled, and re-opening them would be waste.

| # | Result | Where |
|---|---|---|
| **C1** | **The master bound.** $\lVert e^N\rVert \le L^N\lVert e^0\rVert + \sum_n L^{N-n}(\tau^n+\sigma^n+\gamma^n)$, from an exact recursion plus one telescoping identity. No approximation until the Lipschitz step | [[master-error-bound]] §2–§3 |
| **C2** | **The three-way defect split is the right split**, because each term is reduced by a *different* mechanism and one of them — $\sigma$ — is reduced by no amount of iteration. The known $2\times$ failure is exactly the corner $\gamma=0$ bitwise, $\sigma$ maximal, which the two-term version cannot express | [[master-error-bound]] §3.1 |
| **C3** | **$\sigma \le \frac{C_\mu}{\beta}\lVert\Lambda-\tilde\Lambda\rVert\lVert\lambda^\star\rVert$**, and $\beta$ sits in the denominator of both $\sigma$ and $L$ — an ill-conditioned interface hurts additively *and* multiplicatively. Both factors are exactly what one probe returns | [[master-error-bound]] §4, §6 |
| **C4** | **The worst-agent conjecture is a theorem** in the corner $\sigma=\gamma=0$, $L\le1$ — with all three now measurable quantities rather than assumptions, and the $L\le1$ branch structural via incremental passivity rather than fitted | [[master-error-bound]] §6.1, §8 |

**What C1–C4 do *not* do, and this is the whole point of §3 and §4:** they say nothing about whether the hypotheses hold, and nothing outside the envelope.

---

# 3. The envelope — the seven hypotheses the bound silently assumes

**This is the single most useful artifact on this page**, because it is the thing no page currently states: the precise class of systems on which the master bound is a theorem. Each hypothesis maps to exactly one gap, and each gap is the *removal* of that hypothesis.

| # | Hypothesis | Removed by | Holds for the wind farm? |
|---|---|---|---|
| **E1** | The interaction graph is **fixed** for the whole rollout | G9 topology mutation | **yes** |
| **E2** | Interfaces are **static** and **geometrically coincident** | G17 moving $\Gamma$; G14 non-conforming | **yes** (uniform tiles, shared discretization) |
| **E3** | A **single global evolution operator** exists whose restrictions the agents approximate — so that $\tau$ is defined | G2 heterogeneous physics | **yes** (one Navier–Stokes) |
| **E4** | A **single macro-step clock**, or a multirate scheme whose *time-integrated* flux matching is proved conservative | G4 multirate | **yes** (single clock) |
| **E5** | The claim is a **trajectory** claim inside $T_{\text{pred}}$ | G6 chaos | **unverified — $T_{\text{pred}}$ has never been computed** |
| **E6** | The assembly $\mathcal A$ is bounded with $\lVert\mathcal A\rVert$ known, and the blend is at least as accurate as the local solves it blends | G12 assembly consistency | **unverified — 19% of error charged to it, never diagnosed** |
| **E7** | The interconnection is **power-preserving** and every reduction/prolongation pair is **adjoint** — required for the $L\le1$ branch | G3 adjointness; G22 storage | **unverified — the actuator disk was built without the condition stated** |

> **Read the right-hand column.** The flagship case study — the one described as "completed reasonably" — satisfies four of the seven and **has never checked the other three**. It is not known to sit inside the envelope of its own error bound. That is not an argument against the case study; it is an argument that E5, E6 and E7 are checks, not assumptions, and that they are cheap ([[gap-worklist]] W1, W8, and the diagnosis half of W3).

**[AI Inference]:** stating the envelope explicitly is worth more than closing any single gap on it, because it converts "the bound might not apply" from a vague worry into seven yes/no questions, three of which are answerable this week. It is also the natural admissibility check for the compiler of [[general-coupling-scheme]] §4 — a scheme is refusable exactly when it violates a hypothesis the case needs.

---

# 4. IMPORTABLE — solved elsewhere, absent here

No research. Each needs a *decision recorded in the port algebra or the coupling scheme*, and the failure mode of leaving them undecided is the vault's signature one: a silent default chosen per edge, by accident.

| # | Gap | The classical answer | What is missing here |
|---|---|---|---|
| **I1** | **G11** conservative vs. consistent mapping | Flux-like quantities map conservatively (integral preserved); intensive quantities map consistently (pointwise) | A `mapping` field defaulted **per port type**; currently implicit per edge ([[port-algebra-atlas-0.1]] §9.3) |
| **I2** | **G1** cross-points | FETI-DP / BDDC: make corner DOFs **primal** (single-valued), dual elsewhere | Named in 6 files, **decided in none**. [[probed-dtn-coupling]] introduces the problem by going non-overlapping and does not mention it |
| **I3** | **G14** non-conforming interfaces | Mortar projection with an inf-sup-stable multiplier space; $\beta$ is measurable | `mortar` appears only as a cited gap — **no construction page anywhere in the vault** |
| **I4** | **G4** multirate conservation | Match the **time-integrated** flux over the macro-step; stability set by coupling stiffness, not either agent's step; interpolation order caps scheme order | R9 exists as prose, not as a check |
| **I5** | **G3** field↔lumped adjointness | One condition: $\langle e,\mathcal P_\Gamma f_0\rangle_\Gamma=\langle\mathcal R_\Gamma e,f_0\rangle_0$ | Condition never stated in the port algebra; never checked on the actuator disk |
| **I6** | **G13** interface temporal resolution | Schwarz waveform relaxation — represent $\lambda(t)$ to higher order than the exchange cadence | Recorded as a $\Delta t$ constraint, not as an error term |
| **I7** | **G7** runtime a posteriori estimate | Adjoint / dual-weighted residual, localizing error per subdomain and per seam | Differentiability claimed and now unspent **three times** |
| **I8** | **G18** nondimensionalization translation | Mechanical: compose the two `nondim` maps at the port | Declared, never translated |
| **I9** | **G5** decomposition policy | Cut where the exact DtN is closest to local — both factors of $\sigma$ come from a probe, so cut quality is **measurable before any rollout** | Identified as an [AI Inference] in [[generalization-requirements]] G5; never derived or tested |

**[AI Inference]:** I1–I9 are the reason the honest verdict is *"not closed"* rather than *"back to the drawing board."* Nine of the fourteen theory gaps are adoption, and adopting them is a week of writing rules down, not a research programme.

> **Status update, 2026-08-27 — written, and the table's framing was wrong in one respect.** [[interface-transfer-theory]] writes all nine down, and the "what is missing here" column above is now stale for I1, I3 and I5 in particular. **The estimate was right about the effort and wrong about the structure:** these were filed as *nine independent transcriptions*, and **six of them (I1, I3, I4, I5, I6, I8) are one declared object** — a common interface space per seam plus one prolongation per side, whose reduction is forced to be the adjoint. Consequently **I1 needs no `mapping` field at all** (the class is derived, not defaulted), **I5 is the special case $\dim M=1$**, and **I3's *"no construction page anywhere in the vault"* no longer holds** — the construction turns out to be the probed assembly already budgeted, on a space that was never named. I2, I7 and I9 remain genuinely separate. **The three-class sort of §1 stands; what it under-counted was the coupling *between* items inside the IMPORTABLE class.**

---

# 5. OPEN — the actual drawing board

These have no answer in the vault and no clean answer to import. **This is the list that decides whether an implementation push is premature.**

## O1 — G6: there is no claim class beyond $T_{\text{pred}}$ · *the big one*

Past $T_{\text{pred}}\approx\lambda_1^{-1}\ln(\delta_{\text{tol}}/(\tau+\sigma+\gamma))$ the master bound is vacuous, and **no construction on any page changes that** — the divergence is the physics. The framework consequence is that the deliverable must change from a **trajectory** claim to a **statistical** one: invariant measure, energy spectra, structure functions, long-time averages.

**That theory does not exist here.** A vault-wide search finds `shadowing` and `ergodic` in exactly two pages — [[master-error-bound]] §9 and [[generalization-requirements]] G6 — **both of which mention them only to say they are missing.** There is no page developing shadowing under hyperbolicity, no ergodic-average error estimate, no gate set for statistical claims, and no definition of what $\tau$ means when the target is a measure rather than a trajectory.

**Why this is the one that matters most:** turbulence, climate and plasma — the intended domain — are chaotic, so **most of the target application area lies past $T_{\text{pred}}$**, where the framework currently proves nothing and, worse, has no vocabulary for what it *should* prove. Every rollout metric quoted so far is a trajectory metric, and none has been checked against its own horizon.

## O2 — G2: $\tau$ is undefined at a multiphysics seam

E3 assumes a single global operator $\mathcal S_{\Delta t}$ whose restrictions the agents approximate. **Across a fluid–structure or fluid–chemistry seam no such operator exists**, so "agent infidelity" has no referent, and the optimized-Robin coefficient has no symbol to be derived from. This is not an importable fix: it requires deciding what the composed object *is* when the two sides do not discretize a common continuum problem. Until it is settled, the premise of the entire Goal-A analysis is ill-defined exactly where multiphysics begins — which is the framework's whole selling point.

## O3 — G12: no consistency condition on the assembly

OP-5 charges **19% of total error** to the assembly rule. The bound carries $\lVert\mathcal A\rVert$ as a factor of $L$ and otherwise says nothing. **There is no stated condition that the blend be at least as accurate as the local solves it blends** — partition-of-unity theory supplies such conditions for overlapping approximation, but they have not been transcribed, and the measured 19% has never been diagnosed as either a violated condition or an unavoidable cost.

## O4 — G17: moving and deforming interfaces

The port geometry is static. Any $\Gamma$ that moves — FSI, free surfaces, combustion fronts, contact — changes the interface operator every step and invalidates a cached $S$, which is the entire economic argument for [[probed-dtn-coupling]]. `moving interface` appears in exactly two files, both of which are the worklist and the index. **Zero development.**

## O5 — G9: no conservation rule across a topology mutation

Staging, docking and contact make/break reconcile with a fixed-graph bound as a **restart with $e^0\neq0$**, so the $L^N\lVert e^0\rVert$ term reactivates at every event — that much is settled. What has **no rule** is conservation *across* the event: what is required of the state map so that the mutation itself neither creates nor destroys the conserved quantities.

## O6 — G16: port vocabulary completeness is unprovable in the abstract

Five port types are closed *for the cases considered*. `ADVEC`'s variable passenger list is already flagged as where case-study-specific growth reappears, and radiation, phase change, contact/friction and EM coupling are unexercised. **[AI Inference]:** this one never "closes" — the right target is not completeness but a **documented extension procedure**, so that adding a sixth port type is a defined operation rather than a redesign. That reframing is itself an open design question, not a proof.

---

# 6. The verdict, and what follows from it

**Not closed.** Four results proved; nine gaps importable; **six genuinely open**, one of which (O1) removes the framework's guarantees over most of its intended domain.

**But the shape of the answer argues against a long theory detour and against an immediate build.** Specifically:

1. **The core is done.** [[master-error-bound]] is not going to be rederived. Building on it is safe.
2. **The envelope, not the bound, is what is undocumented** (§3). Three of its seven hypotheses are unverified for the one case study that exists.
3. **Nine of fourteen theory gaps are transcription**, not research (§4).
4. **The open six are not evenly distributed** — five of them (O2, O3, O4, O5, O6) are about *widening the envelope*, and every one of them is invisible until you try to specify a case that violates its hypothesis. **O1 alone is a hole inside the current envelope.**

> **Resolution note, added 2026-08-27 (same day).** The conclusion below was acted on immediately: [[end-to-end-architecture-spec]] is the end-to-end outline this paragraph argues for. It uses §3's envelope as its spine, gives each of §5's five envelope-widening gaps a **named slot with a declared interface and a required measurement** rather than a solution, and records the decision O1 was waiting for — **claims typed trajectory inside $T_{\text{pred}}$, statistical outside, with the statistical type declared and refused rather than approximated.** Two findings there bear back on this page: the wind farm's negative composed result would have been a **compile-time refusal** ($\Xi=0$ means the interface problem is *empty*, not hard), and the rocket case study **does not compile**, tripping three of the five slots in its declared scope — which is the concrete form of point 4 below.

**[AI Inference] — the operational conclusion.** Point 4 is the argument for an **end-to-end architecture outline as the next step, before implementation**. The five envelope-widening gaps are precisely the ones an implementation would hit late and expensively, because each is invisible while the only case study is a single-physics, fixed-graph, conforming, single-clock one that satisfies every hypothesis they remove. Writing the architecture down end to end — every layer from expert declaration to emitted diagnostic, with the envelope as an explicit admissibility check — surfaces all five *on paper*, at the cost of a chat rather than a rewrite. It also produces the artifact the framework does not have and needs before any plug-in claim: **a specification of the whole system rather than a set of correct pages about its parts.**

---

## See Also

- [[master-error-bound]] — the four results §2 certifies as closed, and §9 there is O1
- [[generalization-requirements]] — the G-numbered gaps this page re-sorts by *kind of work* rather than by goal or priority
- [[gap-worklist]] — the same gaps as tracked items; §3's unverified hypotheses are W1, W3 and W8
- [[general-coupling-scheme]] — the compiler that should enforce §3's envelope as an admissibility check
- [[probed-dtn-coupling]] — whose cached $S$ economics O4 invalidates, and whose non-overlapping formulation raises I2
- [[port-algebra-atlas-0.1]] — where I1, I2 and I5 must be written down as decisions
- [[composition-error-theory]] — the informal statements C3 makes into a theorem
- [[temporal-error-accumulation]] — the recursion behind C1 and the horizon behind O1
- [[open-problems-atlas-0.1]] — OP-5's 19% assembly share is O3; OP-2's drift is the fitted $L$
- [[interface-transfer-theory]] — **§4's nine importable mechanisms, written down.** Six of them (I1, I3, I4, I5, I6, I8) turn out to be **one declared object**, which revises this page's estimate of the adoption work downward and its estimate of the *coupling between the items* sharply upward — they were filed as nine independent transcriptions and are not independent
- [[plug-in-composition-theorems]] — **three properties §1's three-class sort has no room for**, because every class here is about a mechanism and those are about the composition operation itself: closure under composition, substitution, and conformance. Opens G25–G28
- [[end-to-end-architecture-spec]] — **the artifact §6 argues for, written the same day.** §3's envelope becomes a per-run stamp; O2–O6 become five named slots with declared interfaces and required measurements; O1 becomes a recorded decision
