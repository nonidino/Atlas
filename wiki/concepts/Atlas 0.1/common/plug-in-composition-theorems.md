# Plug-In Composition Theorems — closure, substitution, conformance, and the two horizons

**Type:** Concept page — **framework theory, binding**, version-independent (folder: `Atlas 0.1/common/`)
**Status:** written 2026-08-27, in answer to *"is all of the theory decided — enough for a build that fully outlines accurate composition given the agents, and the plug-in architecture?"* **It was not, and the missing pieces were not on any gap list.** [[generalization-requirements]] tracks twenty-four gaps and [[theory-closure-audit]] sorts them into three classes; **none of them is any of the three properties on this page**, because every existing gap is about a *mechanism* and these are about *the composition operation itself.* Four new gaps are opened here: **G25–G28**.
**Related:** [[end-to-end-architecture-spec]] · [[interface-transfer-theory]] · [[theory-closure-audit]] · [[master-error-bound]] · [[probed-dtn-coupling]] · [[general-coupling-scheme]] · [[port-algebra-atlas-0.1]] · [[generalization-requirements]] · [[expert-library-atlas-0.1]] · [[gap-worklist]] · [[composition-error-theory]] · [[conservation-as-constraint-atlas-0.1]]

> **The one-line version.** A plug-in architecture asserts three things it has never stated: that a composed subsystem **is itself a pluggable expert** (closure), that swapping one expert for another **does not require re-certifying the system** (substitution), and that a capability record **is true** (conformance). All three are derivable or decidable, and three of the answers are uncomfortable. **Closure holds for 8 of 13 record fields, fails for 3, and propagates a hole in 2** — and it proves that $\sigma$ and $\gamma$ at one level of nesting *become* $\tau$ at the level above, so **the three-way error split is not invariant under regrouping.** **Substitution has a certificate computable from two probes and no rollout** — and proves that a strictly better expert can make a composition worse, so *"better agents ⟹ better composition"* is false as stated. **Conformance is nearly free for every field except one**, and that one — `validity` — is **falsifiable but not verifiable**, which makes it the single unverifiable axiom the whole architecture rests on. Finally, abstention supplies a **second horizon** $T_{\text{abs}}\approx\Delta t/(Kp)$ that degrades **linearly in library size**, so for a large library the binding limit on a rollout may be abstention rather than chaos.

---

# 0. Why these three, and why they were invisible

[[end-to-end-architecture-spec]] specifies the composition path for **one graph, one level, one fixed set of experts**. Everything in it is correct and none of it says what happens when the graph is used the way a plug-in architecture is meant to be used:

| The plug-in claim | The property it requires | Where it is stated |
|---|---|---|
| *"Each new subsystem is a port connection, not an integration project"* ([[port-algebra-atlas-0.1]] §5.1) | Composition is **closed** — a cluster of connected agents presents an expert's interface | **nowhere** |
| *"Adding the twentieth expert costs exactly what adding the fourth cost"* ([[port-algebra-atlas-0.1]] §5) | **Substitution** is local — swapping one part does not re-open the others | **nowhere** |
| *"Everything the framework concludes is inherited from labels"* ([[end-to-end-architecture-spec]] §3.1) | The labels are **true** | **nowhere** |

**Why no gap list caught them.** Every G-number is the removal of a *hypothesis* about the physics or the discretization — a moving interface, a mutating graph, a multiphysics seam. These three are properties of the **composition operation**, which no page ever treated as an object with algebraic properties to check. They became visible only when the whole path was written down end to end and then asked to run twice.

**The stakes are not symmetric.** Conformance and substitution are *cheap and now settled*. Closure is where a build could still go back to the drawing board, because if the framework were **not** closed under composition, hierarchy would be impossible and the port algebra's $O(K)$ scaling argument would fail at exactly the library size the roadmap targets. §1 establishes that it is closed, and names precisely the five fields where it is not.

---

# 1. Theorem 1 — closure under composition

## 1.1 Statement

> **Let $\mathcal C\subseteq\mathcal G$ be a connected subgraph whose internal ports are all matched. Then $\mathcal C$, equipped with the interface solve on its internal seams, presents a valid `ExpertCapabilities` record and may be connected as a single expert — provided its internal interface problem is nonsingular, $\beta_{\text{int}}>0$.**

The proviso is the whole content of the condition, and it is already measured: $\beta_{\text{int}}=\sigma_{\min}$ of the internal block of the assembled interface operator ([[probed-dtn-coupling]] §4.3). **A subassembly is a legal expert exactly when its own interface problem is well posed** — which is both the natural condition and a checkable one.

## 1.2 The composite's transmission operator is a Schur complement

Assemble the composite's interface operator on the common space of [[interface-transfer-theory]] §3, $\tilde\Lambda=\sum_{i\in\mathcal C}P_i^\ast\Lambda_iP_i$, and partition the interface degrees of freedom into

- $I$ — **internal**: seams matched inside $\mathcal C$,
- $E$ — **exposed**: ports left open, which become the composite's ports.

Then eliminating the internal trace gives

$$\boxed{\;\tilde\Lambda_{\mathcal C} \;=\; \tilde\Lambda_{EE} \;-\; \tilde\Lambda_{EI}\,\tilde\Lambda_{II}^{-1}\,\tilde\Lambda_{IE}\;}$$

the Schur complement of the internal block. $\tilde\Lambda_{II}$ is invertible on the constrained subspace precisely when $\beta_{\text{int}}>0$, which is the proviso.

**This is the same object the framework already builds.** [[probed-dtn-coupling]] §2.3 solves $S\lambda=\chi$ directly; §1.2 here says that solving it *partially* — eliminating only the internal rows — leaves an operator of exactly the type a single expert exposes. Nesting is not a new mechanism; it is **stopping the existing elimination early.**

### Corollary — grouping does not change the answer

Schur complementation is transitive: eliminating $I_1$ then $I_2$ gives the same complement as eliminating $I_1\cup I_2$ at once (the quotient property of Schur complements). Therefore

> **A hierarchy of nested composites and the flat graph produce the same composed interface operator.** Grouping is free to choose, and can be chosen for cost.

That is what makes hierarchy safe, and it is the answer to whether a build must commit to a nesting depth up front. **It must not, and it may re-group at will.**

## 1.3 The field-by-field result

| field | composes as | status |
|---|---|---|
| `ports` | the **unmatched** ports of $\mathcal C$ | **closes.** Internal ports are eliminated; this is [[port-algebra-atlas-0.1]] §5.1's open-port visibility applied recursively |
| transmission $\tilde\Lambda$ | Schur complement, §1.2 | **closes**, given $\beta_{\text{int}}>0$ |
| `storage` | $H_{\mathcal C}=\sum_{i\in\mathcal C}H_i$ | **closes.** Power-preserving interconnection of passive parts is passive; in operator form, the Schur complement of a positive-real block w.r.t. a positive-real principal block is positive real. This is [[port-algebra-atlas-0.1]] §2 consequence 3 and [[probed-dtn-coupling]] §4.2, used rather than asserted |
| `dt_native` | $\max_{i\in\mathcal C}\texttt{dt\_native}_i$ | **closes**, by R4 — a composite is as slow as its slowest constituent |
| `bc_channel` | determined at each exposed port by the boundary agent there, liftable by R2 | **closes.** A composite can always be probed, so R2 applies at the level above exactly as below |
| `differentiable` | the weakest of the constituents; a composite JVP costs one back-solve with the already-factored $\tilde\Lambda_{II}$ | **closes**, at a stated cost |
| `equivariances` | the subgroup of $\bigcap_i G_i$ that is **also a graph automorphism of $\mathcal C$** | **closes, with a condition no page states.** A symmetry of the parts is a symmetry of the whole only if it maps the internal wiring to itself |
| `claim_types` | intersection | **closes** |
| `L_native` | — | **does not compose.** Constituents with different window sizes leave the composite with no single $L$, and `regime_law` (below) depends on it |
| `regime_law` | — | **does not compose.** The composite's regime is a *tuple* of constituent regimes; scalarizing it is a modelling choice, not a derivation |
| `validity` | $\bigwedge_i \texttt{validity}_i$, evaluated at the **internal solution** | **does not compose usably.** The conjunction is correct and is not evaluable *before* the internal solve — see §1.6 |
| `governing_family` | the common value if all constituents agree; otherwise `composite` | **propagates a hole.** If they disagree, E3 fails for the composite and its $\tau$ is `UNDEFINED` — the `SeamReference` slot climbs the hierarchy |
| `lambda_ref` | requires a reference operator for the composite | **propagates a hole**, for the same reason |

**Eight close, three do not, two propagate an existing hole.** That is the answer, and the three that do not close are all the *same kind* of field: they describe an **operating point** rather than an interface. **[AI Inference]:** the pattern suggests the record has two halves that behave differently under composition — an *interface* half that is algebraic and closes, and a *regime* half that is a modelling description and does not. If that reading is right, the fix for the three failures is not to derive composition rules for them but to accept that **a composite must re-declare its regime half by hand**, which is a bounded, one-line-per-composite cost rather than an obstruction. This is a reading, not a result.

## 1.4 The attribution theorem — and it is the uncomfortable one

What is the composite's **agent infidelity** $\tau_{\mathcal C}$? By definition it is the one-macro-step defect of the composite map against the true restriction of the global evolution operator to $\Omega_{\mathcal C}$. That is exactly what [[master-error-bound]] §7 bounds at $N=1$ with $e^0=0$:

$$\boxed{\;\tau_{\mathcal C}\;\le\;\tau^{\text{int}}\;+\;\sigma^{\text{int}}\;+\;\gamma^{\text{int}}\;}$$

where the superscript denotes the internal per-step terms of $\mathcal C$. The inequality holds whichever convention the internal $\tau$ aggregates under, since it is the same per-step symbol the bound already carries.

> **Read what that says. The transmission infidelity and solve incompleteness of a subassembly become the *agent infidelity* of the expert one level up.** They are not removed by nesting and they are not double-counted — they are **relabelled**.

**Three consequences, in increasing order of how much they matter.**

1. **The three-way split is not invariant under regrouping.** $\tau$, $\sigma$ and $\gamma$ are properties of a decomposition *at a stated depth*, not of the physical problem. Two people who group the same system differently will attribute the same total error differently, and both will be right.
2. **Every emitted defect must carry the depth it was measured at.** Otherwise a nested run's numbers cannot be compared with a flat run's, and a $\tau$ reported at depth 2 will be compared against a $\sigma$ reported at depth 1 as though they were independent terms. **[[end-to-end-architecture-spec]] §10.2's emit contract does not carry a depth field**, and must.
3. **[AI Inference], and it bears on an open problem.** OP-5 charges error shares to "the expert" versus "the architecture" by subtraction. If the expert is itself internally composed — and `WindowNS` over 124 tiles is exactly that — then **part of what OP-5 attributes to the expert is a $\sigma$ from one level down**, wearing a $\tau$ label because of §1.4. This does not invalidate the measurement; it means the label *"expert error"* is depth-relative and the 66%/19% split should be read as *"at the depth the harness cuts"*. Unverified, and checkable by re-cutting the same run at a different depth.

## 1.5 The composite's stability constant

$$L_{\mathcal C}\;\le\;\lVert\mathcal A_{\mathcal C}\rVert\cdot\max_{i\in\mathcal C}\operatorname{Lip}_v(\mathcal E_i)\cdot\Bigl(1+\frac{C_\mu C_{\mathcal G}}{\beta_{\text{int}}}\Bigr)$$

**$\beta_{\text{int}}$ appears here and in the proviso of §1.1, and that is not a coincidence worth passing over.** A subassembly with a badly conditioned internal interface is a bad plug-in *twice*: it is close to being an illegal expert at all, and the composite it does present amplifies everything above it. **Internal conditioning is a property of the plug, not of the socket** — which means it can and should be measured once, by whoever publishes the composite, and shipped with it.

## 1.6 The field that does not close, and why it matters most

`validity` composes logically — the composite is valid where all its parts are — but the conjunction is evaluated at the **internal solution**, which does not exist until the composite has been run. So:

> **A composite's validity predicate is post-hoc.** It cannot be evaluated at declaration time, which is where every other admissibility check in [[end-to-end-architecture-spec]] happens.

Two honest options, and the choice is recorded:

- **A conservative outer approximation** — declare a validity region contained in the intersection of the parts' regions under a bound on the internal solution. Requires an a priori estimate of the internal state that the framework does not have.
- **`validity: unknown-until-run`**, with the conjunction checked *during* the internal solve and a declination raised the moment any part declines.

> **Decision: the second.** A composite declares `validity: deferred`, and its validity check is performed inside its own step, raising a declination that propagates upward. This is the only option that does not require inventing an a priori state estimate, and it is why §3 — abstention semantics — is a prerequisite for hierarchy rather than an independent nicety. **The first option is not refused; it is simply not available yet, and a future a priori estimate would upgrade `deferred` to a declared region.**

---

# 2. Theorem 2 — the substitution lemma

## 2.1 What a swap is

Replace expert $i$ by $\mathcal E_i'$ declaring **the same port list**. That is what "the same plug" means; anything else is a graph edit, not a substitution.

## 2.2 What moves, and by how much

Let $\Delta=\tilde\Lambda_i'-\tilde\Lambda_i$ be the difference of the two probed blocks — **two probes, no rollout.**

**Conditioning.** $\tilde\Lambda=\sum_j\tilde\Lambda_j$, so the assembled operator moves by exactly $\Delta$, and by Weyl's inequality for singular values,

$$\bigl\lvert\,\beta'-\beta\,\bigr\rvert\;\le\;\lVert\Delta\rVert \qquad\Longrightarrow\qquad \beta'\;\ge\;\beta-\lVert\Delta\rVert .$$

**Transmission infidelity.** The exact $\Lambda$ is a property of the true problem and does not move, so $\lVert\Lambda-\tilde\Lambda'\rVert\le\lVert\Lambda-\tilde\Lambda\rVert+\lVert\Delta\rVert$ and

$$\sigma'\;\le\;\frac{C_\mu}{\beta-\lVert\Delta\rVert}\Bigl(\bigl\lVert\Lambda-\tilde\Lambda\bigr\rVert+\lVert\Delta\rVert\Bigr)\bigl\lVert\lambda^\star\bigr\rVert .$$

**Stability.** $L'\le\lVert\mathcal A\rVert\cdot\max\bigl(\operatorname{Lip}_v(\mathcal E_i'),\ \max_{j\neq i}\operatorname{Lip}_v(\mathcal E_j)\bigr)\cdot\bigl(1+C_\mu C_{\mathcal G}/\beta'\bigr)$.

**Agent infidelity.** $\tau_i\to\tau_i'$, additively and with no coupling to the rest — the one term that is genuinely local.

## 2.3 The substitution certificate

Everything above is computable from **two probes of one agent** and quantities the framework already emits. So:

> **Substitution certificate.** A swap preserves admissibility without re-certifying the graph iff
> $$\lVert\Delta\rVert\;<\;\beta-\beta_{\min}$$
> where $\beta_{\min}$ is the conditioning floor the compiler requires. Otherwise the seam must be re-certified and $\sigma$, $L$ and $T_{\text{pred}}$ re-derived.

**This is the plug-in guarantee, stated as an inequality for the first time.** It costs $m+1$ solves on one agent — the cost the framework already pays once per expert — and it is checkable before the new expert is ever run in a rollout.

## 2.4 Passivity is the only modular property, and that promotes G22

Look at what §2.2 requires re-measuring: $\beta$, $\sigma$, $\kappa$, the fitted $L$, $T_{\text{pred}}$. All global, all invalidated by a local change.

Now suppose both $\mathcal E_i$ and $\mathcal E_i'$ are **incrementally passive**. The $L\le1$ branch of [[master-error-bound]] §6.1 follows from a per-part property plus the interconnection structure, and **references no global quantity at all**. Therefore:

> **Modularity theorem. If every expert is incrementally passive and the interconnection is power-preserving, then $L\le1$ survives any substitution of a passive expert for a passive expert, with no re-measurement of anything.** It is the only property in the framework with this closure.

**This changes what G22 is for.** [[generalization-requirements]] G22 records that no expert declares a storage function, and rates it as *"the only structural route to $L\le1$"* — an accuracy argument. It is more than that: **a declared storage function is the enabling condition of plug-in itself**, because it is the one certificate that does not have to be re-earned when a part is swapped. A library of passive experts is composable and re-composable at no certification cost; a library of merely accurate experts must be re-certified on every swap.

**[AI Inference]:** if that is right, the priority of G22 should rise on *both* lists in [[generalization-requirements]], and it is currently on neither list's blocking tier. It is also cheap: [[probed-dtn-coupling]] §4.2 shows the certificate is an eigenvalue of a matrix the probe already assembles.

## 2.5 What is *not* true — and this one will surprise a builder

> **There is no monotonicity theorem. A strictly better expert can produce a strictly worse composition.**

The master bound is monotone in $\tau_i$; **the error is not**, because the bound is an upper bound and because a swap moves $\beta$ as well as $\tau$. Concretely, an expert with smaller $\tau_i$ and smaller benchmark error $\varepsilon_i$ can present a $\tilde\Lambda_i'$ that lowers $\beta$, and $\beta$ sits in the denominator of **both** $\sigma$ and $L$ ([[master-error-bound]] §4, §6) — so it hurts additively and multiplicatively while $\tau$ improves additively once.

**The one-line version for a builder:** *upgrading an expert can degrade the composition, the degradation is in a quantity nobody currently measures, and the substitution certificate of §2.3 is exactly the check that catches it before the rollout.* This is the same failure shape as [[probed-dtn-coupling]] §4.5's finding that a highly accurate frozen checkpoint scores $\Xi=0$ — **accuracy and composability are different axes, and here they are shown to be capable of pointing in opposite directions.**

---

# 3. Theorem 3 — conformance, or: nothing checks that a declaration is true

## 3.1 The gap

[[end-to-end-architecture-spec]] §3.1 states the architecture's central principle: *"Everything the framework is allowed to conclude about a composition is inherited from labels, never from inspecting the box."* Correct, and it has a corollary nobody wrote: **a false label produces a guarantee that does not hold, with no error and no failed gate.** That is the vault's signature failure mode ([[generalization-requirements]] §D) applied to the declaration layer itself, and the declaration layer is the one layer where nothing checks anything.

For a single hand-built case study this is tolerable, because the person who declared the record built the expert. **For a plug-in architecture it is the foundational hole**, because the point of plug-in is that those are different people.

## 3.2 Which fields fail silently, and what tests them

| field | conformance test | cost | silent if false? |
|---|---|---|---|
| `ports`, geometry | schema check against the port algebra | free | no — structural |
| `bc_channel` | impose a non-constant trace and check the response exceeds the probe floor: $\Xi_i>0$ | **the probe itself** | **yes** |
| `storage` (passivity) | $\lambda_{\min}\bigl(\tfrac12(S_i+S_i^{\!\top})\bigr)\ge0$ | **eigenvalue of the probed matrix** | **yes** |
| `equivariances` | apply $g$, compare: $\lVert\mathcal E_i(gu)-g\,\mathcal E_i(u)\rVert$ | one extra solve per generator | **yes** |
| `nondim` | completeness is decidable from the port list; a *wrong value* shows in $\mathcal R(t)$ but is not attributable | free / detectable-not-attributable | **partly** |
| `differentiable` | JVP against a finite difference | cheap | no — disagreement is loud |
| `dt_native` | run at $\Delta t$ and $\Delta t/2$, compare | two short runs | no |
| `regime_law` | evaluate at two known states | cheap | yes, but low-consequence |
| `validity` | **none exists** — §3.4 | — | **yes, and it is the load-bearing one** |

**The result that makes conformance affordable:** the three silent fields with the largest consequences — `bc_channel`, `storage`, `equivariances` — are **all certified by machinery the probe already builds.** One probe per expert returns $\Xi_i$ (certifying `bc_channel`), the symmetric part's spectrum (certifying `storage`), and needs only one extra solve per symmetry generator for `equivariances`. **Conformance is not a new subsystem; it is a report over the probe that is already budgeted.**

> **And the vault has already run two of these tests by accident.** $\Xi=0$ for the frozen wind-farm checkpoint ([[probed-dtn-coupling]] §4.5) *is* a failed `bc_channel` conformance test. OP-6's mirror-equivariance failure *is* a failed `equivariances` conformance test. Both were discovered late, expensively, and framed as findings about a checkpoint. **Both are conformance failures, and both would have been caught at admission by a test costing one probe.**

## 3.3 The conformance certificate

```
ConformanceCertificate:
  expert_id      : name + hash of the frozen weights
  record         : the declared ExpertCapabilities
  measured       : the measured value of every testable field
  residual       : declared − measured, per field
  probe_state    : the state and conditioning the probe was taken at
  suite          : the states validity was NOT falsified on
  verdict        : admit | admit-uncertified | refuse
  date           : when
```

Two fields are load-bearing beyond the obvious. **`expert_id` binds the certificate to a weight hash** — a retrained expert is a different expert and inherits nothing, which is the whole reason plug-in needs a certificate rather than a habit. **`probe_state` records that every measured quantity is local to a state** ([[probed-dtn-coupling]] §7): a conformance certificate is valid at a state and a regime, never globally, and a certificate quoted outside its probe regime is the same error as a trajectory metric quoted past $T_{\text{pred}}$.

## 3.4 `validity` is falsifiable but not verifiable — the one axiom

An expert's `validity` predicate claims to bound where the expert is competent. **Verifying it requires knowing where the expert is wrong, which requires the reference that the predicate exists to make unnecessary.** So:

> **`validity` cannot be certified. It can only be falsified** — by finding a state inside the declared valid region where the expert is measurably wrong against a reference that happens to exist there.

Two consequences, and the second is the one to carry into a build.

1. The certificate must record **"not falsified on suite $X$"**, never "valid". The `suite` field above exists for exactly this, and the honest verdict on `validity` is always provisional.
2. **A plug-in architecture rests on exactly one unverifiable declaration, and it is the one every bound is conditioned on** — [[composition-error-theory]] §4.5's abstention precondition, which *"none of the four constructions removes."* Naming it as the single axiom is worth more than pretending it is checkable: it says where to spend falsification effort, and it says that the correct response to an unfalsified `validity` is `admit-uncertified`, not `admit`.

**[AI Inference]:** this also explains why G19 (reference availability) keeps reappearing under different names. Reference availability is not one gap among twenty-four — it is the resource that the single unverifiable field consumes, so every attempt to make `validity` trustworthy converts directly into a demand for references. Stated that way, G19 is a *cost curve* on the architecture's one axiom rather than an independent gap.

---

# 4. Abstention under composition, and the second horizon

## 4.1 The semantics, decided

[[port-algebra-atlas-0.1]] §8 and [[generalization-requirements]] G8 both require that an expert be able to **decline**. Neither says what the *composition* does when one does. Decided here:

1. **A declination halts the rollout at $t_{\text{decline}}$.** The claim is typed on $[0,t_{\text{decline}})$ and emitted; the run is not a failure, it is a **shorter answer with a stated reason.** This is the only behaviour consistent with the three-verdict system: continuing past a declination is running with a hypothesis known to be false, which is the silent-wrongness class.
2. **A fallback expert is a mid-rollout substitution**, and by §2 it changes the composed operator at $t_{\text{decline}}$. A changed operator mid-rollout is not covered by a bound derived for a fixed one, so it reconciles the same way [[end-to-end-architecture-spec]] §12.3 reconciles a graph mutation: **as a restart with $e^0\neq0$**, reactivating the $L^N\lVert e^0\rVert$ term. An `AbstentionPolicy` therefore shares machinery with `TopologyEvent` and inherits its refusal — **a fallback declared without a state map and an injected-error record is refused.**
3. **A composite declines when any constituent declines** (§1.6), and the declination propagates upward with the identity of the originating agent attached.

## 4.2 The abstention horizon

[[port-algebra-atlas-0.1]] §8 observes that with 20 experts the probability that at least one is out of distribution approaches 1, *"and it gets worse as the vocabulary gets cleaner."* That is a horizon, and it has never been written as one.

Let $p$ be the per-agent per-macro-step declination probability and $K$ the number of agents. Under independence,

$$\Pr[\text{no declination through }N\text{ steps}]\;=\;(1-p)^{KN}\;\approx\;e^{-KpN} \qquad\Longrightarrow\qquad \boxed{\;T_{\text{abs}}\;\approx\;\frac{\Delta t}{K\,p}\;}$$

and the rollout's usable length is

$$T_{\text{usable}}\;=\;\min\bigl(T_{\text{pred}},\ T_{\text{abs}}\bigr).$$

**Three things follow, and the third is the reason this is on a theory page rather than in a worklist.**

1. $T_{\text{abs}}$ degrades **linearly in $K$** — and $K$ is the axis the roadmap grows along, from $K\approx4$ today to $K\approx15$–$20$. $T_{\text{pred}}$ does not depend on $K$ at all.
2. So **for a large enough library the binding horizon is abstention, not chaos** — a limit that is entirely a property of the framework rather than of the physics, and therefore one the framework can actually do something about.
3. **It is the first quantity that makes the cost of a bad `validity` declaration visible.** An expert with a needlessly conservative validity region shortens every rollout it participates in, linearly. Nothing currently prices that, and $p$ is emittable per expert from any run.

**[AI Inference], and the caveats are not small.** Independence is wrong — agents in the same flow go out of distribution together, so the real $T_{\text{abs}}$ is longer than this when regimes are correlated and shorter when a single upstream excursion cascades. $p$ has never been measured for any expert, because no expert declares `validity` at all. The estimate is a scaling law, not a number, and its value is the $K$-dependence rather than the constant.

---

# 5. Routing — G15, given a criterion

[[end-to-end-architecture-spec]] §0.1 flags expert↔subdomain routing as *"a real missing layer between L2 and L1"*, out of scope and *"flagged, not filled."* [[theory-closure-audit]] classifies G15 as engineering rather than theory. **The engineering classification is right about the mechanism and wrong about the criterion**: what to route *by* is a theory question, and the vault has already answered it twice without noticing.

## 5.1 The rule

$$\mathrm{route}(\Omega,r)\;=\;\operatorname*{arg\,max}_{i\,\in\,\mathcal F(\Omega,r)}\ \Xi_i \quad\text{, ties broken by } \min\tau_i,\qquad \mathcal F=\Bigl\{\,i:\ \text{ports match}\ \wedge\ \texttt{validity}_i(\text{state},r)\ \wedge\ \text{certificate valid at } r\,\Bigr\}$$

and **$\mathcal F=\varnothing$ is a refusal, not a nearest-neighbour fallback.**

**Why $\Xi$ and not accuracy.** [[schwarz-iteration-atlas-0.1]] §8 ends on the wish that *boundary-condition flexibility, not accuracy, should be the primary expert-selection criterion*; [[probed-dtn-coupling]] §4.5 turns that wish into a norm, the composability index $\Xi_i=\lVert\Lambda_i^{\text{expert}}\rVert/\lVert\Lambda_i^{\text{ref}}\rVert$. Ranking a library by benchmark accuracy selects for exactly the property [[expert-library-atlas-0.1]]'s axes call invisible to accuracy benchmarks — and §2.5 above shows the two axes can point in opposite directions. **Routing by accuracy is the substitution failure of §2.5 committed automatically, at every subdomain, by the compiler.**

## 5.2 The structural fact about routing

> **Routing is the only layer whose input is a state and whose output is a graph edit.** Every other layer takes a declaration and produces a scheme.

Therefore:

- **Static routing** — decided once at compile time from the declared initial regime — is a compile-time layer (call it L1.5) and disturbs nothing. **This is the only mode admitted today.**
- **Dynamic routing** — re-selecting an expert mid-rollout because the regime moved — **is a topology mutation**, and inherits `TopologyEvent`'s refusal and its restart-with-$e^0\neq0$ reconciliation. It is also, by §4.1, the same object as an abstention fallback.

**[AI Inference]:** that dynamic routing, abstention fallback, and topology mutation are three names for one operation is asserted here and appears nowhere else. If it holds, the framework needs **one** mechanism rather than three, and the `TopologyEvent` slot of [[end-to-end-architecture-spec]] §12.3 is more central than it was filed as — it is not only about staging and docking, it is the mechanism every mid-rollout change routes through.

---

# 6. The four new gaps

Opened here because no existing G-number covers them.

| # | Gap | Status after this page | What remains |
|---|---|---|---|
| **G25** | **Closure under composition.** Is a composed subsystem a pluggable expert? | **Answered**: yes for 8 of 13 record fields, given $\beta_{\text{int}}>0$; grouping-invariant by the quotient property | Three fields (`L_native`, `regime_law`, `validity`) have no composition rule and must be re-declared per composite. The depth-tagging of emitted defects (§1.4) is unimplemented |
| **G26** | **Substitution semantics.** What does swapping one expert cost? | **Answered**: certificate $\lVert\Delta\rVert<\beta-\beta_{\min}$, from two probes. Passivity is the only modular property | No monotonicity theorem exists and none is expected; the certificate is untested on any real swap |
| **G27** | **Conformance.** Nothing checks a declaration is true | **Answered for every field but one**, and nearly free — the probe certifies the three consequential silent fields | `validity` is falsifiable but not verifiable and is the architecture's single axiom. No conformance test has ever been run |
| **G28** | **Abstention under composition.** What does the composition do when an agent declines? | **Decided**: halt and type the claim on $[0,t_{\text{decline}})$; a fallback is a restart. $T_{\text{abs}}\approx\Delta t/(Kp)$ | $p$ is unmeasured for every expert, because no expert declares `validity`. The independence assumption in $T_{\text{abs}}$ is known to be wrong |

---

# 7. Where this page is guessing

| # | The guess | Status |
|---|---|---|
| 1 | **The interface/regime split of the capability record** (§1.3) — that the three non-composing fields fail for one reason and the fix is to re-declare rather than derive | **[AI Inference]** |
| 2 | **That OP-5's expert/architecture split is depth-relative** (§1.4) — plausible from the theorem, unverified, and checkable by re-cutting one run | **[AI Inference]** |
| 3 | **That passivity's modularity should raise G22's priority on both gap lists** (§2.4) — the theorem is derived; the prioritization is a judgement | derived; the priority call is **[AI Inference]** |
| 4 | **That G19 is a cost curve on one axiom rather than an independent gap** (§3.4) | **[AI Inference]** |
| 5 | **$T_{\text{abs}}$'s independence assumption** (§4.2) — known to be wrong in both directions, so the scaling in $K$ is the claim and the constant is not | **[AI Inference]**, flagged in place |
| 6 | **That dynamic routing, abstention fallback, and topology mutation are one operation** (§5.2) — the strongest unification claimed here and the least supported | **[AI Inference]** |
| 7 | **The routing rule's ranking by $\Xi$ with $\tau$ as tie-break** — the *axis* is well supported by two prior pages; the *scalarization* is a design call | **[AI Inference]** |

**What this page is not guessing about:** the Schur-complement form of the composite operator and its transitivity, the congruence that carries positive-realness through nesting, the $\tau_{\mathcal C}\le\tau+\sigma+\gamma$ attribution bound, the Weyl perturbation of $\beta$, and the observation that `validity` admits no verification test. Those are derivations from results this vault already proved.

---

# 8. What this changes

1. **The framework is closed under composition**, so hierarchy is legal and the $O(K)$ port argument survives to the library size the roadmap targets. The condition is $\beta_{\text{int}}>0$ and it is already measured.
2. **Grouping does not change the answer but does change the attribution.** $\sigma$ and $\gamma$ at one depth become $\tau$ at the next, so **every emitted defect must carry its depth**, and the emit contract does not yet.
3. **Substitution has a certificate** — two probes, no rollout — and **passivity is the only property that survives a swap uncertified**, which promotes G22 from an accuracy nicety to the enabling condition of plug-in.
4. **Better is not better.** A more accurate expert can worsen a composition through $\beta$; §2.3's inequality is the check.
5. **Conformance is nearly free and has never been run** — and two of the vault's expensive discoveries ($\Xi=0$, OP-6's equivariance failure) were conformance failures found the slow way.
6. **The architecture rests on exactly one unverifiable declaration**, `validity`, and it should be labelled as an axiom rather than a field.
7. **There are two horizons, not one.** $T_{\text{abs}}\approx\Delta t/(Kp)$ degrades linearly in library size, and for a large library it, not chaos, may be what ends the rollout.
8. **Routing has a criterion** — rank by composability, refuse on an empty feasible set — and dynamic routing is a topology mutation rather than a scheduling decision.

---

## See Also

- [[end-to-end-architecture-spec]] — the one-graph, one-level path this page composes, swaps and certifies; its §10.2 emit contract needs a depth field (§1.4) and its §12.3 slot turns out to be the mechanism every mid-rollout change routes through (§5.2)
- [[interface-transfer-theory]] — the companion page; its $P_i^\ast\Lambda_iP_i$ congruence is what carries passivity through the nesting of §1.3
- [[master-error-bound]] — §7's bound at $N=1$ is §1.4's attribution theorem; §4 and §6's shared $\beta$ is why §2.5's failure mode exists
- [[probed-dtn-coupling]] — supplies $\beta_{\text{int}}$, the passivity spectrum, and $\Xi$; §4.5's $\Xi=0$ is reinterpreted here as a failed conformance test
- [[port-algebra-atlas-0.1]] — §5.1's open ports are §1.3's composite port list; §8's OOD observation becomes §4.2's horizon
- [[generalization-requirements]] — G22 is promoted by §2.4, G19 is reframed by §3.4, G15 gains a criterion in §5, and G25–G28 are added
- [[theory-closure-audit]] — its three-class sort has no room for these three properties, which is the finding of §0
- [[general-coupling-scheme]] — §2's capability record is what §1.3 composes field by field and §3 certifies
- [[expert-library-atlas-0.1]] — where a conformance certificate would live alongside each expert
- [[composition-error-theory]] — §4.5's abstention precondition is §3.4's single axiom, named
- [[conservation-as-constraint-atlas-0.1]] — *enforce, measure, or decline*; §4 specifies what the composition does with the third
- [[gap-worklist]] — where G25–G28 land as tracked items
