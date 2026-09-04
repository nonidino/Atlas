# The End-to-End Architecture Specification — Atlas as one system, with its holes named

**Type:** Concept page — **framework specification, binding**, version-independent (folder: `Atlas 0.1/common/`)
**Status:** written 2026-08-27, executing the operational conclusion of [[theory-closure-audit]] §6: *write the architecture down end to end before implementing, so the five envelope-widening gaps surface on paper rather than late in a build.* This page specifies **no new theory**. It specifies the **system**, states which of the seven hypotheses each layer leans on, and gives each unsolved gap a **named slot with a declared interface** so the architecture has a correctly shaped hole rather than a silent assumption.
**Related:** [[theory-closure-audit]] · [[master-error-bound]] · [[general-coupling-scheme]] · [[port-algebra-atlas-0.1]] · [[generalization-requirements]] · [[gap-worklist]] · [[probed-dtn-coupling]] · [[temporal-error-accumulation]] · [[composition-error-theory]] · [[conservation-as-constraint-atlas-0.1]] · [[expert-library-atlas-0.1]] · [[agent-definition-atlas-0.1]] · [[global-fields-and-topology-atlas-0.1]] · [[schwarz-iteration-atlas-0.1]] · [[symmetry-averaging-atlas-0.1]]

> **The one-line version.** Nine layers, one spine. The spine is the **seven-hypothesis envelope** of [[theory-closure-audit]] §3: every layer declares which hypotheses it consumes, what it does when one fails, and where it **refuses**. Two things come out of writing it end to end that no single-layer page could produce. First, **five named slots** — `SeamReference`, `InterfaceMotion`, `TopologyEvent`, `AssemblyCertificate`, `PortAmendment` — one per envelope-widening gap, each with a declared interface and *no solution*, plus a required measurement so the hole is instrumented rather than merely admitted. Second, a **decision**: claims are typed **trajectory inside $T_{\text{pred}}$, statistical outside**, the statistical type is **declared and refused rather than approximated**, and under that rule **every result currently in the vault types as `UNTYPED`**, because $L$ has never been measured.

---

# 0. Scope, and the two conventions everything below uses

## 0.1 What this page specifies, and what it does not

**In scope:** the **inference-time composition path** — from an expert sitting in a library to a typed, instrumented claim about a coupled rollout.

**Out of scope, deliberately, and named so the omission is not mistaken for coverage:**

| Not specified here | Where it lives | Why it is out |
|---|---|---|
| Training, pretraining, bootstrap of an expert | [[training-and-bootstrap-atlas-0.1]], [[incremental-transfer-roadmap]] | The composition path treats an expert as frozen and declared. Every guarantee below is inherited from the declaration, not from how the weights were obtained |
| **Expert↔subdomain routing from a library** (G15) | unaddressed anywhere | A real missing layer between L2 and L1. This spec assumes the assignment is **declared**, as [[agent-definition-atlas-0.1]] already assumes the agent list is. **Flagged, not filled** |
| Learned graph/edge discovery | [[edge-generation-atlas-0.1]] Mechanism B (deferred) | Declared, not discovered, throughout |
| The interior of an expert | [[expert-library-atlas-0.1]], [[unet-hierarchy-atlas-0.1]] | The composition layer sees ports and a step function, nothing else |

**[AI Inference]:** the nine layers below are the user-and-audit-supplied list, and **this page does not prove the list is complete.** G15 routing is one layer it is already known to be missing; a training-time counterpart to the capability record may be another. Treat "nine layers" as the current best decomposition, not a closure claim — the same status §12.5 gives the port vocabulary.

## 0.2 Convention one — three verdicts, not two

Every gate below returns one of three verdicts. This distinction is **introduced by this page** and is load-bearing everywhere after it, because a spec with only *run* and *refuse* is either unusably strict or silently permissive, and the vault has been bitten by the second.

| verdict | meaning | when it is used |
|---|---|---|
| **`admit`** | run, and the master bound applies with all constants reported | every hypothesis the configuration needs is **checked and holds** |
| **`admit-uncertified`** | **run, but no bound is claimed.** The output is produced, the diagnostics are emitted, and the claim carries the tag naming which hypothesis is unverified | a hypothesis is **unchecked** but its failure would make the answer *unbounded*, not *wrong* — E5, E6, E7 today |
| **`refuse`** | do not run this configuration; report the rule number and the quantity that would have been silently wrong | the configuration belongs to the **silent-wrongness class**: it produces a number, no error, no failed gate, and the number means something other than what it is read as |

> **The rule that decides which:** *refuse* the silent-wrongness class; *decertify* the unverified-hypothesis class. Every Tier-1 item of [[gap-worklist]] (cross-points, reversed mapping, non-conservative subcycling, non-adjoint reduction) is silent-wrongness and therefore a **refusal**. Every unchecked envelope hypothesis is a **decertification**. That single line is the operational form of [[generalization-requirements]] §D's observation that *every strictly-blocking gap is a gap about silence.*

## 0.3 Convention two — how an unsolved slot is specified

There are five things this architecture cannot do and does not know how to do. Each gets a slot rather than an assumption. A slot is specified by five fields, and **the fifth is the one that makes a hole useful rather than merely honest**:

1. **What is missing** — stated as the hypothesis whose removal creates it.
2. **The declared interface** — the signature the eventual solution must present, so the layers around it can be built now.
3. **Default behaviour today** — always a refusal or a decertification, never a plausible-sounding rule.
4. **What any solution must satisfy** — the invariants a future construction is graded against.
5. **The measurement required now** — *the number the missing rule would constrain*, emitted from every run that touches the slot, even though nothing yet says what value it should take.

**[AI Inference]:** field 5 is this page's central methodological bet, and it is the vault's own lesson turned on its unsolved problems rather than its solved ones. The framework's recurring failure has been reporting the term that was easy to compute; the counter-move for a hole is to **emit the number the missing theory will need before the theory exists**, so that when the rule arrives it arrives with a dataset. It costs one line per slot in the emit contract.

---

## 0.4 Convention three — a reported quantity carries the temporal parameters that move it

**Added 2026-09-03 (CS-11). Binding on L7, L8 and L9.** Like §0.2's three verdicts, this is a convention rather than a construction: it does not compute anything new, it says what a number is not allowed to be reported without.

Three temporal parameters silently move a reported quantity in this architecture. They are not the same parameter, they are owned by different layers, and each has been discovered separately by a case study measuring something else:

| parameter | what it moves | how far it moved it, measured | the declaration it requires | found by |
|---|---|---|---|---|
| **exchange interval** $\Delta t_{\text{ex}}$ | $\sigma$, the transmission defect | $\sigma$ ranged $7.70\times$ over five consecutive macro-steps of one seam, and $1.48\times$ between two steps whose scalar lags agreed to $8\%$ | **`sigma_lag`** — the lag, as a number on `MeasuredConstants`, not a phrase inside `probe_state` | **W86**, CS-5 |
| **clock ratio** $r=\Delta t_{\text{ex}}/\Delta t_i$ | nothing, at a seam whose fast agent sub-steps internally — and this is the finding, not the absence of one | $1.00016\times$ over $8\times$ of ratio at a fixed interval; against $525\times$ in $\sigma$ over the same span of ratio when the *interval* moves with it | **the interval, not the ratio.** A multirate $\sigma$ is declared against $\Delta t_{\text{ex}}$; the ratio is quoted as **scope**, because the independence is measured for an agent that can refine its own march and is untested for one that cannot | **W131**, CS-11 |
| **rollout horizon** $N$ | any quantity read off a rollout — a design sensitivity, a cut criterion, a composed defect | $\mathrm dJ/\mathrm d\theta$ **changed sign** between horizons in two independent case studies, on different physics, with every value confirmed by its own finite difference | **`horizon`** — the $N$ the quantity was taken at, and the **validity limit** past which it is not determined | **W128**, CS-10; **W123**, CS-9★ |

### 0.4.1 The rule

> **A quantity read off a rollout is reported with the horizon it was read at, and a quantity read off an exchange is reported with the interval it was read at. A report without them is refused, not decertified** — it is the silent-wrongness class of §0.2 exactly: it produces a number, no error, no failed gate, and the number means something other than what it is read as.

The asymmetry with §0.2's other refusals is worth naming: **these are refusals of a *report*, not of a *run***. The computation is fine. Every finite difference in CS-10 §6 confirms its adjoint to three or four digits, at every horizon, *including the horizon whose sign is wrong*. What is refused is quoting the number without the parameter that determines it.

### 0.4.2 What "a validity limit" means, and why it is $T_{\text{pred}}$ one level down

§11.3 already types a *trajectory* claim by the horizon past which the master bound goes vacuous. A rollout-derived scalar needs the mirror image of that statement — a horizon **before** which it is not yet the number it converges to:

$$N_{\text{valid}}(\varepsilon) \;=\; \min\Bigl\{\,N \;:\; \bigl\lvert q(N') - q(\infty)\bigr\rvert \le \varepsilon\,\lvert q(\infty)\rvert \quad\text{for all } N' \ge N \,\Bigr\}$$

and the two bracket the usable range: **a rollout quantity is meaningful on $N_{\text{valid}}(\varepsilon) \le N \le T_{\text{pred}}/\Delta t$, and that window can be empty**. When it is empty the honest report is that no horizon exists at which the quantity is both converged and inside the bound, which is a statement the architecture can make and currently cannot.

The **sign** is the coarsest case of the same thing and is reported separately, because it is what an optimiser consumes and because it is decidable where a tolerance is not:

$$N_{\text{sign}} \;=\; \min\Bigl\{\,N \;:\; \operatorname{sign} q(N') \text{ is constant for all } N' \ge N \,\Bigr\}$$

**A design search run below $N_{\text{sign}}$ moves the knob the wrong way**, and both CS-10 and CS-11 measured a graph where it does.

### 0.4.3 The declarations, stated as record fields

Three fields, and each is a number rather than prose. This is W86's discipline generalized: that row existed because $\sigma$'s lag *was* quoted, in a sentence nothing could read.

| field | on | meaning | absent |
|---|---|---|---|
| `sigma_lag` | `MeasuredConstants` | the lag $\sigma$ was measured at | **exists since W86.** `solve.rollout` derives the run's own lag and `multiphysics.check_sigma_lag` compares; the comparison can falsify and can never confirm |
| `sigma_interval` | `MeasuredConstants` | the **exchange interval** $\sigma$ was measured at, with the clock ratio beside it as scope | **decertify** — a multirate $\sigma$ without its interval names no scale. CS-11 supplies the law that makes the field evaluable at another interval rather than only comparable |
| `horizon` | every rollout-derived quantity | $N$, plus $N_{\text{valid}}(\varepsilon)$ at a declared $\varepsilon$ and $N_{\text{sign}}$ | **refuse.** This is the one §0.4.1 promotes, and it is promoted rather than decertified because a gradient quoted at an undetermined sign is not a weaker claim than one quoted at a determined sign — it is a differently-signed one |

**The third is the one this convention adds and it is deliberately the strictest.** $\sigma$ without its lag is a bound with unknown tightness, which is a decertification; a sensitivity without its horizon can point the other way, which is not. **[AI Inference]:** the same argument makes $N_{\text{sign}}$ the field an optimiser should be handed rather than $N_{\text{valid}}$ — a search tolerates a $2\times$ error in a gradient's magnitude and does not tolerate its sign — but no optimiser has been run in this loop yet and that ordering is argued rather than measured.

### 0.4.4 Where the emit contract picks it up

§10.2's table gains one row under **claim** — *the per-quantity horizon table*, keyed by quantity, carrying $N$, $N_{\text{valid}}$ and $N_{\text{sign}}$ — which is the same shape §11.3's box already argued $T_{\text{pred}}$ needs and for the same reason: **tolerance is a property of the quantity being claimed, so a run has one $L$ and as many horizons as it has quoted quantities.** The two tables are the two ends of the same window and should be emitted together.

---
# 1. The spine — the envelope as a machine-checkable stamp

[[theory-closure-audit]] §3 states the seven hypotheses under which [[master-error-bound]] is a theorem. This page makes them a **stamp emitted by every run**, with three possible values per hypothesis: `holds` (checked), `fails` (checked), `unchecked`.

| # | Hypothesis | Owned by layer | Checked how | If `fails` | If `unchecked` |
|---|---|---|---|---|---|
| **E1** | Interaction graph **fixed** for the whole rollout | L2, L7 | Static: does the declaration contain a `TopologyEvent`? | `TopologyEvent` slot (§12.3); restart semantics with $e^0\neq0$; **refuse** if no ledger | n/a — always checkable |
| **E2** | Interfaces **static**, and the interface transfer **accounted for** — exactly by coincidence, or by a declared prolongation whose error enters $\sigma$ | L2, L3 | Static: motion class per port. Transfer: coincidence test on the two discretizations, **or** the declared $(M,P_i)$ of §5.2 | `InterfaceMotion` slot (§12.2) for motion — **refuse**. Non-coincidence is **admissible** through a stable declared $P_i$ ([[interface-transfer-theory]] §4.1) and **decertifies** rather than refusing; only an unstable or unimplementable $P_i$ fails it | n/a — always checkable. *A missing declaration is an L3 refusal that sets nothing here — see §1.1* |
| **E3** | A **single global evolution operator** exists, so $\tau$ has a referent | L1, L4, L9 | Static: do all agents at a seam declare the same governing family? | `SeamReference` slot (§12.1); $\tau$ is `UNDEFINED` at that seam; **decertify** | n/a — always checkable |
| **E4** | A **single macro-step clock**, or proved-conservative multirate | L7 | Static: are all `dt_native` equal? Dynamic: is flux matched time-integrated? | R9 required; **refuse** pointwise-in-time matching under multirate | n/a — always checkable |
| **E5** | The claim is a **trajectory** claim inside $T_{\text{pred}}$ | L9 | Requires $\hat L$ (W1). Then arithmetic | Claim types `S-CLAIM`, which is **declared and refused** (§11) | **`UNTYPED`** — the state of every result in the vault |
| **E6** | Assembly bounded with $\lVert\mathcal A\rVert$ known; the blend at least as accurate as what it blends | L6 | *Half checkable*: the partition-of-unity identity is a machine-precision test. *Half open*: the accuracy condition is `AssemblyCertificate` (§12.4) | **refuse** if the PoU identity fails; **decertify** on the accuracy half | **decertify**, and emit $\lVert\mathcal A\rVert$ regardless |
| **E7** | Interconnection **power-preserving**, every reduction/prolongation pair **adjoint** | L3, L1 | Adjointness is a numerical identity test; passivity is $\tfrac12(S_i+S_i^{\!\top})\succeq0$ on the probed block. **Both are measurements** — with no declared pair and no assembled block, neither runs | **refuse** a *measured* non-adjoint field↔lumped pair (silent power leak); **decertify** the $L\le1$ branch if $\pi_i>0$ above the floor of [[probed-dtn-coupling]] §4.2 | **decertify** the $L\le1$ branch; $L$ falls back to fitted. **An L3 refusal for a *missing* declaration lands here, not in `fails`** — see §1.1 |

**Read the `unchecked` column.** Only three hypotheses can be `unchecked` at all — E5, E6, E7 — and those are exactly the three [[theory-closure-audit]] §3 records as unverified for the wind farm. **E1–E4 are static properties of a declaration and are decidable at compile time, before any compute is spent.** That asymmetry is the reason the envelope belongs in the compiler rather than in a review checklist.

## 1.1 A stamp value is not a gate verdict

**The stamp records what was established about a hypothesis; the verdict records what the compiler did.** They are different objects and the table above sets both, so they need separating explicitly — otherwise §13.1's own trace is unreproducible from this page.

They come apart in exactly one direction. A layer may refuse for a **missing declaration**, and a missing declaration establishes nothing about the hypothesis. So:

- **E2 can read `holds` alongside an L3 refusal.** The 2026-08-27 amendment at §5.2 changed the **connection rule**, not the hypothesis. A seam that is static and geometrically coincident satisfies E2 and is still refused for want of the common interface space and prolongation the amended rule requires. That is the wind farm exactly. Note also that **coincidence is itself a declaration that nothing checks** — [[plug-in-composition-theorems]] §3's argument about unverifiable declarations applies to it in full.
- **E7 reads `unchecked` alongside an L3 refusal.** §5.3's *"the only envelope hypothesis whose failure is a refusal"* is a statement about the **connection verdict**. With no declared pair there is nothing to test adjointness on and no block to take a spectrum of, so the stamp value is `unchecked`. Only a *measured* non-adjoint pair, or a measured negative mode in the probed symmetric part, writes `fails`.

**The general rule, and it is worth stating as one: only a measurement writes `fails`.** Absence of evidence is `unchecked`, and a compiler that writes `fails` for a missing field is reporting a schema error as a physics result — which is the same substitution the whole page exists to prevent, committed one level up.

> **The envelope stamp is the artifact this page most wants to exist.** A rollout that emits $(\text{E1}\dots\text{E7})$ alongside its numbers is a rollout whose reader can tell, without reading any theory page, whether the bound applies to it. Nothing in the vault currently emits it, and no result currently in the vault could produce a stamp with fewer than three `unchecked` entries.

---

# 2. The layer stack

| L | Layer | Consumes | Produces | Envelope | Refuses when |
|---|---|---|---|---|---|
| **L1** | Expert declaration & capability record | an expert, frozen | a complete, machine-checkable record, **and the achievable transmission rung** — R1 lifted by R2, decidable from the records alone (§2.1) | enables E3, E7 checks | a record missing a field a downstream layer needs |
| **L2** | Decomposition & cut placement | a physical system, an expert set, **the rung** | a graph: agents, geometries, seams | E1, E2 | non-overlapping cuts with cross-points and no rule |
| **L3** | Port algebra & connection admissibility | the graph plus records | a typed, admissible edge set | E2, E7 | mapping class unset; non-adjoint field↔lumped; new port type outside `PortAmendment` |
| **L4** | Transmission operator construction | the edge set, probe budget | $\tilde\Lambda$ as a matrix, plus $\beta,\kappa,\Xi,\pi$ | E2, E3 | probe null space of the wrong dimension; $\Xi=0$ claimed as a coupling |
| **L5** | Interface solve & accelerator | $\tilde\Lambda$, residual $\mathcal G$ | the converged trace $\lambda^{(k)}$, and $\gamma$ | E1, E4 | $\varepsilon_{\text{tol}}$ tighter than $\min(\hat\tau,\hat\sigma)$; degenerate axis; multiplicative ordering under a composition guarantee |
| **L6** | Assembly | local solves at $\lambda^{(k)}$ | the global state $u^{n+1}$ | E6 | the partition-of-unity identity fails |
| **L7** | Time integration & multirate | the coupled step, clocks | the rollout | E1, E4 | pointwise-in-time flux matching under multirate; $W>1$ without `bc_time_varying` |
| **L8** | Emitted diagnostic set | everything above | the run artifact | reports all seven | a run that cannot emit the stamp |
| **L9** | Typing of claims | the artifact, $\hat L$ | a typed claim, or a refusal | E5 | a trajectory metric quoted past its own horizon |

The layers are **ordered by dependency, not by execution**. L1–L4 are compile-time and run once; L5–L7 run per macro-step; L8–L9 run at emit time. **Every refusal in the right-hand column is decidable before the first expensive solve except two** — the probe null space (L4) and the PoU identity (L6) — and both of those are cheap.

## 2.1 The rung is decided at L1, and the reason is the cross-point trade

**The transmission rung is decided by R1 lifted by R2, and both are decidable from L1's records alone** — R1 reads `bc_channel`, R2 reads `differentiable` and the probe budget. Nothing about the graph is needed. The rung then *fixes the decomposition axis*, because `probed-DtN` requires the non-overlapping view, and the axis is what decides **whether cross-points exist at all**. So the dependency is

$$\text{L1}\ \longrightarrow\ \text{rung}\ \longrightarrow\ \text{L2}\ \longrightarrow\ \text{L3}\ \longrightarrow\ \text{L4}\ \longrightarrow\ \dots$$

and the rung must be settled **before** L2 places a single cut. Deciding it later — §6.2 assembles at the rung, and C7 at L3 already checks compatibility against it — lets a compile check cross-points against the **declared** axis and then switch to the one that has them, which is the silent-wrongness class committed inside the layer whose purpose is to make this trade visible rather than discovered. L4 *assembles* the operator; it does not choose the rung.

**This fires on the first useful configuration.** Give the wind farm a boundary-capable expert and R2 lifts the rung to `probed-DtN`, the axis goes non-overlapping, and L2 must refuse on cross-points that the overlapping decomposition never had. That is §4.3's trade — *"moving to the better transmission operator introduces a difficulty the present scheme does not have"* — and the ordering above is its consequence: **the difficulty arrives at L2, one layer before the operator that caused it.**

---

# 3. L1 — Expert declaration and the capability record

## 3.1 The intuition

An expert is a **frozen black box with a label on the outside**. Everything the framework is allowed to conclude about a composition is inherited from labels, never from inspecting the box. So the label has to carry every property any downstream guarantee depends on — and the recurring discovery of the last several pages is that the labels have been missing exactly the fields the guarantees needed.

## 3.2 The record

Extending [[general-coupling-scheme]] §2 by the fields the other layers of this spec turn out to require. **New in this page:** `mapping`, `reduction`/`prolongation` with an adjointness certificate, `lambda_ref`, `motion_class`, and `claim_types`.

```
ExpertCapabilities:
  # --- from general-coupling-scheme §2 ---
  bc_channel       : none | dirichlet | neumann | robin | ventcell
  bc_time_varying  : bool
  differentiable   : none | jvp | vjp | both
  dt_native        : float
  L_native         : float
  regime_law       : callable
  ports            : [PortDecl]
  storage          : H(u) | none
  equivariances    : [group]
  validity         : predicate(state, conditioning) -> bool

  # --- added by this spec ---
  governing_family : label            # E3: which continuum problem this discretizes
  lambda_ref       : Λ_ref | none     # a per-port reference operator, for τ at a seam (§12.1)
  claim_types      : [trajectory, statistical]   # what the expert is validated to support

PortDecl:
  type             : MECH | ROT | THERM | ELEC | ADVEC
  geometry         : face set
  direction        : bidirectional | in | out
  conservative     : bool             # enforced, or measured only
  mapping          : conservative or consistent      # W5 / G11 — DEFAULTED PER PORT TYPE
  motion_class     : static | prescribed | solution_dependent    # §12.2
  reduction        : R_Γ | none       # field → lumped
  prolongation     : P_Γ | none       # lumped → field
  adjointness      : certificate | none              # W8 / G3
  nondim           : {L, U, rho, ...}
```

**Two fields deserve their own sentence.**

`mapping` is defaulted **per port type, not per edge** — an `ADVEC` flow maps conservatively (integral preserved), a `THERM` effort consistently (pointwise preserved). This is [[port-algebra-atlas-0.1]] §9.3, flagged and unfixed since 2026-08-19; making it a field with a per-type default is what stops it being chosen by accident.

> **Amended 2026-08-27 — four of these fields collapse into one.** [[interface-transfer-theory]] §2 shows that `mapping`, `reduction`, `prolongation` and `adjointness` are not four independent declarations. Declare a **common interface space $M$** for the seam and **one prolongation $P_i:M\to V_i$ per side**; then the reduction is *forced* to be $P_i^\ast$ by power preservation, the mapping class is *derived* (efforts consistent through $P_i$, flows conservative through $P_i^\ast$), and adjointness is not a certificate to attach but the definition of the pair. The `PortDecl` above should read `interface_space : M` and `prolongation : P_i`, with the other four fields dropped. **This is strictly better than a per-type default, because a default can be overridden and a derivation cannot** — the classic backwards-mapping bug needs two declarations that can disagree, and there is now only one.

`governing_family` is the field that makes E3 decidable at compile time. Two agents sharing a seam and declaring different families is exactly the multiphysics seam of §12.1, and it is a **string comparison**, not an analysis.

## 3.3 Envelope, failures, refusals

| relies on | what it must do when it fails | compiler verdict |
|---|---|---|
| Nothing — L1 is where hypotheses become **checkable** | — | — |
| — | `storage: none` ⟹ the $L\le1$ branch of [[master-error-bound]] §6.1 is unavailable; $L$ must be fitted (W1) | **`admit-uncertified`**, E7 `unchecked` |
| — | `validity: none` ⟹ the run cannot abstain; every bound is vacuous wherever the expert is out of distribution and nothing says where that is | **`admit-uncertified`**, at every $K$, with the graph's $K$ reported alongside so the reader can see the exposure grow. **Not a refusal** — see §3.4 |
| — | `lambda_ref: none` at a seam whose two sides declare different `governing_family` ⟹ $\tau$ has no referent | **`admit-uncertified`**, E3 `fails`, $\tau$ emitted as `UNDEFINED` |
| — | `mapping` unset on any port | **`refuse`** — silent-wrongness class |
| — | `adjointness: none` on a port declaring a `reduction`/`prolongation` pair | **`refuse`** — silently disables the passivity theorem (G3 disables G22) |
| — | `differentiable: jvp` declared with **no JVP callable supplied** | **`refuse`** — a contradiction *inside the declaration*, decidable here by inspection. Left off this table it surfaces three layers later as a probe failure at L4, where it reads as a numerical problem rather than the schema error it is |

**The frozen wind-farm checkpoint declares `bc_channel: none`**, and [[general-coupling-scheme]] §2 already identifies that one field as the formal cause of every coupling limitation recorded against it. Under this spec it has a second consequence, worked in §14.1.

## 3.4 The $K$-threshold rule, retracted — and what would reinstate it

An earlier form of the `validity: none` row demanded a **refusal** for any graph with $K$ at or above the threshold where $P(\text{some agent OOD})\to1$, and named no threshold. **A gate with no value is not a gate**, and a spec that carries an unfireable rule is worse off than one that carries none: the rule reads as coverage, and an implementer discovers only at the keyboard that there is nothing to implement. The row above is therefore a **decertification at every $K$**, with the graph's $K$ emitted alongside.

**Retracting it costs less than it looks.** The refusal was never the protection — the decertification is, and it fires at $K=2$ rather than at some large $K$ nobody has reached. What the threshold would add is a place to *stop*, and stopping needs a number.

**What would reinstate it, stated so the work is findable rather than merely deferred:** the threshold is set by the per-expert declination rate $p$ through $P(\text{some agent OOD})=1-(1-p)^K$, so a measured $p$ gives $K^\star(\epsilon)=\lceil\log(1-\epsilon)/\log(1-p)\rceil$ for any tolerated $\epsilon$. That is not a research programme, it is **W13 plus arithmetic** — a `validity` predicate on one real expert supplies $p$, and the same $p$ is already required by the abstention horizon $T_{\text{abs}}\approx\Delta t/(Kp)$ of [[plug-in-composition-theorems]] §4. **Until $p$ exists, both are undefined, and the honest form of both is a decertification that says which measurement it is waiting on.**

---

# 4. L2 — Decomposition and cut placement

## 4.1 The intuition

Where you cut decides how hard the interface problem is, and the framework has until now chosen cuts on **physical** grounds — put a boundary where the governing equations change character ([[agent-definition-atlas-0.1]]) — with nothing said about whether the resulting seam is well conditioned. Those are different criteria and they can disagree: a reaction front is exactly where the physics changes character *and* exactly where the coupling is strongest, which is the worst place to cut.

## 4.2 The criterion

Both factors of the transmission term are properties of the cut:

$$\sigma \;\le\; \frac{C_\mu}{\beta}\,\bigl\lVert\Lambda-\tilde\Lambda\bigr\rVert\,\bigl\lVert\lambda^\star\bigr\rVert$$

so a good cut is one where the exact DtN is **closest to local** — where $\lVert\Lambda-\tilde\Lambda\rVert$ is small for a low rung — and where $\beta=\sigma_{\min}(\tilde\Lambda)$ is comfortable. **Both come from a probe, so cut quality is measurable before any rollout**, by two candidate assemblies and no dynamics.

> **Attribution, kept exact:** this criterion is recorded in [[generalization-requirements]] G5 as an **[AI Inference]** — *identified, never derived and never tested.* This page adopts it as the decomposition policy **and inherits its status**: it is a plausible criterion with a cheap falsification (W16) and no evidence behind it yet. It is not a theorem and this spec does not make it one.

Practical negative form, same status: **do not cut along a shear layer, a wake centreline, or a reaction front.** Cut across weak coupling — a wall, a free-stream boundary, a docking seam — which is also where [[agent-definition-atlas-0.1]]'s physical heuristic points, so the two criteria usually agree and the interesting cases are where they do not.

## 4.3 Envelope, failures, refusals

| relies on | on failure | verdict |
|---|---|---|
| **E1** fixed graph | the decomposition acquires a time axis: `TopologyEvent` (§12.3). Every event is a **restart with $e^0\neq0$**, so $L^N\lVert e^0\rVert$ reactivates | **`refuse`** if an event is declared with no state map and no conservation ledger |
| **E2** static, coincident interfaces | motion ⟹ `InterfaceMotion` (§12.2). Non-coincidence ⟹ mortar projection with an inf-sup-stable multiplier space — **importable (I3), and there is no construction page anywhere in the vault** | **`refuse`** both today: motion because no rule exists, non-coincidence because the mortar is unbuilt (W11) |
| — | **cross-points**: `non-overlapping` decomposition on any graph richer than a chain creates points where 3+ subdomains meet and the transmission conditions are not independent | **`refuse`** until the cross-point rule (W6, primal corner DOFs per FETI-DP/BDDC) is written. A mishandled cross-point **degrades $\beta$ rather than raising an error** — silent-wrongness |

**The cross-point refusal is the one with immediate teeth.** [[probed-dtn-coupling]] is the framework's best construction and it goes non-overlapping; the current 124-tile overlapping decomposition has four subdomains meeting at a corner in many places and has never hit the problem because a partition of unity does not have it. **Moving to the better transmission operator introduces a difficulty the present scheme does not have**, and this layer is where that trade is made visible instead of discovered.

---

# 5. L3 — The port algebra, and what makes a connection admissible

## 5.1 The intuition

A port is a **power socket with a declared shape**. Two agents can be plugged together when the sockets match — and "match" has until now meant two conditions (same type, coincident geometry). The work of the last several pages has been discovering that it means eight.

## 5.2 The admissibility checklist

A connection between agents $A$ and $B$ for port type $P$ is admissible iff:

| # | Condition | Source | Failure verdict |
|---|---|---|---|
| **C1** | Both declare $P$, from the closed vocabulary $\{$`MECH`, `ROT`, `THERM`, `ELEC`, `ADVEC`$\}$, **with the same passenger list on the two faces** | [[port-algebra-atlas-0.1]] §3.2, §5 | **`refuse`** — a sixth type may enter only through `PortAmendment` (§12.5), and a passenger disagreement *is* a sixth type in disguise |
| **C2** | **Geometric compatibility**: the two interface discretizations coincide, **or** a mortar projection exists with a measured $\beta$ above the probe floor | §5 connection rule; G14 | **`refuse`** today — the mortar is unbuilt. *This is the single most restrictive line in the framework for plug-in use* |
| **C3** | **Mapping class** set on both sides and equal, defaulted per port type: flows conservatively, efforts consistently | G11 / W5 | **`refuse`** — reversing it conserves the wrong integral with no signal |
| **C4** | **Nondimensionalization composes**: the two `nondim` maps compose to a defined conversion at the port | G18 / I8 | **`refuse`** — two experts on different reference scales *will not agree at a port*, and the residual will not say why |
| **C5** | **Power-preserving pairing** declared with an orientation: $e_A\rvert_\Gamma=e_B\rvert_\Gamma$ and $f_A\rvert_\Gamma=-f_B\rvert_\Gamma$ | [[port-algebra-atlas-0.1]] §4 | **`refuse`** — without it there is no Dirac interconnection and §6.1's theorem has no hypothesis |
| **C6** | **Adjointness** for every field↔lumped pair: $\langle e,\mathcal P_\Gamma f_0\rangle_\Gamma=\langle\mathcal R_\Gamma e,f_0\rangle_0$ | G3 / W8 / I5 | **`refuse`** — a non-adjoint pair leaks power at the interface and **silently disables the passivity theorem** |
| **C7** | **Rung compatibility**: $\operatorname{rung}(\tilde\Lambda)\le\min_i\operatorname{rung}(\texttt{bc\_channel}_i)$ (R1), lifted by probing (R2) | [[general-coupling-scheme]] §3 | **`admit-uncertified`** if the achievable rung is below what the case needs — $\sigma$ will be large and the run should say so, not stop |
| **C8** | **Validity overlap**: both experts' `validity` predicates admit the anticipated port state | G8 / [[composition-error-theory]] §4.5 | **`admit`** with an abstention hook — the connection is *legal*; whether it is *valid* is decided per step, at runtime, by the predicate |

> **Amended 2026-08-27 — C2, C3 and C6 are one condition.** [[interface-transfer-theory]] §1–§4 replaces all three with a single check: *a common interface space $M$ is declared for the seam and each side declares a stable, implementable prolongation $P_i:M\to V_i$.* **C2** — geometric coincidence — becomes the special case $P_i=\mathrm{id}$, so *"the single most restrictive line in the framework for plug-in use"* is a restriction of the connection rule rather than of the coupling construction, and the mortar is not a separate thing to build: $\tilde\Lambda_i^M=P_i^\ast\Lambda_iP_i$ is assembled by the probe that was already budgeted, on a space whose dimension is set by the same spectral-cutoff argument. **C3** is derived rather than declared. **C6** is the case $\dim M=1$. The failure verdicts are unchanged and the refusal now cites a **missing declaration** rather than an unbuilt mechanism. **[AI Inference]** on the C2 half, and cheap to falsify — probe two agents at mismatched resolutions and watch $\beta$.
>
> **C8 gains a composition-level semantics**, which it did not have: [[plug-in-composition-theorems]] §4 decides that a declination **halts the rollout** and types the claim on $[0,t_{\text{decline}})$, that a fallback expert is a mid-rollout substitution reconciling as a restart with $e^0\neq0$, and that the declination rate $p$ implies a **second horizon** $T_{\text{abs}}\approx\Delta t/(Kp)$ alongside $T_{\text{pred}}$.

> **C8 is the one that gets worse as the framework gets better.** A clean contract makes more connections **legal** without making them **valid**, and with $K$ experts the probability that at least one is out of distribution approaches 1. The port algebra's own §8 records this; the architecture's answer is *enforce, measure, or decline*, and the third verb needs C8 to have somewhere to live.

## 5.3 Envelope, failures, refusals

| relies on | on failure | verdict |
|---|---|---|
| **E7** power-preserving + adjoint | C5 unmet, or a **measured** non-adjoint pair | **`refuse`** — this is the only envelope hypothesis whose failure is a *refusal* rather than a decertification, because a power leak is invisible in every residual the framework reports. **This sets the verdict, not the stamp**: a refusal for a *missing* declaration leaves E7 `unchecked` (§1.1) |
| **E2** static, transfer accounted for | no common $M$ and prolongation $P_i$ declared for the seam | **`refuse`** — a **missing declaration**, not an unbuilt mechanism, and it leaves the stamp untouched (§1.1). Non-coincident geometry *with* a stable declared $P_i$ is admissible and **decertifies** |

**Global fields are not ports and do not pass through this layer.** Gravity and any uniform external field enter each agent's update directly, bypassing the edge mechanism ([[global-fields-and-topology-atlas-0.1]]) — a third category alongside edges and internal state. They contribute to $P_{\text{ext}}$ in the power residual and to nothing else in this spec.

---

# 6. L4 — Construction of the transmission operator

## 6.1 The intuition

$\tilde\Lambda$ is the architecture's answer to *"what does it mean for two agents to agree?"* — and the whole content of [[master-error-bound]] §3.1 is that **iterating harder converges onto the root of whatever question you asked**, so asking the wrong question is a defect no amount of solving removes. This layer builds the question.

## 6.2 The construction

**The rung is already fixed** — L1 chose it by R1 lifted by R2 (§2.1) and L2's axis follows from it. For `probed-DtN`, this layer assembles the block by probing:

$$S_i\,\psi_k \;=\; \partial_n\Bigl(\mathcal E_i[\psi_k]\;-\;\mathcal E_i[0]\Bigr)\Big\rvert_\Gamma ,\qquad k=1,\dots,m$$

$m+1$ Dirichlet solves per block. The basis dimension $m$ is set by **the expert's own measured spectral cutoff**, not chosen — interface content below the scale at which the expert stops representing flow is not information.

Read in the order the probe uses it, the port algebra **is already the DtN interface**: `MECH` maps the imposed flow $\mathbf v$ to the returned effort $\boldsymbol\sigma\!\cdot\!\mathbf n$, and that map is $\Lambda_i$. The layer adds no new declaration; it reinterprets one.

**Emitted here, all from one assembly:** $\beta=\sigma_{\min}(S\rvert_{\text{constrained}})$, $\kappa(S)$, the null-space dimension, $\lambda_{\min}(\tfrac12(S+S^{\!\top}))$ giving the passivity defect $\pi_i$ and its **eigenvector** (which interface mode is amplified), the per-mode optimal Robin coefficient $\alpha^\star_k=(S_i)_{kk}$, and the composability index $\Xi_i=\lVert\Lambda_i^{\text{expert}}\rVert/\lVert\Lambda_i^{\text{ref}}\rVert$.

## 6.3 Envelope, failures, refusals

| relies on | on failure | verdict |
|---|---|---|
| **E2** static $\Gamma$ | a moving interface changes $\Lambda$ every step and **invalidates the cached $S$, which is the entire economic argument for probing**. `InterfaceMotion` (§12.2) supplies a staleness predicate and a re-probe rate | **`refuse`** today; the slot's declared interface is what a future rule plugs into |
| **E3** single global operator | the *rung choice* survives, but $\tau$ does not — see the asymmetry below | **`admit-uncertified`**, E3 `fails` |
| — | probed null space of the wrong dimension | **`refuse`** — a probed $S$ whose measured null space is not the expected dimension **indicts the probe, not the physics**, and is a free correctness check that must be run before anything is built on the matrix |
| — | $\Xi_i=0$ | **`refuse` the claim, not the run** — see §6.4 |

## 6.4 Two consequences worth stating as rules

**(a) $\Xi=0$ is a type refusal.** An expert with no boundary channel has $\partial\mathcal E_i/\partial\lambda\equiv0$, so $\Lambda_i\equiv0$, so $S\equiv0$, so $\mathcal G(\lambda)=-\chi$ for **every** $\lambda$. The interface problem is not ill-conditioned; it is **empty**, and every trace is equally consistent. A run in that configuration is a legitimate object — an **ensemble of independent local solves** — but it is not a composition, and calling its output a coupled result is the mistake that produced the vault's sharpest failure. So the rule is: *run it, and refuse the word "coupled" on the output.* That is a refusal of a claim type, and it belongs here rather than at L9 because it is decidable from one cheap probe.

**(b) Probing survives E3's failure; $\tau$ does not.** **[AI Inference]:** the probe measures the *expert's own response to imposed boundary data*. It never mentions a governing equation, so it is well-defined across a fluid–structure seam where no single global operator exists — and the optimal Robin coefficient, which classically requires Fourier-analyzing a PDE, is **read off a measured diagonal** instead. The transmission layer is therefore **robust to the failure of E3**, while agent infidelity $\tau$ — defined as the gap to the restriction of a global solution — is not. This asymmetry is not stated on any page and it substantially narrows §12.1: the multiphysics hole is not "the coupling is undefined", it is "**the error attribution** is undefined". That is a smaller hole, and a differently shaped one.

---

# 7. L5 — The interface solve and the accelerator

## 7.1 The intuition

Given the question, solve it — and the only thing this layer controls is $\gamma$, the distance from the returned trace to the root of the question that was asked. **$\gamma$ is the smallest term in the bound and the only one the framework has ever reported.**

## 7.2 The choice, and why it is a bound rather than a preference

| accelerator | $\lVert\lambda^{(k)}-\lambda^\dagger\rVert$ | governed by |
|---|---|---|
| Richardson / classical Schwarz | $\rho^{k}\lVert\lambda^{(0)}-\lambda^\dagger\rVert$ | the Schwarz contraction $\rho$; one-level $\rho\to1$ as subdomains shrink |
| Krylov on the interface | $\sim\bigl((\sqrt\kappa-1)/(\sqrt\kappa+1)\bigr)^{k}$ | $\kappa(\tilde\Lambda)$ — *conditioning*, not subdomain count |
| Direct Schur solve | $\sim\kappa\cdot\epsilon_{\text{mach}}$ | negligible |

Ordering is **`additive` by rule** (R5), because multiplicative ordering breaks [[symmetry-averaging-atlas-0.1]]'s exactness and reinstalls graph-cycle path dependence. A direct Schur solve satisfies R5 automatically, being order-free.

## 7.3 Envelope, failures, refusals

| relies on | on failure | verdict |
|---|---|---|
| **E1**, **E4** | the residual $\mathcal G$ is posed per macro-step on a fixed graph; both failures are handled upstream (L2, L7) | — |
| — | $\varepsilon_{\text{tol}} < \min(\hat\tau,\hat\sigma)$ | **`refuse`** — converging the interface far below the dominant terms spends compute to shrink the negligible one **and removes the only alarm the run has**. Tolerance is set relative to the other terms, never absolutely |
| — | $\mathcal K\in\{$newton-krylov, direct-schur$\}$ with $\tilde\Lambda\equiv0$ (R6) | **`refuse`** — the system is empty, not stiff |
| — | `multiplicative` ordering with any composition-layer guarantee active (R5) | **`refuse`** |
| — | any axis whose value provably cannot affect the answer | **`refuse` at config time, and quote the measurement** — the generalized form of the existing `schwarz > 1` guard, which is the framework's first and so far only instance of a principled refusal |

> **The degenerate-axis rule is the template for this whole page.** An axis that cannot affect the answer must be refused when the configuration is read, not discovered by measuring it. Every refusal above is an instance.

---

# 8. L6 — Assembly

## 8.1 The intuition

The local solves come back overlapping and disagreeing slightly; assembly is the rule that turns them into one field. It is the least examined layer in the framework and it carries **19% of measured total error** — a share that survives perfect agents, perfect windows and a perfect interface solve.

## 8.2 What is specified, and what is not

**Checkable today, at machine precision:**

$$\sum_i R_i^{\!\top}\chi_i R_i \;=\; I$$

A partition of unity that fails this identity is producing a weighted average that is not an average. This is a one-line test, it is currently run nowhere, and its failure is exactly the silent-wrongness class. **`refuse` on failure.**

**Also required, and currently unreported:** $\lVert\mathcal A\rVert$, which enters $L$ multiplicatively:

$$L \;\le\; \lVert\mathcal A\rVert\cdot\max_i\operatorname{Lip}_v(\mathcal E_i)\cdot\Bigl(1+\frac{C_\mu\,C_{\mathcal G}}{\beta}\Bigr)$$

**Not specified anywhere, and this is the hole:** any condition that the blend be **at least as accurate as the local solves it blends**. Partition-of-unity theory supplies such conditions for overlapping approximation; they have not been transcribed, and the measured 19% has never been diagnosed as either a violated condition or an unavoidable cost. That is `AssemblyCertificate`, §12.4.

Single-valued-flux assembly (the non-overlapping counterpart) has the same structure with the PoU identity replaced by flux single-valuedness, and the same hole.

## 8.3 Envelope, failures, refusals

| relies on | on failure | verdict |
|---|---|---|
| **E6**, checkable half | PoU identity fails | **`refuse`** |
| **E6**, open half | no accuracy condition exists to check | **`admit-uncertified`**, E6 `unchecked`, and **emit the certificate fields anyway** (§12.4 field 5) |
| — | $\lVert\mathcal A\rVert$ not computed | **`admit-uncertified`** — $L$ is not computable without it, so no bound can be claimed |

**Equivariance averaging runs after assembly** and before emit, where it is exact for a closed group and does not disturb the PoU identity. It is a composition-layer guarantee, so it is what R5 exists to protect.

---

# 9. L7 — Time integration and multirate

## 9.1 The intuition

Two clocks are the problem. One agent wants a step the other cannot take, and the natural fix — exchange fluxes at whatever instants happen to line up — **is not conservative, and the residual will look fine**.

## 9.2 The rules

| rule | statement | why |
|---|---|---|
| **R3** | $W>1$ requires `bc_time_varying` on every agent at that interface | a ring held constant across the window is a $W=1$ scheme wearing a longer name |
| **R4** | the exchange interval cannot go below $\max_i\texttt{dt\_native}$ | a frozen expert's native step constrains *coupling*, not integration. And the asymmetry: **$W$ can be increased, never decreased — the window is the one axis a frozen expert can move along** |
| **R9** | under multirate, conservation is enforced on the **time-integrated** flux over the macro-step, never pointwise in time | pointwise matching across different clocks is not conservative, and the residual is computed correctly and means nothing |
| **W** | $W^\star\approx\min\bigl(1/\ln \hat L,\ W_{\text{SWR}}\bigr)$, clamped by R3 | choose the window so one block is conditioned at $L^W\le e$ |

$L$ is unmeasured, so $W^\star$ is uncomputable, so **$W=1$ is the honest default and the compiler must say that it is defaulting rather than choosing.**

> **The multirate defect is bounded since 2026-09-03 (CS-11, W90), and the bound is in the exchange interval rather than the clock ratio.** R9 covers the flux transient; the *stale trace* over the interval is $\sigma$ and [[master-error-bound]] §4.2 now gives it as $\sigma \le s_\Gamma\,\dot\lambda_{\max}\Delta t_{\text{ex}} + C_2(\dot\lambda_{\max}\Delta t_{\text{ex}})^2$ — first order, two seam constants from one probe, one rate from the run, holding at $1.43\times$ to $2.18\times$ over $2000\times$ of clock ratio. **Eight times of clock ratio at a fixed interval moves it by $1.00016\times$**, which is what lets a bound checked at ratio $5$ apply at the $10^4$–$10^5$ a real conjugate seam runs at and no referent can be built at. The ratio is quoted as **scope** and not as an argument, and the scope has a live edge: a frozen expert cannot refine its own march, so the independence is measured only for an agent that can. `L7/R9/lag` decertifies with this bound's constants rather than with a comparison to another graph's numbers, and the interval it was measured at is a declared field under §0.4.

## 9.3 Envelope, failures, refusals

| relies on | on failure | verdict |
|---|---|---|
| **E4** single clock or proved multirate | multirate with pointwise-in-time flux matching | **`refuse`** — silent-wrongness, and the rocket case needs multirate, so this refusal fires on the framework's own flagship plan |
| **E4** | multirate with time-integrated matching but no interpolation-order accounting: **interface interpolation order caps the scheme's order**, and stability is set by *coupling stiffness*, not by either agent's step | **`admit-uncertified`** — the mechanism is right, the order accounting is not written |
| **E1** fixed graph | a `TopologyEvent` fires between macro-steps: the graph changes, $e^0$ is reset, and the $L^N\lVert e^0\rVert$ term reactivates | **`refuse`** without a ledger (§12.3) |
| — | $W>1$ without `bc_time_varying` (R3) | **`refuse`** |

**A note on where $\lambda(t)$ lives in time.** The interface trace is represented at the exchange cadence, which is floored at the expert's native step — so there is a component of $\sigma$ **in time**, not in space, that convergence does not remove and a frozen expert cannot refine. Schwarz waveform relaxation (representing $\lambda(t)$ to higher order than the exchange cadence) is the only route, it is importable (I6), and it is recorded in the framework today as a $\Delta t$ constraint rather than as an error term. **This spec records it as an error term**, which is a reclassification, not a construction.

---

# 10. L8 — The emitted diagnostic set

## 10.1 The intuition

The framework's most reliable failure has not been computing the wrong thing. It has been **reporting the term that was easy to compute** — and every quantity below is a by-product of machinery the coupled solve already builds.

## 10.2 The contract

Every run emits, every macro-step where the quantity is per-step and once per run otherwise:

| group | quantities | source |
|---|---|---|
| **bound terms** | $\tau$, $\sigma$, $\gamma$, $L$ | W3's three-way split; W1's fit |
| **interface** | $\beta$, $\kappa$, null-space dimension, $\pi_i$ and its eigenvector, $\Xi_i$, $\alpha^\star_k$ | one probe assembly (L4) |
| **conservation** | per-port residual $r_\Gamma$, global power residual $\mathcal R(t)$ | L3, L6 |
| **assembly** *(new)* | $\lVert\mathcal A\rVert$, the PoU identity residual, the `AssemblyCertificate` fields | L6, §12.4 |
| **envelope** *(new)* | the stamp $(\text{E1}\dots\text{E7})\in\{$holds, fails, unchecked$\}^7$ | every layer |
| **claim** *(new)* | claim type, and the **per-claim horizon table** | L9 |
| **slots** *(new)* | re-probe count and staleness (§12.2); mutation ledger rows (§12.3); $\tau$-`UNDEFINED` seam list (§12.1) | §12 |
| **decisions** *(new)* | the refusal / decertification record, each entry citing its rule | every layer |

$$\mathcal R(t)\;=\;\sum_{i}\frac{dE_i}{dt}\;+\;\sum_{\Gamma}\int_\Gamma (e\,f)_\Gamma\,ds\;-\;\sum_i\mathcal D_i\;-\;P_{\text{ext}}$$

> **The emit line is the contract, not instrumentation.** [[master-error-bound]] §7's practical content is one sentence — *the framework measures the smallest term, controls the second-largest, and has never reported the other two* — and this layer is the whole of the fix.

## 10.3 Envelope, failures, refusals

L8 reports on all seven and relies on none. **A run that cannot produce the envelope stamp is `refuse`d**, because a result whose reader cannot tell whether the bound applies is the artifact this entire specification exists to stop being produced.

---

# 11. L9 — The typing of claims

## 11.1 The decision

**The architecture types claims: trajectory inside $T_{\text{pred}}$, statistical outside. The statistical type is declared and refused, never approximated. A run whose $L$ is unmeasured produces `UNTYPED`.**

Three types, and a claim carries exactly one:

| type | means | status in this architecture |
|---|---|---|
| **`T-CLAIM`** | a trajectory claim, certified by the master bound, valid for $t\le T_{\text{pred}}(\delta_{\text{tol}})$ | **fully specified.** Everything in §3–§10 exists to produce one |
| **`S-CLAIM`** | a statistical claim past the horizon — invariant measure, spectra, structure functions, long-time averages | **type reserved, interface declared, estimator absent.** Emitting one is a **refusal** until §11.4's four fields can be filled |
| **`UNTYPED`** | $\hat L$ unknown, so $T_{\text{pred}}$ unknown, so no type is decidable | **the state of every result currently in the vault** |

## 11.2 Why decide it this way

1. **The alternative is the status quo, and the status quo is the failure mode.** Every rollout metric quoted so far is a trajectory metric and none has been checked against its own horizon. A number computed correctly that means something other than it is read as is precisely the class this vault has been caught by most often.
2. **The trajectory half costs nothing to adopt.** It is one arithmetic step past W1, which is one paired rollout.
3. **The cost of adopting is that the framework looks worse**, because most of the intended domain — turbulence, climate, plasma — becomes `S-CLAIM`, which is refused. **That appearance is accurate.** It is the true state of the theory and hiding it does not change it.
4. **The cost of not deciding is a retrofit.** The emit set, the gate set and the general verification protocol (G10) all need to know whether a claim has a type. Left undecided, they get built for trajectory claims only and the statistical path becomes a rewrite of all three. Reserving the type now, empty, keeps the slot open at the cost of one enum.
5. **What is explicitly not decided:** the error theory for a statistical claim. That is O1 and it stays open. This page decides the **typing discipline**, not the **statistical theory** — and the distinction matters, because a typing discipline is a decision and a statistical theory is research.

## 11.3 The horizon has three branches, and only one of them is written down

$T_{\text{pred}}$ appears in the vault in exactly one form, the chaotic one. Read off [[master-error-bound]] §8's specialization table, it has three:

| regime | horizon | character |
|---|---|---|
| $L>1$ | $T_{\text{pred}}\approx\lambda_1^{-1}\ln\bigl(\delta_{\text{tol}}/(\tau+\sigma+\gamma)\bigr)$ | **exponential and short** — the only branch currently written down |
| $L=1$ (passive) | $N_{\text{pred}}\approx\bigl(\delta_{\text{tol}}-\lVert e^0\rVert\bigr)/\max_n(\tau^n+\sigma^n+\gamma^n)$ | **linear and long** — the bound grows linearly, never exponentially |
| $L<1$ (contractive) | unbounded, provided $\max_n(\tau^n+\sigma^n+\gamma^n)/(1-L)<\delta_{\text{tol}}$ | **no horizon at all**; otherwise the claim is invalid at every $N$, not just late ones |

**This connects E7 to E5, which no page does.** Establishing incremental passivity is not only how $L\le1$ is obtained structurally — it is also **what converts a short exponential horizon into a long linear one**, and therefore what decides whether a target lies inside the trajectory claim class at all. The passivity certificate and the claim type are the same investment.

> **[AI Inference], and it changes the emit contract:** $T_{\text{pred}}$ is **not one number per run**. $\delta_{\text{tol}}$ sits inside the logarithm, and tolerance is a property of the *quantity being claimed* — a wake-deficit profile and an integrated power are not held to the same tolerance. So a run has one $L$, one defect, and **as many horizons as it has quoted quantities**. The emitted horizon must be a table keyed by claim, not a scalar. This follows from the formula and is asserted nowhere.

## 11.4 The interface an `S-CLAIM` must present

Reserved, so that when the theory arrives it plugs in rather than restructures. A statistical claim must name:

1. **The functional** of the invariant measure being claimed — a spectrum, a structure function, a long-time average, a stationary distribution.
2. **The averaging window**, and its ratio to the system's decorrelation time.
3. **An estimator with a sampling error** — separate from, and additional to, the model error.
4. **A gate**: what value of the estimator, at what confidence, constitutes agreement with a reference.

**None of the four has a construction in this vault.** There is no page developing shadowing under hyperbolicity, no ergodic-average error estimate, no gate set for statistical claims, and **no definition of what $\tau$ even means when the target is a measure rather than a trajectory** — that last one being the deepest of the four, because $\tau$ is defined as a defect against a *trajectory* of the exact operator.

## 11.5 Envelope, failures, refusals

| relies on | on failure | verdict |
|---|---|---|
| **E5** | $t>T_{\text{pred}}$ | the claim types `S-CLAIM` and is **`refuse`d for publication**, not for computation. The rollout may run; its trajectory metrics may not be quoted |
| **E5** | $\hat L$ unknown | **`UNTYPED`**, and every metric carries the tag. **`admit-uncertified`** |
| — | a trajectory metric quoted past its own horizon | **`refuse`** — this is the single behaviour L9 exists to prevent |

---

# 12. The five named holes

One slot per envelope-widening gap of [[theory-closure-audit]] §5. **None is solved here.** Each is specified by the five fields of §0.3, and the point of the exercise is that the surrounding layers can be built against the declared interface without pretending the hole is not there.

## 12.1 `SeamReference` — O2, $\tau$ undefined at a multiphysics seam

**What is missing.** E3 assumes a single global $\mathcal S_{\Delta t}$ whose restrictions the agents approximate. Across a fluid–structure or fluid–chemistry seam **no such operator exists**, so *"agent infidelity"* has no referent and the whole of Goal-A's analysis is ill-defined exactly where multiphysics — the framework's selling point — begins.

**Declared interface.**
```
SeamReference(seam Γ, agents (A,B)) -> {
    tau_referent : reference object against which each side's τ is defined
    scope        : interface_only | interface_and_interior
    cost         : compute estimate
    availability : bool          # G19: a reference may not exist at all
}
```

**Default behaviour today.** A seam whose two sides declare different `governing_family` and carry no `SeamReference` emits $\tau=$ `UNDEFINED` for both sides, stamps E3 `fails`, and the run is **`admit-uncertified`**. It is not refused, because the composition still *runs* correctly — it is the error *attribution* that has no meaning.

**What any solution must satisfy.** It must (a) reduce to the existing definition when both sides share a governing family, (b) be measurable without a monolithic solve of the whole system, and (c) supply the symbol that an optimized transmission condition is derived from — or explain why that symbol is not needed.

**The measurement required now.** Per seam: the list of seams where $\tau$ is `UNDEFINED`, and — from §6.4(b) — the **operator-norm surrogate** $\lVert\Lambda^{\text{expert}}_i-\Lambda^{\text{ref}}_i\rVert$ on each side, which needs only a per-side reference *operator*, not a global evolution operator.

> **[AI Inference], and it narrows the hole.** That surrogate is well-defined at a multiphysics seam, because probing never mentions a governing equation. So O2 splits: the **interface** component of agent infidelity is definable today from a one-sided reference operator; the **interior** component is what has no referent. The hole is real and smaller than stated, and this decomposition appears on no page. The likely shape of a full answer — a neighbourhood-local monolithic reference solve on a collar around $\Gamma$ — **is a guess and is recorded as one in §13.**

## 12.2 `InterfaceMotion` — O4, moving and deforming interfaces

**What is missing.** E2 assumes static geometry. Any $\Gamma$ that moves — FSI, free surfaces, combustion fronts, contact — changes the interface operator every step and **invalidates a cached $S$, which is the entire economic argument for probed-DtN coupling**. The phrase *moving interface* appears in exactly two files in the vault, both of them trackers. Zero development.

**Declared interface.**
```
InterfaceMotion(port P) -> {
    motion_class     : static | prescribed | solution_dependent
    Gamma_of_t       : callable | none          # prescribed only
    staleness        : predicate(S_cached, t) -> bool   # when must S be re-probed?
    reprobe_cost     : m+1 solves per affected block
    conservation     : how flux is accounted for on a sweeping interface
}
```

**Default behaviour today.** `motion_class` defaults to `static`. Anything else is **`refuse`d**, at L2 and again at L4 — no rule exists, and a moving interface silently invalidating a cached operator is precisely the silent-wrongness class.

**What any solution must satisfy.** (a) Flux accounting across a sweeping interface must conserve — the interface does work on the region it sweeps, and that term is currently absent from $\mathcal R(t)$. (b) The staleness predicate must be cheaper than the re-probe it gates, or the economics of probing collapse. (c) `solution_dependent` motion couples the geometry into the interface solve, which changes $\mathcal G$ from a linear system in $\lambda$ into a system in $(\lambda,\Gamma)$ — a structural change to L5, not a parameter change.

**The measurement required now.** Any run touching a non-static port emits the re-probe count, the measured drift $\lVert S(t+K\Delta t)-S(t)\rVert$, and the fraction of $\mathcal R(t)$ unaccounted at the moving seam. Even a *static* run should emit the drift, since it is one extra assembly and it prices the whole construction.

## 12.3 `TopologyEvent` — O5, conservation across a topology mutation

**What is missing.** Staging, docking and contact make/break change **which agents exist**. That reconciles with a fixed-graph bound as a **restart with $e^0\neq0$**, so the $L^N\lVert e^0\rVert$ term reactivates at every event — that much is settled. What has **no rule** is conservation *across* the event: what is required of the state map so that the mutation itself neither creates nor destroys the conserved quantities.

**Declared interface.**
```
TopologyEvent(t_e) -> {
    graph_before, graph_after : G⁻, G⁺
    state_map                 : M : u⁻ ↦ u⁺
    edge_reinstantiation      : which ports are created, destroyed, rewired
    ledger                    : per conserved functional, the jump ΔC_k = C_k(u⁺) − C_k(u⁻)
    e0_reset                  : the restart error injected at t_e
}
```

**Default behaviour today.** An event declared without a `state_map` **or without a ledger** is **`refuse`d**. An event declared with both is **`admit-uncertified`**, E1 `fails`, and the ledger is emitted with **no constraint on its values** — because no rule says what they should be.

**What any solution must satisfy.** (a) It must say which functionals are required to have zero jump and which are legitimately discontinuous (a jettisoned stage takes its mass and momentum with it — that is not a leak). (b) It must interact correctly with the pooling hierarchy, whose supernode clustering computed before the event is not valid after it. (c) It must specify what happens to a cached $S$ on every port touching a mutated node — almost certainly full invalidation, which links this slot to §12.2's economics.

**The measurement required now.** The ledger itself, plus the injected $\lVert e^0\rVert$, plus the count of macro-steps since the last event — because the bound's initial term is only meaningful relative to that count.

> **This is the clearest instance of the §0.3 pattern.** There is no rule, so the architecture requires the number the rule will constrain. When someone eventually writes the conservation condition for a mutation, it will arrive to find a ledger already being emitted.

## 12.4 `AssemblyCertificate` — O3, no consistency condition on the assembly

**What is missing.** OP-5 charges **19% of total error** to the assembly rule. The bound carries $\lVert\mathcal A\rVert$ as a factor of $L$ and says nothing else. **There is no stated condition that the blend be at least as accurate as the local solves it blends** — partition-of-unity theory supplies such conditions for overlapping approximation and they have not been transcribed, and the 19% has never been diagnosed as either a violated condition or an unavoidable cost.

**Declared interface.**
```
AssemblyCertificate(A, {χ_i}, {R_i}) -> {
    pou_residual   : ‖ Σ Rᵢᵀ χᵢ Rᵢ − I ‖        # CHECKABLE TODAY, machine precision
    norm_A         : ‖A‖                          # enters L multiplicatively
    blend_defect   : α := ‖A({u_i}) − u*‖ − max_i ‖u_i − u*‖ restricted to the overlap
    condition      : THE OPEN FIELD — the inequality α must satisfy
}
```

**Default behaviour today.** `pou_residual` failing is **`refuse`**. `norm_A` unset is **`admit-uncertified`**, since $L$ is not computable without it. `blend_defect` is **emitted and unconstrained**, and `condition` is empty.

**What any solution must satisfy.** It must be checkable without the exact solution $u^\star$, or the certificate is a diagnostic rather than a condition — which is the difference between a rule the compiler can enforce and a number a human reads afterwards. **[AI Inference]:** the transcribable classical statement is likely a bound of the form $\alpha\le C(\delta,\{\chi_i\})\cdot\max_i\lVert\nabla u_i\rVert$ over the overlap, with $C$ growing as the overlap $\delta$ shrinks — which would make the 19% a **cost of the chosen overlap width** rather than a defect. That is a guess at the shape, not a result, and it is recorded in §13.

**The measurement required now.** All four fields, per run, with $\alpha$ measured against the classical reference on the one instrumented step W3 already calls for. **The 19% figure is a subtraction share; $\alpha$ is a direct measurement of the same thing, and the two disagreeing would itself be informative.**

## 12.5 `PortAmendment` — O6, port vocabulary extension

**What is missing.** Five port types are closed *for the cases considered*. `ADVEC`'s variable passenger list is already flagged as where case-study-specific growth reappears, and radiation, phase change, contact/friction and EM coupling are unexercised. **Completeness is unprovable in the abstract**, so the right target is not a proof but a **documented extension procedure** — and that reframing is itself an open design question, not a result.

**Declared interface.** A sixth port type enters only through an amendment presenting all six fields:
```
PortAmendment(new type P) -> {
    bond        : the conjugate pair (e, f) with e·f a power, in W or W m⁻²
    mapping     : the default mapping class for the flow and for the effort
    transfer    : the reduction/prolongation pair, with an adjointness proof
    dtn_reading : which side is imposed and which is returned, so the probe is defined
    distinctness: an argument that P is NOT a special case of an existing type
    exercise    : at least one interface in one case study that P types and the existing five cannot
}
```

**Default behaviour today.** A case study introducing a port type inline is **`refuse`d**. An amendment missing any of the six fields is **`refuse`d**. `ADVEC`'s passenger list is the **first exercise of the procedure and it is overdue**: a variable passenger list is a vocabulary that grows per case, wearing a single type name.

**What any solution must satisfy.** The procedure must keep the $O(K)$ scaling intact — the whole reason the vocabulary is closed is that per-pair contracts scale as $K(K+1)/2$, which is 10 at $K=4$ and 120–210 at the $K\approx15$–20 the roadmap requires. An amendment that requires a bespoke adapter per expert pair defeats the purpose and must be refused on that ground alone.

**The measurement required now.** The count of interfaces in the current case studies that the five types cannot express **without an `ADVEC` passenger extension**, which is the leading indicator of vocabulary pressure and is countable today.

> **This slot does not close, and that is the correct design.** The target is that adding a sixth type is a **defined operation** rather than a redesign. A procedure cannot prove completeness and this one does not try.

---

# 13. Two compile traces — what the spec does to the cases that exist

## 13.1 The wind farm, as built

| layer | outcome |
|---|---|
| L1 | `bc_channel: none`; `storage: none`; `validity: none`; `mapping` unset per port | 
| L3 | **`refuse`** on C3 (mapping unset). Also **`refuse`** on C6 for the actuator disk, which is a field↔lumped reduction built without the adjointness condition ever being stated |
| L4 | $\Xi=0$. $\Lambda\equiv0$, $S\equiv0$, $\mathcal G(\lambda)=-\chi$ for every $\lambda$. **`refuse` the claim "coupled"**, per §6.4(a) |
| L5 | R6 would refuse `krylov`/`direct-schur` for the same reason; `richardson`, $k=1$ survives, and is a single pass |
| L6 | $\lVert\mathcal A\rVert$ never computed. **`admit-uncertified`**, E6 `unchecked` |
| L9 | $\hat L$ unmeasured ⟹ **`UNTYPED`** |
| **stamp** | E1 `holds`, E2 `holds`, E3 `holds`, E4 `holds`, E5 `unchecked`, E6 `unchecked`, E7 `unchecked` |

**The finding worth the whole exercise:** the wind farm's negative composed result was **a compile-time refusal, not a measurement**. $\Xi=0$ is decidable from a probe costing minutes, and it says the interface problem is *empty* rather than *hard* — every trace equally consistent, because no agent's output depends on any agent's input. Under this spec that is caught at L4 before any rollout, and the honest description of the run is **an ensemble of independent local solves**, which is a legitimate object that simply is not a composition.

This is not a retroactive criticism of the case study — the finding is what the case study was built to produce, and it produced it. It is an argument that **the same finding was available for the cost of one probe**, and that the mechanism which would have surfaced it is a compile-time rule rather than a better experiment.

## 13.2 The rocket ascent, as specified

| layer | outcome |
|---|---|
| L1 | agents declare **different `governing_family`** at the fluid–structure seam $b\!-\!c$ (parabolic conduction plus quasi-static elasticity, against confined reacting compressible flow) |
| L2 | the combustion front and the plume boundary are **not static** ⟹ `InterfaceMotion` ⟹ **`refuse`** |
| L3 | `ROT` and the rigid-body trajectory expert are **lumped**, coupling to field agents ⟹ C6 adjointness certificate required ⟹ **`refuse`** without one |
| L4 | E3 `fails` at $b\!-\!c$ ⟹ $\tau$ `UNDEFINED` ⟹ **`admit-uncertified`**; the *rung choice* survives, per §6.4(b) |
| L7 | differing `dt_native` per agent ⟹ multirate ⟹ R9 ⟹ **`refuse`** unless flux is matched time-integrated |
| L9 | `UNTYPED` |
| later | staging, when the F1 ladder reaches it, trips E1 ⟹ `TopologyEvent` ⟹ **`refuse`** without a ledger. *(The 2D case study is scoped to powered ascent without a stage separation, so this does not fire today.)* |

**The rocket case study does not compile**, and it trips **three of the five named slots** in its declared scope, with the fourth arriving with staging. That is the concrete form of [[theory-closure-audit]] §6's argument: these gaps are invisible while the only built case study is single-physics, fixed-graph, conforming and single-clock — and they are all visible on paper the moment the framework's own flagship plan is run through a compiler that checks the envelope.

**[AI Inference]:** the ordering this implies is that **the rocket should not be the next build.** Every refusal above is a genuine missing rule rather than missing code, and three of them are research. The buildable next step is the one whose refusals are all transcription: fix `mapping` and adjointness, measure $L$, assemble one $\tilde\Lambda$, and re-run the wind farm with a boundary-capable expert — which is exactly the Tier 0 and Tier 1 ordering [[gap-worklist]] already has, now with a structural reason rather than a cost estimate behind it.

---

# 14. Where this spec is guessing

Recorded explicitly, because a specification that does not separate its decisions from its inventions is worse than no specification.

| # | The guess | Status |
|---|---|---|
| 1 | **The three-verdict system** (`admit` / `admit-uncertified` / `refuse`) is invented on this page. The split rule — refuse silent-wrongness, decertify unverified hypotheses — is a judgement about which failures are recoverable, not a derived criterion | **[AI Inference]**, load-bearing everywhere |
| 2 | **The nine-layer decomposition is not proved complete.** G15 routing is already known to be a missing layer; there may be others | **[AI Inference]** |
| 3 | **The cut-placement criterion** (§4.2) is inherited from G5 with its original **[AI Inference]** status intact — identified, never derived, never tested. Adopting it as policy does not upgrade it | inherited speculation |
| 4 | **§6.4(b): probing survives E3's failure while $\tau$ does not.** This follows from the probe never referencing a governing equation, but the consequence — that O2 splits into a definable interface half and an undefined interior half — is asserted here and nowhere else | **[AI Inference]** |
| 5 | **The `SeamReference` shape** — a neighbourhood-local monolithic reference on a collar around $\Gamma$ — is a guess at what a solution looks like, offered so the slot is imaginable, not because it is known to work. It also inherits G19: such a reference may not exist or may cost more than the surrogate | **[AI Inference]**, flagged in §12.1 |
| 6 | **The `AssemblyCertificate` blend-defect $\alpha$** is a definition invented here. Partition-of-unity theory has real conditions; they are untranscribed, and $\alpha$ is a placeholder measurement, not the condition. The conjectured form $\alpha\le C(\delta,\{\chi_i\})\max_i\lVert\nabla u_i\rVert$ is a guess at the shape | **[AI Inference]** |
| 7 | **The `PortAmendment` six-field checklist** is a proposed procedure with no evidence it is sufficient. It has never been exercised, and the first exercise (`ADVEC` passengers) may well show a seventh field is needed | **[AI Inference]** |
| 8 | **The `TopologyEvent` ledger** assumes the eventual conservation rule will be a statement about jumps in conserved functionals. It might instead be a condition on the state map itself, in which case the ledger is the wrong measurement | **[AI Inference]** |
| 9 | **§11.3's three horizon branches** are read off the specialization table by direct arithmetic, so they are derived rather than speculative — but **the claim that the passivity investment and the claim-type investment are the same investment** is an interpretation added here | derived; the interpretation is **[AI Inference]** |
| 10 | **The per-claim horizon table** (§11.3) follows from $\delta_{\text{tol}}$ sitting inside the logarithm. The arithmetic is certain; that the emit contract should therefore carry a table rather than a scalar is a design call | **[AI Inference]** |

**What this page is *not* guessing about:** the master bound and its three-term split, the $\sigma$ estimate, the passivity theorem, the nine admissibility rules R1–R9, the seven-hypothesis envelope, the port algebra's closed vocabulary and power residual, and the classification of which gaps are importable. All of those are transcribed from pages that derive them.

---

# 15. What this changes on the worklist

Per [[gap-worklist]], sorted by what the spec actually does to each item. **A specification closes no measurement**, so nothing moves to `done` here.

## 15.1 Closes

**Nothing outright** — and stating that plainly matters more than finding something to claim. The nearest thing to a closure is **the decision W14 was waiting on**: W14 (*type the claims*) needed a decided rule before it could be implemented, and §11.1 supplies it. W14 remains `open` as an implementation item, but it is no longer blocked on a judgement call.

## 15.2 Reframes

| item | reframed how |
|---|---|
| **W5** (G11 mapping) | from *a field to add* to **a compile-time refusal at L3/C3**. Its definition of done gains "the compiler refuses a connection with `mapping` unset" |
| **W6** (G1 cross-points) | from *a rule to write* to **a refusal condition already in the spec** (L2), with the rule itself still open. The refusal exists before the rule does, which is the correct order |
| **W8** (G3 adjointness) | from *a check to run* to **a connection precondition at L3/C6**. This raises its priority: it is no longer a diagnostic on the actuator disk but a gate every field↔lumped port must pass |
| **W9** (G23 capability record) | its definition of done **grows by five fields** — `governing_family`, `lambda_ref`, `claim_types`, and per-port `mapping`, `motion_class`, `reduction`/`prolongation`/`adjointness` |
| **W10** (G24 compiler) | **this page is its specification.** The nine rules become layer gates, plus the three-verdict system, plus the envelope stamp. Its definition of done gains "and emits the E1–E7 stamp" |
| **W4** (G20 emit) | the emit set **grows from nine quantities to eight groups** (§10.2), the new ones being the assembly certificate, the envelope stamp, the claim type and horizon table, the slot measurements, and the decision record |
| **W14** (G6 typing) | from *a decision plus an implementation* to **an implementation of a decided rule**, with three horizon branches instead of one |
| **W16** (G5 decomposition policy) | promoted from Tier 4 to **the specified policy at L2**, with its **[AI Inference]** status preserved verbatim — adopted, not upgraded |
| **W17** (G13 interface temporal resolution) | reclassified from *a $\Delta t$ constraint* to **a component of $\sigma$ in time**, which is where it belongs in the bound |
| **W22** (G9, G17) | **splits into two slots with separate interfaces** — `TopologyEvent` (§12.3) and `InterfaceMotion` (§12.2). They were bundled because both were untouched; they have different declared interfaces and different owning layers |

## 15.3 Adds

Six new items, all of them the **field-5 measurements** of §0.3 — the numbers the missing rules will constrain.

| # | Gap | Task | Definition of done | Cost |
|---|---|---|---|---|
| **W27** | envelope | **Emit the E1–E7 stamp** from every run, with values `holds` / `fails` / `unchecked` | No rollout artifact without a stamp; E1–E4 decided at compile time | small |
| **W28** | G12 | **`AssemblyCertificate`**: emit the PoU identity residual, $\lVert\mathcal A\rVert$, and the blend defect $\alpha$; refuse on a failed identity | Four fields emitted; the identity check in the coupled step; $\alpha$ compared against OP-5's 19% subtraction share | small, and it rides on W3 |
| **W29** | G2 | **`SeamReference`**: declare `governing_family` per expert; emit the $\tau$-`UNDEFINED` seam list and the per-side $\lVert\Lambda^{\text{expert}}-\Lambda^{\text{ref}}\rVert$ surrogate | Multiphysics seams are identified at compile time and their $\tau$ is marked, not silently computed | small to declare |
| **W30** | G17 | **`InterfaceMotion`**: `motion_class` per port; refuse non-static; emit the operator drift $\lVert S(t+K\Delta t)-S(t)\rVert$ even on static runs | A drift number exists, so the cached-$S$ economics are priced rather than assumed | small — one extra assembly |
| **W31** | G9 | **`TopologyEvent`**: state map plus a **conservation ledger** per conserved functional, and the injected $\lVert e^0\rVert$ | An event cannot be declared without a ledger; ledger values are emitted and unconstrained | small to specify |
| **W32** | G16 | **`PortAmendment`**: write the six-field procedure into [[port-algebra-atlas-0.1]] as a binding section, and exercise it once on `ADVEC`'s passenger list | A sixth port type is a defined operation; a case study cannot add one inline | small |

**All six are small, and that is the argument for them.** Each is the instrumentation of a hole rather than an attempt to fill it — the cheapest class of work in the framework, and the class the vault's own history says is most often skipped.

---

# 16. What this page changes

1. **The framework has a specification of the whole system**, not a set of correct pages about its parts. That was the missing artifact, and it is a prerequisite for any plug-in claim.
2. **The envelope becomes a stamp.** Seven hypotheses, decided per run, four of them at compile time before compute is spent.
3. **Refusal becomes the general mechanism it was already an instance of.** The `schwarz > 1` guard is one of roughly fifteen refusals specified here, and the three-verdict system separates *do not run this* from *run it, and claim nothing*.
4. **Five holes are named, interfaced, and instrumented** rather than assumed away. Each emits the number its missing rule will constrain.
5. **Claims are typed, and the decision is recorded**: trajectory inside $T_{\text{pred}}$, statistical outside, statistical refused rather than approximated, and everything currently in the vault `UNTYPED`.
6. **Two case studies were run through it on paper.** One would have been refused at compile time for the finding it was built to measure; the other does not compile, and trips three of the five named slots. **That is what an end-to-end spec is for**, and it cost a page rather than a rewrite.

---

## See Also

- [[theory-closure-audit]] — §3's seven-hypothesis envelope is this page's spine; §5's six open gaps are §12's five slots plus §11's decision
- [[master-error-bound]] — the theorem every layer is arranged to keep valid; §8's specialization table is §11.3's three horizon branches
- [[general-coupling-scheme]] — §2's capability record is L1, the nine rules R1–R9 are distributed across L3–L7, §4's compiler is what this page specifies
- [[port-algebra-atlas-0.1]] — L3's vocabulary and the power residual; §12.5's amendment procedure belongs on that page as a binding section
- [[probed-dtn-coupling]] — L4's construction, and the source of $\beta$, $\kappa$, $\pi$, $\Xi$ and $\alpha^\star$; §12.2 is the slot that prices its cached $S$
- [[generalization-requirements]] — the G-numbered gaps every refusal cites; §E's seven conditions for "generalized" are what this spec is the shape of
- [[gap-worklist]] — §15's closes / reframes / adds, and where W27–W32 land
- [[temporal-error-accumulation]] — L7's windowing and the $W^\star$ rule; §3.7's frequency-sweep reading of the probe
- [[composition-error-theory]] — C8's abstention precondition, which none of the constructions removes
- [[conservation-as-constraint-atlas-0.1]] — *enforce, measure, or decline*, extended here by a fourth state: **refuse at compile time**
- [[expert-library-atlas-0.1]] — where the L1 record lives, and the two axes (composability index, probe class) invisible to accuracy benchmarks
- [[agent-definition-atlas-0.1]] — L2's physical granularity heuristic, which §4.2's conditioning criterion sits beside rather than replaces
- [[global-fields-and-topology-atlas-0.1]] — global fields bypass L3 entirely; its staging requirement is §12.3
- [[schwarz-iteration-atlas-0.1]] — the degenerate-axis refusal L5 generalizes
- [[symmetry-averaging-atlas-0.1]] — the composition-layer guarantee R5 exists to protect, applied at L6
- [[interface-transfer-theory]] — **supplies the rules this page built gates for.** L3's conditions C3 and C6 collapse into one declared object, and §0.1's "nine layers" gains the rule content L3–L7 were refusing on the absence of
- [[plug-in-composition-theorems]] — **three properties this page assumed and did not state**: closure under composition, substitution, conformance. Two findings bear directly here — §10.2's emit contract needs a **depth field**, because a subassembly's $\sigma$ and $\gamma$ become its $\tau$ one level up; and §12.3's `TopologyEvent` turns out to be the mechanism that dynamic routing and abstention fallback also route through. It also fills the G15 layer §0.1 flags as missing, with a criterion
- [[f1-pathmap-and-end-goal]] — the ladder whose later rungs are exactly the five named slots
