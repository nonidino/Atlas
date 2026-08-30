# Interface Transfer Theory — one declared object, and the nine importable mechanisms collapse into four

**Type:** Concept page — **framework theory, binding**, version-independent (folder: `Atlas 0.1/common/`)
**Status:** written 2026-08-27, in answer to *"is all of the theory decided?"* asked of [[end-to-end-architecture-spec]]. It was not. That page built **gates** for the nine importable mechanisms of [[theory-closure-audit]] §4 — *refuse when `mapping` is unset* — but the **rule that sets them exists on no page**, and a compiler that refuses an unset field is unbuildable until something says what to set it to. This page writes the rules.
**Related:** [[end-to-end-architecture-spec]] · [[theory-closure-audit]] · [[probed-dtn-coupling]] · [[port-algebra-atlas-0.1]] · [[master-error-bound]] · [[general-coupling-scheme]] · [[generalization-requirements]] · [[plug-in-composition-theorems]] · [[gap-worklist]] · [[temporal-error-accumulation]] · [[conservation-as-constraint-atlas-0.1]] · [[schwarz-iteration-atlas-0.1]]

> **The one-line version.** The nine importable gaps were filed as nine separate transcription jobs. They are not nine. **Six of them are one construction** — a common interface space $M$ per seam plus one declared prolongation $P_i: M\to\partial\Omega_i$ per side, whose reduction is forced to be the adjoint $P_i^\ast$ — and everything else follows: conservative-vs-consistent mapping is *derived* rather than declared (I1), non-conforming geometry is *removed as a requirement* (I3, and it is the framework's single most restrictive line), field↔lumped adjointness is the special case $\dim M=1$ (I5), multirate is the same rule on the time axis (I4), waveform relaxation is its consistent counterpart (I6), and nondimensionalization is a diagonal factor inside $P_i$ constrained to preserve power (I8). **The remaining three — cross-points, runtime estimation, cut placement — are genuinely separate, and are written down in §7–§9.**

---

# 0. What this page is for, and the mistake it corrects

[[theory-closure-audit]] §4 lists nine gaps as **IMPORTABLE**: solved in the classical domain-decomposition and co-simulation literature, absent here, *"no research required — only adoption and a decision."* That classification is correct. The estimate attached to it — *"a week of writing rules down"* — treated them as nine independent transcriptions.

**They are not independent, and noticing that is the content of this page.** Written out, six of the nine are the same object seen from six directions, and the object is not exotic: it is the pairing that makes a port a *power bond* in the first place ([[port-algebra-atlas-0.1]] §2), applied to the transfer operator instead of to the values.

The consequence matters more than the tidiness. **A single declared object means a single conformance test, a single failure mode, and a single compile-time refusal** — instead of six fields that can each be set wrong independently and four of which fail silently.

## 0.1 Status of each claim on this page

| § | Mechanism | Gap | Status |
|---|---|---|---|
| §2–§3 | Common interface space + adjoint transfer pair | I1/G11, I3/G14, I5/G3 | **classical, transcribed.** The unification is this page's |
| §4 | Non-conforming coupling under probing | I3/G14 | classical mortar; **its interaction with probing is [AI Inference]** |
| §5 | Time-axis transfer: conservative and consistent | I4/G4, I6/G13 | classical; **the tensor-product framing is [AI Inference]** |
| §6 | Nondimensionalization as a power-preserving factor | I8/G18 | mechanical; the power constraint is this page's |
| §7 | Cross-points | I2/G1 | classical (FETI-DP/BDDC), transcribed, **with a decision recorded** |
| §8 | Runtime a posteriori estimation | I7/G7 | classical (DWR); **its inheritance of O2 is [AI Inference]** |
| §9 | Cut placement | I9/G5 | **[AI Inference] throughout**, inherited from G5 and not upgraded |

---

# 1. The object

Fix a seam $\Gamma$ joining agents $A$ and $B$ through one port type. Three spaces, not two:

- $V_A$ — the trace space $A$ can actually accept and return on its own discretization,
- $V_B$ — the same for $B$,
- $M$ — a **common interface space**, chosen once per seam, in which the coupling is expressed.

Classically $M$ is the *multiplier space* and one insists $V_A=V_B=M$; that insistence is exactly the connection rule of [[port-algebra-atlas-0.1]] §5 — *"their interface geometries coincide"* — which [[generalization-requirements]] G14 identifies as **the single most restrictive line in the framework for plug-in use.** Dropping the insistence, and keeping only a declared map into each side, is the whole move.

**The declaration.** Each side declares one operator, the **prolongation**

$$P_i \;:\; M \longrightarrow V_i, \qquad i\in\{A,B\},$$

which takes an interface datum expressed in the common space and presents it to agent $i$ in the form agent $i$ accepts. Nothing else about the transfer is declared.

**The reduction is not declared — it is forced.** The port carries a power bond $P=\int_\Gamma e\,f\,ds$ ([[port-algebra-atlas-0.1]] §2). Requiring that the transfer neither creates nor destroys interface power means the effort produced by agent $i$, when reduced back to $M$, must pair with any $\mu\in M$ exactly as it paired inside $V_i$:

$$\bigl\langle R_i\,e_i,\ \mu \bigr\rangle_M \;=\; \bigl\langle e_i,\ P_i\,\mu \bigr\rangle_{V_i} \qquad \forall\,\mu\in M,\ \forall\,e_i .$$

That is the definition of an adjoint, so

$$\boxed{\;R_i \;=\; P_i^{\ast}\;}$$

with the adjoint taken with respect to the interface $L^2$ pairings that make $e\cdot f$ a power. **One declared operator per side per seam. The other is its adjoint, and any other choice leaks power at the interface.**

---

# 2. Three gaps, one condition

## 2.1 I5 / G3 — field↔lumped adjointness is this condition at $\dim M=1$

[[theory-closure-audit]] I5 states the field↔lumped requirement as $\langle e,\mathcal P_\Gamma f_0\rangle_\Gamma=\langle\mathcal R_\Gamma e,f_0\rangle_0$ and records that it is *"never stated in the port algebra; never checked on the actuator disk."*

That is $R=P^\ast$ with $M=\operatorname{span}\{1\}$. A lumped port is a **one-dimensional common interface space**: the single scalar the lumped side exchanges. The prolongation spreads that scalar over the field side's seam; the reduction is forced to be the corresponding weighted integral, and no other reduction preserves the bond.

> **The actuator disk was not missing a check. It was missing a declaration** — nobody declared $P$, so nobody could say what $R$ had to be, and an $R$ was written by hand. That is the mechanism by which [[generalization-requirements]] G3 *"silently disables G22's passivity theorem"*, stated constructively rather than as a warning.

## 2.2 I1 / G11 — conservative vs. consistent is derived, not chosen

[[port-algebra-atlas-0.1]] §9.3 says an `ADVEC` flow must map **conservatively** (integral preserved) and a `THERM` effort **consistently** (pointwise values preserved), and that Atlas *"is currently making the choice implicitly, per edge, by accident."* [[theory-closure-audit]] I1 asks for a `mapping` field defaulted per port type.

**No field is needed.** The two mapping classes are the two halves of one adjoint pair:

- A **consistent** map is an interpolation: it reproduces constants, i.e. its rows sum to one. Efforts (intensities — temperature, traction, pressure) map this way.
- A **conservative** map preserves integrals: it is the *transpose* of a consistent map in the opposite direction, up to the mass matrices of the two sides. Flows (fluxes — heat rate, mass flux, momentum flux) map this way.

Since the effort map is $P_i$ and the flow map is $P_i^\ast$, and the adjoint of a constant-reproducing interpolation is an integral-preserving projection, the assignment is automatic:

$$\text{effort} \;\xrightarrow{\ P_i\ }\;\text{consistent},\qquad \text{flow} \;\xrightarrow{\ P_i^{\ast}\ }\;\text{conservative}.$$

**Getting it backwards is now not expressible.** The classic partitioned-coupling bug that [[port-algebra-atlas-0.1]] §9.3 names requires two independent declarations to disagree; with one declaration there is nothing to disagree with.

> **This is the strongest single result on this page.** G11 and G3 were filed as two gaps in different tiers of [[generalization-requirements]] — one a silent correctness bug, one a plug-in blocker — and they are one condition. The reason neither page saw it is that G11 was stated in the language of *quantities* and G3 in the language of *operators*, and the connecting fact is that the mapping classes **are** an adjoint pair.

## 2.3 What is still declared, and can still be wrong

$P_i$ itself. A wrong $P_i$ — the wrong interpolation order, the wrong support, the wrong quadrature — is a real error and it is **not** caught by adjointness, because $P_i^\ast$ is wrong consistently with it. Adjointness buys **no power leak**, not accuracy. The accuracy of $P_i$ enters $\sigma$ through §4.2 and is measured there.

---

# 3. The composite interface operator on $M$

With the transfer settled, the probed construction of [[probed-dtn-coupling]] §2 restates on the common space. Probing side $i$ with $\mu_k\in M$ means imposing $P_i\mu_k$ and reducing the returned flux:

$$\bigl(\tilde\Lambda_i^{M}\bigr)_{lk} \;=\; \bigl\langle \mu_l,\ P_i^{\ast}\,\partial_n\bigl(\mathcal E_i[P_i\mu_k]-\mathcal E_i[0]\bigr)\bigr\rangle_M , \qquad\text{i.e.}\qquad \tilde\Lambda_i^{M} \;=\; P_i^{\ast}\,\Lambda_i\,P_i .$$

**Three properties, and each one discharges something.**

1. **It is a Galerkin projection.** $P_i^\ast\Lambda_i P_i$ is the compression of the true DtN onto the common space. Everything the classical theory says about Galerkin compressions applies without modification.
2. **Positive-realness survives.** If $\tfrac12(\Lambda_i+\Lambda_i^{\!\top})\succeq0$ then $\tfrac12\bigl(P_i^\ast\Lambda_i P_i+(P_i^\ast\Lambda_i P_i)^{\!\top}\bigr)=P_i^\ast\,\tfrac12(\Lambda_i+\Lambda_i^{\!\top})\,P_i\succeq0$ — a congruence. So **the passivity certificate of [[probed-dtn-coupling]] §4.2 is unaffected by non-conforming transfer, provided the pair is adjoint.** If the pair is *not* adjoint the expression is $R_i\Lambda_iP_i$ with $R_i\neq P_i^\ast$, the congruence structure is destroyed, and positive-realness is no longer inherited. **That is the precise mechanism of "G3 silently disables G22", and it is one line of linear algebra.**
3. **The seam operator is the sum, as before.** $\tilde\Lambda^M=\sum_i P_i^\ast\Lambda_i P_i$, so $\beta=\sigma_{\min}(\tilde\Lambda^M\vert_{\text{constrained}})$ and $\kappa$ are read off exactly as in [[probed-dtn-coupling]] §4.3 — on $M$ rather than on a shared grid.

> **[AI Inference], and it is the reason this page exists.** Points 1–3 say that the probed formulation *already* accommodates non-conforming interfaces, and that nobody noticed because $M$ was never named as a space distinct from the two sides' traces. The probe never touches either agent's discretization — it hands over a trace and takes back a flux — so the requirement that the two discretizations coincide was never a requirement of the *construction*, only of the *connection rule written before the construction existed*. This is unverified and it is cheap to falsify: probe two agents at deliberately mismatched resolutions, assemble $\tilde\Lambda^M$, and check that $\beta$ degrades only through §4.2's approximation term and not catastrophically.

---

# 4. I3 / G14 — non-conforming interfaces

## 4.1 The requirement that replaces geometric coincidence

[[port-algebra-atlas-0.1]] §5's connection rule reads: *two agents may share an edge for port $P$ iff both declare $P$ and their interface geometries coincide.* Replace it with:

> **Connection rule, revised.** Two agents may share an edge for port type $P$ iff both declare $P$, a common interface space $M$ is declared for the seam, and each side declares a prolongation $P_i:M\to V_i$ that is (a) **stable** — $\lVert P_i\rVert$ bounded — and (b) **implementable**, meaning the agent can accept $P_i\mu$ as boundary data and return a flux.

Geometric coincidence is the special case $M=V_A=V_B$, $P_i=\mathrm{id}$. **Nothing that works today stops working, and the restriction that blocked independently built experts is gone.**

## 4.2 Choosing $M$, and what the choice costs

The classical constraint is an **inf-sup (LBB) condition**: $M$ must not be richer than the trace spaces can test, or the coupled system is singular. Concretely, $M$ must be **no finer than the coarser of the two sides**; the standard safe choice is the trace space of the coarser side, with the end/corner degrees of freedom constrained (see §7).

The cost of a coarse $M$ is a **consistency term in $\sigma$**. The exact trace $\lambda^\star$ is represented only up to its best approximation in $M$, so alongside the operator-approximation term of [[master-error-bound]] §4 there is

$$\sigma_{\text{nc}} \;\le\; \frac{C_\mu}{\beta}\,\bigl\lVert (I-\Pi_M)\,\lambda^\star \bigr\rVert ,$$

with $\Pi_M$ the $M$-projection. This is a genuinely new error term for non-conforming coupling and it belongs in $\sigma$, not in $\tau$ — the agents are innocent of it.

## 4.3 The dimension of $M$ has the same selection rule as the probe basis

[[probed-dtn-coupling]] §2.2 sets the probe basis dimension from the **expert's own spectral cutoff** — content below the scale where the expert stops representing flow is not worth probing, giving $m\approx16$ modes per component for the wind farm. The same argument sets $\dim M$: representing the trace more finely than the *coarser expert can respond to* buys nothing, because $\Lambda_i$ returns noise on those modes.

$$\boxed{\;\dim M \;=\; \min_i\ m_i^{\text{eff}}\;}$$

where $m_i^{\text{eff}}$ is agent $i$'s effective interface resolution. **The multiplier space and the probe basis are the same space**, which is why the mortar construction costs no probes beyond the ones already budgeted.

**[AI Inference]:** this also gives the inf-sup condition an operational form the classical statement lacks here. Rather than proving an LBB constant for a chosen pair of spaces — which requires approximation theory a frozen expert does not have — one *measures* $\beta=\sigma_{\min}(\tilde\Lambda^M)$ on the assembled matrix, exactly as [[probed-dtn-coupling]] §4.3 already proposes. **An inf-sup condition that is measured instead of proved is the only version available to a black box, and it is available for free.**

---

# 5. The time axis — I4 / G4 and I6 / G13 are the same pair again

## 5.1 The framing

Everything above concerns transfer **across** $\Gamma$ at one instant. Under multirate the same question arises **along** $t$: agent $A$ takes $r_A$ substeps per macro-step, $B$ takes $r_B$, and the interface datum must be transferred between two temporal resolutions.

**[AI Inference]:** the transfer is a tensor product of a space rule and a time rule, and **each axis has a conservative member and a consistent member**:

| | conservative (integral preserved) | consistent (pointwise preserved) |
|---|---|---|
| **space** ($\Gamma$) | flow map $P_i^{\ast}$ — I1 | effort map $P_i$ — I1 |
| **time** ($\Delta t$) | time-integrated flux matching — **I4 / G4** | higher-order $\lambda(t)$ representation — **I6 / G13** |

This framing is asserted here and appears on no other page. Its value is that it says I4 and I6 are not two unrelated deferrals — they are **the two halves of the time-axis transfer**, and declaring one without the other is the temporal form of the G11/G3 bug.

## 5.2 I4 / G4 — the conservative half, written as a rule

Rule R9 of [[general-coupling-scheme]] §3 exists as prose. As a rule:

> **R9, stated.** Over a macro-step $[t^n,t^{n+1}]$, the coupling condition on a flow variable is
> $$\int_{t^n}^{t^{n+1}}\!\!\int_\Gamma f_A\,ds\,dt \;+\; \int_{t^n}^{t^{n+1}}\!\!\int_\Gamma f_B\,ds\,dt \;=\;0 ,$$
> evaluated with each side's **own** substep quadrature. Pointwise-in-time matching $f_A(t)=-f_B(t)$ is **refused** whenever $r_A\neq r_B$.

Three consequences worth stating, because each is a thing the vault would otherwise discover late:

1. **The residual of the wrong condition is well-behaved.** Pointwise matching under multirate produces a residual that converges and means nothing — the same silent-wrongness signature as [[schwarz-iteration-atlas-0.1]] §3.1's bitwise-zero residual. This is why R9 must be a **refusal**, not a warning.
2. **Interface interpolation order caps the scheme order.** If $\lambda$ is held constant across the macro-step the coupling is first-order in time regardless of how accurate either agent is internally. **A second-order expert coupled with a zeroth-order interface representation is a first-order method**, and this is the exact temporal analogue of §4.2's $\sigma_{\text{nc}}$.
3. **Stability is set by the coupling stiffness, not by either agent's step.** The macro-step admissible under coupling can be shorter than $\min_i \texttt{dt\_native}$ — which is the *opposite* direction from R4's constraint and is not implied by it. **[AI Inference]:** the relevant stiffness is $\kappa(\tilde\Lambda)$, already printed by the probe, so the multirate stability limit is estimable from the same matrix. Untested.

## 5.3 I6 / G13 — the consistent half

Represent $\lambda(t)$ across the macro-step to higher order than the exchange cadence: this is **Schwarz waveform relaxation**, and it is the only route that reduces the temporal component of $\sigma$ without refining the exchange.

[[generalization-requirements]] G13 records the situation as *"a $\Delta t$ constraint"* and notes its status as an error term is not recorded. Recorded here:

$$\sigma_{\text{time}} \;\le\; \frac{C_\mu}{\beta}\,\bigl\lVert (I-\Pi_W)\,\lambda^\star(\cdot)\bigr\rVert_{[t^n,t^{n+1}]},$$

with $\Pi_W$ the projection onto the temporal representation used across the window. **$\sigma_{\text{time}}$ is not reduced by iterating**, exactly as its spatial counterpart is not — the property that makes $\sigma$ the term worth splitting off ([[master-error-bound]] §3.1) holds on the time axis for the same reason.

This connects to [[temporal-error-accumulation]] §3: a window $W>1$ is what makes a higher-order $\lambda(t)$ representation *possible* — with $W=1$ there is no interval to represent anything over. **So R3's `bc_time_varying` requirement is not just an admissibility rule; it is the precondition for reducing $\sigma_{\text{time}}$ at all**, and an expert declaring `bc_time_varying: false` has a floor on $\sigma$ that no scheme removes.

---

# 6. I8 / G18 — nondimensionalization is a factor of $P_i$, constrained to preserve power

[[port-algebra-atlas-0.1]] §8 lists nondimensionalization mismatch as real, unavoidable and $O(K)$. [[theory-closure-audit]] I8 calls it *"mechanical: compose the two `nondim` maps at the port."* Mechanical, but with one constraint that is not automatic.

Each expert declares reference scales; let $D_i$ be the diagonal operator converting agent $i$'s nondimensional interface variables to SI. Then the prolongation factors as

$$P_i \;=\; D_i^{-1}\,\hat P_i ,$$

with $\hat P_i$ the purely geometric transfer of §1 and $D_i$ the unit conversion. **The constraint:** the port is a power bond, so the effort scale and the flow scale must multiply to the power scale,

$$\boxed{\;s_e^{(i)}\cdot s_f^{(i)} \;=\; s_P^{(i)}\;}$$

for every port type the expert declares. A scale set that does not satisfy this for some declared port is **incomplete or inconsistent**, and either is a compile-time refusal.

**Why this is worth a box rather than a footnote.** Converting effort and flow with independently chosen scales is possible, dimensionally plausible, and silently destroys the power bond — after which $\mathcal R(t)$ of [[port-algebra-atlas-0.1]] §6 measures a unit error rather than a physics error, and reports it in watts. The check costs a multiplication per port and is decidable at compile time from the declaration alone.

**Completeness is decidable per port type.** `MECH` needs a stress scale and a velocity scale; `THERM` a temperature scale and a heat-flux scale; `ADVEC` one pair per passenger. So the compiler can determine, from the port list alone, exactly which scales the record must contain — and refuse a record missing one, before any solve.

---

# 7. I2 / G1 — cross-points, and the decision

At a vertex where three or more subdomains meet, the transmission conditions of the incident seams are **not independent**: each pair imposes continuity, and around a cross-point those constraints are linearly dependent. The classical fix (FETI-DP, BDDC) is to make the cross-point degrees of freedom **primal** — single-valued, solved in a small global coarse problem — and keep the rest of the seam **dual**.

[[theory-closure-audit]] I2 records that cross-points are *"named in 6 files, decided in none"*, and that [[probed-dtn-coupling]] **introduces** the problem by going non-overlapping.

> **The decision, recorded.** The treatment is conditional on the decomposition axis $\mathcal D$ of [[general-coupling-scheme]] §1:
> - $\mathcal D=$ **overlapping** — the partition of unity handles the vertex; **no action**. This is why 124 tiles meeting four-at-a-corner have caused no trouble.
> - $\mathcal D=$ **non-overlapping** (which probed-DtN requires) — cross-point DOFs are **primal**: removed from $M$, made single-valued, and appended to the interface system as constraints. **A non-overlapping decomposition with a cross-point and no primal set is refused at L2.**

**The detection is free and already budgeted.** [[probed-dtn-coupling]] §2.1 makes the probed null space a **free correctness check** — *"a probed $S$ whose measured null space is not the dimension that seam predicts is reporting a defect in the probe, not in the physics."* An untreated cross-point is exactly such a defect: the dependent constraints show up as **extra null directions**, one per unhandled vertex beyond the expected count.

**The expected count is a property of the assembled seam, not of a port type.** Write $n_0(\Gamma)$ for the null-space dimension expected at seam $\Gamma$ — a function of *which agents meet there*, declared on the seam rather than defaulted from the port type:

$$\dim\ker\bigl(\tilde\Lambda^M_\Gamma\bigr) \;=\; \underbrace{n_0(\Gamma)}_{\text{per seam, declared}} \;+\; \underbrace{\#\{\text{untreated cross-points on }\Gamma\}}_{\text{should be }0}$$

> **Measured 2026-08-27, and the machinery above passed its first real use while the constant failed.** Four fluid–fluid `MECH` seams, declared $n_0=1$, probed against `reference.WindowNS`: **measured $\dim\ker=0$ on all four**, in every control, with no spectral gap at the bottom ($\sigma_{14,15,16}=1.05\times10^{-2},\,6.47\times10^{-3},\,5.29\times10^{-3}$) and the constant mode nowhere near null ($\lVert Se_{\text{const}}\rVert=0.0215$ against $\sigma_{\max}=0.120$). That is the **deficit** row of the table below, whose verdict is `admit-uncertified` and whose two readings are *"the probe is not resolving a constraint the physics has"* or *"$n_0(\Gamma)$ is wrong"* — and the second is the case. **$n_0=1$ presumes incompressibility is inside $\Lambda_i$, and [[probed-dtn-coupling]] §2.1 puts the elliptic part *outside* it.** The two statements are incompatible, and this is the measurement that made them collide. **The check itself is vindicated**: a free correctness test caught a real error on its first use, which is what it was built to do — it simply caught one in the theory rather than in the probe. See [[tier0-measurements]] §2.1.
>
> **Resolved the same day, and this section survives with one clause added.** $n_0(\Gamma)$ is a function of *which agents meet there* — and also of **where the elliptic solve lives**. Incompressibility constrains the trace only if the incompressibility constraint is inside $\Lambda_i$. Under [[probed-dtn-coupling]] §2.1's own prescription it is not, so:
>
> $$n_0(\Gamma)\;=\;1\ \text{if the elliptic solve is inside }\Lambda_i,\qquad 0\ \text{if it is in the composition layer}$$
>
> The declaring field is `capability.elliptic_subsolve`. **This is a third input to $n_0$ alongside the seam's agents and the vertex valence**, and like them it is declared rather than derived — a compiler that inferred it would be special-casing a physics it is built never to name. The verdict table below is unchanged and it did its job: the deficit row's two readings were *"the probe is not resolving a constraint"* and *"$n_0(\Gamma)$ is wrong"*, and the second was right for a reason the row could not have anticipated.

The value $n_0=1$ is **incompressibility at a fluid–fluid seam**, and it is the only case for which the constant has ever been written down. At a **field↔lumped** seam $n_0=0$: the lumped side responds in the direction incompressibility leaves open, since an actuator disk returns a thrust for a uniform imposed inflow, so $\tilde\Lambda^M_\Gamma$ there is nonsingular. **Carrying the $1$ as a per-port-type constant produces a false defect report on every rotor face in the wind farm** — which is how this was found, and it is the reason the count belongs to the seam rather than to the port type.

**Both directions of a mismatch get a verdict.** The deficit case needs one as much as the excess, and neither may pass silently:

| measured against $n_0(\Gamma)$ | what it indicts | verdict |
|---|---|---|
| **excess** | untreated cross-points, or a constraint $n_0$ did not anticipate | **`refuse`** — the interface system is singular in a direction nothing handles, and a solve on it is not defined |
| **deficit** | *either* the probe is not resolving a constraint the physics has, *or* $n_0(\Gamma)$ is wrong | **`admit-uncertified`** — which of the two is not decidable from the number alone, and both are worth a reader's attention |
| **equal** | — | no action; the check is free and it passed |

**$n_0$ has to be declared, and the reason is worth being explicit about.** It is not derivable from anything the compiler knows: $n_0=1$ at a fluid–fluid `MECH` seam comes from *incompressibility*, a property of the governing family, and a compiler that inferred it would be special-casing a physics it is built never to name. So $n_0(\Gamma)$ joins the small set of things the seam declares and nothing verifies — the same standing as the mapping class before §2 derived it, and the same standing [[plug-in-composition-theorems]] §3 gives `validity`.

**[AI Inference]:** $n_0(\Gamma)$ also depends on vertex valence, and no page derives it in general — so the **equality** above is a conjecture and is deliberately not the thing enforced. What is enforced is the asymmetric pair of verdicts, and the asymmetry matters precisely because $n_0$ is declared: an $n_0$ declared **too small** costs a refusal that inspection resolves, while an $n_0$ declared **too large masks a cross-point** and passes silently. That is the one direction in which this check can be defeated by an optimistic declaration, it is not detectable from the number alone, and it is the argument for keeping $n_0$ in the conformance suite's falsifiable set rather than treating it as settled. **G1 turns from an uncovered hazard into a line in the probe's existing correctness check** — the second time on this page that a gap dissolves into instrumentation that was already going to be built.

---

# 8. I7 / G7 — runtime a posteriori estimation, and the half of it that survives O2

Every bound in [[master-error-bound]] is *a priori*: evaluating it needs a reference. The classical answer is **dual-weighted residual (DWR)**: for a goal functional $J$, solve the adjoint problem for $z$ and localize

$$J(u)-J(u_h)\;\approx\;\bigl\langle z,\ \mathcal{R}es(u_h)\bigr\rangle \;=\; \sum_i \eta_i \;+\; \sum_\Gamma \eta_\Gamma ,$$

giving a **per-agent** and **per-seam** error contribution at runtime, with no reference solution. This is the fourth identified use of the differentiability claim, and [[generalization-requirements]] G12′ records that it has been claimed and never spent.

**The honest qualification, and it is new.** DWR needs a **residual**, and a residual needs a governing equation. At a multiphysics seam there is none — that is O2 — so:

> **[AI Inference]: DWR inherits O2, and it inherits it with the same split.** The **interface** contribution $\eta_\Gamma$ depends only on the *port* residual — the jump in the power bond across $\Gamma$, which is defined without reference to either side's governing equations. The **interior** contribution $\eta_i$ requires agent $i$'s residual and is undefined where no governing family is declared. This is the identical decomposition [[end-to-end-architecture-spec]] §6.4(b) draws for $\tau$, arrived at independently, and the agreement is mild evidence that the split is a property of the framework rather than of either argument.

Practical consequence: **$\eta_\Gamma$ is available now, at every seam, including multiphysics ones.** It is a runtime error indicator that requires only `differentiable: vjp` and the port residual, and it is the only quantity in the framework that estimates error *while running* without a reference. That is worth more than the full DWR machinery it is a fragment of.

---

# 9. I9 / G5 — cut placement, kept at the status it has

[[generalization-requirements]] G5 proposes a criterion as an **[AI Inference]**: cut where the exact DtN is closest to local — where coupling is weakest — never along a shear layer, wake centreline or reaction front. [[end-to-end-architecture-spec]] §4.2 adopts it as policy *with its speculative status preserved verbatim*, and §14 item 3 records that adopting it does not upgrade it.

**That status is unchanged here.** What this page adds is only a *form* the criterion can be measured in, so that it becomes falsifiable rather than remaining a slogan.

Both factors of the $\sigma$ bound come from one probe, so define a **cut score** from the assembled matrix:

$$\mathcal Q(\Gamma) \;=\; \frac{1}{\beta}\cdot\frac{\bigl\lVert \tilde\Lambda^M-\operatorname{diag}\tilde\Lambda^M\bigr\rVert}{\bigl\lVert \tilde\Lambda^M\bigr\rVert} ,$$

the off-diagonal mass — how far the operator is from local — amplified by the conditioning that multiplies it in the bound. **Lower is better; both factors are printed by the probe already.**

**[AI Inference], and flagged as the weakest claim on this page.** That $\mathcal Q$ is the right scalarization is a guess. The off-diagonal mass in the *Fourier* basis is a locality measure; in an arbitrary basis it is not, and the choice of basis is doing work that is not justified here. The defensible part is narrower and worth separating: **cut quality is measurable before any rollout, from quantities the probe already returns, and it should be reported per seam whether or not $\mathcal Q$ is the right combination of them.**

---

# 10. What this leaves undecided

Recorded so the page is not read as closing more than it does.

| # | Still open after this page | Why |
|---|---|---|
| 1 | **$P_i$ itself is declared and can be wrong.** Adjointness prevents a power leak, never an inaccurate transfer | §2.3; its error is $\sigma_{\text{nc}}$ and must be measured |
| 2 | **The inf-sup constant is measured, not proved** | §4.3. For a frozen expert there is no approximation theory to prove it from; a measured $\beta$ is the only available version and it is state-dependent |
| 3 | **The multirate stability limit is conjectured to follow $\kappa(\tilde\Lambda)$** | §5.2 consequence 3, untested |
| 4 | **The cross-point null-count identity is stated as an inequality, not an equality** | §7 |
| 5 | **DWR's interior half is blocked by O2** | §8 — and no construction here changes that |
| 6 | **The cut score $\mathcal Q$ is a guess at a scalarization** | §9, explicitly |
| 7 | **Nothing here touches $\tau$** | The transfer theory governs $\sigma$ and the passivity route to $L$. Agent infidelity is untouched, as it is by every coupling choice ([[general-coupling-scheme]] §1.1) |

**And the honest framing of the whole page:** this is transcription plus one unification. The unification (§2) is the part that is this vault's, and it is the part most worth checking, because if $R_i=P_i^\ast$ is *not* the right condition at some port type then six mechanisms come apart again rather than one.

---

# 11. What this changes

1. **Six importable gaps become one declared object** — $M$ and $P_i$ per seam — with one conformance test and one refusal.
2. **G11 and G3 are the same condition**, and getting the mapping backwards stops being expressible.
3. **G14 stops being a restriction of the construction** and becomes a restriction of a connection rule written before the construction existed. The revised rule is in §4.1. **[AI Inference]**, and cheap to falsify.
4. **The multiplier space and the probe basis are one space**, so non-conforming coupling costs no additional probes.
5. **Multirate and waveform relaxation are the time-axis pair** of the space-axis mapping rule, and R9 becomes a stated condition with a refusal.
6. **Nondimensionalization gains a power constraint** $s_e s_f=s_P$ that is checkable from the declaration alone, and scale-set completeness becomes decidable per port type.
7. **Cross-points gain a recorded decision** conditional on $\mathcal D$, and a detection that rides on the probe's existing null-space check.
8. **$\eta_\Gamma$ is identified as the one runtime error indicator that survives a multiphysics seam.**

---

## See Also

- [[end-to-end-architecture-spec]] — the layers this page supplies rules for; L3's admissibility checklist gains $M$ and $P_i$ as the declared objects, and C3/C6 collapse into one condition
- [[plug-in-composition-theorems]] — the companion page: closure, substitution and conformance. §3's congruence is what makes passivity survive nesting there
- [[theory-closure-audit]] — §4's nine importable mechanisms, of which this page writes six as one and three as three
- [[probed-dtn-coupling]] — the construction this page re-expresses on a common interface space; §2.2's basis-dimension argument becomes §4.3's rule for $\dim M$
- [[port-algebra-atlas-0.1]] — §5's connection rule is revised in §4.1; §9.3's mapping choice is derived in §2.2; §8's nondimensionalization cost gains the constraint in §6
- [[master-error-bound]] — $\sigma$ gains two named components, $\sigma_{\text{nc}}$ (§4.2) and $\sigma_{\text{time}}$ (§5.3), neither reduced by iteration
- [[general-coupling-scheme]] — R9 is written as a condition in §5.2; R3's `bc_time_varying` is identified as the precondition for reducing $\sigma_{\text{time}}$
- [[temporal-error-accumulation]] — §3's window is what makes §5.3's higher-order interface representation possible at all
- [[generalization-requirements]] — G11, G3, G14, G4, G13, G18, G1, G7, G5 are the nine gaps addressed
- [[conservation-as-constraint-atlas-0.1]] — the conservative half of the space rule is its per-edge enforcement, generalized
- [[schwarz-iteration-atlas-0.1]] — §5.2's silent-residual warning is the same signature as its bitwise-zero result
- [[gap-worklist]] — where the measurements this page implies are tracked
