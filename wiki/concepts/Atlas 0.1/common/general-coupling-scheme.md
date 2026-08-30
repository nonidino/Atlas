# The General Coupling Scheme — Schwarz, generalized to a declaration

**Type:** Concept page — framework specification, **version-independent** (folder: `Atlas 0.1/common/`)
**Status:** written 2026-08-26. Generalizes [[schwarz-iteration-atlas-0.1]] from *a thing done to the wind farm* into **a seven-parameter scheme whose admissible settings are derived from what the agents declare**, so any case instantiates it without redesign.
**Related:** [[master-error-bound]] · [[probed-dtn-coupling]] · [[schwarz-iteration-atlas-0.1]] · [[port-algebra-atlas-0.1]] · [[expert-library-atlas-0.1]] · [[temporal-error-accumulation]] · [[generalization-requirements]] · [[agent-definition-atlas-0.1]] · [[symmetry-averaging-atlas-0.1]] · [[conservation-as-constraint-atlas-0.1]]

> **The one-line version.** Every coupling method the vault has considered — one-pass exchange, halo Schwarz, flux-BC Schwarz, the thrust fixed point, waveform relaxation, the probed Schur solve — is **one point in a seven-dimensional scheme space**. Write the space down once, state which points are *admissible* given what each expert declares it can accept, and the coupling stops being designed per case and starts being **compiled** from the agent library. Each axis maps to exactly one term of [[master-error-bound]], so choosing a scheme *is* choosing an error budget.

---

# 1. The scheme

$$\boxed{\;\Sigma \;=\; \bigl(\ \mathcal D,\ \ \tilde\Lambda,\ \ \mathcal O,\ \ \mathcal K,\ \ \mathcal C,\ \ W,\ \ \varepsilon_{\text{tol}}\ \bigr)\;}$$

| axis | choices | what it sets | bound term |
|---|---|---|---|
| $\mathcal D$ **decomposition** | `overlapping(δ)` · `non-overlapping` | whether a partition of unity or a trace is the primitive | $\rho$, and whether **cross-points** exist (§5) |
| $\tilde\Lambda$ **transmission** | `dirichlet` ≺ `neumann` ≺ `dirichlet-neumann` ≺ `robin(α)` ≺ `optimized-robin` ≺ `ventcell` ≺ `probed-DtN` | the rung of the ladder — *what agreement means* | $\boldsymbol\sigma$ |
| $\mathcal O$ **ordering** | `additive` · `multiplicative` · `restricted-additive` | order-dependence of the sweep | $\rho$; and composability with other guarantees |
| $\mathcal K$ **accelerator** | `richardson` · `krylov` · `newton-krylov` · `direct-schur` | how $\lambda^{(k)}\to\lambda^\dagger$ | $\boldsymbol\gamma$ |
| $\mathcal C$ **levels** | `one` · `two(coarse)` | global information transfer per iteration | $\rho$ at large $N$ |
| $W$ **window** | $1,2,\dots$ macro-steps | step-wise coupling vs. waveform relaxation | $\gamma$ count; and the method's share of $L$ |
| $\varepsilon_{\text{tol}}$ | residual target on $\lVert\mathcal G\rVert$ | when to stop | $\gamma$ |

**$\tau$ appears in no row.** It is a property of the agents, not of the coupling — which is the formal statement of [[schwarz-iteration-atlas-0.1]] §5.3's finding that faithfulness is a *precondition* of coupling helping, not a peer of it. **No choice of $\Sigma$ can reduce $\tau$.**

## 1.1 The vault's existing methods are all points in this space

| method | $\Sigma$ |
|---|---|
| One-pass exchange (R2, adopted) | (`overlapping(0.5D)`, `dirichlet`, `additive`, `richardson`, `one`, $W{=}1$, $k{=}1$) |
| R1 halo variant (tested at W3, not adopted) | same, with halo tokens instead of declared flux |
| The §8.3 thrust fixed point | as above but $\mathcal K$ iterated to a **scalar** criterion — a mis-specified $\varepsilon_{\text{tol}}$, not a different method |
| Schwarz stage S1 (refused) | $k>1$ — refused because under periodic windows $\tilde\Lambda\equiv0$, so the axis is degenerate |
| Stage S2 (`reference.WindowNS`) | $\tilde\Lambda=$`dirichlet` **made real** — the first point in the space with a nonzero $\tilde\Lambda$ |
| [[probed-dtn-coupling]] | (`non-overlapping`, `probed-DtN`, `additive`, `direct-schur`, `two` implicitly, $W{=}1$, roundoff) |
| [[temporal-error-accumulation]] §3 | the same with $W>1$ |

**Seeing them in one table is the argument for the page.** These were treated as six design debates; they are six coordinate settings, and the debates were about which axis mattered rather than which value to pick.

---

# 2. What an expert must declare

The scheme cannot be chosen freely: it is bounded above by what the agents can accept. So each expert publishes a capability record alongside its [[port-algebra-atlas-0.1]] port list.

```
ExpertCapabilities:
  bc_channel      : none | dirichlet | neumann | robin | ventcell
  bc_time_varying : bool                 # can the ring vary within a macro-step?
  differentiable  : none | jvp | vjp | both
  dt_native       : float                # the step it was trained/built at
  L_native        : float                # window physical size
  regime_law      : callable             # e.g. Re_eff = T_s / (nu_p * L^2)
  ports           : [(type, geometry, conservative)]
  storage         : H(u) | none          # for the passivity certificate
  equivariances   : [group]              # for symmetry averaging
  validity        : predicate(state, conditioning) -> bool
```

Two of these are new and both are load-bearing. `storage` is what turns [[master-error-bound]] §6.1 from a hope into a check. `validity` is the abstention precondition — *enforce, measure, or decline*.

**The frozen wind-farm checkpoint declares `bc_channel: none`**, and that single field is the formal cause of every coupling limitation recorded against it.

---

# 3. Admissibility — the rules that make the scheme derivable

**R1 — the interface is as weak as its weakest agent.**
$$\operatorname{rung}(\tilde\Lambda) \;\le\; \min_{i\,\in\,\text{ports of }\Gamma}\ \operatorname{rung}\bigl(\mathcal P_i.\texttt{bc\_channel}\bigr)$$

**R2 — and probing is the exception that makes R1 survivable.** `probed-DtN` requires only `bc_channel ≥ dirichlet`, because the higher rung is built **outside** the expert from Dirichlet-in/flux-out probes ([[probed-dtn-coupling]] §1.1). This is the single most consequential rule on the page: **it decouples the achievable rung from the expert's own interface**, and it is why the ladder is not closed to black boxes.

**R3 — $W>1$ requires `bc_time_varying` on every agent at that interface.** A ring that must be held constant across the window is a $W=1$ scheme wearing a longer name.

**R4 — the exchange interval cannot go below $\max_i \texttt{dt\_native}$.** A frozen expert's native step is a constraint on *coupling*, not on integration. Note the asymmetry [[temporal-error-accumulation]] §3.6 draws: $W$ can always be increased, never decreased — **the window is the one axis a frozen expert can move along.**

**R5 — `multiplicative` is forbidden when any composition-layer guarantee depends on equivalent treatment of agents.** It breaks [[symmetry-averaging-atlas-0.1]]'s exactness and reinstalls graph-cycle path dependence. Atlas is `additive` by rule, not by preference. A `direct-schur` solve satisfies this *automatically*, being order-free.

**R6 — $\mathcal K\in\{$`newton-krylov`,`direct-schur`$\}$ requires $\tilde\Lambda\not\equiv0$.** Otherwise the interface system is not ill-conditioned but **empty**, and every trace is equally consistent.

**R7 — probing method follows `differentiable`:** `jvp` → exact JVP probe; `none` but deterministic and smooth → finite differences with $\epsilon$ set by the reproducibility floor; otherwise → regularized regression over $m'>m$ probes.

**R8 — `two(coarse)` via a classical coarse solve requires an agent that can run at a different $L$ without changing its regime** (check `regime_law`). Via a **probed Schur complement it requires nothing extra**, since every call stays at $L_{\text{native}}$ — this is [[schwarz-iteration-atlas-0.1]] §S3's blocker, and R8 is where it is discharged.

**R9 — under multirate ($\texttt{dt\_native}$ differing across agents), conservation must be enforced on the *time-integrated* flux over the macro-step, not instantaneously.** Pointwise-in-time flux matching between agents on different clocks is not conservative and the residual will not show it. See [[generalization-requirements]] G4.

---

# 4. The compiler

Given a graph and a set of capability records, the scheme is **derived**, not debated:

```
compile_scheme(graph, capabilities, budget):
    Σ.Λ̃  = highest rung admissible under R1, lifted by R2 where a probe is affordable
    Σ.O  = additive                                  # R5, always
    Σ.K  = direct-schur if R6 and R7 satisfiable and budget allows
           else krylov if Λ̃ ≠ 0
           else richardson
    Σ.C  = two(probed-schur) if R8 via probing else one
    Σ.W  = clamp( round(1 / ln L̂), 1, W_max_by_R3 )   # L̂ fitted, or 1 if unknown
    Σ.D  = non-overlapping if Σ.Λ̃ == probed-DtN else overlapping(δ ≥ U∞·Δt)
    Σ.tol= ε such that γ ≪ min(τ̂, σ̂)                 # never tighter than the dominant term
    return Σ, predicted_bound_terms(Σ)
```

Two lines deserve emphasis.

**`Σ.tol` is set relative to the other terms, not absolutely.** Converging the interface far below $\tau$ and $\sigma$ is the precise form of the mistake [[composition-error-theory]] §2.1 warns about — spending compute to make the negligible term smaller while removing the only alarm. **A tolerance tighter than $\min(\hat\tau,\hat\sigma)$ should be refused by the compiler**, the way `schwarz > 1` under periodic windows is already refused at config time.

**`Σ.W` needs $\hat L$, which is unmeasured.** Until it is, $W=1$ is the honest default; the compiler should say so rather than guess.

## 4.1 The coupled macro-step, in full

```
CoupledStep(u^n, graph, Σ):
    λ ← previous step's converged trace          # best available initial guess
    
    def G(λ):                                     # interface residual
        parallel for each agent i:                # additive: same iterate for all
            μ_i  ← Σ.Λ̃.impose(λ, port_i)          # rung-specific: ring, flux, or combination
            u_i  ← E_i.step(R_i u^n, μ_i, W)      # W native steps
            f_i  ← port_flux(u_i, port_i)         # the MECH/THERM/... effort, single-valued
        return Σ.Λ̃.residual({u_i}, {f_i}, λ)

    λ* ← Σ.K.solve(G, tol=Σ.tol)                  # richardson | krylov | newton | direct
    if Σ.C == two:  λ* ← λ* + coarse_correction(λ*, S_probed)

    u^{n+1} ← A({u_i(λ*)})                        # PoU or single-valued-flux assembly
    if equivariances: u^{n+1} ← group_average(u^{n+1})

    emit γ=‖G(λ*)‖, β=σ_min(S), κ(S), π=passivity_defect, R(t), and the bound estimate
    return u^{n+1}
```

**The `emit` line is not optional instrumentation — it is the contract.** [[master-error-bound]] §7 observes that the framework currently reports $\gamma$ and nothing else; this line is where that is fixed, and every quantity on it is a by-product of machinery the solve already built.

---

# 5. The two structural traps

**Cross-points.** `non-overlapping` decomposition creates points where three or more subdomains meet, and there the transmission conditions are **not independent** — naive Robin conditions at a cross-point are ill-posed, and this is the specific difficulty FETI-DP and BDDC exist to handle (by making corner degrees of freedom primal, i.e. continuous by construction). **Overlapping-with-partition-of-unity does not have this problem**, which is why the current tiling has never hit it despite 124 tiles meeting four-at-a-corner.

> **Moving to `probed-DtN` moves to `non-overlapping`, and therefore *introduces* a difficulty the present scheme does not have.** [[probed-dtn-coupling]] does not mention it. Any 2D or 3D graph with more than a chain topology needs a cross-point rule before that construction is built.

**The degenerate-axis trap.** [[schwarz-iteration-atlas-0.1]] refused stage S1 because iterating a periodic window is the identity. Generalized: **an axis whose value cannot affect the answer must be refused at config time, not measured.** The compiler should check R6 before accepting $\mathcal K$, and R3 before accepting $W$, and say which rule it failed — the way the existing `schwarz > 1` guard quotes the bitwise measurement in its error message.

---

# 6. Instantiations

| case | $\mathcal D$ | $\tilde\Lambda$ | $\mathcal K$ | $W$ | binding rule |
|---|---|---|---|---|---|
| Wind farm, **as built** | `overlapping(0.5D)` | `dirichlet` (frozen: **none**) | `richardson`, $k{=}1$ | 1 | **R1** — `bc_channel: none` collapses everything |
| Wind farm, **next** | `overlapping` | `dirichlet` via `WindowNS` | `krylov` | 1 | R6 now satisfiable; R4 fixes the exchange interval |
| Wind farm, **target** | `non-overlapping` | `probed-DtN` | `direct-schur` | $1/\ln\hat L$ | **R2** lifts the rung; §5 cross-point rule needed |
| Rocket ascent | mixed field + lumped | `robin` at fluid–structure, algebraic at lumped ports | `newton-krylov` | multirate | **R9** — differing `dt_native` per agent |
| RBC control | `overlapping` | `dirichlet-neumann` across the conduction/advection split | `krylov` | 1 | R3 — multi-rate boundary layers at $\Delta t/4$ |

The rocket row is the one that exercises the framework rather than the case: **field and lumped agents as first-class peers**, which is F1's central claim and which R9 plus [[generalization-requirements]] G3 are the conditions for.

---

# 7. What this changes

1. **Coupling becomes a declaration, not a design.** A new case supplies a graph and capability records; $\Sigma$ is compiled. That is the difference between a framework and a co-simulation harness built once per problem.
2. **[[expert-library-atlas-0.1]] gains the capability record of §2**, and the two fields that matter most — `bc_channel` and `storage` — are invisible to every accuracy benchmark.
3. **Every run emits the full bound instrumentation** (§4.1), not just $\gamma$.
4. **Refusals are principled and cite a rule.** The existing `schwarz > 1` guard becomes the first instance of a general mechanism rather than a special case.
5. **[[schwarz-iteration-atlas-0.1]]'s missing port-*resolution-order* rule is answered by R5**, and by the observation that a direct Schur solve makes the question moot.

---

## See Also

- [[master-error-bound]] — each axis of §1 maps to one term; choosing $\Sigma$ is choosing an error budget
- [[probed-dtn-coupling]] — the `probed-DtN` rung and rule **R2**, the exception that keeps the ladder open to black boxes; §5 adds the cross-point caveat it omits
- [[schwarz-iteration-atlas-0.1]] — the case-specific ancestor of this page; its additive-only argument becomes R5, its S3 blocker is discharged by R8
- [[temporal-error-accumulation]] — sets $W$; R4's asymmetry (a frozen expert can coarsen but never refine)
- [[port-algebra-atlas-0.1]] — the port list §2's capability record extends; the single-valued flux §4.1 assembles
- [[expert-library-atlas-0.1]] — where capability records live, and why they beat accuracy as a selection axis
- [[generalization-requirements]] — G3 (field↔lumped), G4 (R9's multirate theory), G1 (§5's cross-points)
- [[symmetry-averaging-atlas-0.1]] — the guarantee R5 exists to protect
- [[agent-definition-atlas-0.1]] — the granularity heuristic that produces the graph this scheme couples
- [[interface-transfer-theory]] — **writes R9 as a condition** rather than prose, with the two consequences it did not carry (interface representation order caps the scheme order; stability is set by coupling stiffness). It also identifies R3's `bc_time_varying` as the precondition for reducing $\sigma_{\text{time}}$ at all
- [[plug-in-composition-theorems]] — composes §2's capability record **field by field**: eight fields close under composition, three (`L_native`, `regime_law`, `validity`) do not, and two propagate the `SeamReference` hole upward. It also adds the missing check that the record is *true*
- [[end-to-end-architecture-spec]] — **the whole system this scheme is one layer of.** §2's capability record becomes L1 (extended by five fields), R1–R9 are distributed across L3–L7, and §4's compiler gains the seven-hypothesis envelope as an admissibility check plus a **three-verdict** outcome (`admit`, `admit-uncertified`, `refuse`) in place of a binary one
