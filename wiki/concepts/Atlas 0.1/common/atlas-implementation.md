# The Atlas Implementation — the specification, executed

**Type:** Concept page — **implementation record**, version-independent (folder: `Atlas 0.1/common/`)
**Status:** written 2026-08-27, immediately after building `atlas/` at the vault root; **updated the same day, once the six corrections it opened were written back into the pages they indicted.** This page is the **knowledge-layer record** of what was built, what it decided where the specification was ambiguous, and what it could not do. The code's own documentation lives with the code in `atlas/README.md`; this page is what the wiki needs to know about it.
**Related:** [[end-to-end-architecture-spec]] · [[interface-transfer-theory]] · [[plug-in-composition-theorems]] · [[general-coupling-scheme]] · [[probed-dtn-coupling]] · [[master-error-bound]] · [[port-algebra-atlas-0.1]] · [[gap-worklist]] · [[code-and-papers]] · [[log]]

> **The one-line version.** Nine layers, a seven-hypothesis envelope and a three-verdict compiler, built as 23 modules and 120 tests over `numpy` alone. **A new case study is a graph of declared agent capability records plus port connections, with zero hand-written coupling code** — and that claim is now an executable test rather than an aspiration. The specification survived implementation: nothing in the master bound, the port algebra, the three-verdict system, the transfer unification or the composition theorems needed changing. **Seven smaller things did**, they are the reason this page exists, and **all seven have since been written back into the pages that carried them** — so the theory layer and the code now agree, and no page states a rule its implementation contradicts.

---

# 1. What the layers became

| L | Layer | Module | The refusal it owns |
|---|---|---|---|
| **L1** | expert declaration, capability record | `capability.py` | a record missing a field a downstream layer needs |
| **L1.5** | routing (G15) | `routing.py` | an empty feasible set — never a nearest-neighbour fallback |
| **L2** | decomposition, cut placement | `compiler.py` | motion; an unledgered mutation; untreated cross-points |
| **L3** | port algebra, admissibility | `ports.py`, `transfer.py`, `admissibility.py` | no declared interface space and prolongation; an incomplete or non-power-preserving scale set |
| **L4** | transmission operator | `probe.py` | a probed null space of the wrong dimension |
| **L5** | interface solve, accelerator | `scheme.py`, `compiler.py`, `solve.py` | R6 on an empty operator; a degenerate axis |
| **L6** | assembly | `assembly.py` | the partition-of-unity identity |
| **L7** | time integration, multirate | `compiler.py`, `solve.py` | R9 under multirate; R3 on the window |
| **L8** | emit | `emit.py` | a run that cannot produce the envelope stamp |
| **L9** | typing of claims | `claims.py` | a trajectory metric quoted past its own horizon |

Plus `composition.py` (closure and substitution), `conformance.py` (the suite), `holes.py` (the five slots and the unmeasured constants), and `verdict.py` (the three-verdict system and the decision record).

**Two design choices carry most of the weight.**

`holes.Unmeasured` is a value that **refuses to behave like a number** — `float(L)` raises — so a bound cannot be quoted from a constant nobody measured. It is the mechanical form of §11's decision that an unmeasured $L$ produces `UNTYPED`, and it makes the failure loud at the moment of use rather than at the moment of reading.

`NamedHole.solve()` **raises**. The five slots are declared interfaces with no implementation, and every configuration that reaches one records what it owes: the ledger reports, per run, which field-5 measurements a touched slot still lacks. A static run that never moves an interface still owes the operator drift, because that number prices the cached-$S$ economics the whole probing argument rests on.

---

# 2. Where the specification was ambiguous, and what was decided

Recorded here because a specification that does not separate its decisions from its inventions is worse than no specification, and the same applies to an implementation of one.

> **Closed 2026-08-27, later the same day.** Items 1, 2, 3, 6 and 7 were corrections to the theory, tracked as [[gap-worklist]] W39–W44, and they have since been **written into the pages themselves** — [[end-to-end-architecture-spec]] §1.1, §2.1, §3.4 and §5.3; [[interface-transfer-theory]] §7; [[probed-dtn-coupling]] §2.1 and §4.2; [[port-algebra-atlas-0.1]] §3.2 and §5. The amendment blockquotes that carried them are gone and the pages read correctly in their own body text. **They are kept in the list below because the reasoning is why the code looks the way it does**, and because a reader holding an older copy of the spec will find the difference here. Items 4 and 5 remain implementation choices no page speaks to.

1. **E2 and L3 admissibility are not the same test.** A seam can satisfy the hypothesis — static, geometrically coincident — and still be refused for want of the declaration the amended connection rule requires. That is the wind farm, and [[end-to-end-architecture-spec]] §13.1's own stamp is only reproducible if the two are kept apart.
2. **A missing declaration leaves E7 `unchecked`, not `fails`.** Only a *measured* non-adjoint pair or a *measured* negative passivity mode fails it.
3. **The transmission rung is decided before L2**, because it fixes the decomposition axis and the axis decides whether cross-points exist.
4. **A refused seam is still probed**, on a provisional interface space, marked diagnosis-only. Refusing a connection and then declining to measure it would reproduce the exact failure the wind-farm finding is about — *the same finding was available for the cost of one probe.*
5. **R6 fires per seam, not globally.** One zero block makes the assembled system singular there, and a direct solve over it is not defined.
6. **A passivity defect below the arithmetic noise floor is reported as zero**, with the raw eigenvalue kept alongside.
7. **A null-space deficit is decertified, not refused.** The safe reading of the cross-point identity covers only the excess case; a deficit still indicts either the probe or the declaration.

---

# 3. What it could not do, and why

**The $K$-threshold refusal could not fire, and the spec has withdrawn it.** [[end-to-end-architecture-spec]] §3.3 specified a **refusal** for any graph at or above the $K$ at which $P(\text{some agent OOD})\to1$, and set no threshold. A gate with no value is not a gate, and inventing one would be the failure the whole page is against — so the compiler decertified and said exactly that, and §3.4 now makes the decertification the rule, emits $K$ with it, and names $K^\star(\epsilon)=\lceil\log(1-\epsilon)/\log(1-p)\rceil$ as what a measured declination rate would reinstate. **The threshold is still unset**; what changed is that the spec no longer claims a gate it does not have, and says which measurement (W13) it is waiting on.

**`compile_scheme` never returns `admit`** for any graph that can currently be written. $L$ is unmeasured, so E5 is unchecked; the assembly accuracy condition does not exist, so E6 is unchecked. **That is not a limitation of the code** — it is [[master-error-bound]] §7's sentence, enforced. The best available verdict today is `admit-uncertified`, and reaching `admit` requires W1 and the `AssemblyCertificate` condition, in that order.

**No expert in the library has been run through any of it.** Every measurement remains open. What changed is that each one is now blocked on *taking a measurement* rather than on *building a mechanism*, which is a much cheaper kind of blocked and is the whole content of the [[gap-worklist]] status update.

---

# 4. The two compile traces, as executed rather than on paper

**The wind farm as built** compiles to **`refuse`** with the stamp $(\text{E1 holds},\ \text{E2 holds},\ \text{E3 holds},\ \text{E4 holds},\ \text{E5}/\text{E6}/\text{E7 unchecked})$, reproducing [[end-to-end-architecture-spec]] §13.1 exactly. **No rollout appears anywhere in that trace.** The frozen checkpoint's $\Xi=0$ falls out of the probe machinery rather than being special-cased: its declared boundary response does not depend on the imposed trace, so $\Lambda\equiv0$, so the interface problem is *empty* rather than ill-conditioned, and the word *coupled* is refused on the output while the run itself is not.

**Give the same graph a boundary-capable expert** and R2 lifts the rung to `probed-DtN`, which switches the axis to non-overlapping, which **introduces the cross-point difficulty the overlapping decomposition never had** — and L2 refuses. Declare primal cross-point degrees of freedom and it compiles to `admit-uncertified` with a direct Schur solve. That sequence is [[general-coupling-scheme]] §6's *"wind farm, target"* row, and the refusal in the middle of it is the trade §4.3 predicted, arriving where the spec said it would.

**The rocket does not compile**, and trips `InterfaceMotion`, `SeamReference` and `AssemblyCertificate` in its declared scope, with `TopologyEvent` arriving when staging is declared. Its fluid–structure seam has E3 `fails` and $\tau$ `UNDEFINED` — **and its $\beta$, $\kappa$, passivity spectrum and optimal Robin coefficients are all assembled anyway**, because the probe calls a declared boundary response and never mentions a governing equation. That is §6.4(b) surviving contact with code, and it is the sharpest confirmation the build produced.

---

# 5. What this changes

1. **The specification is executable**, and its two paper traces are now regression tests rather than arguments.
2. **The plug-in claim has an executable form.** A three-agent chain nobody wrote coupling code for compiles from declarations alone.
3. **Nineteen worklist rows moved off `open`**, and the ones that moved to `in progress` are blocked on a measurement rather than a construction.
4. **Six new items, W39–W44**, all corrections to pages rather than new mechanisms — and **all six are now closed**, folded into the body text of the five pages they indicted rather than left standing as notes beside the lines they contradict. W43 closed by retraction: the unfireable refusal is gone and the measurement that would restore it is named.
5. **The transfer unification is confirmed under construction** — with one declared prolongation and the reduction derived, the reversed-mapping bug is *not expressible*, and reproducing the actuator disk's historical hand-written reduction requires deliberately reaching for an escape hatch that the conformance suite then catches.

---

---

# 6. What the first real expert added (2026-08-27)

Running the compiler against `reference.WindowNS` rather than a declared matrix ([[tier0-measurements]]) added **two rules, four capability fields and one missing path**, and every one of them came from a measurement rather than from reading the spec again.

| addition | module | why it exists |
|---|---|---|
| **R10** — refuse a decomposition of an agent with `elliptic_subsolve: embedded` | `capability.py`, `compiler.py` | a global sub-solve is not decomposable; measured at **99.8% of a composed step's defect**, wearing a $\tau$ label |
| **R2b** — gate probed-DtN on `time_discretization` | `capability.py`, `compiler.py` | flux balance is a boundary-value-problem condition; an explicit macro-step poses none. Recorded as an **admit**, since the compiler corrects the rung |
| **halo rule** — `stencil_radius * substeps_per_macro_step` | `capability.py`, `compiler.py` | the overlap must outrun the agent's own domain of dependence, or the blend is contaminated |
| **`MeasuredConstants`** on the graph | `graph.py`, `compiler.py` | **W45**: `unmeasured()` hard-coded its list and consulted nothing, and `L_fitted` was read by nothing. There was an emit path and no ingest path |
| **`GridPartitionOfUnity`**, and `norm_A` corrected | `assembly.py` | the old `norm_A` computed $\lVert I\rVert$ and could never report anything; the dense form needs a $65025$-square matrix for a $255^2$ field |

**Three of the seven original ambiguities were resolved by the same run.** The null-space count is not a property of the seam alone but of the seam *and where the elliptic solve lives*; the flux convention is §2.2's, normatively; and the $\beta$/$\pi$ verdict belongs to the **seam**, not the block — the downstream block reads $\kappa=7061$ where its seam reads $1.196$.

**The package's own numbers moved by one to three orders**: composed macro-step $220\times$, interface conditioning $21.7\to1.20$, operator asymmetry $0.153\to0.002$, and the fitted $L$ from disagreeing with the monolith's to matching it to five decimals. **141 tests**, 21 of them new and written against the rules rather than the case study, so they run without the build repo checked out.

**And the honest end state**: `split-step` compiles to `admit-uncertified` with **zero refusals**, and `admit` is blocked by exactly one named hole — `AssemblyCertificate`'s `condition`, the accuracy rule nobody has written. That is a sharper statement than the one this page opened with.

---

## See Also

- [[end-to-end-architecture-spec]] — the specification this executes. W39, W40 and W43 landed as its new **§1.1** (stamp vs verdict), **§2.1** (L1 → rung → L2) and **§3.4** (the retracted $K$ threshold), plus corrected rows in §1, §3.3 and §5.3
- [[interface-transfer-theory]] — the declared object the whole of L3 is built on; W41 rewrote its §7 null-space count as $n_0(\Gamma)$ with a verdict for each direction of a mismatch
- [[plug-in-composition-theorems]] — closure, substitution and conformance, implemented in `composition.py` and `conformance.py`
- [[general-coupling-scheme]] — the seven-axis scheme and the nine rules the compiler derives and cites
- [[probed-dtn-coupling]] — the construction `probe.py` implements; W42 gave its passivity defect a scale-relative floor in §4.2, and W41 corrected the per-seam null-space check at its origin in §2.1
- [[master-error-bound]] — the theorem every layer is arranged to keep valid, and the reason no graph reaches `admit` today
- [[port-algebra-atlas-0.1]] — the closed vocabulary; W44 put the `ADVEC` passenger list into §5's connection rule and the per-face argument into §3.2
- [[gap-worklist]] — Tier 7, the status update across nineteen rows, and W39–W44, all six now `done`
- [[code-and-papers]] — where the package lands in the build repo
- [[log]] — the full record of the nine findings, under *"The specification is implemented"*
