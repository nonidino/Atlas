# Can an Inequality-Constrained Seam Be Compiled? — rung 8's admissibility question

**Type:** Concept page — **decision**, with one supporting measurement (folder: `Atlas 0.1/common/`)
**Status:** decided 2026-09-10, Tier 42. `scripts/w181_inequality_seam_probe.py`, `out/w181/w181.json`, `tests/test_tier42_inequality_seam.py`. Worklist rows **W165** (the question), **W181** (R7's unchecked premise), **W182** (the named hole).
**Related:** [[gap-worklist]] · [[case-study-ladder-to-f1]] · [[f1-pathmap-and-end-goal]] · [[port-algebra-atlas-0.1]] · [[end-to-end-architecture-spec]] · [[atlas-implementation]] · [[theory-closure-audit]] · [[master-error-bound]] · [[probed-dtn-coupling]] · [[tier0-measurements]]

---

# 0. The one-paragraph result

**No. A seam whose interface condition is an inequality cannot be compiled today, and the reason it looks as though it can is that nothing asks.** Three graphs differing only in the shape of one side's `boundary_response` — linear, kinked at the probe base, kinked behind a gap — compile to the **same verdict under the same eighteen decisions in the same order**, with $\beta$ moving $0.13\%$ between them. At the solved interface trace the framework's own reported residual is $5\times10^{-16}$ and the port-matching condition it is standing in for is violated at $4\%$ of the trace scale: a ratio of $7\times10^{13}$, against a linear control where the two are the same number. That is this package's **silent-wrongness class** by its own definition, so the verdict is a **named hole whose compiler response is a refusal, not a decertification** — and the refusal has to rest on a **declaration**, because the measurement that would find it cannot: at a positively homogeneous kink the $\varepsilon$ sweep is stable to **nine digits**, which is the same nine the smooth control returns.

**What that changes for rung 8.** Its first module stops being *couple a tyre* and becomes *let a seam say its condition is not an equation, refuse it, and emit the number the missing rule will constrain*. That module needs no contact expert. Rung 8 itself still does — §8 says exactly where and why.

---

# 1. The question, and the half of it that is already settled

[[gap-worklist]] W165 split rung 8 in two and the split is not re-litigated here.

**The bond exists.** Contact traction $\mathbf t$ and slip velocity $\mathbf v$ have a product $\mathbf t\cdot\mathbf v$ with units of W m$^{-2}$, which is what [[port-algebra-atlas-0.1]] §3.1 requires of `MECH`. Friction is a *dissipative* bond rather than a storage one, and the port algebra never asked a bond to be lossless — §6's residual $\mathcal R(t)$ has a declared-dissipation term precisely for this. **No `PortAmendment` is needed and the vocabulary stays at five.**

**Admissibility is the open half, and it is this page's subject.** The variables are typed; the *condition relating them* is not an equation. Normal contact is a complementarity triple,

$$g \ge 0,\qquad p \ge 0,\qquad g\,p = 0,$$

read as *the bodies cannot interpenetrate, the contact cannot pull, and at most one of the two is ever slack*. Tangentially, Coulomb's law is a disjunction:

$$\lVert \mathbf t_T\rVert < \mu p \;\Rightarrow\; \dot{\mathbf s} = \mathbf 0 \quad(\text{stick}),\qquad
\lVert \mathbf t_T\rVert = \mu p \;\Rightarrow\; \dot{\mathbf s} = -\gamma\,\mathbf t_T,\ \gamma \ge 0 \quad(\text{slip}).$$

Both are the same object in different clothes: a **variational inequality**, equivalently a complementarity problem, equivalently a *piecewise* smooth relation with a switching set the solution itself decides.

**Why that is not a small generalisation.** Every layer of this compiler is arranged around one sentence, and it is written down in exactly one place — `atlas/admissibility.py:281`, in C5's docstring:

> `e_A = e_B and f_A = -f_B on the seam.`

Two equalities. The composition layer's central object, `solve.InterfaceProblem` (`atlas/solve.py:89`), is their discrete form:

$$G(\lambda) = S\lambda - \chi \;=\; 0 .$$

An inequality seam replaces that root-finding problem with a **linear complementarity problem**: find $\lambda$ with

$$\lambda \ge 0,\qquad S\lambda - \chi \ge 0,\qquad \lambda^{\!\top}\!\left(S\lambda - \chi\right) = 0 .$$

Same $S$, same $\chi$, **different problem** — different solution set, different solver, different error theory, different certificate. The LCP is not a perturbation of the linear system; it is a different question with the same data, and all four branches of `solve_interface` answer the first one.

> **Intuition, for a reader who wants it before the algebra.** An equality seam is a handshake: whatever leaves one side enters the other, always, and the only question is what the common value is. A contact seam is a handshake that may or may not be happening, and *whether it is happening* is part of the answer rather than part of the setup. A compiler built to find the common value will always find one — including when the right answer is *the hands are not touching*.

---

# 2. The enumeration: which of the nine layers assume an equality seam

Read off `atlas/compiler.py`, `atlas/admissibility.py`, `atlas/probe.py`, `atlas/solve.py` and `atlas/envelope.py` at `fcab1e4`, function by function, in compile order. **Cited by line.** The column that matters is the third: *silent* means the layer proceeds and its output is wrong in a way nothing in the run reports.

| L | function | line | does an inequality break it? |
|---|---|---|---|
| **L1** | `_l1_records` | `compiler.py:278` | **No — and that is the defect.** `missing_fields()` (`capability.py:550`) requires `dt_native`, `governing_family`, `boundary_response`, a JVP if one is claimed, and a per-port `effective_resolution`. Nothing about smoothness, branch structure or condition class. A contact agent produces a **complete** record. L1 is the layer a declaration would have to live in, and there is no field to put it in |
| **L1.5** | `_l1_5_routing` | `compiler.py:522` | **No.** Routes candidate experts by required port type. Indifferent |
| **L2** | `_l2_decomposition` · E1 | `compiler.py:587` | **Yes, silently, and it is undeclarable.** E1 is *"the interaction graph is fixed for the whole rollout"* (`envelope.py:54`) and is stamped `holds` whenever `graph.topology_events` is empty. A contact patch's **active set** changes during the rollout: a seam carrying power at $t_1$ carries none at $t_2$. That is an interaction-graph change with no `TopologyEvent` behind it — and one cannot be declared, because a `TopologyEvent` fires at a **declared time** $t$ with a `state_map` and a `ledger`, and a contact switch happens at a time the solution decides |
| **L2** | `_l2_decomposition` · `InterfaceMotion` | `compiler.py:596` | **It fires — for the wrong reason.** A contact patch's boundary moves with the solution, so `MotionClass.SOLUTION_DEPENDENT` is the one field in the schema a contact seam can truthfully set, and setting it **refuses**. Measured (§3.5): declared on the *linear control*, the identical refusal fires, alone. A rule that cannot tell the two apart is not adjudicating the difference between them |
| **L2** | `_r10_elliptic`, `_halo_rule`, `_cut_policy`, cross-points | `1304`, `1771`, `764`, `618` | **No.** They read `elliptic_subsolve`, `governing_family`, `stencil_radius`, `required_halo` and the decomposition axis. None reads anything about the condition |
| **L3** | `_c1_vocabulary` | `admissibility.py:90` | **No, correctly.** Both sides declare `MECH`; traction $\times$ slip velocity is a power. W165's own finding, confirmed at the code |
| **L3** | `_c2367_declared_transfer` | `admissibility.py:117` | **No.** $\dim M$, $\lVert P\rVert$, implementability, and the adjointness residual are properties of the **linear transfer maps** $P_i: M \to V_i$, which stay linear whatever the condition on the trace is |
| **L3** | `_c4_nondimensionalization` | `admissibility.py:245` | **No.** $s_e s_f = s_P$ holds for contact scales as for any other |
| **L3** | **`_c5_power_pairing`** | **`admissibility.py:278`** | **Yes — and it is the one place the equality is written down.** The docstring (line 281) states $e_A=e_B,\ f_A=-f_B$; the *check* is that `connection.orientation` is a non-empty string (line 286). The condition C5 names is therefore verified nowhere at compile time and becomes the runtime port residual $r_\Gamma$ ([[port-algebra-atlas-0.1]] §4). At a contact seam $e_A=e_B$ **holds** — traction is continuous — and $f_A=-f_B$ is **false whenever the patch is open**, because two separated bodies' normal velocities are independent. So the framework's one universal physical-plausibility monitor **reads as a fault exactly where the physics is right** |
| **L3** | `_c9_response_half` | `admissibility.py:302` | **No.** Both sides can declare `EFFORT` |
| **L3** | `_c7_rung`, `_c8_validity` | `admissibility.py:345`, `396` | **No** — and C8 is the only existing hook that could carry it. `validity` is per-agent, per-state, and a declination **halts the rollout** (`solve.py:45`). Declaring *valid only while in contact* turns a routine separation into an abstention, which is the wrong verdict for a normal physical event |
| **L3** | `_l3_global_fields` | `compiler.py:1173` | **No.** W117's provenance check on `GlobalField` |
| **L4** | `probe_block` (`probe.py:883`) → `_columns_finite_difference` | `probe.py:911`, `912`, `845` | **Yes, silently, and it is measured.** Line 845 forms the one-sided chord $\big(f(b+\epsilon d)-f(b)\big)/\epsilon$. At a switch that chord is **one of two one-sided Jacobians**, selected by the sign of $d$ and by whether $\epsilon$ reaches the switch, and the returned `ProbedBlock` records no such choice |
| **L4** | `_fill_diagnostics` | `probe.py:963` | **Yes, silently.** $\beta$, $\kappa$, the null dimension, `operator_content`, `alpha_star` and the passivity spectrum are all computed from that single matrix, and all are healthy on a chord. This is `L4/operator-content`'s shape (W68/W71) one cause along: *the numbers are not wrong, what is wrong is reading them as statements about the physics* |
| **L4** | `L4/E7/passivity` | `compiler.py:2288` | **No — it gains information.** It fires on $\lambda_{\min}\!\big(\mathrm{sym}\,S\big) < -\text{tol}$, and on a genuinely non-associated friction tangent it fires **correctly**. The one rule contact makes sharper rather than blunter (§4b) |
| **L5** | `_l5_l7_scheme` → `solve_interface` | `compiler.py:2341`, `solve.py:132` | **Yes, structurally, and this is the deepest break.** `InterfaceProblem.residual` (`solve.py:102`) is affine. `DIRECT_SCHUR` is `lstsq` (line 155); `KRYLOV` is `_gmres_like` (262); `ROBIN_RICHARDSON` (181) and plain Richardson (168) are stationary iterations on the same residual. **None projects onto a cone.** All four solve the equation; the seam poses the LCP |
| **L6** | `_l6_assembly` (R11), `_r12_conservative_assembly`, `_w49_sigma_branch` | `2796`, `2978`, `3211` | **No.** L6/C1's identity $\lvert A(u)-u^\star\rvert^2=\sum_i\chi_i\lvert u_i-u^\star\rvert^2-V_\chi$ holds for any weights summing to one; R11 is convexity of $\chi$; R12 is the commutator $C(\sum_i\chi_iu_i)=\sum_i\nabla\chi_i\!\cdot\!u_i$. All indifferent to the seam's condition class |
| **L7** | `_r9_flux_matching`, `_r10b_exchange_cadence`, `_clocks` | `2663`, `2548`, `2629` | **Yes, and it is R7's shape one layer along.** R9 requires the **time-integrated** flux to match and checks three things: the clocks nest, both sides supply `boundary_response_integrated`, and the composition actually calls it. All three are about **endpoints**. Nothing asks whether the integrand is continuous *inside* the interval — and a macro-step containing a stick–slip transition has an integrand that is not |
| **L8** | `_l8_emit` | `compiler.py:3329` | **No** — and the artifact is therefore not recoverable evidence. It carries `E1: holds`, a scheme, a bound-terms block and a stamp, and no field anywhere records that the seam's condition was an inequality |
| **L9** | `_l9_claims` | `compiler.py:3275` | **No — and it is the only layer whose theory survives intact.** $L$ is an amplification constant and [[master-error-bound]] needs **Lipschitz continuity**, not differentiability. A two-branch piecewise-affine map is Lipschitz with $L=\max\big(\lVert J_+\rVert,\lVert J_-\rVert\big)$. The top of the error theory is fine; the instrument that feeds it is not |
| **E1–E7** | `envelope.py:51–105` | — | **E1 false and stamped `holds`** (undeclarable). **E2 false and caught**, by the motion refusal that also fires on the control. **E3** is the ordinary multiphysics failure (tyre and road declare different families, $\tau$ `UNDEFINED`) and is not specific to this. **E4, E6** indifferent. **E5** survives via L9. **E7 fails, correctly** |

**The count, stated plainly.** Of the layers, **two break silently** — L4's instrument and L5's solve. **One is stamped false with no way to declare otherwise** — L2's E1. **One reports a correct physical fact as a fault** — L3's C5 residual. **One gains information** — L4's E7. **The rest are indifferent.** And **L1, which is where the declaration would have to live, has no field for it.**

**That last row is the verdict in embryo.** Nothing here is a rule that needs widening. Every rule that reads a declaration reads one that exists and is correct; the thing that does not exist is a declaration saying what class of condition the seam carries.

---

# 3. The measurement

`scripts/w181_inequality_seam_probe.py`, `out/w181/w181.json`, $5.8$ s on `numpy` alone. **No contact solver was written, no tyre case study exists, and none is needed** — the entire finding is available from the declaration plus the *shape* of one callable.

## 3.1 The fixture

Three two-agent graphs on one `MECH` seam, $\dim M = 8$, conforming, identity prolongations. **Every declaration is identical across the three.** The only difference is agent B's `boundary_response`:

$$\text{smooth}:\ f(t)=At+b \qquad
\text{kink-at-base}:\ f(t)=At+b+c\,\mathrm{relu}(\mathbf w\!\cdot\!t)\,\mathbf w \qquad
\text{kink-at-gap}:\ f(t)=At+b+c\,\mathrm{relu}(\mathbf w\!\cdot\!t-g)\,\mathbf w$$

with $A$ symmetric positive definite (so E7 has nothing to say and cannot confound), $c=0.6$, $g=10^{-3}$.

**Why `relu` is the right stand-in and not a caricature.** Solve the complementarity triple for one of its two variables and a rectifier is what comes out: a penalty normal law is $p=k\max(0,-g)$ exactly, and a regularised stick–slip law is a $\min$ of the same family. What is *not* claimed is that this is contact physics — there is no Hertzian pressure distribution, no $\mu$, no patch. The claim is about **what the nine layers see**, and what they see is a callable.

Both one-sided Jacobians are available in closed form, which is the horizon the measurement needs: $J_-=A$ and $J_+=A+c\,\mathbf w\mathbf w^{\!\top}$, with

$$\frac{\lVert J_+-J_-\rVert_2}{\lVert J_-\rVert_2}=0.261463 .$$

A $26\%$ difference between the two operators the seam has, depending on which side of the switch you stand on.

## 3.2 The compile does not differ

| fixture | verdict | decisions | rules identical to control | $\beta$ | $\kappa$ | $\lVert S\rVert_2$ | passivity defect | stamp E1–E7 |
|---|---|---|---|---|---|---|---|---|
| smooth | `admit-uncertified` | 18 | — | $3.561717$ | $1.226504$ | $4.368459$ | $0$ | `h h h h u u h` |
| kink-at-base | `admit-uncertified` | 18 | **yes** | $3.566386$ | $1.320814$ | $4.710534$ | $0$ | `h h h h u u h` |
| kink-at-gap | `admit-uncertified` | 18 | **yes** | $3.566234$ | $1.306621$ | $4.659715$ | $0$ | `h h h h u u h` |

*Identical* means the full `(layer, rule, verdict, subject)` sequence of all eighteen decisions, in order. $\beta$ spans $0.13\%$ across the three. The envelope stamp is character-for-character the same, **E1 included**.

**This is the admissibility question answered by experiment.** If an inequality-shaped seam reaches the same verdict under the same rules as a linear one, then no layer is asking, and a rule cannot be *widened* to ask — it has to be written.

## 3.3 The residual the framework reports, against the residual it posed

For affine agents $S\lambda-\chi$ is **identically** $\sum_i P_i^{\ast}f_i(P_i\lambda)$ — the port-matching condition itself. With a kinked agent the two part company, and nothing in a run reports the second. Computed at the trace the direct Schur solve returns:

| fixture | reported $\lVert S\lambda-\chi\rVert/\lVert\chi\rVert$ | true $\lVert\sum_i P_i^{\ast}f_i(P_i\lambda)\rVert/\lVert\chi\rVert$ | ratio |
|---|---|---|---|
| smooth **(control)** | $8.020\times10^{-16}$ | $9.659\times10^{-16}$ | $1.204$ |
| kink-at-base | $4.751\times10^{-16}$ | $4.420\times10^{-2}$ | $9.30\times10^{13}$ |
| kink-at-gap | $5.451\times10^{-16}$ | $4.031\times10^{-2}$ | $7.39\times10^{13}$ |

**The framework reports a machine-precision interface solve on a seam whose matching condition is violated at $4\%$ of the trace scale.** On the linear control, run through the identical instrument, the two numbers are the same number. That is the silent-wrongness class stated in the package's own terms — *"no exception, no failed gate, and a number that looks like"* a converged solve — and it is why §6's verdict is `refuse` rather than `admit-uncertified`.

## 3.4 R7's premise, and the step nobody's floor sets

`scheme.RULES["R7"]` reads, in full:

> *the probing method follows differentiable: jvp $\to$ exact JVP; **deterministic and smooth** $\to$ finite differences with the step set by the reproducibility floor; otherwise regularized regression over more probes than modes*

`ExpertCapabilities.probe_route` (`capability.py:532`) is three lines: `has_jvp` $\to$ `jvp`; `deterministic` $\to$ `finite-difference`; else `regression`. **There is no third clause.** A stick–slip transition is deterministic and not smooth, so the rule's stated premise is false and the branch it selects computes a difference quotient across a switch.

Measured, and it is worse than the row predicted in two ways:

- **No decision anywhere cites R7.** Across all three compiles, decisions whose rule contains `R7`: **zero**. `probe_route` is called once, at `probe.py:911`, inside the probe, and its result reaches the record only as the `ProbedBlock.route` string. A rule nothing cites cannot be appealed to and cannot be audited.
- **And the step is not set by the reproducibility floor for anybody.** `probe.py:912` is `step = budget.fd_step or max(1e-2, 100.0 * caps.reproducibility_floor)`. A floor below $10^{-4}$ loses to the constant. Counted over the package: **$0$ of $24$** constructible agents and **$0$ of $19$** resolvable declared floors have their step set by their floor — every declaration in `atlas/cases/` is either `np.finfo(float).eps` or $10^{-6}$, and Poseidon's default is $10^{-5}$. All $24$ take the finite-difference route. The $10^{-2}$ floor is a sensible guard against differencing into noise; what it is not is the mechanism R7's sentence describes.

**This is R10's and W136's shape a third time** — a rule that states a hypothesis it does not check — and it is the first of the three found **before** the physics, from the declaration alone, which is what the row predicted it would be.

## 3.5 What happens when the seam is declared truthfully

`MotionClass.SOLUTION_DEPENDENT` is the only field in the whole schema a contact seam can truthfully set and that any rule reads. Set it and:

| fixture | verdict | refusing rules |
|---|---|---|
| kink-at-gap | `refuse` | `L2/InterfaceMotion`, and nothing else |
| smooth **(control)** | `refuse` | `L2/InterfaceMotion`, and nothing else |

**The control is the whole point.** The refusal a truthfully declared contact graph receives today is about the **patch moving**, it fires identically on a seam with no inequality anywhere in it, and it therefore **cannot** be the framework's answer to W165. Worse, it *masks* the question: an as-declared compile refuses at L2 and never reaches L3, so the admissibility question is not merely unanswered, it is unasked. Drop the motion declaration — which is what rung 8 would do first anyway, since a quasi-static patch at fixed geometry is the cheap starting configuration — and §3.2's table is what you get.

---

# 4. The three predictions W165 foresaw, restated against the numbers

## 4a. The probe measures a chord across a kink — **confirmed, and the row's own discriminator is wrong**

**The prediction.** *"`probe_block` perturbs the trace and reads a linear response; at a stick–slip boundary the response is not differentiable, so the block it returns depends on the probe amplitude in a way no `epsilon` sweep will settle."*

**The positive control, which W165 correctly identified as already in hand.** Every $\varepsilon$ sweep in this vault has come back stable to seven digits. Here, nine decades, $\varepsilon\in[10^{-8},10^0]$, drift measured as $\lVert S(\varepsilon)-S(10^{-8})\rVert_2/\lVert S(10^{-8})\rVert_2$:

| fixture | max drift | digits stable |
|---|---|---|
| smooth **(control)** | $2.493\times10^{-10}$ | **9** |
| kink-at-base | $2.334\times10^{-10}$ | **9** |
| kink-at-gap | $1.372\times10^{-1}$ | **0** |

**The chord is confirmed and the diagnostic is refuted.** At a kink sitting **at** the probe base the sweep is as stable as the smooth control's — to nine digits, over nine decades — and the matrix it returns so stably is $J_+$ to $1.7\times10^{-16}$, while $J_-$ sits $26\%$ away and is never mentioned. The cause is one line of algebra: $\mathrm{relu}$ is **positively homogeneous**, $\mathrm{relu}(\varepsilon x)=\varepsilon\,\mathrm{relu}(x)$, so the one-sided quotient is *exactly* independent of the step. **A kink at the base is invisible to an amplitude sweep, and W165's proposed test would have passed it.**

The offset kink is the case the sweep does see, and it sees it for a reason worth writing down: for $\varepsilon \le g = 10^{-3}$ the returned block is **bit-identical to the linear control** — $\lVert S\rVert = 4.368458912$ and $\beta = 3.561716938$ on both, to the last digit printed — and it crosses at $\varepsilon = 10^{-2}$, which is **the compiler's own default step**. So the default probe straddles a gap of $10^{-3}$ and reports a partly engaged contact, and a more careful probe at a smaller step reports no contact at all, and neither says which.

This is [[epsilon-halo-measurement]]'s finding in a different register: *the proposal's own wording names the wrong test*. Correcting W165's clause is more useful than confirming it, and the corrected statement is:

> **A probe amplitude sweep is not evidence of differentiability.** Stability distinguishes a smooth response from an offset kink and **cannot** distinguish it from a homogeneous one. The instrument that separates all three is a **two-sided** probe that declines where the two sides disagree beyond its own floor — which is **W180**'s done-when, arrived at from the other direction. W180 is a jump by architecture and this is a kink by physics; one instrument covers both, and W180's own `[AI Inference]` said so before this was measured.

## 4b. `L4/E7/passivity` will fire, and correctly — **not measured here, and deliberately not**

**The prediction stands as W165 wrote it.** Coulomb friction is **non-associated**: its slip rule is not the normality rule of its yield surface, so the consistent tangent is genuinely non-symmetric — as *physics*, not as convention. `L4/E7/passivity` reads $\lambda_{\min}$ of $\mathrm{sym}\,S$ and will report a defect, and the defect will be real.

**The contrast case is W138**, where the non-symmetry on `front_wing`'s `wet` seam was an **orientation artefact**: each side reported traction against its own outward normal where the residual is written in a shared one, and naming `effort_normal` took the assembled operator from $\lambda_{\min}=-3.488$ to $+3.469$ — a sign, and the defect went to exactly zero.

**Two operators, both non-symmetric, one repairable by a declaration and one not.** That is the discriminating pair the rule has never had, and W165 is right that most of rung 8's value is there.

**Why nothing was measured for it in this tier, stated so it is not mistaken for an omission.** A synthetic non-symmetric block would confirm only that `_fill_diagnostics` reads the smallest eigenvalue of the symmetric part correctly, which is arithmetic and which `tests/test_probe_and_envelope.py` already covers on a symmetric indefinite block. The pair is informative **only** if the non-symmetry arrives from a real non-associated flow rule, because the question the pair answers is whether anything **other than the analyst's prior knowledge** separates the two cases. A fixture cannot answer that, and building the expert that can is out of this tier's scope by the brief. It is clause 4 of §8's done-when, and it is the clause rung 8 cannot be finished without.

## 4c. R7's finite-difference branch is not licensed — **confirmed, and sharpened**

§3.4 is the evidence. The prediction holds exactly as stated, and two things were found beside it that the row did not foresee: **no decision cites R7 at all**, and **the reproducibility floor sets the step for none of the $19$ declarations in the package**. The first means the defect cannot be audited from a run artifact; the second means R7's sentence describes a mechanism that is inoperative.

**Scope note, because a `max` is not automatically a bug.** The $10^{-2}$ constant is a guard against differencing into arithmetic noise and is defensible on its own terms. What is recorded here is the **mismatch between the rule's sentence and the code**, which is the same class as W136 and R10 — and for contact specifically it matters directly, because $10^{-2}$ is *large* relative to any gap a tyre patch has.

---

# 5. What an inequality seam would need, stated as the missing rule

Before the verdict, the shape of the thing that is missing — because *"a named hole"* is only a useful answer if it names what would fill it.

Write the seam's condition as a **variational inequality** on the interface space $M$: find $\lambda \in K$ with

$$\big\langle S\lambda - \chi,\ \mu - \lambda\big\rangle \;\ge\; 0 \qquad \text{for all } \mu \in K,$$

where $K\subseteq M$ is the **admissible cone** — $\{\lambda:\lambda\ge0\}$ for unilateral normal contact, the Coulomb friction cone $\{(\lambda_N,\lambda_T): \lVert\lambda_T\rVert \le \mu\lambda_N\}$ for the full problem. $K = M$ recovers $S\lambda=\chi$ exactly, which is why the equality case is the special case and not the other way round.

Three consequences, each of which is a thing the framework would have to say:

1. **Existence and uniqueness need a different hypothesis.** For $K$ a closed convex cone, Lions–Stampacchia gives a unique solution when $S$ is **coercive** on $K$ — which for the symmetric case is $\beta > 0$ restricted to $K$, not on all of $M$. So the framework's $\beta$ is the right quantity measured on the **wrong set**, and a rule would have to say which. **For Coulomb friction $K$ is not even convex in the coupled variables**, and uniqueness genuinely fails above a critical $\mu$ — that is a fact about the physics and the compiler is entitled to refuse it, but it must refuse it *saying that*.
2. **The error theory changes character.** [[master-error-bound]]'s $\sigma$ term routes transmission error through an interface solve with $1/\beta$ as the amplifier. Under a VI the corresponding estimate carries a **Céa-type** constant and an extra term for the error in the **active set**, which has no analogue in the equality theory at all.
3. **And the defect is not a norm.** Two traces can be close in $\lVert\cdot\rVert_M$ and sit on opposite sides of the switch. Every defect this framework reports is a norm.

**[AI Inference]:** consequence 3 is the reason the repair is a declaration rather than a measurement, and it generalises past contact. Any seam whose condition has a **discrete** component — an active set, a regime label, a phase — carries information that no interface-space norm can express, and the framework's whole diagnostic vocabulary is norms. Contact is the first instance to arrive; free surfaces, cavitation, phase change and flow-regime transitions are the same shape. Not measured, and none of those is declared anywhere in this vault.

---

# 6. The verdict: a named hole, refused at L3, on a declaration

**Yes, it is a named hole**, by this package's own definition of one (`atlas/holes.py`; `README.md`, *"What is deliberately not filled in"*): the specification left a rule unwritten, the compiler must refuse or decertify, and it must emit **the number the missing rule will constrain**.

Four candidate responses were considered and three are rejected on the package's own rules.

| candidate | verdict |
|---|---|
| **Widen an existing rule** | **Rejected.** §2's enumeration says no rule has the premise. Every rule that reads a declaration reads one that exists and is correct |
| **A new `FailureClass`** | **Rejected — unnecessary.** `FailureClass.SILENT_WRONGNESS` already names it, and §3.3 *measures* that the class is right: machine-precision residual, $4\%$ violation, $10^{13}$ apart. Inventing a fourth class for a case the third already covers would weaken the three |
| **`ADMIT_UNCERTIFIED`** | **Rejected.** The split rule is *refuse the silent-wrongness class, decertify the unverified-hypothesis class*. This is not an unverified hypothesis — the hypothesis is verified **false** and the run proceeds anyway |
| **`refuse` at L3, on a declared condition class** | **Adopted.** §6.1 |

## 6.1 The shape, following C9's precedent exactly

**Why a declaration and not a test — and this is now measured rather than argued.** C9/W66 was added because *both value-based routes were measured and neither discriminates*. The same is true here and the evidence is §4a: at a positively homogeneous kink the amplitude sweep returns the smooth control's nine digits. No test the framework owns separates them, so the framework has to be **told**.

    SeamCondition         EQUALITY (default) | COMPLEMENTARITY
    Connection.condition  SeamCondition = EQUALITY
    L3/C10                refuse COMPLEMENTARITY, citing a missing rule
    NamedHole             SeamCondition, with its required field-5 measurements

Four properties this shape has, and each is the reason to prefer it:

- **Every existing graph is bit-identical.** `EQUALITY` is what all twenty case graphs mean and none of them changes. That is what makes the change cheap, and it is the first thing to assert.
- **The refusal cites a missing declaration turned into a missing rule**, which is the form `TopologyEvent`'s refusal already takes (`compiler.py:567`): *"no rule exists, and the architecture requires the ledger the missing rule will constrain"*.
- **It is checked at L3, where the condition lives.** C5 is the only place the equality is stated and C10 sits beside it. A compile that declares `COMPLEMENTARITY` refuses **before** L4 probes a chord and **before** L5 solves an equation, which is where both silent failures are.
- **And it unmasks the question.** Today a truthfully declared contact graph refuses at `L2/InterfaceMotion` for a reason that fires on a linear control (§3.5). With C10 the graph refuses for the reason it should, and a *quasi-static* patch — no motion declared — refuses where today it admits.

## 6.2 The three numbers the missing rule will constrain

`NamedHole.solve()` raises; it does not guess. What the slot must emit instead:

1. **The branch gap** $\lVert J_+-J_-\rVert_2/\lVert J_-\rVert_2$ at the declared switch — the size of the disagreement between the two operators the seam has. Measured here at $0.261463$ on the fixture; a real contact law supplies it from its own two branches at no extra cost.
2. **The switch rate** — transitions of the active set per macro-step, per seam. This is what E1's stamp is currently asserting is zero, and it is the quantity that says whether the seam is a perturbed equality seam or a genuinely combinatorial one.
3. **The violated-condition residual** — $\lVert\sum_i P_i^{\ast}f_i(P_i\lambda)\rVert$ evaluated at the trace the equality solve returned, which §3.3 measures at $4\times10^{-2}$ against a reported $5\times10^{-16}$. **This is the number that makes the refusal auditable**: it is the exact amount by which the answer the framework would have run with fails the condition it claimed to enforce.

---

# 7. What rung 8's module becomes

W165's `[AI Inference]` — *"if the answer is a named hole then the module's job changes from coupling a tyre to emitting the measurement the missing rule would constrain, which is a different and much smaller module"* — **holds, and this page is the evidence for it.**

**The small module, buildable with no contact expert:**

1. `SeamCondition` + `Connection.condition` + `L3/C10` + the `NamedHole` entry. A schema change with a default; roughly the size of `ResponseHalf`'s (W66).
2. A **two-branch seam declaration**: a switching function $\phi(\lambda)$ and the two one-sided responses, so the probe can return $J_-$ and $J_+$ instead of one chord.
3. `probe_block` returning both blocks at a declared switch, with the gap reported, and **declining** where the two disagree beyond its own floor. Shared with **W180**, which wants the same instrument for a jump by architecture.

**What that module is worth, even though it ends in a refusal.** It converts an `admit-uncertified` that is wrong at $4\%$ into a refusal that says why, on a graph no case study has to build; it clears the masking so the next person to declare a contact seam gets the right rule number; it supplies W180 with its instrument; and it is the only one of rung 8's three findings that is available before an expert exists.

**What it does not do**, and the distinction matters for §8: it does **not** finish rung 8. Clause 4 below needs a real non-associated tangent, and there is nothing in the build repo's `solvers/` that has one — the only match for *contact* is HLLC's contact discontinuity, a gas-dynamics wave.

---

# 8. Done-when for rung 8, written so it cannot be quietly declared finished

Rung 8 is done when **all five** hold. The fifth exists because the first four can all be true while the ladder's own gate has silently been replaced by an easier one.

1. **A seam can say its condition is not an equation, and saying so refuses.** `Connection` carries a declared condition class; a graph declaring `COMPLEMENTARITY` is refused at L3 with a rule number and the quantity that would otherwise have been silently wrong. The refusal names §6.2's three numbers and emits them.
2. **Every existing graph is unchanged.** All twenty graphs in `atlas/cases/` compile to a byte-identical artifact against their pre-change output, and a test asserts it. A schema change that moves an unrelated verdict has not been scoped, it has been guessed.
3. **The probe returns both branches, or declines.** At a declared switch `probe_block` returns $J_-$ and $J_+$ with the gap, and declines — naming the amplitude — wherever the two disagree beyond the probe's own reproducibility floor. **W180 closes on the same instrument or this clause is not met**: two rows asking for one probe and getting two is the drift this vault keeps catching.
4. **The passivity pair is run and is discriminating.** `L4/E7/passivity` is measured on a **genuinely non-associated** friction tangent and on W138's orientation artefact, through one unmodified instrument, and something **other than the analyst's knowledge of which is which** separates them. If nothing does, that is the finding and it is recorded as one — a rule that fires identically on a repairable convention error and on irreducible physics is a rule whose output cannot be acted on, which is worth knowing and is the reason W165 priced this clause as most of the rung's value.
5. **The gate is met or explicitly replaced, in writing, on both pages.** [[f1-pathmap-and-end-goal]] §3's rung-8 gate is *"load transfer; temperature-dependent grip"*. **A refusal cannot meet a gate phrased as an accuracy target.** So either that gate is met with a running contact coupling, or it is replaced — and the replacement is written on this page, on [[case-study-ladder-to-f1]] and in §3.3's table of the pathmap, with the reason. Rung 8 may **not** be marked done against a gate nobody restated.

> **The clause that is doing the work is 5.** Clauses 1 to 3 are a tier of work and are honestly completable without a tyre. Clause 4 is not, and a rung marked *done* on 1–3 would be the ladder's own failure mode — [[f1-pathmap-and-end-goal]] §3: *"a rung is complete when its gate passes; rungs are not skipped, and a failed gate stops the ladder rather than being worked around."*

---

# 9. What this page does NOT claim

- **It does not claim contact is uncoupleable.** Complementarity coupling is standard in multibody and computational contact mechanics; semi-smooth Newton and active-set methods solve these problems routinely. The claim is about **this compiler**, which has no such branch and no declaration that would select one.
- **It does not claim the fixture is contact physics.** `relu` is the shape a complementarity condition takes once it is solved for one variable. There is no $\mu$, no patch, no Hertzian distribution, no stick–slip iteration anywhere in this tier.
- **It does not measure prediction 4b**, and §4b says why in full. The passivity pair needs an expert this vault does not have.
- **It does not claim R7's $10^{-2}$ step floor is a defect.** It records that R7's *sentence* and `probe.py`'s *code* describe different mechanisms and that the sentence's one is inoperative for every declaration in the package.
- **It does not settle L7.** R9's integration across a switch is enumerated in §2 and argued, not measured. It needs a two-branch response and a multirate seam, which is the small module's step 2 plus one clock ratio, and it was not done here.
- **And §5's generalisation past contact is an `[AI Inference]`** and is labelled as one in place. Free surfaces, cavitation and phase change are named as the same shape; none of them is declared anywhere in this vault and none was measured.

---

## See Also

- [[gap-worklist]] — W165 (the question and its budget), W181 (R7's premise), W182 (the named hole), W180 (the shared two-sided probe), W138 (the orientation artefact §4b contrasts against)
- [[case-study-ladder-to-f1]] — §14.1's rung table and §15.3's critical path, where rung 8 sits
- [[f1-pathmap-and-end-goal]] — §3's rung-8 gate, which §8 clause 5 binds
- [[port-algebra-atlas-0.1]] — §3.1's `MECH` bond, which covers the variables; §4's per-port residual, which §2 shows reads as a fault at an open patch
- [[end-to-end-architecture-spec]] — §5, the amended connection rule C1–C9 that C10 would join
- [[atlas-implementation]] — the nine layers as built, and the three-verdict split §6 applies
- [[theory-closure-audit]] — §6's argument for settling the architecture on a page before settling it in a module, which is why this page exists before rung 8's
- [[master-error-bound]] — §4's $\sigma$ and the $1/\beta$ amplifier, whose hypothesis §5 says changes under a variational inequality
- [[probed-dtn-coupling]] — the probe whose chord §4a measures
- [[epsilon-halo-measurement]] — the other row whose own proposed test turned out to name the wrong quantity
- [[tier0-measurements]] — *every diagnostic healthy while the scheme solved the wrong equation*, which §3.2 reproduces on a fixture
