# Wind-Farm Case Study — Results, Phases W0–W3

**Type:** Results page (folder: `Atlas 0.1/case-study-wind-farm-wake/implementation/`)
**Status:** W0–W3 complete, 2026-08-20. W4–W6 not started.
**Code:** branch **`atlas-0.1-windfarm`** of [github.com/nonidino/physics-foundation-model](https://github.com/nonidino/physics-foundation-model), commit `2645cd0`. Full build narrative and every number in context: [[wind-farm-implementation-log]].
**Related:** [[impl-wind-farm-guide]] (the phases and gates), [[spec-wind-farm-wake-atlas-0.1]] (binding numbers), [[port-algebra-atlas-0.1]], [[f1-pathmap-and-end-goal]], [[prior-art-and-novelty-atlas-0.1]]

> **What this page is for.** The log records how each number was arrived at; this page records *what the numbers are* and what may and may not be claimed from them. Read §5 before quoting anything.

---

# 1. The headline

**The go/no-go passed.** A frozen, independently pretrained incompressible-Navier–Stokes checkpoint, handed a body force it never saw in training, keeps the streamwise momentum budget closed across a rotor to within $\mathbf{2.6\times10^{-3}}$ of the thrust an exact actuator-disk expert computed — against a $5\%$ threshold, teacher-forced, **before any projection**, and at the same level a purpose-built spectral Navier–Stokes solver achieves on the identical problem.

**Falsification criterion F2 of [[f1-pathmap-and-end-goal]] is not triggered.** Frozen experts, at least this pair, at least at one interface, compose without fine-tuning.

Two qualifications sit on that sentence and neither is optional: **the gate is weak** (§5.1), and **the same checkpoint cannot survive the rollout the spec asks for** (§5.2).

---

# 2. Gate results

| Gate | Criterion | Result | Status |
|---|---|---|---|
| **W0** ingredients | checkpoint loads; unforced rollout stable to $t=60$; decoder mode logged; native $\Delta t$ recorded | loads (20.77 M params, 0.21 s/forward, CPU); rollout completes 1200 steps with mean flow held exactly, **but fluctuation energy grows past its initial value at $t=42.5$ and reaches $2.47\times$ at $t=60$** | **conditional pass** — envelope in §5.2 |
| **W1** geometry | disjoint; curves on both boundaries; pairs $=$ declared; diameter 4 | partition exact *pointwise*; areas $192.0$ vs. $192.0$; $15=15$ pairs; normals verified $\text{src}\to\text{dst}$ everywhere; tokens match the spec table exactly (1144); **diameter is 3, not 4** | **pass** (+2 spec corrections) |
| **W2** disk expert | $C_P^{\max}=16/27$ at $a=1/3$ to machine precision | $<5\times10^{-16}$, and verified to be the maximum by scan; $C_T'=2$ to the same; $\int\mathbf f=-T$ to $10^{-13}$ on misaligned lattices | **pass** |
| **W3** go/no-go | $r_T<5\%$ pre-projection under R1 **or** R2 | $r_T = 6.2\times10^{-4}$ (R1), $2.6\times10^{-3}$ (R2) — **both** | **pass** |

---

# 3. The W3 measurement in full

$$r_T=\frac{\bigl|\,T_{\text{disk}}-|\Phi_x(V_1)|\,\bigr|}{T_{\text{disk}}},\qquad \Phi_x(V)=\oint_{\partial V}\Bigl[(\mathbf u\!\cdot\!\mathbf n)u_x+p\,n_x-\tfrac{1}{\mathrm{Re}_{\text{eff}}}\partial_nu_x\Bigr]ds$$

measured in the unsteady form $\dot M_V+\Phi_x(V)=\int_V f_x$, which holds at every instant rather than only once settled.

| variant | band | reference solver | **expert, teacher-forced** | expert, cold start |
|---|---|---|---|---|
| **R1** Schwarz halo | $0.5\,D$ | $8.9\times10^{-4}$ | $\mathbf{6.2\times10^{-4}}$ | $6.1\times10^{-5}$ |
| **R2** flux-BC tokens | $0.0625\,D$ (one expert patch) | $1.1\times10^{-3}$ | $\mathbf{2.6\times10^{-3}}$ | $2.8\times10^{-3}$ |

Field agreement at the **cold-start converged state** — the expert released from uniform flow to find its own fixed point over 300 steps, with only the band. Not constrained by $r_T$, and the reason the pass is meaningful:

| | R1 | R2 |
|---|---|---|
| $\langle U_d\rangle$ relative error | $0.80\%$ | $1.07\%$ |
| thrust relative error | $1.59\%$ | $2.14\%$ |
| field rel-$L_2$ (normalized by the **perturbation**, not the field) | $0.21$ | $0.18$ |

**The cold-start column is the stronger evidence and should be quoted in preference to the teacher-forced one.** Teacher forcing hands the expert the reference's converged state, so the expert can only be observed relaxing *away* from a fixed point that is not its own; at 60 steps ($t=3$, roughly one window transit) it is mid-transient, $\langle U_d\rangle$ having excursed to $0.858$ (R1) and $0.682$ (R2). The cold start asks the harder question — released from uniform flow with nothing but the band, where does this expert's own steady state sit? — and the answer is **within $1\%$ of a real Navier–Stokes solver's**. No part of that is inherited from an answer it was handed.

**Both variants passing is the outcome [[impl-wind-farm-guide]] §4.2 calls "the declared-edge contract is validated. Proceed, use R2."** Mechanism A of [[edge-generation-atlas-0.1]] needs no change; halo exchange is not required. **R2 adopted.**

## 3.3 The band sweep — the cliff is at $\text{band} = U_\infty\Delta t$

R1 and R2 are two points on one axis: how much of the neighbour's field is written into the upstream band. Sweeping it at the spec's $\Delta t=0.05$, 65 min of CPU:

| band | cells | reference $r_T$ | expert $r_T$ (cold) | reference $\langle U_d\rangle$ | expert $\langle U_d\rangle$ | error |
|---|---|---|---|---|---|---|
| $0.0625\,D$ **(R2)** | 4 | $1.1\times10^{-3}$ | $2.8\times10^{-3}$ | 0.7641 | 0.7559 | $1.07\%$ |
| $0.125\,D$ | 8 | $2.7\times10^{-3}$ | $1.8\times10^{-3}$ | 0.8169 | 0.8071 | $1.21\%$ |
| $0.25\,D$ | 16 | $2.4\times10^{-3}$ | $7.7\times10^{-4}$ | 0.8541 | 0.8454 | $1.02\%$ |
| $0.5\,D$ **(R1)** | 32 | $8.9\times10^{-4}$ | $6.1\times10^{-5}$ | 0.8876 | 0.8806 | $0.80\%$ |
| $0.75\,D$ | 48 | $9.8\times10^{-6}$ | $2.1\times10^{-4}$ | 0.9229 | 0.9206 | $0.25\%$ |

Read alone this says a wide halo helps and a thin one does not break — flat to $\sim1\%$ across a twelvefold range, no threshold. **That reading was wrong, and repeating the sweep at $\Delta t=0.25$ shows why** (2026-08-21):

| band | cells | $\text{band}/(U_\infty\Delta t)$ | $\langle U_d\rangle$ error at $\Delta t=0.25$ | at $\Delta t=0.05$ |
|---|---|---|---|---|
| $0.0625\,D$ | 4 | $0.25$ | $\mathbf{19.6\%}$ | $1.07\%$ |
| $0.125\,D$ | 8 | $0.5$ | $2.27\%$ | $1.21\%$ |
| $0.25\,D$ | 16 | $\mathbf{1.0}$ | $0.427\%$ | $1.02\%$ |
| $0.5\,D$ | 32 | $2.0$ | $0.140\%$ | $0.80\%$ |
| $0.75\,D$ | 48 | $3.0$ | $0.197\%$ | $0.25\%$ |

**There is a cliff, and the first sweep simply never crossed it.** At $\Delta t=0.05$ the flow advects $0.05\,D$ per macro-step, so even the four-cell band is $1.25\times U_\infty\Delta t$ — every point sampled sat on the safe side. The governing quantity is not the band width but its ratio to the distance the flow crosses in one coupling step:

$$\boxed{\;\text{band width}\;\ge\;U_\infty\,\Delta t_{\text{macro}}\;}$$

Below the knee the band is overrun within a single step — material advects clean through the interface layer before it can constrain the inflow — and the interface stops transmitting. At $\Delta t=0.25$ a four-cell band is overrun fourfold and $\langle U_d\rangle$ is wrong by $19.6\%$; at ratio $1.0$ the error is back to $0.43\%$.

**What this changes.** Mechanism A of [[edge-generation-atlas-0.1]] survives — the interface still carries only declared port values, not neighbour interior, and R2 remains the adopted variant. What does not survive is the claim that *four cells* is enough as a fixed number. The minimum band is a function of the coupling step, and W5 must size it from the rule rather than inherit $0.0625\,D$ from W3. **[AI Inference]:** this is a Courant condition on the *interface* rather than on the solver, and it appears to be a general property of writing boundary data into the state of an operator that accepts no boundary conditions — so it should apply to every later Atlas case study using the same edge mechanism, not only to this one.

**One confound, stated because it forbids the obvious reading.** The reference's own $\langle U_d\rangle$ moves from $0.764$ to $0.923$ across the sweep: a wider band pins more of the window to uniform inflow, which reduces the effective blockage. **The band width changes the problem, not only its numerical treatment**, so the $r_T$ column may not be compared down its length — the only valid cross-band comparison is expert against reference *at the same band*, which is the error column. The two entries at $0.125\,D$ in the first sweep are a duplicate configuration and agree to every digit printed, which is the determinism check the sweep gets for free.

## 3.1 What had to be built before the number existed

Four properties of the checkpoint made the guide's W3 as written non-executable, and each was resolved by construction rather than approximation:

1. **No forcing input.** Poseidon maps velocity to velocity; forced Navier–Stokes appears in its paper only as a downstream *finetuning* task. The force enters by Strang splitting — half impulse, one expert step, half impulse, Leray projection after each. Over the periodic window the applied impulse is exact, because the projection touches every Fourier mode except $k=0$ and the total force *is* the $k=0$ mode.
2. **No pressure.** The incompressible datasets fill the density and pressure channels with the constants $1$ and $0$, so the checkpoint predicts no pressure — and $\Phi_x$ needs $p$ on the control-volume faces. Recovered from $\nabla^2p=\nabla\!\cdot\!\mathbf f-\nabla\!\cdot\!((\mathbf u\!\cdot\!\nabla)\mathbf u)$, spectrally; matches the analytic Taylor–Green pressure to $2\times10^{-15}$. A derivation, not a model.
3. **A periodic pressure cannot carry a mean gradient.** With the disk force alone, the budget was short by exactly $\langle f_x\rangle V_{\text{slab}}$ — a steady $30.4\%$ residual at a converged state with $\dot M=10^{-12}$, so not a transient and not the expert. Driving the window with a uniform $+T/A_\Omega$ so the *total* force has zero mean is the standard periodic wind-farm LES construction (Calaf, Meyers & Meneveau 2010); the residual falls to $10^{-3}$.
4. **Teacher forcing is an exchange, not a frozen thrust.** Each side takes the other's value once per step, no iteration. Freezing $T$ makes the turbine blind to its own inflow — precisely the failure [[spec-wind-farm-wake-atlas-0.1]] §4.2 forbids for turbine 2 — and, when the wake recirculates, runs away and reverses the flow.

## 3.2 The reference solver

A pseudo-spectral vorticity–streamfunction solver (~100 lines) drives the **identical** probe code, so a difference between it and the expert is a difference between them. It was not in the plan and earned its place three times: it caught the frozen-thrust runaway, it exposed the missing mean pressure gradient at a state where the residual could not be blamed on a transient, and it establishes the floor $r_T$ the expert is judged against.

This is [[impl-wind-farm-guide]] §5's argument arriving a phase early — build the metric before the system that produces numbers nobody can check.

**[AI Inference]:** the same solver is most of what W9's monolithic baseline needs. Running it over the undivided domain now looks cheap rather than optional, which would move [[impl-wind-farm-guide]] §7.3 from "do not skip" to "already have".

---

# 4. What W0 found out about the checkpoint

The three that change how it must be used.

## 4.1 It destroys a uniform flow — and the fix is a change of frame

A spatially uniform flow is an **exact steady solution** of the periodic incompressible equations at any viscosity. Fed $\mathbf u=(1,0)$, the checkpoint returns $\bar u=0.969$ after one step and $-0.083$ after 1200. Its pretraining set (NS-Sines) is zero-mean by construction; a wind farm is nothing but mean flow.

For constant $\mathbf U$, $\;\mathbf u(\mathbf x,t)=\mathbf U+\mathbf u'(\mathbf x-\mathbf Ut,\,t)$ with $\mathbf u'$ solving the same equations at zero mean. Exact — not a first-order splitting — because Navier–Stokes is Galilean invariant and $\mathbf U$ does not vary in space. The adapter subtracts the window mean, advances the fluctuation, re-zeroes its mean (recording the drift the expert attempted, since for unforced periodic flow the mean is conserved exactly and any drift is model error), translates by $\mathbf U\Delta t$ — *which is the mean advection term* — and adds $\mathbf U$ back. One-step error $3.3\%\to0.55\%$; $\bar u$ after 1200 steps $\to1.000$ exactly.

**[AI Inference]:** the translation in step 4 is where interface data enters once a window stops being periodic, because it pulls in upstream material that in the coupled system belongs to the neighbour. If that holds at W5, the Galilean split is not a separate mechanism sitting beside the edge layer — it is where the edge layer acts.

## 4.2 Its effective viscosity is not a number

Fitting an exact Taylor–Green decay across the whole trained lead range, regressing $\ln(\text{ratio})$ on lead so a fixed per-step bias separates from real decay:

| $\lambda$ | $\nu$ | $r^2$ | decay over the lead range |
|---|---|---|---|
| $0.500\,D$ | $1.8\times10^{-5}$ | 0.27 | $+0.001$ |
| $0.250\,D$ | $-1.4\times10^{-5}$ | 0.92 | $-0.010$ |
| $0.125\,D$ | $4.9\times10^{-4}$ | **0.998** | $+0.685$ |

Above the cutoff there is no decay distinguishable from the $\sim2\%$ per-step bias; below it decay is clean and exponential. A Newtonian fluid gives the same $\nu$ at every $k$. **The checkpoint behaves like an LES with a spectral cutoff**, and any $\mathrm{Re}_{\text{eff}}$ quoted for it describes that cutoff rather than a fluid property.

## 4.3 The window size is not a free parameter

With $x=Lx_p$, $u=U_su_p$, $t=T_st_p$ and $T_s=L/U_s$, substituting into the incompressible equations gives $\nu_{\text{ours}}=\nu_pLU_s$ and hence

$$\boxed{\;\mathrm{Re}_{\text{eff}}=\frac{T_s}{\nu_p\,L^2}\;}$$

Requiring one macro-step to be one *native* expert step fixes $T_s$; requiring the freestream to land inside the expert's velocity distribution fixes $U_s$; $L$ follows. **$\mathrm{Re}_{\text{eff}}$ is then not a dial** — and doubling the physical size one window covers *quarters* it. Covering $\Omega$ with fewer, larger windows is not a performance tuning knob; it changes the physics being solved.

**[AI Inference]:** this looks like a general property of composing a fixed-resolution pretrained operator, not a quirk of Poseidon. If so it belongs in [[expert-library-atlas-0.1]] as a constraint on how finely a domain must be cut, alongside the governing-family cut rule — the two are independent and both binding.

---

# 5. Reporting rules, applied

Per [[impl-wind-farm-guide]] §9 and [[conservation-as-constraint-atlas-0.1]]'s enforce-or-measure rule.

## 5.1 The W3 gate is necessary and weak

Run against null models at the full horizon:

| control | $r_T$ | passes $5\%$? | field rel-$L_2$ | $\langle U_d\rangle$ error |
|---|---|---|---|---|
| identity (returns its input) | $3.1\times10^{-1}$ | no | 2.16 | $88.6\%$ |
| **rigid advection** — translation at the mean; no pressure, no viscosity, no dynamics | $9.5\times10^{-3}$ | **yes** | 0.26 | $3.0\%$ |
| reference $+\,5\%$ divergence-free noise | $4.6\times10^{-3}$ | **yes** | 0.21 | $0.02\%$ |

Run again in the **exact teacher-forced configuration the headline number comes from** — started at the reference's converged state, band filled from it:

| control | R1 $r_T$ | R1 $\langle U_d\rangle$ error | R2 $r_T$ | R2 $\langle U_d\rangle$ error |
|---|---|---|---|---|
| identity | $6.1\times10^{-2}$ | $64.7\%$ | $2.2\times10^{-1}$ | $62.2\%$ |
| rigid advection | $7.5\times10^{-3}$ | $2.0\%$ | $1.1\times10^{-3}$ | $8.2\%$ |

**The identity fails the gate in both variants**, which settles the obvious objection to a teacher-forced measurement: the headline $r_T$ is not achievable by simply holding the state the expert was handed. Rigid advection still passes.

At this horizon $r_T<5\%$ rejects only a do-nothing operator. **It constrains the momentum budget, not the flow.** The frozen expert beats rigid advection on every measure — $r_T$ $2.6\times10^{-3}$ vs. $9.5\times10^{-3}$, $\langle U_d\rangle$ error $1.1\%$ vs. $3.0\%$, field rel-$L_2$ $0.18$ vs. $0.26$ — and that comparison, not the bare $r_T$, is what the W3 pass rests on. Asserted as a regression test so a future change that makes the gate discriminating surfaces as a failure to re-examine.

## 5.2 The spec's rollout cannot be run at the spec's timestep

Unforced, closed periodic window, at the W3/W5 scaling ($L=2\,D$), to $t=60$:

| lead | $\Delta t_{\text{macro}}$ | calls | $E'/E'_0$ at $t=60$ | peak $E'/E'_0$ | wake retained at $t=7$ |
|---|---|---|---|---|---|
| $0.10$ (native) | $0.05$ **(spec)** | 1200 | $1.79$ | $1.79$ | $20.0\%$ |
| $0.20$ | $0.10$ | 600 | $13.71$ | $14.74$ | $75.7\%$ |
| $0.35$ | $0.175$ | 343 | $4.04$ | $4.04$ | — |
| $\mathbf{0.50}$ | $\mathbf{0.25}$ | 240 | $\mathbf{0.086}$ | $\mathbf{1.000}$ | $\mathbf{68.2\%}$ |

**Lead $0.5$ is the only drift-free choice**: peak ratio exactly $1.000$ means the fluctuation energy never once exceeds its initial value — monotone decay, which is what an unforced 2D flow must do. It is also $5\times$ cheaper than the spec's step, and it keeps $68\%$ of a wake against an analytic $78.5\%$.

Lead $0.2$ has the best wake retention of any step tested, and $160\times$ the drift. It is a trap: the quantity it optimizes is the one measured over $t=7$, and the quantity it ruins only shows up by $t=60$.

**This inverts [[impl-wind-farm-guide]] §7.2's warning in an interesting way.** The guide says a smaller communication step is unavailable as a remedy because the expert has a native $\Delta t$. Measured: going *below* native is indeed bad (composing two half-native steps misses one native step by $18.5\%$, against $2.0\%$ for composing two native steps) — **and stepping *at* native is unstable while stepping at $5\times$ native is not.** The frozen expert has a *minimum viable coupling step well above its nominal native lead*, and the two are different numbers. Neither is documented anywhere in the checkpoint's own materials.

**Adopted: $\Delta t_{\text{macro}}=0.25$, lead $0.5$**, which by §3.3's rule requires a band of at least $0.25\,D$ (16 cells), not W3's four. Re-running the W3 probe at that step over the identical physical horizon keeps the gate: $r_T=1.1\times10^{-2}$ at a $0.5\,D$ band with $\langle U_d\rangle$ within $0.140\%$ of the reference — *better* field agreement than at $\Delta t=0.05$. And the null controls that made the W3 gate weak (§5.1) now all fail it, so the gate becomes discriminating at the adopted step rather than merely satisfied.

## 5.2b Neither checkpoint can carry a wake unaided — and Poseidon-B is worse

Gate W0 asks for a stable rollout; it does not ask whether anything *survives* the rollout. The case study needs a velocity deficit to reach turbine 2 at $7\,D$, i.e. to survive $t=7$. A Gaussian deficit uniform in $x$ is the right probe because it has a closed-form answer: for any parallel flow $\mathbf u=(U(y),0)$ the nonlinear term $(\mathbf u\!\cdot\!\nabla)\mathbf u = u_x\partial_xu$ vanishes identically and the pressure gradient with it, so the exact evolution is pure diffusion,

$$\sigma^2(t)=\sigma_0^2+2\nu t,\qquad A(t)=A_0\,\sigma_0/\sqrt{\sigma_0^2+2\nu t},$$

which at $\nu=10^{-3}$ ($\mathrm{Re}=1000$, the bottom of the spec's band, hence the *slowest* defensible recovery) retains $78.5\%$ at $t=7$.

| at $t=7$, $\Delta t=0.05$ | Poseidon-T | Poseidon-B | analytic |
|---|---|---|---|
| wake retained | $20.0\%$ | $\mathbf{-0.5\%}$ | $78.5\%$ |
| half-life | $1.75$ | $0.05$ (one step) | $33.8$ |
| spurious / signal | $1.33$ | $1.00$ | $0$ |

**The no-wake control is what makes these numbers mean anything.** Run with no deficit at all, both checkpoints still develop a profile — depth $0.209$ for Poseidon-T and $0.095$ for Poseidon-B by $t=10$ — because the expert emits spurious structure on every call and it accumulates on a closed window. Peak-to-trough depth therefore measures signal plus noise, and for Poseidon-T at $t=10$ the noise alone *exceeded* the wake. Retention is therefore reported as the projection of the profile onto the initial wake shape, the same device §4.2 uses for Taylor–Green modes. Poseidon-B's $-0.5\%$ says its entire residual is its own invention: control depth $0.0946$ against $0.0950$ with a wake, a difference of $0.4\%$ of the initial deficit.

At the adopted $\Delta t=0.25$ Poseidon-T retains $68.2\%$, close to the analytic $78.5\%$. **The dissipation is per *call*, not per unit time** — the same finding as the drift — so a coarser coupling step is simultaneously the fix for stability and the fix for wake survival.

**None of this is a wake result** in the sense of §5.3; it is a statement about what the operator does to a known analytic solution on a closed window. In the coupled system the upstream band resupplies the deficit every step, so this is the worst case rather than the operating case. It is recorded because a checkpoint that halves a wake every $1.75$ convective times cannot produce a defensible W8 or W10 no matter how the coupling is arranged.

## 5.3 Nothing here is a wake result

The probe window is $2\,D$ square, giving $50\%$ blockage against the spec's $8\,D$-tall domain. That is affordable *only because $r_T$ is an identity and holds at any blockage*. **No wake-recovery, array-efficiency or deficit number may be quoted from W0–W3** — those are W8 and W10, and under [[case-study-wind-farm-wake-2d-atlas-0.1]]'s 2D discount they are corridors, not predictions, even then.

## 5.4 Conservation claims

Of the fifteen declared ports, **zero are yet enforced and zero are yet measured** — W3 exercised one interface in isolation and the per-port residual machinery is W4. The thrust projection of [[spec-wind-farm-wake-atlas-0.1]] §7.3 does not exist yet, which is why every W3 number is pre-projection by construction rather than by discipline. Mass is at **level 4**, not the spec's level 5: scOT has no stream-function head, so cross-interface mass conservation is a spectral Leray projection (exact and cheap *on a periodic window only*) rather than agreement on $\psi$ at interface endpoints.

> **Superseded in part, 2026-08-22.** The parenthesis above — *on a periodic window only* — turned out to be the whole W5 story, and taking it seriously changes the mass claim. The coupled domain now solves the pressure with the boundary conditions it actually has (Neumann inlet and slip walls, **Dirichlet outlet**), diagonalized exactly by a cosine transform, and the assembled field is divergence free to $5.6\times10^{-14}$ over the whole domain including its boundary cells. **Mass conservation reaches the spec's $10^{-8}$ on the velocity-decoder path**, by a route the spec did not anticipate, and without a stream function. What is *not* superseded is the level-5 language: this is not agreement on $\psi$ at interface endpoints, and C2 remains unenforced. See [[wind-farm-implementation-log]], entry 2026-08-22.

## 5.5 Integration cost

Integrating the *first* frozen expert cost roughly four times what the zero-parameter closed-form expert cost, and essentially all of that was reverse-engineering an undocumented interface — channel layout, asymmetric $u/v$ normalization, trained lead times, the mean-flow failure, the dissipation cutoff — rather than writing coupling code. Falsification criterion **F3** predicts the second frozen expert is much cheaper, since the port and adapter machinery now exists. Full ledger in [[wind-farm-implementation-log]].

---

# 6. Corrections banked

| Where | What | Consequence |
|---|---|---|
| [[spec-wind-farm-wake-atlas-0.1]] §5.1 | Graph diameter is **3**, not 4 — the quoted path $I\to R_1\to N\to F\to R_2$ is not a geodesic | Documentation only. $n_{\text{mp}}=4$ kept as conservative |
| [[spec-wind-farm-wake-atlas-0.1]] §3 | The rotor notch is **sub-token** at the spec's own $0.25$ spacing, so a boolean cell-centre mask loses it entirely and double-counts the rotor | Agent masks carry exact fractional occupancy; the discrete coverage identity holds |
| [[spec-wind-farm-wake-atlas-0.1]] §7.2 | Level-5 mass conservation is **unavailable** with this checkpoint | Level 4, and the periodic-window discount must not be inherited by W5 |
| [[impl-wind-farm-guide]] §4.1 | "Give the fluid expert the disk's $\mathbf f_{\text{disk}}$" presumes a forcing input the expert does not have | Operator splitting, logged as a design decision |
| Own code | `disk.py` indexed $[x,y]$ while the adapter used $[y,x]$ — both self-consistent, mutually transposed, and **every W2 test passed** | Per-component gates cannot catch cross-component conventions; the first integration is where they appear |

---

# 7. One unresolved anomaly

In the **no-band periodic configuration** — no interface at all, an infinite streamwise array at $2\,D$ spacing — the slab balance does not close, leaving a residual of order $T$ that persists at $\dot M\sim10^{-7}$ and flips sign as the control volume grows. The same operator closes to $10^{-16}$ unforced and to $10^{-3}$ in both R1 and R2.

Working hypothesis, **untested**: the split scheme's fixed point is not a fixed point of the continuous equations, and the band prevents the offset accumulating. No number from that configuration is quoted anywhere on this page. It is not on the W3 path, but it is on W5's, where several agents will run without an upstream band.

---

# 8. What W4 inherits

- **R2 adopted**; R1 kept as the positive control. Its band must be sized by §3.3's rule, $\text{band}\ge U_\infty\Delta t$ — at the adopted step that is $0.25\,D$ (16 cells), not W3's four.
- **$\Delta t_{\text{macro}}=0.25$ (lead $0.5$) is the adopted coupling step**, §5.2 — the only drift-free choice, $5\times$ cheaper than the spec's, and the one at which the W3 gate becomes discriminating.
- **Poseidon-T stays**; Poseidon-B was evaluated and rejected (§5.2b) — it removes the drift by annihilating the flow, retaining $-0.5\%$ of a wake for $1.6\times$ the compute.
- **Wake survival is the binding constraint on W8/W10**, not interface consistency. §5.2b is the number to beat, and the reference solver is what it must be measured against.
- The reference solver exists and should be promoted to the W9 monolithic baseline.
- $\mathrm{Re}_{\text{eff}}$ is a cutoff description; the window-size constraint of §4.3 means the eight agents cannot be tiled at an arbitrary window size.
- Per-port residuals, the global power residual $\mathcal R(t)$, and the wake diagnostics are all still to be built — and per [[impl-wind-farm-guide]] §5, validated on cases with known answers **before** the coupled system produces numbers nobody can check. §5.1 above is the argument for taking that instruction literally.

---

## See Also

- [[wind-farm-implementation-log]] — the build narrative, every number in context, the integration-hours ledger
- [[impl-wind-farm-guide]] — phases W0–W6 and gates W0–W11
- [[spec-wind-farm-wake-atlas-0.1]] — the binding spec these results are measured against
- [[f1-pathmap-and-end-goal]] — F2 (composability without fine-tuning) is what W3 tested
- [[prior-art-and-novelty-atlas-0.1]] — §2.1 predicted the fixed-point convergence question; §5.2 here is the first measurement bearing on it
- [[edge-generation-atlas-0.1]] — Mechanism A survives W3 unchanged
- [[poseidon-pde-foundation-model]] — the checkpoint
- [[conservation-as-constraint-atlas-0.1]] — the enforce-or-measure rule these claims are graded against; C1 moved from measured to enforced on 2026-08-22, C2 did not
