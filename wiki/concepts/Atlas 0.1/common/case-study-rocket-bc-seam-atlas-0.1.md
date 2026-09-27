# Rocket b–c Seam — the first rocket physics through Atlas, and the wall that was never listening

**Type:** Concept page — **case-study summary** (folder: `Atlas 0.1/common/`)
**Status:** opened 2026-09-17. **CS-21**, and the first rung (R0) of PoC 4. Every number here is quoted from `out/w300.json` and `out/w302.json`; nothing is estimated.
**On timings.** The machine ran at 1400 MHz of a 3800 MHz maximum throughout, and **the power state changed during the session** — battery at 46% when the first arms ran, AC at 98% by the last. So every duration on this page spans two power states and none of them is a property of the code. **Costs are quoted in solver calls and CFL sub-steps**, which are deterministic; the seconds are commentary, and §7.1 measures how far they drift.
**Code:** `atlas/cases/rocket_experts.py` (the two real experts), `atlas/cases/rocket.py` (the graph, three declaration levels), `scripts/w300_rocket_bc_seam.py` (R0's gates), `scripts/w302_declaration_ledger.py` (R1's ledger), `scripts/w310_rocket_r0_complete.py` (R0's completion), `scripts/w312_plane_seam_converged.py`, `tests/test_tier76_rocket_experts.py`, `tests/test_tier79_r0_complete.py`
**Build repo pinned at `0a407b7`**, branch `atlas-0.1-windfarm`. It is a second checkout and it can drift; a number quoted from here carries the commit or it carries nothing.
**Related:** [[case-study-thermal-strain-atlas-0.1]] · [[gap-worklist]] · [[probed-dtn-coupling]] · [[port-algebra-atlas-0.1]] · [[tier0-measurements]] · [[interface-transfer-theory]] · [[end-to-end-architecture-spec]] · [[composition-error-theory]] · [[master-error-bound]] · [[expert-library-atlas-0.1]] · [[generalization-requirements]] · [[case-study-ladder-to-f1]]

---

## 1. The question

`atlas/cases/rocket.py` has existed since the architecture spec was written, and it has always been a **declaration exercise**. Every agent's `boundary_response` was `capability.linear_response` on a seeded random matrix, so the graph's seven agents, seven typed edges and 49 decertifications were all statements about a record that nothing physical backed. **No rocket physics had ever been run through Atlas.**

Meanwhile the physics existed and passed its own tests: `compressible2d.py` (MUSCL + minmod + HLLC under SSP-RK2), `thermostruct2d.py` (backward-Euler Q1 conduction plus quasi-static plane stress), and five more modules, 54 tests green in the build repo. The two halves had never been connected.

So the question is narrow and it is the whole of R0's first rung:

> **Wire one seam to the real solvers, probe it, and find out what the probe says — including whatever it says that nobody wanted to hear.**

The answer to the narrow question is that the seam probes cleanly — a finite $\beta$, a null space of the declared dimension, and an operator content $\omega$ — but **the first $\beta$ this page published was wrong twice over, and §10 is the correction**. The answer to the wider question is **W301**, and between them they are the reason this page is longer than the rung deserved.

**Read §10 before quoting any number from §3.** The corrected seam operator is $\beta = 0.123188$, $\kappa = 3.7141$, at the consistent interface state $\lambda^{\ast} = 1035.487$ K with `effort_normal` declared.

> **And read §18.8 before quoting any gas-side number from §2 to §17.** Every one of them was taken with an isothermal wall that was saturated (W301) and leaked mass (W332). Re-derived with both fixed, the corrected operator is $\beta = 0.229896$, $\kappa = 2.392$ at $\lambda^{\ast} = 1133.885$ K. **CS-11's bound holds at the declared interval** ($2.2782\times$ loose, where §14 reported it violated at $0.2702$), and there is no knee. The shell reaches solidus at $2.95$ s, before the 5 s probe base (W335). The sections below keep what they recorded, each with a one-line pointer.
>
> **And read §19 before quoting any number from the marched episode (§17.4, §18).** Its bookkeeping is exact and it reproduces bitwise. But its injector delivers $17.7\%$ more than declared (W337). Its throat seam loses $3\%$ of the mass flow (W340). Its stationary walls do work (W339). And every wall flux in it is a property of the grid (W342).

---

## 2. How it is built

### 2.1 The seam, exactly

| | agent `b`, the chamber | agent `c`, the airframe |
|---|---|---|
| solver | `compressible2d.Compressible2D` | `thermostruct2d.ThermoStruct2D` |
| mesh | $112 \times 80$ body-fitted cells | $232 \times 8$, the upper panel |
| extent | $z \in [0.12,\ 0.40]$ | $z \in [-4.0,\ 0.70]$ |
| the seam face | its `jmax` wall, **112 cells** | the part of its inner face with cell centres in $[0.12,\ z_{\text{throat}}]$ — **14 cells of 232** |
| channel | `BC("wall_noslip", {"T_wall": ...})`, a Dirichlet channel in temperature | its own `_robin` surface term, $-k\,\partial T/\partial n = h\,(T - T_{\text{gas}})$ |
| `response_half` | `FLOW` — the trace is a temperature (effort), the response is $q_n/T$ | `FLOW`, same reason |

**Nothing is reimplemented and nothing is re-derived.** Both solvers are imported verbatim under the private name `atlas_build_solvers` (both repositories have a top-level `atlas`), and both meshes come from `grid.build_blocks` — the same call `data/generate.py` builds its own solvers from, so the cells a number is measured on are the cells that repo's own runs march. `generate.py::_iface_for` maps `("b","c")` to `("b", 0, "jmax")`, and the upper shell panel sits at $y = +h_{\text{in}}(z)$, which is how the face was chosen rather than guessed.

### 2.2 The operating point, and why it carries a clock *and* a horizon

Every number comes from the build repo. `data/sweep.py` declares the corpus ranges $p_c \in [2,8]\times 10^{6}$ Pa and $T_c \in [2200, 3400]$ K, with $\gamma = 1.22$ and $R = 361$ J/(kg K); the probe sits at the **midpoint**, $p_c = 5.0 \times 10^{6}$ Pa and $T_c = 2800$ K. The gas's conduction-limited film coefficient, the one `generate.py::_step_structure` hands to `step_thermal`, is measured rather than assumed:

$$h \in [156.84,\ 315.11]\ \mathrm{W/(m^2\,K)}, \qquad \bar h = 183.87$$

a factor of **2.01 along one wall**, which matters in §5.

**And then the shell does not settle.** Marched under that $h$ at its own 50 ms clock, the seam wall passes 294.6 K at 0.05 s, 335.3 K at 1 s, 449.4 K at 5 s, 557.6 K at 10 s and 1386.6 K at 100 s — **still rising**. There is no steady state during a burn, so a probe base here is a state *at a time*, and the time is part of the declaration. That is [[case-study-vehicle-march-atlas-0.1]]'s convention (CS-10 found a sensitivity that changed **sign** with its rollout horizon) applied to a base rather than to an output. The headline probe is taken at **5.00 s of burn**, where the seam wall is 449.4197 K in the mean and 690.0180 K at its hottest point.

> **A finding that fell out of the setup, not out of a gate.** The shell's own `validity` predicate — the one 7 of 7 agents were missing, here declaring the Al–Li solidus at 770 K — **declines at $t = 10.35$ s.** The declared airframe material against the declared chamber, with no liner and no regenerative cooling, reaches solidus in ten and a third seconds of burn. That is what a validity predicate is *for*, and it earned its keep the first time it was asked.

> **W334 (§18.8):** with the wall conducting, $\bar h = 211.40$, the wall at 5 s is $478.96$ K in the mean and $870.77$ K at its hottest, and the predicate declines at **$2.95$ s**. That is before the declared probe base, which is W335.

The hottest point is at $z = 0.396$, the throat end, and the cause is geometric: in the converging section the inner surface is sloped, so the Robin face length is $\sqrt{\mathrm{d}z^2 + \mathrm{d}h^2} = 1.1662\,\mathrm{d}z$ while the element mass is unchanged. Seventeen per cent more heat into the same metal.

### 2.3 Where the fixture's `M_EFF = 12` stops being true

`rocket.py` declared `effective_resolution = 12` on every port of all seven agents. The airframe mesh is uniform over 4.7 m, so a cell is 20.259 mm and the chamber wall gets **14 of them**. At the 4-cell cutoff wavelength every classical-solver case in this vault uses (`ground_effect.LAMBDA_CUT_CELLS`; `wake_array` uses 8, for a frozen checkpoint):

$$m^{\text{eff}} = \frac{2 n_{\text{cells}}}{\lambda_{\text{cut}}} + 1 \quad\Longrightarrow\quad m^{\text{eff}}_c = 8, \qquad m^{\text{eff}}_b = 57$$

and $\dim M = \min_i m_i^{\text{eff}} = \mathbf{8}$, however finely the gas declares its own side. The fixture's 12 was a convenience of a graph with no geometry in it. **The interface is genuinely non-conforming** — 112 cells against 14 over the same 0.28 m — which is what the declared prolongation pair is for.

The two sides' cells are not uniform either, so the interface basis is orthonormalised against the real arc-length Gram rather than assumed orthonormal in the closed form; the residual $\lVert P^{\mathsf T} \mathrm{diag}(w) P - I\rVert_\infty$ is $2.2\times 10^{-16}$.

---

## 3. R0's four gates

### 3.1 Gate 4 first — the control that must PASS

`thermostruct2d`'s conduction step is linear, so the shell's response to the seam trace **exists in closed form** and a probe harness that cannot reproduce it is wrong about the harness rather than about the physics. Assembling it from the same matrices `step_thermal` builds, with $E$ the face-averaging operator and $G = \partial f_{\text{in}} / \partial T_{\text{gas}}$:

$$A\,T^{n+1} = \tfrac{M}{\Delta t}T^{n} + f_{\text{in}}(T_{\text{gas}}) + f_{\text{out}}, \qquad \frac{\partial q}{\partial T_{\text{gas}}} = \mathrm{diag}(h)\,\bigl(I - E A^{-1} G\bigr)$$

| convention | probe against the assembled operator | verdict |
|---|---|---|
| `heat`, the pseudo-bond $q_n$ | $8.2929\times 10^{-9}$ | **passes**, at the floor |
| `entropy`, the declared bond $q_n/T$, against the analytic **Jacobian** | $3.0231\times 10^{-6}$ | **passes** |
| `entropy` probe against the **affine** operator | $0.99982$ | **the control's own control** |

The floor is not a matter of opinion: CG converges to `rtol = 1e-10` and the probe's finite-difference step is $10^{-2}$, so $10^{-8}$ is the best any affine response can do. The measurement sits 1.2× above it.

The third row is why the first two mean something. THERM's declared bond is $(T,\ q_n/T)$ — chosen in [[port-algebra-atlas-0.1]] precisely so that effort times flow is a power — and **dividing by $T$ is what makes the response nonlinear in the effort**. The pseudo-bond would have been affine. If the entropy response had been affine after all, gate 4 would have passed for a reason that is not the one claimed; it is off by 100%, so it did not. This is W74's cause stated as algebra rather than as a caveat.

### 3.2 Gates 1 and 2 — the seam, and what it cost

> **SUPERSEDED BY §10, and the reader should go there before quoting any number in this section.** The $\beta$ below is measured at two *mismatched* bases and with the *wrong assembly convention* — and the two errors nearly cancelled, which is why it looked unremarkable. The corrected value at the consistent interface state with the convention declared is $\beta = 0.123188$, $\kappa = 3.7141$. §10 has the full table and the evidence.
>
> **And by §18.8 (W334).** With the fixed wall, this section's configuration reads $0.146475$ and the corrected one $0.229896$.

At $\dim M = 8$, burn 5.00 s, gas cadence $10^{-5}$ s:

| quantity | value |
|---|---|
| $\beta = \sigma_{\min}(S)$ | $\mathbf{0.149619}$ |
| $\kappa$ | $2.22739$ |
| null dimension | $\mathbf{0}$ — declared $0$, **excess $0$** |
| $\omega$, operator content | $0.302781$ |
| passivity $\lambda_{\min}$ | $-0.33326$ |
| smallest block's share | $0.169472$ |

and per block, with $m + 1 = 9$ solver calls each:

| block | calls | share | $\lVert S\rVert_F$ | $\omega$ |
|---|---|---|---|---|
| `b`, the gas | 9 | $1.168$ | $0.69964$ | $0.3028$ |
| `c`, the shell | 9 | $0.1695$ | $0.095915$ | $0.3151$ |

**Cost, measured in both currencies as the gate requires**: 9 `boundary_response` calls per block — $\dim M + 1$, the base probe plus one column per interface mode — which for the gas is **306 CFL sub-steps**, and 7.070 s of wall time on a battery-derated laptop. The sub-step count is the honest currency; the seconds are the one that rots — $7.070$ here against the $6.532$ an *identical* earlier run reported, 8% apart within one evening on one machine.

The null check is the one that could have embarrassed the declaration and did not. A conjugate-heat-transfer seam constrains nothing the way incompressibility constrains a MECH trace — a uniform temperature shift produces a uniform flux change — so $n_0(\Gamma) = 0$, and the probe agrees. [[tier0-measurements]] W2 records the opposite outcome on a fluid–fluid seam, where the measured null dimension was 0 against a declared 1 and **the declaration was the thing at fault**; here the two agree, which is worth one line rather than none.

### 3.3 Gate 3 — the control that must FAIL, on real physics

W66 established by three legs that `response_half` cannot be inferred: no property of the returned numbers distinguishes an effort from a flow. It established this on **fixtures built to fail**. Here are the same three legs against two real solvers:

| leg | verdict | `L3/C9` refusals |
|---|---|---|
| both `FLOW` — the truth | `admit-uncertified` | **0** |
| shell flipped to `EFFORT` | `refuse` | **1** |
| gas flipped to `EFFORT` | `refuse` | **1** |
| **both** flipped to `EFFORT` | `admit-uncertified` | **0** |

The fourth row is the finding and it is negative: **a matching pair of wrong halves passes silently**, on real physics, exactly as it does on a fixture. The rule catches disagreement, which is all it ever claimed to catch, and the vault's own summary of W66 — *"a declaration nothing checks"* — survives contact with two solvers that were not written to make the point.

Note also the first row: **the two-agent b–c graph reaches `admit-uncertified` with zero refusals.** The shell's clock is 50 ms and the gas's is the probe cadence, so the ratio is far above 1 and `build_bc_graph` declares `FluxMatching.TIME_INTEGRATED`; both sides supply `boundary_response_integrated`, so `L7/R9` **and** `L7/R9/quadrature` both clear. **That is R2's declaration condition met on the one seam where both sides are real** — and it is only half of R2. The other half, the composed defect at the graph's declared 50:1 ratio measured against CS-11's bound, is **not done here** and is the honest statement of where that rung stands (§9).

### 3.4 The declarations, measured rather than asserted

`probe.support_reach` pokes one seam cell and reports which cells move — two solves, no spectral threshold.

| agent | reach | nonzero | fraction | `consistent_with` | declared |
|---|---|---|---|---|---|
| `c`, the shell | 7 of 14 | **14** | $1.0000$ | `('embedded',)` | `EMBEDDED` |
| `b`, the gas | **0** | **1 of 112** | $0.0089$ | `('exposed', 'none')` | `NONE` |

The shell's profile falls $2.515\times 10^{-2} \to 1.009\times 10^{-4} \to 1.619\times 10^{-5} \to \ldots \to 3.403\times 10^{-8}$: **globally supported and strongly decaying at once**, which is exactly what a backward-Euler solve should look like and is not a contradiction — `support_reach` thresholds at exactly-nonzero while $\omega$ weights by magnitude.

Both declarations are corroborated. **One of the two corroborations is worthless**, and §4 is why.

> **W334 (§18.8):** with W301 fixed, the gas's poke reaches 24 cells and moves 39 of 112, so its `reach` now reads the solver.

---

## 4. W301 — the isothermal wall is saturated, and the trace never reaches the solver

> **Fixed in the solver at Tier 86 (§18.1), and re-derived at §18.8.** No wall temperature now gives a bitwise-identical field in either gas agent. This section records the defect as it was found.

### 4.1 The mechanism

`compressible2d.py` line 170, setting the ghost cell for an isothermal no-slip wall:

$$T_g = \max\bigl(2\,T_{\text{wall}} - T_i,\ 20\bigr)$$

The clamp is a **positivity guard and it is correct**: below $T_{\text{wall}} = T_i / 2$ a linear ghost extrapolation gives a negative temperature, and $\rho_g = p / (R\,T_g)$ would divide by it. What it *also* does is turn the Dirichlet channel into a **saturated** one below

$$T_{\text{wall}}^{\ast} = \frac{T_i + 20}{2}$$

with no diagnostic anywhere.

### 4.2 The measurement, and why it is stated bitwise

"Dominated by a film coefficient" and "never reached the solver" are different claims, and only an exact-zero test separates them. So the assertion is bitwise equality of the post-step field.

**The rocket chamber**, $T_i = 2800$ K, threshold $1410$ K:

| $T_{\text{wall}}$ | $\max\lvert\Delta T_i\rvert$ | $\max\lvert\Delta p\rvert$ | bitwise equal to the 300 K field |
|---|---|---|---|
| $300.00$ | $0$ | $0$ | **True** |
| $449.42$ — *the probe base* | $\mathbf{0}$ | $\mathbf{0}$ | **True** |
| $690.00$ | $4.579$ | $117.6$ | False |
| $1500.00$ | $1049$ | $1.620\times 10^{4}$ | False |

**And `thermal_seam` — this vault's flagship multiphysics case study — is in the same regime**, $T_i = 900$ K, threshold $460.0$ K, declared probe base $400$ K:

| $T_{\text{wall}}$ | $\max\lvert\Delta T_i\rvert$ | bitwise equal to the 400 K field |
|---|---|---|
| $300.00,\ 350.00,\ 400.00,\ 440.00,\ 459.00$ | $0$ | **True** |
| $460.00$ | $1.2669\times 10^{-5}$ | False |

The transition is at **exactly** the algebraic threshold, to the digit.

### 4.3 What the probed block is actually made of

Poke one cell of the trace at the rocket's probe base and **1 of 112** response cells moves — the poked one, by $0.175064$; the other 111 by bit-zero. Compare that single number against the algebraic derivative of `generate.py::_wall_flux` with the interior field **held frozen**,

$$\phi = \frac{k_{\text{gas}}}{\delta n}\,\frac{T_i - T_w}{\tfrac{1}{2}(T_i + T_w)}, \qquad \frac{\partial \phi}{\partial T_w}\bigg|_{T_i\ \text{frozen}}$$

and the ratio of measured to predicted is $\mathbf{0.999996923}$.

> **So the chamber gas's block of the b–c seam operator contains no solver.** It is the post-processing flux formula differentiated about a field that never saw the trace. The same poke on `thermal_seam`'s gas gives $0.0925536$ with bit-zero elsewhere — reproducing to three digits the $9.3\times 10^{-2}$ that [[tier0-measurements]] recorded for W68/W71.

**What is new here is the mechanism, not the observation, and the distinction is worth stating precisely.** `probe.operator_content`'s own docstring already records that *"in the spatial domain the gas block is **exactly** diagonal — a delta at one seam cell moves that cell's flux by 9.3e-2 and every other cell by bit-zero"*. That is right, it is reproduced above, and nothing here overturns it. What that measurement did **not** have was a cause, and the cause it was read with — *"the difference of two film coefficients"*, a statement about competing **scales** — turns out to be the wrong kind of explanation. There is no competition of scales. The trace does not enter the solver at all, so the block is `generate.py`'s post-processing formula differentiated about a frozen field, and it would look **identical for any interior solver whatsoever**.

Three things follow that a ratio-of-scales reading does not license. The block has a **threshold** rather than a gradient — it changes qualitatively at $T_{\text{wall}} = (T_i+20)/2$, measured at exactly $460.0$ K. The gas's $\Xi$, its composability index, would be a property of `_wall_flux` and not of `compressible2d`. And the seam's $\omega$ is not measuring "how much operator survives under the boundary condition"; on this side there is no operator underneath to survive.

The shell, by contrast, is a genuine operator: 14 of 14 cells respond, `exact_zeros` is 0.

### 4.4 Why no existing rule catches it, and the one that could

**`ProbedBlock.is_empty` misses it** because $\Lambda \neq 0$ — the algebraic term depends on the trace perfectly well.

**`operator_content` misses it too, and this is the part worth arguing about.** $\omega = \lVert S - cI\rVert_F / \lVert S\rVert_F$ was built for exactly this job — W68/W71's *"a probed block is supposed to be the expert's DtN map; it can instead be the expert's boundary condition"* — and on this seam the gas block reads $\omega = 0.3028$, three hundred times above its $10^{-3}$ floor. No rule fires. The reason is that $\omega$ measures departure from a **scalar** multiple of the identity, and this block is bitwise *diagonal* with a diagonal that **varies by 2.01 along the wall** because $h$ does. A saturated channel with a non-uniform film coefficient is far from scalar and carries no operator content at all.

**The discriminator already exists and has never been read.** `SupportReach.exact_zeros` has been emitted since Tier 0:

| agent | $n$ | nonzero | `exact_zeros` | $n-1$ |
|---|---|---|---|---|
| rocket `b`, gas | 112 | 1 | **111** | 111 |
| `thermal_seam` gas | 48 | 1 | **47** | 47 |
| rocket `c`, shell | 14 | 14 | **0** | 13 |

A saturated channel reads `exact_zeros` $= n - 1$ exactly; a genuine operator does not.

> **MEASURED AT TIER 77, AND IT REFUSES THE RULE. See §11.** Swept across every port of every case graph that builds, the signature fires on **15 of 110 real-callable ports — 13.6%** — of which only **2** are actually saturated. Being exactly diagonal turns out to be an ordinary property of a lumped expert, and this vault has many. The proposal below stands as a *necessary* condition and is **not usable as a rule**.

This is [[tier0-measurements]] W76's shape repeated one object down — *the first rule in the vault that reads a field the probe has emitted since Tier 0 and no layer had ever consumed* — and it costs two solves, which `support_reach` already spends.

### 4.5 The part that cannot be fixed by a rule

**[AI Inference]:** the saturation is **undetectable through the L1 interface by construction**, and that is a sharper statement than "no rule catches it." `ExpertCapabilities` exposes `boundary_response` and nothing else; "did the imposed trace reach the solver's state" is a question about the expert's internals, which L1 forbids anyone from asking. What §4.4 offers is a *necessary* condition visible in the response alone — bitwise-zero off-diagonal — not a sufficient one. An expert whose genuine response happened to be exactly diagonal would trip it, and an expert that saturated only partially would not. This is unverified beyond the three agents measured here.

**And the consequence for a declaration is concrete.** Both gas agents declare `bc_channel = DIRICHLET`. That is true of the code path and false of the operating point: at its own probe base the channel transmits nothing. `support_reach`'s corroboration of the gas's `elliptic_subsolve = NONE` in §3.4 is therefore **vacuous** — it read local because it was saturated, not because the solver is explicit. The declaration happens to be right; the evidence for it is void.

---

## 5. Predictions, registered before any arm ran

Fifteen were written down before the first measurement (`PREDICTIONS-tier76`, reproduced in `out/w300.json`'s provenance). **Eleven held and four were refuted.** The four are the useful part.

### 5.1 The four that failed

**P3 — $\omega$ on the shell block would be below $0.10$. Measured $0.3151$.**
The reasoning was that the shell's thermal diffusion length over one step, $\sqrt{\alpha \Delta t} = 1.57$ mm with $\alpha = k/(\rho c_p) = 4.938\times 10^{-5}\ \mathrm{m^2/s}$, is far below a 20.3 mm cell, so the block is near-diagonal, so it is near a multiple of the identity. **The first two steps are right and the third does not follow.** The control settles it: force $h$ uniform and re-probe.

| arm | $\omega$ | spatial diagonal spread | off-diagonal fraction |
|---|---|---|---|
| $h$ as measured | $0.326700$ | $2.977\times$ | $5.896\times 10^{-3}$ |
| $h$ forced uniform | $\mathbf{0.081904}$ | $1.377\times$ | $5.576\times 10^{-3}$ |

The off-diagonal fraction barely moves — the block **is** near-diagonal, as predicted — and $\omega$ falls by $3.99\times$, straight into the range P3 named. So P3's physics was right and its inference about what $\omega$ measures was wrong, and the same mistake is the one §4.4 catches the rule making.

**P4 — 13 solver calls per block at $\dim M = 12$. Measured 9, at $\dim M = 8$.**
The *rule* held exactly ($m+1$); the $m$ did not, because the fixture's uniform `M_EFF = 12` is not attainable on a wall the airframe mesh resolves with 14 cells (§2.3). A prediction refuted by a declaration correction rather than by a measurement, which is worth distinguishing.

**P9 — declaring `time_discretization` on all seven agents removes exactly two decertification records, `L4/R2b/W46` and `L2/R10/halo`. It removes one.**
This is a claim **both plan documents make** — `ROCKET-POC-PLAN.md` §2 and `ATLAS-CLASSICAL-PLAN.md` §7 both say declaring it *"lifts both `L4/R2b/W46` AND `L2/R10/halo`"* — and the one-factor-at-a-time ledger shows it is wrong. `L2/R10/halo` is lifted by the **decomposition** being non-overlapping, and would have been lifted with `time_discretization` still unknown.

What declaring it does to `L2/R10/halo` is more interesting than removing it: the rule stops being undecidable and starts being *decided*, and the answer is that the gas agents' domain of dependence **needs an overlap of 33,446 cells**. The chamber has 8,960. **[AI Inference]:** an overlapping Schwarz-type decomposition of these gas agents at the YAML's 1 ms clock is therefore impossible in principle — one macro step's domain of dependence is $3.7\times$ the whole subdomain — which is an independent argument for the non-overlapping declaration §6 arrives at on geometric grounds. Untested against an actual overlapping run.

**P13 — declaring the shell's `elliptic_subsolve = EMBEDDED` adds an `L2/R10` refusal. It adds none.**
The reason was already printed in the baseline output and I did not read it carefully enough when predicting: R10 is checked against the agents the decomposition **cuts**, and `c` is the sole agent of its `governing_family`, owns its whole region, and has no artificial face for an overlap to outrun (W136). The rocket's shell is exempt. The prediction was "the one with teeth"; it had none.

### 5.2 The eleven that held

$\beta$ finite and positive (P2) and the null space at the declared $0$ (P1); both halves of gate 4 (P5, P6); all three of W66's legs (P7, P8); the record counts for `storage` and `validity` (P10: exactly 7; P11: exactly 27, being $7 + 19 + 1$) with their cross-check closing to 13 as written; the shell's global reach (P12); the decomposition correction (P14, **with the risk it flagged materialising** — `L2/C2` went away and `L2/C3/W57` arrived in its place); and the $\omega$ diagnosis (P15).

---

## 6. R1 — the declaration ledger, one factor at a time

A count alone is not a result, and a count taken with five things changed at once is worse. Every arm below moves **exactly one** declaration off the Tier-75 fixture.

> **Re-derived at §18.8 (W334).** The `real` level is unchanged. The `known` level reads 48, not 47, and that is Tier 77's doing rather than the wall's: `effort_normal` on b–c makes the stubs' difference indefinite, which adds one `L4/E7/passivity`.

Baseline, `declarations="fixture"`: **`refuse`, 10 refusals, 49 decertifications** — and the refactor reproduces it **byte for byte**, not merely decision for decision. `w189_artifact_control` compares the rocket's whole emitted artifact against a capture taken before this tier, and pinning that control to the fixture level rather than spending an exemption on it keeps it at **40 of 40 identical**. The cost is that `build` puts no level tag in the graph's `name` or `note` at that level; the benefit is that a change which quietly rewrote every note the graph has would fail, where a count of matching decisions would not.

Two pre-existing tests did have to move, and both for the right reason: `tau` at b–c is no longer `UNDEFINED` (both sides now declare `lambda_ref`, which is precisely what `L1/E3`'s own message says declaring it does), and `R9` is no longer the refusal. **The diagnosis was kept rather than rewritten** — `TestRocketAscent` is pinned to the fixture level and a new `TestRocketAscentWithRealExperts` pins what the new level does instead.

| arm | decerts | what moved |
|---|---|---|
| known fields (`time_discretization`, `stencil_radius`, `elliptic_subsolve`, `substeps_per_macro_step`, `lambda_ref`) | $-1$ | `L4/R2b/W46` $-1$ |
| decomposition `NON_OVERLAPPING` | $-1$ | `L2/C2` $-1$, `L2/R10/halo` $-1$, **`L2/C3/W57` $+1$** |
| clocks from the YAML (50:1, not 100:1) | $0$ | **nothing** |
| `flux_matching = TIME_INTEGRATED` | $0$ | refusals: `L7/R9` $-1$, **`L7/R9/quadrature` $+1$** |
| `storage` + `validity`, all 7 **[counting control]** | $\mathbf{-34}$ | `L1/E7` $-7$, `L1/C8` $-7$, `L3/C8` $-19$, `L1/3.4` $-1$ |

**The four structural arms are exactly additive** — the sum of the parts equals the whole `known` level, rule by rule, which is a check that could have failed and did not.

Two rows deserve their own sentence.

**The clock correction buys nothing.** Moving from the fixture's 100:1 to the YAML's 50:1 moves no decertification at all. It is still the right declaration — the YAML says in its own header that it is the single source of truth, and both plan documents quote it — but its value is correctness, not verdict movement, and reporting it as the latter would have been dishonest.

**`TIME_INTEGRATED` trades one refusal for another, and the trade is progress.** `L7/R9` clears; `L7/R9/quadrature` arrives, naming `['d', 'e', 'f']` — the agents that declare time-integrated matching and supply no `boundary_response_integrated`. Before the real experts arrived that list was `['b', 'c', 'd', 'e', 'f']`. So R2's condition is not one declaration but two, and the second one is discharged agent by agent as the physics lands.

### 6.1 The counting control, and the warning it carries

The $-34$ row is an **instrument, not a claim**: it attaches trivially-true `storage` and `validity` callables to all seven agents, including five that are still seeded random matrices. A stub's storage is fiction and `lambda _s=None: 1.0` is not an energy.

It exists so that R1's ledger is a measurement. But it also measures something nobody asked for: **34 of 49 decertifications — 69% — are controlled by two fields that a graph of random matrices can declare, and the compiler cannot tell.** Adding the `known` fields on top gives a `known` level at **47 decertifications on a graph with no physics in it whatsoever**.

> This is `governing_family`'s promotion problem (W69) generalised from one field to a layer. The verdict movement between `fixture` and `known` is **a measure of the paperwork, not of the model**, and the only defence currently available is that `rocket.build` names its level in the graph's own `note` so a certificate carries which one it was compiled at.

### 6.2 The residue, named

`declarations="real"` — the `known` fields on the five stubs, plus real physics, real `storage` and real `validity` on `b` and `c`:

**`refuse`, 10 refusals, 43 decertifications.** Identical at the YAML's 1 ms gas clock (662.76 s to compile) and at the reduced $10^{-5}$ s cadence (6.7 s) — same verdict, same 43, same breakdown.

| rule | count | why it survives |
|---|---|---|
| `L2/InterfaceMotion` | 9 (refusal) | the combustion front and the plume boundary. **Stage C research, untouched, and it should stay refusing** |
| `L7/R9/quadrature` | 1 (refusal) | `d`, `e`, `f` supply no `boundary_response_integrated`. A stub problem, not a bound problem |
| `L1/E3` | 7 | the families genuinely differ. **But it is a different 7 and it now says something else — see below** |
| `L1/E7` | 5 | the five agents still on stubs have no `storage`. **Down from 7** |
| `L1/C8` | 5 | the same five have no `validity`. **Down from 7** |
| `L3/C8` | 17 | the same, per seam. **Down from 19** |
| `L1/3.4` | 1 | the OOD exposure grows with $K=7$ and $p$ is unmeasured |
| `L2/C3/W57` | 1 | substructuring: no cut criterion exists. **Arrived with the correct decomposition** |
| `L4/probe-base` | 1 | **new, and only real physics could produce it** — see below |
| `L4/E7/passivity` | 1 | **new** — defect $3.966\times 10^{-1}$; the symmetric part has a negative mode, so the $L \le 1$ branch is unavailable |
| `L5/eps_tol`, `L8/W56` | 2 | $\tau$, $\sigma$ and four more constants unmeasured on this graph |
| `L6/E6` | 1 | no assembly declared, so $\lVert \mathcal A\rVert$ is not computed |
| `L7/R9/order` | 1 | the interface representation order caps the scheme order |
| `L9/E5` | 1 | `UNTYPED`: $L$ has never been measured for this graph |

### 6.3 `L1/E3` holds its count and changes its meaning, and $\tau$ stops being undefined

The residue table's most misleading row is `L1/E3`, because 7 records before and 7 records after conceals two real changes.

**The multiphysics seams moved to where the physics puts them.** `data/generate.py` sets `rxn = self.reaction if a == "a" else None`, so `a` is the only reacting agent; the fixture labelled `a`, `b` and `e` alike as reacting, which is wrong for two of the three and was **invisible** because it made `a-b` and `e-b` agree. Correcting it from source:

$$\{\texttt{b-c},\ \texttt{c-d},\ \texttt{e-f}\} \quad\longrightarrow\quad \{\texttt{a-b},\ \texttt{b-c},\ \texttt{c-d}\}$$

`e-f` stops being a family boundary — the nozzle outflow and the plume solve the same equations with different parameters — and `a-b` becomes one, which is the only place in this vehicle where a reaction term genuinely appears on one side and not the other.

**And $\tau$ stops being undefined anywhere in the graph:** `tau_undefined_seams` goes from **7 to 0**. E3 still *fails*, because the families really do differ and no monolithic reference covers both sides — but the decertification it emits now reads *"**tau is NOT undefined here**: both sides declare `lambda_ref`, so a reference PAIR exists and tau is measurable against the tightly coupled trajectory."* That is W83's move, made for a rocket: a reference trajectory was never the same thing as a shared governing family, and declaring one turns an unmeasurable attribution into a measurable one without either family moving.

**A caution that belongs with it.** This is the *paperwork* half again — five of the seven `lambda_ref` declarations name solvers the agent is not yet running. What they buy is that swapping a real expert in would **measure** that expert's infidelity instead of emitting `UNDEFINED`; what they cost is one more unverifiable field per agent, which is W69's column and not a free move.

**The rest of the movement is not monotone either, and that is the result.** Six decertifications went away and **two arrived that the fixture could not have produced**:

- **`L4/probe-base`.** The two sides were linearised about states differing by 1374 on $M$ — **876% of the base norm**. The gas linearises about the wall it sees (449 K) and the shell about the gas it sees (2800 K); each is honest alone, and $\Lambda_M = \sum_i P_i^{\ast}\Lambda_i P_i$ is a sum of Jacobians, which is a Jacobian only if the terms share a point. This is W74's class firing on a rocket, and `thermal_seam`'s two sides were 500 K apart; these are nearly three times worse.
- **`L4/E7/passivity`.** A negative mode in the symmetric part of the assembled seam. **[AI Inference]:** W138 records that a fluid–solid seam is often written with both sides positive in one shared direction, so that the well-posed condition is their *difference* and adding them assembles a matrix no scheme differentiates. This seam declares no `effort_normal`, so both blocks entered with $+1$. Whether the defect is that convention or a genuinely amplified mode is **not settled here** and is the natural first question for the next rung.

### 6.4 The fixture was not neutral, and E7 is where it shows

The envelope stamp moves once and only once in this whole tier, and it moves at the step where physics arrives:

| level | E1 | E2 | E3 | E4 | E5 | E6 | **E7** |
|---|---|---|---|---|---|---|---|
| `fixture` | holds | fails | fails | fails | unchecked | unchecked | **holds** |
| `known` | holds | fails | fails | fails | unchecked | unchecked | **holds** |
| `real` | holds | fails | fails | fails | unchecked | unchecked | **fails** |

`_responder` builds each stub as

$$A = g\,\bigl(A_0 + A_0^{\mathsf T} + 3I\bigr), \qquad A_0 \sim \mathcal N(0,\,0.1)$$

which is **symmetric and diagonally dominant by construction**, hence positive definite. So E7 — the passivity hypothesis, the one that decides whether the $L \le 1$ branch of the master bound is available — held on the fixture *for a reason that had nothing to do with a rocket*, and it is the first hypothesis real physics breaks.

**A fixture is a set of assumptions, and this one was optimistic exactly where it mattered.** Nothing was wrong with writing it that way — a responder that claims a JVP must supply a consistent one, and a well-conditioned symmetric matrix is the cheapest way to get that. The lesson is narrower and worth carrying: **the hypotheses a stub satisfies are the ones nobody has tested**, and E7 sat green through every compile of this graph since it was written.

---

## 7. Two numbers about probing that generalise past this seam

> **Re-derived at §18.8 (W334).** With the fixed wall, and in the difference assembly Tier 77 declared, the cadence costs $0.03\%$ and the horizon $11.2\%$ ($\beta$ $0.210346 \to 0.189202$ over 0.5–10 s). The ordering stands, and the horizon now dominates by about $390\times$. The 5 and 10 s rows are both past solidus now (W335).

### 7.1 The cadence costs 0.3% and 83×

The chamber's CFL sub-step at the operating point is $2.990008\times 10^{-7}$ s, so the YAML's own $\Delta t_b = 10^{-3}$ s is about 3350 explicit sub-steps. Probing there is a quarter of an hour for one seam. Over a 100× cadence range:

| $\Delta t_{\text{gas}}$ | $\beta$ | $\kappa$ | $\omega$ | cost |
|---|---|---|---|---|
| $10^{-6}$ | $0.149201$ | $2.2034$ | $0.2987$ | 36 sub-steps, $0.8$ s |
| $10^{-5}$ | $0.149619$ | $2.2274$ | $0.3028$ | 306 sub-steps, $6.8$ s |
| $10^{-4}$ | $0.149639$ | $2.2396$ | $0.3048$ | 3051 sub-steps, $62.7$ s |

$\beta$ moves by a factor of $\mathbf{1.003}$ while the cost moves by $\mathbf{84.75\times}$ **in sub-steps**.

The cost is quoted in sub-steps because the seconds do not hold still: two runs of this identical sweep, hours apart on one machine, gave wall-clock ratios of $87.0$ and $78.4$ — an $11\%$ spread — while the sub-step ratio was $84.75$ both times. So a reduced probe cadence is a **measured** trade here rather than a convenience, and `rocket.build(gas_dt=None)` still takes the YAML's clock for anyone who wants to pay it.

### 7.2 The horizon costs 26.5%, which is 88× more

| burn | $T_{\text{wall}}$ mean (max) | valid | $\beta$ | $\kappa$ | shell's share |
|---|---|---|---|---|---|
| $0.5$ s | $316.78$ ($375.06$) | yes | $0.169621$ | $2.3763$ | $0.0935$ |
| $1.0$ s | $335.25$ ($432.25$) | yes | $0.166803$ | $2.3477$ | $0.1059$ |
| $2.5$ s | $383.07$ ($554.61$) | yes | $0.159537$ | $2.2880$ | $0.1344$ |
| $5.0$ s | $449.42$ ($690.02$) | yes | $0.149619$ | $2.2274$ | $0.1695$ |
| $10.0$ s | $557.65$ ($859.15$) | **no** | $0.134092$ | $2.1628$ | $0.2222$ |

$\beta$ moves by $\mathbf{1.265\times}$, and the shell's share of the seam operator more than doubles as the wall heats. **The probe base's horizon matters about $90\times$ more than the probe's own cadence** — $26.496\%$ against $0.294\%$ — which is the ordering a reader would be least likely to guess and the reason both are reported.

The last row is quoted from a state the shell's own `validity` predicate declines. It is included **because** it is invalid: a $\beta$ at 10 s of burn is a number about a melted airframe, and leaving it out would have hidden the fact that the curve runs straight through the boundary of the declaration.

---

## 8. What this tier did not do, and why

**MECH is not wired.** The b–c edge carries MECH as well as THERM and the MECH response is stubbed **deliberately rather than by omission**. `solve_mechanical` is quasi-static and free-body — it removes the three planar rigid-body modes by projection — so the response to a net pressure load is the part of the load that is *not* the thrust, and the missing part is the force the trajectory agent integrates. And MECH's declared bond is $(\text{traction},\ \text{velocity})$ while a quasi-static solve returns a **displacement**, so turning it into the declared flow needs a time derivative whose convention is what R2 is about. The MECH port is the field-to-lumped coupling W97 closed at CS-10 wearing a different hat, and it belongs with the trajectory agent rather than ahead of it.

**`L2/InterfaceMotion` is untouched.** Nine ports refuse and they should. Agent `a`'s combustion front is `solution_dependent` and the plume boundary is `prescribed`; a moving interface silently invalidates a cached operator, which is the silent-wrongness class, and no rule exists. `test_interface_motion_still_refuses_nine_ports` asserts the count at **every** declaration level, so no future tier removes it by accident.

**Five agents are still stubs**, and `rocket.build` says so in each record's `note` and in the graph's own. A mixed record whose reader cannot tell which port is physics would be a worse artefact than an honest stub.

---

## 9. What follows

| next | why |
|---|---|
| `a`, `e`, then `d`, `f`, `g` | R0's remaining rungs. The pattern is now proven twice and the cost is known: $\dim M + 1$ calls per block, and the cadence trade is measured |
| **W301's detector** | `support_reach.exact_zeros == n - 1` as an L4 check, with the necessary-not-sufficient caveat §4.5 states |
| **W301's physics** | every THERM number in [[tier0-measurements]]'s Tier 13–16 range was taken with the gas wall saturated. The *conclusions* about the port algebra look robust; the numbers describing the gas's transverse response are not numbers about the gas |
| `effort_normal` at b–c | to settle whether `L4/E7/passivity` is the convention or a real mode |
| a common `seam_base` | `L4/probe-base` has a named remedy: `assemble_seam` takes one |
| the five stubs' `boundary_response_integrated` | discharges `L7/R9/quadrature` agent by agent |

---


---

## 10. Tier 77 — the correction, and two errors that nearly cancelled

**Added 2026-09-17, the same day.** §3.2's $\beta$ was published *with its own decertification attached*: `L4/probe-base` said the two sides were linearised 1374 apart on $M$, and the compiler's own wording is that such a number is *"not merely the wrong point but no point at all."* Publishing it was the right call — the alternative was to publish nothing — but leaving it there would not be. This section settles it, and **the answers went against five of the eight predictions registered for them.**

### 10.1 The evidence is a root, not an argument

`assemble_seam` sums the two blocks unless a seam declares `effort_normal`. The rocket declared none, so the sum was assembled. Scanned over the admissible interval $[250, 2800]$ K — the two reservoirs the seam can physically reach, by `w87`'s convention:

| | sign changes over $[250, 2800]$ K |
|---|---|
| **SUM** — Steklov–Poincaré, what Tier 76 assembled | **NONE.** The scalar stays in $[43.2,\ 61.5]$, essentially flat |
| **DIFFERENCE** — what `effort_normal` declares | **one**, bracketed $(1015,\ 1270)$, root at $\mathbf{1035.487}$ K |

The sum cannot have a root, and the reason is algebraic rather than numerical. `generate.py::_step_structure` hands the **gas's own** conduction-limited $h$ to the **shell's** Robin channel, so both sides carry the same film coefficient:

$$q_{\text{gas}} = h\,(T_i - \lambda), \qquad q_{\text{shell}} = h\,\bigl(\lambda - T_{\text{face}}(\lambda)\bigr)$$

$$\Longrightarrow\qquad q_{\text{gas}} + q_{\text{shell}} = h\,\bigl(T_i - T_{\text{face}}(\lambda)\bigr)$$

and $\lambda$ **cancels**. The sum's root is where the shell's face reaches the gas's near-wall temperature — thermal equilibrium, not a state one 50 ms macro step can reach.

> **A method note, because it cost something.** The first version of this stage ran Newton blind on that residual and was killed after **204 s of CPU without converging**. That was not a solver problem and no amount of damping would have fixed it. A scan costs one solver call per point, shows a sign change or its absence directly, and *is* the evidence; Newton is for after a root is bracketed. This vault already holds the rule that a comparison is not a result until it has been marched past where the curves could cross. The sibling: **a root is not a result until something has shown the residual changes sign.**

### 10.2 The corrected numbers, and the near-cancellation

Four configurations — two bases against two assemblies, at $\dim M = 8$:

| assembly | base | $\sigma_{\min}$ | $\kappa$ | symmetric spectrum | one-signed |
|---|---|---|---|---|---|
| **SUM** *(Tier 76, as published)* | each side's own | $0.149619$ | $2.2274$ | $[-0.33326,\ -0.14962]$ | **yes** |
| SUM | $\lambda^{\ast} = 1035.487$ K | $\mathbf{0.0013150}$ | $\mathbf{142.41}$ | $[-0.005560,\ +0.18727]$ | **NO — mixed** |
| DIFFERENCE | each side's own | $0.200562$ | $2.2212$ | $[-0.44549,\ -0.20056]$ | yes |
| **DIFFERENCE** *(corrected)* | $\lambda^{\ast} = 1035.487$ K | $\mathbf{0.123188}$ | $\mathbf{3.7141}$ | $[-0.45753,\ -0.12318]$ | yes |

> **W334 (§18.8): the near-cancellation was the old wall's.** With the fixed wall, $\lambda^{\ast} = 1133.885$ K and the corrected $\beta = 0.229896$ ($\kappa = 2.392$). The doubly-wrong configuration sits $36\%$ below it, and the nearest is the right assembly at the wrong base. Put the old wall back on today's code and this table returns to within $0.25\%$, and to $5\%$ on the near-singular sum.

**Tier 76's number was wrong twice, and the two errors nearly cancelled.** $0.149619$ against the corrected $0.123188$ is 21% apart — close enough to look like an ordinary number. Fix *either* error alone and it moves much further: the right base with the wrong assembly gives $0.0013150$, a factor of **114 down**; the wrong base with the right assembly gives $0.200562$, up 34%. **The configuration that was wrong in both respects landed nearest the one that is right in both.** That is the most uncomfortable thing this tier measured, and it is pinned by `test_two_errors_that_nearly_cancelled`.

### 10.3 What E7's failure actually was — and W306

Three of the four rows above are **one-signed**: every eigenvalue of the symmetric part carries the same sign. A one-signed *negative* operator is passive up to a **global sign**, and the interface problem $S\lambda = \chi$ is indifferent to it, because $\chi$ flips with $S$. Only one row — the sum at the consistent base — has a genuinely **mixed** spectrum, which is an amplified interface mode.

`L4/E7/passivity` rejects all four identically, because it computes

$$\texttt{passivity\_defect} = \lvert \lambda_{\min}\rvert \quad\text{when}\quad \lambda_{\min} < -\texttt{tol}$$

and **never reads $\lambda_{\max}$**. So:

> **W306.** The passivity diagnostic cannot distinguish *"this operator is passive and my sign convention is upside down"* from *"this operator amplifies an interface mode"*. Those are different statements with different remedies, and the quantity that separates them is one line — the sign of $\lambda_{\max}$ — which the probe already has in hand and nothing computes.

So Tier 76 §6.2's reading of `L4/E7/passivity` as a finding the fixture could not have produced is **half right**: it is real, it could not have come from a stub, and it is **not** an amplified mode. The registered prediction that it was physics rather than convention is refuted.

### 10.4 And `thermal_seam`'s E7 holds by an accident

Both conjugate-heat-transfer seams in this vault have the **same sign structure**: the fluid block is negative — a hotter interface leaves the gas less to transfer — and the solid block positive, because a hotter interface drives more into the metal. So the difference is always one-signed and the sum always cancels. Whether the *sum* is definite is therefore a question of which block **dominates**, and that is a modelling choice rather than a property of the physics:

| | solid block | fluid block | ratio | the sum |
|---|---|---|---|---|
| `thermal_seam` | $+1.099$ | $-0.0855$ | $\mathbf{12.9}$ | **definite**, $\sigma_{\min} = 1.0136$ |
| rocket b–c | $+0.2363$ | $-0.2254$ | $\mathbf{1.05}$ | near-singular and mixed |

`thermal_seam` declares `H_IN_NOMINAL = 500` **independently of its gas**; the rocket follows `generate.py` and hands the shell the gas's own $h = 183.87$, so its two blocks nearly cancel.

> **W334 (§18.8):** with the fixed wall, the rocket's blocks are $+0.3266$ and $-0.2641$, a ratio of $1.24$. Its gas block is now cleanly negative where it was faintly mixed, and the sum is still near-singular and indefinite. `thermal_seam` stays at $12.9$. W307 stands.

> **[AI Inference]:** this means `thermal_seam`'s `E7: holds` — quoted across Tier 13–16 — is **not evidence that conjugate-heat-transfer seams are passive**. It is a property of a 500. Any CHT seam whose two sides carry comparable film coefficients should be expected to land where the rocket did. Unverified beyond these two seams, and tracked as **W307**.

### 10.5 The predictions, and most of them failed

Eight were registered before their arms ran. **Three held, five were refuted**, and the refutations are the content. (Q6 ran later the same day and is refuted too — §11 — taking the tally to three held and six refuted.)

| | prediction | outcome |
|---|---|---|
| **Q1** | $\lambda^{\ast} \in [449, 470]$ K, pinned to the metal by its thermal mass | **refuted** — $1035.487$ K. The mechanism assumed the *sum* condition; under the difference the interface sits between the two sides, as equal film coefficients demand |
| **Q2** | $\beta$ moves by less than $2\times$ when re-based | **refuted** — $100\times$ down under the sum. Its mechanism depended on Q1 and fell with it |
| **Q3** | the mismatched $\beta$ lies inside the swept range | **held, and uninformative** — the range spans $209\times$, so nearly anything would be inside. Reported as a weak test, not a hit |
| **Q4** | declaring `effort_normal` makes the defect *worse* | **refuted in its conclusion.** The norm does rise, as predicted; but I tested only one of the two differences and reasoned from magnitude instead of checking both orientations |
| **Q5** | E7's failure is physics, not convention | **refuted** — §10.3 |
| **Q6** | the W301 detector flags exactly two agents | **refuted** — 15 of 110 real-callable ports, and only 2 of those are saturated. §11 |
| **Q7** | the sum has no admissible root, the difference does | **held qualitatively and decisively; refuted quantitatively** — the predicted band $[1300, 1550]$ K missed the measured $1035.487$ by 28%, because the estimate ignored the $1/T$ in the declared bond |
| **Q8** | if Q7 holds, Q4 is refuted and E7's failure is the convention | **held** |

**Q4 and Q5 failed against a note in the code I was already using.** `assemble_seam`'s own docstring says, of a fluid–solid seam: *"Adding them assembles a matrix no scheme differentiates, and `L4/E7/passivity` then reports a defect that is the convention rather than an amplified mode."* It says it of MECH; it is equally true of THERM. **That is the second tier running in which a prediction failed against a statement already written in the module under test** — Tier 76's P13 was the same shape. The lesson is cheap and specific: *read the docstring of the function whose output you are about to predict.*

### 10.6 What is declared, and what is deliberately not

`effort_normal = "b"` is declared on **b–c:THERM only**. Both sides return heat positive in the direction gas $\to$ shell, which is $+y$ at this wall; read off the meshes, $+y$ is agent `b`'s own outward normal — its `jmax` face normal is $[0, +1]$ — and is against the shell's inner-face normal, which points back into the gas.

**Which of the two agents is named carries no physical content**: it flips a global sign on $S$, and the interface problem is unchanged because $\chi$ flips with it. What it changes is whether E7's test passes, which is exactly W306.

**The other six edges are not declared.** Their sign structure has not been measured, and a declaration copied across a graph on the strength of one seam is the defaulting this vault keeps finding. The rocket's compile is therefore unchanged at **`refuse`, 10 refusals, 43 decertifications** — the correction moves numbers, not the verdict.

---


---

## 11. Tier 77 — the W301 detector, measured and refused

§4.4 proposed `SupportReach.exact_zeros == n - 1` as the L1-visible signature of a saturated boundary channel, and §4.5 flagged it as *necessary and not sufficient*. **Q6 predicted it would flag exactly two agents. Swept across the vault it flags 15 of 110 real-callable ports — 13.6% — and the insufficiency turns out to be the common case rather than a corner.**

### 11.1 The rate, and what the false positives are

141 ports across 8 case graphs; 13 cases skipped and named, because their `build()` needs a velocity field the sweep has no business inventing. The rate is read **per kind**, because a fixture built as `linear_response(A)` with a diagonal $A$ trips the signature trivially and says nothing about an expert:

| kind | ports | tripped | rate |
|---|---|---|---|
| real callable | $110$ | $\mathbf{15}$ | $\mathbf{13.6\%}$ |
| `linear_response` fixture | $31$ | $4$ | $12.9\%$ |

And the 15 real-callable trips break down as:

| | count | what they are |
|---|---|---|
| **genuinely saturated** | $\mathbf{2}$ | `thermal_seam.gas`, `rocket.b` — the two §4 is about |
| $n = 1$ lumped agents | $2$ | `powertrain.MGU`, `car_graph.MGU` on `shaft:ROT`. With one cell, *"the poked cell responds and no others"* is **vacuous** |
| legitimately diagonal experts | $11$ | `ROTOR`, `FLUID`, `RAD`, `PASS` across `powertrain`, `car_graph`, `cooling_loop` — lumped or gain-type responses with no spatial coupling, several with peaks that are round numbers ($3$, $5$) |

Precision on real-callable ports is $2/15 = \mathbf{13\%}$, or $2/13$ once the vacuous $n = 1$ cases are removed. **That is not a rule; it is an 87% false-positive rate.**

> **W334 (§18.8):** swept again with W301 fixed, the signature trips on 13 of 110, and **none** is saturated. The two genuine cases were the two that stopped tripping.

Two controls bracket the measurement and both behaved — an exactly-diagonal responder trips, a one-cell banded one does not — so the 13.6% is a rate about the vault and not about the instrument.

### 11.2 The negative sharpens §4.5 rather than overturning it

§4.5's **[AI Inference]** was that the saturation is undetectable through the L1 interface *by construction*, because *"did the imposed trace reach the solver's state"* is a question about internals L1 forbids asking, and that `exact_zeros` is only a necessary condition. **The measurement says the necessary condition is far too weak to carry a rule.** Being exactly diagonal is what a lumped expert *is*.

The discriminator §4.2 actually used — a **bitwise-identical field response across a range of imposed traces** — needs the expert's internal state. There is no L1-visible substitute in sight, and this tier did not find one.

> So **W301's status changes from *"open, with the detector identified and measured"* to *"open, and the proposed detector is refused as a rule"***, with a number and a named reason. What stays open is whether any L1-visible signature exists at all. The underlying finding — that both gas agents in this vault probe a wall their solver never sees — is untouched: it was established bitwise in §4.2 and does not depend on the detector.

### 11.3 And the sweep's own first version had the finding's shape

It probed `caps.ports[0]` only, and reported that the rocket's `b` did **not** trip — because at the `real` level that agent's first port is `a:MECH`, still a seeded random matrix, while the one backed by `compressible2d` is `c:THERM`. **A sweep that probes one port of a mixed record measures the port it happened to pick**, which is the same lesson §8 states about mixed records being labelled. Fixed to cover every port; the rates above are over all 141.

---


---

## 12. Tier 78 — W306 implemented, and it was not a rocket detail

§10.3 named W306: `passivity_defect` is $\lvert\lambda_{\min}\rvert$ and nothing read $\lambda_{\max}$, so a **one-signed negative** operator — passive up to a global sign that $S\lambda = \chi$ is indifferent to, because $\chi$ flips with $S$ — reported the same number as a genuinely **mixed** spectrum. This section closes it for the diagnostic, and the tier's own gate came first.

### 12.1 The gate: is the rocket the only place this matters?

A rule change that serves one case study is a fix, and should be called one. So before touching `probe.py`, every buildable case was compiled and every seam it probes classified — **60 seams across 8 cases**:

| | count |
|---|---|
| seams classified | $60$ |
| reporting a passivity defect | $\mathbf{4}$ |
| of those, **MIXED** — a genuinely amplified mode | $1$ |
| of those, **ONE-SIGNED** — a global sign | $\mathbf{3}$ |

The three one-signed ones are the rocket's `b-c:THERM` and **`car_graph`'s `J1_core_strip` and `J3_rotor_strip`** — the latter two on a graph that *does* declare `effort_normal` and has it **inverted**. The single mixed one is `thermal_strain`'s `thermal-pressure` seam, at $\lambda \in [-1.93\times10^{9},\ +3.23\times10^{11}]$.

**So three quarters of the passivity defects in this vault are sign conventions**, and the one the rule was written for is the minority case.

> **Since Tier 82 (W308, §12.4), and re-derived at §18.8:** two defects remain. One is the rocket's b–c, one-signed, whose defect is $0.4939$ with the fixed wall. The other is `thermal_strain`'s mixed one.

### 12.2 What was implemented, and what deliberately was not

`ProbedBlock` and `SeamOperator` now carry `passivity_lambda_max` and `sign_structure` $\in$ {`positive`, `negative`, `mixed`, `zero`}, classified against **the same tolerance the defect is clipped at** — so an eigenvalue at arithmetic-noise level does not turn every operator into `mixed`. Both are emitted, and `L4/E7/passivity` now names the structure and the remedy. On the rocket it reads:

> *"the spectrum is ONE-SIGNED NEGATIVE ($\lambda_{\max} = -2.193\times10^{-1} < 0$) … a global SIGN rather than an amplified mode … the seam declares `effort_normal='b'`, so try the OTHER side."*

**The verdict is unchanged, deliberately.** A negative-definite operator still does not give the $L \le 1$ branch and still needs fixing before it can, so it is still a decertification. Changing what the rule *decides* would move every artifact that has one, and the evidence for that change is not in hand. What changes is that the certificate now says which failure it is.

**A number that explains why the fix is cheap:** negating $S$ changes no singular value, so $\beta$ and $\kappa$ — the quantities the master bound actually uses — are untouched by the orientation. Only E7's test moves. That is pinned by `test_a_global_sign_leaves_the_singular_values_alone`.

### 12.3 What a schema extension costs a byte-identity control, measured

Two new keys on every seam move every artifact that has one: **7 of the 11 live keys** in the W189 control. Naming seven exemptions would have gutted it, so the live test now requires the difference to be **confined to the new keys** — delete them from both sides and the bytes must match again. That is a strictly stronger claim than *"these moved and I meant it"*, and it is mechanical rather than a list of names.

The two costs are separable and were measured apart:

| | live keys moved |
|---|---|
| compute the fields, do **not** emit them | $1$ of $11$ — only `thermal_strain`, whose *message* changed |
| emit them | $8$ of $11$ |

So the diagnostic is nearly free and the **publication** is what costs the control — which is the right way round, since publication is the whole point.

### 12.4 W308 — found, measured, deferred, and closed at Tier 82

`car_graph`'s two inverted declarations are a two-character fix: naming the other agent gives a **positive definite** operator with every singular value unchanged. `J2_heat` already names the right one, which is the control that makes this a finding rather than a blanket flip.

It was **not** fixed here. [[poc3-racelab-car-graph]] records those seams as earning `L4/E7/passivity` in *"the package's standing decertification set"*, alongside a decision count of $191$ decisions, $156$ admits, $34$ decertifications and one refusal. Correcting the declarations changes those counts, so it belonged with an audit of that page rather than as a side effect of a diagnostic tier. Pinned meanwhile by a test that asserts both structures, the flip's effect, and that `J2_heat` is untouched — so a future change cannot quietly alter any of it.

> **Closed 2026-09-21 (Tier 82).** The audit found those counts recorded in **exactly one sentence**, and the movement is as narrow as Tier 78 predicted: **$191$ decisions $\to$ $189$, decertifications $34 \to 32$, admits unchanged at $156$, the one refusal unchanged** — it was never these seams', it is the clocks. All three seams now read `positive`, `E7/passivity` fires on neither strip seam, and **every singular value is bitwise identical** (maximum relative difference $0.0$). That verifies on a real graph what §12 argued in principle: negating $S$ changes no singular value, so $\beta$ and $\kappa$ are untouched and only E7's test moves. `car_graph` is not in the W189 census, so the correction costs the byte-identity control nothing. **The blast radius is asserted by its own test**, which flips the declarations back and requires the old counts to reappear — so $191 \to 189$ stays a measurement of *this* change rather than of everything since.

**[AI Inference]:** the shape is likely general. A seam whose two sides both report in one shared direction needs `effort_normal`, and this vault now has four examples — one FSI (`front_wing`, W138), one CHT (the rocket), and two lumped-to-field joins (`car_graph`) — against zero counter-examples. Whether *every* such seam is one-signed rather than mixed is not established; `thermal_strain` shows a mixed spectrum exists.

---


---

## 13. Tier 79 — R0 complete, and what the other six agents cost to believe

Tier 76 wired two of the seven agents. This wires the remaining five — `a`, `e`, `d`, `f`, `g` — so **every agent's `boundary_response` now steps a build-repo solver on the build repo's own mesh, and nothing in the rocket graph is a seeded random matrix.** That is R0's completion.

It also found that **most of what it measured is not yet believable**, for a reason worth more than the numbers.

### 13.1 Two kinds of port, and they are not interchangeable

The b–c seam is a **wall**. Almost every other seam in this rocket is a **plane the flow crosses**, and they need different channels:

| | channel | trace | response |
|---|---|---|---|
| **wall** | `BC("wall_noslip", {"T_wall": ...})` | wall temperature | the conduction-limited flux `generate.py` exchanges |
| **plane** | `BC("prescribed", {"state": ...})` | total enthalpy $h_0$ | the mass flux $\rho\,\mathbf u\cdot\mathbf n$ through the face |

Agent `g` needs a third: it carries the plume as a **blanked hole**, not a face, so its `f` port drives `Compressible2D.hole_state` — a field whose own comment says *"set by the coupler"* and which nothing in this vault had ever set. The hole's ghost is **one state for the whole band**, so that port's effective resolution is genuinely 1, and it is declared as such rather than pretended otherwise.

### 13.2 The correction that matters most: Tier 76's cadence justification does not transfer

Tier 76 reduced the gas probe cadence and justified it by measuring that $\beta$ moved $0.3\%$ over a $100\times$ range. **That measurement was taken on a wall.** A wall's response is *algebraic* in the trace — `_wall_flux` subtracts $T_{\text{wall}}$ directly — so it converges at any cadence. A plane's response is the interior state after the imposed ghost has influenced it, which needs the wave to cross cells.

Measured side by side, on the nozzle exit plane and the chamber wall:

| `dt_scale` | sub-steps | **plane** `e→f`, $\max\lvert\partial R/\partial h_0\rvert$ | cells responding | **wall** `b→c`, $\max\lvert\partial R/\partial T\rvert$ |
|---|---|---|---|---|
| $10^{-3}$ | 8 | $4.66\times10^{-11}$ | 11 of 96 | $0.193867$ |
| $10^{-2}$ | 72 | $2.18\times10^{-10}$ | 29 of 96 | $0.193867$ |
| $10^{-1}$ | 720 | $1.43\times10^{-7}$ | **96 of 96** | $0.193722$ |
| $1$ (declared) | 7274 | $9.80\times10^{-6}$ | 96 of 96 | — |

**The wall is flat to four digits across the whole range. The plane moves five orders and has not converged at the declared clock.** Its *support* saturates at $10^{-1}$ — the wave finally crosses — while its *magnitude* keeps growing, because influence keeps accumulating.

> So **every ADVEC number in §13.3 is the operator the probe could see in 8 sub-steps, not the seam's operator.** They are reported because the difference between them and a converged one is the finding, not because they are measurements of a seam. The reduced cadence remains justified for wall ports, where it was measured; it was over-generalised to planes, and that was my error.

The price of doing it properly is measured, not estimated: one response call on agent `e` at its declared clock is $\approx 268$ s, so a full $\dim M = 41$ probe of `e-f` is 84 calls and about **7.2 hours**.

### 13.2a The anchor — one plane seam paid for, and what the cadence was hiding

To make the claim above load-bearing rather than rhetorical, `e-f` was probed twice at **the same declared interface space** ($\dim M = 8$, declared rather than budget-capped — see below) and two cadences a factor of $1000$ apart:

| | `dt_scale` $=10^{-3}$ | `dt_scale` $=1$ (declared) | ratio |
|---|---|---|---|
| CFL sub-steps | $45$ | $\mathbf{41\,445}$ | $921\times$ |
| $\beta$ | $6.79758\times10^{-7}$ | $\mathbf{2.91212\times10^{-5}}$ | $\mathbf{42.8\times}$ |
| $\kappa$ | $1.06819$ | $1.42432$ | $1.33\times$ |
| $\omega$ | $0.0212435$ | $0.0907455$ | $4.27\times$ |
| `e`'s share of the seam | $0.0637$ | $\mathbf{0.362}$ | $5.68\times$ |
| wall cost | $1$ s | $2786$ s | $2786\times$ |

The **sign structure is `positive` at both**, the null dimension is $0$ against a declared $0$ at both, and neither is empty — so the *qualitative* reading of the seam survives the reduced cadence. Every *magnitude* does not.

> **W334 (§18.8):** with the fixed wall, $\beta$ goes $6.98986\times10^{-7} \to 3.37917\times10^{-5}$, a factor of $48.3$, and `e`'s share goes $3.7\% \to 24.9\%$. The converged $\dim M = 41$ probe now agrees: $53.8\times$ and $25.0\%$. The argument below stands with smaller numbers.

> **The block share is the consequence that bites.** At the reduced cadence agent `e` carries $6.4\%$ of the assembled operator and reads as a passenger; converged it carries $36\%$. [[tier0-measurements]] W76 established that a substitution certificate can only see a perturbation bounded by the swapped agent's own block, so **a certificate taken at the reduced cadence would be blind to `e`** — it would admit any replacement of the nozzle, whatever the replacement did. That is not a small quantitative error; it is the difference between a test and a formality.

**And a note on how the anchor was made affordable, because the obvious route does not work.** `ProbeBudget(max_modes=8)` shrinks the interface space and leaves the **declared** prolongation at its original width, so `Prolongation.prolong` raises `dim mismatch`: a budget cannot narrow a space somebody else declared. The narrower space has to be *declared*, which `build_rocket_real(m_cap=...)` now does — and being a declaration, it is reported: $\dim M = 8$ against the $41$ this seam's meshes support, so the anchor is a number about a coarser interface than the graph's own.

### 13.3 The seven seams, probed

At `dt_scale` $=10^{-3}$, **with the ADVEC rows carrying §13.2's caveat**:

| seam | type | $\dim M$ | $\beta$ | $\kappa$ | $\omega$ | sign | null | cost |
|---|---|---|---|---|---|---|---|---|
| a–b | ADVEC | 41 | $8.75\times10^{-7}$ | $1.0052$ | $0.00089$ | positive | $0=0$ | 84 calls |
| e–b | ADVEC | 41 | $5.18\times10^{-8}$ | $16.885$ | $0.1626$ | positive | $0=0$ | 84 calls |
| **b–c** | **THERM** | 8 | $\mathbf{0.21934}$ | $2.3131$ | $0.3178$ | negative | $0=0$ | 18 calls |
| **c–d** | **THERM** | 93 | $\mathbf{0.13603}$ | $1.1358$ | $0.01988$ | positive | $0=0$ | 94 calls |
| e–f | ADVEC | 41 | $5.58\times10^{-7}$ | $1.3018$ | $0.0368$ | positive | $0=0$ | 84 calls |
| d–g | ADVEC | 25 | $9.73\times10^{-17}$ | $6.27\times10^{11}$ | $0.6111$ | mixed | $0=0$ | 52 calls |
| **g–f** | ADVEC | 77 | $2.58\times10^{-12}$ | $33.786$ | $0.7039$ | mixed | $0=0$ | **EMPTY** |

> **W334 (§18.8):** with the fixed wall, e–b reads $\beta = 6.79\times10^{-7}$, $\kappa = 1.28$, and e–f moves $12\%$. a–b, b–c, c–d and g–f move under $2\%$. d–g reads `positive` at a $\beta$ of $9\times10^{-17}$, where the sign of the smallest eigenvalue is round-off.

**The null check runs, and passes, on all seven** — which it could not do on `rocket.py`'s own graph, because those connections declare no `expected_null_dim` at all and `excess_null_directions` is therefore `None` right across it.

**And `g-f` is empty: $\Lambda = 0$.** Not ill-conditioned — *empty*, which `CASE-STUDY-GUIDE` lists as mistake 6. The cause is a declaration error and it is exact: `rocket.py` declares ADVEC on that seam, the shear layer's normal is precisely $+y$, the flow is $+z$, so $\rho\,\mathbf u\cdot\mathbf n \equiv 0$ for **any** trace. Perturbing $h_0$ by 1000 J/kg moves the response by $4.2\times10^{-10}$, which is the solver's own transverse-velocity noise. **The config agrees**: its `edge_list` gives `g-f` the types `(heat, fluid)`, not `conservation`.

### 13.4 W309 — `grid.Block`'s face normals are index-oriented, so every gas seam needs `effort_normal`

Tier 77 derived `effort_normal = "b"` for b–c from the geometry of one wall. Measured across every gas block, the situation is general:

$$\mathbf n_i[0] = \mathbf n_i[-1] = (+1, 0), \qquad \mathbf n_j[:,0] \cdot \mathbf n_j[:,-1] > 0$$

They are $[e_y, -e_x]$ and $[-e_y, e_x]$ of the edge vectors — a **consistent orientation, not an outward normal**. So on every seam between two blocks, both sides report their flux against the same physical direction, which is exactly the shape `assemble_seam`'s W138 note describes and where the Steklov–Poincaré sum is the wrong object.

The rule that follows — **`effort_normal` names the agent for which the seam is its `imax` or `jmax` face** — reproduces the b–c answer that was derived independently a tier earlier, which is what makes it a rule rather than a pattern. `c-d` is the one edge it cannot reach, because `c` is a `ThermoStruct2D` with no block; there the shell is named, since `d`'s `jmin` index normal $+y$ *is* the shell's outward normal.

### 13.5 W311 — a second undeclared interface, and its consequence measured

The airframe `c` spans $z \in [-4.0, 0.70]$ and the nozzle `e` spans $[0.40, 0.70]$. At $z = 0.55$, `domains.contains` puts the nozzle interior in `e` and the shell thickness in `c`: **they share a boundary and no edge declares it.** `wall_b_c` stops at the throat and `wall_c_d` is the *outer* wall, so no interface curve runs along the diverging nozzle's inner surface at all.

This is the second one. `contours.plane_d_g`'s own docstring already records *"a real, currently UNDECLARED d-f interface"*. So **`domains.verify` checks that the agents partition the box and that every declared edge lies on both its agents' boundaries — and nothing checks the converse**, that every shared boundary has an edge.

**The consequence was predicted and the prediction was refuted.** I predicted the nozzle wall's film coefficient would exceed the chamber's by more than $2\times$; measured:

| | $\bar h$ | min | max |
|---|---|---|---|
| chamber wall (declared) | $190.81$ | $159.26$ | $365.16$ |
| **nozzle wall (undeclared)** | $\mathbf{281.27}$ | $171.22$ | $507.81$ |

a ratio of $\mathbf{1.4741}$, not $>2$. So the omission is **material but smaller than claimed**: 47% more heat transfer per unit area, over the 0.30 m of wall that is the hottest part of the engine, into a shell whose own `validity` predicate already declines at 10.35 s of burn.

> **W334 (§18.8):** $1.332$. With the wall conducting, the chamber wall's $\bar h$ rose to $211.30$ and the nozzle wall's stayed at $281.35$. The shell's predicate now declines at $2.95$ s.

### 13.6 W301's signature, cross-checked — and it is about `wall_noslip`, not about the solver

Tier 76 found the isothermal wall saturated. If that were a property of `compressible2d` rather than of that one boundary condition, the plane ports would show it too. They do not:

| | ports | tripping `exact_zeros == n-1` |
|---|---|---|
| **wall** | 2 | 1 |
| **plane** | 10 | 1 |

and **both exceptions are informative**:

- `d.c:THERM` is a wall and does **not** trip. Same BC, same solver — but `d` is the atmosphere at 255.7 K against a 255.7 K wall, so the clamp threshold $(T_i + 20)/2 \approx 138$ K is far below the imposed trace and the channel is **not saturated**. That is the mechanism confirmed from the other side: saturation needs a wall much colder than the gas, which is the engine's situation and not the atmosphere's.
- `f.g:ADVEC` is a plane and *does* trip, **vacuously** — its response is the identically-zero shear-layer operator of §13.3, so "one cell responds and no others" is a statement about noise.

> **W334 (§18.8):** with W301 fixed, `b.c:THERM` moves 10 of 112 cells and no longer trips, so no wall port does. `f.g` still trips, vacuously.

### 13.7 W313 — every $\beta$ in this vault is in raw physical units

`Prolongation` carries a `nondim_diag`, `interface-transfer-theory` §6 specifies it as the diagonal unit conversion $D_i$ with $P_i = D_i^{-1}\hat P_i$, and **no case in this vault sets it** — grep returns zero.

So a $\beta$ here is in whatever units its port's bond happens to have: $0.219$ on a THERM seam is $\mathrm{W/(m^2\,K)}$ per kelvin, and $8.75\times10^{-7}$ on an ADVEC seam is $\mathrm{kg/(m^2\,s)}$ per $\mathrm{J/kg}$. **They are not comparable, and nothing in the certificate says so.** Every *within-seam* comparison this vault has published — base against base, convention against convention, cadence against cadence — is unaffected, because the units are constant across those. What is not safe is reading a table of $\beta$ across seams of different port types as though the numbers ranked anything.

### 13.8 The graph, compiled

Seven agents, seven seams, all real:

| | fixture (Tier 75) | **real (Tier 79)** |
|---|---|---|
| verdict | `refuse` | `refuse` |
| refusals | 10 | **4** |
| decertifications | 49 | **22** |

> **W334 (§18.8):** `refuse`, 4 refusals, **20** decertifications with the fixed wall. `L4/block-share` goes $3 \to 2$, and `L4/E7/passivity` goes $2 \to 1$ as d–g's round-off sign flips.

The four refusals are `L2/InterfaceMotion` $\times 3$ and `L5/R6` $\times 1$ — **and the second one is the framework catching this tier's own declaration error, unprompted.** §13.3 found the g–f ADVEC operator empty by hand; `L5/R6` refuses the graph for it independently, and names the reason:

> *"newton-krylov and direct-schur require a nonzero transmission operator; otherwise the interface system is not ill-conditioned but empty. Seams `['g-f']` assemble to a zero block … iterating an empty interface problem converges instantly and means nothing — the degenerate-axis trap, generalized: an axis whose value cannot affect the answer must be refused when the configuration is read, not discovered by measuring it."*

That is the PoC's stated product demonstrated on its own author: a coupling that would have run, produced numbers and meant nothing, **refused with the reason** — and refused at *compile* time, which is exactly what the last clause of that message asks for.

**The InterfaceMotion count is 3 rather than 9 because this graph declares one port type per seam where the fixture declared three** — the same three moving faces, the combustion front and the two plume boundaries, refusing for the same reason.

> **And they were briefly *gone*, which is the most useful mistake in this tier.** The first version of `build_rocket_real` declared `motion_class=STATIC` on every port, and the compile came back with **1 refusal instead of 4**. That is the "declare it away" failure in its purest form: the one genuine research hole in this graph, removed by a default in a builder rather than by any argument. It is now carried from `rocket.AGENTS` explicitly, and `PORT_MOTION` says why in the code.

---

## 14. Tier 80 — R2's remaining half: the composed defect at 50:1, and the bound that breaks at it

> **Re-derived at §18.8 (W334), and the headline reverses.** With the fixed wall, **the bound HOLDS at the declared interval**, $2.2782\times$ loose. $\sigma$ is first order over the whole $250\times$ (exponents $0.9997$–$1.0021$), and there is no knee. The constants are $s_\Gamma = 1.1627\times10^{-3}$, $C_2 = -1.228\times10^{-6}$ and $\dot\lambda = 32.20$ K/s, and 10 of 12 predictions hold. This section keeps what Tier 80 recorded on the old wall. Its method — anchors, registered extrapolations, the ratio control — is what found the reversal.

R2 asked for two things: declare `flux_matching=TIME_INTEGRATED` with a `boundary_response_integrated` on each side, and **report the composed defect at 50:1 beside CS-11's bound**. The first was declared at Tier 76. It did not *clear* until Tier 79, and that dependency is the first thing worth stating.

### 14.1 R2's gate could not clear until R0 closed

Tier 76's residue (§6.2) carries `L7/R9/quadrature` as a **refusal** — *"`d`, `e`, `f` supply no `boundary_response_integrated`"*. The declaration was in place and the graph refused anyway, because three of the seven agents were stubs. Tier 79 wired them. Only then:

> `L7/R9` **admit** — *"time-integrated flux matching over an exchange interval of $0.05$, with each side quadrature'd on its own clock: `b x50000, c x1, d x10000, e x50000, f x10000`. The clocks nest, every side supplies `boundary_response_integrated`, and `solve.coupled_step` calls it."*

The agents at a multirate seam are `['b', 'c', 'd', 'e', 'f']`, which is W194's narrowing of R9's premise doing its work — the rule asks the agents *at* a multirate seam, not all seven. The quadrature counts above are at the probe cadence $\texttt{dt\_scale} = 10^{-3}$; at the declared clocks they read `b x50, d x10, c x1`.

> **A declaration is not a capability.** R2's first half was written down three tiers before anything in the graph could satisfy it, and nothing in the declaration said so — the refusal did.

What survives is `R9/lag` and `R9/order`, both `admit-uncertified`. **`R9/lag` is the term this tier measures**, and its own message names the exit: *"declare sigma with the `sigma_lag` it was measured at and expect it to be the term that decides this graph."*

### 14.2 What CS-11's bound is, and what had to be re-measured to use it

[[case-study-brake-thermal-atlas-0.1]] §3 supplies

$$\sigma(\Delta t_{\text{ex}}) \;\le\; s_\Gamma\,\lVert\delta\lambda\rVert \;+\; C_2\,\lVert\delta\lambda\rVert^2, \qquad \lVert\delta\lambda\rVert \;\le\; \dot\lambda_{\max}\,\Delta t_{\text{ex}}$$

with **two constants it calls properties of the SEAM** and **one rate it calls a property of the RUN**. If that split is real, the constants must be re-measured here and the form must still hold. If it is not, CS-11 is a description of one brake. Three things make the rocket a genuinely different test.

**The referent is built against the declaration, and the alternative is priced rather than hidden.** Both columns march the same gas from the same state over the same interval and sub-step it the same number of times; the only difference is the trace it is handed — the referent's own path from a tightly coupled rollout at the gas's clock, or its value at the start held constant. The functional is **the gas's own wall flux in both columns**, because CS-11 found that reading the slope through one side while reading the defect through the other made its bound over-predict by $5.7\times$–$8.6\times$ and look exactly like a lag-profile effect.

**The chamber has a steady state and the shell does not, so one side is settled and the other is marched.** The gas is a fed reservoir with an outflow, so it settles: pre-marched $6\times10^{-3}$ s of gas time (about one flow-through at $0.30$ m and $50$ m/s), with the relative state change falling $7.49\times10^{-3} \to 2.32\times10^{-4} \to 9.05\times10^{-5}$ over three checkpoints. The shell is marched $5.00$ s of burn, as Tier 76 established it must be.

**The ratio is pinned at 50 in every row and the interval moves.** CS-11 pinned the interval and moved the ratio; this is the complement, so the two statements check each other on a second seam. §14.6 runs CS-11's own version as well.

### 14.3 The rocket's own constants — and one CS-11 never tested

| shift [K] | $\sigma$ | $\sigma/$shift | log ratio |
|---|---|---|---|
| $0.01$ | $1.22289\times10^{-5}$ | $1.22289\times10^{-3}$ | — |
| $0.03$ | $3.66857\times10^{-5}$ | $1.22286\times10^{-3}$ | $1.0000$ |
| $0.1$ | $1.22274\times10^{-4}$ | $1.22274\times10^{-3}$ | $0.9999$ |
| $0.3$ | $3.66723\times10^{-4}$ | $1.22241\times10^{-3}$ | $0.9998$ |
| $1$ | $1.22126\times10^{-3}$ | $1.22126\times10^{-3}$ | $0.9992$ |
| $3$ | $3.65391\times10^{-3}$ | $1.21797\times10^{-3}$ | $0.9975$ |
| $10$ | $1.20650\times10^{-2}$ | $1.20650\times10^{-3}$ | $0.9921$ |

$$s_\Gamma = \mathbf{1.222889\times10^{-3}}\ \mathrm{K}^{-1}, \qquad C_2 = \mathbf{-1.639139\times10^{-6}}\ \mathrm{K}^{-2}$$

**$C_2$ is negative, and CS-11's is $+1.4288\times10^{-5}$.** The rocket's $\sigma$ is *sub*-linear in a uniform shift where the brake's was super-linear — the log-ratio column falls away from $1$ rather than rising. So the second-order term bends the two seams in opposite directions. It stays a correction in the range that matters (at a $1.3$ K lag it is $0.17\%$ of the linear term), but *"CS-11's bound form transfers"* is a claim with a caveat: **the quadratic coefficient does not even keep its sign.**

**And the test CS-11 did not run.** CS-11 measured $s_\Gamma$ at one interval and then *used* it as a constant in the interval, which is what lets a bound be checked cheaply and spent expensively. Whether it **is** one was never checked. Here it is:

| $\Delta t_{\text{ex}}$ [s] | $s_\Gamma$ | rel to base |
|---|---|---|
| $5\times10^{-5}$ | $1.22002\times10^{-3}$ | $0.9977$ |
| $2\times10^{-4}$ | $1.222889\times10^{-3}$ | — |
| $1\times10^{-3}$ | $1.23530\times10^{-3}$ | $1.0101$ |

> **$s_\Gamma$ moves by $1.0125\times$ over $20\times$ of exchange interval.** That is the licence the extrapolation needs, and without it the anchor's bound would rest on an assumption rather than a measurement.

### 14.4 The run-derived constant carries a burn time, and the cost of not saying which is $2.58\times$

CS-11 measured $\dot\lambda$ at a settled state. The rocket's shell has none — Tier 76 renamed `settle()` to `march()` for exactly that reason — so the rate is a function of when in the burn it is taken:

| burn [s] | wall mean [K] | $\dot\lambda$ [K/s] | rel to 5 s |
|---|---|---|---|
| $0.5$ | $316.7842$ | $5.15996\times10^{1}$ | $1.9877$ |
| $1$ | $335.2537$ | $4.38490\times10^{1}$ | $1.6891$ |
| $5$ | $449.4197$ | $\mathbf{2.59594\times10^{1}}$ | $1.0000$ |
| $10$ | $557.6469$ | $1.99816\times10^{1}$ | $0.7697$ |

$\dot\lambda$ spans $\mathbf{2.582\times}$ over the burn window, against CS-11's single settled $11.94$ K/s — the rocket's is $2.174\times$ CS-11's at the declared $5$ s base. Worth setting beside Tier 76's §7.2: **the rate is about twice as horizon-sensitive as $\beta$ was** ($2.58\times$ against $1.265\times$ over the same window), so a $\sigma$ quoted without its burn time is looser than a $\beta$ quoted without one.

### 14.5 The defect, at a pinned 50:1, over 250x of interval

| $\Delta t_{\text{ex}}$ [s] | ratio | lag [K] | measured $\sigma$ | bound | bound/measured | order | cost |
|---|---|---|---|---|---|---|---|
| $4\times10^{-6}$ | $1$ | $0$ | $\mathbf{0}$ | $0$ | — **control** | — | — |
| $2\times10^{-4}$ | $50$ | $5.17895\times10^{-3}$ | $2.53345\times10^{-6}$ | $6.33324\times10^{-6}$ | $2.4999$ | — | $28$ s |
| $1\times10^{-3}$ | $50$ | $2.58905\times10^{-2}$ | $1.39546\times10^{-5}$ | $3.16601\times10^{-5}$ | $2.2688$ | $1.060$ | $135$ s |
| $5\times10^{-3}$ | $50$ | $1.29434\times10^{-1}$ | $6.67552\times10^{-5}$ | $1.58256\times10^{-4}$ | $2.3707$ | $0.973$ | $652$ s |
| $\mathbf{5\times10^{-2}}$ **(declared)** | $50$ | $1.292034$ | $\mathbf{5.837575\times10^{-3}}$ | $1.577278\times10^{-3}$ | $\mathbf{0.2702}$ | $\mathbf{1.9417}$ | $9259$ s |

> **Over the first $25\times$ the bound holds and is loose by $2.269\times$ to $2.500\times$, against CS-11's own $1.426\times$–$2.178\times$ on the brake. At the declared interval it is VIOLATED: $\mathbf{0.2702}$, under-predicting the measured defect by $\mathbf{3.7\times}$.**
>
> **W334 (§18.8): that violation was the old wall's.** Re-derived, the declared row reads lag $1.581489$ K, $\sigma = 8.05768\times10^{-4}$, bound/measured $\mathbf{2.2782}$ and order $0.9997$. $\sigma/$lag is $5.055$–$5.095\times10^{-4}$ over all $250\times$.

**That row cost $2.57$ hours and it is the only reason any of this is known.** The cheap sweep is internally consistent, first order, and agrees with CS-11 to within a factor — and it is wrong about the one interval the graph actually runs at. A bound published from §14.5's first three rows would have carried $25\times$ of clean supporting evidence and failed at the declared clocks.

**The failure is not in the lag, and that is what makes it a finding rather than a measurement error.** Three quantities were checked across the whole $250\times$:

| | $2\times10^{-4}$ | $1\times10^{-3}$ | $5\times10^{-3}$ | $5\times10^{-2}$ |
|---|---|---|---|---|
| lag$/\Delta t_{\text{ex}}$ [K/s] | $25.8948$ | $25.8905$ | $25.8869$ | $25.8407$ |
| profile peakedness | $1.6839$ | $1.6835$ | $1.6833$ | $1.6811$ |
| $\sigma/$lag | $4.8918\times10^{-4}$ | $5.3899\times10^{-4}$ | $5.1575\times10^{-4}$ | $\mathbf{4.5181\times10^{-3}}$ |

**The trace drift is linear to $0.2\%$ over $250\times$ and the profile's shape does not move. The gas's response per unit lag jumps $8.8\times$ in the last decade**, taking the order from $1.060$ and $0.973$ to $\mathbf{1.9417}$ — first order to very nearly second. So $\sigma$ stops being first order in the interval somewhere between $5\times10^{-3}$ and $5\times10^{-2}$ s, and **no choice of $C_2$ repairs it**: the rocket's $C_2$ is *negative* (§14.3), so the quadratic term pulls the bound further down, not up.

**Ratio 1 is the control and its defect is exactly zero, bitwise.** One gas call *is* the interval, so there is nothing to be stale about. It carries no tightness ratio, and quoting $0/0$ as one would be W106's mistake.

**First order in the interval — over the cheap sweep, and only there**, at $1.060$ and $0.973$. Over that range it joins CS-9's volumetric splitting ($1.003$), CS-10's field-to-lumped lag ($1.258$, $1.165$) and CS-11's multirate lag: four case studies, four coupling kinds, order one. **The declared row leaves that family at $1.9417$**, which is the first time anything in this vault has measured a coupling defect *out of* first order rather than into it — and it took an interval an order of magnitude beyond where any of the four checked.

**A cross-check that fell out unasked.** The lag scales as $5.17895\times10^{-3} \to 2.58905\times10^{-2} \to 1.29434\times10^{-1}$ for intervals $5\times$ apart — ratios of exactly $5.0$ and $5.0$ — so $\dot\lambda$ derived from the sweep is $25.89$ K/s against the $25.9594$ K/s the shell's own single step gave in §14.4. **Two unrelated routes, $0.3\%$ apart.**

**And the looseness has a mechanism, quantified rather than attributed.** CS-11 ascribed its own $1.4$–$2.2\times$ to **W86** — $s_\Gamma$ is measured by displacing the trace *uniformly* and a run's lag is not uniform. Here that is a number. The lag profile's peakedness $\lVert\delta\lambda\rVert_\infty / \lVert\delta\lambda\rVert_{\text{rms}}$ is $1.6839$, $1.6835$, $1.6833$ across the three intervals — **fixed**, so the profile scales with the interval without changing shape. And the measured $\sigma/\text{lag}$ is $4.89$–$5.39\times10^{-4}$ against $s_\Gamma = 1.2229\times10^{-3}$: the real non-uniform lag produces $\mathbf{40\%}$–$\mathbf{44\%}$ of what a uniform shift of the same norm would. The reciprocal, $2.27$–$2.50$, **is** the tightness column for those three rows. **So over the cheap sweep the bound's looseness is the profile effect and nothing else** — which is why it looked so well understood, and why the declared row is not a bigger version of the same thing: there the profile is unchanged ($1.6811$) and the response per unit lag has moved instead.

### 14.6 CS-11's own control, replicated — and 1900x sharper

Pin the interval and move the ratio by refining the gas's own stability step through its CFL number:

| CFL | sub-steps | $\Delta t_{\text{stable}}$ [s] | ratio | $\sigma$ | relative |
|---|---|---|---|---|---|
| $0.40$ | $1400$ | $2.85714\times10^{-7}$ | $700$ | $2.533447\times10^{-6}$ | $1.000000$ |
| $0.20$ | $2800$ | $1.42857\times10^{-7}$ | $1400$ | $2.533447\times10^{-6}$ | $1.000000$ |
| $0.10$ | $5500$ | $7.27273\times10^{-8}$ | $2750$ | $2.533447\times10^{-6}$ | $1.000000$ |

> **$3.93\times$ of clock ratio at a fixed interval moves $\sigma$ by $1.0000000836\times$.** CS-11 measured $8\times$ of ratio moving it by $1.000160\times$.

**A reading of exactly $1.000000$ is the signature of a floor, so it was checked rather than quoted.** The stored values are $2.53344704701930033$, $2.53344721751540963$ and $2.53344725881998366 \times 10^{-6}$: **pairwise different, none bitwise equal**, with a maximum relative spread of $8.360\times10^{-8}$. The numbers genuinely move and genuinely do not matter. Had they been bitwise identical, the control would have been vacuous for the same reason **W301** makes the saturated channel vacuous, and this table would say nothing.

The scope CS-11 attached still applies and is worth repeating: this holds for a fast agent that **sub-steps internally at its own stability limit**, which `Compressible2D` does. A frozen learned expert whose `dt_native` is fixed by its weights is the case where the ratio re-enters, and neither case study can see it.

### 14.7 The control that makes "the bound holds" mean something

If CS-11's own constants happened to bound the rocket too, §14.5 would be evidence of nothing.

| $\Delta t_{\text{ex}}$ [s] | measured $\sigma$ | rocket bound | ratio | **CS-11's bound** | ratio |
|---|---|---|---|---|---|
| $2\times10^{-4}$ | $2.53345\times10^{-6}$ | $6.33324\times10^{-6}$ | $2.4999$ | $4.91828\times10^{-5}$ | $\mathbf{19.4134}$ |
| $1\times10^{-3}$ | $1.39546\times10^{-5}$ | $3.16601\times10^{-5}$ | $2.2688$ | $2.45881\times10^{-4}$ | $\mathbf{17.6200}$ |
| $5\times10^{-3}$ | $6.67552\times10^{-5}$ | $1.58256\times10^{-4}$ | $2.3707$ | $1.22943\times10^{-3}$ | $\mathbf{18.4169}$ |

**P6 predicted CS-11's constants would fail to bound the rocket, and it was refuted: they bound it, by $17.6\times$ to $19.4\times$.** The honest reading is not *"the bound transfers"*:

> The two seams' constants differ by $7.8\times$ ($s_\Gamma$ ratio $0.1288$; $C_2$ ratio $-0.1147$, sign included), **which is what proves they are seam properties**. Borrowing happened to land on the conservative side here. That is the *sign of the discrepancy*, not a property of the method — had the rocket's seam been the more sensitive one, the same borrowing would have been unsafe by the same factor. **A borrowed $s_\Gamma$ is a coin flip, and this one landed safe.**

### 14.8 W315 — the build repo's shell-to-gas trace is rank 1

`generate.py::_wall_T` returns

```python
return float(np.mean([T.mean() for T in self.T_shell]))
```

**one number, the mean over every panel of every station**, and hands it to every gas agent's `wall_noslip` BC. The other direction is per-station, by `np.interp` on normalized index. So the build repo's own coupler transports a **rank-1** shell-to-gas trace, while this vault probes that seam at $\dim M = 8$ and Tier 76 reported $\beta$, $\kappa$ and $\omega$ on it.

The wall the gas would see per station is $369.3286$–$690.0180$ K, a spread of $320.6894$ K. The wall `generate.py` hands it is $449.4197$ K.

| $\Delta t_{\text{ex}}$ [s] | $\sigma$, per-station trace | $\sigma$, `generate.py`'s scalar | ratio |
|---|---|---|---|
| $2\times10^{-4}$ | $2.53345\times10^{-6}$ | $3.11872\times10^{-6}$ | $1.2310$ |
| $1\times10^{-3}$ | $1.39546\times10^{-5}$ | $1.62731\times10^{-5}$ | $1.1661$ |

So the collapse costs $17\%$–$23\%$ on this tier's quantity — material, and much smaller than the rank suggests. **The reason it is not worse is §14.5's mechanism**: $\sigma$ responds to the lag's norm far more than to its shape, which is the same fact that keeps the bound only $2.3\times$ loose.

> **The declaration and the implementation disagree about the dimension of the interface, and the vault's side is the richer one.** `Compressible2D`'s `wall_noslip` accepts a per-face array — Tier 79 relied on it — so this is a capability the solver has and the coupler does not use. Every $\dim M = 8$ number this case study has published on b–c describes a coupling the build repo would have to be changed to realise.

### 14.9 W316 — the shell is linearised about a gas temperature the gas does not have

`ShellAgent.march` drives the shell with `T_CHAMBER` and `base_trace` returns the same $2800$ K. Measured at the settled state, the gas's own near-wall cell is **$1490.151$–$2804.191$ K, mean $2464.176$ K** — a ratio of $0.8801$ in the mean and a factor of $1.88$ across the window. The two sides' **film coefficients** agree to under a percent ($183.872$ from the march against $185.339$ settled); their **temperatures** do not.

This is W74's class on the one direction of this seam nobody had checked. Tier 77 measured the other direction at $876\%$ of the base norm and it moved $\beta$ by $114\times$; this is the same class, one direction along, priced on $\sigma$:

| $\Delta t_{\text{ex}}$ [s] | declared trace | two-way referent | ratio |
|---|---|---|---|
| $2\times10^{-4}$ | $2.53345\times10^{-6}$ | $9.96800\times10^{-7}$ | $0.3935$ |
| $1\times10^{-3}$ | $1.39546\times10^{-5}$ | $1.53224\times10^{-5}$ | $1.0980$ |

**The ratio changes direction between the two intervals, and that is the finding rather than a wobble.** Driving the shell with the gas's real near-wall temperature puts a **step** into its boundary condition at $t=0$ — the state was marched under $2800$ K and the referent hands it $2464$ K — and a startup transient decays over a fixed time, so it dominates a short interval and washes out of a long one. Spread across the two: $2.79\times$. That is why the declared trace is the default here and the two-way version is a control: **a referent that is more physical and less consistent with the state it starts from is not a better referent.**

> **W334 (§18.8):** with the fixed wall the gap is 44 K rather than 336 K, and the two-way referent moves $\sigma$ by $0.101\times$ and $0.160\times$, no longer changing direction. The startup-step reading carries at the new size, on the arithmetic §18.8 gives: a 3% flux step on a half-millimetre face node. Closing it properly means marching the shell under the gas's own near-wall temperature from the start, which moves the probe state every number since Tier 76 was taken at.

### 14.10 W317 — a third undeclared interface, and the declared seam is the minority

W311 derived `e-c` from the geometry; `contours.plane_d_g`'s docstring recorded `d-f`. This one is named in **the build repo's own source**, as the stated reason `_wall_T` returns a scalar:

> *"the per-station map is available (`T_shell`) but the gas agents' wall BC takes a per-face array only on the faces that touch the shell, and `a`'s lateral walls are an **UNDECLARED a-c interface** (see the implementation log)."*

Measured on the shell's own inner face ($232$ cells, $4.7254$ m of arc), taking each cell by its midpoint:

| window | cells | arc [m] | share of the engine-side wall |
|---|---|---|---|
| **a-c**, undeclared | $6$ | $0.121552$ | $16.6\%$ |
| **b-c**, declared | $14$ | $0.299018$ | $\mathbf{40.7\%}$ |
| **e-c**, undeclared | $15$ | $0.313893$ | $42.7\%$ |
| engine-side total ($z \in [0, 0.70]$) | $35$ | $0.734463$ | — |

> **The one declared conjugate seam covers $40.7\%$ of the wall that actually has gas behind it, and $6.3\%$ of the shell's inner face.** W311 reported one undeclared interface; with `a-c` the undeclared majority is $59.3\%$ — and W311 already measured `e-c` as the hotter part ($\bar h = 281.27$ against the chamber's $190.81$).

**And the build repo knows.** Its own comment names an interface its own config does not declare, and the consequence — a scalar wall temperature for every gas agent (§14.8) — is not a shortcut but a *forced* one: a per-face array cannot be handed to a face whose interface has no edge.

### 14.11 The anchor — the declared interval, and the bound it breaks

Every row in §14.5 is at the declared $50{:}1$, but the declared **exchange interval** is $5\times10^{-2}$ s — the shell's own step, which is where `R4` floors it — and the sweep stops $10\times$ short of it. That row is the one the gate names, and it costs what it costs: the chamber marches at $65\,204$–$70\,070$ s of wall per second of gas time on this machine, measured per row rather than estimated, so two columns of $5\times10^{-2}$ s was budgeted at $\approx 1.8$ hours. **It cost $2.57$**, and the miss is itself a number worth keeping — see below.

**Before it ran, the core's three rows were made to commit to it**, so the anchor can refute the extrapolation rather than decorate it:

| | registered | measured | |
|---|---|---|---|
| **A1** | lag $\in [1.22958,\ 1.35902]$ K | $\mathbf{1.292034}$ | **held** |
| **A2** | $\sigma \in [5.5\times10^{-4},\ 8.0\times10^{-4}]$ | $\mathbf{5.837575\times10^{-3}}$ | **failed, $8.7\times$ high** |
| **A3** | bound/measured $\in [2.0,\ 2.8]$ | $\mathbf{0.2702}$ | **failed** |

The disjunction was registered before the run:

> *If **A1 holds and A2 fails**, $\sigma$ is not first order out to the declared interval, and the bound's usefulness stops somewhere between $5\times10^{-3}$ and $5\times10^{-2}$. That is the most interesting outcome available here, and it is why the anchor is worth the time.*

> **W334 (§18.8):** re-derived, **A3 holds** at $2.2782$. A1 and A2 fail on their registration: they were set from the old core's $25.89$ K/s, and the fixed wall heats the shell at $31.70$. The rule they encode lands within $0.23\%$ on the lag and $0.57\%$ on $\sigma$. The disjunction below was answered by the wall, not by the seam.

**A1 held to four digits and A2 failed by $8.7\times$.** So the extrapolation's *premise* — that the trace drifts linearly, at a rate the shell's own single step predicts — is exactly right, and its *conclusion* is wrong, because the quantity that stopped being linear is the seam's response and not the trace.

**Cost, measured rather than estimated:** $9259$ s ($2.57$ h), $364\,093$ gas sub-steps, **zero Modern Standby events** over the window, at $100\%$ charge on AC. That is $92\,590$ s of wall per second of gas time, about $40\%$ above the sweep's own $65\,204$–$70\,070$ — the chamber accelerates as it develops, so a longer march is dearer per unit of gas time, not merely longer.

> **What this costs the rest of the vault.** [[case-study-brake-thermal-atlas-0.1]] §4's argument is that a bound written in the *interval* rather than the *ratio* can be checked at a ratio you can afford and applied at the $10^4$–$10^5$ a real conjugate seam runs at. That argument survives — §14.6 confirms the ratio is not the variable, $1900\times$ more sharply than CS-11 could. **What does not survive is extrapolating in the INTERVAL**, and CS-11 never claimed to; its own table spans $400\times$ of interval and stops. The rocket shows what happens one decade past where the checking stopped, and the answer is that the order changes.

**And the mechanism is bounded but not measured.** The chamber is $0.30$ m at a $50$ m/s inlet, so one flow-through is $\approx 6\times10^{-3}$ s: the sweep's largest interval is $\mathbf{0.83}$ flow-throughs and the anchor is $\mathbf{8.3}$. **[AI Inference]:** the order changes when the exchange interval crosses the fast agent's own residence time, because below it the gas cannot carry a wall perturbation out of the domain and above it the perturbation is advected through repeatedly. What supports this is the arithmetic above and the fact that nothing about the trace moved; what would settle it is $s_\Gamma$ measured at an interval inside the gap — flat means the growth is in how the run's lag couples, climbing means the seam's own linear constant stops being constant. That measurement is one stage (`--stages knee`), four marches of $10^{-2}$ s — $1.67$ flow-throughs, placed to straddle the threshold — and **$\approx 70$ minutes on AC**.

**It was attempted and it is not in this record.** The run died ten minutes in, during its settle, with an empty stderr and no Modern Standby event to blame, and **the cause is not known** — the anchor had survived nearly three hours by the identical launch. So the mechanism is named, priced and unpaid: it refines an inference, and the gate does not turn on it.

### 14.12 The predictions

> **W334 (§18.8):** re-derived, **10 of 12 hold**. P3, P4 and P9 hold, since the anchor no longer breaks them. **P11 fails** at $1.156\times$ against the registered $1.2\times$. P6 is refuted as before, by $18.70\times$.

**Eleven of twelve held on the core; the anchor moved three of them to failed, so the tier closes at eight of twelve.** Both runs recorded **zero Modern Standby events** (the driver counts them itself — see §14.13).

| | claim | got | |
|---|---|---|---|
| P1 | `L7/R9` admits on the real graph | `True` | **held** |
| P2 | the ratio-1 control is exactly zero, not small | $0.0$ | **held** |
| P3 | $\sigma$ first order: every exponent in $[0.90, 1.10]$ | $1.060$, $0.973$, $\mathbf{1.942}$ | **refuted by the anchor** |
| P4 | the bound with rocket constants holds everywhere graded | `False` | **refuted by the anchor** |
| P5 | $\dot\lambda$ exceeds CS-11's and lies in $[20, 200]$ K/s | $25.96$ | **held** |
| P6 | CS-11's own constants do **not** bound the rocket | $17.62$ | **refuted** |
| P7 | $4\times$ of ratio at fixed interval moves $\sigma$ by $<1.05\times$ | $1.0000000836$ | **held** |
| P8 | $\dot\lambda$ at $0.5$ s and $5$ s differ by $>2\times$ | $2.582$ | **held** |
| P9 | tightness within an order of magnitude of CS-11's | $[0.270, 2.500]$ | **refuted by the anchor** |
| P10 | $s_\Gamma$ constant in the interval to better than $1.5\times$ | $1.0125$ | **held** |
| P11 | the rank-1 trace changes $\sigma$ by $>1.2\times$ | $1.2310$, $1.1661$ | **held** |
| P12 | the two-way referent moves $\sigma$ by $>1.5\times$ somewhere | $0.3935$, $1.0980$ | **held** |

> **Three predictions that held on 25x of interval were refuted by one row 10x further out — and the row was registered, priced and paid for precisely because the core could not reach it.** That is the whole argument for anchors, stated by a case that would otherwise have published a false bound with a clean table behind it.

**P6 is the useful one and §14.7 says why.** It was also flagged as likely to fail *before* the transfer stage ran, from the slope stage's first row — $s_\Gamma$ came in $7.8\times$ below CS-11's, which fixes the direction of the discrepancy and hence the answer. A prediction refuted by a number measured two stages earlier is a prediction that should have been re-registered, and saying so is cheaper than pretending the order was not visible.

### 14.13 A cost this tier could not quote until it measured its own machine

Three readings of the same quantity, same code, same chamber:

| | s of wall per second of gas time | why |
|---|---|---|
| first sizing | $\approx 215\,000$ | **spanned a five-minute Modern Standby**; `perf_counter` counts S0 idle as work |
| on battery, clean | $111\,496$ (3 samples, spread $1.041$) | $1.4$ GHz |
| on AC, settled | $65\,204$–$70\,070$ | measured per sweep row |

> **The same measurement differs by $3.3\times$ depending on whether the laptop was plugged in and whether it slept.** The standby was found by reading System events $506$/$507$, not by noticing a slow run — it looks identical to a slow run. The driver now records `BatteryStatus`, the clock and the standby count beside every cost it prints, and the tier's own record carries `standby_during_run = 0`.

This is why [[tier0-measurements]]'s convention of quoting **sub-steps** rather than seconds keeps earning itself: $1400$, $2800$ and $5500$ sub-steps in §14.6 are the same numbers on any machine, and the seconds beside them are not.

### 14.14 What this tier did not do

- **MECH is still stubbed on every seam**, and the trajectory agent is still unbuilt. W305 is untouched.
- **W316 is measured and not closed.** Closing it moves the probe state every number since Tier 76 was taken at, which is a re-baselining and not a fix.
- **W317 is reported and not fixed**, for W311's reason: adding the edge changes `config/atlas_0_1.yaml`, which is the build repo's file.
- **`sigma` is measured and not yet declared** on the graph. `build_rocket_real` now takes `measured=`, and `L7/R9/lag`'s own message asks for it; the number it should carry is the anchor's, at the native interval, which is what §14.11 is for.
- **`L2/InterfaceMotion` is untouched** and still refuses. It should.

---

---

## 15. Tier 81 — the trajectory agent, and the MECH bond that has no null space to measure

The last part of the original brief. R0's ordering named the trajectory agent — *"lumped, dim M = 1 — the cheapest of all"* — and it was never built; MECH is declared on seven of `rocket.py`'s edges and no channel has ever carried it. **W305 asked for "a MECH response on b–c whose null space is measured against the rigid-body count, coupled to the trajectory agent rather than ahead of it".** Two of those three clauses turned out not to be answerable as written, and saying why is the tier.

### 15.1 The trajectory agent, and why its port is 3-dimensional

`solvers/trajectory.py` is used verbatim, which its own docstring insists on: *"written once and reused **verbatim** by Atlas's non-learned `rigid_body` expert … not a data-generation helper that gets replaced later"*, and, on the integrator, *"RK4 is correct here; do not 'upgrade' it to leapfrog"* — because mass is expelled, drag dissipates and thrust does work, so the system is not conservative. The wrapper adds a port, a `storage`, a `validity` and nothing else.

**Its probed block is analytic, and measured it is exactly that:**

$$B \;=\; \frac{\partial(\text{velocity after one step})}{\partial(\text{rigid load})} \;=\; \operatorname{diag}\!\left(\frac{\Delta t}{m},\ \frac{\Delta t}{m},\ \frac{\Delta t}{I}\right)$$

| | measured | analytic |
|---|---|---|
| translations | $1.0000000000\times10^{-6}$, $1.0000000013\times10^{-6}$ | $\Delta t/m = 1.0\times10^{-6}$ |
| rotation | $5.0000000000\times10^{-6}$ | $\Delta t/I = 5.0\times10^{-6}$ |
| off-diagonal | $\mathbf{0}$, exactly | the three planar modes do not mix |

Maximum relative departure $2.565\times10^{-10}$. **This is the first agent in the rocket graph whose probed block can be checked against a closed form**, and it is the cheapest by a wide margin — a response is microseconds against the chamber's minutes.

> **The brief said $\dim M = 1$ and the port is 3.** A planar rigid body has two translations and one rotation, and the structure's own `_rigid_modes` returns a $[2n, 3]$ basis, orthonormal here to $1.78\times10^{-15}$. One is the count for a single scalar channel; this bond is not one.

Its `validity` declines two ways, both read off the sources rather than chosen: `atmosphere.py` is the 1976 standard tabulated to $86$ km, and `rhs` floors the mass at $10^{-6}$ kg — so a burn past the propellant load returns a number rather than failing, and the floor is a guard rather than a licence.

### 15.2 The coupling variable already exists inside the build repo, and is discarded

`ThermoStruct2D.solve_mechanical` is a **free-body** solve: it removes the three planar rigid-body modes by projection rather than by pinning nodes, and its own comment says why — *"any pin that also blocks a component of uniform thermal expansion manufactures stress in a plate that should have none"*. `_solve_free` does it with a bordered Lagrange system

$$\begin{bmatrix} K & V \\ V^{\mathsf T} & 0\end{bmatrix}\begin{bmatrix} u \\ \ell \end{bmatrix} = \begin{bmatrix} f \\ 0\end{bmatrix}$$

and returns `lu.solve(rhs)[: f.size]`. Because $KV = 0$ and $V$ is orthonormal, left-multiplying the first block row by $V^{\mathsf T}$ gives

$$\boxed{\ \ell = V^{\mathsf T} f\ }$$

exactly — **the net force and torque on the body, which is precisely what `trajectory.rk4_step` integrates.** The build repo computes the trajectory's input as a Lagrange multiplier and slices it off on the very next character.

> **So the structure and the trajectory already share an interface variable, and nothing declares it.** That is this vault's recurring shape — W311's undeclared nozzle wall, W317's `a-c`, W315's rank-1 trace — in its sharpest form yet: not a quantity nobody computed, but one computed and thrown away inside a single expression.

The identity is pinned as **pure algebra on a synthetic system** where $K$, $f$ and $V$ are controlled, rather than on the real shell where observing $\ell$ would mean changing the build repo. The load is recovered here by the surface integral instead — the same construction `generate.py::_compute_loads` uses for `F_aero`.

**And a control that says the map is physical.** Under a *uniform* ambient traction the net rigid load is $(-3.55\times10^{-15},\ -5.68\times10^{-14},\ 0.1386)$: **the two forces cancel to machine precision and the torque does not.** That is the shell's thickness talking — inner and outer faces have equal projected area so a uniform pressure exerts no net force, but they sit at different radii so it exerts a **couple**. The trajectory must receive that couple and the structure must not, which is exactly what the projection arranges. Both halves are asserted, because only the pair rules out a sign error.

The rigid-load map $\partial(V^{\mathsf T}f)/\partial(\text{traction})$ is $3\times14$ with singular values $2.8739\times10^{-3}$, $4.4638\times10^{-4}$, $3.2478\times10^{-5}$ — **rank 3**, so a 14-cell window is enough to drive the whole body, with the torque mode $88.5\times$ weaker than the leading translation.

### 15.3 The MECH response on b–c, and the null space it does not have

MECH's bond is $(\text{traction},\ \text{velocity})$ and `solve_mechanical` is quasi-static: it returns a **displacement**. W305 flagged that the flow needs a time derivative *"whose convention is what R2 decides"* — and R2 decided, by declaring `FluxMatching.TIME_INTEGRATED` over the macro-step. So the flow is $v = \delta u/\Delta t$ at that clock. It is a declaration, it is reported, and §14's lesson applies directly: **a quantity divided by $\Delta t$ is not cadence-free, so a $\beta$ measured here carries its clock.**

The operator is $14\times14$ and costs $0.08$ s a response — **four orders of magnitude cheaper than a gas response**, which is why this tier is minutes where Tier 80 was hours. Its singular values:

$$5.242\times10^{-5},\ 1.123\times10^{-7},\ 4.574\times10^{-9},\ 7.195\times10^{-10},\ \ldots,\ 5.942\times10^{-14}$$

$$\beta = 5.941852\times10^{-14}, \qquad \kappa = \mathbf{8.822\times10^{8}}$$

Read the null dimension at a tolerance and you get whatever you asked for:

| relative tolerance | rank | **null dim** |
|---|---|---|
| $10^{-2}$ | 1 | $\mathbf{13}$ |
| $10^{-3}$, $10^{-4}$ | 2 | $12$ |
| $10^{-6}$ | 6 | $8$ |
| $10^{-8}$ | 11 | $\mathbf{3}$ |
| $10^{-10}$, $10^{-12}$ | 14 | $\mathbf{0}$ |

> **W305 asked for the null space "measured against the rigid-body count", and at $10^{-8}$ it is exactly 3. That number is an artefact of the threshold and not a measurement.** The consecutive singular-value ratios are $466$, $24.6$, $6.36$, $3.31$, $2.92$, $2.95$, … — after the second mode they settle at about three per mode and **never gap**. A genuine null space announces itself as a jump of many orders; here nothing separates signal from null, so any null dimension in $[0, 13]$ is available for the asking.

This is [[tier0-measurements]] W106's mistake with the sign reversed. W106 was about a *reproducibility floor* of exactly zero making a movement test meaningless; this is about a *spectrum with no floor* making a rank test meaningless. **The tier's test therefore asserts that two tolerances DISAGREE** — the only way to state "there is no answer here" as something that can fail. If the operator ever grew a real null space, four decades of tolerance would agree on it.

**And the contrast with THERM on the same fourteen cells is the most striking number in the tier.** Tier 77 measured b–c THERM at $\kappa = 3.7141$ ($2.392$ with the fixed wall, §18.8, so the contrast stands). MECH is $8.82\times10^{8}$ — **eight orders apart, on the same seam, the same cells, the same operating point.** The reason is structural and it is already on the record: the THERM response is nearly *algebraic* (W301 — `_wall_flux` subtracts $T_{\text{wall}}$ directly), while the MECH response is a quasi-static elliptic solve, which is a **smoothing** operator and therefore compact, with geometrically decaying singular values. *Port type, not physics regime, is what decides whether a seam operator has a rank you can name.*

### 15.4 The gas cannot respond at all, and a declaration cannot see why

`Compressible2D` has eight boundary kinds — `wall_slip`, `wall_noslip`, `symmetry`, `inlet_massflow`, `freestream`, `outflow`, `prescribed`, `extrapolate` — and **none of them takes a wall velocity**. `_reflect(no_slip=True)` sets the ghost momentum to $-U$, which is a stationary wall, and `wall_noslip` reads only `T_wall` and `no_slip_mask`. Checked against the source rather than asserted, including the negative: no `u_wall`, `v_wall`, `wall_velocity` or `wall_speed` appears anywhere in the file.

> **So the gas can SUPPLY a traction — it has the wall pressure — and cannot RESPOND to one. The MECH bond at every gas–solid seam in this graph is one-sided by a missing capability, not by a modelling choice, and the declaration is identical either way.** W97 found a field-to-lumped MECH seam one-sided by $\mathrm{Re}_h$; this one is one-sided by a boundary condition that does not exist.

That is why MECH is still not wired into the seven-agent graph. Assembling it would produce a seam whose gas block is identically zero — the `L5/R6` empty-operator refusal that Tier 79 earned on g–f ADVEC, arrived at for a different reason and carrying a different remedy: g–f needed the *declaration* corrected, and this needs the *solver* extended.

### 15.5 The c–trajectory seam, which is two-sided

Composing the structure's hand-over with the body's response gives the seam in the rigid-mode space, $3\times14$:

$$\beta = 3.972877\times10^{-11}, \qquad \kappa = \mathbf{295.51}$$

against b–c MECH's $8.82\times10^{8}$ — **three million times better conditioned.** And it is genuinely two-sided: **the body responds to a load with a velocity natively.** It is the one port in this graph that needs no time derivative supplied and declares none, which is the exact contrast with §15.3's $\delta u/\Delta t$ and the reason the pair is worth building together.

### 15.6 What was deliberately not done

**The trajectory agent is built and measured and is NOT in the compiled graph.** *(Re-derived at §18.8: 4 refusals, 20, 82 — and still Tier 79's count, which Tier 79's re-derived record also moved to 20.)* The seven-agent compile is unchanged at `refuse`, **4 refusals, 22 decertifications, 81 admits** — exactly where Tier 79 left it, asserted rather than assumed, because *a builder that constructs declarations is a place where a refusal can be lost without anybody deciding to lose it* (Tier 79's own lesson). Adding an eighth agent would move the count, and it should move it for a reason that has been argued rather than as a side effect of a tier about something else.

Three things have to land before that argument can be made, and none of them belongs here:

- **MECH needs a responding gas side**, which needs a moving-wall boundary condition in the build repo (§15.4);
- **the `c`–trajectory edge is not in `config/atlas_0_1.yaml`'s `edge_list`** — a fourth undeclared interface, and like W311 and W317 it is the build repo's file;
- **the rigid load must be handed over by the structure rather than recomputed** beside it, which means `solve_mechanical` returning its own multiplier (§15.2).

**Five of five registered predictions held**, over a $61$ s run — the whole tier costs less than one gas response at the declared clock.

---

---

## 16. Tier 83 — W312 priced, and one of the five seams removed from the bill

W312 left every ADVEC number in §13.3 as *"the operator the probe could see in 8 sub-steps"*, with one seam paid for and the rest carrying a label. Converging the rest was quoted at *"~7.2 hours for a full $\dim M = 41$ seam"* — a figure extrapolated from **one** measured call on agent `e`, at $268$ s, on a machine that Tier 80 later showed was running at $1.4$ GHz on battery.

### 16.1 The bill, computed without paying it

The cost of a probe call is the number of CFL sub-steps it takes, and `Compressible2D.max_stable_dt` returns that **without marching anything** — one evaluation per agent, $0.01$ s for all six.

| agent | block | cells | $\Delta t_{\text{model}}$ [s] | $\Delta t_{\text{CFL}}$ [s] | sub-steps | s / call |
|---|---|---|---|---|---|---|
| `a` | $48\times80$ | $3840$ | $10^{-3}$ | $8.6171\times10^{-7}$ | $1160$ | $13.9$ |
| `b` | $112\times80$ | $8960$ | $10^{-3}$ | $3.0210\times10^{-7}$ | $\mathbf{3310}$ | $92.6$ |
| `e` | $120\times96$ | $11520$ | $10^{-3}$ | $2.8004\times10^{-7}$ | $3571$ | $128.4$ |
| `d` | $184\times48$ | $8832$ | $5\times10^{-3}$ | $1.7207\times10^{-6}$ | $2906$ | $80.1$ |
| `f` | $152\times80$ | $12160$ | $5\times10^{-3}$ | $5.1703\times10^{-6}$ | $967$ | $36.7$ |
| `g` | $152\times96$ | $14592$ | $5\times10^{-3}$ | $1.0491\times10^{-5}$ | $477$ | $21.7$ |

The per-call seconds are **calibrated**, not directly measured: Tier 80 measured the chamber at $92\,590$ s of wall per second of gas time on AC, which fixes the cost of one sub-step at agent `b`'s size, and the rest scale by cell count. **It cross-checks twice.** `b`'s $3310$ sub-steps reproduces `ChamberGasAgent`'s own docstring figure of *"about 3350"*, derived independently at Tier 76. And `e` comes out at $128.4$ s against Tier 79's **measured** $268$ s — a factor of $2.09$, which is the battery-versus-AC ratio Tier 80 established at $2.6$, on a call whose sub-step count is fixed. *Two independent routes to the same number, one of them the figure being replaced.*

Per seam, at each seam's own $\dim M$ rather than at `e-f`'s $41$:

| seam | $\dim M$ | calls | hours |
|---|---|---|---|
| `a-b` | $41$ | $84$ | $1.24$ |
| `d-g` | $25$ | $52$ | $0.74$ |
| `e-b` | $41$ | $84$ | $2.58$ |
| `e-f` | $41$ | $84$ | $1.93$ |
| `g-f` | $77$ | $156$ | $1.27$ |
| **total** | | $\mathbf{460}$ | $\mathbf{7.75}$ |

> **So the whole job is $7.75$ hours, not the $7.2$ hours *per seam* W312 implied.** The row's figure was right about `e-f` and wrong as a unit: $\dim M$ is not $41$ everywhere — it is $25$ on `d-g` and $77$ on `g-f` — and four of the six agents are cheaper per call than `e`, which is the most expensive one in the graph and the one the estimate was taken on. **An estimate extrapolated from the worst case, on the wrong power state, was high by a factor and low by a factor at the same time.**

### 16.2 And one seam comes off the bill for nothing

`g-f` is $1.27$ of those hours and **does not need running**. Tier 79 found its ADVEC operator identically zero and attributed it to the shear layer's normal being perpendicular to the flow. That attribution is now checked directly, and it is exact:

- over the whole hole band $j \in [29, 66]$, the face normal is $n_z = \mathbf{-0.000\times10^{0}}$ and $n_y = \mathbf{1.000000}$ — not small, **exactly** zero and one;
- the plume flows at $u_z = 256.4$ m/s.

So the declared flow $\rho\,\mathbf u\cdot\mathbf n = \rho(u_z n_z + u_y n_y)$ collapses to $\rho\,u_y$: **the transverse mass flux through a band the flow is parallel to**, which Tier 79 measured at $4.2\times10^{-10}$ — the solver's own noise. **A cadence changes how long the solver runs; it does not change a normal.** Converging this seam would converge noise, at $1.27$ hours.

That leaves **$6.48$ hours** for the four seams where a cadence could actually move something, and it is a cost that has been priced rather than guessed.

### 16.3 What is not decided here

The $6.48$ hours are **not spent**. W312's own definition of done — *"one plane seam probed at its declared clock, and the ADVEC numbers either replaced or labelled"* — was met at Tier 79 by the second branch: the numbers are labelled, in §13.2 and §13.3, and the anchor established what the label costs ($\beta$ moving $42.8\times$, agent `e`'s block share going $6.4\%$ to $36\%$). Replacing them is a separate decision with a price now attached to it, and the price is small enough that it is worth making deliberately rather than by drift.

---

---

## 17. Tiers 84–85 — the knee measured, every ADVEC seam converged, and the vehicle marched

### 17.1 The knee: the bound's LINEAR constant is what fails

> **Re-derived at §18.8 (W334): there is no knee.** With the fixed wall, $s_\Gamma$ at $10^{-2}$ s is $1.0077\times$ its small-interval value, the order is $1.00$ at every interval, and the bound holds throughout at $2.28\times$–$2.30\times$. What follows is what the old wall produced, and the registered disjunction answered it correctly for that wall.

§14.11 left the residence-time mechanism as an [AI Inference], named a measurement that would settle it, and priced it. It was run. At $\Delta t_{\text{ex}} = 10^{-2}$ s — **1.67 chamber flow-throughs**, straddling the threshold:

| | value |
|---|---|
| lag | $2.588180\times10^{-1}$ K |
| $\sigma$ | $3.275484\times10^{-4}$ |
| $\sigma/$lag | $1.265555\times10^{-3}$ |
| bound/measured | $\mathbf{0.9660}$ — already violated |
| $s_\Gamma$ **here** | $2.088233\times10^{-2}$ K$^{-1}$ — $\mathbf{17.0762\times}$ the small-interval value |

$s_\Gamma$ was flat to $1.0125\times$ over $5\times10^{-5}$ to $10^{-3}$ (§14.3). It is **seventeen times larger** at $10^{-2}$. That settles the disjunction the stage registered before it ran:

> *if $s_\Gamma$ has climbed, the seam's own response to a displacement grows with the march duration, and the bound's **first** constant stops being a constant somewhere in this decade — the bound does not merely need a bigger $C_2$, its linear term is wrong.*

**It climbed.** So the failure at the declared interval is not a missing quadratic term — no $C_2$, of either sign, repairs a linear coefficient that is itself a function of the interval. And the order across the whole table now reads

$$1.060,\quad 0.973,\quad \mathbf{2.295},\quad 1.790$$

placing the break **between 0.83 and 1.67 flow-throughs** — at one residence time, where the inference said it would be. The violation also starts *earlier* than the anchor showed: bound/measured is already $0.9660$ at $10^{-2}$, not just $0.2702$ at $5\times10^{-2}$.

### 17.2 Every ADVEC seam converged, at matched $\dim M$

> **Re-derived at §18.8 (W334).** e–b moves $82.6\times$, not $1090\times$, because its cheap probe is no longer near-singular. e–f moves $53.8\times$, and Tier 79's $\dim M = 8$ anchor now reads $48.3\times$, so the correction below reverses. d–g's sign change was round-off at $\beta \approx 9\times10^{-17}$. Its converged share ($29\%$) and g–f's exact $0.0$ stand.

W312's four remaining seams, plus `g-f` run to remove doubt rather than because the price demanded it:

| seam | $\beta$ reduced | $\beta$ declared | ratio | sign same? | share movement |
|---|---|---|---|---|---|
| `a-b` | $8.74547\times10^{-7}$ | $4.08487\times10^{-5}$ | $46.7\times$ | yes | `a` $0.0001 \to 0.0001$ |
| `e-b` | $5.17669\times10^{-8}$ | $5.64308\times10^{-5}$ | $\mathbf{1090\times}$ | yes | `e` $0.996 \to 0.755$ |
| `e-f` | $5.58034\times10^{-7}$ | $1.23716\times10^{-5}$ | $22.2\times$ | yes | `e` $0.231 \to \mathbf{0.765}$ |
| `d-g` | $9.72545\times10^{-17}$ | $9.26906\times10^{-7}$ | $\mathbf{9.53\times10^{9}}$ | **NO** | `d` $0.0001 \to \mathbf{0.288}$ |
| `g-f` | $2.57829\times10^{-12}$ | $8.60692\times10^{-13}$ | $0.334\times$ | yes | `g` $\mathbf{0.0 \to 0.0}$ |

> **`d-g` is the result.** At the reduced cadence it was numerically empty — $\beta$ at machine epsilon, $\kappa = 6.27\times10^{11}$, agent `d` carrying $10^{-4}$ of the seam. Converged, $\beta$ moves by **nine and a half billion**, $\kappa$ collapses to $1055$, the **sign structure changes**, and `d` carries $\mathbf{29\%}$. By W76 a substitution certificate taken at the cheap cadence could not see `d` at all.

**And `g-f` is the control that makes `d-g` mean something.** Both looked empty at the reduced cadence. One woke up by nine orders; the other went *down* ($0.334\times$), stayed at noise, and kept `g`'s share at **exactly 0.0 at both cadences**. So `d-g` was *small* and `g-f` is *structurally empty* — Tier 83's geometric argument ($n_z = -0.000\times10^{0}$ exactly) survives a direct test instead of standing on inference. It cost $950$ s to stop guessing.

**One correction to Tier 79.** Its anchor reported $\beta$ moving $42.8\times$ on `e-f`. At the true $\dim M = 41$ it moves $\mathbf{22.2\times}$ — the anchor was taken at $\dim M = 8$ and overstated the movement about twofold. The direction and the lesson stand; the factor does not.

### 17.3 How it was made affordable, and the speedup figure that is wrong

`probe_block` is a sequential loop, but its calls are **independent**: the traces are `base` and `base + step·direction_k`, all fixed before any response is computed, and each `respond` restarts its agent from its own `_U0`. So the framework was left as the authority and only the arithmetic moved — record the request set with a zero-returning recorder, fill it across a process pool, then re-run `probe_block` from the cache.

Two controls make that sound. A **determinism control** records the set twice and refuses to continue unless the two agree. And a **round-trip control**: at one cadence on both arms, the cached route reproduces a direct probe **bit for bit** ($\beta = 9.72545\times10^{-17}$ either way, $\kappa$, $\omega$ and every block share identical). The cache is a no-op on the numbers.

> **The script prints a $9.98\times$ speedup and that figure is wrong.** It is measured against *contended* per-call times — the same calls, run ten-wide, each take $2.35\times$ longer than they would alone. Against the uncontended serial estimate the real gain is $\mathbf{4.24\times}$, and the gap is exactly the per-call inflation the scaling test predicted at 10–12 workers. A speedup quoted against a denominator the parallelism itself inflated is a speedup measured against nothing.

### 17.4 The vehicle, marched

The first coupled run in this case study: `generate.CoupledEpisode` advancing all six gas blocks, then the shell, then the loads, then the rigid body — the trajectory agent Tier 81 built, doing its job inside the build repo's own coupler rather than beside it.

**Two declarations, both reported.** `coarsen = 4` (blocks $12\times20$ to $46\times12$), and $\Delta t_{\text{macro}} = 5\times10^{-3}$ s against the config's own $5\times10^{-2}$. The second is a **10× reduction and it is the right direction**: $5\times10^{-3}$ s is $0.83$ chamber flow-throughs, which §17.1 places **below the knee**, where $\sigma$ is still first order and CS-11's bound still holds. At the config's own macro step ($8.3$ flow-throughs) it does not.

> **W334 (§18.8):** there is no knee, and the bound holds at the config's own $5\times10^{-2}$ s too, $2.28\times$ loose. The $10\times$ reduction is conservative rather than necessary.

Cost: $30$ macro steps, $626\,165$ CFL sub-steps, $1647$ s — $54.9$ s a step against the $49.2$ s priced beforehand.

### 17.5 W322 — the engine decelerates the vehicle

| $t$ [s] | $y$ [m] | $v_y$ [m/s] | $m$ [kg] | $F_{y,\text{thrust}}$ [N] |
|---|---|---|---|---|
| $0.0000$ | $25000.000$ | $1044.3619$ | $50000.00$ | — |
| $0.0050$ | $25005.222$ | $1044.2616$ | $49998.12$ | $-5.165\times10^{5}$ |
| $0.0750$ | $25078.243$ | $1042.0187$ | $49971.73$ | $-1.138\times10^{6}$ |
| $0.1450$ | $25151.104$ | $1039.7451$ | $49945.35$ | $-1.138\times10^{6}$ |

**The vehicle is slowing down under thrust**, at $-20$ to $-35$ m/s². `generate.py::_compute_loads` sets

```python
F_thrust = (-thrust * np.cos(th), -thrust * np.sin(th))   # body -z is 'up'
```

with `thrust` a positive momentum-flux integral over the nozzle exit and `th` the state's $\theta$. `initial_state` sets $\theta_0 = \pi/2$, so $F_{y,\text{thrust}} = -\text{thrust}$: **downward, in a state vector whose $y$ is unambiguously altitude** — `rhs` subtracts `gravity(y)` from the same component.

**This is not an integrator artefact and the audit is what establishes that.** Each step's $\mathrm{d}v_y/\mathrm{d}t$ was reconciled against the loads the integrator was actually handed, and it agrees to $\mathbf{1.00001}$ at every step. The trajectory is integrating its inputs correctly; the inputs have the wrong sign.

> **And the audit caught its own bug first, which is why it is trustworthy.** Its first version paired `loads[i]` with the step out of `rigid[i]` and reported ratios of $2.04$, $1.05$, $1.62$ — noise-looking, and easy to write up as "the loads are inconsistent". They were not: `run()` seeds both lists with the pre-run state and appends **after** `step_macro`, so `loads[k]` is the load that *produced* `rigid[k]`. The tell was that measured$[i]$ equalled predicted$[i+1]$ to five digits — a clean one-step lag, which is the signature of an off-by-one and not of a broken integrator. *A control that reports a real defect and a bookkeeping error identically has not yet been calibrated.*

---

## 18. Tier 86 — the seams rebuilt, and a wall that passed mass

Opened 2026-09-23, after [[rocket-episode-seam-audit]] found that §17.4's episode was pieced together wrongly: geometry that did not meet, and a coupler that joined the solvers by array index. **This tier changes the build repo.** Both checkouts carry the same fixes, uncommitted: `atlas-0.1-windfarm` at `0a407b7` (the pin this case study declares), and `atlas-0.1` at `a00d3af`. Every marched number below carries the diff hash of the tree that produced it.

**A correction to §17.4 first.** "30 macro steps" counted frames. `run()` seeds the lists with the pre-run state, so 30 frames are **29 steps**, covering $0.145$ s.

### 18.1 The audit's defects, and what closed each

| row | defect | fix | gate | reading |
|---|---|---|---|---|
| **W327** | shell and `d` had no node at either nozzle kink; coarsening subsampled nodes and the blank mask; `g`'s hole sat at $[-0.500, 0.625]$ | axial breakpoints at both kinks and every agent's $z$ extent; blocks **rebuilt** at the coarse resolution; the hole on `f`'s node lines | G1, G2, G3 | polylines $2.78\times10^{-17}$ m apart; **0** raster points claimed by no block or by two; hole error $0$ |
| **W323** (+W315) | the shell read its walls by array index — 95% of stations off, by up to $4.08$ m — from one scalar wall temperature | each station averages the gas wall cells at its own $z$; every gas wall reads the skin's own temperature by $z$; over the tank barrel the config **declares** the wall (adiabatic, ambient pressure) | C1, C2, C3 | $0.0002$ m inside (half a cell is $0.005$), $0.0375$ m outside (half of `d`'s cell is $0.075$), barrel exactly $0$ |
| **W324** | the lower panel was loaded inside out | `ShellMesh` takes its gas faces from the geometry (`grid.shell_faces`); the traction sign follows the face | E1 | §18.2 |
| **W325** | aero loads on W309's index normals, no shear: drag $0.000$, side force $1781.8$ N/m | both panels on the body's outward normal, plus wall shear $\mu u_t/\Delta n$ | E2 | §18.2 |
| **W322** | widened: the body→world rotation, not just the thrust's sign | nose at body $-z$; thrust and aero share one proper rotation | E2, E3 | thrust up, drag to the tail and down |
| **W326** | conserved states crossed gas models unconverted; the plume hole got `f`'s transverse mean | `_as_gas` at equal $p$, $T$, $\mathbf u$; the hole gets `f`'s edge rows, lower and upper separately | C4, C5, E5 | conversion error $6.66\times10^{-16}$; wake at $0.293$ of its hottest inflow's stagnation temperature |
| **W328** | the freestream ignored $\alpha$ and the vehicle's speed | the relative wind from the rigid state | unit test | |
| **W330** | records took one wall and one panel, index-oriented | both walls, both panels, outward normals; a–c, e–c and d–f recorded (config `recorded_interfaces`); d–g clipped to its declared span | unit tests | d–g $+$ d–f tile `d`'s outlet exactly |
| **W331** | plane seams remapped by index; d–g carried nothing into `g` | remaps by position; `d`'s outlet feeds `g`'s inlet and `f`'s outer inlet | unit tests | |
| **W329** | the page | SI units on one scale per field; labels from the config | — | |
| **W301** | the isothermal channel saturates at the $20$ K ghost floor | the wall face's conduction taken one-sided from $T_w$: $k(T_i)(T_i - T_w)/\Delta n$, exactly what `_wall_flux` hands the shell | unit tests | exact to $10^{-12}$; 300 K and 600 K walls were bitwise identical before |

W311 and W317, the undeclared nozzle-wall and injector-wall interfaces, are now **exchanged by position and recorded**. Declaring them stays a decision about the graph. All the tests that pinned these defects were rewritten to pin the fixes. Each keeps its original diagnosis as a control that the old behaviour is really gone.

### 18.2 The first fixed march — 11 of 13, and the two it failed

The gates were written before the march (`scripts/w323_seams_fixed.py`). The march went to `out/w321_seams_only`: 29 steps, $636\,902$ sub-steps, $v_y$ $1044.3619 \to 1046.1708$ m/s. **The vehicle climbs.** Thrust is $+1.142\times10^{6}$ N/m upward, and the acceleration audit reconciles every step to $3.46\times10^{-4}$.

**E1 and E2 failed.** The panels' mirror residual was $2.74\times10^{-4}$ against $10^{-6}$. The side force was $1.17\times10^{-5}$ of the drag, against $10^{-6}$. The thresholds stayed where they were registered, and the failures were traced rather than excused. Per block, per step, $\max|p - p_{\text{mirror}}| / \max|p|$:

| run | `e` (nozzle), step 3 | `e`, worst | `d` (air), worst | shell $u_y$, worst |
|---|---|---|---|---|
| pre-fix (`out/w321_prefix`) | $4.6\times10^{-3}$ | $6.0\times10^{-2}$ | $3\times10^{-14}$ | $2.00$ (inside out) |
| seams rebuilt | $2.5\times10^{-3}$ | $4.6\times10^{-2}$ | $6\times10^{-7}$ | $2.7\times10^{-4}$ |

**The asymmetry was already in the pre-fix run, and it is not in the seams.** The pre-fix run's nozzle went from $2\times10^{-15}$ to $4.6\times10^{-3}$ in one 5 ms step. Its walls then had one scalar temperature, and its outlet takes nothing from the plume, so nothing asymmetric entered it. The rebuilt seams made the asymmetry *visible*. The shell now reads the nozzle wall by position, so it bent asymmetrically. The air, reading the shell, followed at $10^{-7}$. And a side force of $10^{-7}$ of one panel's normal load is $10^{-5}$ of the drag, because the drag is about twenty times smaller than one panel's normal integral.

### 18.3 W332 — the isothermal wall passed mass

The engine (`a`, `b`, `e`) was marched **alone**, with the coupler's own wiring but both walls pinned at $288.15$ K, so nothing asymmetric can enter from outside (`scripts/w332_nozzle_symmetry.py`). The nozzle's asymmetry at the end of each 5 ms step, inviscid:

| nozzle flux | nozzle walls | 5 ms | 10 ms | 15 ms | 20 ms |
|---|---|---|---|---|---|
| HLLC | isothermal, **as it was** | $1.9\times10^{-15}$ | $1.9\times10^{-13}$ | $1.4\times10^{-8}$ | $\mathbf{7.1\times10^{-4}}$ |
| HLLE | isothermal, as it was | $7.6\times10^{-16}$ | $1.7\times10^{-15}$ | $2.0\times10^{-15}$ | $1.4\times10^{-12}$ |
| Rusanov | isothermal, as it was | $1.1\times10^{-15}$ | $1.0\times10^{-15}$ | $1.0\times10^{-15}$ | $1.1\times10^{-15}$ |
| HLLC | adiabatic | $1.7\times10^{-15}$ | $1.3\times10^{-15}$ | $1.8\times10^{-15}$ | $1.3\times10^{-15}$ |
| HLLC | slip | $2.0\times10^{-15}$ | $1.6\times10^{-15}$ | $1.4\times10^{-15}$ | $1.9\times10^{-15}$ |
| HLLC | isothermal, **fixed** | $1.7\times10^{-15}$ | $1.3\times10^{-15}$ | $1.8\times10^{-15}$ | $1.3\times10^{-15}$ |

The records are `out/w332/*.json`, written by `scripts/w332_nozzle_symmetry.py` (`--old-ghost` restores the inviscid flux's old ghost). The first exploratory runs are archived beside them in `scratch_logs/`, and the script reproduces them. HLLE's $1.4\times10^{-12}$ is inherited from `b` rather than grown in the nozzle, and it is gone by 25 ms. Inviscid, the fixed isothermal wall gives **exactly** the adiabatic wall's numbers, as it should: with no viscous terms, a wall's temperature has nothing to act through. So the mode needs **both** HLLC and the isothermal wall. It grows about $10^{5}$ per 5 ms, from round-off to saturation in four steps. It lives in the wall-adjacent cells of the diverging section, rows 5–12, where the wall column runs at Mach 0.9 beside a Mach 1.9 core.

**The mechanism.** `_fill_side` builds the isothermal ghost at the cell's pressure with

$$T_g = \max(2T_w - T_i,\ 20\ \text{K}), \qquad \rho_g = \frac{p}{R\,T_g} = \frac{T_i}{T_g}\,\rho_i ,$$

so beside $2500$ K gas and a $288$ K wall the ghost is **125 times denser** than the cell. That pair is then handed to the Riemann solver as the wall face's two states. For a true reflection — equal density and pressure, normal velocity reversed — HLLC's contact speed

$$S_M = \frac{p_R - p_L + \rho_L u_L (S_L - u_L) - \rho_R u_R (S_R - u_R)}{\rho_L (S_L - u_L) - \rho_R (S_R - u_R)}$$

has $u_L = -u_R$, $S_L = -S_R$ and $\rho_L = \rho_R$, so its numerator is $-\rho u (S_L + S_R) = 0$ **exactly**. The face carries pressure and nothing else. With $\rho_L = 125\,\rho_R$ the cancellation is gone. For gas moving into the wall at $50$ m/s, $S_M \approx +49$ m/s: **the wall face passes mass**, at $123\times$ the gas's own normal mass flux $\rho\,|v|\,a$ (`out/w332_wall_mass.json`). As a closed box on the coupler's own engine blocks, gas pushed at both walls, the net mass rate over $\rho\,|v|\,L_{\text{walls}}$ is:

| block | before W332 | after |
|---|---|---|
| `a` | $168$ | $3.6\times10^{-15}$ |
| `b` | $193$ | $7.4\times10^{-16}$ |
| `e` | $119$ | $1.5\times10^{-15}$ |

**The fix** (`Compressible2D.residual`): the inviscid flux sees every wall as a **mirror at the cell's own density**, and only the viscous operator sees the isothermal ghost. A wall's temperature reaches the gas by conduction, which the viscous terms and W301's override carry. The inviscid flux of an impermeable wall is its pressure, and the mirror problem gives exactly that. Blocks without an isothermal wall use a single ghost array as before, so their residual is **unchanged bit for bit**. That keeps every case without isothermal walls out of the blast radius.

**This refutes a sentence written earlier the same day.** W301's docstring said the ghost *"still sets the inviscid flux and the interior gradients, and neither needs it to be exact."* The first half is what W332 measured to be wrong.

**[AI Inference]:** every isothermal-wall result in this vault before Tier 86 carries this leak, including the thermal seam (Tiers 15–16) and the rocket (Tiers 76–85). Where the gas beside the wall has no normal velocity — a settled wall layer — the Riemann problem has $u_L = u_R = 0$ and equal pressures. Then $S_M = 0$ exactly whatever the densities, so a steady wall is barely touched. Transients, like the rocket's start-up and every probe's poke, are where it acted. §18.6 lists what moved in the pinned tests.

### 18.4 W333 — the forebody slot counted as airframe

`d` is one two-panel map over its whole range, $z \in [-5, 0.7]$. Ahead of the nose ($z < -4$) its inner boundary runs on at the airframe's half-height, bordering `forebody_slot`, which the config declares unmodelled. The coupler made that boundary **no-slip, at the nose's temperature**. That is a metre of flat plate before the vehicle began: its boundary layer arrived at the nose already grown, its shear was added to the drag, and its record was filed under the c–d seam. The first fixed march's c–d record was $11.454$ m long, against a skin whose outer face is $9.4542$ m.

**The fix.** The config now declares the slot's wall, `gas_wall: slip_adiabatic`: a streamline of the freestream it lies along. The solver gained a per-station `isothermal_mask` beside the existing `no_slip_mask`, and the slot's stations are slip and adiabatic. Records and loads stop at the nose. Gates C7 and C8: the masks match the airframe's extent exactly, c–d equals the outer face to $2.2\times10^{-16}$, and changing the slot's cells changes the aero load by exactly $0$.

**One pin moved with it.** `rocket_experts.WALL_BC_PARAMS` records what `wall_noslip` reads (W305/W319). It is now `T_wall`, `no_slip_mask` and `isothermal_mask`. None is a velocity, so W319's finding stands.

### 18.5 The re-march — 17 of 17

The four new gates (C6, C7, C8, E6) were written into `scripts/w323_seams_fixed.py` before the re-march, and E1–E5 kept their thresholds. **As a dry run against the seams-only march, E6 read $4.57\times10^{-2}$ and failed**, so the gate can fail, and C6–C8 passed on the fixed code. The re-march went to `out/w321`: build repo diff `b6bbcec4240a5943`, archived beside the run.

| gate | seams-only march | **this march** | threshold |
|---|---|---|---|
| E1 — panels mirror images | $2.74\times10^{-4}$ ✗ | $\mathbf{6.46\times10^{-9}}$ | $10^{-6}$ |
| E2 — $\lvert$side$\rvert$ / drag, worst step | $1.17\times10^{-5}$ ✗ | $\mathbf{6.30\times10^{-14}}$ | $10^{-6}$ |
| E3 — climbs | $+1.809$ m/s | $+1.747$ m/s | $> 0$ |
| E4 — acceleration audit | $3.46\times10^{-4}$ | $2.48\times10^{-4}$ | $0.02$ |
| E5 — wake energy bound | $0.293$ | $0.290$ | $1.02$ |
| E6 — engine mirror, worst | $4.57\times10^{-2}$ ✗ (dry run) | $\mathbf{2.18\times10^{-15}}$ | $10^{-8}$ |
| C6 — closed-box wall mass | $168$ / $193$ / $119$ (old residual) | $3.63\times10^{-15}$ | $10^{-12}$ |
| C7, C8 — the slot | — | pass; c–d $= 9.4542$ m exactly | |

G1–G3 and C1–C5 read as before. **The engine stays symmetric to round-off for the whole 0.145 s.** The sideways thrust is $2\times10^{-10}$ N/m where it was $-175.8$, and the side force is $-3.5\times10^{-11}$ N/m.

| | seams-only march | this march |
|---|---|---|
| $v_y$ | $1044.3619 \to 1046.1708$ m/s | $1044.3619 \to 1046.1087$ m/s |
| thrust, last step | $1.142\times10^{6}$ N/m | $1.113\times10^{6}$ N/m |
| drag, last step | $832.0$ N/m | $838.1$ N/m |
| nozzle exit / injector mass flow, steps 10–29 | $1.092$ | $0.976$ |

Thrust is $2.5\%$ lower. The drag is $0.7\%$ higher, because the airframe's boundary layer now starts at the nose instead of a metre upstream, and a thinner layer carries more shear.

**[AI Inference]:** in the seams-only march, $9\%$ more mass left the nozzle than entered at the injector. That is what walls adding mass would look like, and the nozzle there was oscillating, which is when the leak acts. The records are cell-centred traces, and near the throat the area changes by $\sim2.5\%$ within half a cell, so a few percent of either ratio is where the trace is taken. The $11$-point swing between the runs is larger than that. It is not a conservation measurement.

Cost: 29 steps, $647\,637$ CFL sub-steps, $4149$ s of wall time. The laptop moved from battery to mains mid-run and shared the machine with test suites for part of it, so the seconds describe this run and nothing more.

### 18.6 What moved in the vault's pinned tests, and one finding re-measured

The vault's 488 tests that load the build repo were re-run on the final code. Every pin that moved was rewritten to pin the fix, and each keeps its original diagnosis as a control that the old behaviour is really gone:

| test | was | now |
|---|---|---|
| Tier 76, 79: W301's saturation signature | the wall bitwise identical at 300 K and 449 K; `exact_zeros == n - 1` | the wall transmits at its own probe base; `exact_zeros < n - 1` (the chamber's poke reached 11 of 112 cells with W301 alone, where it reached 1) |
| Tier 80: W315, W317 | one scalar wall temperature; a–c unrecorded | per-cell by position; a–c recorded; the conjugate wall's shares recomputed (the declared seam is 40.8%) |
| Tier 15 | the gas support was one cell | global, with the poke's neighbour below $10^{-2}$ of its own cell |
| Tier 16: the thermal seam's consistent interface temperature | $372.1377$ K | $372.4964$ K with W301, then $\mathbf{372.2359}$ K with W332 |
| Tier 81: what `wall_noslip` reads | `T_wall`, `no_slip_mask` | plus `isothermal_mask` (W333) |
| Tier 80: W316 at $10^{-4}$ s | near-wall mean below 2800 K, spread over 100 K | $2802$ K, spread about 4 K; the old residual restores the old reading, kept as the control |

One more was already red before any of this: Tier 80's jump test went red the day Tier 84 added its knee row to `out/w314.json`. It compared the last two rows, and the new row sits between them. It also assumed one exponent for the last decade, which the knee row splits into $2.295$ and $1.790$. It now reads the jump against the flat part of the sweep and checks the two exponents separately.

**W316, re-measured.** The Tier-80 `setup` stage was re-run with both fixes (`out/w332_w316_setup.json`). The chamber's settled near-wall cell is $2696$–$2804$ K, mean $2756$ K, where Tier 80 recorded $1490$–$2804$ K, mean $2464$ K. The ratio to the 2800 K the shell is linearised about goes from $0.8801$ to $0.9841$. **The finding stands at about an eighth of its size.** The shell is still linearised about a hotter gas than its wall layer holds, by $1.6\%$ rather than $12\%$. The rest was the saturated channel and the leak.

### 18.7 What this tier did not do

- **It did not re-derive Tiers 76–85's gas-seam numbers (W334) — *paid later the same day, §18.8*.** As first recorded: the vault's probe wrapper gives every rocket gas agent an isothermal wall, so every THERM and ADVEC number in `out/w300` … `out/w314` was taken with W301's saturated channel and W332's leak. Their tests pass because they read those records. §18.6's W316 shows how large the difference can be. Re-deriving them is **about 7 hours**, from the records' own wall times. (It was first priced at 9, which counted the knee row twice.)

  **Where the time goes.** The gas solver is explicit, so its step is held to the CFL limit, $\Delta t \le C\,\Delta x/(\lvert u\rvert + c)$. The chamber's smallest cells are $0.88$ mm and sound runs at $1100$ m/s in 2800 K gas, so $\Delta t \approx 3.0\times10^{-7}$ s. A simulated millisecond is therefore about 3300 steps of 8960 cells, and costs 93–112 s of wall.
  - The multirate study marches the chamber twice per exchange interval: once with the trace following the skin, once with it held stale. The declared 50 ms interval alone is 100 ms of gas, $364\,093$ steps and $2.57$ h. The knee row adds $1.65$ h, and the shorter intervals about 50 min.
  - The converged flow-through seams are finite-difference Jacobians: up to 42 marches per seam side, one per boundary mode plus the base. They took $1.84$ h on a 10-worker pool.
  - The wall-seam probes cost seconds.

  The two columns of each interval start from the same state and never read each other, so they can run side by side. On mains, that plausibly halves the total. It is still multi-hour work, so it waits for the owner. Until it runs, those pages quote the old wall. *(The owner approved it. With the columns forked and the anchor run beside the core on a rented 48-core CPU, it took 38 minutes.)*
- **It did not declare a–c, e–c, d–f or c–f.** They are exchanged and recorded, and whether the model's graph should declare them is a modelling decision.
- **It did not commit anything.** Both checkouts, and the vault, carry the changes uncommitted.
- **The forebody slot is still a slot.** W333 made its wall honest. It did not model the flow around the nose, which is what removing the slot would take: a `d` block that closes across the axis ahead of the vehicle.

### 18.8 W334 — Tiers 76–85 re-derived, and what the old wall had been saying

**Done the same day.** Every record from `out/w300` to `out/w314` was re-run with W301 and W332 fixed, on a rented 48-core EPYC 9655: Linux, Python 3.11, and the laptop's numpy, scipy and pyyaml versions on OpenBLAS. The build repo went as a copy of the tree this page pins, identified in every record as `0a407b7+dirty:b6bbcec4240a5943`. The old records moved to `out/pre_w332/`, and the new ones took their names (`scripts/w334_swap_records.py`). `scripts/w334_compare_records.py` diffs each pair leaf by leaf and lists every label that changed.

**The box reproduces the laptop.** Its `setup` stage matches the laptop's `out/w332_w316_setup.json` to $10^{-15}$ relative, on a different BLAS, in 158 s where the laptop took 862. The multirate study's two columns ran as forked processes, and its anchor ran as a separate job whose result was replayed into the record. Both routes were checked bitwise against the sequential driver before the box was rented. The whole job took 38 minutes of rental and about \$0.40. The anchor alone, $363\,725$ sub-steps, took 1417 s where Tier 80's took 9259.

**Attribution first, because not every difference is the wall's.** Three other things changed between the original runs and these:

- **Tier 77 declared `effort_normal = "b"` on b–c:THERM.** So `w300` and `w304` now assemble the *difference* where Tier 76 assembled the sum. The ledger's `known` level (`w302`) gains one `L4/E7/passivity` from its stubs, $47 \to 48$. Neither change is the wall's. The `real` level is unchanged: `refuse`, 10 refusals, 43 decertifications.
- **Tier 82 fixed W308**, so `w306`'s two `car_graph` strip seams read `positive`. That leaves the rocket's b–c as the vault's only one-signed seam (defect $0.5085 \to 0.4939$), beside `thermal_strain`'s mixed one.
- **Tier 86's own geometry fixes** (W323, W327). Where they could matter, they are separated below by an old-wall control: today's code with W301 and W332 reverted.

Tiers 81–82 otherwise added MECH and trajectory code, which no gas record reads.

| tier | quantity | old wall | **fixed wall** | reading |
|---|---|---|---|---|
| 76 | chamber $\bar h$ (min–max), W/(m² K) | $183.87$ ($156.84$–$315.11$) | $211.40$ ($165.09$–$421.79$) | the wall now conducts |
| 76 | seam wall at 5 s, mean (max) | $449.42$ ($690.02$) K | $478.96$ ($870.77$) K | |
| 76 | the shell's `validity` declines at | $10.35$ s | $\mathbf{2.95}$ **s** | **before the probe base — W335** |
| 76 | a one-cell poke on the gas moves | 1 of 112 cells | 39 of 112 | W301's signature gone |
| 76 | gate 4; the affine control | $8.29\times10^{-9}$; $0.99982$ | $5.94\times10^{-9}$; $0.99980$ | stands |
| 76 | $\beta$ over $100\times$ of cadence; over 0.5–10 s of burn | $0.29\%$; $26.5\%$ | $0.03\%$; $11.2\%$ | the horizon still dominates (now in the difference assembly) |
| 77 | $\lambda^{\ast}$ | $1035.487$ K | $\mathbf{1133.885}$ K | still one sign change, in $(1015, 1270)$ |
| 77 | corrected $\beta$ ($\kappa$) | $0.123188$ ($3.7141$) | $\mathbf{0.229896}$ ($2.392$) | |
| 77 | shell/gas block ratio | $1.05$ | $1.24$ | W307 stands |
| 77 | `exact_zeros == n - 1` trips | 15 of 110, 2 saturated | 13 of 110, **none** saturated | refused more firmly |
| 79 | e–b at the cheap cadence, $\beta$ ($\kappa$) | $5.18\times10^{-8}$ ($16.9$) | $6.79\times10^{-7}$ ($1.28$) | |
| 79 | nozzle wall's $\bar h$ over the chamber's | $1.474$ | $1.332$ | W311 smaller again |
| 79 | compile | `refuse`, 4, 22 | `refuse`, 4, **20** | `L4/block-share` $3\to2$, `L4/E7/passivity` $2\to1$ |
| 80 | $s_\Gamma$; $C_2$ | $1.2229\times10^{-3}$; $-1.639\times10^{-6}$ | $1.1627\times10^{-3}$; $-1.228\times10^{-6}$ | $C_2$ still negative |
| 80 | $\dot\lambda$ at 5 s; its span over the burn | $25.96$ K/s; $2.58\times$ | $32.20$ K/s; $2.94\times$ | |
| 80 | **bound/measured at the declared $5\times10^{-2}$ s** | $\mathbf{0.2702}$ — **violated** | $\mathbf{2.2782}$ — **holds** | **§14.11's finding was the old wall's** |
| 80 | CS-11's constants bound the rocket by | $17.6\times$–$19.4\times$ | $18.70\times$–$18.79\times$ | P6 refuted as before |
| 80 | the rank-1 trace moves $\sigma$ by | $1.231\times$, $1.166\times$ | $1.156\times$, $1.155\times$ | **P11 now fails** |
| 80 | the two-way referent moves $\sigma$ by | $0.393\times$, $1.098\times$ | $0.101\times$, $0.160\times$ | see below |
| 80 | predictions | 8 of 12 | **10 of 12** | |
| 81 | c–trajectory $\kappa$; b–c MECH | $295.51$; $8.82\times10^{8}$ | $283.66$; unchanged | W318 stands |
| 84 | $s_\Gamma$ at $10^{-2}$ s, over its small-interval value | $17.08\times$ | $\mathbf{1.0077\times}$ | **there is no knee** |
| 84 | $\beta$'s movement, cheap to converged: e–b; e–f; d–g | $1090\times$; $22.2\times$; $9.53\times10^{9}$ | $82.6\times$; $53.8\times$; $1.27\times10^{10}$ | |

**1. The bound holds at the declared interval (§14.5, §14.11, §17.1).** Tier 80's central finding was that CS-11's bound, clean over $25\times$ of interval, is violated at the one interval the graph declares. It read $0.2702$, a $3.7\times$ under-prediction, with the order going to $1.94$. Tier 84 then found a knee at one chamber flow-through, where $s_\Gamma$ climbed $17\times$. **All of it was the old wall's.** Re-derived:

| $\Delta t_{\text{ex}}$ [s] | lag [K] | $\sigma$ | $\sigma/$lag | bound/measured | order |
|---|---|---|---|---|---|
| $2\times10^{-4}$ | $6.34113\times10^{-3}$ | $3.20547\times10^{-6}$ | $5.0550\times10^{-4}$ | $2.3000$ | — |
| $1\times10^{-3}$ | $3.17023\times10^{-2}$ | $1.60489\times10^{-5}$ | $5.0624\times10^{-4}$ | $2.2966$ | $1.0008$ |
| $5\times10^{-3}$ | $1.58484\times10^{-1}$ | $8.05141\times10^{-5}$ | $5.0803\times10^{-4}$ | $2.2882$ | $1.0021$ |
| $10^{-2}$, *the knee row* | $3.16893\times10^{-1}$ | $1.61243\times10^{-4}$ | $5.0882\times10^{-4}$ | $2.2843$ | $1.0019$ |
| $\mathbf{5\times10^{-2}}$ **(declared)** | $1.581489$ | $8.05768\times10^{-4}$ | $5.0950\times10^{-4}$ | $\mathbf{2.2782}$ | $0.9997$ |

$\sigma$ per unit lag is flat to $0.8\%$ over $250\times$ of interval, and every exponent is $1$ to $0.3\%$. **There is no knee**: $s_\Gamma$ at $10^{-2}$ s is $1.0077\times$ its small-interval value. The bound is loose by $2.28\times$–$2.30\times$ at every interval, the declared one included, inside the band CS-11 found on the brake. So the rocket is a second seam on which CS-11's form, written in the interval, holds a decade past where it was checked. §14.11's residence-time inference, which §17.1 took as confirmed, explained an artefact. Which of the two wall defects produced the knee is not separated here; the old-wall control for this stage would cost the anchor again.

**The anchor's registered predictions, read honestly.** A1 and A2 were registered from the old core's lag rate, $25.89$ K/s, and both fail. The fixed wall heats the shell faster ($31.70$ K/s), so the lag is $1.5815$ K where $1.2943 \pm 5\%$ was registered. $\sigma$ misses the top of A2's range by $0.7\%$, because it rides on that lag. A3 holds. The rule the three encoded is that the lag is linear in the interval and $\sigma$ is a fixed multiple of it. Applied to the new core's own graded rows, that rule lands within $0.23\%$ on the lag and $0.57\%$ on $\sigma$. This is a consistency check, not a registration, because the core and the anchor ran side by side.

**The predictions go from 8 of 12 to 10 of 12.** P3, P4 and P9, which the anchor had refuted, now hold. **P11 now fails**: the rank-1 trace moves $\sigma$ by $1.156\times$ and $1.155\times$, under the registered $1.2\times$. P6 is refuted as before.

**2. The two errors that nearly cancelled were the old wall's (§10.2).** At the cadence §10.2 used, $10^{-5}$ s (`scripts/w334_tier77_table.py`):

| assembly | base | old wall, as published | **fixed wall** | old wall on today's code |
|---|---|---|---|---|
| SUM *(Tier 76)* | each side's own | $0.149619$ | $0.146475$ | $0.149262$ |
| SUM | $\lambda^{\ast}$ | $0.0013150$ | $0.0047232$ | $0.0012479$ |
| DIFFERENCE | each side's own | $0.200562$ | $0.197386$ | $0.200182$ |
| **DIFFERENCE** *(corrected)* | $\lambda^{\ast}$ | $0.123188$ | $\mathbf{0.229896}$ | $0.124841$ |

(each at its own wall's $\lambda^{\ast}$). The doubly-wrong configuration is now $36\%$ *below* the corrected value rather than $21\%$ above it. The nearest configuration to the corrected one is now the right assembly at the wrong base, as a reader would expect. **The last column is the control**: W301 and W332 reverted on today's code reproduce the published table to $0.25\%$ on three rows, and to $5\%$ on the near-singular sum, whose value is steep in its base. So the near-cancellation belongs to the wall, and the small residue to Tier 86's other changes. The sum at $\lambda^{\ast}$ is still the one mixed spectrum, so §10.3's reading of E7 stands. `test_two_errors_that_nearly_cancelled` now asserts the fixed wall's ordering and keeps the old wall's as its control.

**3. W301's signature is gone, and its detector is refused more firmly (§4, §11).** Both gas agents now transmit at their probe bases. In both saturation scans every wall temperature except the reference's own gives a different field, where the old scans had runs of bitwise-identical ones. A one-cell poke on the chamber moves 39 of 112 cells where it moved 1. Swept again, `exact_zeros == n - 1` trips on **13 of 110** real-callable ports, and **none** of them is saturated. The two ports that were saturated are the two that stopped tripping. As a saturation detector its precision is now $0/13$.

**4. The solidus arrives at 2.95 s, before the declared probe base (§2.2) — W335.** With the wall conducting, the chamber's film coefficient is $\bar h = 211.40$ where it was $183.87$. The shell's own `validity` predicate now declines at **$2.95$ s** of burn, where it declined at $10.35$. At the declared 5 s base the seam wall is $478.96$ K in the mean and $\mathbf{870.77}$ K at its hottest, $100$ K past the Al–Li solidus. §7.2 quoted one row from a state the predicate declines, deliberately. **Now every b–c number at the 5 s base is quoted from one.** At 2.5 s the wall is still valid (hottest $672.5$ K). Moving the base is a decision about what the probe is for, so W335 records it and does not make it.

**5. The flow-through seams (§13.2a, §17.2).** Converged at matched $\dim M$:

| seam | movement, old wall | **fixed wall** | converged $\beta$ | share, cheap $\to$ converged |
|---|---|---|---|---|
| a–b | $46.7\times$ | $46.7\times$ | $4.085\times10^{-5}$ | `a` $0.0002$ |
| e–b | $1090\times$ | $82.6\times$ | $5.607\times10^{-5}$ | `e` $0.998 \to 0.803$ |
| e–f | $22.2\times$ | $53.8\times$ | $3.374\times10^{-5}$ | `e` $0.136 \to 0.250$ |
| d–g | $9.53\times10^{9}$ | $1.27\times10^{10}$ | $1.164\times10^{-6}$ | `d` $0.0001 \to 0.285$ |
| g–f | $0.334\times$ | $1.49\times$ | noise | `g` exactly $0.0$ at both |

e–b's movement shrank because its cheap-cadence probe was near-singular on the old wall ($\kappa = 16.9$) and is not now ($1.28$). Its converged value barely moved. **Tier 84's "correction to Tier 79" reverses.** The $\dim M = 8$ anchor on e–f now reads $48.3\times$ against $53.8\times$ converged at $\dim M = 41$, and both put `e`'s share at $0.249$–$0.250$. On the old wall they were $0.362$ and $0.765$. **d–g's "sign-structure change" was round-off.** At the cheap cadence its $\beta$ is $9\times10^{-17}$, where the sign of the smallest eigenvalue carries no information, and it read `mixed` on one run and `positive` on the other. Its converged numbers carry the finding, and they stand: `d` goes from $10^{-4}$ of the seam to $29\%$. g–f stays structurally empty.

**What stood.** Gate 4 and its affine control. W66's three legs. The ledger at the `real` level. The CFL-ratio control: $1.0000000003\times$, with the three values pairwise different, so it is not a floor. $s_\Gamma$ is constant in the interval, to $1.0021\times$. CS-11's constants bound the rocket, and §14.7's coin-flip reading of that. W315's collapse costs far less than its rank suggests. W318's contrast holds: THERM at $\kappa = 2.39$ against MECH's $8.82\times10^{8}$, still eight orders. W307 holds, and so does the ordering of horizon over cadence.

**W316's two-way control, re-read.** The two-way referent now moves $\sigma$ by $0.101\times$ and $0.160\times$, the same side of 1 at both intervals, where the old wall's $0.393\times$ and $1.098\times$ changed direction. §14.9 read the old pair as a startup step: the shell had been marched under 2800 K, and the referent hands it the gas's own near-wall temperature. The step is now 44 K rather than 336 K. It still cuts the two-way lag to $35\%$ of the declared one at $2\times10^{-4}$ s. **[AI Inference]:** the step lands on the shell's face node, about half a millimetre of the 8 mm, eight-layer skin, whose heat capacity is $\rho c\,\delta/2 \approx 1.2\times10^{3}$ J/(m² K). A $3\%$ cut in the face's heat flux, $\approx 1.5\times10^{4}$ W/m², then changes its drift by about $12$ K/s, the same order as the $20$ K/s measured. On the old wall the same arithmetic gives a flux cut near $20\%$ and a drift that reverses, which is the old pair's larger, direction-changing movement. So §14.9's reading stands at the new size. It is not measured beyond this.

**One evaluator was more permissive than its prose.** Tier 81's R5 says the seven-agent graph is left "where Tier 79 left it", and its code checks only the verdict and the four refusals. The re-derived records moved the uncertified admits $22 \to 20$, and R5 read `held` throughout. The prose's claim is now checked against Tier 79's own record by `test_tier81`, and it held on both walls ($22 = 22$, $20 = 20$). The registered evaluator is left as registered.

**What moved in the tests.** Tier 80's gate tests now pin the re-derived record. The old wall's readings (the anchor's violation, the jump, the second-order exponents) are kept as controls that read `out/pre_w332/w314.json`. The standby test accepts the box's `-1`, meaning no event log to count, only beside the same record's evidence that PowerShell was absent. Tier 77's near-cancellation test was rewritten before the run, against the re-measured root, with the old wall as its control.

---

## 19. Tier 87 — W336: the episode's residuals, and what "correct" can mean here

Opened 2026-09-24, on the owner's question: *can we evaluate the residuals, and how do we know this is correct?* The episode of §18 passes 17 of 17 gates. Those gates test symmetry, geometry, the coupler's bookkeeping and the trajectory's integration. None of them measures whether the coupled run **conserves** what it exchanges, or how far its numbers are from the ones a finer run would give.

**The script** is `scripts/w336_episode_residuals.py`, with `scripts/w336_thermal_seam_diag.py` beside it. Its records are in `out/w336/`, and its tests are `tests/test_tier87_episode_residuals.py`. Every arm ran on a rented 48-thread Threadripper 7960X, from the shipped script (SHA-256 `10612cb0…eae33d`). The build repo went as `0a407b7+dirty:b6bbcec4240a5943`, the tree the episode was marched on.

### 19.1 Three questions, and the one the build repo already answers

"Is it correct" is three questions, and they have different answers:

1. **Does the code solve its equations?** This is code verification. The build repo already has it, per solver:
   - an isentropic vortex converging at order $\ge 1.7$;
   - the Sod shock tube's wave positions;
   - free-stream preservation on the curvilinear nozzle;
   - a Stokes shear layer, and the Blasius growth law;
   - the erf conduction slab;
   - exact thermal stress;
   - Tsiolkovsky for the trajectory;
   - the nozzle's centreline Mach against area–Mach, within $5\%$ on a $120 \times 24$ grid.
2. **Does this run satisfy its discrete equations, and do its seams conserve?** These are the residuals, §19.2–§19.6.
3. **How far is it from the answer?** This is discretization error, §19.7, and validation against physics known independently, §19.8.

Nothing before this tier asked the second and third of the coupled system.

### 19.2 The instrument, and why its numbers can be believed

A finite-volume update changes a block's content only through its boundary faces, so per block and per macro step

$$\sum_{\text{cells}} V\,\bigl(U^{\text{after}} - U^{\text{before}}\bigr) \;=\; \sum_{\text{sub-steps}} \tfrac12\,\Delta t \sum_{\text{stages}} \;\sum_{\text{boundary faces}} \text{inflow} \;+\; \text{reaction}.$$

The **ledger** wraps the solver's own flux functions and reads the arrays its residual differences. It does not recompute them. It weights them as SSP-RK2 does: $U^{n+1} = U^n + \tfrac12\Delta t\,[r(U^n) + r(U^{(1)})]$, so both stages at $\tfrac12\Delta t$. The accumulation is face by face, so every boundary can be read separately: walls, the injector, each side of each plane seam.

Four controls, all pinned by the tests:

| control | reading |
|---|---|
| a uniform stream through a channel, by hand | $\rho u H\,\Delta t$ in and out to $10^{-12}$; walls exactly zero |
| the same march with and without the ledger | **bitwise** equal |
| a budget with the inflow side left out | the residual is exactly that side's inflow |
| the episode's engine with W332 reverted, one step | the walls pass $\mathbf{5.2\%}$ of $\dot m\,\Delta t$, where the fixed wall passes $5\times10^{-17}$ |

**The re-run is the recorded run.** Altitude, attitude, vertical velocity and mass match `out/w321` **bitwise at all 29 steps**, on a different machine and BLAS.

**One evaluator was fixed while the audit ran.** Gate A4 was first evaluated component by component. The lateral position and velocity are $10^{-20}$-level round-off in a symmetric flight, so dividing each by its own recorded value read $O(1)$ "deviations". A4 now measures position, velocity, attitude and mass each against its own magnitude. The change and its reason are in the script, and both readings are in the record.

### 19.3 What closes exactly

| account | residual |
|---|---|
| every block's content against its boundary inflows (mass, both momenta, energy), every step | $1.2\times10^{-13}$ |
| mass through the engine's and the airframe's walls, face by face | $5\times10^{-17}$ of $\dot m\,\Delta t$ |
| the shell's energy change against its Robin heat | $1.1\times10^{-11}$ of its stored energy |
| the engine's mass account, and the two routes to the thrust | $10^{-15}$ (identities of the bookkeeping) |
| the loads' cell-centred exit thrust against the solver's face flux | $3.7\times10^{-8}$ |

The solver conserves. The walls are impermeable, which is W332's fix holding inside the coupled run. The shell's backward-Euler step conserves to its own CG tolerance. Everything below is therefore about the **coupling and the model**, not about a leak in a solver.

### 19.4 W337 — the injector delivers 17.7% more than it declares

Over the last ten steps the injector face passes $\mathbf{1.1772}$ times the declared $\dot m^{\ast} = 376.87$ kg/(s m): the ideal choked flow for $p_c = 8$ MPa and $T_c = 3400$ K. The engine run alone gives $1.1774$ at coarsen 4 and $1.1783$ at coarsen 2, so the excess does not depend on the grid.

**[AI Inference], from the boundary condition's code.** `inlet_massflow` does not prescribe a flux. It builds a ghost state at $T_{\text{inject}} = 900$ K with $\rho u = \dot m/A$ and the interior's pressure, and the face flux is then the Riemann solution between that cold, dense ghost and the burning gas in `a`'s first cells. The build repo's own nozzle test uses the same condition with no reaction, injecting at the chamber temperature, where ghost and cell agree. So the excess would appear only with combustion. Not isolated here.

**What it costs the episode.**
- The chamber runs at a volume-mean $9.04$ MPa, $1.13\times$ the declared $8$.
- The nozzle receives $1.147\,\dot m^{\ast}$, after the throat seam's loss (§19.5).
- The vehicle loses mass at the *declared* rate while the nozzle expels $1.147\times$ that, so the mass-loss/outflow ratio is $0.872$.

The episode's operating point is not the sweep point it is labelled with.

### 19.5 The plane seams: two kinds of error, told apart by the coupling step

A seam between two blocks exchanges frozen states once per macro step. Each side computes its own face flux, so their difference is mass, momentum or energy the coupling creates. Running the coupled episode at $\Delta t_{\text{macro}} = 10$, $5$ and $2.5$ ms separates a *lag*, which shrinks with the step, from a *floor*, which does not. Mean $\lvert$mass created$\rvert$ per step over 30–60 ms, relative to the flow through the seam:

| seam | how states cross | 10 ms | 5 ms | 2.5 ms | reading |
|---|---|---|---|---|---|
| a\|b | identical transverse grids | $0.0196$ | $0.0130$ | $0.0012$ | lag |
| e\|f | overlap average (integral-preserving) | $1.2\times10^{-3}$ | $3.2\times10^{-4}$ | $9.9\times10^{-5}$ | lag, order $1.97$ |
| **b\|e**, the throat | **point interpolation**, 20 cells against 24 | $0.073$ | $\mathbf{0.029}$ | $\mathbf{0.030}$ | **floor — W340** |
| d\|f | air handed to the exhaust gas's model | $0.335$ | $0.368$ | $0.371$ | **floor — W341** |
| g\|f | the plume hole, both gas models | $0.277$ | $0.221$ | $0.232$ | **floor — W341** |

**W340.** `_remap` interpolates cell states by position (`np.interp`), which preserves no integral. At the throat, where `b` has 20 transverse cells and `e` 24, about **$3.0\%$ of the mass flow vanishes every step**. The loss roughly halves when the grid does ($-3.1\%$ at coarsen 4, $-1.8\%$ at coarsen 2, at 30 ms), as an interpolation error should. The only seam built with `_cell_average`, e|f, shows no floor at all.

**W341.** `f` runs the exhaust gas's $(\gamma, R)$ for everything in it, air included. `_as_gas` (W326) hands states across at equal $p$, $T$, $\mathbf u$, so the density changes by $R_{\text{air}}/R_{\text{gas}} = 287/361 = 0.795$ in one direction and the inverse in the other. A mass flux cannot survive that, and the d|f and g|f floors of $20$–$37\%$ are mostly that ratio. This is a limit of a single-species plume block, not a coding slip: $p$ and normal velocity are what a contact carries, and $T$ and $\rho$ are not.

**The throat seam also destroys momentum**, $1.7\%$ of the thrust. That is exactly the gap between the two routes to the thrust: $1.1321\times10^{6}$ N/m from the forces on the injector and the walls, $1.1135\times10^{6}$ from the exit plane's momentum.

### 19.6 The thermal seams

**W339 — the stationary wall does work.** Per side and per 5 ms step, the gas loses $626.2$ J/m at the engine's walls. The shell gains $597.3$. The diagnostic splits the gas's side: the one-sided conduction W301 installs is $596.6$, and the shell matches it to $\mathbf{0.13\%}$, so the thermal coupling conserves heat. The remaining $29.6$ J/m, $\mathbf{4.7\%}$, is the viscous energy flux's *work* term. The solver builds the face value of $\mathbf u\cdot\boldsymbol\tau$ as the average of the cell's and the mirrored ghost's. That average is not zero, although a wall that does not move can do no work. Energy disappears at the wall face: $4\times10^{-6}$ of the engine's energy throughput, but $4.7\%$ of the heat that reaches the airframe. It is the same at every coupling step ($0.0474$, $0.0474$, $0.0473$), because it is a property of the face, not of the lag.

**W338 — the skin radiates to the air beside it.** `step_thermal` folds the radiative coefficient into the outer Robin term, whose target is the adjacent gas's temperature, not the ambient. Averaged over the late steps, that costs the skin $0.95$ J/m per step per panel: $0.16\%$ of its engine-side heat input. A real bug with a negligible effect here.

**W342, first half — the airframe's heating has the wrong sign at the episode's resolution.** At Mach 3.5 the boundary layer's recovery temperature is several hundred kelvin above the $288$ K skin, so the air should heat it. At coarsen 4 the conduction-limited flux runs the other way: the air *gains* $1.9$ J/m per step per panel from the skin. Coarsen 2 and 1 reverse it, to $+8.5$ and $+70$ J/m per step per panel. **[AI Inference]:** at coarsen 4 the cell beside the wall is too thick to hold the boundary layer's heated gas, so it reads closer to the free stream's static $222$ K than to the recovery temperature. Not measured directly. The magnitudes are small beside the engine's $600$.

### 19.7 Convergence

**The coupling step** (coarsen 4; at or around 60 ms):

| quantity | 10 ms | 5 ms (the episode) | 2.5 ms | order | Richardson |
|---|---|---|---|---|---|
| $v_y$ gained by 60 ms [m/s] | $0.5915$ | $0.6797$ | $0.7164$ | $1.27$ | $0.7424$ |
| shell inner face, hottest [K] | $293.374$ | $293.456$ | $293.499$ | $0.93$ | $293.547$ |
| drag [N/m] | $849.00$ | $848.60$ | $848.68$ | flat | |
| exit thrust, 40–60 ms mean [N/m] | $1.1256\times10^{6}$ | $1.1214\times10^{6}$ | $1.1138\times10^{6}$ | not monotone | |

The velocity the vehicle gains by 60 ms is **$8\%$ below** its extrapolated value at the episode's 5 ms step, first order in the step. Most of that is the start-up, when the thrust sampled at 5 ms swings hardest.

**The grid** (each subsystem alone, 30 ms, walls at 288.15 K):

| quantity | coarsen 4 | coarsen 2 | coarsen 1 | reading |
|---|---|---|---|---|
| injector $\dot m / \dot m^{\ast}$ | $1.1774$ | $1.1783$ | lost | a property of the condition, not the grid |
| throat seam b\|e | $-3.1\%$ | $-1.8\%$ | lost | an interpolation error |
| chamber $p$ (volume mean) [MPa] | $9.18$ | $9.30$ | lost | |
| exit Mach (mass mean) | $2.359$ | $2.389$ | lost | |
| exit thrust [N/m] | $1.132\times10^{6}$ | $1.148\times10^{6}$ | lost | |
| engine wall heat, per side per step [J/m] | $623$ | $1249$ | lost | **doubles per halving** |
| airframe drag [N/m] | $855.4$ | $700.4$ | $642.0$ | pressure $832.7 \to 650.6 \to 525.4$ (order $0.54$); shear $22.7 \to 49.7 \to 116.7$ (diverging) |
| airframe heat, per panel per step [J/m] | $-1.9$ | $+8.5$ | $+69.8$ | sign reversed at coarsen 4 |

**The full-resolution engine's record was lost.** Its run finished on the rented box, but the session that was to collect it ended first, and the box idled until the account's credit ran out and it could not be restarted. Its first two steps had been printed to the box's log and read off during the session, so the three grids can be compared at $t = 10$ ms, still in the start-up transient. This comparison was not registered:

| at 10 ms | coarsen 4 | coarsen 2 | coarsen 1 | order |
|---|---|---|---|---|
| throat $\dot m / \dot m^{\ast}$ (start-up) | $0.5955$ | $0.5900$ | $0.5868$ | $0.78$ |
| chamber $p$ (volume mean) [MPa] | $9.491$ | $9.534$ | $9.594$ | differences grow |
| exit Mach (mass mean) | $2.342$ | $2.373$ | $2.390$ | $0.85$ |
| exit thrust, step mean [N/m] | $5.514\times10^{5}$ | $5.488\times10^{5}$ | $5.470\times10^{5}$ | $0.52$ |
| engine wall heat, per side [J/m] | $581$ | $1151$ | $2256$ | doubles, $\times1.98$ and $\times1.96$ |

**W342, second half — every wall flux is a property of the grid.** The model's heat transfer is `generate.py`'s conduction-limited $h = k/(\Delta n/2)$: laminar, with no wall model. The skin friction is $\mu u_t/\Delta n$. Halving the wall cell doubles both, and neither has converged at coarsen 1. The engine's *integral* quantities (ṁ, chamber pressure, exit Mach, thrust) move about $1.3\%$ per refinement and are the part of this episode a reader can quote.

**This reaches W335.** The solidus at $2.95$ s (§18.8) was computed with the same conduction-limited $h$ on the full-resolution chamber. **[AI Inference]:** it is therefore a property of that mesh as much as of the vehicle. A real chamber's turbulent boundary layer would carry far more heat than this laminar model, so the true time is likely shorter, not longer. Not measured.

### 19.8 Validation: the nozzle against ideal theory

At the flow the nozzle actually receives, $1.147\,\dot m^{\ast}$, ideal theory at area ratio $3.0$ gives the thrust below:

| | ideal | episode (late mean) | ratio |
|---|---|---|---|
| thrust, quasi-1D [N/m] | $1.1374\times10^{6}$ | $1.1135\times10^{6}$ | $0.979$ |
| thrust, with the planar divergence $\sin\alpha/\alpha = 0.9887$ ($\alpha = 14.93°$) | $1.1261\times10^{6}$ | | $\mathbf{0.989}$ |
| exit Mach (mass mean) | $2.420$ | $2.359$ | $0.975$ |
| exit velocity [m/s] | $2309$ | $2238$ | $0.969$ |
| exit pressure, scaled to the flow [Pa] | $5.82\times10^{5}$ | $6.10\times10^{5}$ | $1.049$ |

**Given its inflow, the nozzle is right to about $1\%$ in thrust.** The rest of the gap is plausibly the viscous wall layer and the coarse grid, and §19.7 has it moving about $1.4\%$ per refinement. The thrust's excess over the *declared* operating point is W337's, not the nozzle's.

### 19.9 The predictions

Registered in the script before any arm ran:

| | prediction | got | |
|---|---|---|---|
| A1 | every block closes to $10^{-10}$ | $1.2\times10^{-13}$ | held |
| A2 | walls pass under $10^{-12}$ of $\dot m\,\Delta t$ | $5\times10^{-17}$ | held |
| A3 | the shell closes to $10^{-9}$ of its stored energy | $1.1\times10^{-11}$ | held (re-based after a smoke test, before the audit) |
| A4 | the re-run reproduces the record to $10^{-9}$ | bitwise in the flight components | held (evaluator fixed during the run, §19.2) |
| P1 | the injector delivers $\dot m^{\ast}$ to $2\%$ | $1.1772$ | **failed — W337** |
| P2 | a\|b and b\|e create under $1\%$ | $1.5\times10^{-4}$; $\mathbf{3.0\%}$ | **failed — W340** |
| P3 | the two thrust routes agree to $2\%$ | $1.7\%$ | held |
| P4 | cell-centred and face thrust agree to $2\%$ | $3.7\times10^{-8}$ | held |
| P5 | engine heat lost and gained agree to $2\%$ | $4.7\%$ | **failed — W339** |
| P6 | vehicle mass loss and nozzle outflow agree to $2\%$ | $12.8\%$ | **failed — W337, W340** |
| P7 | the radiation target moves the outer heat under $5\%$ | $81\%$ | **failed**, and ill-posed: the outer heat itself is nearly zero at coarsen 4 |
| P8 | seam creation at least linear in the step | $2.19$; $1.33$ | **failed — the throat's floor** |
| P9 | $v_y$ at 60 ms moves under $0.01\%$ from 5 to 2.5 ms | $0.0035\%$ | held, **and uninformative**: $v_y$ is dominated by the $1044$ m/s it started with; the velocity *gained* moves $5\%$ |
| P10 | throat $\dot m$ within $3\%$ of ideal at coarsen 1 | — | **not measured**: the 30 ms record was lost (§19.7) |
| P11 | coarsen-4 thrust within $5\%$ of coarsen 1 | — | **not measured** as registered; $0.8\%$ at 10 ms, unregistered |
| P12 | engine wall heat at coarsen 1 over twice coarsen 4's | — | **not measured** as registered; $3.88\times$ at 10 ms, unregistered |
| P13 | coarsen-4 drag within $10\%$ of coarsen 1 | $33\%$ | **failed — W342** |

### 19.10 So — is the episode correct?

**To round-off:**
- conservation inside every block;
- impermeable walls;
- the shell's energy;
- the trajectory's integration of the loads it is handed (E4);
- the run's reproducibility, across machines.

**To about a percent:** the nozzle's thrust given its inflow, and the engine's integral quantities against the grid.

**Wrong, with a named cause:**
- the operating point (W337, $+17.7\%$ mass flow);
- the throat seam (W340, $3\%$ of the mass flow per step);
- work at stationary walls (W339, $4.7\%$ of the wall heat);
- the plume's seams (W341, $20$–$37\%$);
- the skin's radiation (W338, negligible).

**Not known at this resolution:**
- every wall flux (W342): engine heat, skin friction, and the airframe's heating, sign included;
- the velocity gained, which carries an $8\%$ coupling-step error by 60 ms.

So the episode is a correct demonstration of the coupled machinery, with its bookkeeping exact and its failures located. It is **not** a quantitative prediction of this vehicle. W337, W338, W339 and W340 are code: each has a clear fix and a test that pins it. W341 and W342 are modelling decisions — a two-gas plume, a wall model or a finer episode — with costs attached in §19.7.

---

## 20. Tier 88 — the four fixes, the re-march, and a throat floor that was not what it looked like

Opened 2026-09-26 from [[rocket-episode-pickup]], by a session with none of the context the episode was built in. **This tier changes the build repo**, on both branches: commit **`c1ccac6`** on `atlas-0.1-windfarm`, which this page pins, and **`edc3315`** on `atlas-0.1`. `git diff 76edd8b c1ccac6 | sha256sum` begins `409bd71838fd6dc0`, so every record naming that identity is a record of `c1ccac6`. The records name the tree they ran on: `76edd8b+dirty:409bd71838fd6dc0` for the march, the audit and the coupling-step study, and `76edd8b+dirty:547399c4359a14b2` for the first build's records that carry over (§20.2 says why they can). The Tier 87 records are kept in `out/w336_pre_w337` and `out/w321_pre_w337`.

### 20.1 The four fixes

| row | defect | fix | build-repo test and its control | reading, before → after |
|---|---|---|---|---|
| **W337** | the injector's face flux was the Riemann solution between a 900 K ghost and the burning gas | `_inlet_face_flux` prescribes the face's inviscid flux: mass $\dot m''$; momentum $(\dot m'' u + p)\,\mathbf n$ with $u = \dot m''/\rho$ and $\rho = p/(R\,T_{\text{inject}})$ at the cell's pressure; energy $\dot m''\,(c_p T_{\text{inject}} + u^2/2)$; $Y = 0$ | a burning strip carries $\dot m''$ to $10^{-14}$; the old face carries $1.46\,\dot m''$ | $1.1772 \to 1.000000$ (worst audited step $7\times10^{-14}$, every grid to coarsen 1) |
| **W338** | radiation folded into the outer Robin term against the adjacent gas | `_outer_robin`: $h = h_{\text{out}} + h_{\text{rad}}$ against $T_{\text{target}} = T_g + \dfrac{h_{\text{rad}}}{h_{\text{out}} + h_{\text{rad}}}\,(T_\infty - T_g)$, which is the old assembly bit for bit wherever $T_g = T_\infty$ | $h_{\text{out}} = 0$ against 2000 K air: the skin cools to $T_\infty$ by exactly the linearised radiation; the old target heats it | slip $0.94$ J/m per step per panel $\to 3\times10^{-16}$ |
| **W339** | the no-slip face's viscous energy flux kept the average of $\mathbf u\cdot\boldsymbol\tau$ between the cell and its ghost | `_isothermal_wall_conduction` sets that flux to the conduction alone: one-sided from $T_w$ at isothermal stations, zero at adiabatic ones (the forebody slot, any wall without `T_wall`); the shear is untouched | the face carries $k(T_i - T_w)/\Delta n$ to $10^{-13}$ and an adiabatic face exactly $0$; the old face carries $0.32\%$ of work on a sheared layer | wall work $29.65 \to 0$ exactly; the shell receives the conduction to $1.3\times10^{-3}$ at the third step and $2.4\times10^{-4}$ late in the run (it was $4.7\%$ short) |
| **W340** | the throat lost $3.0\%$ of the mass flow every step | every plane seam hands over an overlap average on the faces' node edges, **two ghost layers deep on a two-way seam** (a\|b, b\|e, f\|g) and **one on a one-way seam** (e→f, d→f, d→g) — §20.2; d–g made two-way, two layers each way, in §20.8 | a channel cut in two computes the uncut channel's flux through the cut **bit for bit**; one repeated layer creates $2.8\%$ | throat $-3.0\% \to +2.8\times10^{-4}$ late, and a lag in the coupling step |

Each fix keeps its diagnosis as a control. The vault's `w336_episode_residuals.revert` puts any of them back on today's code, and with all four reverted the solvers' residuals, the shell's step and the coupler's wiring are **bitwise** those of build repo `adc470b`. That was checked against a checkout of `adc470b` with random states in every block.

### 20.2 W340 — the floor was the ghost's depth, not the interpolation

§19.5 read the throat's floor as point interpolation between `b`'s 20 transverse cells and `e`'s 24. Three observations supported that: the floor did not move with the coupling step, it halved with the grid, and e\|f, the one seam built with an overlap average, had none. The briefed fix was that overlap average. **Before any march, a test with no lag in it refuted the diagnosis.** It takes the pre-fix run's engine at one instant, rebuilds it in SI from the record, and wires the throat three ways. It then evaluates each block's own face fluxes at that same instant with the ledger (`scripts/w340_throat_diag.py`, `out/w340_throat_diag.json`). Mass created at b\|e, as a fraction of $\dot m$:

| instant | point interpolation (`adc470b`) | overlap average, one layer (the briefed fix) | overlap average, two layers |
|---|---|---|---|
| 50 ms | $-3.042\times10^{-2}$ | $-3.086\times10^{-2}$ | $-4.2\times10^{-5}$ |
| 100 ms | $-3.060\times10^{-2}$ | $-3.109\times10^{-2}$ | $+4.2\times10^{-5}$ |
| 145 ms | $-3.051\times10^{-2}$ | $-3.100\times10^{-2}$ | $+1.2\times10^{-5}$ |

**The mechanism.** A `prescribed` ghost repeated in both layers makes the receiver's reconstructed face state equal to the neighbour's **cell-centre** state, half a cell from the face. The neighbour's own flux uses its state reconstructed **at** the face. In a steep axial gradient those differ. At the throat the gas accelerates through Mach 1, and §19.5 measured the floor there, not at the chamber's a\|b. Handed the neighbour's first two cell layers, each overlap-averaged, the receiver reconstructs the face from the same four-cell stencil a single block would use. With matching transverse grids, both halves of a cut channel then compute the uncut channel's flux through the cut bit for bit. The build repo's test `test_W340_two_layers_make_the_seam_one_face` pins this, and its one-layer control creates $2.8\%$. The prescribed boundary now takes a layered state, $[N_G, n, n_v]$ with layer 0 against the face, the convention `hole_state` already used.

**All three clues fit this mechanism as well as the one they were read as supporting.** The error is spatial, so it does not move with the step. It is half a cell, so it halves with the grid. And at e\|f the source's own ghost is the problem's opposite, as the next paragraph shows. ***The same observations can fit two mechanisms; only a test that removes one of them tells them apart.***

**And a regression, caught part-way through the fixed run.** The first fixed build handed *every* plane seam two layers. Its audit's coupling-step study, read at 60 ms, showed e\|f gone from a lag ($1.2\times10^{-3}$, $3.2\times10^{-4}$, $9.9\times10^{-5}$ at 10, 5 and 2.5 ms) to a $1.8\%$ floor ($2.1\times10^{-2}$, $1.8\times10^{-2}$, $1.8\times10^{-2}$). `e`'s exit is an outflow boundary, whose ghost is its own edge cell repeated. So `e`'s face state *is* its cell centre, and a receiver handed two layers reconstructs a slope the source never used. Frozen at one instant, e\|f creates $2.0\%$ with two layers and $1.5\times10^{-16}$ with one: mass flux is linear in the conserved state, so a supersonic one-way seam handed the source's own face state is exact. **The rule is to hand over the layers that reproduce the source's own face state.** That means two where each side's ghost is the other's cells, and one where the source sits on an outflow boundary. Every coupled stage was re-run on the second build. The first build's records are kept in `out/w336_t88_twolayer`, `out/w321_t88_twolayer_partial` and `out/w321_t88_twolayer_box`. Its uncoupled records carry over because the two builds differ only in the one-way seams, which neither the engine alone, nor `d` alone, nor the engine's thermal diagnostic can see. The coarsen-4 engine run, the coarsen-4 external run and the thermal diagnostic are **bitwise equal** on both builds.

Both changes came after Tier 88's predictions were registered (§20.3); the script records them beside the predictions, with the times.

### 20.3 The re-march and the re-audit

**The march** ran on the rented box and is the record: `out/w321`, 29 steps, $652\,106$ sub-steps, 2879 s. The laptop's replication (`out/w321_laptop`, 7748 s, Windows and MKL against Linux and OpenBLAS) matches it **bitwise in altitude, attitude, $\omega$ and mass**. $v_y$ differs by $2.3\times10^{-13}$ m/s, one unit in the last place of 1045, and the gas fields by $10^{-13}$. The vehicle climbs $1044.3619 \to 1045.7753$ m/s, where the pre-fix run reached $1046.1087$. The last step's thrust is $9.739\times10^{5}$ N/m, where it was $1.113\times10^{6}$, and the drag is unchanged at $838.2$ N/m. **All 17 of Tier 86's gates pass on it** (`out/w323_seams_fixed_t88.json`): E1 $3.87\times10^{-9}$, E2 $1.15\times10^{-13}$, E4 $2.48\times10^{-4}$, E5 $0.286$, E6 $2.20\times10^{-15}$.

**The late-time account** (mean of the last 10 audited steps):

| | pre-fix (Tier 87) | **fixed** |
|---|---|---|
| injector / throat / exit, over $\dot m^{\ast}$ | $1.1772$ / $1.1469$ / $1.1469$ | $\mathbf{1.0000}$ / $1.0009$ / $1.0009$ |
| b\|e mass, momentum, energy created | $-3.0\%$, $-1.7\%$, $-3.0\%$ | $+2.8\times10^{-4}$, $+2.0\times10^{-4}$, $+2.6\times10^{-4}$ |
| thrust, exit-plane and wall routes [N/m] | $1.1135$ and $1.1321\times10^{6}$ | $9.7458$ and $9.7434\times10^{5}$: agree to $2.5\times10^{-4}$ |
| engine heat, gas lost and shell gained [J/m per side per step] | $624.9$, $595.3$ | $600.35$, $600.49$ |
| radiation slip [J/m per step per panel] | $0.94$ | $3\times10^{-16}$ |
| chamber pressure over the declared 8 MPa | $1.130$ | $\mathbf{0.980}$ |
| vehicle mass loss over nozzle outflow | $0.872$ | $\mathbf{0.9991}$ |
| exit Mach, velocity [m/s], pressure [Pa] | $2.359$, $2238$, $6.10\times10^{5}$ | $2.362$, $2246$, $5.33\times10^{5}$ |
| thrust over ideal planar theory, given the inflow | $0.989$ | $\mathbf{0.992}$ |
| d\|f, g\|f (W341); d\|g (W343) | $-37\%$, $+23\%$; $-4.1\%$ | $-38\%$, $+23\%$; $-2.1\%$ |

**The predictions.** Tier 87's, re-read on the fixed run, and Tier 88's, registered at 11:05 before the fixed episode was marched or audited:

| | prediction | got | |
|---|---|---|---|
| A1–A4 | the gates | $1.2\times10^{-13}$; $3.5\times10^{-17}$; $1.1\times10^{-11}$; bitwise at 29 of 29 steps | held |
| P1, P2, P5, P6, P7, P8 | the four defects' readings | $1.0000$; $5.5$ and $3.5\times10^{-4}$; $2.4\times10^{-4}$; $9\times10^{-4}$; $5\times10^{-16}$; $3.2$ and $9.4$ | **held** (all failed at Tier 87) |
| P3, P4, P9 | | $2.5\times10^{-4}$; $4\times10^{-8}$; $4.5\times10^{-6}$ | held |
| P10 | throat within 3% of ideal at coarsen 1 | $1.077$ at 30 ms | **failed**, §20.4 |
| P11, P12 | thrust within 5% of coarsen 1; wall heat more than doubles | $0.7\%$; $3.98\times$ | held (first measured) |
| P13 | drag within 10% of coarsen 1 | $33\%$ | failed — W342, as before |
| Q1, Q2 | injector exact to $10^{-6}$, every audited step, every grid | $7\times10^{-14}$; $3\times10^{-14}$ at coarsen 1 | held |
| Q3, Q4 | no wall work; thermal seam under $0.5\%$ | $0$; $2.4\times10^{-4}$ | held |
| Q5 | what remains of the thermal seam shrinks at least 1.5× per halving | $1.24\times$, $1.44\times$ | **failed** |
| Q6 | radiation slip under $10^{-9}$ of the radiation | $6\times10^{-16}$ | held |
| Q7, Q8, Q9 | throat a lag, below a\|b's, under $0.3\%$ late | $5.8\times$ and $9.5\times$; $1.4$ against $2.9\times10^{-2}$; $3.5\times10^{-4}$ | held |
| Q10–Q13 | chamber, nozzle, mass, thrust routes | $0.980$; $0.992$; $9\times10^{-4}$; $2.5\times10^{-4}$ | held |
| Q14 | all four reverted reproduce the pre-fix audit, rigid state bitwise | readings to $1.1\times10^{-13}$, flight components bitwise, $x$ and $v_x$ not | **failed**, below |

**Q5 failed honestly.** The engine's thermal seam, $4.7\%$ at every coupling step before, now reads $4.7$, $3.8$ and $2.6\times10^{-4}$ at 10, 5 and 2.5 ms over 30–60 ms. That is a hundredfold smaller and shrinking with the step, but by $1.24\times$ and $1.44\times$, not the $1.5\times$ registered. **[AI Inference]:** the shell's backward-Euler step takes the gas's end-of-step state while the gas conducted against the shell's start-of-step temperature, so the lag should be first order. A residue of this size may carry a second, non-lag part, such as the shell's CG tolerance or the per-station averaging of $h(T_g - T_w)$. Not separated.

**Q14 repeated A4's first mistake.** With all four fixes reverted, today's code reproduces the pre-fix audit's first two steps: altitude, attitude, $v_y$, $\omega$ and mass **bitwise**, and the injector ($1.1780325709$), throat seam, wall heat and radiation slip to $1.1\times10^{-13}$. The lateral $x$ and $v_x$ are round-off around zero ($10^{-20}$, $10^{-17}$), and they differ in their last bits between the pre-fix audit's Zen 4 box and this Zen 3 one. The prose said "rigid state bitwise", so Q14 reads FAILED, and the registered evaluator is left as registered. §19.2 had already named this trap. ***A registered prediction can repeat a mistake the same page names; the evaluator's test with a synthetic flight used lateral round-off equal on both sides, so it could not see it.***

### 20.4 The coupling step and the grid

**The coupling step** (30–60 ms mean $\lvert$mass created$\rvert$ per step, relative to the flow through the seam):

| seam | 10 ms | 5 ms | 2.5 ms | Tier 87 at 10 / 5 / 2.5 ms | reading |
|---|---|---|---|---|---|
| a\|b | $5.6\times10^{-2}$ | $2.9\times10^{-2}$ | $3.1\times10^{-3}$ | $2.0$, $1.3$, $0.12\times10^{-2}$ | lag |
| **b\|e** | $8.2\times10^{-2}$ | $1.4\times10^{-2}$ | $\mathbf{1.5\times10^{-3}}$ | $7.3$, $2.9$, $\mathbf{3.0}\times10^{-2}$ | **lag** (was a floor) |
| e\|f | $1.4\times10^{-3}$ | $5.3\times10^{-4}$ | $1.7\times10^{-4}$ | $1.2$, $0.32$, $0.099\times10^{-3}$ | lag |
| d\|f, g\|f | $0.35$, $0.29$ | $0.38$, $0.22$ | $0.38$, $0.23$ | as before | floor — W341 |
| d\|g | $5.9\times10^{-2}$ | $2.1\times10^{-2}$ | $2.1\times10^{-2}$ | $7.0$, $3.7$, $4.1\times10^{-2}$ | floor — W343, §20.6 |
| engine thermal | $4.7\times10^{-4}$ | $3.8\times10^{-4}$ | $2.6\times10^{-4}$ | $4.7\times10^{-2}$ at all three | residue, §20.3 |

The velocity the vehicle gains by 60 ms is $0.5337$, $0.5798$ and $0.5845$ m/s. Richardson gives $0.585$, so the episode's 5 ms step sits **$0.8\%$** below it, where it sat $8\%$ below at Tier 87. The 40–60 ms thrust ($1.036$, $1.000$, $0.976\times10^{6}$ N/m) still moves with the step: that window is the chamber's start-up, and the late-time thrust is the table's.

**The grid** (each subsystem alone, 30 ms). The full-resolution engine ran to the end this time: 6 steps, $10\,388$ s.

| quantity | coarsen 4 | coarsen 2 | coarsen 1 | reading |
|---|---|---|---|---|
| injector $\dot m/\dot m^{\ast}$ | $1.000000$ | $1.000000$ | $1.000000$ | exact at every grid |
| throat $\dot m/\dot m^{\ast}$ | $1.0742$ | $1.0732$ | $1.0765$ | the start-up at 30 ms, the same at every grid |
| chamber $p$ (volume mean) [MPa] | $8.244$ | $8.308$ | $8.334$ | order $1.31$, Richardson $8.352$ |
| exit Mach (mass mean) | $2.361$ | $2.390$ | $2.410$ | order $0.52$ |
| exit thrust [N/m] | $1.0428\times10^{6}$ | $1.0443\times10^{6}$ | $1.0502\times10^{6}$ | within $0.7\%$ |
| engine wall heat, per side per step [J/m] | $596$ | $1190$ | $2370$ | doubles, $\times2.00$ and $\times1.99$ — W342 |
| airframe drag [N/m] (pressure; shear) | $855.8$ ($833.2$; $22.7$) | $700.9$ ($651.5$; $49.4$) | $641.4$ ($522.3$; $119.1$) | as before — W342 |
| airframe heat, per panel per step [J/m] | $-0.38$ | $+11.2$ | $+58.5$ | sign still reversed at coarsen 4 — W342 |

**P10 failed on a window chosen for the wrong engine.** Its 30 ms was registered when the injector over-delivered and the chamber settled high. With the injector exact, the chamber is still draining its initial fill at 30 ms, and the throat passes $1.07\,\dot m^{\ast}$ at every grid. Late in the audited run it passes $1.0009$ at coarsen 4. That is the throat's start-up, not its grid convergence, which this tier did not measure at a settled state.

### 20.5 Tiers 76–85 re-derived, and what the fixes moved there

W334's procedure, done twice on the same box and tree. Every Tier 76–85 record was re-run with the fixes (`out/t88`) and with all four reverted (`out/t88_revert`, through `scripts/box/with_revert.py`), and the old records moved to `out/pre_w337/` (`scripts/t88_swap_records.py`). **The control reproduces W334's records.** Nothing moves by $10^{-3}$ except round-off-level quantities on a different machine:
- W318's MECH singular values, $10^{-11}$ to $10^{-13}$ at $\kappa \approx 9\times10^{8}$;
- d–g's $\beta$ of $9\times10^{-17}$;
- g–f's $\beta$, which W334 called noise;
- a CG floor, and timings.

So nothing else changed since W334 that these records can see, and every difference below is the fixes'.

**What the fixes moved** (same box, fixed against reverted): **nothing this page quotes, at the precision it quotes it.** The b–c seam's $\beta$ moves at $10^{-9}$. $\lambda^{\ast} = 1133.884859$ K is identical, and so are the corrected $\beta = 0.229896$ and $\kappa = 2.392$. The declared interval's bound/measured goes $2.27819 \to 2.27821$, and $s_\Gamma$ at $10^{-2}$ s stays $1.0077\times$. The compile stays `refuse`, 4, 20, and the two-way referent's ratio goes $0.10114 \to 0.10122$. What did move:
- **Threshold counts:** a one-cell poke on the chamber now moves **36** of 112 cells, where it moved 39; `b`'s `c:THERM` saturation-sweep row reaches 25 cells, not 23; `d`'s `c:THERM` 9, not 8.
- **Round-off-level quantities:** the MECH operator's small singular values, by 2%.

`test_the_fixes_move_the_tier76_85_records_only_where_named` pins that list. **W337 does not reach these records**, contrary to the pickup's plan: the vault's agent `a` probes with a freestream inlet, not `inlet_massflow`. **W338 does**, through the shell's `d:THERM` port, whose outer face is driven against a gas temperature that is not $T_\infty$. W340 reaches only the coupler. The thermal seam of Tiers 15–16, re-run live with W339 fixed and reverted, moves its consistent interface temperature $372.235888 \to 372.235740$ K and its mismatched $\beta$ $0.37597146 \to 0.37597140$. Both pins hold at their stated precision.

### 20.6 W343 — d\|g is coupled one way across a subsonic band

Reading the fixed run's seams turned up a floor Tier 87 had not tabulated: d\|g creates $-2.1\%$ at 5 and at 2.5 ms, and it created $-4.1\%$ before the fixes. Both blocks are air, so it is not W341. Frozen at one instant, the handover is exact: `g` is handed $73.802$ of `d`'s $73.802$. The loss is in `g`'s own inflow faces, which pass $70.686$, $-4.2\%$. The shortfall sits in the four cells beside the plume, where `g`'s first column is **subsonic** (Mach $0.52$–$0.93$), while `d`'s outlet there runs at Mach $3.5$. **[AI Inference]:** the underexpanded plume, at many times the ambient pressure, drives a disturbance into `g` that reaches its inflow plane. `g`'s Riemann problem there is then no longer upwind, and `d`, behind an outflow boundary, never hears it. The seam is declared, but coupled one way. Whether d–g should be two-way (`d`'s outlet taking `g`'s cells) or its non-conservation declared is a graph decision, recorded as **W343**. *Made two-way the same day, which halves the loss and leaves a floor: §20.8.*

### 20.7 Cost, and what this tier did not do

**Compute.** One vast.ai instance, an EPYC 7B13 host with 64 effective cores at $\$0.607$/h, the owner's cap being $\$0.70$/h. It ran 3.8 h for $\$2.34$ of credit ($\$9.63 \to \$7.29$), including about 50 minutes idle after its last job while the session was paused. Three contracts were created on offers above the cap or on a busy host. One came from a create the owner had rejected in the permission prompt, which still reached the API and started. All three were destroyed within minutes, for about $\$0.001$. The box ran with its own six-hour stop deadline. It was destroyed after every file was hash-checked, and the instance list read `[]`.

**Not done, and the owner's:** W341 (a two-gas plume), W342 (a wall model or a finer episode), W335 (the probe base past solidus), W343 (d–g one way; *made two-way in §20.8*), and declaring the a–c, e–c, d–f and c–f interfaces. One vault test that fails independently of this tier, `test_tier45`'s W189 control on `front_wing`, fails identically on the pre-fix build and is left for its own session.

### 20.8 W343 — d\|g made two-way, and the floor it leaves

The owner decided the same day to couple d–g both ways.

**The change.** It is in the build repo on both branches: commit **`98df350`** on `atlas-0.1-windfarm`, which this page pins, and **`462fff8`** on `atlas-0.1`. The records name `c1ccac6+dirty:bb2c392cf522a319`, and `git diff c1ccac6 98df350 | sha256sum` begins `bb2c392cf522a319`, so every record naming that identity is a record of `98df350`. `out/w321/build_repo_at_launch.diff` is that diff.
- `d`'s outlet stays an `outflow` boundary. Where `g` lies across it, its ghost takes `g`'s first two live cell layers, overlap-averaged onto `d`'s outlet cells. The mechanism is `coupled_state`, with `coupled_mask` true where at least half of a `d` cell faces `g`'s live cells.
- `g`'s inlet takes two of `d`'s layers where it took one. That is §20.2's rule: a seam whose ghosts are each other's cells takes two layers.
- `CoupledEpisode.D_G_TWO_WAY = False` puts the one-way wiring back, bit for bit `c1ccac6`'s, and `w336_episode_residuals.revert(M, "W343")` sets it.

Three build-repo tests pin it:
- a masked outflow takes its neighbour's layers there and keeps its own elsewhere;
- where the two grids match, a channel cut at a two-way outflow seam computes the uncut channel's flux;
- `d` takes `g` exactly where `g` lies across its outlet.

The suites pass: 92 tests on `atlas-0.1-windfarm` and 96 on `atlas-0.1`.

**Before the run**, §20.2's frozen-instant test was run on three instants of Tier 88's march. Two-way coupling moved d\|g from $-1.9$ to $-2.1\%$ (one way) to $-1.1$ to $-1.2\%$. The residue sat in the two subsonic `g` cells beside the plume, for two reasons:
- one `d` cell straddles the plume line with $35\%$ of it facing `g`'s live cells, so it stays an ordinary outflow;
- `d`'s $0.2$ m cells span `g`'s $0.13$ m ones.

That reading is recorded beside the predictions in the script, not as a record.

**The predictions**, registered at 17:37 EDT before the two-way episode was marched (`PREDICTIONS_W343`):

| | prediction | got | |
|---|---|---|---|
| S1 | d\|g under $1.5\%$ at 5 ms (30–60 ms mean) | $1.133\%$ | held |
| S2 | a floor: 5 and 2.5 ms both in $[0.3,\ 1.5]\%$ and within $30\%$ of each other | $1.133\%$ and $1.126\%$ | held |
| S3 | velocity gained and last-step drag within $1\%$ of Tier 88's | $1.413382$ against $1.413386$ m/s; $839.51$ against $838.24$ N/m | held |
| S4 | the engine blind to it: chamber $p$ within $10^{-4}$ of Tier 88's, throat within $10^{-4}$ of the declared flow | $p$ to $3\times10^{-15}$; throat $1.00089$ | **failed**, below |
| S5 | A1–A4 on the new audit, and all 17 of Tier 86's gates on the new march | all four held; 17 of 17 | held |
| S6 | with W343 reverted, the first two audited steps reproduce Tier 88's | every compared quantity bitwise | held |

**S4 failed on its own wording.** Its second clause compared the throat with the **declared** flow, and Tier 88's throat already ran at $1.00089$ of that late in the run (§20.3). The engine itself did not move:
- the throat, chamber, exit flow and thrust match Tier 88's at every audited step to $10^{-13}$;
- the late throat matches to ten digits.

The engine sees d–g only through the altitude, at round-off. ***A clause meant as "unchanged" has to compare with the run it claims is unchanged; this one compared with a target the old run had not met either.***

**The coupling step** (30–60 ms mean $\lvert$mass created$\rvert$, relative to the flow through the seam):

| seam | 10 ms | 5 ms | 2.5 ms | Tier 88 (one way) at 10 / 5 / 2.5 ms | reading |
|---|---|---|---|---|---|
| **d\|g** | $5.09\times10^{-2}$ | $\mathbf{1.133\times10^{-2}}$ | $\mathbf{1.126\times10^{-2}}$ | $5.90$, $2.13$, $2.08\times10^{-2}$ | still a floor, about half the size |
| d\|f; g\|f | $0.36$; $0.29$ | $0.39$; $0.22$ | $0.38$; $0.23$ | $0.35$, $0.38$, $0.38$; $0.29$, $0.22$, $0.23$ | W341's floors, barely moved |
| a\|b, b\|e, e\|f, engine thermal | | | | | identical to Tier 88's |

The velocity gained by 60 ms is $0.53375$, $0.57983$ and $0.58449$ m/s, the same as Tier 88's to five digits.

**And the seam now rings.** One way, d\|g's late series was flat at $-2.1\%$, within $3\times10^{-4}$. Two way, it alternates step by step:
- at 5 ms: $-1.0$, $-1.6$, $-0.9$, $-1.5$, … $-1.1$, $-1.3\%$, the swing shrinking from $0.6\%$ at 50 ms to $0.2\%$ at 145 ms;
- at 2.5 ms: wider, $-0.5$ against $-1.9\%$, and shrinking too.

The mean is the floor at both steps. **[AI Inference]:** the coupler wires `d`, `f` and `g` from one another's start-of-step states and then advances each, a Jacobi exchange. A disturbance crosses the seam and comes back in two macro steps. The lagged loop's alternating mode is the ringing, and it decays because the loop's gain is below one. Wiring `g` from `d`'s end-of-step state (Gauss–Seidel) would test this. Not run.

**What the seam reaches.**
- **The engine run alone.** Re-run at coarsen 4 on a different box, it equals Tier 88's in every reading. So the engine's coarsen-2 and coarsen-1 records carry over; `out/w336` holds them as copies.
- **The external flow run alone** (30 ms). It gives the same drag to six figures at coarsen 4, 2 and 1.
- **The diagnostics.** The thermal diagnostic moves only at round-off. The old-wall control's first step shows `d`'s outflow rate moving by $1.2\times10^{-4}$ and the airframe's loads by $5\times10^{-9}$.
- **The coupled run's drag.** It matches Tier 88's to $4\times10^{-8}$ through 20 ms. From 30 ms, when the plume reaches the seam, it reads $0.4\%$ higher, easing to $0.17\%$ by 145 ms. **[AI Inference]:** the plume's disturbance now enters `d` through the subsonic band and reaches the aft body. The path was not traced.

**What is left is W344.** The handover preserves each layer's integral. But on non-matching grids, a subsonic face's flux is a nonlinear function of both sides' states, and the two sides reconstruct different states on different faces. So no choice of ghost states makes their fluxes equal cell by cell across $0.2$ m against $0.13$ m cells. **[AI Inference]:** exact conservation needs one of two things:
- the seam's flux computed once, on the common refinement of the two face grids, and applied to both sides (a mortar-type conservative exchange);
- the two blocks sharing a node line along the seam, as Tier 86 arranged for the other seams.

That is a coupler design decision, recorded as **W344**.

**Compute.** One vast.ai instance with 128 threads at $\$0.211$/h, under the owner's $\$0.70$/h cap.
- Twelve jobs ran in 40 minutes: the march 2384 s, the audit 2371 s, the full-resolution external flow 1465 s.
- Credit went from $\$7.29$ to $\$7.12$.
- All 41 files were hash-checked on download, the instance was destroyed, and the list read `[]`.

**Records.** Tier 88's records were moved to `out/w321_t88` and `out/w336_t88` first, with checksums verified. The new ones are `out/w321`, `out/w336`, `out/w323_seams_fixed_w343.json`, and the box's logs in `out/w343/box_logs`. The page was rebuilt with the two-way seam's notes. `tests/test_tier88_w343.py` pins all of it.

---

**Worklist rows opened by this page:** W300 (the seam itself), **W301** (the saturated channel and its detector), W302 (the declaration ledger), **W303** (`effort_normal` at b–c — *closed at Tier 77, §10*), **W304** (the common seam base — *closed at Tier 77, §10*), W305 (MECH and the trajectory agent), **W306** (the passivity diagnostic cannot tell a global sign from an amplified mode — *implemented at Tier 78, §12*), **W307** (`thermal_seam`'s E7 holds by a film-coefficient accident), **W308** (`car_graph` declares `effort_normal` and has it inverted on two of three seams), **W309** (`grid.Block` face normals are index-oriented, so every gas seam needs `effort_normal`), **W310** (R0 complete — all seven agents on real physics), **W311** (a second undeclared interface, the nozzle wall), **W312** (a plane port's probe needs a cadence a wall's does not), **W313** (`Prolongation.nondim_diag` is specified and no case sets it, so every beta in this vault is in raw units), **W314** (R2's remaining half — the composed defect at 50:1 against CS-11's bound, *§14*), **W315** (`generate.py` collapses the shell to one number, so the build repo's shell-to-gas trace is rank 1, *§14.8*), **W316** (the shell is linearised about a gas temperature the gas does not have, *§14.9*), **W317** (a third undeclared interface — `a-c`, named by the build repo itself, and the declared seam is 40.7% of the conjugate wall, *§14.10*). **W305** (the trajectory agent and MECH on b-c — *§15*), **W318** (the b-c MECH operator has no spectral gap, so `expected_null_dim` is not a well-posed declaration for a compact response — *§15.3*), **W319** (`Compressible2D` has no wall-velocity boundary condition, so every gas-solid MECH bond is one-sided by a missing capability — *§15.4*), **W320** (`_solve_free` computes the trajectory's input as a Lagrange multiplier and discards it — *§15.2*). **W321** (the graph marched, not just probed — *§17.4*), **W322** (the engine decelerates the vehicle — *§17.5*; *closed and widened to the body rotation at Tier 86, §18.1*). **Tier 86:** **W323**–**W331** (the seams rebuilt — *§18.1*), **W332** (the isothermal wall passed mass — *§18.3*), **W333** (the forebody slot counted as airframe — *§18.4*), **W334** (Tiers 76–85's gas-seam records predate W301 and W332 — *priced at §18.7, re-derived at §18.8: the anchor's violation, the knee and the near-cancellation were the old wall's*), **W335** (the shell's validity now declines at 2.95 s, before the declared 5 s probe base — *§18.8*); W301 fixed in the solver, W315 closed, W316 re-measured at an eighth of its size (*§18.6*). **Tier 87:** **W336** (the episode's residuals — *done, §19*), **W337** (the injector delivers 17.7% more than it declares — *§19.4*), **W338** (the skin radiates to the air beside it — *§19.6*), **W339** (a stationary no-slip wall does work — *§19.6*), **W340** (non-conforming plane seams are handed over by point interpolation — *§19.5*), **W341** (the plume block runs one gas model for exhaust and air — *§19.5*), **W342** (every wall flux in the episode is a property of the grid — *§19.6, §19.7*). **Tier 88:** W337, W338, W339 and W340 *closed* (*§20.1*; W340's floor was the ghost's depth, *§20.2*), W334's procedure re-run for the fixes (*§20.5*), and **W343** (d\|g is coupled one way across a subsonic band beside the plume — *§20.6*; *made two-way, §20.8*). **Tier 88, continued:** **W344** (d\|g's two-way floor, $1.1\%$, from non-matching grids across the subsonic band — *§20.8*). See [[gap-worklist]].
