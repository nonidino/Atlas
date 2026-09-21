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

---

## 4. W301 — the isothermal wall is saturated, and the trace never reaches the solver

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

### 13.6 W301's signature, cross-checked — and it is about `wall_noslip`, not about the solver

Tier 76 found the isothermal wall saturated. If that were a property of `compressible2d` rather than of that one boundary condition, the plane ports would show it too. They do not:

| | ports | tripping `exact_zeros == n-1` |
|---|---|---|
| **wall** | 2 | 1 |
| **plane** | 10 | 1 |

and **both exceptions are informative**:

- `d.c:THERM` is a wall and does **not** trip. Same BC, same solver — but `d` is the atmosphere at 255.7 K against a 255.7 K wall, so the clamp threshold $(T_i + 20)/2 \approx 138$ K is far below the imposed trace and the channel is **not saturated**. That is the mechanism confirmed from the other side: saturation needs a wall much colder than the gas, which is the engine's situation and not the atmosphere's.
- `f.g:ADVEC` is a plane and *does* trip, **vacuously** — its response is the identically-zero shear-layer operator of §13.3, so "one cell responds and no others" is a statement about noise.

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

The four refusals are `L2/InterfaceMotion` $\times 3$ and `L5/R6` $\times 1$ — **and the second one is the framework catching this tier's own declaration error, unprompted.** §13.3 found the g–f ADVEC operator empty by hand; `L5/R6` refuses the graph for it independently, and names the reason:

> *"newton-krylov and direct-schur require a nonzero transmission operator; otherwise the interface system is not ill-conditioned but empty. Seams `['g-f']` assemble to a zero block … iterating an empty interface problem converges instantly and means nothing — the degenerate-axis trap, generalized: an axis whose value cannot affect the answer must be refused when the configuration is read, not discovered by measuring it."*

That is the PoC's stated product demonstrated on its own author: a coupling that would have run, produced numbers and meant nothing, **refused with the reason** — and refused at *compile* time, which is exactly what the last clause of that message asks for.

**The InterfaceMotion count is 3 rather than 9 because this graph declares one port type per seam where the fixture declared three** — the same three moving faces, the combustion front and the two plume boundaries, refusing for the same reason.

> **And they were briefly *gone*, which is the most useful mistake in this tier.** The first version of `build_rocket_real` declared `motion_class=STATIC` on every port, and the compile came back with **1 refusal instead of 4**. That is the "declare it away" failure in its purest form: the one genuine research hole in this graph, removed by a default in a builder rather than by any argument. It is now carried from `rocket.AGENTS` explicitly, and `PORT_MOTION` says why in the code.

---

## 14. Tier 80 — R2's remaining half: the composed defect at 50:1, and the bound that breaks at it

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

**The ratio changes direction between the two intervals, and that is the finding rather than a wobble.** Driving the shell with the gas's real near-wall temperature puts a **step** into its boundary condition at $t=0$ — the state was marched under $2800$ K and the referent hands it $2464$ K — and a startup transient decays over a fixed time, so it dominates a short interval and washes out of a long one. Spread across the two: $2.79\times$. That is why the declared trace is the default here and the two-way version is a control: **a referent that is more physical and less consistent with the state it starts from is not a better referent.** Closing it properly means marching the shell under the gas's own near-wall temperature from the start, which moves the probe state every number since Tier 76 was taken at.

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

**A1 held to four digits and A2 failed by $8.7\times$.** So the extrapolation's *premise* — that the trace drifts linearly, at a rate the shell's own single step predicts — is exactly right, and its *conclusion* is wrong, because the quantity that stopped being linear is the seam's response and not the trace.

**Cost, measured rather than estimated:** $9259$ s ($2.57$ h), $364\,093$ gas sub-steps, **zero Modern Standby events** over the window, at $100\%$ charge on AC. That is $92\,590$ s of wall per second of gas time, about $40\%$ above the sweep's own $65\,204$–$70\,070$ — the chamber accelerates as it develops, so a longer march is dearer per unit of gas time, not merely longer.

> **What this costs the rest of the vault.** [[case-study-brake-thermal-atlas-0.1]] §4's argument is that a bound written in the *interval* rather than the *ratio* can be checked at a ratio you can afford and applied at the $10^4$–$10^5$ a real conjugate seam runs at. That argument survives — §14.6 confirms the ratio is not the variable, $1900\times$ more sharply than CS-11 could. **What does not survive is extrapolating in the INTERVAL**, and CS-11 never claimed to; its own table spans $400\times$ of interval and stops. The rocket shows what happens one decade past where the checking stopped, and the answer is that the order changes.

**And the mechanism is bounded but not measured.** The chamber is $0.30$ m at a $50$ m/s inlet, so one flow-through is $\approx 6\times10^{-3}$ s: the sweep's largest interval is $\mathbf{0.83}$ flow-throughs and the anchor is $\mathbf{8.3}$. **[AI Inference]:** the order changes when the exchange interval crosses the fast agent's own residence time, because below it the gas cannot carry a wall perturbation out of the domain and above it the perturbation is advected through repeatedly. What supports this is the arithmetic above and the fact that nothing about the trace moved; what would settle it is $s_\Gamma$ measured at an interval inside the gap — flat means the growth is in how the run's lag couples, climbing means the seam's own linear constant stops being constant. That measurement is one stage (`--stages knee`), four marches of $10^{-2}$ s — $1.67$ flow-throughs, placed to straddle the threshold — and **$\approx 70$ minutes on AC**.

**It was attempted and it is not in this record.** The run died ten minutes in, during its settle, with an empty stderr and no Modern Standby event to blame, and **the cause is not known** — the anchor had survived nearly three hours by the identical launch. So the mechanism is named, priced and unpaid: it refines an inference, and the gate does not turn on it.

### 14.12 The predictions

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

**And the contrast with THERM on the same fourteen cells is the most striking number in the tier.** Tier 77 measured b–c THERM at $\kappa = 3.7141$. MECH is $8.82\times10^{8}$ — **eight orders apart, on the same seam, the same cells, the same operating point.** The reason is structural and it is already on the record: the THERM response is nearly *algebraic* (W301 — `_wall_flux` subtracts $T_{\text{wall}}$ directly), while the MECH response is a quasi-static elliptic solve, which is a **smoothing** operator and therefore compact, with geometrically decaying singular values. *Port type, not physics regime, is what decides whether a seam operator has a rank you can name.*

### 15.4 The gas cannot respond at all, and a declaration cannot see why

`Compressible2D` has eight boundary kinds — `wall_slip`, `wall_noslip`, `symmetry`, `inlet_massflow`, `freestream`, `outflow`, `prescribed`, `extrapolate` — and **none of them takes a wall velocity**. `_reflect(no_slip=True)` sets the ghost momentum to $-U$, which is a stationary wall, and `wall_noslip` reads only `T_wall` and `no_slip_mask`. Checked against the source rather than asserted, including the negative: no `u_wall`, `v_wall`, `wall_velocity` or `wall_speed` appears anywhere in the file.

> **So the gas can SUPPLY a traction — it has the wall pressure — and cannot RESPOND to one. The MECH bond at every gas–solid seam in this graph is one-sided by a missing capability, not by a modelling choice, and the declaration is identical either way.** W97 found a field-to-lumped MECH seam one-sided by $\mathrm{Re}_h$; this one is one-sided by a boundary condition that does not exist.

That is why MECH is still not wired into the seven-agent graph. Assembling it would produce a seam whose gas block is identically zero — the `L5/R6` empty-operator refusal that Tier 79 earned on g–f ADVEC, arrived at for a different reason and carrying a different remedy: g–f needed the *declaration* corrected, and this needs the *solver* extended.

### 15.5 The c–trajectory seam, which is two-sided

Composing the structure's hand-over with the body's response gives the seam in the rigid-mode space, $3\times14$:

$$\beta = 3.972877\times10^{-11}, \qquad \kappa = \mathbf{295.51}$$

against b–c MECH's $8.82\times10^{8}$ — **three million times better conditioned.** And it is genuinely two-sided: **the body responds to a load with a velocity natively.** It is the one port in this graph that needs no time derivative supplied and declares none, which is the exact contrast with §15.3's $\delta u/\Delta t$ and the reason the pair is worth building together.

### 15.6 What was deliberately not done

**The trajectory agent is built and measured and is NOT in the compiled graph.** The seven-agent compile is unchanged at `refuse`, **4 refusals, 22 decertifications, 81 admits** — exactly where Tier 79 left it, asserted rather than assumed, because *a builder that constructs declarations is a place where a refusal can be lost without anybody deciding to lose it* (Tier 79's own lesson). Adding an eighth agent would move the count, and it should move it for a reason that has been argued rather than as a side effect of a tier about something else.

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

**Worklist rows opened by this page:** W300 (the seam itself), **W301** (the saturated channel and its detector), W302 (the declaration ledger), **W303** (`effort_normal` at b–c — *closed at Tier 77, §10*), **W304** (the common seam base — *closed at Tier 77, §10*), W305 (MECH and the trajectory agent), **W306** (the passivity diagnostic cannot tell a global sign from an amplified mode — *implemented at Tier 78, §12*), **W307** (`thermal_seam`'s E7 holds by a film-coefficient accident), **W308** (`car_graph` declares `effort_normal` and has it inverted on two of three seams), **W309** (`grid.Block` face normals are index-oriented, so every gas seam needs `effort_normal`), **W310** (R0 complete — all seven agents on real physics), **W311** (a second undeclared interface, the nozzle wall), **W312** (a plane port's probe needs a cadence a wall's does not), **W313** (`Prolongation.nondim_diag` is specified and no case sets it, so every beta in this vault is in raw units), **W314** (R2's remaining half — the composed defect at 50:1 against CS-11's bound, *§14*), **W315** (`generate.py` collapses the shell to one number, so the build repo's shell-to-gas trace is rank 1, *§14.8*), **W316** (the shell is linearised about a gas temperature the gas does not have, *§14.9*), **W317** (a third undeclared interface — `a-c`, named by the build repo itself, and the declared seam is 40.7% of the conjugate wall, *§14.10*). **W305** (the trajectory agent and MECH on b-c — *§15*), **W318** (the b-c MECH operator has no spectral gap, so `expected_null_dim` is not a well-posed declaration for a compact response — *§15.3*), **W319** (`Compressible2D` has no wall-velocity boundary condition, so every gas-solid MECH bond is one-sided by a missing capability — *§15.4*), **W320** (`_solve_free` computes the trajectory's input as a Lagrange multiplier and discards it — *§15.2*). See [[gap-worklist]].
