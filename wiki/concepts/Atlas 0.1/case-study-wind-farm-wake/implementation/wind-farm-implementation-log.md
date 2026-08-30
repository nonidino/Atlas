# Wind-Farm Case Study — Implementation Log

**Type:** Implementation log — append-only build record (folder: `Atlas 0.1/case-study-wind-farm-wake/implementation/`)
**Guide:** [[impl-wind-farm-guide]] · **Spec:** [[spec-wind-farm-wake-atlas-0.1]] · **Figure:** [[wind-farm-agent-graph-figure]]

Format: `## [YYYY-MM-DD] <phase> | <status> — <what happened>`, status ∈ {built, fixed, optimized, measured, blocked, reverted}.

---

## Standing record — fill these in as they are determined

These four facts condition every number this case study produces. **Fill them in at W0 and W5; do not leave them to be reconstructed.**

| Fact | Value | Determined at |
|---|---|---|
| Fluid expert used (checkpoint + provenance) | **`camlab-ethz/Poseidon-T`** (scOT, 20,774,444 params), pretrained on NS-Sines / CE-KH / CE-Gauss on the periodic unit square, $T_{\text{final}}=1$, 21 snapshots. Loaded frozen: `.eval()`, `requires_grad_(False)` on every parameter, every forward under `no_grad`. Priority 1 of [[impl-wind-farm-guide]] §1.1; Walrus deferred as too large for a 15.4 GB CPU-only machine | W0, 2026-08-20 |
| Decoder mode ($\psi$ / Leray / soft) | **Velocity-direct + spectral Leray — level 4.** scOT has no stream-function head, so the level-5 path of spec §7.2 is unavailable. On the *periodic* window the Leray projection is spectral and exact: $\lVert\nabla\!\cdot\!\mathbf u\rVert_\infty$ goes from $9.08$ raw to $1.6\times10^{-13}$ for one FFT pair — far cheaper than the Poisson-solve-per-agent the guide budgets. **That discount does not survive to a non-periodic agent boundary, and W5 must not assume it does.** — *and W5 did assume it, which is the 2026-08-21 block. Superseded 2026-08-22: the coupled domain now uses a DCT-diagonalized Poisson solve with Neumann inlet/walls and a **Dirichlet outlet**, mass conservation moves from level 4 "documented" to $5.6\times10^{-14}$, and the Poisson solve the guide budgets as expensive costs $3\%$ of one expert sweep. The stream-function head is still absent; the level-5 path is reached by a different route* | W0, 2026-08-20; superseded 2026-08-22 |
| Native $\Delta t$ of the expert | **0.1 in expert time units**, measured rather than assumed. scOT was trained all2all on lead times $t/20$ for $t\in\{0,2,\dots,14\}$, i.e. $\{0,0.1,\dots,0.7\}$. Composing two native steps reproduces one double-length step to $2.0\%$; two half-native steps miss one native step by $18.5\%$ — an order of magnitude worse, confirming sub-native stepping is out of distribution. Under the adopted scaling this is $\Delta t=0.05$ in case-study units, exactly the spec's $\Delta t_{\text{macro}}$ | W0, 2026-08-20 |
| **Coupling step $\Delta t_{\text{macro}}$** — the spec says $0.05$; measurement says otherwise | **$0.25$ (expert lead $0.5$).** The only lead whose unforced $E'$ never exceeds its initial value at $t=60$ (peak ratio $1.000$), $5\times$ cheaper than the spec's step, and retaining $68\%$ of a wake at turbine 2 against an analytic $78.5\%$. Dissipation and spurious injection are both *per call*, so a coarser step fixes stability and wake survival together | W0 revisit, 2026-08-21 |
| Interface variant adopted (R1 halo / R2 flux-BC) | **R2 (flux-BC tokens).** Both passed W3, so by the guide's outcome table the declared-edge contract is validated and R2 is the one to use. R1 is kept as the positive control. **Band width is not a constant**: it must satisfy $\text{band}\ge U_\infty\Delta t_{\text{macro}}$ (2026-08-21), which at the adopted step is $0.25\,D$ = 16 cells, not W3's four | W3, 2026-08-20; width rule 2026-08-21 |
| **Adapter scaling** — not in the original table, and it conditions every number here | $L=2\,D$ per $128\times128$ window, $U_s=4\,U_\infty$, $T_s=L/U_s=0.5$, so one macro-step of $0.05$ is exactly one native expert step. W0 used $L=1,\;U_s=2$; W3 needs $L=2$ so that the $1\,D$ rotor does not span the full window height | W0/W3, 2026-08-20 |
| **Galilean decomposition** — forced by W0/E1 | The expert does **not** preserve a uniform flow, which is an exact steady solution at any viscosity: one step takes $\bar u$ from $1$ to $0.969$, and 1200 steps to $-0.083$. Its pretraining (NS-Sines) is zero-mean by construction, and a wind farm is nothing but mean flow. The adapter therefore evolves the fluctuation in the frame moving with the window mean and transports the mean by an exact spectral translation. A change of frame, not a correction | W0, 2026-08-20 |

## Integration-hours ledger

Falsification criterion F3 of [[f1-pathmap-and-end-goal]] asks whether per-expert integration cost stays flat as experts are added. **It cannot be reconstructed after the fact.** Log hours as they are spent.

**Caveat on the units.** W0–W3 were built by an agent in one continuous session, so these are *shares of one session's effort*, not human hours, and they are not comparable to a person's integration time. They are recorded anyway because F3 asks whether per-expert cost stays flat as experts are added, and a consistently measured relative number still answers that. Wall-clock compute is listed separately because that part *is* directly comparable.

| Component | Share of W0–W3 effort | Compute (wall clock) | Notes |
|---|---|---|---|
| Disk expert (`disk.py`) | ~8% | < 1 s | Zero parameters, ~200 lines, exact. Went in almost without friction — which is the point of a closed-form expert |
| Fluid-expert adapter (`adapters.py`) | ~30% | ~6 min (the W0 rollout) | The expensive one, and nearly all of the cost was **discovering what the checkpoint actually does**: channel layout, asymmetric $u/v$ normalization, trained lead times, the mean-flow failure, the dissipation cutoff. None of that is in the model card |
| Geometry + ports (`geometry.py`, `ports/types.py`) | ~22% | < 2 s | Passed first time on the physics; the two iterations were quadrature (switched to exact rectangle overlap) and the spec's diameter claim |
| Probe + reference solver (`probe.py`, `reference.py`) | ~35% | ~20 min | Includes a pseudo-spectral reference solver that was not in the plan and paid for itself three times over — see the W3 entry |
| Metrics (`metrics.py`, `residual.py`, `analytic.py`) | ~25% | < 20 s | Cheap in compute, and the *only* component so far whose bugs were all caught by its own gate rather than downstream. Building it before the system is what made that possible |
| Viewer (`gui/windfarm.py`, `windfarm.html`) | ~12% | — | Reimplements no physics, so it cost adapter work rather than modelling work. Caught the Betz bug |

**The number worth carrying forward:** integrating the *first* frozen expert cost roughly four times what the closed-form expert cost, and essentially all of that was reverse-engineering an undocumented interface rather than writing coupling code. If F3 holds, the *second* frozen expert should be much cheaper, because the port and adapter machinery now exists. If it is not cheaper, F3 is in trouble.

## Gate ledger

| Gate | Status | Number | Date |
|---|---|---|---|
| W0 ingredients | **pass at the adopted configuration** | Poseidon-T, 20.77 M params, 0.21 s/forward on CPU. At the spec's $\Delta t=0.05$ the unforced rollout reaches $t=60$ but $E'$ grows to $2.47\times$ — recorded 2026-08-20 as a conditional pass. **At the adopted $\Delta t=0.25$ (lead $0.5$) it passes outright: peak $E'/E'_0=1.000$, no growth anywhere, mean flow held exactly, $68\%$ wake retention at turbine 2.** Poseidon-B (157.73 M) was evaluated 2026-08-21 and rejected — it passes the gate by annihilating the flow ($E'$ constant to six figures from $t=4.5$, wake retention $-0.5\%$) | 2026-08-20; revised 2026-08-21 |
| W1 geometry | **pass** | Partition exact — every sampled point lies in exactly one agent; areas sum to $192.0$ against a domain area of $192.0$; 15 declared edges $=$ 15 materialized pairs, no extras and no omissions beyond the two documented rotor lateral faces; normals verified $\text{src}\to\text{dst}$ at every sampled point; token counts match the spec table **exactly**, total 1144. BFS diameter **3**, not the spec's 4 | 2026-08-20 |
| W2 disk expert | **pass** | $C_P^{\max}=16/27$ at $a=1/3$ to $<5\times10^{-16}$; $C_T'=2$ at $a=1/3$ to the same; $C_T(a)$ matches theory to $10^{-15}$ across $a\in[0.05,0.35]$; $\int\mathbf f_{\text{disk}}\,dV = -T$ to $10^{-13}$ on four lattices including ones deliberately misaligned with the strip; $a>0.4$ raises rather than extrapolating | 2026-08-20 |
| **W3 go/no-go** ($r_T$ pre-projection) | **pass, both variants** | Teacher-forced, pre-projection: **R1 $r_T=6.2\times10^{-4}$, R2 $r_T=2.6\times10^{-3}$**, against a $5\%$ threshold — two orders of magnitude of margin. Reference-solver floor $8.9\times10^{-4}$ (R1) and $1.1\times10^{-3}$ (R2). Field agreement with the reference: $\langle U_d\rangle$ within $0.80\%$ (R1) / $1.07\%$ (R2), $T$ within $1.59\%$ / $2.14\%$, field rel-$L_2$ $0.21$ / $0.18$. **Read the W3 entry before quoting $r_T$ — the gate is weak at $\Delta t=0.05$.** Re-validated 2026-08-21 at the adopted $\Delta t=0.25$ over the identical horizon: $r_T=1.1\times10^{-2}$, $\langle U_d\rangle$ within $0.140\%$, field rel-$L_2$ $0.128$ — better field agreement, and the null controls that made the gate weak now **fail** it | 2026-08-20; re-validated 2026-08-21 |
| W4 metrics | **pass** | All three items to machine precision: uniform-flow null (worst port $1.1\times10^{-16}$, $\mathcal R=0$, symmetry $=0$); Jensen recovery ($a$ to $5.6\times10^{-16}$, $k$ to $1.4\times10^{-16}$; BPA $\sigma$ to $2.8\times10^{-16}$); $\mathcal R$ without $P_{\text{ext}}$ equals the extracted power to $1.1\times10^{-16}$. Four bugs found — velocity-as-flux, occupancy-as-quadrature-weight (48% work error), constant-deficit-as-Gaussian, and the Betz canary firing on the viewer | 2026-08-21 |
| W5 coupling | **partial — one item still blocked, for a new reason** | Rollout reaches $t=60$ (240 steps, 64 min, all finite); ports 15/15; fixed point max 5, mean 3.02, converged every step; Betz exactly $16/27$; declared shear interfaces transmit cleanly (no jump at $y=\pm2$). **Mass: was $L_2\le6.6\times10^{-2}$ "documented"; now $5.6\times10^{-14}$ — the spec's $10^{-8}$ met on the velocity path** (2026-08-22 pressure solve). **Pre-projection $r_T$ was $0.73$ at $32\times$ its null floor and unreadable; now $0.14$ / $0.047$ at $2.5\times$ floor with a $3.6$–$6.7\%$ error bar.** **Post-projection $r_T$ still BLOCKED** — the pressure was one of two blocks; the second is that spec §7.3's min-norm correction direction is a *uniform* field, which the momentum balance is Galilean-invariant under ($dg/d\lambda=10^{-6}$). $P_2/P_1$ $1.140\to1.138$: the pressure was **not** its cause | 2026-08-21 |
| **W9 monolithic baseline** | **built and converged** (out of order — the pressure fix made it cheap) | `reference.ChannelNS`: 2D NS on the real domain with the real BCs, validated against two exact potential-flow laws. Steady to drift $3\times10^{-5}$; upstream induction $5.7\%$ at $x=-1$, insensitive to halving $\nu$, halving $dx$, and thinning the strip $2.5\times$. **$P_2/P_1 = 0.40$–$0.43$, inside W10's band.** Wake still $28.3\%$ deep at $x=16$ — in 2D wakes barely recover | 2026-08-22 |
| W6 Betz canary | **pass** | $C_P^{\max}=0.592593$ against Betz $0.592593$ — exactly $16/27$ at $a=1/3$, never above, every step. The canary fired first on the gate script itself, which normalized by $\langle U_d\rangle$ instead of the undisturbed local inflow $U_{\text{ref}}=\langle U_d\rangle/(1-a)$ and reported $C_P=2.0$ — the identical error W4 records it catching on the viewer | 2026-08-23 |
| W7 symmetry | **fail — cause reattributed 2026-08-23** | mirror residual $2.77\times10^{-2}$ at $t=5$ and $7.42\times10^{-2}$ at $t=10$, against a $10^{-6}$ gate, **growing rather than settling**. First read as guide §10's path-dependent message passing around the graph cycle; **that is wrong**. The frozen expert is **not mirror-equivariant**: one call on a *uniform inflow* returns $\max\lvert u-\mathcal Mu\rvert = 2.42\times10^{-2}$, $2.4\times10^4$ times the gate, with no graph, ports, agents or tiling anywhere in the loop. Deterministic; grows to $1.02\times10^{-1}$ over eight calls; `SolverExpert` through the identical harness gives $10^{-15}$. **OP-6** | 2026-08-23 |
| W8 wake recovery | **fail** | far-wake Gaussian fit and BPA corridor both missed. Cause is OP-3, not the coupling: the checkpoint retains $1.6\%$ of a wake at $x=16$ where the 2D baseline retains $28.3\%$ | 2026-08-23 |
| **W9 decomposed vs. monolithic** | **fail as stated — and the ratio is the result** | Composition error $A-B=\mathbf{0.273}$ against the gate's $5.5\times10^{-3}$, a $50\times$ miss. **But expert error $B-C=1.762$, so composition is $6.5\times$ smaller than the checkpoint's own contribution.** `B` is the same 124 tiles with the agents removed. **The monolithic run also inverts $P_2/P_1$ ($1.220$ against the partitioned $1.143$)** — so the case study's most visible defect is *not caused by Atlas's decomposition*: remove every agent and port and it persists. Only replacing the expert with a solver fixes it ($0.404$) | 2026-08-23 |
| W10 array efficiency | **fail** | $P_2/P_1=1.197$ against $[0.4,0.8]$. Attributed by W9 to the expert, not the composition — the no-agent run gives $1.220$ and the classical solver $0.404$ | 2026-08-23 |
| W11 power residual | **not measurable — OP-4** | $\lvert \mathcal R\rvert =1.61$ against $1\%$ of $0.669$. **The instrument was validated first** (W4's identity to $10^{-12}$; outflux to $2\times10^{-4}$ and dissipation to $3\times10^{-5}$ on smooth fields), so the fault is the gate's: its dissipation term needs a single $\nu$, and **$81\%$ of the field's dissipation lives below $0.125\,D$**, exactly the cutoff where W0 established this checkpoint has no single $\nu$ | 2026-08-23 |
| Scaling sweep $N=2,3,5,8$ | not started | | |

## Operating envelope of the frozen expert — read before W5

Measured at W0, and binding on everything downstream.

| Constraint | Value | Consequence |
|---|---|---|
| Autoregressive horizon at the spec's $\Delta t_{\text{macro}}=0.05$ | stable to $t\approx42$, drifts past it | The spec's $t_{\text{end}}=60$ rollout **cannot be run at the spec's timestep with this checkpoint** |
| Stable horizon at lead $\ge0.5$ ($\Delta t_{\text{macro}}\ge0.25$) | $t=60$ with $E'/E'_0=0.06$ | Available, at $5\times$ the coupling step and with heavy smoothing |
| Per-call noise floor | $0.031\,U_\infty$ peak emitted from an *exactly uniform* field; the identity lead reproduces its own input only to $0.23\%$ | Nothing measured below $\sim3\%$ of $U_\infty$ in a single step is signal |
| Effective viscosity | **not a single number.** Modes at $\lambda\ge0.25\,D$ show no decay above the $\sim2\%$ per-step bias; at $\lambda=0.125\,D$ decay is clean and exponential, $\nu=4.9\times10^{-4}$, $r^2=0.998$ | The checkpoint behaves like an LES with a spectral cutoff, not a Newtonian fluid. Any $\mathrm{Re}_{\text{eff}}$ quoted in this case study describes that cutoff |
| Window size is not a free parameter | $\mathrm{Re}_{\text{eff}} = T_s/(\nu_p L^2)$ | Doubling the physical size one window covers **quarters** $\mathrm{Re}_{\text{eff}}$. Covering $\Omega$ with fewer, larger windows is not a performance knob — it changes the physics being solved |

---

## [2026-08-19] project | planned — case study specified, figure drawn, guide written

[[spec-wind-farm-wake-atlas-0.1]] (binding numbers), [[wind-farm-agent-graph-figure]] (plan view + port graph + schedules), and [[impl-wind-farm-guide]] (phases W0–W6, gates W0–W11) are complete. Nothing built yet.

**One correction already banked before any code exists.** Drawing the partition to scale exposed that the rotor strips ($\lvert x\rvert\le0.05$) overlapped agent $I$, and that the bypass corridors were tabulated as $x\in[-6,18]$ while the edge table placed the $I\!-\!B^\pm$ interface at $x=0$ — a contradiction inside the spec itself. The partition is now disjoint: rotors at $x\in[0,0.1]$ and $x\in[7,7.1]$, bypasses at $x\in[0,18]$, with $N$ and $W$ carrying a notch. Left in place this would have failed W1's disjointness assertion, or worse, passed silently and produced double-counted tokens.

**Three things to hold onto when the build starts:**

1. **Report pre-projection residuals.** The §6.3 projection drives $r_T$ to machine zero by construction; a post-projection number proves only that the projection works. The scientific result is the residual *before* it.
2. **Do not train anything.** If a phase appears to need training, the phase or the expert is wrong. Fine-tuning "a little" destroys the claim the case study exists to test.
3. **A smaller timestep is not available as a remedy.** Classical co-simulation fixes coupling instability by shrinking the communication step; a frozen expert has a native $\Delta t$ and going below it is out of distribution. If instability appears the responses are halo exchange, IQN-ILS, or a negative result.

---

*Build entries begin below.*

---

## [2026-08-20] W0 | built, measured — Poseidon-T brought up; two findings that changed the adapter

**Code:** branch `atlas-0.1-windfarm` of `nonidino/physics-foundation-model`, `src/atlas/cases/windfarm/adapters.py`, `scripts/windfarm_w0_bringup.py`, results in `results/windfarm/w0/`.

**Hardware.** Intel Core Ultra 7 155H, 16 cores, 15.4 GB RAM, **no CUDA** (Intel Arc iGPU, torch 2.7.1+cpu). One 128×128 forward pass takes **0.21 s**. That is fast enough that W0–W3 never needed a GPU; W5 and the $N=2,3,5,8$ sweep will.

Six experiments, in the order their results feed each other.

### E1 — the checkpoint destroys a uniform flow

A spatially uniform flow is an **exact steady solution** of the periodic incompressible equations at any viscosity. Fed $\mathbf u = (1,0)$, Poseidon-T returns a mean of $0.969$ after one step and $-0.083$ after 1200. It is pulling the field toward zero mean, which is what its pretraining distribution contains — NS-Sines initial conditions are zero-mean by construction. **A wind farm is nothing but mean flow, so used naively this checkpoint cannot do the case study at all.**

The fix is a change of frame, not a fudge. For a constant mean velocity $\mathbf U$,

$$\mathbf u(\mathbf x,t) = \mathbf U + \mathbf u'(\mathbf x - \mathbf Ut,\;t)$$

with $\mathbf u'$ solving the same equations at zero mean. The transformation is **exact**, not a first-order splitting, because Navier–Stokes is Galilean invariant and $\mathbf U$ does not vary in space. So `step()` subtracts the window mean, asks the expert to advance the fluctuation, re-zeroes the fluctuation mean (recording what the expert *tried* to do to it as `last_mean_drift`, since for unforced periodic flow the mean is exactly conserved and any drift is model error), translates by $\mathbf U\Delta t$ — which *is* the mean advection term the expert was not asked to do — and adds $\mathbf U$ back.

With that: one-step error $3.3\%\to0.55\%$, and $\bar u$ after 1200 steps $-0.083\to1.000$ exactly.

**The naive numbers are kept in the record rather than deleted.** The size of the naive failure is the measurement, and it is what justifies the adapter carrying a change of frame at all.

**[AI Inference]:** step 4 of that sequence — the translation — is where interface data enters once a window stops being periodic, because the translation pulls in material from upstream that in the coupled system belongs to the neighbouring agent. That makes the Galilean split the natural home for the R1/R2 decision rather than a second mechanism sitting beside it. Worth checking at W5, where the windows really are non-periodic.

### E2 — the axis convention, measured rather than assumed

scOT reads `solution[..., 0:2]` and reshapes, with a per-dataset `transpose` flag, so which array axis "horizontal velocity" runs along is a property of the data files and is not fixed by the model card. Advecting a vortex in a uniform stream and watching which way it goes: displacement $(+0.2047, +0.0037)$ against an expected $(+0.2000, 0)$ — **axis 1 is $x$**, advection ratio $1.024$.

Worth recording how the first attempt failed. The vortex was specified by its *vorticity* and induced a velocity of $0.015$, which is **below the expert's own noise floor of $0.031$** measured in E1 — so the tracker followed the noise and returned nonsense. Specifying the peak fluctuation velocity instead fixed it. A probe below the instrument's noise floor measures the instrument.

### E3 — effective viscosity is not a single number

Fitting an exact Taylor–Green decay across the whole trained lead range, per wavenumber, regressing $\ln(\text{ratio})$ on lead so that a fixed per-step bias separates from genuine decay:

| $\lambda$ | $\nu$ | $r^2$ | bias | decay over lead range |
|---|---|---|---|---|
| $0.500\,D$ | $1.8\times10^{-5}$ | 0.27 | 0.979 | $+0.001$ |
| $0.250\,D$ | $-1.4\times10^{-5}$ | 0.92 | 0.980 | $-0.010$ |
| $0.125\,D$ | $4.9\times10^{-4}$ | **0.998** | 1.096 | $+0.685$ |

At $\lambda\ge0.25\,D$ there is no decay above the $\sim2\%$ per-step bias — the "decay" is a lead-independent offset, and a single-ratio estimate turns it into a viscosity six times too large. At $\lambda=0.125\,D$ decay is clean and exponential. A Newtonian fluid gives the same $\nu$ at every $k$; this does not. **The checkpoint behaves like an LES with a spectral cutoff.** $\mathrm{Re}_{\text{eff}}\approx1000$ (at the probe scaling, $255$) describes that cutoff and should never be quoted as a fluid property.

This matters directly for W3: the rotor strip is $\Delta_d=0.1\,D$ thick, which puts its Fourier content **inside the dissipation range**. The expert will smear the momentum sink.

### E4 — decoder mode: level 4, and cheaper than budgeted

No stream-function head exists in scOT, so spec §7.2's level-5 path is unavailable and the guide's fallback applies. Raw decoded divergence is $\lVert\nabla\!\cdot\!\mathbf u\rVert_\infty = 9.08$, about $9\times$ the natural scale $\lvert\mathbf u\rvert/L$ — the decode carries substantial grid-scale noise, which spectral differentiation amplifies. The Leray projection takes it to $1.6\times10^{-13}$ for one FFT pair, changing the field by $1.2\%$ in rel-$L_2$.

One bug worth recording because it will recur. The projection at first only reduced divergence from $74$ to $1.3$, not to machine zero, because of the **Nyquist mode**: the Nyquist coefficient of a real field is real, multiplying it by $ik$ makes it imaginary, and taking the real part of the inverse transform silently discards it — so the divergence operator and the projection meant to cancel it disagreed at exactly one mode. Zeroing $k$ at Nyquist once, in one shared helper, fixed it. **Level 4 here is much cheaper than the "Poisson solve per agent per step" the guide budgets, but only because the window is periodic. W5's agents are not, and must not inherit the assumption.**

### E5 — the gate, and where it fails

Unforced rollout from uniform flow plus $10\%$ ambient turbulence (spec §2.1's $I_\infty$), 1200 steps of $\Delta t=0.05$ to $t=60$, 262 s. No NaN, mean flow held exactly, and total energy within $2.9\%$.

But the fluctuation channel — the part the expert is actually integrating — decays to a minimum of $0.109\,E'_0$ at $t=3.6$, then **grows monotonically**, passing $E'_0$ at $t=42.5$ and reaching $2.47\,E'_0$ at $t=60$. Two-dimensional unforced turbulence cannot gain energy, so this is unambiguous and needs no reference solution.

Sharper still: started from an **exactly uniform** flow, where the true answer is exactly zero fluctuation forever, 1200 steps manufacture a $7.8\%$ velocity field out of nothing. The expert has a spurious attractor and any long rollout drifts toward it.

**So gate W0 as literally specified — "stable to $t=60$ with no energy drift" — fails at the spec's timestep.** It holds to $t\approx42$.

### E7 — the drift is not a simple per-step accumulation

Since W0 failed, the actionable question is what envelope it holds in. Sweeping the lead time to the same physical horizon $t=60$:

| lead | $\Delta t_{\text{macro}}$ | steps | $E'$ from uniform (true: 0) | $E'/E'_0$ from turbulent |
|---|---|---|---|---|
| 0.10 | 0.050 | 1200 | $1.49\times10^{-2}$ | $2.47$ |
| 0.20 | 0.100 | 600 | $9.40\times10^{-2}$ | $4.95$ |
| 0.35 | 0.175 | 343 | $6.4\times10^{-4}$ | $7.48$ |
| 0.50 | 0.250 | 240 | $1.6\times10^{-4}$ | $0.063$ |
| 0.70 | 0.350 | 171 | $1.1\times10^{-4}$ | $0.011$ |

Non-monotone, with a sharp transition between $0.35$ and $0.50$. At leads $\ge0.5$ the checkpoint is stable to $t=60$ and the fabricated field stays at $\sim1.5\%$ of $U_\infty$ — but the fluctuations are also wiped out ($E'$ down $87\times$), which is over-dissipative rather than correct.

**This inverts the guide's §7.2 warning in an interesting way.** The guide says a smaller communication step is unavailable as a remedy because the expert has a native $\Delta t$ and going below it is out of distribution. Measured: going below native is indeed bad (E6: $18.5\%$ against $2.0\%$ for composing native steps), *and* stepping **at** native for 1200 steps is unstable while stepping at $5\times$ native is not. The frozen expert has a **minimum viable coupling step well above its nominal native lead**, and it is not the same number. Neither the model card nor the paper states it.

### Gate W0

Conditional pass, envelope recorded above. The guide says an unstable unforced rollout should stop the project, and the rollout is stable in a well-defined envelope that W1–W3 sit entirely inside — the whole of W3 is 300 steps at $t\le15$, comfortably inside the $t\approx42$ limit at the native step. **W5 and W6 are the phases this constrains**, and the decision there is between a $5\times$ coarser coupling step, a shorter horizon than the spec's $t=60$, or a different checkpoint. Recorded now rather than discovered then.

---

## [2026-08-20] W1 | built — geometry and ports, no model involved

**Code:** `src/atlas/cases/windfarm/geometry.py`, `src/atlas/ports/types.py`, `tests/atlas/windfarm/test_w1_geometry.py`. `verify()` runs at import, following the Phase-1 scaffold's rule that a broken partition should fail the moment anything imports Atlas.

`src/atlas/ports/` is kept free of wind-farm specifics: it is the port algebra of [[port-algebra-atlas-0.1]] and every later case study reuses it. The five types are a closed enum; `port("momentum")` raises, so the pre-port-algebra vocabulary cannot come back by accident. `MappingMode.apply` implements the conservative/consistent split directly, so the two cannot be confused at a call site.

**Gate W1: pass.** Every item.

- **Disjointness and coverage as an identity, not a tolerance.** The rocket's `contains` used closed sets and took care to probe only cell centres. Here every interface is axis-aligned and lands exactly on the token lattice, so closed sets would put $(0,1)$ in both $I$ and $N$. The boxes are half-open on exactly the faces they share with a downstream neighbour, which makes *exactly one agent contains every point of $\Omega$* true pointwise. Areas sum to $192.0$ against a domain area of $192.0$.
- **Interfaces on both boundaries, and normals, in one test.** Step off each curve by $10^{-7}$ in both directions and ask `owner()` who is there. That single assertion covers curve-lies-on-both-boundaries, correct pairing, and normal orientation at once, and it cannot be fooled by a sign convention written down in a comment.
- **Materialized pairs $=$ declared pairs**, 15 each, found by probing both sides of every candidate line rather than by reading the edge table — so an interface the table forgot would still show up. The only shared boundary left undeclared is the two rotors' lateral faces, $0.206$ against the documented $0.2$ (the excess is probe resolution), which spec §3.1 declares as a deliberate omission. Anything else undeclared fails.
- **Token counts match the spec table exactly**, not within the $10\%$ the gate allows: 192/20/192/256/20/176/144/144, total **1144**. Per-agent spacing is $0.25$ in $N$ and $F$ and $0.5$ elsewhere, with the rotor strips at $0.05\times0.1$ — which is what reproduces the spec's own numbers.

**Two corrections to the spec, both recorded rather than silently patched.**

1. **The graph diameter is 3, not 4.** The spec quotes 4 via $I\to R_1\to N\to F\to R_2$, but that path is not a geodesic — $I\to N\to F\to R_2$ is one hop shorter, and BFS over the declared edge list gives 3. `n_mp = 4` is kept: it matches the Phase-1 scaffold, and one round more than the diameter is conservative rather than wrong. This is a documentation error, not an architectural one.

2. **The notch is sub-token and a boolean mask loses it.** $N$'s rotor notch is $0.1$ wide and $N$'s tokens are $0.25$, so no cell *centre* falls inside the notch and a boolean mask would carve out nothing while the rotor's own tokens counted the same area again. Agent masks are therefore **fractional** — `AgentGrid.frac` is the exact area fraction of each cell inside the agent — and the discrete coverage identity $\sum_{\text{agents}}\text{frac}\cdot\text{area} = \lvert\Omega\rvert$ holds. This is a real consequence of the spec's own token spacing, not a discretization convenience.

Occupancy was first computed by $8\times8$ subsampling, which reported $N$'s area as $11.90625$ against a true $11.9$ — the notch does not land on eighths of a $0.25$ cell. Every region here is a box minus boxes, so the overlap has a closed form; using it makes the number exact. A $0.05\%$ error would never have surfaced as anything but a slightly wrong flux integral much later.

---

## [2026-08-20] W2 | built — the actuator disk, against theory

**Code:** `src/atlas/cases/windfarm/disk.py`, `tests/atlas/windfarm/test_w2_disk.py`. Zero parameters.

**Gate W2: pass**, every item to machine precision.

- $C_P = 4a(1-a)^2$ maximal at $a=1/3$ with value $16/27$ to $<5\times10^{-16}$, and verified to be the *maximum* by scanning $2\times10^6$ values of $a$, not just evaluated at the design point.
- $C_T' = 2$ at $a=1/3$ to $<5\times10^{-16}$; $C_T(a) = 4a(1-a)$ to $10^{-15}$ across $a\in[0.05,0.35]$.
- $\int\mathbf f_{\text{disk}}\,dV = -T$ to $10^{-13}$ on four lattices, **including ones deliberately misaligned with the strip**. The force field weights each cell by the exact fraction of its area inside the strip, so this holds on any lattice — which matters because a discretization losing $3\%$ of the thrust would appear as a $3\%$ interface residual at W3 and be blamed on the fluid expert.
- $a>0.4$ raises; $a\in(0.35,0.4]$ is clamped to $0.35$ with the clamp recorded in the returned state.
- The local-induction form is checked against the freestream form where they must agree, and separately checked to *disagree* the way it should: halving the inflow quarters the thrust. That is the property turbine 2 depends on.
- The `ROT` port carries $(\tau,\omega)$ with $\tau\omega = P$ to $10^{-13}$. The split needs a rotor-speed model the disk does not have, so a design tip-speed ratio of $7.5$ is **declared**; nothing downstream depends on it, because the port is unconnected and only the product enters $P_{\text{ext}}$.

**One bug that gate W2 could not have caught on its own.** `disk.py` was written with arrays indexed $[x,y]$ while `adapters.py` — matching what W0/E2 *measured* about the expert — uses $[y,x]$. Both modules were internally consistent and mutually transposed, and every W2 test passed, because a disk centred in a square window looks much the same either way. It surfaced only when the two were first driven together. The convention is now $[y,x]$ everywhere, and `test_force_field_orientation_matches_the_expert_convention` asserts the strip is a tall narrow column rather than a short wide row. **Per-component gates cannot catch cross-component conventions; the first integration test is where they appear.**

---

## [2026-08-20] W3 | measured — go/no-go: PASS, both R1 and R2

**Code:** `src/atlas/cases/windfarm/probe.py`, `src/atlas/cases/windfarm/reference.py`, `scripts/windfarm_w3_probe.py`, `tests/atlas/windfarm/test_w3_probe.py`. Results in `results/windfarm/w3/`.

### The result

Teacher-forced, **pre-projection** (no thrust projection exists yet — §6.3 is W5):

| variant | band | reference solver | expert, teacher-forced | expert, cold start |
|---|---|---|---|---|
| **R1** Schwarz halo | $0.5\,D$ | $8.9\times10^{-4}$ | $\mathbf{6.2\times10^{-4}}$ | $6.1\times10^{-5}$ |
| **R2** flux-BC tokens | $0.0625\,D$ (one expert patch) | $1.1\times10^{-3}$ | $\mathbf{2.6\times10^{-3}}$ | $2.8\times10^{-3}$ |

Against a $5\%$ threshold: two orders of magnitude of margin, and the expert is at the reference solver's own floor. By the guide's outcome table, **both passing means the declared-edge contract is validated — proceed, use R2.**

Field agreement at the **cold-start** converged state — the expert released from uniform flow to find its own fixed point over 300 steps with only the band — which the $r_T$ gate does not constrain: $\langle U_d\rangle$ within $0.80\%$ (R1) and $1.07\%$ (R2), thrust within $1.59\%$ and $2.14\%$, field rel-$L_2$ (normalized by the *perturbation*, not the field, so the uniform stream cannot flatter it) $0.21$ and $0.18$.

**The cold-start column is the stronger evidence.** Teacher forcing hands the expert a fixed point that is not its own, so all it can be observed doing is relaxing away from it; at 60 steps ($t=3$, about one window transit) it is mid-transient, with $\langle U_d\rangle$ excursed to $0.858$ (R1) and $0.682$ (R2). The cold start asks the harder question and answers it: released from uniform flow, this expert's own steady state sits within $1\%$ of a real solver's, and none of that is inherited from an answer it was given.

### The gate is weak, and this qualifies the result

A gate nothing can fail grades nothing, so the metric was run against null models:

| control | $r_T$ | passes $5\%$? | field rel-$L_2$ | $\langle U_d\rangle$ error |
|---|---|---|---|---|
| identity (returns its input) | $3.1\times10^{-1}$ | no | 2.16 | 88.6% |
| **rigid advection** (translation at the mean; no pressure, no viscosity, no dynamics) | $9.5\times10^{-3}$ | **yes** | 0.26 | 3.0% |
| reference $+\,5\%$ divergence-free noise | $4.6\times10^{-3}$ | **yes** | 0.21 | 0.02% |

Repeated in the **exact teacher-forced configuration the headline comes from** — started at the reference's converged state, band filled from it — the identity fails in both variants ($r_T = 6.1\times10^{-2}$ under R1, $2.2\times10^{-1}$ under R2, with $\langle U_d\rangle$ off by $65\%$ and $62\%$), which settles the obvious objection to a teacher-forced measurement: the headline number is not achievable by holding the state the expert was handed. Rigid advection still passes.

At the full horizon the $r_T$ criterion rejects only a do-nothing operator. **$r_T<5\%$ is necessary and weak: it constrains the momentum budget, not the flow.** The frozen expert beats rigid advection on every measure — $r_T$ $2.6\times10^{-3}$ against $9.5\times10^{-3}$, $\langle U_d\rangle$ error $1.1\%$ against $3.0\%$, field rel-$L_2$ $0.18$ against $0.26$ — and that comparison, not the bare $r_T$, is what makes the pass meaningful. Asserted as a test (`test_the_gate_does_not_reject_rigid_advection_and_this_is_a_limitation`) so that a future change making the gate discriminating shows up as a failure to re-examine rather than passing quietly.

### The band sweep: no cliff

R1 and R2 are two points on one axis — how much of the neighbour's field is written into the upstream band. Sweeping it (65 min CPU), the expert tracks the reference to $0.25$–$1.21\%$ on $\langle U_d\rangle$ across a twelvefold range of band width, with **no threshold below which it breaks down**: 4 cells $1.07\%$, 8 cells $1.21\%$, 16 cells $1.02\%$, 32 cells $0.80\%$, 48 cells $0.25\%$. Cold-start $r_T$ improves monotonically with band width ($2.8\times10^{-3}$ down to $6.1\times10^{-5}$), so a halo helps — it just is not required. **Four cells, one of the expert's own patch tokens, is enough.**

**A confound that forbids the obvious reading.** The reference's own $\langle U_d\rangle$ moves from $0.764$ to $0.923$ across the sweep, because a wider band pins more of the window to uniform inflow and reduces the effective blockage. The band width therefore changes the *problem*, not only its numerical treatment, and the $r_T$ column may not be read down its length; only expert-against-reference at a fixed band is a valid comparison. The two runs at $0.125\,D$ are a duplicated configuration and agree to every printed digit — a free determinism check.

### Four things the probe had to settle first

**The expert has no forcing input.** Poseidon takes a velocity field and returns a velocity field; forced Navier–Stokes appears in the Poseidon paper only as a downstream *finetuning* task, and nothing here is finetuned. The force therefore enters by Strang splitting — half impulse, one expert step, half impulse, with a Leray projection after each impulse to absorb the pressure. Over the whole periodic window the applied impulse is preserved exactly, because the projection touches every Fourier mode except $k=0$ and the total force *is* the $k=0$ mode.

**There is no pressure to read.** The incompressible datasets fill the density and pressure channels with the constants 1 and 0, so the checkpoint predicts no pressure at all — and the momentum balance needs $p$ on the control-volume faces. Pressure is not an independent field in incompressible flow: it is recovered from $\nabla^2p = \nabla\!\cdot\!\mathbf f - \nabla\!\cdot\!((\mathbf u\!\cdot\!\nabla)\mathbf u)$, spectrally, and matches the analytic Taylor–Green pressure to $2\times10^{-15}$. A derivation, not a model.

**A periodic pressure cannot carry a mean gradient.** With the disk force alone the balance was short by exactly $\langle f_x\rangle V_{\text{slab}}$ — a rock-steady $30.4\%$ residual at a converged state with $dM/dt = 10^{-12}$. Not a transient, not the expert: a missing term. Driving the domain with a uniform $+T/\text{Area}$ so the *total* applied force has zero mean is the standard construction for periodic wind-farm LES (Calaf, Meyers & Meneveau 2010) and makes the problem well posed. The residual drops to $10^{-3}$. Asserted in `test_counter_force_is_what_closes_the_balance_not_the_solver`.

**Teacher forcing is an exchange, not a frozen thrust.** Each side is handed the other's value once per step with no fixed-point iteration. Freezing $T$ at its freestream value instead is a different problem: with an inflow band it settles ($\langle U_d\rangle = 0.94$ under R1), but with the wake recirculating through a periodic window it runs away and reverses the flow ($\langle U_d\rangle = -0.42$ at $t=20$) — a fixed force plus a residence time that grows as the strip slows is a positive feedback, and only fresh inflow breaks it. The reason not to freeze $T$ is the physics rather than the stability, though: a frozen thrust is independent of local inflow, which is exactly the failure spec §4.2 forbids for turbine 2.

### The reference solver was not in the plan and should have been

A pseudo-spectral vorticity–streamfunction solver (`reference.py`, ~100 lines) drives the *identical* probe code — same splitting, same balance, same control volume — so a difference between it and the expert is a difference between them. It earned its place three times: it caught the frozen-thrust runaway, it exposed the missing mean pressure gradient at a converged state where the residual could not be blamed on a transient, and it establishes the floor $r_T$ that the expert is judged against. Without it, "the expert gives $2.6\times10^{-3}$" is a number; with it, it is a result.

This is guide §5's argument arriving a phase early: build the metric before the system that produces numbers nobody can check. **[AI Inference]:** the same solver is most of what W9's monolithic baseline needs, and running it over the undivided domain at W6 now looks cheap rather than optional.

### Realization details, and their limits

* **Probe window $2\,D$ square**, $L=2$, $U_s=4$, $128^2$ cells at $dx = 0.0156\,D$. One diameter per window (the W0 scaling) would put the $1\,D$ rotor across the full window height, leaving the flow nowhere to go around — a blockage problem, not a wake problem. At $2\,D$ the blockage is still $50\%$ against the spec's $8\,D$-tall domain. That is a limitation of the probe, and it is affordable *only because $r_T$ is an identity and holds at any blockage*. No wake number should be quoted from this window.
* **R1 and R2 as band widths.** With an operator that accepts no boundary conditions, the neighbour's data has to be written into the state itself, in a band along the upstream edge. R2 is one *expert patch* wide (4 cells — scOT's own `patch_size`), which is the closest thing to "the declared interface carries the port values and nothing else". R1 is $0.5\,D$ of neighbour interior. The band is also what breaks periodicity, erasing the wake that would otherwise wrap around and re-enter as inflow.
* **Control volume is a full-height slab**, so its lateral faces coincide and cancel identically. A volume hugging $\lvert y\rvert\le0.5$ instead adds two faces carrying most of the flux, whose discretization error would land on $r_T$ and be indistinguishable from the expert being wrong.
* **The unsteady form is what is measured**: $d/dt\int_V u_x\,dV + \Phi_x(V) = \int_V f$, true at every instant. The spec's steady $\Phi_x = -T$ is reported alongside and agrees once settled. The balance operator closes to $10^{-16}$ on an unforced reference solution, which is what licenses reading $10^{-3}$ from it.

### One unresolved anomaly, recorded rather than buried

In the **no-band periodic configuration** — no interface at all, an infinite streamwise array at $2\,D$ spacing — the slab balance does *not* close, leaving a residual of order $T$ that persists at $dM/dt\sim10^{-7}$ and flips sign as the control volume grows. The same operator closes to $10^{-16}$ unforced and to $10^{-3}$ in both R1 and R2. The working hypothesis is that the split scheme's fixed point is not a fixed point of the continuous equations, and that the band prevents the offset accumulating — **untested**. No number from that configuration is quoted anywhere. It is not on the W3 path, but it is on W5's, where several agents will run without an upstream band.

### Gate W3: pass

$r_T = 6.2\times10^{-4}$ (R1) and $2.6\times10^{-3}$ (R2), teacher-forced, pre-projection, both far inside $5\%$. **R2 adopted.** The frozen expert responds to a body force it never saw in training well enough to keep the momentum budget closed at the level a real solver does — falsification criterion F2 is not triggered.

---

## [2026-08-21] W0 | measured — Poseidon-B evaluated and rejected; coupling step and band rule decided

Three questions were open after W0–W3: whether the fluctuation-energy drift was a capacity artefact of a 21 M-parameter model, what coupling step W5 should use, and how wide the interface band has to be. One experiment answered the first and turned up the other two.

### Poseidon-B removes the drift by killing the flow

`camlab-ethz/Poseidon-B`, 157.73 M parameters — $7.6\times$ Poseidon-T — loads through the identical adapter with no code change (same `image_size`, `num_channels`, `use_conditioning`, `learn_residual`), and costs only $1.6\times$ the CPU time per forward ($0.33$ s against $0.21$ s).

It **passes** gate W0 where Poseidon-T fails. It should not be adopted.

| | Poseidon-T | Poseidon-B |
|---|---|---|
| $E'/E'_0$ at $t=60$ | $2.470$ (fails) | $0.007$ (passes) |
| $E'$ from $t=4.5$ to $t=60$ | grows | $6.7580\times10^{-5}\to6.7589\times10^{-5}$ |
| noise floor, uniform IC | $3.1\times10^{-2}$ | $7.7\times10^{-2}$ |
| one-step rel-$L_2$, uniform IC | $5.5\times10^{-3}$ | $1.3\times10^{-2}$ |
| dissipation cutoff | $0.125\,D$ | $0.25\,D$ |
| **wake retained at turbine 2** | $20.0\%$ | $\mathbf{-0.5\%}$ |

Poseidon-B's fluctuation energy is **constant to six significant figures from $t=4.5$ to $t=60$**, and its $u_{\max}$ is constant from $t=2$. That is not stability; it is a strongly attracting fixed point reached in about one convective time. It is worse than Poseidon-T on every axis measured independently of the gate — $2.5\times$ the noise floor, and a dissipation cutoff at *twice* the wavelength, meaning it erases structures twice as large.

**A gate that a model passes by destroying the field is a gate reporting the wrong thing.** W0 asks for a stable rollout and does not ask whether anything survives it. That gap is what §5.2b of [[results-w0-w3-wind-farm]] now closes with the wake-persistence measurement, and it is the reason the checkpoint decision could not have been made from the W0 gate alone.

### The measurement that decided it, and the control that made it readable

A Gaussian deficit uniform in $x$ is a parallel flow, so $(\mathbf u\!\cdot\!\nabla)\mathbf u\equiv0$ and the exact evolution is pure diffusion with a closed-form amplitude — an analytic answer, not another model's opinion. At $\nu=10^{-3}$ the deficit retains $78.5\%$ at $t=7$.

The first version of this measurement used peak-to-trough depth and was **uninterpretable**: run with *no wake at all*, Poseidon-T still developed a profile of depth $0.209$ by $t=10$ and Poseidon-B one of $0.095$, because both emit spurious structure every call and it accumulates on a closed window. For Poseidon-T the noise alone exceeded the signal. Adding a no-wake control on the identical path, and reporting retention as the projection of the profile onto the initial wake shape, separated the two — and turned Poseidon-B's apparently-respectable residual into what it is: $0.0946$ with no wake against $0.0950$ with one, i.e. **nothing**.

**The lesson is the one W3 already recorded in a different form.** A metric that cannot distinguish the signal from what the model invents cannot grade the model, and the only way to find out is to run the null case through the identical path. Both times the null control changed the conclusion rather than confirming it.

### The coupling step: lead $0.5$, and lead $0.2$ is a trap

Repeating the drift sweep at the W3/W5 scaling ($L=2\,D$; the earlier sweep was at $L=1$ and the two are not comparable), unforced to $t=60$, against wake retention at $t=7$:

| lead | $\Delta t$ | calls | $E'/E'_0$ at $t=60$ | peak $E'/E'_0$ | wake at $t=7$ |
|---|---|---|---|---|---|
| $0.10$ (native) | $0.05$ **(spec)** | 1200 | $1.79$ | $1.79$ | $20.0\%$ |
| $0.20$ | $0.10$ | 600 | $13.71$ | $14.74$ | $75.7\%$ |
| $0.35$ | $0.175$ | 343 | $4.04$ | $4.04$ | — |
| $\mathbf{0.50}$ | $\mathbf{0.25}$ | 240 | $\mathbf{0.086}$ | $\mathbf{1.000}$ | $\mathbf{68.2\%}$ |

**Adopted: $\Delta t_{\text{macro}}=0.25$, lead $0.5$.** Peak ratio exactly $1.000$ — the fluctuation energy never exceeds its initial value at any point in the rollout, which is the condition an unforced 2D flow must satisfy. It is $5\times$ cheaper than the spec's step and retains $68\%$ of a wake against the analytic $78.5\%$.

Lead $0.2$ looked like the answer for an hour: it has the best wake retention of anything tested, $75.7\%$ against an analytic $78.5\%$. It also has $160\times$ the drift of lead $0.5$. **It optimizes the quantity visible at $t=7$ and ruins the one that only appears by $t=60$** — which is exactly why the guide's instruction to run the long horizon rather than assume it is not optional.

Both effects have the same cause: **the dissipation and the spurious injection are both per *call*, not per unit time.** Halving the number of calls more than triples wake retention. A coarser coupling step is simultaneously the fix for stability, the fix for wake survival, and $5\times$ cheaper — which is a suspiciously good deal, and §3.3 is where the bill arrives.

### The band-CFL rule — and a correction to W3's "no cliff"

W3 concluded from a band sweep at $\Delta t=0.05$ that there is **no cliff**: four cells was as good as forty-eight, to within $1\%$. Repeating that sweep at $\Delta t=0.25$:

| band | cells | $\text{band}/(U_\infty\Delta t)$ | $\langle U_d\rangle$ error at $\Delta t=0.25$ | at $\Delta t=0.05$ |
|---|---|---|---|---|
| $0.0625\,D$ | 4 | $0.25$ | $\mathbf{19.6\%}$ | $1.07\%$ |
| $0.125\,D$ | 8 | $0.5$ | $2.27\%$ | $1.21\%$ |
| $0.25\,D$ | 16 | $\mathbf{1.0}$ | $0.427\%$ | $1.02\%$ |
| $0.5\,D$ | 32 | $2.0$ | $0.140\%$ | $0.80\%$ |
| $0.75\,D$ | 48 | $3.0$ | $0.197\%$ | $0.25\%$ |

**The cliff exists; the first sweep never crossed it.** At $\Delta t=0.05$ the flow advects $0.05\,D$ per step, so the smallest band tested was still $1.25\times U_\infty\Delta t$ — every point sampled was on the safe side, and the flatness was an artefact of the range, not a property of the mechanism. The rule is

$$\text{band width}\;\ge\;U_\infty\,\Delta t_{\text{macro}},$$

below which the band is overrun inside a single step: the flow crosses the interface layer before it can constrain the inflow, and the interface stops transmitting.

**Mechanism A survives; a number in it does not.** The interface still carries only declared port values rather than neighbour interior, so R2 stays adopted. What is retracted is "four cells is enough" as a constant — the minimum band is a function of the coupling step, and at the adopted $\Delta t=0.25$ it is $0.25\,D$, sixteen cells. **[AI Inference]:** this is a Courant condition on the *interface* rather than on the solver, and since it follows from writing boundary data into the state of an operator that accepts no boundary conditions, it should hold for every later Atlas case study using this edge mechanism.

### W3 re-validated at the adopted step

Re-running the probe at $\Delta t=0.25$ over the identical physical horizon ($t=15$ reference, $t=3$ expert — `--quick` at the coarse step covers the same time as the full run at the fine one):

| | $\Delta t=0.05$ | $\Delta t=0.25$ |
|---|---|---|
| $r_T$, teacher-forced, $0.5\,D$ band | $6.2\times10^{-4}$ | $1.1\times10^{-2}$ |
| $\langle U_d\rangle$ error vs reference | $0.80\%$ | $\mathbf{0.140\%}$ |
| field rel-$L_2$ | $0.21$ | $\mathbf{0.128}$ |
| rigid advection control | **passes** the gate | **fails** |
| reference $+5\%$ noise control | **passes** | **fails** |

**Gate W3 holds at the adopted step, and holds better.** Field agreement improves, and the two null controls that made the gate weak — the limitation W3 recorded and asserted as a test — now both fail it. The gate is discriminating at $\Delta t=0.25$ in a way it was not at $\Delta t=0.05$. The $r_T$ value itself rises an order of magnitude, but so does the reference solver's own floor ($8.9\times10^{-4}\to1.9\times10^{-2}$), which is the splitting error of a coarser step and is a property of the balance operator, not of the expert.

`test_the_gate_does_not_reject_rigid_advection_and_this_is_a_limitation` still passes, because it is written against $\Delta t=0.05$. It should be re-pointed at the adopted step in W4, where it will need inverting — the gate now *does* reject rigid advection.

### Gate W0, restated

**Pass at the adopted configuration.** Poseidon-T at lead $0.5$: unforced rollout to $t=60$ with peak $E'/E'_0=1.000$ — no growth anywhere — no NaN, mean flow held exactly, and $68\%$ wake retention at turbine 2. The failure recorded on 2026-08-20 was a failure at the *spec's* timestep, not of the checkpoint, and the resolution is a coupling step the spec did not anticipate.

---

## [2026-08-21] W4 | built — metrics validated against known answers, and a viewer

**Code:** `src/atlas/ports/residual.py`, `src/atlas/cases/windfarm/analytic.py`,
`src/atlas/cases/windfarm/metrics.py`, `scripts/windfarm_w4_metrics.py`,
`tests/atlas/windfarm/test_w4_metrics.py`. Viewer: `gui/windfarm.py`,
`gui/windfarm.html`, `run_windfarm_gui.bat`, `tests/atlas/windfarm/test_w4_gui.py`.

`src/atlas/ports/residual.py` is framework code, not case-study code: it is the measurement half of [[port-algebra-atlas-0.1]] and every later case study reuses it.

### Gate W4: pass, all three items to machine precision

| item | result |
|---|---|
| uniform flow: every $r_\Gamma$, $\mathcal R$, symmetry $=0$ | worst port $1.1\times10^{-16}$; $\mathcal R = 0$; symmetry $= 0$ |
| Jensen field: diagnostics recover the parameters given | $a$ to $5.6\times10^{-16}$, $k$ to $1.4\times10^{-16}$, at three $(a,k)$ pairs; BPA $\sigma(x)$ to $2.8\times10^{-16}$ at four stations |
| $\mathcal R$ is exactly the extracted power | $\mathcal R$ without $P_{\text{ext}} = 0.2962962963$ against $P = T\langle U_d\rangle$; difference $1.1\times10^{-16}$ |

The third item's reading is worth stating, because it is not "$\mathcal R$ equals the extracted power" as a target. **The balance with $P_{\text{ext}}$ omitted must equal the extracted power, and with it included must vanish** — which is the sharpest available statement that $P_{\text{ext}}$ *closes* the budget rather than being a free term sized to absorb whatever is left over. A test asserts the negative too: a ledger built with $P_{\text{ext}}=0$ must be wrong by more than 10% of the extracted power, or all three gate items would pass on a metric measuring nothing.

### Every null test is paired with a case the metric must reject

A metric that returns zero because it is not looking at anything returns zero on a uniform flow. So each of the gate's null items has a partner: the symmetry residual is checked on a sheared field, the corridor is checked against a wake that never recovers (failure mode 1 of the guide's triage table), the port residual is checked on two sides that disagree by a known factor, and the Jensen fit is checked on a field that is not Jensen.

### Four bugs the gate found, three of them mine

**1. Velocity is not a flux.** The port convention is that a flow is positive along the interface normal, so two agents that agree report values summing to zero. That is true of the mass flux and — by Newton's third law — of the traction, but **not of the velocity**, which is a property of the point and does not flip with the normal. Comparing it as a sum made every `MECH:flow` port read exactly $r=2.0$. Caught by the uniform-flow null test on the first run, which is what it is for.

**2. Fractional occupancy is not a quadrature weight.** The area integrals first multiplied a whole-cell Gauss quadrature by the cell's area fraction. That is only right for an integrand constant across the cell, and the disk body force is not: it lives entirely inside agent $N$'s notch, so $N$ was credited with a share of a force belonging to $R_1$ and the total work came out **48% too large**. Cells are now clipped to the region exactly — every agent here is a box minus boxes, so the clip has a closed form.

**3. A constant deficit was reported as a perfect Gaussian.** $r^2 = 1 - SS_{\text{res}}/SS_{\text{tot}}$ is $0/0$ where the deficit has no variance, and returning $1.0$ for it declared a Jensen **top hat** a perfect Gaussian at $x=14$ — far downstream the top hat is wider than the sampled window, so the deficit is constant across it. That would have let gate W8's "self-similar Gaussian far wake" pass on the exact opposite of one. **Found by reading the gate script's output, not by the tests**; the test only checked $x=7$. It now checks every station.

**4. The Betz canary fired on the viewer.** Reading the field at the rotor face and calling it $\langle U_d\rangle$ fed the disk expert its own inflow and reported $C_P = 2.0$ against a limit of $0.5926$. An engineering wake model carries no induction upstream of the rotor, so the disk velocity must be formed as $U_{\text{ref}}(1-a)$ with $U_{\text{ref}}$ the *undisturbed local* inflow — for turbine 2 that is turbine 1's wake, which is the distinction spec §4.2 exists to preserve. Both turbines now sit at exactly $16/27$, and $P_2/P_1$ is $0.593$ (Jensen) and $0.520$ (BPA), inside W10's band. **A canary that never fires is decoration; this one fired on its first real use.**

### Two limitations, asserted as tests rather than written in a comment

* **The centreline cannot distinguish Jensen from Bastankhah–Porté-Agel.** Fitting the Jensen law to a Gaussian wake's centreline gives $r^2 = 0.9993$ — both decay roughly as $x^{-2}$ — while returning $a = 0.52$, past the physical limit. Only the *lateral profile* discriminates, which is why gate W8 asks for a self-similar Gaussian rather than a centreline match.
* **Conservative and consistent mapping agree wherever coverage is complete.** They differ only under *partial* coverage, where a destination segment reaches past the end of the source. So the fifteen interface residuals are insensitive to the choice — the distinction becomes live at a three-agent corner, and $(0,\pm2)$, $(3,\pm2)$, $(7,\pm2)$ are exactly that (spec §13 open question 3).

**The BPA corridor is NaN near the rotor**, not wide. It is a far-wake model whose amplitude carries a square root that goes imaginary close in, and `valid_from` moves with $k^*$ — so the "corridor" inverts there and a correct wake can fall outside it. Reporting no corridor is honest; reporting a wrong one is not. The report carries the fraction of $x$ actually graded, so "inside the corridor" can never be claimed over a range where there was none.

### The viewer

`python gui/windfarm.py --port 8001 --open`, or `run_windfarm_gui.bat`. Four tabs: the plan view at true scale with the partition, all fifteen interface curves, normals and token lattices over the field; wake corridors and per-station Gaussian fits; the fifteen port residuals with the $\mathcal R(t)$ breakdown; and the recorded W0–W4 result JSONs.

**Its one design rule is that it reimplements no physics** — every number comes from `cases.windfarm.*`, and `test_w4_gui.py` checks the endpoints against direct calls into those modules rather than against hand-written expectations, so a viewer that grew its own copy of the equations would fail rather than drift. It needs no data and no checkpoint.

Worth having for two reasons beyond looking at pictures. The 2026-08-19 geometry error was found by *drawing the partition*, and this draws it from the same module the assertions run against. And fifteen ports produce fifteen numbers; colouring the interfaces by residual turns "something is leaking" into "that edge is".

### Gate W4: pass

Metrics exist and recover known answers before any coupled system can produce numbers nobody can check — which is the whole point of the guide's ordering, and it has paid for itself four times before W5 has started.

---

## [2026-08-21] W5 | built, one gate item blocked — the eight-agent system runs; the thrust projection does not

**Code:** `src/atlas/cases/windfarm/couple.py`, `scripts/windfarm_w5_couple.py`,
`tests/atlas/windfarm/test_w5_couple.py`. Batched expert stepping added to `adapters.py`.

### The scaffold was not reusable, and that is the F3 finding

[[impl-wind-farm-guide]] §6.1 says to reuse the Phase-1 scaffold verbatim and swap the identity experts for real ones. **That is not possible.** `atlas.geometry.domains` builds every agent grid from `atlas.solvers.grid`, whose blocks come from the rocket's nozzle contours, and the agent set, edges and conditioning all come from `config/atlas_0_1.yaml`, which describes the rocket. It is a *rocket* scaffold, not a case-study-agnostic one.

**The layer that did transfer was the port algebra.** `atlas.ports` was reused without a single change. That is the concrete answer to falsification criterion F3 so far: the reusable substrate is the *interface contract*, not the model plumbing.

### An agent is no longer atomic

The expert is a fixed $128\times128$ operator on a square window; the agents are $6\times8$, $18\times2$, $0.1\times1$. One window per agent is impossible — a non-square agent mapped to a square window is anisotropically stretched, which is not a symmetry of Navier–Stokes, and sizing the window to the largest agent puts $\mathrm{Re}_{\text{eff}} = T_s/(\nu_p L^2)$ through the floor. So every fluid agent is **tiled**: 124 windows over six fluid agents.

The consequence is worth naming. **The tiles inside one agent exchange through the same band mechanism the declared interfaces use, but those seams are not ports.** They are not in the spec's edge table and no residual may be read as if they were. *The frozen expert's input shape has imposed a decomposition finer than the one Atlas declares* — and nothing in the framework describes it.

Batching matters: the checkpoint costs $0.117$ s for one window and $0.022$ s per window at batch 32, so the sweep is 2.7 s rather than 14.5 s.

### Gate W5

| item | result |
|---|---|
| every port materializes; pairs = declared | **pass** — 15 declared, 15 materialized, 2 undeclared-by-design |
| fixed point $\le 6$ iterations | **pass** — 3–4 with IQN-ILS |
| mass residual $<10^{-8}$ or documented | **documented** — see below |
| post-projection $r_T<10^{-8}$ | **BLOCKED** — architectural, see below |
| pre-projection $r_T$ recorded per step | **pass**, but **not trustworthy** — see the null control volumes below; the same pressure that blocks enforcement contaminates the measurement |
| flow stays physical (added) | **pass** with the projection off |

### The blocked item, and why it is architectural rather than a bug

Spec §7.3 derives the min-norm thrust projection **in stream-function space**, where the correction preserves $\nabla\!\cdot\!\mathbf u=0$ identically because $\nabla\!\cdot\!\nabla^\perp\equiv0$. W0 established this checkpoint has no $\psi$ head, so the case study is on the level-4 path.

Half of the construction survives that. The correction *direction* can be made divergence-free by Leray-projecting it, and the scalar quadratic solves exactly: the measured residual goes to $10^{-15}$ in one closed-form step. **The half that does not survive is the constraint value.** $g$ contains a pressure flux, the pressure is recovered from the velocity, and that recovery assumes a *periodic* window. W3's probe was a periodic window, so there it was sound. A rotor window inside the coupled domain is not, and with a body force present the recovered pressure is wrong by enough that $g$ reads **100–450% of $T$** while the identical operator on a turbine-free volume reads **3%**.

Correcting to that target reversed the flow through the disk within a few steps. With the projection off, the same system runs stably.

**Two null control volumes settle where the fault is.** The same operator, the same normalization, on volumes with **no body force in them** — one in the quiet inflow, one $2.5\,D$ downstream of rotor 1 in strongly sheared unsteady wake:

| control volume | residual, as a fraction of $T$ |
|---|---|
| null, quiet inflow | $0.5$–$2.2\%$ |
| null, **wake** — disturbed flow, no force | $0.2$–$5\%$ |
| **rotor** | $\mathbf{65}$–$\mathbf{72\%}$ — 19–300$\times$ the wake floor |

The operator survives a hard flow. What it does not survive is a volume containing a **net body force**, which is exactly the case a periodic pressure recovery cannot represent — the same ill-posedness W3 named and solved with a counter-force, reappearing where a counter-force is not available because the domain has a real inlet.

So **both the enforcement and the measurement of C2 are blocked by the same missing pressure**, and the honest conclusion is not "the frozen expert has a 70% momentum error". The expert conserves momentum to a few percent everywhere the operator can be trusted, including in the wake; the rotor control volume is the one place the instrument does not work. **So conservation law C2 drops from "enforced, level 5" to "measured" in the coupled configuration**, and the reason is traceable to the decoder mode recorded at W0. The enforcement ledger of spec §7.6 needs that row changed: on the velocity-decoder path, *four* enforced ports become *zero*, and all fifteen are measured.

Fixing it needs a Neumann Poisson solve on the window — not a different projection. Recorded rather than attempted, per the guide's rule that a failed gate stops the phase and an architectural finding gets said out loud.

### Four bugs found on the way, each now a test

**1. The fixed point was solving a different problem than the step.** It iterated the two rotor tiles alone, then `step()` took a separate full sweep — so the converged $T$ was not consistent with the $\langle U_d\rangle$ the sweep produced: $T=0.624$ against $\langle U_d\rangle = 0.598$, which the disk says should give $0.357$. Since $C_T'=2$ makes $T$ and $\langle U_d\rangle^2$ the same number, that gap is checkable and is now asserted every step. The fixed point iterates the full sweep; its converged iterate *is* the macro-step, so no extra sweep is needed.

**2. The Galilean frame must be spatially constant.** `step_many` gave each window its own mean. For one isolated window that is exactly right — and invisible. Assembling overlapping tiles each shifted by a *different* amount is not a Galilean transformation at all, and it tears the seams: the turbine-free control-volume residual went from $4.8\times10^{-4}$ at $t=0.25$ to $0.37$ by $t=5$. One frame for the whole domain fixed it, 7$\times$ better.

**3. A periodic pressure cannot carry a mean gradient — again.** W3 found this and solved it with a counter-force. Here it returned as per-tile Leray projection: the projection leaves $k=0$ alone, so a tile holding a rotor loses mean momentum every step with nothing to balance it. In a real incompressible flow the disk does **not** slow the fluid inside the strip — it imposes a pressure jump, and the pressure decelerates the flow over the whole streamtube, most of it upstream. Producing that needs a pressure solve over a region big enough to hold the streamtube; a $2\,D$ tile is not, the domain is. The projection is now done once on the assembled field, which is also what C1 asks for — conservation "over every agent *and every union of agents*" is not something a per-tile projection can deliver however exact it is on each tile.

**4. Two more Courant conditions, both the same rule as W3's band.** The strip is $\Delta_d=0.1$ thick but the flow crosses $U_\infty\Delta t = 0.25$ in one step, so the impulse removes more momentum than the fluid carries. And a control volume clipped by an off-centre tile edge silently dropped **27%** of the applied force. Both are now guards: `disk_thickness >= U_inf * dt` raises, and the measurement window is centred on the rotor rather than borrowed from the nearest tile. **[AI Inference]:** the pattern across W3 and W5 is that *every* feature of the model thinner than $U_\infty\Delta t$ — interface band, actuator strip, control-volume margin — is overrun within one coupling step. That looks like a general design rule for coupling a frozen operator with a fixed native $\Delta t$, and it is worth stating as one rather than rediscovering it a fourth time.

A fifth, smaller: a binary strip mask rounds a 6.4-cell strip to 6 or 7 and loses 6% of the thrust, which reads as an interface residual and gets blamed on the expert. `disk.body_force_field` — the W2-tested routine that weights by exact rectangle overlap — is now used instead of a second, worse implementation.

### The mass residual, split

Inside a window the Leray projection is spectral and exact: intra-tile divergence is $10^{-13}$. The **assembled** field is a different object, and after the global projection it is the one that is controlled. The spec's $10^{-8}$ belongs to the stream-function path and is not available here; this is the documented alternative the W0 fallback allows, and reporting one number for both would hide which is which.

### The rollout to $t=60$, and what it says about W7, W8 and W10

240 macro-steps, 64 min, all finite. Fixed point max 5 iterations, mean 3.02, converged at every step. $C_P$ exactly $16/27$ at both rotors — the Betz canary sits at the limit, as it must at $a=1/3$, and never above it. Minimum $\langle U_d\rangle$ over the whole run $0.643$.

**$P_2/P_1 = 1.16$ over $t\ge20$, and the reason is not the wake.** Turbine 2 out-producing turbine 1 is backwards for an aligned pair and far outside W10's $0.4$–$0.8$ band, and the obvious explanation — the wake never arrives — is **wrong**. The centreline at $t=15$:

| $x$ | $-3$ | $-1$ | $0.5$ | $2$ | $4$ | $6.9$ | $9$ | $12$ | $16$ |
|---|---|---|---|---|---|---|---|---|---|
| deficit | $13.4\%$ | $26.0\%$ | $41.1\%$ | $27.1\%$ | $20.7\%$ | $28.4\%$ | $16.1\%$ | $6.5\%$ | $3.4\%$ |

There *is* a wake, it recovers from $41\%$ at the disk to $21\%$ by $x=4$, and it is still $28\%$ deep just upstream of turbine 2. What is wrong is the other end: **the upstream induction is far too strong** — $26\%$ deficit one diameter upstream of turbine 1 and still $13\%$ three diameters upstream, against roughly $10\%$ and $2\%$ for a real disk. Turbine 1's own blockage therefore slows its inflow *more* than turbine 2's wake slows turbine 2's, and the ratio inverts.

That traces back to the same place as the blocked projection. The global Leray projection treats the domain as **periodic in $x$**, so the disks' pressure jump propagates around the domain instead of being absorbed by an inflow and an outflow. It is a better treatment than the per-tile projection it replaced — the system is stable and a physical wake exists — but it is still not an inflow/outflow pressure solve. **One diagnosis covers the blocked projection, the excessive induction and the inverted array efficiency, and one fix addresses all three: a pressure solve with real boundary conditions.**

**A positive result, and it is the one the case study exists to test.** A fine scan across the declared $y=\pm2$ shear interfaces shows no jump: at $x=2$ the profile runs $0.899$, $0.916$, $1.001$, $1.044$ through $y=1.95,\,2.00,\,2.05,\,2.10$ — a smooth shear layer with the bypass acceleration on the outside. **The declared-edge contract transmits.** An earlier reading of a deep deficit exactly at $y=2$ looked like an interface artefact and was not one; it was the wake being wider at that instant, and the finer scan is what distinguished them.

**W7 will fail as things stand.** The same scan gives a mirror asymmetry of about $7\%$ at $t=5$ ($u=0.842$ at $y=+1.75$ against $0.905$ at $y=-1.75$) where the gate asks for $<10^{-6}$ under symmetric inflow. That is consistent with the sweep-order sensitivity below, and it is what [[impl-wind-farm-guide]] §10 predicts for path-dependent message passing around the graph cycle.

### Scaffold action item 4, answered

The open question was whether the expert has to interleave with the message-passing layers. In a coupled solver that is the sweep order: **Jacobi** (every agent steps from the same old state, then all exchange) against **Gauss-Seidel** (each agent steps from a state its upstream neighbours have already updated). Both run stably and converge in the same 3–4 iterations. At $t=10$ they differ by $3.1\%$ (turbine 1) and $4.0\%$ (turbine 2) in $\langle U_d\rangle$, with Gauss-Seidel consistently the less dissipative of the two.

**So ordering matters, at the few-percent level — neither negligible nor fatal.** It is the same magnitude as the symmetry breaking, which is what one would expect if both come from the same source: information reaching an agent through different path lengths around the cycle.

### The gate had to be strengthened before it could fail

The first version reported **PASS on a rollout whose flow ran backwards through the rotor** — every item it checked was satisfied while $\langle U_d\rangle = -0.22$. `flow_stays_physical` and the Betz canary are now gate items. A gate that cannot fail on a reversed flow is not grading anything, which is the same lesson W3 recorded about $r_T$ and rigid advection.

---

## [2026-08-22] W5 | fixed, measured — the pressure solve; C1 met at machine precision, C2 still blocked and for a new reason

**Code:** `src/atlas/cases/windfarm/pressure.py` (new), `couple.py`, `scripts/windfarm_w5_couple.py` (`--pressure`), `tests/atlas/windfarm/test_w5_pressure.py` (new, 21 tests).

The 2026-08-21 entry ended with one diagnosis covering three symptoms — the blocked thrust projection, the excessive upstream induction, and the inverted array efficiency — and one fix proposed for all three: *a pressure solve with real boundary conditions*. That fix is now built. **It resolved one of the three, and the measurement says the other two have a different cause.** Recorded that way round because the prediction was made here and it was wrong.

### What was actually periodic

Two operators, both spectral, both correct on the window W3 validated them on, and neither correct where W5 used them.

`global_project` projected the assembled field with an FFT, so the domain wrapped in $x$: what left the outlet arrived at the inlet. `pressure_from_velocity` recovered $p$ on a rotor window by the same route. Against an **analytic** Taylor–Green pressure the second can be graded exactly:

| window | $n=32$ | $64$ | $128$ | $256$ |
|---|---|---|---|---|
| periodic $[0,2\pi)^2$ — a true period | — | — | $5\times10^{-16}$ | — |
| **non-periodic sub-window** $x\in[0.3,1.9]$, $y\in[-0.7,0.9]$ | $0.557$ | $0.590$ | $0.641$ | $0.704$ |
| the same window, Neumann solve | $1.9\times10^{-3}$ | $4.6\times10^{-4}$ | $1.2\times10^{-4}$ | $2.9\times10^{-5}$ |

**The error grows under refinement.** An inaccurate operator converges; an inconsistent one does not, and that is the difference between a discretization error and the wrong equation. A $56$–$70\%$ error on the control-volume faces is exactly the size needed to make $g$ read $100$–$450\%$ of $T$ at a rotor while reading $3\%$ on a turbine-free volume.

### The replacement, and why it is direct rather than iterative

Two Poisson solves, each diagonalized **exactly** by a discrete cosine transform, so the cost stays $O(N\log N)$ — measured $0.08$ s on the $512\times1536$ domain against $2.7$ s for one 124-window expert sweep. The guide budgets "a Poisson solve per agent per step" as the expensive fallback; at $3\%$ of a sweep it is not.

* **Mass.** Neumann at the pinned inlet and the two slip walls, **Dirichlet at the open outlet**; DCT-IV in $x$, DCT-II in $y$. The Dirichlet face is the whole point — it is the one boundary free to absorb the pressure the rotors displace, and its eigenvalues $-\sin^2(\pi(k+\tfrac12)/n)/h^2$ never vanish, so the solve needs no compatibility condition and no null-space handling.
* **Momentum.** Neumann on all four faces of a control volume, with the data taken from the normal momentum equation itself,
  $$\frac{\partial p}{\partial n} = \mathbf n\cdot\Bigl[\mathbf f - \frac{\partial\mathbf u}{\partial t} - (\mathbf u\cdot\nabla)\mathbf u + \nu\nabla^2\mathbf u\Bigr].$$
  Singular by construction, and **the compatibility defect is returned rather than absorbed** — it is the error bar on every residual computed from it, and it is now recorded every step.

**One correction banked inside the fix.** The first version projected with the compact five-point Laplacian and measured with the collocated centred difference. Those are different operators: on a collocated grid $D\!\cdot\!G$ is a Laplacian on a $2h$ stencil, not the compact one. The divergence fell only from $0.64$ to $0.40$ in $L_2$ while the face field being thrown away sat at $10^{-12}$ — *the exactly divergence-free object was not the one being stored.* Rebuilding the operator as exactly $D\!\cdot\!G$ fixed it. **A projection is only a projection if the operator inverted is the operator measured.** The near-null checkerboard modes of the wide stencil are harmless, and it is worth saying why because they look alarming: $I - G(DG)^{-1}D$ has norm 1, since $D$ annihilates precisely the modes $(DG)^{-1}$ amplifies.

### What the fix delivered

24 macro-steps, Jacobi, $\Delta t=0.25$, identical in every other respect:

| | periodic (2026-08-21) | **outflow** |
|---|---|---|
| assembled mass divergence, $L_2$ | $3.23\times10^{-2}$ | $\mathbf{5.57\times10^{-14}}$ |
| assembled mass divergence, $L_\infty$ | $2.54$ | $5.72\times10^{-13}$ |
| $r_T$ pre, R1 / R2 | $0.725$ / $0.706$ | $\mathbf{0.140}$ / $\mathbf{0.047}$ |
| null CV, inflow / wake | $7.3\times10^{-3}$ / $2.2\times10^{-2}$ | $1.9\times10^{-2}$ / $5.6\times10^{-2}$ |
| **rotor residual / null-wake floor** | $\mathbf{32\times}$ | $\mathbf{2.5\times}$ |
| pressure error bar | not available | $3.6\%$ / $6.7\%$ of $T$ |
| $P_2/P_1$ | $1.140$ | $1.138$ |
| centreline deficit at $x=-1$ | $17.8\%$ | $17.3\%$ |

**Two results worth keeping separate.**

1. **C1 is no longer a documented alternative — it is met.** Spec §7.2 says level-5 mass conservation is unavailable without a stream-function decoder, and the 2026-08-21 entry reported $L_2\le6.6\times10^{-2}$ as the fallback the W0 path allows. The assembled field is now divergence free to $5.6\times10^{-14}$ **over the whole domain including the boundary cells**, in the operator that telescopes over any union of cells to that union's net boundary flux — which is exactly what C1 asks for, and which the old interior-only diagnostic (it trimmed four cells from each edge) could not have detected a boundary error in at all. The spec's $10^{-8}$ is met on the velocity path, by a route the spec did not anticipate.
2. **The instrument works at the rotor now.** C2 was blocked because the rotor control volume read $19$–$300\times$ the turbine-free floor, so no residual measured there could be attributed to anything. It now reads $2.5\times$ the floor and carries its own error bar — and R2's $r_T=4.7\times10^{-2}$ sits essentially *at* its $6.7\times10^{-2}$ pressure defect, i.e. at the instrument's resolution rather than above it.

### The prediction that failed, and the one that replaced it

$P_2/P_1$ moved from $1.140$ to $1.138$. Upstream induction moved from $17.8\%$ to $17.3\%$ at one diameter. **The periodic pressure was not what caused either.**

**[AI Inference]:** the likeliest remaining cause is that this is a *2D* case study and the theory it was called wrong against is 3D. A 3D actuator disk at $a=1/3$ induces about $0.106\,a$ at one diameter upstream, i.e. $\sim3.5\%$, and the "roughly $10\%$ and $2\%$" of the last entry is that picture. In 2D the rotor is an infinite-span strip, blockage in an $8\,D$ channel with slip walls is $12.5\%$, and upstream influence decays far more slowly. So $17\%$ at one diameter may be substantially *correct 2D physics* rather than an artefact — which is the sort of thing [[case-study-wind-farm-wake-2d-atlas-0.1]]'s own 2D discount exists to warn about. **This is checkable and should be checked before anything else is blamed:** the pseudo-spectral reference solver built at W3 already exists, takes the same actuator disk, and running it on this geometry answers whether $17\%$ is simply the 2D answer. If it is, then W10's $0.4$–$0.8$ band was set from 3D intuition and it is the band that needs revising, not the coupling.

### Fixing the pressure exposed a second, independent block on C2

With the pressure correct, the thrust projection was switched on. It destroyed the flow — $\langle U_d\rangle$ at $770$ times freestream on the first step and $-4879$ by step 24 — **while reporting post-projection $r_T = 7.5\times10^{-11}$, four orders of magnitude inside the gate.**

The cause is neither the pressure nor the target. It is the correction *direction*, and it is degenerate. The direction

$$\mathbf w = \mathbb P_{\text{Leray}}\bigl[\mathbb 1[x_l\le x\le x_r]\,\hat{\mathbf x}\bigr]$$

varies only in $x$ and points along $x$, so **every** Fourier mode of it is pure gradient and the projection removes all of them; what survives is $k=0$ alone. The "min-norm divergence-free direction" of spec §7.3 is a **uniform field** — measured, $\max|w_u|$ equals its own mean to every digit. A uniform velocity added to an incompressible flow is a Galilean shift, and with the pressure now recovered from correct Neumann data (which contains $-\partial_t\mathbf u$) the momentum balance is invariant under one: the pressure flux moves by $-6.19\times10^{-2}\lambda$ and $dM/dt$ by $+6.19\times10^{-2}\lambda$, cancelling to five figures. Measured sensitivity $dg/d\lambda = 1.02\times10^{-6}$ against a residual of $1.01\times10^{-1}$, hence $\lambda = 9.85\times10^{4}$.

**And the reason this was invisible before is the very defect that was just fixed.** A periodic pressure cannot carry a mean gradient, so it could not perform that cancellation, and $\beta$ was spuriously nonzero. *The same missing degree of freedom made the target wrong and made the direction appear viable.* Two bugs that concealed each other.

A guard now refuses any correction that would move the field by more than half its own RMS, records the sensitivity, and marks the gate item BLOCKED rather than passing. With it, `project=True` runs stably ($u_{\max}=1.4$ over six steps) and reports C2 honestly as unenforced. **This is the third appearance of one lesson**: W3's $r_T$ could not reject rigid advection, W5's first gate reported PASS on a reversed flow, and now a projection graded its own arithmetic while wrecking the field. A gate that measures whether a solve converged is not measuring the physics the solve was for.

The fix for C2 is a correction direction localized near the disk rather than uniform over the window. Recorded, not attempted, per the guide's rule that a failed gate stops the phase.

### Gate W5, restated

| item | before | now |
|---|---|---|
| ports materialize | pass | pass |
| fixed point $\le6$ | pass (3–4) | pass (3) |
| mass residual | documented, $L_2\le6.6\times10^{-2}$ | **pass at $5.6\times10^{-14}$** — the spec's $10^{-8}$, on the velocity path |
| post-projection $r_T<10^{-8}$ | BLOCKED (pressure wrong) | **BLOCKED (direction degenerate)** — a different cause, still blocked |
| pre-projection $r_T$ recorded | pass, not trustworthy | **pass, and now readable** — $2.5\times$ its floor, with an error bar |
| rollout reaches $t=60$ | pass | pass |
| flow stays physical | pass | pass |

**Still PARTIAL, and the remaining block is a different one than it was.** That is progress, and it should not be rounded up to a pass.

---

## [2026-08-22] W5/W9 | measured — the monolithic baseline built; 17% upstream induction is **not** 2D, and the AI inference that said it might be is retracted

**Code:** `reference.ChannelNS` (new), `analytic.induction_2d_strip` / `induction_3d_disk` (new), `scripts/windfarm_induction_check.py` (new). Results in `results/windfarm/w5fix/`.

The entry above proposed, as an **[AI Inference]**, that W5's excessive upstream induction might be correct 2D physics graded against 3D theory, and said it was cheaply checkable. It was checked. **It is wrong, and the retraction is the result.**

### The three answers, and how they were obtained

Two are exact potential-flow results and cost nothing. The third is a real solver on the real configuration.

* **3-D actuator disk**, the vortex-cylinder law $1-u/U = a[1 + (x/R)/\sqrt{1+(x/R)^2}]$: $3.5\%$ at one diameter upstream. This is what W5 originally graded itself against.
* **2-D actuator strip**, unbounded: the 2D wake is bounded by two semi-infinite trailing vortex sheets rather than a closed cylinder, giving $1-u/U = (2a/\pi)[\pi/2+\arctan(x/h)]$, $h=D/2$ — $9.8\%$ at one diameter. **The inference was right that 2D is nearly $3\times$ stronger than 3D**, and that much of it survives.
* **2-D Navier–Stokes on the wind-farm domain itself**: $\mathbf{5.7\%}$.

The last is `ChannelNS`, and it exists because the 2026-08-22 pressure fix made it possible — it reuses `project_outflow` for exactly the boundary conditions the coupled system now has. Same actuator disk, same domain, same pinned inlet and slip walls and open outlet, same local-induction thrust law. **The only thing that differs from the coupled run is that the frozen expert is replaced by a solver**, which is what makes it [[impl-wind-farm-guide]] §7.3's W9 baseline as well as this measurement. The W3 entry predicted it would come cheap once a reference solver existed; it did.

**Converged, and insensitive to everything it should be insensitive to.** Upstream induction is a potential-flow effect, so a value that moved with viscosity or resolution would not be an answer:

| run | $\nu$ | $dx$ | strip | deficit at $x=-1$ | $P_2/P_1$ |
|---|---|---|---|---|---|
| coarse | $0.03$ | $0.0625$ | $0.25$ | $5.7\%$ | $0.403$ |
| half $\nu$ | $0.015$ | $0.0625$ | $0.25$ | $5.9\%$ | $0.246$ |
| fine grid | $0.03$ | $0.03125$ | $0.25$ | $5.6\%$ | $0.421$ |
| thin strip | $0.03$ | $0.0625$ | $0.10$ | $5.7\%$ | $0.429$ |

Halving the viscosity, halving the cell, and thinning the strip by $2.5\times$ each move it by at most $0.2$ points, at a steady-state drift of $3\times10^{-5}$. **$5.7\%$ is the answer.**

### The verdict

| $x$ | 3-D theory | 2-D theory | **2-D NS, this domain** | **W5 coupled** |
|---|---|---|---|---|
| $-3$ | $0.5\%$ | $3.5\%$ | $\mathbf{0.7\%}$ | $\mathbf{14.1\%}$ |
| $-1$ | $3.5\%$ | $9.8\%$ | $\mathbf{5.7\%}$ | $\mathbf{25.9\%}$ |

**The coupled system's upstream induction is $4.5\times$ the monolithic baseline's on an identical configuration.** It is a defect, not 2D physics, and the inference above is retracted. That it lands between the 3D and unbounded-2D laws is itself a sanity check on the solver: confinement raises it above nothing, and averaging over the rotor span rather than the exact centreline lowers it below the unbounded peak.

**And it is not even a settled defect — it is drifting.** Tracked over the coupled rollout at $x=-1$: $9.3\%$ at $t=2$, $13.9$, $17.3$, $22.1$, $25.0$, $25.9\%$ at $t=20$. It passes *through* the correct answer at about $t=1$ and keeps going. A steady blockage should equilibrate within one or two convective times, as the baseline does. **Something is accumulating**, which is a sharper and more findable statement than "the number is too big".

### The larger finding the same table produced

Reading the downstream columns was not the point of the exercise and is the more important result:

| $x$ | **2-D NS baseline** | **W5 coupled** |
|---|---|---|
| $+2$ | $44.2\%$ | $26.1\%$ |
| $+9$ | $49.8\%$ | $13.6\%$ |
| $+16$ | $\mathbf{28.3\%}$ | $\mathbf{1.6\%}$ |

**In genuine 2D the wake barely recovers at all** — $28\%$ still standing sixteen diameters downstream — because 2D has no three-dimensional entrainment to refill it and the channel is closed at the sides. The coupled system has $1.6\%$. The frozen expert is dissipating the wake roughly twenty times too fast, which is the per-call dissipation W0 measured (§5.2b: half-life $1.75$ convective times) arriving in the coupled system exactly as W0 warned it would.

**So the inverted array efficiency has two causes, not one, and they push the same way.** Turbine 1's inflow is over-throttled by $4.5\times$ too much upstream induction; turbine 2's inflow is under-throttled because the wake that should reach it has been dissipated. Both raise $P_2/P_1$, which is why it inverts.

### W10's band is vindicated, which was also in doubt

The entry above suggested W10's $0.4$–$0.8$ array-efficiency band might have been set from 3D intuition and need revising. **It does not.** The 2D monolithic baseline gives $P_2/P_1 = 0.403$ (coarse), $0.421$ (fine), $0.429$ (thin strip) — inside the band, at its lower edge, which is where a 2D case study with a poorly-recovering wake should sit. The band was right; the coupled system's $1.14$ is genuinely wrong.

**Two AI inferences from the previous entry, both now settled and both by the same measurement: one retracted, one refuted.** Recorded at this length because the entry above stated them as the cheap next check, and a proposed check that is run and reported is worth more than one that is only proposed.

### What this hands W6–W11

* **The W9 monolithic baseline exists** and is validated against two closed forms. §7.3 moves from "do not skip" to "already have", which is what the W3 entry's inference predicted.
* **W8 has its target**: the wake to reproduce is the baseline's, and in 2D that wake does not recover. Any Jensen or Bastankhah–Porté-Agel corridor fitted here must be checked against $28\%$ at $x=16$, not against engineering-model intuition from 3D field data.
* **The two defects are separable** and should be attacked separately: the induction drift is in the coupling, the wake dissipation is in the checkpoint and was quantified at W0.

---

## [2026-08-23] W6–W11 | built, measured — the metrics finally reach the system; W9 attributes the damage to the checkpoint, not to the decomposition

**Code:** `couple.CoupledField` / `energy_per_agent` / `CoupleConfig.monolithic`, `metrics.report(d_energy_dt=...)`, `scripts/windfarm_w6_experiments.py`, `scripts/windfarm_w9_compare.py`, `tests/atlas/windfarm/test_w6_bridge.py` (8 tests). Results in `results/windfarm/w6/`, `results/windfarm/w9/`.

### The gates were blocked on an adapter, not on physics

W4 built every metric — fifteen per-port residuals, the mirror symmetry residual, the Jensen/BPA corridors, the global power residual $\mathcal R(t)$ — and validated each against a field with a known answer, exactly as [[impl-wind-farm-guide]] §5 demands. Then it stopped, because **the metrics take an analytic `Field` with a `velocity(x,y)` method and the coupled system produces a lattice.** There was no way to point one at the other. Gates W7, W8, W10 and W11 had been unreachable since 2026-08-21 for that reason and no other.

`CoupledField` is thirty lines of bilinear sampling plus a pressure and a body force. **One adapter unblocked four gates**, which is worth recording as a planning lesson: the guide sequenced the metrics before the system to avoid unverifiable numbers, and the cost of that ordering is that the join between them is nobody's deliverable and gets skipped.

### The adapter needed the same discipline the metrics got, and it caught two bugs

Both would have reached a gate as a physics claim.

1. **Cell averages are not point values.** `body_force` returned a bilinear interpolation of the lattice force field. That field is exact in its own sense — each cell holds the strip's area fraction, so the discrete sum is exactly $-T$ on any lattice, which is what gate W2 tests — but `metrics._integrate_rect` *point-samples* at Gauss nodes, and a strip $6.4$ cells wide is where that bites. W4's validated identity (the balance with $P_{\text{ext}}$ omitted equals the extracted power) read $0.2708$ against an exact $0.2963$: **an $8.6\%$ error, all of it in the work term.** The W4 entry records the same mistake in a different costume — *"fractional occupancy is not a quadrature weight"*, a $48\%$ work error. Both times the fix is the closed form: the strip is a rectangle and the force is piecewise constant, so a point evaluation is exact. The identity now holds to $10^{-12}$.

2. **`d_energy_dt` defaults to zero and the system is not steady.** `power_ledger`'s docstring says so plainly — zero is exact for the steady analytic fields W4 validated against and *must be supplied by the caller for anything else*. Omitting it inflated $|\mathcal R|$ to $88\%$ of extracted power at $t=5$, essentially all of it the missing term. It is now computed per agent on the identical quadrature, and **whether it was supplied is recorded in the result** rather than left for a reader to infer.

**The adapter is validated, which is what makes the failures below attributable.** It reproduces W4's identity exactly; on smooth analytic fields it recovers the outflux term to $2\times10^{-4}$ and dissipation to $3\times10^{-5}$ relative; a symmetric field reads a mirror residual of exactly $0$ even when the $\pm y$ sample points miss cell centres, which matters because W7's threshold is $10^{-6}$ and nearest-neighbour sampling would manufacture an asymmetry of order half a cell.

### Gate W6 — pass, and the canary fired on me

$C_P^{\max} = 0.592593$ against Betz $= 0.592593$. Exactly the limit at $a=1/3$, never above, at every step.

The first version of the script reported $C_P = 2.0$. **The normalization is the entire canary**: $C_P$ is referenced to the *undisturbed local* inflow $U_{\text{ref}} = \langle U_d\rangle/(1-a)$ — for turbine 2 that is turbine 1's wake, not the domain inlet — and dividing by $\langle U_d\rangle$ itself feeds the disk its own induction. That is precisely the error the W4 entry records the canary catching on the viewer, reproduced here in new code four days later. **A canary that fires twice on the same mistake in different code is doing its job better than a test would.**

### Gate W9 — the result, and it exonerates the decomposition

Three runs, differing in exactly one thing each. `A` is the eight-agent system. `B` is the **same 124 tiles at the same positions with the agents taken away** — ownership follows no declared interface and no port materializes. `C` is `ChannelNS`, a real 2D Navier–Stokes solver on the same domain.

| pair | isolates | rel-$L_2$ |
|---|---|---|
| $A-B$ | **composition error** — what declaring agents and exchanging ports costs | $\mathbf{0.273}$ |
| $B-C$ | **expert error** — what the frozen checkpoint costs | $\mathbf{1.762}$ |
| $A-C$ | total | $1.618$ |

**Composition error is $6.5\times$ smaller than expert error.** The decomposition into declared agents is not what is damaging this simulation; the checkpoint is. That is the single sharpest statement this case study has produced, and §7.3 is right that it cannot be made without the baseline.

Read at the rotors, it is sharper still:

| run | $\langle U_d\rangle$ R1 | R2 | $P_2/P_1$ |
|---|---|---|---|
| eight agents | $0.6502$ | $0.6798$ | $1.143$ |
| **monolithic (no agents)** | $0.6521$ | $0.6969$ | $\mathbf{1.220}$ |
| classical 2D NS | $0.7894$ | $0.5837$ | $\mathbf{0.404}$ |

**The monolithic run inverts too — slightly worse than the partitioned one.** So the inverted array efficiency that has been the case study's most visible defect since 2026-08-21 is **not caused by Atlas's decomposition at all.** Remove every agent and every port and it is still there. Only replacing the frozen expert with a solver fixes it.

**Gate W9 fails as stated**, because $0.273$ exceeds the expert's own single-step error of $5.5\times10^{-3}$ by $50\times$ — decomposing the domain costs about fifty times what one call of the expert costs. That is a real negative result about the composition layer and it should not be rounded away by the favourable ratio above. Both statements are true: composition error is large against the *gate's* yardstick and small against the *expert's* contribution to the total.

### The other gates

| gate | result | number |
|---|---|---|
| W6 Betz canary | **pass** | $C_P^{\max} = 0.592593$, exactly $16/27$ |
| W7 symmetry | **fail** | mirror residual $2.77\times10^{-2}$ at $t=5$, $7.42\times10^{-2}$ at $t=10$, against $10^{-6}$ — and **growing**. Cause reattributed to operator non-equivariance, OP-6 |
| W8 wake recovery | **fail** | far-wake Gaussian fit and BPA corridor both missed |
| **W9 decomposed vs monolithic** | **fail** | composition $0.273$ vs $5.5\times10^{-3}$; but $6.5\times$ below expert error |
| W10 array efficiency | **fail** | $P_2/P_1 = 1.197$ against $[0.4,0.8]$ — **and the monolithic run gives $1.220$** |
| W11 power residual | **not measurable** | see below — this one is the gate's fault, not the system's |

W7's failure is what [[impl-wind-farm-guide]] §10's triage predicts for path-dependent message passing around a graph cycle, and it grows with time rather than settling — the same signature as OP-2's induction drift, and plausibly the same cause.

### Gate W11 is not measurable with this checkpoint — OP-4

$|\mathcal R| = 1.61$ against $1\%$ of an extracted power of $0.669$: off by $240\times$, not by a few percent. **The instrument was checked before the claim was made**, and it is sound. So the failure is in the field or in the gate.

It is in the gate. The ledger's dissipation term is $\nu\int|\nabla\mathbf u|^2$, which is dominated by the smallest resolved scales. Progressively smoothing the coupled field and recomputing:

| smoothing radius | $0$ | $1$ cell | $2$ | $4$ | $8$ ($0.125\,D$) |
|---|---|---|---|---|---|
| dissipation, % of raw | $100$ | $74$ | $58$ | $39$ | $\mathbf{19}$ |

**$81\%$ of the dissipation lives below $0.125\,D$** — exactly the wavelength W0 identified as this checkpoint's dissipation cutoff, and exactly the band in which W0 established that a single $\nu$ is meaningless. The log's own words from 2026-08-20: *"the checkpoint behaves like an LES with a spectral cutoff, not a Newtonian fluid."* A power balance needs a dissipation term; a dissipation term needs a viscosity; **this expert does not have one.** Recorded as OP-4 in [[open-problems-atlas-0.1]] rather than reported as a conservation failure of the framework, with three candidate resolutions and an argument for the one that generalizes.

### What this session changes about the case study's overall story

Before today the honest summary was *"the framework composes frozen experts stably and conservatively, and does not yet reproduce the physics."* W9 sharpens the second half: **the physics it fails to reproduce is failed by the expert, not by the composition.** Remove Atlas entirely — same tiles, no agents, no ports — and the array efficiency is still inverted and the wake still over-dissipates. What Atlas costs on top is $0.273$ against the expert's $1.762$.

That is a considerably better position for falsification criterion F2 than the raw gate table suggests, and a considerably worse one for [[expert-library-atlas-0.1]]'s assumption that a pretrained checkpoint can be dropped in as an expert without qualification.

---

## [2026-08-23] composition layer | built, measured — PoU assembly and a solver expert; **the dominant error is the windowed architecture, and an earlier attribution is corrected**

**Code:** `couple.scatter_pou` / `taper_1d` / `tile_weight`, `CoupleConfig.blend` and `.expert_kind`, `reference.SolverExpert`, `scripts/windfarm_w9_compare.py --expert --blend`, `tests/atlas/windfarm/test_w9_harness.py` (14 tests). Results in `results/windfarm/seams.txt`, `results/windfarm/w9/`.

Two pieces of machinery, built together because neither is measurable without the other, and a result that revises the previous entry.

### Why a solver-backed expert

W9 put composition error at $0.273$ against an expert error of $1.762$. **A coupling layer cannot be developed under a $6.5\times$ noise floor** — any change worth a tenth of the composition error is a sixtieth of the total. `SolverExpert` swaps the frozen checkpoint for `SpectralNS` while holding *everything* else identical: same tiling, same window size, **same per-window periodicity**, same Galilean framing, same explicit impulse, same assembly. It preserves a uniform flow exactly, where the checkpoint takes its mean from $1$ to $0.969$.

The per-window periodicity is deliberately *not* fixed. If it were, a difference between this and the frozen run would confound "learned vs exact" with "periodic vs not" and neither could be attributed. The window's periodicity is part of the harness, not part of what is under test — and §"the finding" below is why that mattered.

### Partition-of-unity assembly

`scatter` gave each cell to its **nearest** tile and discarded what every other tile predicted there. Tiles overlap by $0.5\,D$ — 32 cells — so much of the domain has two to four independent predictions and exactly one was kept; and because the choice is a hard nearest-tile switch, adjacent cells across an ownership boundary came from *different expert calls*. `scatter_pou` blends them with a smoothstep partition of unity.

Smoothstep because $S(t) + S(1-t) = 1$ **identically**, so a pairwise overlap is a genuine partition before normalization rather than a smoothing; and because it is $C^1$, so the blend introduces no gradient kink where the taper saturates — which the power ledger's dissipation term would pick up as readily as a jump.

**Blending is within an agent only.** Blending across agents would drive every port residual to zero by erasing the distinction the framework exists to test — a metric that reads perfect because it stopped looking.

$|\Delta u|$ between horizontally adjacent cells, 8 macro-steps:

| | same tile | diff tile | p99 | excess over interior |
|---|---|---|---|---|
| frozen / nearest | $9.89\times10^{-4}$ | $2.92\times10^{-3}$ | $2.87\times10^{-2}$ | $2.95\times$ |
| **frozen / pou** | $8.79\times10^{-4}$ | $\mathbf{1.52\times10^{-3}}$ | $\mathbf{1.27\times10^{-2}}$ | $\mathbf{1.73\times}$ |
| solver / nearest | $1.42\times10^{-3}$ | $2.14\times10^{-3}$ | $2.26\times10^{-2}$ | $1.51\times$ |
| **solver / pou** | $1.48\times10^{-3}$ | $1.90\times10^{-3}$ | $1.66\times10^{-2}$ | $1.29\times$ |

$-48\%$ in the mean and $-56\%$ at the 99th percentile for the frozen expert. **It helps the checkpoint far more than the solver**, which is what one expects if the discarded predictions differ mostly by per-call noise — and is a small independent confirmation that the checkpoint's per-call noise is real.

### A null control corrected the premise this was built on

The measurement that motivated the work showed agent-boundary cell pairs at $4.07\times$ the interior, which read as interface error. **The monolithic field — no agents at all — shows $4.31\times$ at the identical sites.** Those boundaries sit on the wake/bypass shear layers and the rotor planes, where the flow genuinely has large gradients. So most of that ratio is *physics*, and only the tile-seam excess is artefact. PoU removes the artefact and correctly leaves the physics alone. **Fourth time in this case study a null control has changed a conclusion rather than confirming it.**

### The finding, and the correction it forces

Running the full $2\times2$ to $t=20$ makes the attribution clean for the first time:

| configuration | composition error $A-B$ | 8-agent $P_2/P_1$ | monolithic $P_2/P_1$ |
|---|---|---|---|
| frozen / nearest | $0.273$ | $1.143$ | $1.220$ |
| frozen / pou | $0.297$ | $1.002$ | $0.983$ |
| solver / pou | $\mathbf{0.234}$ | $\mathbf{0.889}$ | $0.907$ |
| **`ChannelNS`, undivided** | — | $\mathbf{0.404}$ | — |

Decomposing the $P_2/P_1$ error, a gap of $0.739$ from $1.143$ to the correct $0.404$:

* **assembly rule (PoU): $0.141$, or $19\%$**
* **checkpoint (frozen $\to$ exact solver): $0.113$, or $15\%$**
* **everything left: $0.485$, or $66\%$**

And "everything left" is not the expert and not the assembly. **It is the windowed architecture** — tiling the domain into 124 periodic $2\,D$ windows and stepping them at $\Delta t = 0.25$. With a *perfect* solver inside every window and no agent decomposition whatsoever, $P_2/P_1$ is $0.907$ against the undivided $0.404$: **wrong by a factor of $2.2$ with nothing learned anywhere in the loop.**

**This corrects the previous entry.** That entry read $B-C = 1.762$ as "expert error" and concluded the checkpoint dominates. $B-C$ compares a *tiled, periodic-windowed* run against an undivided solver, so it was never expert error alone — it conflated the checkpoint with the windowing, and the conflation flattered the framework by charging the whole difference to the expert. The solver run separates them, and the larger share belongs to Atlas.

The honest restatement: **the checkpoint is a real defect but a minority one; the dominant error is that a fixed-window operator forces the domain to be solved in periodic $2\,D$ patches.** W5 already recorded the mechanism — *"the frozen expert's input shape has imposed a decomposition finer than the one Atlas declares"* — and this is its price, measured.

**What is not yet separated.** The $66\%$ still mixes three things: per-window periodicity, the macro-step ($\Delta t = 0.25$ against a CFL-stable $2.0\times10^{-3}$ at the coupled system's own resolution), and the band mechanism. Ranking those is the next measurement, and it is cheap now that `SolverExpert` exists — vary one at a time with a perfect interior operator. Recorded as **OP-5**.

### What this changes about the decision that was deferred

The checkpoint-swap question was parked pending the OPs. This narrows it sharply: **swapping the expert buys at most $\sim15\%$ of the error.** A better checkpoint is worth having and is not the leverage. The leverage is in how the domain is decomposed for a fixed-window operator — larger windows, overlapping-Schwarz iteration to convergence rather than one pass, or a finer macro-step — and all three are framework work that does not need a new expert.

---

## [2026-08-23] W7 | reattributed — the frozen expert is not mirror-equivariant, and the guide's prediction was a false confirmation

**Code:** the mirror test against `FrozenFluidExpert.step_many` and `reference.SolverExpert.step_many`. **Recorded as OP-6** in [[open-problems-atlas-0.1]].

### Why this was tested at all

The plan was a rollout: run the *monolithic* (no-agent) configuration and read its mirror residual, on the reasoning that if the agents are what break symmetry, removing them restores it. Two things came first and made the rollout unnecessary.

**The geometry is exactly symmetric**, so it is excluded. The grid satisfies $\max\lvert y_c + y_c^{\text{rev}}\rvert = 0$; the tiling has $124$ tiles with $y$-centres $\{0,\pm0.7,\pm1.5,\pm2.1,\pm2.5,\pm3.5\}$ and **zero mirror-unpaired tiles** in both the eight-agent and monolithic builds; every agent box is either mirror-symmetric or paired ($B^+\!/B^-$). Cost: seconds, no rollout.

**Equivariance is a property of the operator before it is a property of the composition.** If $E(\mathcal M\mathbf u)\ne\mathcal M E(\mathbf u)$, then no arrangement of agents, tiles, ports, blending or Schwarz iteration can produce a symmetric field, and the rollout would only have measured how fast an already-doomed gate fails. That is a one-call test, not a $t=5$ one.

### The measurement

$y$-mirror: $\mathcal M[u](x,y)=u(x,-y)$, $\mathcal M[v](x,y)=-v(x,-y)$. On exactly mirror-symmetric input, one call at $\Delta t=0.25$:

| symmetric input | $\max\lvert u-\mathcal Mu\rvert$ | $\max\lvert v-\mathcal Mv\rvert$ |
|---|---|---|
| **uniform inflow $u\equiv1$** | $\mathbf{2.42\times10^{-2}}$ | $3.59\times10^{-2}$ |
| wake deficit | $3.16\times10^{-2}$ | $3.58\times10^{-2}$ |
| symmetric jet | $2.79\times10^{-2}$ | $3.46\times10^{-2}$ |
| symmetric counter-rotating $v$ | $2.58\times10^{-2}$ | $3.68\times10^{-2}$ |

The gate is $10^{-6}$. **A uniform inflow comes back asymmetric by $2.4\times10^4$ times the gate in one call.**

That field deserves emphasis on its own: it is W4's null control and it is what W0 certified the checkpoint's mean-flow handling on. **The mean is preserved to $10^{-16}$ and the symmetry is destroyed at $10^{-2}$, and neither gate looked.** A null control is only null with respect to the property it checks.

### Systematic, not per-call noise

Two calls on identical input agree to $\max\lvert u_a-u_b\rvert = 0$ **exactly** — deterministic, so this is a fixed property of the operator. On a closed window it accumulates:

| calls | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| $\max\lvert u-\mathcal Mu\rvert$ | $3.16\times10^{-2}$ | $4.69\times10^{-2}$ | $7.12\times10^{-2}$ | $1.02\times10^{-1}$ |

bracketing W7's measured $2.77\times10^{-2}$ at $t=5$ and $7.42\times10^{-2}$ at $t=10$. Monotone growth from a per-call bias is also **OP-2's exact signature**, and OP-2's proposed diagnostic — does the drift follow call count or clock — is the same experiment.

### The positive control

`SolverExpert` (`SpectralNS`) through the **identical** harness, convention and input: $1.06\times10^{-15}$ relative in $u$, $1.17\times10^{-17}$ in $v$. A sign error in my own reflection operator would have failed the solver too. This is what `SolverExpert` was built for, used for the first time as a control rather than a baseline.

### The mechanism is keyed to the attention grid, not the flow

Antisymmetric part of the output on the **featureless** uniform input, averaged over $x$, transformed in $y$ ($128$ cells):

| $k$ | 4 | 8 | 16 |
|---|---|---|---|
| wavelength | $32$ cells | $16$ cells | $8$ cells |
| amplitude | $6.00\times10^{-1}$ | $5.19\times10^{-1}$ | $1.41\times10^{-1}$ |

A clean comb on an input with no structure whatsoever. The checkpoint's `patch_size` is $4$ and `window_size` is $16$ patches $=64$ cells, giving a shifted-window stride of $8$ patches $=\mathbf{32}$ **cells** — the dominant wavelength exactly, with $16$ and $8$ as its harmonics.

**[AI Inference]:** this is SwinV2's shifted-window attention. A patch grid maps to itself under reflection but the *shifted* partition does not, because the shift is applied from one corner — so the mirrored field is cut into different windows and attends over different neighbourhoods. If that is right the defect is architectural and shared by every Swin-based neural operator, which makes it an [[expert-library-atlas-0.1]] selection criterion rather than a fact about this checkpoint. Isolating it means disabling the shift and re-running this test: cheap, not yet done.

### What this corrects, and why the correction is uncomfortable

The gate ledger read W7's failure as *"exactly what [[impl-wind-farm-guide]] §10 predicts for path-dependent message passing around the graph cycle."* The guide named the mechanism in advance, the number arrived, and the magnitude was plausible. **It is wrong.** The operator produces the same order of asymmetry with no graph at all.

Sweep-order path dependence may still be present — nothing here rules it out. What is ruled out is that it is *needed*, and it has never been shown to contribute anything.

**This is the fifth time a control has overturned a conclusion in this case study rather than confirming it, and the first time the overturned conclusion was one the guide predicted in advance.** The earlier four were mistakes of measurement; this one is a mistake of *inference from a correct prediction*. A predicted mechanism arriving at the predicted magnitude is weaker evidence than it feels like, because a wrong mechanism of the right size is indistinguishable from a right one until something is removed.

### What it changes

W7 is **unwinnable with this checkpoint as the gate is written** — not a coupling defect, and no amount of framework work touches it. But the fix is a *framework* fix to an *expert* defect, which is the most interesting shape a problem can have here:

$$\tilde E(\mathbf u) \;=\; \tfrac12\bigl[E(\mathbf u) + \mathcal M^{-1}E(\mathcal M\mathbf u)\bigr]$$

Two expert calls, nothing trained, and W7 passes **by construction** on any mirror-symmetric configuration — with the same construction generalizing to any symmetry group the problem declares. **[AI Inference]:** this is a stronger result for Atlas than a gate passing, because it is a capability the frozen expert provably does not have alone: *the composition layer restoring a symmetry the expert lacks*. It also belongs next to [[conservation-as-constraint-atlas-0.1]] — Noether ties symmetries to conserved quantities, so a broken discrete symmetry is the same class of defect as a broken conservation law, and the enforce-or-measure rule should plausibly cover declared symmetries as well as declared conserved quantities.


---

## [2026-08-23] OP-5 | qualified — the $66\%$ is a subtraction bucket, and the number it is measured against moves

**Code:** none. A review of the composition-layer entry above; the qualification is filed in full in [[open-problems-atlas-0.1]] under OP-5.

The ranking survives — windowing $>$ assembly $>$ checkpoint — and the practical conclusion is unchanged: an expert swap is not the leverage. Four things about the $66\%$ do not survive.

**The third factor was never varied at any level.** The $2\times2$ crossed {assembly} $\times$ {expert}, and **all four cells are windowed** — deliberately, so the expert swap would be attributable. So no run in the table has windowing off and the rest of the coupled path on; the only windowing-off leg is `ChannelNS`, which differs in solver, BCs, timestep and nondimensionalization at once. $66\%$ is therefore *everything that differs between `SolverExpert`-in-124-windows and `ChannelNS`-undivided*, plus every interaction term the additive split has nowhere to put. **That is the shape of the error the entry above corrects, repeated one level down.**

**The target moves.** The 2026-08-22 baseline sweep is quoted as *"insensitive to everything it should be insensitive to."* True of the induction column ($5.7\%$, flat — potential flow). The $P_2/P_1$ column in the same table runs $0.246$–$0.429$: a spread of $0.183$, about $25\%$ of the $0.739$ gap being decomposed. **The verdict was read off the flat column and applied to the sensitive one.** Array efficiency measures wake *survival*; it should be strongly $\nu$-dependent, and it is.

**A Reynolds mismatch may sit inside the bucket, and it is the cheapest thing to check.** $\nu=0.03$ at $D=U_\infty=1$ is $\mathrm{Re}_D\approx33$, against the spec's declared $[10^3,10^4]$ and W0's analytic check at $\nu=10^{-3}$. At $dx=0.0625$ that gives a cell Reynolds number of $2.08$ — on the central-difference limit, so $\nu$ looks **grid-dictated rather than chosen**, and the "fine grid" run then halved $dx$ at fixed $\nu$ without spending what it bought. Whether `SpectralNS`'s in-window $\nu_p$ maps through the $L=2D,\,U_s=4U_\infty,\,T_s=0.5$ scaling to `ChannelNS`'s physical $\mathrm{Re}_{\text{eff}}$ is recorded nowhere. Sign is consistent: the coupled system reads over-viscous, and viscosity raises $P_2/P_1$.

**The split is additive on a cubed quantity.** $P_2/P_1=(U_2/U_1)^3$ exactly at every row — $(0.6798/0.6502)^3=1.1429$, $(0.5837/0.7894)^3=0.4042$. In velocity space the shares are $15/13/73$, not $19/15/66$.

**Next measurement, and it is not the three-way sweep.** The missing **null control for the windowing factor**: sweep $N=124\to\sim31\to\sim8\to1$ windows through the identical harness with `SolverExpert`, rescaling $\nu_p$ at each level to hold $\mathrm{Re}_{\text{eff}}=T_s/(\nu_p L^2)$ fixed. That is what a solver expert is *for* — window size is not a free parameter for the checkpoint, but $\nu_p$ is an explicit dial that cancels the $L^2$, turning this case study's most-quoted confound into a controlled variable. Converging to $0.404$ as $N\to1$ confirms and quantifies windowing in one sweep; plateauing short says most of the $66\%$ is harness.

**[AI Inference]:** OP-5's candidates (2) and (3) may be one knob. If `SolverExpert` substeps internally to CFL then $\Delta t=0.25$ is an **exchange interval**, not an integration step, and only the frozen expert conflates them — its native lead forces the two equal. Worth settling before building a three-legged sweep with two legs.

---

## [2026-08-23] OP-5 | planned — Schwarz iteration specified, and the plan reorders OP-5's candidates

**Code:** none. Plan in full at [[schwarz-iteration-atlas-0.1]]; planned surfaces are `couple.schwarz_sweep`, `CoupleConfig.schwarz`, and an interface-residual reporter on every port and tile seam.

**Read the terminology note first.** "Schwarz" already names the **R1 halo** interface variant in this case study — built, passed W3 at $r_T=6.2\times10^{-4}$, **not adopted** in favour of R2 flux-BC tokens. R1-vs-R2 is *what data crosses an interface*; this is *how many times per macro-step*. Orthogonal, and R2 needs no change. Nothing here reopens the W3 decision.

**The loop is not one-pass, and that changes the scope.** Spec §8.3's thrust fixed point re-enters the fluid solve, and W5 measured it at **3.02 iterations mean, 5 max, converged every step**. It is stopped by $|T^{(k+1)}-T^{(k)}|/T<10^{-4}$ — **a scalar describing one rotor.** So the field exchange is already iterated roughly three times per macro-step under a criterion that never looks at whether the interfaces agree. **Verify before scoping:** confirm a fixed-point iteration re-runs the full 124-window sweep rather than re-evaluating the disk expert against a cached field. If it caches, the machinery does not exist and the estimate changes.

**The finding, and it is a precondition rather than a fix.** A Schwarz fixed point is the global solution *only if each local solve is the exact restriction of the global operator*. **A periodic window is not** — it identifies its streamwise edges, so imposing distinct neighbour data on them contradicts the operator's own structure. Iterating therefore converges to the fixed point of the wrong map, making the tiles agree with each other more precisely while leaving them all wrong the same way — **and the interface residual, the only visible symptom, is exactly what iteration removes.** So OP-5's candidate (1) is a *precondition* of candidate (3) helping at all, not a peer of it. OP-5 updated.

**[AI Inference]:** this also explains a number already on the table. The 2026-08-23 harness read monolithic $P_2/P_1 = 0.907$ against partitioned $0.889$ — removing every agent made it slightly *worse*. Under this reading that is expected: the defect is the periodic local solve, and the monolithic build has just as much of it.

**The prediction, recorded before the build.** Within a macro-step the problem is an evolution, not an elliptic one, and sufficient overlap gives finite-step convergence. Every transport mechanism sits inside the overlap — advection $U_\infty\Delta t = 0.25\,D$ against a $0.5\,D$ tile overlap, diffusion $\sqrt{\nu\Delta t}\approx0.016\,D$ — and the one mechanism that would defeat one-pass exchange, elliptic pressure coupling, is **already global** since the 2026-08-22 DCT Poisson solve. **So iterating tile seams should buy very little.** The discriminating test: declared **ports** are marginal where seams are comfortable, because the band rule sets $\text{band}=U_\infty\Delta t_{\text{macro}}=0.25\,D$ *exactly on* the one-iteration condition. An effect at ports and not at seams confirms the reasoning; the reverse refutes it.

**Additive ordering is required, not preferred.** A multiplicative sweep would reinstall the graph-cycle path dependence the W7 entry above just ruled out, **and would break [[symmetry-averaging-atlas-0.1]]'s exactness** — mirror-paired tiles visited at different points in the iteration no longer see equivalent data, so W7 degrades from *exact by construction* to *approximate and untested*. **First case in this build of two composition-layer guarantees conflicting.**

**Next action is S0, and it changes no behaviour.** Report the per-iteration interface residual $\max_{\text{seams}}\lVert\mathbf u^{(k+1)}-\mathbf u^{(k)}\rVert_{\infty,\text{overlap}}/U_\infty$ against the *existing* loop, separately for ports and seams, at $k=1,2,3$. If it has already reached the expert's per-call noise floor ($\sim3\times10^{-2}\,U_\infty$, W0) by $k=3$, iteration cannot help and stages S1–S3 should not be built. **Cheapest decisive measurement OP-5 has.**

---

## [2026-08-23] OP-5 | built, measured — the non-periodic window solve; and Schwarz iteration is provably a no-op without it

**Code:** `reference.WindowNS` (new), `SolverExpert.window_bc` / `.force_mode` / `step_many(bc=)`, `CoupleConfig.window_bc` / `.schwarz` / `.schwarz_tol` / `.force_mode`, `CoupledSystem._sweep` (Schwarz loop) / `._sweep_once` / `._overlap_masks`, `StepRecord.schwarz_*`, `invariants.symmetry.SymmetrizedExpert` (ring forwarding), `scripts/windfarm_schwarz.py`, `scripts/windfarm_reynolds_sweep.py`, `tests/atlas/windfarm/test_schwarz.py` (11 tests). Existing suite: **175 passed**, no regressions.

[[schwarz-iteration-atlas-0.1]] planned this in four stages and predicted the iteration would buy little. Building it turned the prediction into a proof, and the proof reorders the work.

### The two facts that decide the shape of the fix, both read off the code

**1. `SolverExpert` sub-steps internally to CFL.** `SpectralNS.step` computes `n_sub = ceil(dt / (cfl·dx/umax))`. So $\Delta t = 0.25$ is **not an integration step for the solver — it is an exchange interval**, and OP-5's candidates (2) *the macro-step* and (3) *one-pass exchange* are **one knob, not two**. Only the frozen expert genuinely conflates them, because its native lead forces integration step and exchange interval to be equal. This settles the question [[schwarz-iteration-atlas-0.1]] §2.2 flagged as "verify before scoping".

**2. The assembled sweep is a constant map in the iteration index — bitwise.** `_advance` gathers each window out of the global field at $t^n$ and the operator returns one field; nothing carries information from one pass to the next. Two sweeps from the same base with the same $\mathbf f$ agree to **`np.array_equal`**, not to a tolerance. **There is no boundary channel to iterate on**, so a Schwarz loop over periodic windows is not a weak fix — it is exactly a no-op costing `schwarz`$\times$ the expert calls.

That is the plan's §3.1 as an executable fact rather than an argument, and it is why `schwarz > 1` is now **refused at config time** unless `window_bc='dirichlet'`.

### `WindowNS` — the missing boundary channel

2-D incompressible NS on one $128^2$ window with a Dirichlet ring, one cell wide, ramped linearly from $t^n$ to the neighbours' current iterate. The interior initial condition stays pinned at $t^n$; only the ring moves. Explicit fractional step, skew-symmetric centred advection, wide-stencil projection.

**Validated against exact answers before it was wired to anything:**

| check | result |
|---|---|
| uniform flow (exact steady solution at any $\nu$) | **preserved exactly — $0.0$, not machine precision** |
| Taylor–Green vs. analytic, $n=64/128/256$ | $2.97/0.94/0.48 \times10^{-2}$ — second order, error is interior not boundary |
| measured divergence after projection | $7.6\times10^{-16}$ |
| Gaussian wake vs. **analytic** pure diffusion | within $\mathbf{0.44\%}$ — no spurious dissipation |
| the wrap | inflow-edge $y$-structure $<0.02$ against `SpectralNS`'s $>0.2$ |

The uniform-flow row is the one worth pausing on. The frozen checkpoint fails it ($1\to0.969$) and `SpectralNS` passes only by carrying the mean analytically outside the vorticity. **A Dirichlet window needs neither trick — the inflow ring *is* the mean**, so the boundary condition preserves it. The Galilean decomposition and the spectral shift are therefore *switched off* in this mode, not ported: both exist solely to work around periodicity, and `spectral_shift` is itself a periodic translation that would reintroduce the wrap.

### Two mistakes made and caught, because both would have read as small boundary errors

**The projection inverted a different operator from the one it applied.** Taking the divergence with `np.gradient` and subtracting `np.gradient` of $\phi$ composes into the *wide* $2h$ Laplacian while the DCT inverted the compact five-point one — interior divergence $2.5$ on a rough field instead of zero. This is exactly what `pressure.wide_eigenvalues_neumann` was written to document, one module over.

**And the obvious fix was worse.** A MAC projection — interpolate to faces, correct, average back — gives machine-zero *face* divergence, and its `centre → face → centre` round trip is the filter $[1,2,1]/4$, gain $0.962$ per sub-step at a 16-cell wavelength. At 51 sub-steps to a macro-step that is $0.962^{51}\approx0.14$: **an 86% loss of exactly the scales the wake lives at, inside the projection, in a case study whose open defect is over-dissipation.** Rejected on that arithmetic and the wide-stencil operator used instead, which is the codebase's own documented choice for collocated storage.

### A composition-layer guarantee needed extending, exactly as predicted

[[symmetry-averaging-atlas-0.1]] wraps the expert in a group average. The transmission ring is a **vector field on the same window**, so it has to travel with it: mirroring the interior while leaving the ring pointing the old way hands the operator a boundary condition belonging to the unmirrored problem — **and the average still comes back looking symmetric**, which is the failure mode that class exists to prevent rather than to create. `SymmetrizedExpert` now transforms `bc` with `g.field` alongside `force` and `frame`. Tested: symmetric interior + symmetric ring returns mirror-symmetric to $10^{-12}$.

This is the concrete form of the plan's §4 warning that two composition-layer guarantees can conflict. Additive ordering was chosen precisely so that they compose; the ring is the second thing that had to be told about the group.

### The Reynolds mismatch is real, it is in the code, and the target moves

[[open-problems-atlas-0.1]]'s 2026-08-23 qualification raised this as a question. Answered from the source: `CoupledSystem.__init__` sets $\nu = 1/255$, so every coupled run is at $\mathrm{Re}_D = 255$. `ChannelNS` produced the $0.404$ target at $\nu = 0.03$, i.e. $\mathrm{Re}_D = 33$ — **$7.7\times$ apart** — and $\nu=0.03$ at $dx=0.0625$ is a cell Reynolds number of $2.08$, the central-difference limit, so it was set by the grid rather than chosen as physics.

Re-running the baseline with $dx$ refined in step to hold cell Reynolds at $2.08$:

| $\nu$ | $\mathrm{Re}_D$ | grid | $\langle U_d\rangle$ | $P_2/P_1$ |
|---|---|---|---|---|
| $0.03$ | $33$ | $128\times384$ | $(0.7894,\,0.5837)$ | $\mathbf{0.4044}$ |
| $0.015$ | $67$ | $256\times768$ | $(0.7733,\,0.4975)$ | $\mathbf{0.2663}$ |

The first rung reproduces the published number **and its disk velocities** exactly, which is what makes the second trustworthy. **Doubling $\mathrm{Re}$ moves the target by $34\%$**, and the coupled system sits nearly four times further along that axis again. So the $0.739$ gap OP-5's shares are computed from is measured against the wrong number, the true gap at matched Reynolds is *larger*, and the correction runs against the framework rather than for it.

Both rungs are read at $t=20$ to match the coupled runs and neither is steady (drift $2.3\times10^{-2}$, $3.4\times10^{-2}$), so they are like-for-like at an instant rather than steady-state answers. The $\mathrm{Re}_D=133$ and $255$ rungs need $512\times1536$ and $1024\times3072$ grids and were not run — the machine was needed for the rollouts, and three memory-bandwidth-bound jobs starved each other to $\sim3\%$ CPU.

### Cost, and the asymmetry that is itself the result

A Dirichlet sweep of 124 windows takes $\sim91$ s against $\sim7$ s for the periodic path — roughly $\mathbf{14\times}$ — and every second of that is the Galilean frame. The periodic path advects the *fluctuation* ($u_{\max}\approx0.05$, two sub-steps); the Dirichlet path advects the *flow* ($u_{\max}\approx1.35$, fifty-one). A frozen expert cannot pay this cost at all, having no ring to impose; a solver can. **The fix for OP-5's dominant term is available to a solver and unavailable to a frozen checkpoint**, which is a statement about [[expert-library-atlas-0.1]]'s selection criteria rather than about this code.

### What is running, and the first signal

Rollouts to $t=20$ at matched settings, `periodic` against `dirichlet`, everything else held. Per-step JSONL is flushed so a partial run is still a result.

**The `periodic` leg is complete, and it validates the harness:** $P_2/P_1 = \mathbf{0.8889}$ at $t=20$, against the $0.889$ the 2026-08-23 composition-layer entry recorded for solver/PoU. Four significant figures on a number produced by different code eight days later, so the Dirichlet comparison is like-for-like rather than merely similar. Trajectory, which the earlier entry did not record: the ratio **starts inverted and falls through unity at $t\approx9.5$** — $1.0596$ at $t=4$, $1.0179$ at $t=8$, $0.9841$ at $t=10$, $0.8889$ at $t=20$ — and is *still falling* at the horizon. So $0.889$ is a reading at $t=20$, not a settled value, and the same caveat now attaches to every published number taken there.

Centreline deficit at $t=20$: $14.6\%$ at $x=-1$ (against the monolithic baseline's $5.7\%$), $19.2\%$ at $x=+2$, $19.3\%$ at $x=+9$, $3.5\%$ at $x=+16$ (baseline: $44.2$, $49.8$, $28.3\%$). Both defects OP-2 and OP-3 name are present and unchanged, which is what makes the Dirichlet leg's job legible.

**The `dirichlet` leg is unfinished** — a Dirichlet sweep is $\sim14\times$ the periodic one, so $t=20$ is an overnight job.

**Step 1 is bitwise identical between the two.** That is the correct answer and a useful one: at $t=0$ the field is uniform, both operators preserve a uniform flow exactly, and the rotor impulse is applied identically — so the two paths *cannot* differ until structure exists. It also means the divergence, when it appears, is attributable to the window boundary condition and to nothing else in the loop.

At $t=0.5$, the second step where they can differ:

| | $\langle U_d\rangle$ R1 | R2 | $P_2/P_1$ |
|---|---|---|---|
| periodic | $0.8448$ | $0.8485$ | $1.0130$ |
| **dirichlet** | $0.8223$ | $0.8190$ | $\mathbf{0.9880}$ |

**The sign is right and it is right immediately**: turbine 2 reads *below* turbine 1 with a non-periodic window and *above* it with a periodic one, which is the case study's most visible defect appearing and not appearing under a one-line change of boundary condition. **This is two steps into an eighty-step run and is recorded as a signal, not a result.** $P_2/P_1$ is a $7D$-range quantity and means nothing until the wake has transited, $t\gtrsim7$; the number to enter in the gate ledger is the one at $t=20$, against a target that the Reynolds ladder above says is no longer $0.404$.

What the plan already commits to in advance, for the Schwarz stage that does now exist: the effect should be **small at tile seams and larger at declared ports**, because the $0.5\,D$ tile overlap is twice the advective distance per macro-step while the band rule pins ports at exactly $U_\infty\Delta t$. If it comes out the other way round the reasoning is wrong somewhere, and that is worth more than the fix.

---

## [2026-08-23] OP-5 | corrected — a pinned ring traps the wake inside every window; the entry above measured the wrong boundary condition

**Code:** `WindowNS.transmission` (`'characteristic'` default, `'dirichlet'` control), `WindowNS._balance_flux`, `_pin(inflow_only=)`, `SolverExpert.window_transmission`, `CoupleConfig.window_transmission`, `--transmission` on `scripts/windfarm_schwarz.py`. Tests now 15.

The entry above pinned the **whole** boundary ring, which is what classical alternating Schwarz prescribes. For an elliptic problem that is right. For an advection-dominated one it is **ill-posed** — prescribing velocity on an *outflow* boundary over-determines the problem — and here the consequence is not subtle.

### The measurement, on a single forced window

Six macro-steps, uniform inflow ring, a thrust strip in the middle, reading the centre row:

| ring | at the strip | 10 cells downstream | last interior cell | ring cell |
|---|---|---|---|---|
| **pinned (`dirichlet`)** | $0.8687$ | $0.7992$ | $\mathbf{0.9864}$ | $1.0000$ |
| **`characteristic`** | $0.8307$ | $0.7347$ | $\mathbf{0.7049}$ | $0.7051$ |

**With the ring pinned, the wake is forced back to freestream at the outflow cells.** The deficit cannot leave the window. Across 124 overlapping tiles that annihilates the wake at *every seam* — which is [[open-problems-atlas-0.1]]'s OP-3 defect, manufactured by the repair meant to remove it. It would have looked like a fix and behaved like the disease.

`characteristic` pins the ring only where flow **enters** and lets it leave under zero normal gradient. The wake exits intact: last interior $0.7049$ against a row minimum of $0.7031$.

### This retracts the previous entry's early signal

That entry reported periodic $P_2/P_1 = 1.0130$ against Dirichlet $0.9880$ at $t=0.5$ and called the sign flip "the right direction". **The number is from the pinned-ring configuration and is withdrawn as a measurement of the fix.** It may well have been the *wrong mechanism giving a plausible-looking answer* — a trapped wake at turbine 1's window also lowers $P_2/P_1$, by starving turbine 2 rather than by shading it correctly. The run is kept as the control the corrected one is read against, and nothing from it is quoted as the fix.

**This is the sixth time in this case study a control has overturned a conclusion, and the first time the overturned conclusion was one recorded the same session.** Its predecessor — the 2026-08-23 W7 reattribution — was a *predicted* mechanism arriving at the *predicted* magnitude and still being wrong. This one is a *repair* producing a *favourable* sign for a reason that had not been checked.

### Two consequential bugs behind the correction, both caught by the divergence gate

**The extrapolated outflow must not reach the projection.** Feeding a zero-gradient rim into the pressure solve makes the window's net boundary flux inconsistent, `_poisson` absorbs the defect as a uniform constant, and a uniform constant in the right-hand side is a *quadratic* in $\phi$ whose gradient is nonzero everywhere. Measured: **the projection added $4.8\times10^{-2}$ of interior divergence to a field that arrived with $2.4\times10^{-14}$.** The zero-gradient value is a *stencil* value for the advection and diffusion stages, and is now excluded from the projection's input.

**And the window's flux has to balance.** An all-Neumann pressure solve is singular and solvable only if as much leaves as enters. With inflow prescribed and outflow free, nothing enforces that. `_balance_flux` puts the defect back on the boundary the flow leaves through, rather than spreading it over the interior — which is physically just the outflow passing whatever the inflow delivers. After both fixes: interior divergence $9.9\times10^{-14}$, compatibility mismatch $1.0\times10^{-16}$.

### The interior scheme is untouched, and the trade-off is stated rather than hidden

Taylor–Green with the **exact** ring on every face isolates the interior scheme from the transmission condition. Before and after all of the above: $2.97\times10^{-2}$, $9.42\times10^{-3}$, $4.77\times10^{-3}$ at $n=64,128,256$ — identical numbers, still second order.

Under `characteristic` the same test reads $1.40\times10^{-1}$, $1.18\times10^{-1}$, $1.01\times10^{-1}$ — an order of magnitude worse and **barely improving with resolution**, which is the correct signature of an *inconsistent boundary condition* rather than a bad scheme. Taylor–Green genuinely has structure at the outflow that a zero normal gradient cannot represent. The wind farm's outflow is a wake in a uniform stream and nearly $x$-invariant, which is why the analytic parallel-flow wake test passes under `characteristic` to $0.3\%$. **Both numbers are recorded so neither is quoted as the other**, and a configuration whose outflow carries real structure should expect the first.

### What is running

`dirichlet` + `characteristic` to $t=20$, against the completed `periodic` leg's $0.8889$. The pinned-ring run is not being repeated: its mechanism is established by a two-minute unit test, which is better evidence than an eight-hour rollout.

---

## [2026-08-24] OP-5 | measured, reverted — the characteristic run diverges; the boundary treatment is exonerated and the published $0.889$ is a transient

**Code:** no change. `results/windfarm/schwarz/dirichlet_s1_pou_impulse_characteristic.jsonl`, `periodic_s1_pou_impulse_t60.jsonl`.

### The corrected run blew up

Nine of eighty steps, then NaN.

| $t$ | $P_2/P_1$ | $\langle U_d\rangle$ R1 | R2 | $\max\lvert u\rvert$ | energy |
|---|---|---|---|---|---|
| $0.75$ | $1.2839$ | $0.7340$ | $0.7978$ | $1.230$ | $0.5012$ |
| $1.25$ | $0.7772$ | $0.7544$ | $0.6936$ | $1.304$ | $0.5034$ |
| $1.50$ | $0.2310$ | $0.7463$ | $0.4579$ | $1.433$ | $0.5079$ |
| $1.75$ | $-0.4637$ | $0.6513$ | $\mathbf{-0.5041}$ | $1.948$ | $0.5317$ |
| $2.00$ | $-3.1888$ | $1.0445$ | $-1.5374$ | $2.597$ | $0.6179$ |
| $2.25$ | NaN | | | | |

**Energy grows monotonically**, which a disk that only removes momentum cannot cause: energy is being injected numerically. Turbine 2's disk velocity reverses at $t=1.75$.

**The $0.777$ at $t=1.25$ recorded in the previous entry is withdrawn.** It was never a signal — the ratio was already thrashing ($1.00\to1.11\to1.28\to1.01\to0.78$) on a diverging trajectory, and the early steps looked calm only because the instability had not yet grown. Reporting a number from step five of a failing run was the mistake, not the number.

### The boundary treatment is exonerated, by the null control

Rotors off, uniform inflow: the exact answer is a uniform flow forever, so any drift is the tiling and boundary treatment alone.

| transmission | $\max\lvert u-1\rvert$ | $\max\lvert v\rvert$ | energy |
|---|---|---|---|
| `characteristic` | $\mathbf{0.0}$ | $\mathbf{0.0}$ | $0.500000$ |
| `dirichlet` | $\mathbf{0.0}$ | $\mathbf{0.0}$ | $0.500000$ |

Five macro-steps, 124 tiles, **exactly zero** in both. So the tiling, the partition-of-unity assembly, the global projection and *both* transmission conditions are exactly consistent with no forcing. **The instability is rotor-coupled**, and the seventh null control in this case study to move a conclusion rather than confirm one.

### The first mechanism proposed for it is wrong

The obvious explanation was that zero-gradient outflow assumes no streamwise gradient and is therefore badly wrong for a tile whose outflow edge cuts through a rotor strip. **Measured, and it is not that.** A single window with the force strip swept from mid-window to the outflow edge is *stable at every position* under `characteristic` — energy decays in all four — while `dirichlet` is the one that grows mildly at two of them:

| strip position in window | `characteristic` | `dirichlet` |
|---|---|---|
| $0.50$ | decaying | **growing** |
| $0.75$ | decaying | **growing** |
| $0.88$ | decaying | decaying |
| $0.95$ | decaying | decaying |

**A single window is not where this lives.** What the same check did turn up is geometry worth having: **both rotors straddle a declared agent seam, and each is seen at two very different positions by tiles of different agents.**

| rotor | tile | agent | position in window |
|---|---|---|---|
| R1 $x=0$ | 26, 27 | `I` | $0.75$ — outflow third |
| R1 $x=0$ | 31 | `N` | $0.25$ |
| R2 $x=7$ | 46 | `F` | $0.75$ — outflow third |
| R2 $x=7$ | 49 | `W` | $0.25$ |

Partition-of-unity blending is **within an agent only** — deliberately, so that declared ports stay real interfaces. So at the rotor plane the two agents' predictions are *not* reconciled, and they are made under opposite boundary treatments: agent `I`'s tiles are extrapolating outflow immediately behind the rotor while agent `N`'s tiles have their inflow pinned there. That disagreement sits exactly where the body force is largest and exactly where the framework declines to average. Whether it is the seed is the next measurement.

### It is not the seed either, and the mechanism is **not** confirmed

Two measurements, both against the hypothesis.

**One forced macro-step locates the action but cannot attribute it.** The largest transverse velocity sits at $(x,y) = (-0.01,\,-0.51)$ — the rotor plane at the blade tip — and is $0.3788$ on **port cells** against $0.0633$ on tile seams, a factor of six. But `characteristic` and `dirichlet` return *identical* fields at step 1 (same $\max\lvert v\rvert$, same $\langle U_d\rangle = 0.8894$, same energy $0.500125$), for the same reason step 1 of the rollouts was bitwise identical: the field is still uniform and the two conditions have nothing yet to disagree about. So this locates **where** the physics is violent, not **which** treatment breaks.

**Handed the same developed wake, the two agents' tiles disagree *less* under `characteristic`, not more.** Tiles 26/27 (agent `I`, rotor at $0.75$) against tile 31 (agent `N`, rotor at $0.25$), over their $64\times83$-cell overlap:

| transmission | $\max\lvert\Delta u\rvert$ | mean $\lvert\Delta u\rvert$ | $\max\lvert\Delta v\rvert$ |
|---|---|---|---|
| `characteristic` | $\mathbf{0.0878}$ | $\mathbf{0.0316}$ | $0.1301$ |
| `dirichlet` | $0.2985$ | $0.1031$ | $0.3213$ |

**A factor of $3.3$ the wrong way for the hypothesis.** The condition that diverges is the one that makes the declared interface *more* consistent — which is worth keeping in view, because it means cross-agent disagreement and stability are not the same axis, and a coupling layer tuned to minimise interface residual would have preferred the unstable one.

**So: two candidate mechanisms proposed, two refuted, and the cause is open.** What survives is the null control's verdict — the instability is rotor-coupled and is not a property of the tiling, the assembly, or either boundary treatment in isolation. Recorded as unresolved rather than narrated into a third guess; the next thing to try is the iteration, which is the classical remedy for exactly this class of failure and is on the plan's path anyway.

### The spec's own horizon says $0.889$ was never converged

The `periodic` solver/PoU leg run to the spec's $t_{\text{end}}=60$ rather than $t=20$:

| $t$ | $5$ | $10$ | $20$ | $30$ | $40$ | $50$ | $60$ |
|---|---|---|---|---|---|---|---|
| $P_2/P_1$ | $1.0573$ | $0.9841$ | $\mathbf{0.8889}$ | $0.8563$ | $0.8438$ | $0.8391$ | $\mathbf{0.8373}$ |

Drift over $t=50\to60$ is $1.9\times10^{-3}$ — converged. **So the $0.889$ every composition-layer number is quoted against is a transient reading at an arbitrary horizon; the settled value is $0.8373$.** That is about $7\%$ of the $0.739$ gap OP-5 decomposes, so every share moves again.

**It does not rescue the framework, and that is the important half.** $0.8373$ is still far above any plausible target, so the defect is architectural rather than an artefact of stopping early. Steady-state centreline deficit at $t=60$: $15.1\%$ at $x=-1$ against the monolithic baseline's $5.7\%$, and $20.4\%$ / $22.3\%$ / $8.0\%$ at $x=+2$ / $+9$ / $+16$ against $44.2$ / $49.8$ / $28.3\%$. **OP-2's excess blockage and OP-3's over-dissipation both survive to convergence** — they were never transients either.

**[AI Inference]:** three of OP-5's four quoted numbers have now moved for reasons that have nothing to do with the windowed architecture — the metric's cube, the baseline's Reynolds number, and now the horizon. The ranking has survived every one. That is worth stating as a property of the *ranking* rather than continuing to defend the shares: **windowing $>$ assembly $>$ checkpoint is robust, and no digit of $19/15/66$ is.**

### Where this leaves the fix

Two transmission conditions, two distinct failures: pinned traps the wake inside every window, characteristic diverges once the rotors are on. Neither is usable one-pass. **This is the classical situation in which explicit Dirichlet–Neumann coupling is unstable and the classical remedy is to iterate** — so the stability question and the Schwarz measurement the plan wanted have collapsed into a single experiment, on the only faithful local solve available. Recorded before running it, as the plan requires.

---

## [2026-08-24] OP-5 | measured — Schwarz iteration makes it **worse**; the instability is not transmission lag

**Code:** `--checkpoint-every` / `--resume` and field-state persistence in `scripts/windfarm_schwarz.py`. State in `results/windfarm/schwarz/dirichlet_s2_pou_impulse_characteristic_state/`.

Stopped at 4 of 8 steps, deliberately, once the answer was unambiguous. The remaining hour would have bought a NaN already visible in the trend.

### The measurement

`schwarz=2`, characteristic ring, everything else identical to the `schwarz=1` run, against that run at the same instants:

| $t$ | run | ratio | $\langle U_d\rangle$ R1 | R2 | energy | fp iters | converged | seam residual |
|---|---|---|---|---|---|---|---|---|
| $0.25$ | `schwarz=1` | $1.0000$ | $0.8708$ | $0.8708$ | $0.500171$ | — | — | — |
| | **`schwarz=2`** | $1.0388$ | $0.8458$ | $0.8566$ | $0.500244$ | $4$ | yes | $0.047$ |
| $0.50$ | `schwarz=1` | $1.1076$ | $0.7976$ | $0.8252$ | $0.500524$ | — | — | — |
| | **`schwarz=2`** | $1.2785$ | $0.7476$ | $0.8114$ | $0.500968$ | $6$ | yes | $0.046$ |
| $0.75$ | `schwarz=1` | $1.2839$ | $0.7340$ | $0.7978$ | $0.501181$ | — | — | — |
| | **`schwarz=2`** | $1.6930$ | $0.6580$ | $0.7842$ | $0.502161$ | $5$ | yes | $0.050$ |
| $1.00$ | `schwarz=1` | $1.0104$ | $0.7625$ | $0.7652$ | $0.501958$ | — | — | — |
| | **`schwarz=2`** | $\mathbf{2.1776}$ | $\mathbf{0.5938}$ | $0.7697$ | $0.503676$ | $7$ | **NO** | $\mathbf{0.128}$ |

**Iterating the exchange made every indicator worse at once.** Turbine 1's disk velocity collapses ($0.846\to0.594$ where the one-pass run held $0.76$), the ratio runs away upward, energy climbs faster, the seam residual jumps $2.5\times$ in a single step, and at $t=1.00$ **the thrust fixed point hits its iteration cap without converging**.

It also fails by a *different route*: `schwarz=1` collapsed turbine **2**, `schwarz=2` collapses turbine **1**.

### What this rules out, which is the point of running it

**The instability is not transmission lag.** That was the whole hypothesis: one-pass explicit Dirichlet–Neumann coupling is the classical unstable case and iterating to convergence is the classical remedy. If stale boundary data were the cause, more iterations would help. **More iterations hurt, monotonically.** So the remedy that the plan, the prior-art page and the classical theory all point at is not the remedy here.

**And the non-convergent fixed point is a gate, not a nuisance.** [[spec-wind-farm-wake-atlas-0.1]] §8.3 and [[impl-wind-farm-guide]] §6.4 both say so in advance: failure to converge means *the fluid operator responds non-monotonically to a body force*, which is the failure mode [[prior-art-and-novelty-atlas-0.1]] §2.1 predicted for frozen experts and which the guide calls "the most informative single measurement in this build". It has now fired — on a **solver**, not a frozen checkpoint. Whatever it indicates, it is not a property of a learned operator.

**[AI Inference]:** the two facts together point away from the coupling layer entirely and at the **disk–fluid coupling** itself. Iterating the field exchange tightens the interface at the same time as it lets the rotor's own feedback loop run further per macro-step, and the quantity that runs away is the disk velocity, not the seam. The null control already showed the tiling and both boundary treatments are *exact* with the rotors off. Three mechanisms have now been proposed for this instability and three refuted; the one region never yet varied is the actuator-disk model under the local-induction thrust law at $\Delta t = 0.25$.

### Process, changed

Every run now persists field state: `last_good.npz` after every step, periodic checkpoints, and the last finite state **written on divergence**, plus `--resume`. Verified rather than assumed — resuming from a $t=0.25$ checkpoint and stepping once reproduces $t=0.50$ at $P_2/P_1 = 1.0130$, $\langle U_d\rangle = (0.8448, 0.8485)$, identical to the original. The two rollouts before this one kept only per-step scalars, so the field at the step before each blowup was lost and neither could be post-mortemed or restarted. This run kept $t=0.50$, $t=1.00$ and `last_good`.

---

## [2026-08-24] OP-5 | the thrust loop is exonerated by two independent measurements, two of my own claims are retracted, and the instability is located: **the upstream induction has the wrong sign**

**Code:** no change. Artifacts: `results/windfarm/schwarz/op5_mechanism_2026-08-24.json`, `relax_probe.json`.

Four mechanisms had been proposed for the divergence and three refuted. This entry refutes the fourth, withdraws two claims made in the two entries above, and identifies a fifth that survives every check the others failed.

### 1. The disk–fluid loop gain is a contraction, bounded by $1/2$

Perturb turbine 1's thrust by $5\%$, run one full $124$-tile sweep, read the change in disk velocity — the loop the fixed point actually iterates:

$$S=\frac{d\langle U_d\rangle}{dT},\qquad \frac{dG}{dT}=C_T'\,\langle U_d\rangle\,S,\qquad \text{slope}=(1-\theta)+\theta\,\frac{dG}{dT}$$

| state | $T_0$ | $\langle U_d\rangle$ | $S$ | $dG/dT$ | slope |
|---|---|---|---|---|---|
| uniform, $t=0$ | $0.4444$ | $0.9243$ | $-0.1703$ | $-0.315$ | $0.343$ |
| checkpoint $t=0.50$ | $0.5589$ | $0.6692$ | $-0.1702$ | $-0.228$ | $0.386$ |
| checkpoint $t=1.00$ | $0.3566$ | $0.5126$ | $-0.1702$ | $-0.175$ | $0.413$ |

The gain gets **weaker** as the run degrades, because $dG/dT$ scales with the operating point $\langle U_d\rangle$ and that is what is collapsing. More thrust always means a slower disk, so $S<0$ and the slope is confined to $[0,\,1/2)$ **at every state**: it cannot reach $1$. Expansiveness would need $\lvert\langle U_d\rangle\rvert>2.9$ at the measured $\lvert S\rvert$, and even the reversed $\langle U_d\rangle=-0.5041$ of the earlier rollout gives $0.585$. **This closes the mechanism by a bound rather than by three samples.**

### 2. Confirmed independently: plain relaxation converges monotonically at the state that failed

Secant off, cap raised to $12$, same map, same $t=1.00$ state:

| iter | $T$ (R1) | $\langle U_d\rangle$ | rel |
|---|---|---|---|
| $0$ | $0.35663$ | $0.51263$ | $3.571\times10^{-1}$ |
| $3$ | $0.28243$ | $0.52526$ | $2.366\times10^{-2}$ |
| $6$ | $0.27727$ | $0.52614$ | $1.631\times10^{-3}$ |
| $10$ | $0.27690$ | $0.52620$ | $4.626\times10^{-5}$ |

Converged in $11$, **monotone on both rotors**, observed contraction ratio settling at $\mathbf{0.410}$ against the independently measured slope of $\mathbf{0.413}$. Two measurements by different routes agreeing to three digits.

### 3. Retraction — the §8.3 gate did not fire on the physics

The entry above read the non-convergent fixed point at $t=1.00$ as evidence that "the fluid expert responds non-monotonically to the body force", which [[spec-wind-farm-wake-atlas-0.1]] §8.3 names as the gate's meaning, and called it the most informative measurement in the build.

**Withdrawn.** The response is monotone — measured directly ($S=-0.1702$ at that state against $-0.1703$ at $t=0$) and then demonstrated by a monotone converging iteration on the same state. What actually happened is arithmetic: a $0.410$ contraction starting from $\text{rel}=0.357$ reaches $0.357\times0.410^{7}=6.9\times10^{-4}$ after seven iterations, and the tolerance is $10^{-4}$. **The cap is too small for the map, and nothing about the fluid is implicated.** I proposed the secant accelerator as the culprit; the data does not support that either — plain relaxation was also still short of tolerance at iteration $7$. The cap alone explains it.

**This makes the gate itself unreachable by construction.** With $\theta=0.5$ and $dG/dT\in(-0.32,0)$ the slope can never beat $\approx0.34$, so $\sim\!10$ iterations are needed from a cold start and `max_iter = 6` demands six. The spec's "typically 2–4 iterations because the map is a contraction" is right that it contracts and wrong about the rate. **[AI Inference]:** $\theta=0.5$ is the wrong relaxation here — the native map already contracts at $\lvert dG/dT\rvert\le0.32$, so under-relaxation only slows it; $\theta=1$ would give slope $0.175$ at this state and converge in $\sim\!5$. Defensive under-relaxation for a map that never needed it.

### 4. Retraction — energy was not being "injected numerically"

The entry above called monotonic energy growth something "a disk that only removes momentum cannot cause". **Withdrawn.** Measured: $\langle u\rangle=1.000000000000$ to machine precision at every checkpoint, and $\int u\,dy=8.000000$ at **every** $x$ station. That is mass conservation in an incompressible domain with impermeable walls, and it is mandatory, not a defect. With the mean pinned by it,

$$E=\tfrac12\big(\langle u\rangle^2+\operatorname{Var}u+\operatorname{Var}v\big)$$

so energy growth **is** variance growth — redistribution at conserved flux. Verified to eight digits at both checkpoints. The force removes $\Delta\langle u\rangle=-1.2357\times10^{-3}$ and the projection returns it exactly, net $-3.3\times10^{-16}$, at every thrust from $0.2$ to $2.0$ and from both a uniform and a developed field. Structural, and correct.

### 5. Two more candidates refuted at zero cost

**Overlap is adequate.** Information travels $0.25$ per exchange; the smallest agent overlap is $0.5$. Ratios: `I` $3.0$, `N` $4.0$, `F` $2.0$, `W` $2.29$, `B±` $2.33$ — all $\ge2$.

**The cross-agent handover at turbine 1 is smooth.** Ownership switches `I` to `N` at $x=-0.0078$, one cell upstream of the rotor face, and switches `F` to `W` one cell upstream of turbine 2. But the measured jump in $u$ across the R1 seam is $0.3\times$ the background $\lvert du\rvert$ — no discontinuity. (R2's `F`/`W` seam does show $4$–$9.5\times$ background, which is worth keeping.)

### 6. What survives: the upstream induction is **inverted**, and growing coherently

Centreline deficit $1-u/U_\infty$, against the monolithic reference and the earlier coupled run measured when the outflow projection was adopted:

| $x$ | monolithic | coupled (W5) | window $t=0.50$ | window $t=1.00$ |
|---|---|---|---|---|
| $-3.0$ | $+0.0062$ | $+0.069$ | $-0.0042$ | $-0.0253$ |
| $-1.0$ | $+0.0473$ | $+0.173$ | $-0.0044$ | $\mathbf{-0.1425}$ |
| $-0.5$ | $+0.0911$ | — | $+0.0942$ | $+0.0811$ |
| $+0.5$ | $+0.3842$ | $+0.331$ | $+0.2902$ | $+0.6065$ |

**Upstream of $x\approx-0.75$ the sign is wrong.** A disk decelerates the flow ahead of it; this one accelerates it, to $14\%$ *above* freestream one diameter upstream, while the bypass at $y=+2$ sits at $0.949$ — exactly inverted from the physical pattern, which is slow core and fast bypass. The correct sign does appear inside $x>-0.75$.

And it grows as one object:

| $x$ | $t=0.50$ | $t=1.00$ | factor |
|---|---|---|---|
| $-3.0$ | $-0.0042$ | $-0.0253$ | $\times6.08$ |
| $-2.0$ | $-0.0091$ | $-0.0564$ | $\times6.18$ |
| $-1.5$ | $-0.0170$ | $-0.1076$ | $\times6.33$ |

**A common growth factor at three well-separated stations is a single spatially-coherent growing mode, not local noise.** This is the first candidate consistent with every constraint the others failed: rotor-driven (the null control is exactly zero), invisible to the local disk loop (which contracts), indifferent to transmission condition and to Schwarz iteration count (it is not a seam), absent from the monolithic reference, and growing fast enough to reach the observed blowup.

### 7. Where it comes from — narrowed, not yet closed

**Not the window solver.** A single forced window under `characteristic` gives the *correct* upstream sign at every station ($+0.0013$ at $x=-0.9$ rising to $+0.075$ at the strip). So the inversion is produced by the assembly, not by the local solve.

**A prime suspect is the macro-step.** [[spec-wind-farm-wake-atlas-0.1]] §8.1 prescribes $\Delta t_{\text{macro}}=0.05$; the code runs $\mathbf{0.25}$, five times larger. Under `force_mode="impulse"` the whole macro-step's thrust lands as one instantaneous velocity change over the $0.25$-thick strip:

| $T$ | $f$ | $\Delta u=f\,\Delta t$ | $\langle U_d\rangle$ | $\Delta u/\langle U_d\rangle$ |
|---|---|---|---|---|
| $0.4444$ | $1.778$ | $0.444$ | $0.9243$ | $0.481$ |
| $0.3566$ | $1.426$ | $0.357$ | $0.5126$ | $\mathbf{0.696}$ |

**Seventy per cent of the disk velocity, removed in a single instantaneous hit**, then handed to a pressure solve to redistribute. At the spec's $\Delta t=0.05$ it would be $14\%$. **[AI Inference]:** an impulse that large is exactly the kind of forcing whose pressure response a Poisson solve will spread further upstream than a physically integrated force would, and the sign inversion sits in the induction zone where that response lives. Two tests separate this cleanly and neither has been run: `force_mode="rhs"` at the same $\Delta t$ (same cost as the existing run), and $\Delta t=0.05$ as the spec asks (five times the cost). Recorded before running either, as the plan requires.

---

## [2026-08-24] OP-2/OP-5 | the inverted induction is a **timestep** artifact: at the spec's $\Delta t$ the sign is correct, and it is not the force discretization

**Code:** no change. Single-window sweep; artifact `results/windfarm/schwarz/op5_mechanism_2026-08-24.json`.

The entry above named two candidate tests for the inverted upstream induction — `force_mode="rhs"` at the same step, and the spec's $\Delta t=0.05$ — and predicted the impulse discretization was the likely cause. **One of those two guesses was wrong and the cheap test found out before the expensive one finished.**

### The sweep

One `WindowNS` window, `characteristic` ring, the *faithful* rotor geometry (strip thickness $0.25$, halfspan $0.5$, $T=0.4444$), integrated to $t=2.0$ at three macro-steps under both force modes. Centreline deficit $1-u/U_\infty$ upstream of the disk:

| $\Delta t$ | mode | sub-steps | $\langle U_d\rangle$ | $x=-0.7$ | $x=-0.5$ |
|---|---|---|---|---|---|
| $0.25$ | impulse | $47$ | $0.4687$ | $-0.00017$ | $+0.00925$ |
| $0.25$ | **rhs** | $51$ | $0.9186$ | $\mathbf{-0.00886}$ | $\mathbf{-0.00413}$ |
| $0.05$ | impulse | $14$ | $0.8145$ | $+0.01007$ | $+0.01967$ |
| $0.05$ | rhs | $10$ | $0.8793$ | $+0.00960$ | $+0.02483$ |
| $0.0125$ | impulse | $3$ | $0.8239$ | $+0.01183$ | $+0.03286$ |
| $0.0125$ | rhs | $3$ | $0.8874$ | $+0.00747$ | $+0.02139$ |

**The wrong sign occurs only at $\Delta t=0.25$.** At the spec's $0.05$ and at $0.0125$ both force modes give a positive upstream deficit — the physical slow-core pattern — and the two modes converge toward each other as the step shrinks, which is what consistency requires. At $0.25$ they do not merely disagree, they disagree by a factor of two in disk velocity ($0.4687$ against $0.9186$) and straddle zero in induction sign.

### The prediction in the entry above is falsified

That entry proposed the impulse as the prime suspect, on the grounds that it removes $70\%$ of the disk velocity in one instantaneous hit. **`rhs` at the same step is *worse*, not better** — a stronger jet at both stations. So the defect is not *where* the force enters. It is the exchange interval itself: at $\Delta t=0.25$ neither discretization of the force is resolved, and the pressure response to an under-resolved forcing is what inverts the induction.

**[AI Inference]:** this reframes the knob. $\Delta t=0.25$ was adopted deliberately — [[open-problems-atlas-0.1]] OP-3 records it as the mitigation for the frozen expert's per-*call* dissipation, since fewer calls means less damping. That trade was made when the expert was a frozen checkpoint. With a solver expert the dissipation argument does not apply at all, and the same choice now buys an inverted induction and a divergent rollout. **A parameter chosen to work around one expert's defect was carried over to a configuration that does not have that defect** — which is a composition-layer failure mode worth naming in its own right, and not one this case study set out to look for.

### What this does and does not settle

It settles the *sign* mechanism on a single window and identifies the responsible parameter. It does **not** yet show that the full 124-tile assembly recovers at $\Delta t=0.05$: the single window reproduced the correct sign even at $\Delta t=0.25$ under `impulse`, and the tiled system did not, so the assembly contributes something the single window does not capture. $\langle U_d\rangle$ in this sweep also does not converge to momentum theory's $2/3$ — the window is $2.0$ wide, so the pinned ring sits one diameter from the disk and suppresses the induction it is trying to measure. **Only the sign and the $\Delta t$-dependence should be read from this table**, not the disk velocities.

The full-scale runs are the ones that decide it, and both are approved: `rhs` at $\Delta t=0.25$ is running as the control this table predicts will *fail*, and $\Delta t=0.05$ follows.

---

## [2026-08-24] OP-5 | **mechanism found** — the characteristic outflow is inconsistent once the outflow carries streamwise structure, and the inconsistency is self-amplifying

**Code:** no change. `results/windfarm/schwarz/dirichlet_s1_pou_rhs_characteristic.jsonl` and its state directory.

### The `rhs` control ran, and the prediction held

Predicted in the entry above, from a seconds-long single-window sweep, that `rhs` forcing at $\Delta t=0.25$ would **not** remove the inverted induction. It did not:

| $x$ | rhs $t=0.50$ | rhs $t=1.00$ |
|---|---|---|
| $-3.0$ | $-0.00247$ | $-0.00564$ |
| $-1.5$ | $-0.01918$ | $-0.03731$ |
| $-1.0$ | $-0.02352$ | $-0.06091$ |

Still a jet, still growing. **Where the force enters is not the mechanism**, now established at full scale rather than on one window.

**And `rhs` diverged *earlier*, not later** — NaN at $t=1.50$ against impulse's $t=2.25$ — and by a different route. Impulse degraded visibly over four steps ($1.28 \to 0.78 \to 0.23 \to -0.46$); `rhs` read $P_2/P_1 = 1.1051$ with $\langle U_d\rangle = (0.821, 0.849)$ at $t=1.25$ and overflowed on the next step. A run that looks healthy one step before it dies is a different failure from one that visibly decays into it.

### The refinement test that located it

Take the last good field, feed one window to `WindowNS`, **switch the rotor force off**, freeze the ring, and vary only the number of sub-steps inside the macro-step:

| transmission | 48 sub-steps | 190 | 379 |
|---|---|---|---|
| `dirichlet` | $1.2182$ | $1.2182$ | $1.2183$ |
| **`characteristic`** | $1.1730$ | $\mathbf{3.1044}$ | $\mathbf{11.8699}$ |

`dirichlet` is converged to four decimals. **`characteristic` diverges under refinement, with no rotor force at all.** Refining the time step *worsens* the answer, which is never a stability problem — it is the signature of a boundary condition that is not consistent with the equations. The outflow velocity is what runs away: mean $u$ on the outflow rim goes $0.9191 \to 0.8527$ over 48 sub-steps, $0.9195 \to \mathbf{-0.4108}$ over 190, and $0.9195 \to \mathbf{-3.7375}$ over 379. **The outlet reverses.**

### Two things it is *not*

**Not the flux patch.** `_balance_flux` applies a correction once per sub-step with a magnitude that does not scale with the sub-step, which looked like the obvious culprit. Under-relaxing it by $4\times$ and by $20\times$ changes the outcome by nothing: $11.87$, $11.87$, $11.90$ at 379 sub-steps. It is a symptom.

**Not zero-gradient outflow as such.** A clean synthetic wake — Gaussian deficit, $x$-invariant, no forcing — is stable under *both* transmissions and converges cleanly: energy $0.404697 \to 0.404728$ across the same refinement. A zero normal gradient is *exact* for an $x$-invariant outflow, and behaves perfectly when it is.

### What it is, and why it compounds

The condition is exact when the outflow is $x$-invariant and wrong in proportion to the streamwise structure crossing it. A window $2.0$ wide sitting $1.5$ downstream of a disk has an outflow that is **still developing**, so the condition is inconsistent there — and the error it commits *is itself streamwise structure*, which makes the next application more wrong. The refinement sensitivity tracks the pathology exactly:

| state | jet at $x=-1$ | 48 sub-steps | 190 | 379 |
|---|---|---|---|---|
| $t=0.50$ | $-0.0235$ | $1.2177$ | $1.2330$ | $1.6287$ |
| $t=1.00$ | $-0.0609$ | $1.1372$ | NaN | NaN |
| $t=1.25$ | $-0.1387$ | $1.1730$ | $3.1044$ | $11.8699$ |

A uniform field is exactly stable, an early field is mildly sensitive, a developed field blows up. **Positive feedback, seeded by the rotor and sustained by the boundary condition.**

### This retracts the reasoning that adopted `characteristic`

The 2026-08-23 entry recorded that Taylor–Green under `characteristic` gives $1.40\times10^{-1}$, $1.18\times10^{-1}$, $1.01\times10^{-1}$ at $n=64,128,256$ — "barely improving with resolution, which is the correct signature of an *inconsistent boundary condition*" — and then argued the wind farm was safe because "the wind farm's outflow is a wake in a uniform stream and nearly $x$-invariant."

**The diagnosis was right and the exemption was wrong.** It holds for the initial condition and fails for the developed one: a window whose outflow sits between the rotor and the far field is exactly where the wake is *not* yet $x$-invariant. The evidence for this was recorded, correctly labelled, and reasoned past. **[AI Inference]:** the general form is worth naming — a measurement was taken, its meaning was identified correctly, and it was then excused by an argument about the *intended* configuration rather than tested on it. The test that would have caught it is the one run here, costs seconds, and was available the same day.

### It also explains the results that made no sense

- **Why the null control passed exactly.** Rotors off and uniform inflow gives an $x$-invariant outflow, where the condition is exact. The control was passed trivially and could not have detected this.
- **Why more Schwarz iterations made it worse.** Each sweep is a full set of sub-steps, so `schwarz=2` doubles the sub-steps per ring refresh — twice the accumulation between resets. The iteration was not converging an interface, it was integrating an inconsistent boundary twice as long.
- **Why $\Delta t = 0.05$ looked better in the single-window sweep.** The ring is refreshed each macro-step, so the error accumulates *between refreshes*. A smaller macro-step means fewer sub-steps per refresh — $14$ against $47$ — and less drift before the reset. The $\Delta t$ finding stands, but as a modulator of this mechanism rather than a cause in its own right.
- **Why both transmissions failed differently.** `dirichlet` is stable and traps the wake; `characteristic` releases the wake and is inconsistent. **Neither is a usable open boundary**, and that is the real state of the build.

### The fix this points at

Not a tuning change. The window needs a genuine open boundary: a **convective (Orlanski) outflow**, $\partial_t u + U_c\,\partial_x u = 0$ on the outlet, which is consistent for structure being advected out rather than assuming there is none. That is the standard remedy for exactly this failure and it is a real piece of work, not a flag. Recorded before attempting it.

---

## [2026-08-24] OP-5 | **the divergence is fixed, the inverted induction is not** — an open-boundary condition was being applied to 117 interior tile seams; the wake-trapping that justified it was measured on an unrepresentative window

> **Corrected below by the verification run.** This entry was written before the fix was tested in the full 124-tile system. It prevents the blow-up; it does **not** correct the wrong-sign upstream induction. Read the verification entry before quoting the word "fixed".

**Code:** `WindowNS._convect_outflow`, `WindowNS._balance_flux_scaled`, `WindowNS._faces`, `open_faces=` on `_pin` / `step_batch` / `SolverExpert.step_many`, `couple.tile_open_faces`, wired in `CoupledSystem._advance`.

### The first fix attempt failed, and recording that is the point

The entry above proposed a convective (Orlanski) outflow as the remedy. Built it, and on its own **it is not one**:

| state | transmission | 48 sub-steps | 190 | 379 |
|---|---|---|---|---|
| $t=0.50$ | `characteristic` | $1.2177$ | $1.2330$ | $1.6287$ |
| $t=0.50$ | `convective` | $1.2280$ | $\mathbf{5.4066}$ | $\mathbf{28.16}$ |

Worse than the condition it replaced. The reason is instructive: `characteristic` overwrites the whole outflow rim with its interior neighbour every sub-step, which also **erases** the uniform bias `_balance_flux` leaves behind. Take the overwrite away and the bias has nothing to erase it. Replacing the additive flux patch with a shape-preserving rescale (`_balance_flux_scaled`) recovered two states out of three and still failed the first — the scale factor was measured at $0.9996$–$1.0006$, so the flux patch was never the driver either. **Two repairs aimed at the wrong object.**

### The actual defect

An open-boundary condition is a statement about the **domain**. The code applies one to whichever faces of *any* tile the flow happens to leave through. On an interior tile seam the neighbour's data is not merely available — **it is the right answer**, which is what classical Schwarz is. Extrapolating over it discards good data and substitutes a condition that is inconsistent wherever streamwise structure crosses.

**Only 7 of the 124 tiles have a face on the real outlet.** The other 117 were being given an open boundary on interior seams. Marking the faces geometrically, and taking the ring hard everywhere else:

| state | open faces | 48 | 190 | 379 | spread |
|---|---|---|---|---|---|
| $t=0.50$ | all (as built) | $1.2177$ | $1.2330$ | $1.6287$ | $0.4109$ |
| $t=0.50$ | **geometric** | $1.2454$ | $1.2454$ | $1.2453$ | $\mathbf{0.0000}$ |
| $t=1.00$ | all (as built) | $1.1372$ | NaN | NaN | — |
| $t=1.00$ | **geometric** | $1.2716$ | $1.2716$ | $1.2717$ | $\mathbf{0.0000}$ |
| $t=1.25$ | all (as built) | $1.1730$ | $3.1044$ | $11.8699$ | $10.6968$ |
| $t=1.25$ | **geometric** | $1.2182$ | $1.2182$ | $1.2183$ | $\mathbf{0.0002}$ |

Converged at every state, and `characteristic` and `convective` now return **identical** numbers — because on an interior tile both correctly reduce to the `dirichlet` control, which is what they should always have done there.

### The retraction this forces, and it is the one that started everything

The 2026-08-23 entry abandoned `dirichlet` transmission on this measurement: *"With the ring pinned, the wake is forced back to freestream at the outflow cells. The deficit cannot leave the window."* That test built the ring as `np.ones(...)` — **a synthetic uniform freestream**. That is the situation at the *domain outlet* and at no interior seam.

What an interior tile is actually given, read from the assembled field at the same state that diverged: a ring carrying $0.9744$ on its outflow face — **the wake, not freestream**. Run at both refinements, `dirichlet` returns $0.97440$ and $0.97440$: it reproduces the wake exactly and is converged. **A pinned ring does not trap the wake on an interior window, and never did.**

So the entire move from `dirichlet` to `characteristic` — and the divergence, the killed rollouts, and four of the refuted mechanisms — rest on a control whose boundary data was not representative of the 117 windows it was generalized to. **[AI Inference]:** the pattern is worth more than the bug. The test was correct, well-motivated, and correctly interpreted *for the configuration it set up*; what was never checked is whether that configuration is the one the system runs. A synthetic ring is exactly the kind of convenience that makes a test easy to write and silently changes what it is a test of. The cheap check — read the ring the real system supplies and look at it — costs one line and was available throughout.

### Status

`dirichlet` was the stable choice all along and is now correct everywhere except the 7 outlet tiles, which get a genuine open condition. `convective` is kept and defaulted off: it is the right condition *for those 7 faces* and its value there is untested, since a refinement test on an interior tile can no longer see it. **What is fixed is the instability; whether $P_2/P_1$ improves is a separate question and unmeasured.** The rollout that answers it has not been run.

### Locked in

Two regression tests. `test_only_the_outlet_tiles_declare_an_open_face` asserts the geometry: one face at most, always `x`-hi, exactly $7$ tiles. `test_an_interior_tile_is_substep_converged_under_every_transmission` asserts the behaviour, and does it in three parts so it cannot pass vacuously -- `dirichlet` calibrates what "converged" means for this field and scheme rather than a threshold chosen to pass, `open_faces=None` must **still diverge** so the defect stays reachable, and both open conditions must then converge with the geometric mask.

Building the second one turned up something worth keeping. The first synthetic field -- a wake varying on the *domain* scale -- gave an all-open spread of $2.9\times10^{-4}$ and **did not reproduce the bug at all**. It appears only once the streamwise ramp is sharp and sits on the face under test: $4.4\times10^{-2}$ against $1.2\times10^{-4}$ fixed, a factor of $354$. **A gentle wake is too benign to show this**, which is the same reason the original single-window checks passed and is now recorded in the test that would otherwise have inherited the blind spot.

---

## [2026-08-24] OP-5 | verified in the full system: the blow-up is gone, the inverted induction is **not**, and the entry above overstated the word "fixed"

**Code:** `ckpt_dir` now keyed on `--out` rather than on the config tag. `results/windfarm/schwarz/fix_verify_from_t125.jsonl`.

The controlled test the fix deserved: resume from the state one macro-step before the `rhs` rollout died, change **nothing but the open-face declaration**, and step past the point of death.

### The divergence is genuinely gone

| $t$ | unfixed | **fixed** |
|---|---|---|
| $1.25$ | $P_2/P_1 = 1.1051$, $\langle U_d\rangle = (0.8208, 0.8486)$ | — (resumed from here) |
| $1.50$ | **NaN** | $P_2/P_1 = 1.0216$, $\langle U_d\rangle = (0.8491, 0.8552)$ |
| $1.75$ | — | $P_2/P_1 = 0.9522$, $\langle U_d\rangle = (0.8683, 0.8542)$ |

Both steps finite, the thrust fixed point converging in 3 iterations each time, from the identical state that overflowed one step later without the change. And $P_2/P_1$ crosses **below 1** at $t=1.75$ for the first time in any non-periodic rollout — the physically correct ordering, turbine 2 below turbine 1.

### The induction is still inverted, and the entry above should not have said "fixed"

| $x$ | monolithic | $t=0.50$ | $t=1.00$ | $t=1.50$ | $t=1.75$ |
|---|---|---|---|---|---|
| $-3.0$ | $+0.0062$ | $-0.0025$ | $-0.0056$ | $-0.0285$ | $-0.0276$ |
| $-1.5$ | — | $-0.0192$ | $-0.0373$ | $-0.0838$ | $-0.0713$ |
| $-1.0$ | $+0.0473$ | $-0.0235$ | $-0.0609$ | $-0.1219$ | $\mathbf{-0.0985}$ |
| $-0.5$ | $+0.0911$ | $+0.0300$ | $+0.0111$ | $-0.0546$ | $-0.0592$ |

$-0.0985$ against a monolithic $+0.0473$: **still a jet, still the wrong sign, still an order of magnitude from the reference.** The previous entry's title claimed a fix on the strength of a sub-step refinement test on one tile. That test was sound and the defect it found was real, but a boundary condition that is now *consistent* is not the same as an induction that is now *correct*, and the two were run together in one word.

### One step of reversal, recorded as one step

The last step decreased the jet at all four upstream stations — ratios $0.97$, $0.95$, $0.85$, $0.81$ at $x=-3, -2, -1.5, -1$ — the first decrease anywhere in this case study. Energy fell for the first time as well ($0.504366 \to 0.504254$), as did $\max u$ ($1.4392 \to 1.4019$).

**This is one step and is recorded as one step.** Twice this session a number read off an early step of a rollout was later withdrawn — the $0.9880$ sign flip and the $0.777$ ratio — both because a quantity was quoted before the trajectory it sat on was known. Four indicators turning together is more than either of those had, and it is still four indicators on one step. **The claim on the record is that the blow-up is prevented. Whether the induction recovers is not yet a measurement.**

### A process defect the run itself exposed

`ckpt_dir` was keyed on the configuration tag rather than on `--out`, so this verification run — which differed from the rollout it resumed only in `--out` — wrote into that rollout's state directory and **overwrote the `last_good.npz` it had just resumed from**. The $t=1.25$ field is gone; the two earlier checkpoints and the JSONL survive, and the state is reproducible, so nothing irreplaceable was lost.

It is still exactly the failure the checkpointing rule was adopted to prevent, reintroduced by the naming of the directory that implements it. Now keyed on `--out`. **[AI Inference]:** worth noting that the safeguard failed in a way the safeguard could not see — every individual save succeeded, the divergence path worked, and the loss came from two runs agreeing on a name. A rule about *keeping* artifacts needs a companion rule about *not colliding* on them, and only the first had been written down.

---

## [2026-08-24] OP-2/OP-5 | the induction reversal was not one step: eight macro-steps of monotone recovery, and the station nearest the rotor has **changed sign**

**Code:** unchanged from the entry above. `results/windfarm/schwarz/fix_rollout_from_t175.jsonl` (+ `_state/`), continued from the verification run's last good state at $t=1.75$.

The previous entry closed with *"this is one step and is recorded as one step... the claim on the record is that the blow-up is prevented. Whether the induction recovers is not yet a measurement."* It is now a measurement.

### $P_2/P_1$

| $t$ | 1.50 | 1.75 | 2.00 | 2.25 | 2.50 | 2.75 | 3.00 | 3.25 | 3.50 |
|---|---|---|---|---|---|---|---|---|---|
| $P_2/P_1$ | $1.0216$ | $0.9522$ | $0.9092$ | $0.8998$ | $0.8973$ | $0.8799$ | $0.8602$ | $0.8381$ | $\mathbf{0.8144}$ |
| $\langle U_d \rangle_1$ | $0.8491$ | $0.8683$ | $0.8801$ | $0.8834$ | $0.8784$ | $0.8706$ | $0.8654$ | $0.8628$ | $0.8608$ |
| $\langle U_d \rangle_2$ | $0.8552$ | $0.8542$ | $0.8526$ | $0.8528$ | $0.8472$ | $0.8342$ | $0.8230$ | $0.8135$ | $0.8038$ |

Monotone after $t=2.00$, and the *rate* of decrease grows rather than decays ($-0.0025$, $-0.0174$, $-0.0197$, $-0.0221$, $-0.0237$ over the last five steps). The thrust fixed point converged in 3 iterations at every step, and mass closure held at $4.3 \times 10^{-14}$. The mechanism is visible in the two disk velocities and is the right one: turbine 1 recovers toward freestream as its spurious over-induction drains, while turbine 2 sinks as a real wake finally reaches it.

### The upstream induction, $1 - u/U_\infty$

| $x$ | monolithic | $t{=}1.50$ | $1.75$ | $2.00$ | $2.50$ | $3.00$ | $3.25$ | $3.50$ |
|---|---|---|---|---|---|---|---|---|
| $-3.0$ | $+0.0062$ | $-0.0285$ | $-0.0276$ | $-0.0267$ | $-0.0260$ | $-0.0250$ | $-0.0246$ | $-0.0241$ |
| $-1.5$ | — | $-0.0838$ | $-0.0713$ | $-0.0656$ | $-0.0595$ | $-0.0527$ | $-0.0500$ | $-0.0476$ |
| $-1.0$ | $+0.0473$ | $-0.1219$ | $-0.0985$ | $-0.0777$ | $-0.0605$ | $-0.0492$ | $-0.0453$ | $\mathbf{-0.0418}$ |
| $-0.5$ | $+0.0911$ | $-0.0546$ | $-0.0592$ | $-0.0461$ | $-0.0220$ | $-0.0059$ | $-0.0021$ | $\mathbf{+0.0013}$ |

**Monotone toward the reference at all four stations, every step, for eight steps** — and at $x=-0.5$ the induction has **changed sign**. That is the first correctly-signed upstream induction anywhere in this case study's coupled runs: every previous one, periodic or windowed, produced either a jet or an over-strong blockage. At $x=-1.0$ the magnitude has fallen by a factor of $2.9$.

The downstream profile at $t=3.50$ is a $31.6\%$ deficit at $x=+2$, $20.3\%$ at $x=+7$, $13.0\%$ at $x=+9$ and $0.0\%$ at $x=+16$, against a 2-D monolithic that still holds $28.3\%$ at $x=16$. So **OP-3 is untouched and plainly visible**: the wake is still dissipated far too fast. That is a separate defect from the induction, and nothing here was meant to address it.

### What this does and does not establish

It establishes **direction and persistence**, which is exactly what one step could not. The open-face fix did not merely stop the blow-up; the wrong-sign induction it was masking drains away on its own once the boundary condition is consistent.

It does **not** establish the answer. $P_2/P_1 = 0.8144$ against a monolithic $0.403$–$0.429$, and $-0.0418$ against $+0.0473$ at $x=-1$ — still the wrong sign one diameter out, still roughly a factor of two out on the ratio, and the sign change at $x=-0.5$ is one station of four. A trend that is monotone over eight steps is not a limit, and this case study has twice quoted an early-rollout number that a longer rollout withdrew. **What is on the record is that the trajectory is heading toward the reference from a state that used to diverge. Where it settles is unmeasured, and the run that answers it is the $t=20$ rollout.**

**[AI Inference]:** the recovery timescale is worth noting for its own sake. The monolithic baseline equilibrates its blockage in one or two convective times; this is eight macro-steps ($\Delta t = 0.25$, so $2$ convective times) into the recovery and still moving. If the settling time is genuinely comparable, the remaining gap may be transient rather than structural. But that is a hypothesis for the $t=20$ run to test, not a result — and the identical reasoning applied to the periodic rollout in W9 was **wrong** there: it drifted monotonically to $25.9\%$ instead of settling. A monotone trend has already fooled this case study once from exactly this position.

### On having under-claimed

The previous entry could have said "the induction is recovering" on the strength of four indicators turning together, and it would have turned out to be right. It said "one step" instead, and the cost of that caution was one paragraph. **The two withdrawn numbers earlier in this case study ($0.9880$ and $0.777$) were both cases where the same caution was available and was not taken.** Being right early is not the same as having measured it, and the distinction is only cheap to maintain while it does not yet matter.

---

## [2026-08-24] infra | the windowed solve is 95% batched stencils and memory-bandwidth-bound; a torch backend, verified on CPU because there is no GPU here

**Code:** `src/atlas/cases/windfarm/backend.py` (new), `WindowNS(backend=, device=)`, forwarded through `SolverExpert` and `CoupleConfig`; `scripts/windfarm_profile_step.py`, `scripts/windfarm_backend_check.py`, `vastai_windfarm.sh` (all new); `--backend/--device/--publish/--resume-release` on `windfarm_schwarz.py`; `push_assets` in `noether11.train.publish`. Runbook: [[vast-ai-windfarm-runbook]].

The rollouts are the bottleneck on every remaining question — $182$ s per macro-step, $\sim6$ h for $t=20$ — so "rent a GPU" was the obvious move. It was measured before it was made.

### Where the time is

Profiling one macro-step (124 windows, $128\times128$):

| owner | self time | share |
|---|---|---|
| window solve (`reference.py` stencils) | $173.1$ s | $\mathbf{95.2\%}$ |
| numpy ufuncs / BLAS | $7.6$ s | $4.2\%$ |
| assembly + thrust fixed point | $0.6$ s | $0.3\%$ |
| global pressure solve | $0.2$ s | $\mathbf{0.1\%}$ |

$122\,756$ Python calls in $182$ s, so essentially none of it is interpreter overhead: it is batched elementwise work on a `[124, 128, 128]` float64 array, $16$ MB a pass, $\sim1700$ passes per macro-step. Notably the **global pressure solve is 0.1%** — the piece that looks expensive, being global and serial across tiles, is free.

### The cheap alternative, checked first and refuted

21 of 22 cores were idle, and numpy releases the GIL inside large ufunc loops, so a thread pool over the batch dimension needs no copies:

| threads | 1 | 4 | 8 | 11 | 22 |
|---|---|---|---|---|---|
| speed-up | — | $2.29\times$ | $\mathbf{2.81\times}$ | $2.28\times$ | $2.29\times$ |

**Saturates at 8 and gets worse at 22.** That is a memory-bandwidth wall, not a core shortage, and it settles two questions at once: renting a bigger *CPU* box would buy almost nothing, and a GPU's advantage here is real and specific — on a 5-point stencil the edge is $\sim1$ TB/s against a desktop's tens of GB/s, not FLOPs. **The recommendation to rent a GPU is therefore a measurement, not a preference**, which is the standard this case study has had to learn to hold.

### The design constraint: there is no GPU on this machine

So the port could not be "write CUDA and hope". `WindowNS` is now written **once** against a small backend interface; numpy is a transparent pass-through, and `step_batch` takes numpy in and hands numpy back out, so the assembly, the ledger, the metrics and every existing test are untouched. torch runs the identical code on CPU, so `tests/atlas/windfarm/test_backend.py` pins torch-CPU against numpy across all three transmissions, the `open_faces` mask, ramped rings and body forcing. **`device='cuda'` is the only variable left untested off the box** — a flag, not a code path — and `scripts/windfarm_backend_check.py` closes it there before any rollout starts.

Tolerances are calibrated rather than chosen: a $10^{-6}$ relative change in $\nu$ is physically negligible and still moves the answer far more than the backends differ, and that ratio is asserted. Without it the equivalence tests could pass by comparing two things that are trivially equal — the same vacuity that let the open-face bug survive its own unit tests.

### Two things the port surfaced

**The DCT is a matmul against a matrix built by scipy itself**, not a hand-rolled FFT reordering. Deriving DCT-II from an FFT by hand is easy to get wrong by $\sqrt{2}$ in the $k=0$ row alone, and that slip does not raise — it yields a smooth, plausible pressure field with a wrong constant mode, inside a solve that is *already singular* in the constant mode. The matmul is also the better GPU shape, being the one compute-bound piece.

**The body force stayed a numpy array while the fields became tensors.** That works on CPU through numpy's interop protocol — with a `DeprecationWarning` — and **raises on CUDA**. It would have passed every test on this machine and failed on the rented box. It was caught only because the warning was read instead of ignored, which is the whole argument for `-W error::DeprecationWarning` on a port like this.

### Durability, and a safeguard that failed on its own terms

`--publish TAG` mirrors both the per-step JSONL and the last good `.npz` to a GitHub Release as the run proceeds; `--resume-release TAG` continues on a fresh box. Both halves matter differently: the `.npz` lets a run continue, and the JSONL **is the result** — a run killed after fifteen hours still said something.

The generic upload reuses the training path's hardening (bounded `gh` calls, closed stdin, one upload in flight) rather than re-deriving it. Extracting that shared path exposed a hazard worth recording: **the in-flight lock is *coalescing*, so leaking it is silent.** An early return between acquiring it and handing it to the worker would hold it forever, and the symptom is not a crash — every later upload reports "previous upload still in flight" and skips, so the run keeps going and saves nothing, which is precisely the failure the sync exists to prevent. There is now one `_abandon_staging` path every early return must use, and the regression test for it was **verified to fail when the release is removed** rather than assumed to cover it.

**[AI Inference]:** the measured CPU rate is $\sim1.8$ GB/s of effective traffic for `_ddx`, far below what the DRAM can do, which points at the per-call `np.empty_like` — a fresh $16$ MB allocation, $1700$ times a step — rather than at raw bandwidth. If so, part of the GPU's win will come from its caching allocator rather than its memory system, and a caching allocator on CPU would recover some of it for free. **Not measured; not a result.**
