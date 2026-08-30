# Wind-Farm Case Study — Phased Implementation Guide

**Type:** Implementation spec — agent task (folder: `Atlas 0.1/case-study-wind-farm-wake/implementation/`)
**Status:** Not started. **This page is written to be executed by an agent holding only this page plus [[spec-wind-farm-wake-atlas-0.1]].** Everything needed to build is here or in that spec; everything else is context.
**Design pages:** [[spec-wind-farm-wake-atlas-0.1]] (binding numbers), [[wind-farm-agent-graph-figure]] (the picture), [[port-algebra-atlas-0.1]] (interface contract), [[case-study-wind-farm-wake-2d-atlas-0.1]] (why this scenario), [[f1-pathmap-and-end-goal]] (why it matters).
**Build log:** append to [[wind-farm-implementation-log]].

---

# 0. Before writing any code — read this section fully

## 0.1 What is being built, in one paragraph

Eight agents covering a 2D hub-height plane of a two-turbine wind farm, coupled through fifteen declared ports. Five agents run a **frozen, pretrained incompressible-Navier–Stokes network**; two run a **zero-parameter algebraic actuator disk**. The disks and the fluid experts exchange momentum across four interfaces where flux matching is *exactly enforceable* because one side is closed-form. **Nothing is trained. No dataset is generated.** The deliverable is a coupled rollout plus the measurements in §8.

## 0.2 The three rules that override convenience

1. **Do not train anything.** If a phase seems to need training, the phase is wrong or the expert is wrong — stop and record it in the log. Fine-tuning "just a little" destroys the entire claim, which is that *independently pretrained frozen experts can be composed*.
2. **Report pre-projection numbers, always.** The conservation projection (§6.3) makes the residual zero by construction. A post-projection residual of $10^{-12}$ proves the projection works and nothing else. **The scientific result is the residual *before* projection.** Any table showing only the post-projection number is wrong.
3. **A failed gate stops the phase.** Do not work around a gate. Log it, and if the finding is architectural, say so — several of the most valuable outcomes here are negative results (§9).

## 0.3 Repository placement

Code home: the **`atlas-0.1` branch** of `github.com/nonidino/physics-foundation-model`, in `src/atlas/`, parallel to `src/noether11/`. Add:

```
src/atlas/
  cases/windfarm/
    geometry.py        # agent rectangles, notches, interface curves   [W1]
    disk.py            # actuator-disk expert, zero parameters         [W2]
    adapters.py        # nondim + port mapping for the frozen expert   [W3]
    couple.py          # fixed-point macro-step, projection            [W5]
    metrics.py         # r_T, R(t), Nu-analogues, wake diagnostics     [W4]
    run.py             # CLI entry point                               [W6]
  ports/
    types.py           # MECH / ROT / THERM / ELEC / ADVEC             [W1]
    residual.py        # per-port residual + global power residual     [W4]
tests/atlas/windfarm/  # one test module per phase gate
```

`src/atlas/ports/` is **not** case-study code — it is the port algebra of [[port-algebra-atlas-0.1]] and will be reused by every later case study. Keep it free of wind-farm specifics.

## 0.4 Conventions

- **Everything nondimensional.** $D=1$, $U_\infty=1$, $\rho=1$. Convective time unit $D/U_\infty$. Never mix in SI inside the solver; convert only at I/O boundaries.
- **Sign convention:** interface normal $\mathbf n$ points from the **first-named** agent to the second. Flows are positive along $\mathbf n$. Get this wrong once and every residual in §8 is meaningless — assert it in `geometry.py` and test it.
- **`float64` for all interface integrals**, whatever dtype the expert runs in. The residuals being measured are small; do not measure them in `float32`.
- **Seed everything, log the seed.** Frozen experts are deterministic, but the tokenizer's patch placement may not be.

---

# 1. Phase W0 — Verify the two ingredients exist

**Goal:** confirm the case study is buildable at all, before writing case-study code. Half a day. **Do not skip — this is where the project most plausibly dies, and it should die cheaply.**

## 1.1 Locate a fluid expert

In priority order:

1. **Poseidon** — `camlab-ethz/poseidon`, HF `camlab-ethz`. Public, incompressible-NS pretraining among its six operators. Needs the Stage-1 donor adapter of [[incremental-transfer-roadmap]].
2. **Walrus** — `PolymathicAI/walrus`, HF `polymathic-ai/walrus`. Broadest coverage; heaviest adapter.

**Record in the log which one was used and why.** This single choice conditions every number the case study produces.

## 1.2 Determine the decoder mode — this decides §6.2

Ask of the chosen checkpoint: **can it expose a stream function $\psi$, or does it decode velocity directly?**

- [[noether-1.0-rbc]] has a stream-function head natively.
- [[decoder-1.1]] made the direct-increment head the default and **retains the stream-function head as an optional structure-preserving decode mode.** If it can be engaged, engage it.
- Poseidon/Walrus decode velocity directly.

| Available | Mass conservation | Cost |
|---|---|---|
| stream function $\psi$ | **level 5, exact, free** | none |
| velocity only | level 4 — Leray projection $\mathbf u^*=\mathbf u-\nabla\Delta^{-1}(\nabla\!\cdot\!\mathbf u)$ | a Poisson solve per agent per step |
| velocity only, cheap path | level 2 — soft divergence residual, **reported not enforced** | none, but the conservation claim weakens |

**Log the choice explicitly.** §6.3's projection is much cleaner in $\psi$-space, so this is not a minor implementation detail.

## 1.3 Gate W0

- [ ] Checkpoint loads; a forward pass runs on a uniform-flow initial condition.
- [ ] Unforced 2D rollout stable to $t=60$ with no energy drift and no NaNs.
- [ ] Decoder mode determined and logged.
- [ ] Native $\Delta t$ of the expert recorded. **This is a hard constraint later** — §7.2.

**If the unforced rollout is unstable, stop.** Nothing downstream can succeed, and the finding belongs in the log as a statement about the checkpoint, not about Atlas.

---

# 2. Phase W1 — Geometry and ports (no model)

**Goal:** the agent partition and the fifteen interface curves, as data, verified without any network involved.

## 2.1 Build

`ports/types.py` — the five port types as a closed enum, each with its effort and flow variable names, units, and **mapping mode**:

| Port | Effort | Flow | Effort maps | Flow maps |
|---|---|---|---|---|
| `MECH` | traction $\mathbf t=\boldsymbol\sigma\!\cdot\!\mathbf n$ | velocity $\mathbf v$ | **consistent** | **conservative** |
| `ADVEC` | (carrier) | $\dot m=\rho\,\mathbf u\!\cdot\!\mathbf n$ | — | **conservative** |
| `ROT` | torque $\tau$ | $\omega$ | consistent | conservative |
| `THERM`, `ELEC` | declared, unused here | | | |

**Conservative vs. consistent is preCICE's distinction and it is load-bearing:** a *flow* must map so its **integral** over the interface is preserved; an *effort* must map so its **pointwise values** are preserved. Getting it backwards is a classic partitioned-coupling bug that produces plausible-looking, quietly wrong answers.

`cases/windfarm/geometry.py` — the eight agent regions and fifteen interface curves exactly as tabulated in [[spec-wind-farm-wake-atlas-0.1]] §3 and §5. $N$ and $W$ carry a **notch** where the rotor strip sits; use the active-mask channel the Phase-1 scaffold already built for the rocket's agents $b$ and $e$.

## 2.2 Gate W1 — all assertions, no model

- [ ] **Disjointness:** no two agent regions overlap. *(This is the assertion the geometry bug of [[wind-farm-agent-graph-figure]] would have failed.)*
- [ ] **Coverage:** the union of agents equals $\Omega$ up to the notches.
- [ ] **Every interface curve lies within $\epsilon_{\text{tol}}$ of *both* agents' boundaries.** Port the scaffold's existing import-time assertion verbatim.
- [ ] Materialized agent-pair set **equals** the declared set — both directions, no extras, no silent omissions.
- [ ] Normals point first-named → second-named, everywhere.
- [ ] BFS over the declared graph gives diameter 4; `n_mp = 4`.
- [ ] Token counts within 10% of the spec table; total ≈ 1,144.

---

# 3. Phase W2 — The actuator-disk expert

**Goal:** a zero-parameter expert, unit-tested against closed-form theory. Roughly 40 lines and a test file. **Build it before anything is coupled** — it is the only component with an exact answer.

## 3.1 Build — `cases/windfarm/disk.py`

$$C_T'=\frac{4a}{1-a},\qquad T=\tfrac12\rho A\,C_T'\,\langle U_d\rangle^2,\qquad P=T\,\langle U_d\rangle,$$
$$\mathbf f_{\text{disk}}=-\frac{T}{A\,\Delta_d}\,\hat{\mathbf x}\quad\text{inside the strip.}$$

**Use the local-induction $C_T'$ form, not the freestream form.** $U_\infty$ is undefined inside a wake; referencing the domain inlet would make turbine 2's thrust independent of the wake it sits in, **silently deleting the coupling this whole case study exists to test.** $\langle U_d\rangle$ is the disk-averaged streamwise velocity on the rotor's **upstream** face, supplied by the fluid expert.

Clamp $a\le0.35$. Momentum theory breaks down above $a\approx0.4$ (the turbulent-wake state); treat that as out of scope rather than modelling it.

Expose a `ROT` port carrying $(\tau,\omega)$ with $P=\tau\omega$. **Leave it unconnected** — the extracted power leaves the model there and must appear in $P_{\text{ext}}$ of §6.4.

## 3.2 Gate W2

- [ ] $C_P=4a(1-a)^2$ maximized at $a=1/3$ with value $16/27$, **to machine precision**.
- [ ] $C_T'=2$ at $a=1/3$.
- [ ] $C_T=4a(1-a)$ matches theory across $a\in[0.05,0.35]$.
- [ ] Integrating $\mathbf f_{\text{disk}}$ over the strip returns exactly $-T$.
- [ ] $a>0.4$ raises, not silently extrapolates.

---

# 4. Phase W3 — Single-interface probe, teacher-forced

**Goal:** the **go/no-go for the entire case study**, at the smallest possible scope. One turbine, two agents, no graph, no message passing.

## 4.1 Build

Take $I$ and $N$ across $R_1$ only. Feed each side the other's ground-truth quantity in turn:

- Give the disk a prescribed $\langle U_d\rangle$; check it returns the right $T$. *(Already covered by W2 — do it anyway as an integration check.)*
- Give the fluid expert the disk's $\mathbf f_{\text{disk}}$ as a body force and check the **momentum deficit it produces across the strip** against the disk's $T$.

Run both interface-condition variants:

- **R1 — overlapping (Schwarz) halo.** Agents overlap by $h$ rows; the edge layer supplies each agent's halo tokens from its neighbour's interior, so the expert always sees a locally complete stencil that looks like the interior of a full domain. **This is the positive control** — no out-of-distribution boundary at all.
- **R2 — non-overlapping flux-BC tokens.** The declared interface carries the port values as boundary-condition tokens, per Mechanism A as already built. **This is the real test**, and it is the out-of-distribution path.

## 4.2 Gate W3 — the go/no-go

$$r_T=\frac{\bigl|\,T_{\text{disk}}-|\Phi_x(V_1)|\,\bigr|}{T_{\text{disk}}}<5\%\quad\text{under at least one of R1, R2, \textbf{before} any projection.}$$

**Interpretation matters more than the number:**

| Outcome | Meaning |
|---|---|
| R1 and R2 both pass | The declared-edge contract is validated. Proceed, use R2. |
| R1 passes, R2 fails | **Halo exchange is a required part of the framework.** A first-class finding — it changes [[edge-generation-atlas-0.1]] Mechanism A. Proceed with R1 and log it prominently. |
| Neither passes | The frozen expert cannot respond correctly to a body force it never saw in training. **Stop.** This is falsification criterion F2 of [[f1-pathmap-and-end-goal]] and it is a real result — write it up rather than working around it. |

---

# 5. Phase W4 — Metrics and diagnostics (still no coupling)

**Goal:** every measurement in §8, implemented and validated on cases with known answers, **before** the coupled system can produce numbers nobody can check.

Building metrics before the system that produces them is deliberate: it is the only way to avoid tuning the system until the metric looks good.

## 5.1 Build — `ports/residual.py` and `cases/windfarm/metrics.py`

- **Per-port residual** $r_\Gamma=\lVert f_A+f_B\rVert_\Gamma\big/\tfrac12(\lVert f_A\rVert+\lVert f_B\rVert)_\Gamma$, for all 15 ports.
- **Thrust residual** $r_T$ per rotor, pre- and post-projection, reported separately.
- **Global power residual** $\mathcal R(t)=\sum_i \dot E_i+\sum_\Gamma\int_\Gamma ef\,ds-\sum_i\mathcal D_i-P_{\text{ext}}$, with $P_{\text{ext}}$ the two open `ROT` ports.
- **Wake diagnostics:** centreline deficit vs. $x$; lateral profile at $x\in\{1,3,5,7,10,14\}$; Gaussian fit quality; Jensen ($k=0.075$) and Bastankhah–Porté-Agel corridors.
- **Symmetry:** $\lVert\mathbf u^{B^+}-\mathcal M\mathbf u^{B^-}\rVert/\lVert\mathbf u\rVert$.
- **Array efficiency** $P_2/P_1$.

## 5.2 Gate W4 — validate the metrics, not the model

- [ ] On a **uniform flow with no turbines**, every $r_\Gamma=0$, $\mathcal R=0$, symmetry $=0$ to floating-point.
- [ ] On an **analytically constructed Jensen wake field**, the wake diagnostics return the Jensen parameters they were given. *(A metric that cannot recover a known answer cannot grade a model.)*
- [ ] $\mathcal R$ is exactly the extracted power when fed the analytic actuator-disk solution.

---

# 6. Phase W5 — Couple the graph

**Goal:** the eight-agent system running end to end.

## 6.1 Wire the scaffold

Reuse the Phase-1 scaffold verbatim. Replace the identity placeholder experts with (a) the frozen fluid forward and (b) the disk. The MLP gate is a **fixed two-way map** — assert it is exactly that, so nothing learned sneaks in.

**Resolve scaffold action item 4 here.** [[atlas-0.1-implementation-log]] (2026-08-09) left open whether experts must interleave with message-passing layers or whether an intra-agent token mixer is needed — with real experts in the slots this becomes measurable for the first time. Test both orderings and log the difference.

## 6.2 Mass conservation

**If $\psi$ is available (level 5):** flux across a vertical cut equals the **endpoint difference** of $\psi$,
$$\int_{y_a}^{y_b}u_x\,dy=\psi(x_0,y_b)-\psi(x_0,y_a).$$
So intra-agent divergence vanishes identically, and cross-interface mass conservation reduces to **the two agents agreeing on $\psi$ at the two endpoints of the shared curve**. Implement as a single shared value per interface endpoint that both agents read. A handful of scalars, exact, free.

**Watch the three-agent corners:** $(0,\pm2)$, $(3,\pm2)$, $(7,\pm2)$ are each shared by three agents. The rocket never had one. Assert all three agree.

**Otherwise:** the W0 fallback — Leray projection or a reported divergence residual.

## 6.3 Thrust projection at the four disk faces

The disk side is exact; correct the fluid side with a **minimum-norm perturbation applied in $\psi$-space**, which preserves $\nabla\!\cdot\!\mathbf u=0$ identically because $\nabla\!\cdot\!\nabla^\perp\equiv0$ whatever the correction is.

With $g(\psi)=\Phi_x(V_i)[\psi]+T_i$ (quadratic in $\psi$), set $\psi=\hat\psi+\lambda\mathbf w$, $\mathbf w=\nabla_\psi g/\lVert\nabla_\psi g\rVert^2$, and solve the scalar quadratic $\alpha\lambda^2+\lambda+g(\hat\psi)=0$ with $\alpha=\tfrac12\mathbf w^\top\nabla^2_\psi g\,\mathbf w$:

$$\lambda=\frac{-1+\sqrt{1-4\alpha g(\hat\psi)}}{2\alpha}\ (\alpha\neq0),\qquad \lambda=-g(\hat\psi)\ (\alpha\to0),$$

taking the root of smaller magnitude. Closed form, one step, no iteration.

**Log $g(\hat\psi)$ — the pre-projection residual — every step.** That is the measurement; $\lambda$ is just the fix.

## 6.4 The fixed-point coupling

$T$ depends on $\langle U_d\rangle$ depends on $\mathbf f_{\text{disk}}$ depends on $T$. Per macro-step:

$$T^{(k+1)}=(1-\theta)T^{(k)}+\theta\cdot\tfrac12\rho AC_T'\langle U_d\rangle^2\bigl[\mathbf f(T^{(k)})\bigr],\qquad\theta=0.5,$$

initialized from the previous macro-step, converged to $|T^{(k+1)}-T^{(k)}|/T<10^{-4}$. Expect 2–4 iterations; the map is a contraction for $a\le0.35$.

**If it needs more than ~6, switch to quasi-Newton interface acceleration (IQN-ILS)** rather than tuning $\theta$. This is solved technology in preCICE; there is no reason to rediscover it.

**Non-convergence is a gate, not a bug to smother.** It would mean the frozen expert responds **non-monotonically** to a boundary perturbation — the specific failure mode predicted in [[prior-art-and-novelty-atlas-0.1]] §2.1, and the most informative single measurement in this build. Log the iteration trace before changing anything.

## 6.5 Gate W5

- [ ] Every port materializes; agent-pair set equals declared set.
- [ ] Fixed point converges in $\le6$ iterations at every macro-step, all configurations.
- [ ] Mass residual $<10^{-8}$ ($\psi$ path) or documented ($\mathbf u$ path).
- [ ] Post-projection $r_T<10^{-8}$; **pre-projection $r_T$ recorded per step**.
- [ ] Coupled rollout reaches $t=60$ without NaN.

---

# 7. Phase W6 — Run the experiments

## 7.1 Configuration sweep

| Axis | Values |
|---|---|
| Spacing | $4D$, $7D$, $10D$ |
| Induction $a$ | $0.20$, $1/3$ |
| Inflow | uniform; **gust ramp** $U(t)=1+0.15\tanh((t-20)/2)$ |
| Interface variant | R1 halo, R2 flux-BC (whichever passed W3, plus the other if both did) |

**Run the gust case even if uniform works.** A composed system that settles to a steady wake has not tested temporal coupling at all, and steady agreement is a much weaker result than it looks.

## 7.2 Time stepping

$\Delta t_{\text{macro}}=0.05$ (CFL $\approx0.5$ at token spacing $0.25$, unit velocity). Rollout to $t=60$ ($\approx2.5$ domain transits, 1,200 steps).

**The constraint that has no classical analogue:** if the coupling proves unstable, **you cannot fix it by shrinking $\Delta t$**, because the expert was trained at a native $\Delta t$ and going below it is out of distribution. In classical co-simulation a smaller communication step is the universal remedy; here it is unavailable. If instability appears, the response is halo exchange, IQN-ILS, or a negative result — not a smaller step. Record which was used.

## 7.3 The monolithic baseline — do not skip

Run the frozen fluid expert **over the undivided domain** with both body forces applied directly. This is the gold standard for W9 in §8 and it costs one extra run. Without it there is no way to separate composition error from expert error, and the case study loses its sharpest measurement.

---

# 8. Acceptance gates

| ID | Gate | Criterion |
|---|---|---|
| W0 | Ingredients | Checkpoint loads; unforced rollout stable to $t=60$; decoder mode logged |
| W1 | Geometry | Disjoint; curves on both boundaries; pairs = declared; diameter 4 |
| W2 | Disk expert | $C_P^{\max}=16/27$ at $a=1/3$ to machine precision |
| W3 | **Go/no-go** | $r_T<5\%$ teacher-forced, pre-projection, under R1 or R2 |
| W4 | Metrics | All metrics zero on uniform flow; recover known Jensen parameters |
| W5 | Coupling | Fixed point $\le6$ iterations; mass $<10^{-8}$; rollout completes |
| W6 | Betz canary | Measured $C_P\le16/27$ at both turbines, every step |
| W7 | Symmetry | Mirror residual $<10^{-6}$ under symmetric inflow |
| W8 | Wake recovery | Self-similar Gaussian far wake; inside the Jensen/BPA corridor |
| **W9** | **Decomposed vs. monolithic** | Eight-agent RMSE against the undivided run below the expert's own single-step error |
| W10 | Array efficiency | $P_2/P_1\in[0.4,0.8]$ at $7D$ — **a wide band on purpose** (§9.2) |
| W11 | Power residual | $\lvert \mathcal R(t)\rvert <1\%$ of extracted power, with per-port breakdown |

**W3, W4 (pre-projection) and W9 are the results.** Everything else is a precondition or a canary.

## 8.1 The scaling sweep — schedule it, do not defer it

$N=2,3,5,8$ turbines in a row, same everything else. Plot composition error against interface count.

**This is the cheap early warning for falsification criterion F1** of [[f1-pathmap-and-end-goal]] — the question of whether composition scales at all, which rung 9 decides expensively. Sub-linear growth is the thesis holding. **Super-linear growth is a reason to stop the whole program and rethink, not to press on to the next case study.** Treat the result accordingly.

---

# 9. Reporting rules

## 9.1 Always three curves

Decomposed vs. monolithic vs. analytical corridor. Never one, never two.

## 9.2 The 2D discount, stated every time

2D turbulence has an **inverse** energy cascade and no vortex stretching, so 2D wakes persist longer and recover by a different mechanism than real ones. Consequently:

- **W1–W7, W9, W11 are internal-consistency measurements** — identities, cascade-independent, and trustworthy.
- **W8 and W10 are corridors and sanity bands** — not predictions, and never to be quoted as wake-model accuracy.

The distinction between an identity and a correlation is the difference between an honest result and an overclaim here.

## 9.3 Never claim a conserved quantity that was only measured

Four ports are enforced; eleven are measured. Any statement that "conservation is enforced at the interfaces" is false for eleven of the fifteen. See [[conservation-as-constraint-atlas-0.1]]'s enforce-or-measure rule.

## 9.4 Track integration hours

Record actual hours spent on the disk expert, the adapter, and the coupling, separately. **This is the first data point for falsification criterion F3** ([[f1-pathmap-and-end-goal]]) — whether per-expert integration cost stays flat as experts are added. It cannot be reconstructed later, so log it as it happens.

---

# 10. Failure-mode triage

| Symptom | Most likely cause | Meaning |
|---|---|---|
| Wake never recovers; permanent deficit | shear ports 7–10, 14–15 transmit mean flux but not turbulent transport | **Fundamental** — the declared-edge contract cannot carry entrainment. A finding about edge typing, not a bug |
| $r_T$ large, one-signed | expert misreads the body force; or control volume mismatched to the strip | Geometry/adapter error. Fixable |
| $r_T$ large, noisy | expert genuinely out of distribution on the forced strip | Use halo (R1); log as an R1-vs-R2 result |
| Fixed point diverges | non-monotone response to $\mathbf f$ | **Most serious possible outcome** — F2. Frozen experts may not be composable without fine-tuning |
| Symmetry breaks (W7) | path-dependent message passing around the graph cycle | The cycle matters; motivates ordering constraints or symmetrized message passing |
| $C_P>16/27$ | sign error in the projection, or double-counted momentum removal | Caught by W6 — this is exactly why the canary exists |
| Error super-linear in $N$ | error accumulation per interface | **F1. The headline result of the case study**, and a stop condition |

---

# 11. Definition of done

- [ ] All gates W0–W11 recorded pass or fail, with numbers, in [[wind-farm-implementation-log]].
- [ ] The $N=2,3,5,8$ scaling curve exists.
- [ ] Pre- and post-projection $r_T$ reported separately everywhere.
- [ ] Integration hours logged per component.
- [ ] R1-vs-R2 decided and written up.
- [ ] Scaffold action item 4 (expert/MP interleaving) resolved with a measurement.
- [ ] A short results page filed in this folder, with §9's reporting rules applied.

**The case study succeeds if it produces trustworthy numbers, including bad ones.** Rungs 1–4 of [[f1-pathmap-and-end-goal]] exist to answer whether composition of frozen experts preserves physical validity — an unanswered question whose answer is useful in either direction. A clean negative result here is worth more than a positive result obtained by quietly fine-tuning something.

---

## See Also

- [[spec-wind-farm-wake-atlas-0.1]] — binding geometry, equations, conservation and enforcement
- [[wind-farm-agent-graph-figure]] — the picture, plus the agent and edge schedules
- [[port-algebra-atlas-0.1]] — port semantics and mapping modes
- [[wind-farm-implementation-log]] — append build entries here
- [[f1-pathmap-and-end-goal]] — rung 2; falsification criteria F1–F3
- [[prior-art-and-novelty-atlas-0.1]] — why the fixed-point convergence gate is the interesting one
- [[impl-atlas-0.1-phase1-scaffold]] — the scaffold being reused
