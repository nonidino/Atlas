# Thermal Strain — is a volumetric coupling a bond the port algebra can type?

**Type:** Concept page — **case-study summary** (folder: `Atlas 0.1/common/`)
**Status:** opened 2026-09-01. The ninth real case study, and the first exercise of `PortAmendment`'s extension procedure. Everything here is quoted from `out/w94/w94.json`; nothing is estimated.
**Code:** `atlas/cases/thermal_strain.py` (the declarations, the three routes and the amendment submission), `scripts/w94_thermal_strain.py` (the driver), `tests/test_tier23_thermal_strain.py`
**Related:** [[port-algebra-atlas-0.1]] · [[gap-worklist]] · [[case-study-ladder-to-f1]] · [[tier0-measurements]] · [[end-to-end-architecture-spec]] · [[conservation-as-constraint-atlas-0.1]] · [[composition-error-theory]] · [[interface-transfer-theory]] · [[f1-pathmap-and-end-goal]]

---

## 1. The question, and why it is Phase B's first row

[[port-algebra-atlas-0.1]] §3 declares a **closed** vocabulary of five port types and says the burden of proof is on any addition. All five are **surface** bonds: a traction against a velocity, a temperature against an entropy flux, a torque against an angular velocity. Two ledger rows say that is not enough.

- **W70** found the first physics in this vault the vocabulary could not express — a thermoelastic shell whose conduction and elasticity are coupled through *thermal strain*, a volume term — and closed it by **reframing**. In `atlas/cases/thermal_seam.py` ([[tier0-measurements]] §13-16) that coupling is one expert's internals and it is one-way, so nothing needed a bond; what was missing was disclosure of the output.
- **W94** then found the case with no such escape. An actuator disk against a frozen operator is **two-way volumetric**: under the only decomposition axis the checkpoint supports, the disk sits inside the overlap, both windows cover it, and there is no surface between them. The row has stood open since Tier 16, and [[case-study-ladder-to-f1]] §4 schedules CS-9 as *"the single highest-leverage missing piece of vocabulary"* — three of the four Phase C subsystems (brake, tyre, battery) are blocked on it.

So the question is not *"can we couple these two things"* — plainly we can, since one solver already does. It is:

> **Is thermal strain a bond the port algebra can type, and if not, what exactly is the obstruction?**

## 2. How it is built

**One solver, cut along the physics rather than the domain.** `thermostruct2d.ThermoStruct2D` from the build repo — the same file `thermal_seam.py` uses, imported unmodified — is split into

| agent | owns | its step | its own surface port |
|---|---|---|---|
| `cond` | $T$ | backward-Euler conduction with Robin faces | `inner:THERM`, open — the drive enters here |
| `elas` | $\boldsymbol u,\ \boldsymbol\sigma$ | quasi-static plane stress with thermal eigenstrain | `outer:MECH` |

**Both agents own the whole region.** This is the first **co-located** graph in the vault: there is no cut, no overlap, no halo and no partition of unity, because $\Omega_{\text{cond}}=\Omega_{\text{elas}}=\Omega$. `Decomposition` has no member for it; the build declares `OVERLAPPING` with the overlap equal to the domain and says so in a comment rather than picking one silently.

**The geometry is the shell segment, and the streak is load-bearing.** $0.20\,\mathrm{m}\times8\,\mathrm{mm}$, $48\times6$ Q1 elements, aluminium–lithium defaults, heated on the inner face by a $600\,$K Gaussian gas streak of width $0.03\,$m at $h=2000\,\mathrm{W\,m^{-2}K^{-1}}$ and cooled outside by convection plus radiation, marched $40$ macro-steps of $50\,$ms. A *uniform* $\Delta T$ makes the whole question collapse — §5 says why — so the temperature field has to be genuinely two-dimensional, and it is produced by the solver rather than prescribed.

**The referent is free.** The unsplit `ThermoStruct2D` solve is right there. Nothing here derives a ground truth.

### 2.1 The coupling operator, extracted rather than re-derived

The thing that has to cross is the thermal load `solve_mechanical` assembles inside itself,

$$\boldsymbol f_{\text{th}} = \int_\Omega \mathsf B^{\!\top}\mathsf D\,\boldsymbol\varepsilon_0\,\mathrm dV = G\,(\boldsymbol T-T_{\text{ref}}),\qquad \boldsymbol\varepsilon_0=\alpha\,\Delta T\,(1,1,0)^{\!\top},$$

and $G$ is lifted out of that method's own quadrature loop so the term can be *named* — which is what "carry it across as a declared bond" requires. It is **asserted against the expert**, not assumed: on the expert's own free-body solve,

$$\frac{\big\lVert K\boldsymbol u-(I-VV^{\!\top})G\,\Delta T\big\rVert}{\lVert G\,\Delta T\rVert}\;=\;5.3\times10^{-15}\ \text{to}\ 1.3\times10^{-14}$$

over a uniform, a transient and a random $\Delta T$. If that failed, every number below would be measuring an arithmetic mistake.

### 2.2 Three routes, because two of them are declarable today

| route | what crosses | declarable now? |
|---|---|---|
| **`volumetric`** | the whole eigenstrain field over $\Omega$ | **no** — needs a sixth port type |
| **`surface-mech`** | the classical equivalent thermal pressure $\mathbf t=\beta\,\Delta T\,\mathbf n$ on the body's own boundary | yes, a real `MECH` bond on a real surface |
| **`global-field`** | the eigenstrain declared a `GlobalField` on `elas` | yes, and it bypasses L3 entirely |

`build(route="volumetric")` asks `ports.spec_for("THERMEL")` for the type and **is refused** — `NamedHoleError` from `holes.PORT_AMENDMENT`, at the line where a sixth type would have to enter. Nothing catches it. That refusal is the deliverable.

### 2.3 Two modes, and the second is what W94 actually asks about

`ThermoStruct2D` as built is **one-way**: `step_thermal` takes no displacement. That is W70's shape and it is the referent the schedule names. But W94's case is two-way, so the reverse half — the Biot / Gough–Joule term $-T_0\beta\,\mathrm{tr}\,\dot{\boldsymbol\varepsilon}$ in the energy equation — is restored. **`ThermoStruct2D` is not edited**: the conduction agent declines to call `step_thermal` and assembles its own step from that solver's own $M$, $K$ and Robin blocks, which is `window_ns` and `thermal_seam`'s move of supplying a part rather than changing one — the reverse term is a source, not a change to the operator.

**The reverse operator is $G^{\!\top}$: the same matrix transposed.** That reciprocity is what makes the pair a *bond* rather than two unrelated couplings, and it is exhibited here rather than asserted — the test checks $\boldsymbol u\cdot G\Delta T = \Delta T\cdot G^{\!\top}\boldsymbol u$ exactly.

---

## 3. The gate: does the split reproduce the monolith's stress field?

**Yes, and the interesting number is the second one.**

| split | relative stress error | note |
|---|---|---|
| synchronous (lag $0$) | $0$, **bit for bit** | a control on the plumbing |
| lagged one macro-step | $7.1162\times10^{-3}$ | the splitting error |
| lagged two macro-steps | $1.4301\times10^{-2}$ | |

The synchronous split is *identical arithmetic in identical order*, so its zero is a control and not a measurement — **W106**'s warning, taken seriously: a floor of exactly zero bounds nothing, so nothing here is quoted as "within reproducibility".

The lagged split is what a two-agent exchange at a shared clock actually does, and it is **first order in $\Delta t$**:

| $\Delta t$ | $0.2$ | $0.1$ | $0.05$ | $0.025$ |
|---|---|---|---|---|
| relative stress error | $2.857\times10^{-2}$ | $1.425\times10^{-2}$ | $7.116\times10^{-3}$ | $3.556\times10^{-3}$ |

with $\log_2$ ratios $1.0034,\ 1.0019,\ 1.0009$. **This is the number a bond would have had to bound**, and it is a splitting error rather than a transmission defect — the distinction §6 of this page turns on.

**The two-way half behaves the same way.** Restored, the reverse coupling moves $T$ by $1.104\,$K and $\boldsymbol\sigma$ by $3.78\times10^{-3}$ relative; the fixed point contracts in $7$ iterations at a coupling number $\delta = T_0\beta^2/(\rho c_p D_{\text{plane}}) = 9.075\times10^{-3}$. A **staggered** split — one pass, the reverse term lagged one step — lands $6.20\times10^{-5}$ from the converged pair, recovering $98.3\%$ of the coupling it exists to capture.

---

## 4. What the closed vocabulary can express, and how each way is wrong

### 4.1 `surface-mech` — right shape, wrong by three orders of magnitude

The divergence theorem splits the thermal load **exactly**:

$$\underbrace{\int_\Omega \beta\,\Delta T\,\nabla N_k\,\mathrm dV}_{G}\;=\;\underbrace{\oint_{\partial\Omega}\beta\,\Delta T\,N_k\,\mathbf n\,\mathrm ds}_{G_{\text{surf}}}\;-\;\underbrace{\int_\Omega N_k\,\beta\,\nabla(\Delta T)\,\mathrm dV}_{-G_{\text{body}}}$$

and $G=G_{\text{surf}}+G_{\text{body}}$ holds to $8\times10^{-16}$–$1.0\times10^{-15}$. $G_{\text{surf}}$ is precisely what a surface `MECH` bond on the body's own boundary can deliver; $G_{\text{body}}$ is precisely what it cannot.

On the transient, $G_{\text{body}}$ is only $9.58\%$ of the load — and dropping it is not a $10\%$ error:

| quantity | value |
|---|---|
| $\lVert G_{\text{body}}\Delta T\rVert/\lVert G\Delta T\rVert$ | $0.0958$ |
| $\lVert\boldsymbol u_{\text{surf}}\rVert/\lVert\boldsymbol u_{\text{full}}\rVert$ | $233.3$ |
| relative stress error | $1279.0$ |

**The two halves of the load nearly cancel.** An $8\,$mm strip $25$ diameters long is very soft in bending, $G_{\text{surf}}$ carries a large bending component, and $G_{\text{body}}$ removes almost all of it — so a $10\%$ defect in the load is a $233\times$ defect in the displacement. That the *answer* is then $1279\times$ out is compounded by §4.3's cancellation.

### 4.2 The positive control, which is the reason W70 never saw this

At a **uniform** $\Delta T$, $\nabla\Delta T\equiv0$, so $G_{\text{body}}=0$ identically — measured at $6.83\times10^{-17}$ of the load — and the surface route is exact to $1.03\times10^{-13}$ in displacement, with zero stress on a free body (the build repo's own M1 free-expansion oracle, reached through this file's stress recovery).

> **The classical equivalent-thermal-pressure reduction is exact exactly when the temperature has no gradient.** W70's shell was $8\,$mm of aluminium at a Biot number of $3\times10^{-4}$ — very nearly isothermal — which is why one-way, nearly-uniform thermal strain looked like it needed no vocabulary at all. Put a gradient in it and the reduction has nowhere to put $\beta\nabla\Delta T$.

### 4.3 The cancellation every relative error here sits on

$\boldsymbol\sigma=\mathsf D(\boldsymbol\varepsilon-\boldsymbol\varepsilon_0)$, and on this trajectory

$$\lVert\mathsf D\boldsymbol\varepsilon\rVert = 3.0442\times10^{9}\,\mathrm{Pa},\qquad \lVert\mathsf D\boldsymbol\varepsilon_0\rVert = 3.0445\times10^{9}\,\mathrm{Pa},\qquad \lVert\boldsymbol\sigma\rVert = 2.5186\times10^{7}\,\mathrm{Pa},$$

a **$120.9\times$ cancellation**: the stress is a $0.83\%$ residual of two terms agreeing to three digits. That is the hard norm and the right one — stress is what a structural engineer is paid to predict — and it is stated here so no reader takes $1279\times$ for a claim about the load.

### 4.4 `global-field` — exactly right, and the dangerous one

`GlobalField` bypasses L3 entirely, so declaring the eigenstrain a global field carries the whole term and reproduces the monolith **bit for bit**. And because there is then no seam, nothing is ever asked for: no scale set, no prolongation, no adjoint, no null space, no `response_half`, no $\tau$, no $\sigma$, no $\beta$.

Compiled side by side on identical physics:

| | `surface-mech` | `global-field` |
|---|---|---|
| stress error | $1279\times$ | $0$, exact |
| envelope $\mathrm{E3}$ | `fails` | **`holds`** |
| envelope $\mathrm{E7}$ | `fails` | `unchecked` |
| decertifications | $9$ | $6$ |
| decertifies at, uniquely | `L1/E3`, `L4/E7/passivity`, `L4/block-share`, `L4/operator-content` — **all four seam properties** | `L3/graph`, *"no connections declared: this is not a composition"* |
| unmeasured constants | $L,\ \sigma,\ \tau,\ C_\mu,\ \lVert A\rVert$ | the same, plus $\beta$ — no seam, no probed operator |

> **Declaring a coupling *out* of the port algebra improves its envelope stamp**, because the hypotheses that would have failed are the ones a seam defines. Four seam-level decertifications are traded for one line saying *"this is not a composition"* — which is true, and says nothing whatever about whether the coupling the page actually performs is sound. This is the silent-wrongness class in its purest form: right answer, zero certificate, and nothing in the compile says so. It is declarable today by anyone.

**W117, 2026-09-02 — this is now refused rather than merely warned about.** `GlobalField` gained a `produced_by` field and `L3/global-field` reads it; this route declares `produced_by=("cond",)` against `applies_to=("elas",)` and is **refused**. The route is kept exactly as written, because a rule with nothing to fire on is not a rule. Two facts made the gap larger than this page recorded: `global_fields` was referenced *nowhere* in the compiler, so the class had no reader at all, and the dataclass held no field saying where a global field's value comes from — the discriminator a rule needs. The rule was run against every live declaration in the vault before adoption and fires on **exactly this one of seven** ([[gap-worklist]] Tier 22, W117a).

**It is not uniformly better, and that matters for reading it.** $\beta$ is a property of a probed seam, so it becomes *unmeasurable* on the seamless route and the unmeasured list gets one item longer. The trade is specific: **two envelope hypotheses stop failing and four seam checks stop being run**, in exchange for one constant that can no longer be measured and one decertification that carries no diagnostic content.

### 4.5 Two things the probe said about the `surface-mech` seam

Both are free, both came from running the compile rather than from reasoning, and each is a small standing result.

**The interface problem is one-sided to *machine zero*.** The two blocks of the assembled operator measure $\lVert\Lambda_{\text{cond}}\rVert = 0$ exactly and $\lVert\Lambda_{\text{elas}}\rVert = 5.619\times10^{11}$. A temperature field does not know its boundary is moving, so the conduction agent's response is constant in the trace: **the interface problem this route poses is *empty*, not hard** — `CASE-STUDY-GUIDE` mistake 6 and spec §6.4(a), measured rather than reasoned about. It is a sharper instance than **W97**'s rotor seam, whose one-sidedness is a *ratio* set by the cell Reynolds number; here the ratio is zero.

**And $n_0(\Gamma)$ has a fourth row.** The seam was declared `expected_null_dim=0` and `L4/null-space` refused it. Inspection resolved it in a minute, which is the trade the guide says to make: the null direction is the **constant mode** to $5.7\times10^{-13}$ at $\sigma_{\min}/\sigma_{\max} = 2.4\times10^{-15}$, and its physical name is **rigid-body translation normal to the face** — a uniform normal velocity on the only constrained face of a *free* elastic body is a rigid motion, so it produces no strain and no reaction. The guide's table had rows for fluid–fluid `MECH` ($1$, from incompressibility), field↔lumped ($0$) and CHT `THERM` ($0$); a solid–solid `MECH` seam on a free body gets $1$ **for none of those reasons**. It is kinematics, and it now has its own row.

---

## 5. The amendment: `PortAmendment`, exercised for the first time

**Verdict: `refuse`.** Two of six fields fail, and a seventh is missing.

| field | verdict | why |
|---|---|---|
| 1 **bond** | **fails** | the pair exists and its product is a power density, and *the pairing is not unique* — §5.1 |
| 2 mapping | holds | the consistent/conservative pair is the adjoint of the declared prolongation and the derivation never mentions the carrier's dimension |
| 3 transfer | holds **vacuously** | $P=R=I$ on a shared mesh; $\dim M/\dim V = 1$, so there is no reduction because there is no cut |
| 4 **dtn_reading** | **fails** | defined and unaffordable — §5.2 |
| 5 distinctness | **holds**, and it is the strongest field | §4.1's exact decomposition, with §4.2's control |
| 6 exercise | holds | this case study, plus W94's actuator disk |

### 5.1 Field 1: the bond exists and the pairing does not

The conjugate pair is $\big(\Delta T,\ \beta\,\mathrm{tr}\,\dot{\boldsymbol\varepsilon}\big)$ with $\beta = E\alpha/(1-\nu) = 2.403\times10^{6}\,\mathrm{Pa\,K^{-1}}$ and a product in $\mathrm{W\,m^{-3}}$. That the measure changes from $\mathrm ds$ to $\mathrm dV$ is survivable on its own. What is not survivable is that **there are two readings and they are not the same number**:

$$P_\Omega^{(A)}=\int_\Omega\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}_0\,\mathrm dV = -5.1949\times10^{-2}\,\mathrm{W/m},\qquad P_\Omega^{(B)}=\int_\Omega\beta\,\Delta T\,\mathrm{tr}\,\dot{\boldsymbol\varepsilon}\,\mathrm dV = +1.7638\times10^{2}\,\mathrm{W/m},$$

differing by $3395\times$, with

$$P_\Omega^{(A)}+P_\Omega^{(B)}=\frac{\mathrm d}{\mathrm dt}\Big(\boldsymbol u^{\!\top}G\,\Delta T-\tfrac12\Delta T^{\!\top}H\,\Delta T\Big)$$

exactly — a **total derivative of an energy neither agent owns**, verified to converge as $\Delta t\to0$ (measured $175.57$ against $169.03$ at $\Delta t = 0.2$, $176.52$ against $176.12$ at $\Delta t=0.0125$).

The mechanism is the free energy, and it is arithmetic rather than opinion. The elastic stored energy $\tfrac12\!\int\!\boldsymbol\sigma:\mathsf D^{-1}\!:\boldsymbol\sigma\,\mathrm dV$ expands as

$$\tfrac12\boldsymbol u^{\!\top}K\boldsymbol u\;\underbrace{-\;\boldsymbol u^{\!\top}G\,\Delta T}_{\text{bilinear}}\;+\;\tfrac12\Delta T^{\!\top}H\,\Delta T,$$

and the middle term is a function of **both** agents' states. Measured, $\lvert E_\times\rvert = 4.93\times10^{2}\,\mathrm{J/m}$ against an elastic energy of $1.84\times10^{-1}\,\mathrm{J/m}$: **$2680\times$**. There is no way to write $\Psi = \Psi_{\text{cond}}(T)+\Psi_{\text{elas}}(\boldsymbol u)$, and $\mathcal R(t)$'s first sum presumes one.

For a *free* body the ratio is exact rather than incidental: $K\boldsymbol u = (I-VV^{\!\top})G\Delta T$ and $V^{\!\top}\boldsymbol u = 0$ give $\boldsymbol u^{\!\top}K\boldsymbol u = \boldsymbol u^{\!\top}G\Delta T$, so $E_\times = -2\,E_{\text{strain}}$ identically. **The joint term is twice the term the elasticity agent could own.**

### 5.2 Field 4: defined, and not affordable

Which side is imposed and which returned is perfectly well defined — `cond` imposes $\Delta T$, `elas` returns $\beta\,\mathrm{tr}\,\boldsymbol\varepsilon$ — and under the two-way mode the interface problem is genuinely non-empty ($7$ fixed-point iterations). What fails is the economics that make probed-DtN a method at all: the trace space **is** the state, $\dim M = \dim V = 343$, so a finite-difference probe is $344$ full solves to build an operator that *is* the coupled solve. And this is decidable **from the declaration** — co-dimension $0$ — rather than from any measurement.

### 5.3 The seventh field, which the first exercise found

`holes.PORT_AMENDMENT`'s own `inference_note` said the six-field checklist was *"a proposed procedure with no evidence it is sufficient ... the first exercise may well show a seventh field is needed."* It does.

> **Field 7 — `support`.** The **co-dimension of the bond's carrier**, and, if it is $0$, the argument that the two agents' free energies are **additive**.

Fields 1–6 are *all* satisfiable by a co-located pair, and not one of them notices that $\mathcal R$'s premise has failed. A procedure that cannot refuse on that is not gating the thing that matters. It is now [[port-algebra-atlas-0.1]] §10.1, binding.

### 5.4 The constructive result

> **A port is a bond on an interface of co-dimension $\ge1$ between agents whose free energies add. Thermal strain is a bond of co-dimension $0$ between agents whose free energies do not add. It is a bond and it is not a port.**

This is **W70's conclusion reached from the other side and sharpened**. W70 said *"a sixth port type is not the fix, because the object is not a bond"*; measured, the object **is** a bond — a conjugate pair whose product is a power density, with $G^{\!\top}$ as its reciprocal half — and what it is not is a *port*. The difference is not pedantry, because it names the instrument: an **operator splitting with a splitting-error bound**, which §3 measured at $7.1\times10^{-3}$ per macro-step of lag and first order in $\Delta t$, rather than a transmission condition. That is what CS-11 (brake), CS-15 (tyre) and the battery row should now be built on.

---

## 6. $\mathcal R(t)$, and where §6 of the port algebra applies

The gate asks that the global power residual close with the volumetric term included. It does, and the more useful half of the answer is *where*.

**The thermal balance closes on its surface port alone.** $\mathrm dE_{\text{th}}/\mathrm dt = 5.006438\times10^{4}$ against $P_\Gamma = 5.006429\times10^{4}\,\mathrm{W/m}$ — seven digits, and **no volumetric term appears**, because `ThermoStruct2D` truncates the reverse half.

**The elasticity agent's balance is closed by the volumetric term and by nothing else.** The body is traction-free, so $\mathrm dE_{\text{el}}/\mathrm dt = -P_\Omega$, and the residual is first order in $\Delta t$:

| $\Delta t$ | $0.2$ | $0.1$ | $0.05$ | $0.025$ | $0.0125$ |
|---|---|---|---|---|---|
| $\lvert\mathrm dE_{\text{el}}/\mathrm dt+P_\Omega\rvert/\lvert P_\Omega\rvert$ | $6.02\times10^{-2}$ | $2.97\times10^{-2}$ | $1.47\times10^{-2}$ | $7.34\times10^{-3}$ | $3.66\times10^{-3}$ |

$\log_2$ ratios $1.020,\ 1.011,\ 1.006,\ 1.003$ — the residual is the time differencing, not the bond.

**And the global $\mathcal R$ cannot see any of it.** $P_\Omega/P_\Gamma = -1.04\times10^{-6}$ and $E_{\text{el}}/E_{\text{th}} = 1.17\times10^{-6}$. Including the volumetric term moves the relative global residual from $2.909\times10^{-6}$ to $1.871\times10^{-6}$ — a real $36\%$ improvement, and both numbers are the same order as the backward-Euler differencing error, so the term is at the noise floor of the quantity §6 asks to be reported every macro-step.

> [[port-algebra-atlas-0.1]] §6 calls $\mathcal R(t)$ *"the cheapest possible physical-plausibility monitor"*. On a co-located split it is blind to the coupling under test by six orders of magnitude, and **the right residual is the receiving subsystem's own balance**. That bounds where §6 applies; it does not retract it.

---

## 7. Four framework defects the case study found without looking for them

(§4.5 has two more findings of the same kind, about the seam rather than the framework: an interface problem that is one-sided to machine zero, and a fourth row for $n_0(\Gamma)$.)

### 7.1 W113 — a graph with no connections could not emit an artifact

`_l3_connections` stamped $\mathrm{E2}$ `unchecked` when a graph had no seams, and `emit.validate` refuses an unchecked $\mathrm{E2}$ outright — its own words, *"a defect in the compiler, not a property of the case"*, because $\mathrm{E1}$–$\mathrm{E4}$ are static properties of a declaration. The two rules contradicted each other and a zero-connection graph raised `EmitRefused` and produced nothing. Latent until the `global-field` route, **the first graph in this vault with zero seams**. Fixed: $\mathrm{E2}$ *holds vacuously* over an empty interface set, and the L3 decertification already says the thing that matters.

### 7.2 W114 — R10 refuses a co-located split and its derivation does not reach one

`L2/R10` refuses any graph with two agents one of which declares `elliptic_subsolve=embedded`, and its sentence is *"the graph decomposes the domain … so the decomposition changes the operator rather than restricting it"*. **A co-located split cuts no domain**: both agents own all of $\Omega$, and the quasi-static elasticity solve is over exactly the region it was over in the monolith. The rule is stated over `elliptic_subsolve` alone and never consults whether the decomposition cuts the agent, so it fires where its own derivation does not reach — and it refuses **every** graph in this case study, for a reason that has nothing to do with the bond.

R10 is right about the class it was derived on and this is a scope defect, not a contradiction. Recorded as `thermal_strain.R10_SCOPE` so the refusal is never read as being about the coupling.

### 7.3 W116 — the splitting error has no slot to be declared in

The gate's second number, $7.116\times10^{-3}$, is exactly the kind of quantity `MeasuredConstants` exists to carry: measured, reproduced over an $8\times$ sweep, first order in the exchange interval. **There is no field for it.** The nearest is $\sigma$, the *transmission* infidelity, which the master bound consumes as an interface-power quantity measured between two assemblies; this is a *lagged-operator-splitting* defect measured in relative stress, because there is no interface to take a power over.

Declaring it as $\sigma$ would be **W56 with the sign reversed** — not a bound quoted from a constant nobody measured, but a bound quoted from a constant somebody measured *and that means something else*. So `MEASURED_W94` declares `tau = 0` (true, and free: a real solver against itself), leaves $\sigma$ unmeasured, and `SPLITTING_ERROR` sits beside it as a module constant nothing consumes.

**Found by nearly doing the wrong thing.** The first version of this case study declared the splitting error as $\sigma$ on the seamless route and not on the surface one, which made §4.4's comparison asymmetric and produced a headline — *"the exact route drops $\tau$ and $\sigma$ from its unmeasured list"* — that was a property of the declaration rather than of the routes. §4.4 above is the symmetric comparison, and it is smaller and sharper.

### 7.4 A structural agent has no `split-step` escape

The move `window_ns` and `thermal_seam` both make — expose the elliptic part, let the composition layer own it, take stable explicit sub-steps beside it — **has no analogue for quasi-static elasticity**. There is no time derivative to sub-step. An elasticity agent is `EMBEDDED` or it is not an agent, so CS-8's advice to *"start from the exposed-agent plus `ProjectedAssembly` column"* does not transfer to a structural agent, and CS-12's wing FSI will meet this first.

---

## 8. What this case study cannot say

- **One material, one geometry, one linear constitutive law.** The bilinear cross term is a fact about *linear thermoelasticity*. **[AI Inference]:** every volumetric two-way coupling has one, because a coupling that is not a boundary exchange is a term in the free energy density and a term in a density is bilinear in the two fields at leading order. Argued, not measured.
- **The two-way half is small here.** $\delta = 9.1\times10^{-3}$ for aluminium. W94's actuator disk is not obviously in the same regime, and nothing here measures it there.
- **The splitting error is quoted at one lag and one operating point.** Same discipline as `sigma_lag` ([[gap-worklist]] W86): $7.116\times10^{-3}$ at a lag of one $50\,$ms macro-step, on the streak transient at $40$ steps, in *relative stress* and **not in interface power** — there is no interface to take a power over. It is therefore **not comparable** with `thermal_seam`'s $\sigma$ ([[tier0-measurements]] §17), and `MEASURED_W94`'s `source` field says so. **It is therefore not declared at all** — see §7.3.
- **Nothing here reaches `admit`.** R10 refuses every route (§7.2), $L$ is unmeasured (W1), and no assembly is declared because there is nothing to assemble.

---

## 9. Rows this closes and opens

| row | outcome |
|---|---|
| **W94** | **closed.** A two-way volumetric coupling has a bond and does not have a port, and the obstruction is localized to fields 1, 4 and 7. The actuator disk's third option — two ports at the nearest rings, `geometrically_coincident=False` — is now readable as what it is: a *surface approximation of a co-located bond*, and §4.1 is the measurement of what such an approximation drops |
| **W32** | **closed.** The six-field procedure is written into [[port-algebra-atlas-0.1]] §10 as a binding section, exercised once, and it **refused** — and the exercise found a seventh field, which is exactly the outcome `holes.py`'s `inference_note` predicted |
| **W70** | **confirmed from the other side.** Reframed once as *"one expert's internals and one-way"*; measured here as *"a bond and not a port"*. §4.2 explains why the one-way shell could never have shown it: at a Biot number of $3\times10^{-4}$ the body force is nearly zero |
| **W113** | **opened and closed in this tier** — §7.1 |
| **W114** | **opened.** R10's scope, §7.2 |
| **W115** | **opened.** `Decomposition` has no member for a co-located split, and `OVERLAPPING` with the overlap equal to the domain is a declaration nothing checks |
| **W116** | **opened.** A co-located split's own measured defect is not declarable: `MeasuredConstants` has a slot for a *transmission* infidelity and none for a *splitting* one — §7.3 |

## See Also

- [[port-algebra-atlas-0.1]] — §10 is the binding amendment procedure this case study wrote and then failed
- [[tier0-measurements]] — §13-17 carry `thermal_seam`, the same solver at a *surface* seam, and W70's original framing behind `VOLUMETRIC_COUPLING_SCOPE`
- [[case-study-ladder-to-f1]] — §4 Phase B, which scheduled this, and CS-11 / CS-15, which were waiting on it
- [[gap-worklist]] — Tier 22: W94, W32 and W70 closed, and the four rows §9 opens
- [[end-to-end-architecture-spec]] — §12.5's `PortAmendment` slot, and L2/R10
- [[conservation-as-constraint-atlas-0.1]] — per-port enforcement against measurement, which §6 bounds the scope of
- [[composition-error-theory]] — why $\mathcal R(t)$ is a falsifier rather than a bound, now with a measured blind spot
