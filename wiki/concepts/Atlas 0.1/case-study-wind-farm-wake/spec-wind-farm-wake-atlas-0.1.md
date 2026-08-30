# Atlas 0.1: Wind-Farm Wake — Full Case-Study Specification

**Type:** Core Concept — Case-Study Specification (folder: `Atlas 0.1/case-study-wind-farm-wake/`)
**Status:** Specification, not built. This is the binding, self-contained statement of the scenario: geometry, agents, governing equations, edges, interface quantities, conservation laws and their enforcement, time stepping, and acceptance gates. The rationale for choosing this scenario — the expert/data screen that eliminated wildfire, aircraft, car, battery and reentry — lives in [[case-study-wind-farm-wake-2d-atlas-0.1]] and is not repeated here.
**Related Concepts:** [[case-study-wind-farm-wake-2d-atlas-0.1]], [[wind-farm-agent-graph-figure]], [[impl-wind-farm-guide]], [[port-algebra-atlas-0.1]], [[f1-pathmap-and-end-goal]], [[agent-definition-atlas-0.1]], [[edge-generation-atlas-0.1]], [[expert-library-atlas-0.1]], [[conservation-as-constraint-atlas-0.1]], [[graph-tokenizer-atlas-0.1]], [[unet-hierarchy-atlas-0.1]], [[global-fields-and-topology-atlas-0.1]], [[impl-atlas-0.1-phase1-scaffold]], [[00-atlas-0.1-overview]]

> **Reading order.** §1–2 fix the physics and the nondimensionalization. §3 is the agent table, §4 the governing equations per family, §5 the edge table, §6 the interface-quantity contract. **§7 is the load-bearing section**: which conservation laws hold, in what integral form, and by exactly what mechanism each is enforced. §8–12 cover time stepping, boundary conditions, gates, and the declared/learned/closed-form ledger.

---

# 1. The scenario in one paragraph

Two horizontal-axis wind turbines stand in line with the wind, seven rotor diameters apart, on flat terrain. The upstream machine extracts momentum from the flow and leaves behind a **wake** — a region of reduced velocity and elevated turbulence — which spreads laterally, recovers by entraining momentum from the undisturbed flow beside it, and arrives at the downstream machine still depleted. The downstream turbine therefore produces less power, and the whole farm produces less than the sum of its isolated turbines. Predicting that deficit, and manipulating it by yawing the upstream rotor to deflect its wake, is the central problem of wind-farm control. **We model the hub-height horizontal plane.**

This is a multi-agent problem in the exact sense Atlas means: distinct physical components (rotors, wake regions, undisturbed flow) governed by different mathematics (algebraic momentum theory vs. a PDE), exchanging a conserved quantity (momentum) across declared interfaces, where getting the exchange wrong produces an answer that is not merely inaccurate but **unphysical** — a turbine above the Betz limit, or a wake that never recovers.

---

# 2. Physical setting and nondimensionalization

## 2.1 Dimensional reference values

A utility-scale onshore machine, for concreteness:

| Symbol | Quantity | Value |
|---|---|---|
| $D$ | rotor diameter | $120$ m |
| $U_\infty$ | freestream hub-height wind speed | $9$ m s$^{-1}$ |
| $\rho$ | air density | $1.225$ kg m$^{-3}$ |
| $a$ | axial induction factor (design point) | $1/3$ |
| $s$ | streamwise turbine spacing | $7D = 840$ m |
| $I_\infty$ | ambient turbulence intensity | $0.06$–$0.12$ |

## 2.2 Nondimensionalization — the ruler

Following [[conditioning-and-constants-1.1]]'s ruler/dial/arrow contract, length and velocity are **rulers** (divided out), not dials:

$$\tilde{\mathbf x}=\mathbf x/D,\qquad \tilde{\mathbf u}=\mathbf u/U_\infty,\qquad \tilde t = tU_\infty/D,\qquad \tilde p = p/(\rho U_\infty^2).$$

The convective time unit is $D/U_\infty=13.3$ s. A particle traverses the $24D$ domain in $\tilde t=24$. Tildes are dropped from here on; **every quantity below is nondimensional unless carrying units.**

## 2.3 The Reynolds number, stated honestly

$\mathrm{Re}_D=U_\infty D/\nu\approx7\times10^7$. Nothing in this project resolves that, and no expert was trained near it. The plan-view model therefore runs at an **effective Reynolds number** $\mathrm{Re}_{\text{eff}}=U_\infty D/\nu_{\text{eff}}$ set by an eddy viscosity, and $\mathrm{Re}_{\text{eff}}\in[10^3,10^4]$ is supplied as a **dial** (conditioning token), inside the band the fluid expert's 2D-turbulence pretraining covers.

**This is a model, not a DNS, and the spec says so in three places on purpose.** The consequence is recorded in §11 as the 2D discount: wake *recovery rates* are quantitatively unreliable; interface *consistency* is not affected, because it is an identity, not a correlation.

## 2.4 Coordinates and domain

$x$ streamwise, $y$ lateral, origin at the base of turbine 1's tower projected into the plane.

$$\Omega=[-6,\;18]\times[-4,\;4]\quad\text{(in units of }D\text{)}$$

Turbine 1 at $x=0$, turbine 2 at $x=7$. Rotors span $|y|\le0.5$. The wake corridor is $|y|\le2$; the bypass corridor is $2<|y|\le4$. At the domain exit the Jensen half-width is $0.5+k\cdot11\approx1.3$ for onshore $k=0.075$, comfortably inside the corridor — so the corridor width is a justified choice, not a guess.

---

# 3. The agent set

Eight agents forming a **disjoint** cover of $\Omega$. Rotor agents are strips of thickness $\Delta_d=0.1$ immediately downstream of the rotor plane; the wake agents carry a matching notch.

| ID | Agent | Domain | Governing family | Expert | State | Tokens |
|---|---|---|---|---|---|---|
| $I$ | inflow corridor | $x\in[-6,0)$, $\lvert y\rvert\le4$ | F1 incompressible NS | `fluid` (frozen) | $(u_x,u_y,p)$ | 192 |
| $R_1$ | rotor 1 | $x\in[0,0.1]$, $\lvert y\rvert\le0.5$ | F2 actuator disk | `disk` (closed form) | $(a,C_T',T,P)$ | 20 |
| $N$ | near wake | $x\in[0,3]$, $\lvert y\rvert\le2$, **minus $R_1$** | F1 | `fluid` | $(u_x,u_y,p)$ | 192 |
| $F$ | far wake | $x\in(3,7)$, $\lvert y\rvert\le2$ | F1 | `fluid` | $(u_x,u_y,p)$ | 256 |
| $R_2$ | rotor 2 | $x\in[7,7.1]$, $\lvert y\rvert\le0.5$ | F2 | `disk` | $(a,C_T',T,P)$ | 20 |
| $W$ | exit wake | $x\in[7,18]$, $\lvert y\rvert\le2$, **minus $R_2$** | F1 | `fluid` | $(u_x,u_y,p)$ | 176 |
| $B^+$ | bypass, north | $x\in[0,18]$, $2<y\le4$ | F1 | `fluid` | $(u_x,u_y,p)$ | 144 |
| $B^-$ | bypass, south | $x\in[0,18]$, $-4\le y<-2$ | F1 | `fluid` | $(u_x,u_y,p)$ | 144 |

> **Geometry correction, 2026-08-19.** An earlier revision placed the rotor strips at $\lvert x\rvert\le0.05$ and the bypass corridors at $x\in[-6,18]$. Both **overlapped their neighbours** — $R_1$ with $I$ on $x\in[-0.05,0)$, and $B^\pm$ with $I$ on all of $x\in[-6,0)$ — and the second contradicted this page's own edge list, which places $I\!-\!B^\pm$ at $x=0$. The partition above is disjoint and consistent with §5. **The error was found by drawing the graph** ([[wind-farm-agent-graph-figure]]); left in place it would have fired the scaffold's import-time assertion that every interface curve lies on *both* agents' boundaries.
>
> Two consequences worth stating. **Upstream of the first rotor the full height is one agent** — there is no wake there to separate from a bypass, so the split begins at $x=0$. And **$N$ and $W$ are rectangles with a notch** where the rotor strip sits, handled by the same active-mask channel the Phase-1 scaffold already built for the rocket's agents $b$ and $e$.

**Total ≈ 1,144 tokens** — deliberately matched to the 1,114 the Phase-1 scaffold already runs at ([[impl-atlas-0.1-phase1-scaffold]]), so memory and step-time measurements carry over.

## 3.1 Why this partition and not another

- **The near/far wake split at $x=3$** is a regime boundary, not a convenience. Upstream of $\approx3D$ the wake is dominated by **shear-layer roll-up** and the velocity deficit is roughly top-hat; downstream it is **self-similar and Gaussian**, dominated by turbulent entrainment. Two local regimes, one governing family — which per [[expert-library-atlas-0.1]]'s cut rule is *one expert conditioned continuously*, not two experts. Getting a single frozen expert to handle both, distinguished only by conditioning, is a live test of that invariant.
- **$B^\pm$ are separate agents rather than one two-component agent.** A single agent with two disconnected components would be the first non-simply-connected agent in Atlas and would exercise untested tokenizer and pooling behaviour. Splitting them avoids that unknown — and buys a free diagnostic: under symmetric inflow, $B^+$ and $B^-$ must remain mirror images, so **symmetry breaking is a detectable bug** with no reference data required.
- **The bypass is an agent at all** because wake recovery *is* momentum entrainment from the bypass flow. A wake decomposition without a bypass neighbour cannot recover and will hold a permanent deficit. This is the partition's most load-bearing choice and its most visible failure mode.
- **The rotor is a strip, not a line.** A zero-thickness agent has no interior tokens and no way to hold state. $\Delta_d=0.1$ gives it an upstream face, a downstream face, and a well-posed body-force density. Its lateral faces ($|y|=0.5$, length $0.1$) are $2\%$ of its perimeter, sit inside the wake agent's notch, and are **not declared as edges** — a deliberate omission, recorded here so it is a decision rather than an oversight.

## 3.2 Per-agent conditioning vector

Per [[agent-definition-atlas-0.1]], each agent carries its own conditioning token:

$$z_i=\big[\;\log\mathrm{Re}_{\text{eff}},\;\; I_{\text{loc}},\;\; \Delta t_i,\;\; \chi_{\text{wake}},\;\; \chi_{\text{disk}}\;\big]$$

with $I_{\text{loc}}$ the local turbulence intensity, and $\chi_{\text{wake}},\chi_{\text{disk}}$ one-hot role flags. **$\chi$ flags are the only place the agent's identity enters** — the weights are shared across all five fluid agents.

---

# 4. Governing equations

## 4.1 F1 — two-dimensional incompressible Navier–Stokes (five agents)

$$\boxed{\;\partial_t\mathbf u+(\mathbf u\cdot\nabla)\mathbf u=-\nabla p+\frac{1}{\mathrm{Re}_{\text{eff}}}\nabla^2\mathbf u+\mathbf f,\qquad \nabla\cdot\mathbf u=0\;}$$

with $\mathbf f=\mathbf 0$ in $I,N,F,W,B^\pm$; the body force is nonzero only inside the rotor strips (§4.2). In vorticity form, which is how the fluid expert internally represents the state,

$$\partial_t\omega+(\mathbf u\cdot\nabla)\omega=\frac{1}{\mathrm{Re}_{\text{eff}}}\nabla^2\omega+\nabla\times\mathbf f,\qquad \omega=\partial_xu_y-\partial_yu_x,$$

and where a stream function is used, $\mathbf u=\nabla^\perp\psi=(\partial_y\psi,\,-\partial_x\psi)$, so that $\nabla^2\psi=-\omega$.

**The stream-function representation is not incidental — §7.2 shows it makes mass conservation across every interface exact and free.** [[noether-1.0-rbc]] has this head natively; [[decoder-1.1]] retired it as the default but retains it as an optional structure-preserving decode mode. **This spec requires that mode to be engaged.** §7.2 gives the fallback if the selected checkpoint cannot.

## 4.2 F2 — actuator-disk momentum theory (two agents)

The rotor is replaced by a permeable surface that removes streamwise momentum without resolving blades. In the classical freestream-referenced form, with $A$ the swept width ($=1$ in nondimensional units, the rotor diameter):

$$U_d=U_\infty(1-a),\qquad U_w=U_\infty(1-2a),$$
$$C_T=4a(1-a),\qquad C_P=4a(1-a)^2,\qquad T=\tfrac12\rho AU_\infty^2C_T .$$

$C_P$ is maximized at $a=1/3$, giving the **Betz limit**

$$\boxed{\;C_P^{\max}=\tfrac{16}{27}\approx0.5926\;}$$

**The freestream form is unusable for turbine 2** and this matters more than it looks: $U_\infty$ is not defined inside a wake, and using the domain inlet value would make turbine 2's thrust independent of the wake it sits in — silently destroying the coupling the case study exists to test. The spec therefore uses the **local-induction (Calaf/Meyers) form**, standard in actuator-disk CFD:

$$\boxed{\;C_T'=\frac{C_T}{(1-a)^2}=\frac{4a}{1-a},\qquad T=\tfrac12\rho AC_T'\,\langle U_d\rangle^2,\qquad P=T\,\langle U_d\rangle\;}$$

where $\langle U_d\rangle$ is the disk-averaged streamwise velocity **measured by the fluid expert on the rotor's upstream face**. Every turbine now reads its own local inflow, and turbine 2 automatically produces less because its $\langle U_d\rangle$ is lower. At $a=1/3$, $C_T'=2$.

The body force distributed over the rotor strip:

$$\mathbf f_{\text{disk}}(\mathbf x)=-\frac{T}{A\,\Delta_d}\,\hat{\mathbf x}\,\mathbb 1[\mathbf x\in R_i]\;=\;-\frac{C_T'\langle U_d\rangle^2}{2\Delta_d}\,\hat{\mathbf x}\,\mathbb 1[\mathbf x\in R_i].$$

**Validity limit:** momentum theory breaks down for $a\gtrsim0.4$ (the turbulent-wake state, where an empirical $C_T$ correction is required). The spec stays inside $a\le0.35$ and treats $a>0.4$ as out of scope rather than modelling it.

**Parameter count of this expert: zero.** It is algebraic, exact, and sits at **level 5 of the physics-encoding spectrum** — the same placement [[expert-library-atlas-0.1]] gave the rocket's `rigid_body` expert, and for the same reason: there is no point spending model capacity on something with a closed form.

## 4.3 What is *not* modelled

No blade aerodynamics, no tip vortices, no tower shadow, no nacelle, no generator, no structural response, no atmospheric stratification, no Coriolis, no terrain. Each of these is an *agent to add later* (Phase G of [[case-study-wind-farm-wake-2d-atlas-0.1]]), not a term to bolt on.

---

# 5. The typed edge list

Fifteen edges over eight agents. Every edge is **declared and geometric** — Mechanism A of [[edge-generation-atlas-0.1]]; no RL instantiation in v0.

| # | Edge | Interface curve | Ports |
|---|---|---|---|
| 1 | $I\!-\!R_1$ | $x=0$, $\lvert y\rvert\le0.5$ | `MECH`, `ADVEC` |
| 2 | $I\!-\!N$ | $x=0$, $0.5<\lvert y\rvert\le2$ | `MECH`, `ADVEC` |
| 3 | $I\!-\!B^+$ | $x=0$, $2<y\le4$ | `MECH`, `ADVEC` |
| 4 | $I\!-\!B^-$ | $x=0$, $-4\le y<-2$ | `MECH`, `ADVEC` |
| 5 | $R_1\!-\!N$ | $x=0.1$, $\lvert y\rvert\le0.5$ | `MECH`, `ADVEC` |
| 6 | $N\!-\!F$ | $x=3$, $\lvert y\rvert\le2$ | `MECH`, `ADVEC` |
| 7 | $N\!-\!B^+$ | $y=+2$, $0<x\le3$ | `MECH` (tangential) |
| 8 | $N\!-\!B^-$ | $y=-2$, $0<x\le3$ | `MECH` (tangential) |
| 9 | $F\!-\!B^+$ | $y=+2$, $3<x<7$ | `MECH` (tangential) |
| 10 | $F\!-\!B^-$ | $y=-2$, $3<x<7$ | `MECH` (tangential) |
| 11 | $F\!-\!R_2$ | $x=7$, $\lvert y\rvert\le0.5$ | `MECH`, `ADVEC` |
| 12 | $F\!-\!W$ | $x=7$, $0.5<\lvert y\rvert\le2$ | `MECH`, `ADVEC` |
| 13 | $R_2\!-\!W$ | $x=7.1$, $\lvert y\rvert\le0.5$ | `MECH`, `ADVEC` |
| 14 | $W\!-\!B^+$ | $y=+2$, $7<x\le18$ | `MECH` (tangential) |
| 15 | $W\!-\!B^-$ | $y=-2$, $7<x\le18$ | `MECH` (tangential) |
| — | $R_1$ shaft, $R_2$ shaft | lumped | **`ROT`, unconnected** |

*Relabelled 2026-08-19 to [[port-algebra-atlas-0.1]]'s closed port set. Previously `momentum`, `mass`, `shear`, `conservation`.* Three consequences:

- **`shear` disappears.** It was `MECH` with a tangential normal — the tangential and normal components of one traction vector, not two quantities. This resolves open question 1 of §13 structurally rather than by argument.
- **`conservation` disappears.** Every port conserves by construction; the disk faces are distinguished not by a type label but by the fact that **one side is closed-form**, which is what makes their residual *enforceable* rather than merely measurable (§7).
- **Two `ROT` ports appear that the old vocabulary could not express.** Each rotor extracts shaft power that leaves the model. Under the old scheme that was an absence; as an unconnected port it is a **measurable power flow**, and rung 7 of [[f1-pathmap-and-end-goal]] connects it to a generator rather than integrating one. The isothermal incompressible setting means **no `THERM` port anywhere** and `ADVEC` degenerates to volumetric flux at constant $\rho$.

## 5.1 Structural properties worth naming

- **The graph has cycles.** $I\to N\to F\to W$ and $I\to B^+\to$ (via shear edges) $\to W$ are two distinct paths between the same agents. Information can circulate. The rocket's graph was near-chain; this one is not, and it is the first Atlas graph where **path-dependence of message passing is a real concern**. Agent-graph diameter is 4 ($I\to R_1\to N\to F\to R_2$), so $n_{\text{mp}}=4$ carries over unchanged from the scaffold.
- **`shear` is a new edge-type label.** It carries transverse momentum flux across a wake/bypass boundary. Per [[expert-library-atlas-0.1]]'s cut rule it is **not** a new expert: the same incompressible-NS system produces it, so it is an output channel.
- **The four `conservation`-typed edges are all disk faces.** They are the only interfaces where one side computes the crossing quantity *exactly* — which is what makes §7.3's check two-sided and sharp.
- **Every declared edge has a non-degenerate geometric interface**, and the import-time assertion from the Phase-1 scaffold (every curve lies within $\epsilon_{\text{tol}}$ of *both* agents' boundaries) must pass unchanged. Note the near-miss the rocket hit: edges 2 and 12 span only the annular part of the plane *outside* the rotor, and declaring them over the full $|y|\le2$ would repeat exactly the $d\!-\!g$ bug recorded in [[atlas-0.1-implementation-log]] on 2026-08-09.

---

# 6. The interface-quantity contract

Each edge materializes a set of boundary tokens carrying specified quantities. With $\mathbf n$ the unit normal from the first-named agent to the second:

| Port | Effort $e$ | Flow $f$ | $e\cdot f$ | Mapping |
|---|---|---|---|---|
| `MECH` | traction $\mathbf t=\boldsymbol\sigma\!\cdot\!\mathbf n=-p\mathbf n+\tfrac{1}{\mathrm{Re}_{\text{eff}}}(\nabla\mathbf u+\nabla\mathbf u^{\!\top})\!\cdot\!\mathbf n$ | velocity $\mathbf v$ | mechanical power density | effort **consistent**, flow **conservative** |
| `ADVEC` | (carrier) | mass flux $\phi_m=\mathbf u\cdot\mathbf n$ | $h_0\phi_m$ | **conservative** |
| `ROT` | torque $\tau$ | angular velocity $\omega$ | shaft power | lumped, unconnected here |

The **conservative vs. consistent** column is preCICE's distinction and Atlas was previously making it implicitly, per edge, by accident: a *flow* must map so its **integral** over the interface is preserved; an *effort* must map so its **pointwise values** are preserved. Getting this backwards is a classic partitioned-coupling bug.

**Bidirectionality.****Bidirectionality.** `momentum` and `mass` edges are symmetric: both agents read and write. The `conservation` edges at disk faces are **asymmetric** — the disk expert *writes* $T$, the fluid expert *reads* it as a body force and *writes back* $\langle U_d\rangle$, which the disk expert reads. This is a genuine two-way algebraic–PDE coupling and it is resolved per macro-step by the fixed-point iteration of §8.3.

---

# 7. Conservation laws and how each is enforced

This is the section the case study exists for. Four laws, at three different points on the physics-encoding spectrum, with a different enforcement mechanism each.

## 7.1 The laws, stated as integrals

Let $\Gamma$ be any interface curve and $V$ any agent or union of agents with boundary $\partial V$.

**(C1) Mass.** For incompressible flow, over every agent and every union of agents:
$$\oint_{\partial V}\mathbf u\cdot\mathbf n\;ds=0 .$$

**(C2) Streamwise momentum across a turbine.** For the control volume $V_i$ spanning a rotor from its upstream to its downstream face:
$$\underbrace{\oint_{\partial V_i}\big[(\mathbf u\cdot\mathbf n)u_x+p\,n_x-\tfrac{1}{\mathrm{Re}_{\text{eff}}}\,\partial_nu_x\big]ds}_{\textstyle \Phi_x(V_i)\;=\;\text{net streamwise momentum outflux}}\;=\;-\,T_i .$$
In words: **the momentum the flow loses equals the thrust the rotor gains.** This is the interface contract in its most testable form, because the left side is computed by the `fluid` expert and the right side by the `disk` expert, independently.

**(C3) Global momentum balance.** Over the whole domain,
$$\Phi_x(\Omega)=-\sum_i T_i ,$$
which must hold as a consequence of C2 plus exact interface matching — so any discrepancy localizes a leak to a specific fluid–fluid edge.

**(C3b) Global power residual.** Because every port is an effort–flow pair, the whole graph admits one scalar diagnostic ([[port-algebra-atlas-0.1]] §6):
$$\mathcal R(t)=\sum_{i\in\mathcal A}\frac{dE_i}{dt}+\sum_{\Gamma\in\mathcal P}\int_\Gamma (ef)_\Gamma\,ds-\sum_i\mathcal D_i-P_{\text{ext}},\qquad P_{\text{ext}}=\sum_i P_i\ \text{(the two open \texttt{ROT} ports)}.$$
The extracted shaft power leaves through the unconnected `ROT` ports, so it appears explicitly in $P_{\text{ext}}$ rather than vanishing. **Report $\mathcal R(t)$ every macro-step.**

**(C4) Energy extraction bound.** For each turbine,
$$P_i=T_i\langle U_d\rangle_i\quad\text{and}\quad C_{P,i}=\frac{P_i}{\tfrac12\rho AU_{\text{ref},i}^3}\le\frac{16}{27},$$
with $U_{\text{ref},i}$ the *undisturbed* inflow for turbine $i$ — for turbine 2 that is the local wake velocity, not the domain inlet, and the choice must be stated in any reported number.

## 7.2 Enforcing C1 — mass, at **level 5** (hard, by construction)

**The result that makes this free.** Across a vertical cut $\Gamma=\{x=x_0,\;y\in[y_a,y_b]\}$, with $\mathbf u=\nabla^\perp\psi$:

$$\int_{y_a}^{y_b}u_x\,dy=\int_{y_a}^{y_b}\partial_y\psi\,dy=\psi(x_0,y_b)-\psi(x_0,y_a).$$

**The mass flux through any interface equals the difference of the stream function at its two endpoints.** Therefore:

1. Within an agent, $\nabla\cdot\mathbf u=0$ holds identically because $\nabla\cdot\nabla^\perp\psi\equiv0$ (mixed partials commute) — nothing to enforce.
2. Across an interface, C1 reduces to a **two-scalar continuity condition**: the two agents must agree on $\psi$ at the two endpoints of the shared curve. Enforce by construction — the edge layer writes a single shared $\psi$ value at each **interface endpoint node**, and both agents read it.

So mass conservation over the entire eight-agent graph collapses to agreement on $\psi$ at the interface endpoints — a handful of scalars, enforced exactly, at zero cost. **[AI Inference]:** this appears to generalize to any 2D incompressible Atlas decomposition, and is a stronger statement than [[conservation-as-constraint-atlas-0.1]] currently makes for the mass channel; it is worth promoting to that page if it survives implementation.

**Fallback if the chosen checkpoint cannot expose $\psi$.** [[decoder-1.1]] made the direct-increment head the default and the stream-function head optional. If the optional head is unavailable, mass drops to **level 4**: apply a Leray projection $\mathbf u^*=\mathbf u-\nabla\Delta^{-1}(\nabla\cdot\mathbf u)$ per agent, which requires a Poisson solve per agent per step and is the expensive path — or accept **level 2** (a soft divergence penalty) and report the residual rather than claiming conservation. *This choice must be recorded in the build log, not left implicit.*

## 7.3 Enforcing C2 — thrust/momentum at the disk faces, at **level 5** (hard, by projection)

The disk side is exact by construction: $T_i$ is algebraic. The fluid side is a neural prediction and will not match. The enforcement is a **minimum-norm correction on the fluid side that makes the match exact**, applied in stream-function space so that C1 is not destroyed in the process.

Let $\hat\psi$ be the fluid expert's predicted stream function on the rotor control volume, and

$$g(\psi)\;\equiv\;\Phi_x(V_i)[\psi]+T_i$$

the constraint residual. $\Phi_x$ is **quadratic** in $\psi$ (the convective term is quadratic; the pressure and viscous terms are linear). Seek

$$\psi=\hat\psi+\lambda\,\mathbf w,\qquad \mathbf w=\frac{\nabla_\psi g(\hat\psi)}{\lVert\nabla_\psi g(\hat\psi)\rVert^2},\qquad g(\hat\psi+\lambda\mathbf w)=0 .$$

Because $g$ is quadratic, $g(\hat\psi+\lambda\mathbf w)=\alpha\lambda^2+\lambda+g(\hat\psi)$ with $\alpha=\tfrac12\mathbf w^\top\nabla^2_\psi g\,\mathbf w$, so $\lambda$ is the **root of a scalar quadratic** — closed form, no iteration:

$$\lambda=\frac{-1+\sqrt{1-4\alpha\,g(\hat\psi)}}{2\alpha}\quad(\alpha\ne0),\qquad \lambda=-g(\hat\psi)\quad(\alpha\to0),$$

taking the root of smaller magnitude. Three properties make this the right mechanism:

- **It is exact**, not a penalty. $g=0$ after one step, to floating-point.
- **It preserves C1 identically**, because the correction is a perturbation of $\psi$ and $\nabla\cdot\nabla^\perp\equiv0$ regardless of what $\lambda$ is. *This is the reason for doing the whole thing in stream-function space.*
- **It is minimum-norm**, so among all corrections achieving the constraint it perturbs the expert's prediction least — the correct posture toward a frozen expert one is not entitled to retrain.

The same projection is applied at all four disk-face ports (edges 1, 5, 11, 13), one scalar constraint each. **These four are singled out not by a type label but by the structural fact that one side is closed-form** — the criterion of [[conservation-as-constraint-atlas-0.1]]'s enforce-or-measure rule.

## 7.4 Enforcing momentum at fluid–fluid edges, at **level 2–4** (soft, and honestly so)

Edges 2, 3, 4, 6, 7–10, 12, 14, 15 have **no exact side**. Both agents are neural, so there is no ground truth to project onto — only a consistency requirement that the two agents' interface values agree. Enforcement is therefore weaker by necessity:

$$\mathcal L_{\text{iface}}=\sum_{e\in\mathcal E_{\text{ff}}}w_e\left\lVert\,\Pi_n^{(A)}\big|_{\Gamma_e}-\Pi_n^{(B)}\big|_{\Gamma_e}\right\rVert^2 .$$

**But nothing is trained here** — the experts are frozen — so this is a *diagnostic*, not a loss. It is reported per edge every step, and a systematic sign in the residual localizes which interface is leaking. **Stating this plainly is important:** Atlas's flux-matching story is exactly enforceable only where one side is closed-form, and is a measurement everywhere else. The disk edges are the exception that lets us calibrate what "good" looks like for the rest.

**[AI Inference]:** if the fluid–fluid residuals turn out comparable in magnitude to the pre-projection disk residuals, that is evidence the frozen expert is self-consistent across cuts and the declared-edge contract works. If they are much larger, the mismatch is in the edge layer's interpolation, not in the expert.

## 7.5 Enforcing C4 — the Betz bound, at **level 5** (hard, by parameterization)

$C_P=4a(1-a)^2$ with $a\in(0,0.4]$ **cannot exceed $16/27$** — the bound is a property of the formula, not a constraint applied to it. Enforcement is therefore by parameterization: the disk expert never outputs $C_P$ directly, only $a$, and clamps $a\le0.35$.

The bound then becomes a **test of the composed system rather than of the disk expert**: if the *measured* power (from the fluid-side momentum deficit times disk velocity) exceeds $16/27$ of the available flux, the coupling is generating energy, and the projection of §7.3 has been wired with a sign error or the wrong control volume. It is a canary, not a constraint.

## 7.6 Summary — the enforcement ledger

| Law | Where | Spectrum level | Mechanism | Exact? |
|---|---|---|---|---|
| C1 mass, intra-agent | all fluid agents | 5 | $\mathbf u=\nabla^\perp\psi$ | yes, identically |
| C1 mass, cross-interface | all `ADVEC` ports | 5 | shared $\psi$ at interface endpoints | yes |
| C2 momentum | 4 disk-face `MECH` ports | 5 | min-norm quadratic projection in $\psi$ | yes, one closed-form step |
| C2/C3 momentum | 11 fluid–fluid `MECH` ports | 2 (diagnostic) | reported residual; no training to penalize it | no |
| C3b power residual $\mathcal R$ | whole graph | — (diagnostic) | computed from port values alone | measured every step |
| C4 Betz | 2 rotors (`ROT`) | 5 | parameterization in $a$ | yes, by construction |

---

# 8. Time stepping

## 8.1 Timescales

| Process | Nondimensional time | Physical |
|---|---|---|
| Convection across one rotor diameter | $1$ | 13.3 s |
| Domain transit | $24$ | 5.3 min |
| Wake meandering period | $\sim10$–$30$ | 2–7 min |
| Rotor response to inflow change | $\ll1$ (algebraic) | instantaneous |

Two orders of magnitude, against the rocket's four. **Multi-rate subcycling is therefore not required in v0** and is deferred to Phase F, where it becomes a deliberate stress test rather than a necessity. $\Delta t_{\text{macro}}=0.05$ uniformly (CFL $\approx0.5$ at the finest token spacing of $0.25$ under unit velocity).

## 8.2 The disk expert has no timestep

It is algebraic: given $\langle U_d\rangle$ at time $t$, it returns $T(t)$ at time $t$. It is evaluated once per macro-step, and it introduces no stability constraint of its own.

## 8.3 Resolving the two-way coupling within a macro-step

$T$ depends on $\langle U_d\rangle$, which depends on $\mathbf f_{\text{disk}}$, which depends on $T$. Resolve by **fixed-point iteration under relaxation**, per macro-step:

$$T^{(k+1)}=(1-\theta)\,T^{(k)}+\theta\cdot\tfrac12\rho AC_T'\big\langle U_d\big\rangle^2\!\big[\mathbf f(T^{(k)})\big],\qquad \theta=0.5 .$$

Initialize $T^{(0)}$ from the previous macro-step. Converge to $|T^{(k+1)}-T^{(k)}|/T<10^{-4}$, typically 2–4 iterations because the map is a contraction for $a\le0.35$. **Convergence failure is itself a gate** — it would indicate the fluid expert responds non-monotonically to the body force, which is a physical-plausibility failure worth catching early.

## 8.4 Rollout horizon

$\tilde T_{\text{end}}=60$ (about 13 min, $\approx2.5$ domain transits, 1,200 macro-steps) — long enough for the wake to establish, reach the downstream rotor, and pass through several meandering periods.

---

# 9. Boundary conditions of the composite

These are the *outer* boundaries of the union of all agents — not edges, and not the concern of any expert individually.

| Boundary | Condition |
|---|---|
| Inlet, $x=-6$ | $u_x=U(y,t)$, $u_y=0$. Base case uniform $U=1$; **gust case** $U(t)=1+0.15\tanh\!\big((t-20)/2\big)$ to force unsteadiness (§11 risk 4 of [[case-study-wind-farm-wake-2d-atlas-0.1]]) |
| Outlet, $x=18$ | convective outflow $\partial_t\mathbf u+U_c\partial_x\mathbf u=0$, $U_c=1$ |
| Lateral, $\lvert y\rvert=4$ | slip / symmetry, $u_y=0$, $\partial_yu_x=0$ |

**No ground boundary.** This is the whole point of the hub-height plan view: the wall-bounded shear that would be the dominant out-of-distribution feature for a frozen expert trained on homogeneous 2D turbulence is simply absent from the slice. The lateral slip condition additionally means the domain is closer to the periodic-like setting the expert was trained in than a no-slip box would be.

**There is no global uniform field.** Unlike the rocket's gravity ([[global-fields-and-topology-atlas-0.1]]), nothing here bypasses the edge mechanism. Gravity enters only through stratification, which §4.3 defers to Phase G.

---

# 10. Declared / learned / closed-form ledger

Per [[00-atlas-0.1-overview]] invariant 1 and the physics-encoding spectrum:

| Component | Status | Level |
|---|---|---|
| Agent set, agent domains | **declared** | — |
| Edge list, **ports**, interface curves | **declared** (Mechanism A, [[port-algebra-atlas-0.1]]) | — |
| Expert assignment (fluid vs. disk) | **declared**, MLP gate is a fixed 2-way map | — |
| Fluid dynamics inside an agent | **learned, frozen** — no training in this case study | 1–2 |
| Rotor thrust, power, induction | **closed form** | 5 |
| Mass conservation | **closed form** (stream function) | 5 |
| Thrust/momentum matching at disk faces | **closed form** (projection) | 5 |
| Momentum matching at fluid–fluid edges | **measured**, not enforced | 2 |
| Betz bound | **closed form** (parameterization) | 5 |

Two experts. One frozen and used five times; one with zero parameters used twice. **Nothing in this case study is trained.**

---

# 11. Acceptance gates

Milestones in the style of [[00-atlas-0.1-implementation-plan]]. Numbers are targets to be revised on first measurement, not results.

| ID | Gate | Criterion |
|---|---|---|
| **W0** | Scaffold imports | All 15 interface curves within $\epsilon_{\text{tol}}$ of both agents' boundaries; materialized agent-pair set equals the declared set exactly |
| **W1** | Disk expert unit test | $C_P^{\max}=16/27$ at $a=1/3$ to machine precision; $C_T(a)$ curve matches theory; $C_T'=2$ at $a=1/3$ |
| **W2** | Frozen expert bring-up | Unforced uniform-flow rollout stable to $\tilde t=60$ with no drift; $\lVert\nabla\cdot\mathbf u\rVert_\infty<10^{-10}$ (level-5 path) or $<10^{-3}$ (fallback) |
| **W3** | Mass conservation | $\lvert\oint_{\partial V}\mathbf u\cdot\mathbf n\rvert<10^{-8}$ for every agent and every union, all steps |
| **W4** | **Thrust agreement** (primary gate) | $r_T=\lvert T_{\text{disk}}-\lvert\Phi_x(V_i)\rvert\rvert/T_{\text{disk}}<5\%$ *before* projection; $<10^{-8}$ after. **Report both** — the pre-projection number is the honest measure of whether the experts agree, and the post-projection number only proves the projection works |
| **W5** | Coupling convergence | Fixed-point iteration converges in $\le6$ steps at every macro-step, all configurations |
| **W6** | Betz canary | Measured $C_P\le16/27$ at both turbines, all steps |
| **W7** | Symmetry | Under symmetric inflow, $\lVert \mathbf u^{B^+}-\mathcal M\mathbf u^{B^-}\rVert/\lVert\mathbf u\rVert<10^{-6}$ ($\mathcal M$ = mirror) |
| **W8** | Wake recovery | Far-wake deficit collapses to a self-similar Gaussian; centreline recovery inside the Jensen ($k=0.075$) / Bastankhah–Porté-Agel corridor |
| **W9** | Decomposed vs. monolithic | Eight-agent rollout RMSE against the same frozen expert run undivided, below the expert's own single-step error |
| **W10** | Array efficiency | $P_2/P_1\in[0.4,0.8]$ at $7D$ — **a wide band on purpose**, read as a sanity check under the 2D discount of §2.3, never as a prediction |
| **W11** | Global power residual | $\lvert\mathcal R(t)\rvert$ below 1% of total extracted power, all steps, with a per-port breakdown reported |

**W4 and W9 are the two that matter.** W4 asks whether two experts computing the same physical quantity by different means agree; W9 asks whether decomposing a system into agents costs anything. Every other gate is either a precondition or a canary.

---

# 12. Failure modes and what each would mean

| Symptom | Most likely cause | What it would mean for Atlas |
|---|---|---|
| Wake never recovers; permanent deficit | shear edges 7–10, 14–15 carrying no effective entrainment | The declared-edge contract cannot transmit turbulent transport, only mean flux — a **fundamental** finding about edge typing, not a bug |
| $r_T$ large and one-signed before projection | fluid expert misreads the body force; or control volume mismatched to the strip | Adapter/geometry error; fixable |
| $r_T$ large and noisy | frozen expert genuinely out of distribution on the forced strip | Argues for halo/Schwarz exchange over flux-BC tokens — the R1-vs-R2 decision |
| Fixed-point iteration diverges | non-monotone response of the expert to $\mathbf f$ | The frozen expert is not usable as a coupled component without fine-tuning; the most serious possible outcome |
| Symmetry breaks (W7) | path-dependent message passing around the graph cycle (§5.1) | The cycle matters; would motivate ordering constraints or symmetrized message passing |
| $C_P>16/27$ | sign error in the projection, or double-counted momentum removal | Caught immediately by W6; the reason the canary exists |
| Composition error grows super-linearly in $N$ (Phase F) | error accumulation per interface | **The framework does not scale**, and that is the headline result of the whole case study |

---

# 13. Open questions

1. ~~**Does the `shear` edge type need a distinct treatment from `momentum`?**~~ **Resolved 2026-08-19 by [[port-algebra-atlas-0.1]]:** no. Both are components of one traction vector $\mathbf t=\boldsymbol\sigma\!\cdot\!\mathbf n$, so both are the `MECH` port. The port formulation settles this structurally rather than by argument.
2. **Should the near/far wake split exist at all?** If a single conditioned expert handles both regimes, agents $N$ and $F$ could merge — which would be positive evidence for [[expert-library-atlas-0.1]]'s cut rule and a simpler graph. Testable directly by running both partitions.
3. **Is the interface-endpoint $\psi$ agreement of §7.2 sufficient, or does the corner where three agents meet need special handling?** The points $(0,\pm2)$, $(3,\pm2)$, $(7,\pm2)$ are shared by three agents each. The rocket never had a three-agent corner.
4. **Does the projection of §7.3 accumulate bias over a long rollout?** Minimum-norm at each step is not minimum-norm over a trajectory. Measure the cumulative $\lVert\lambda\mathbf w\rVert$ against the state norm.
5. **Whether the fluid expert should be interleaved with message-passing layers** — [[atlas-0.1-implementation-log]]'s open action item 4, which this case study makes measurable for the first time.

---

## See Also

- [[case-study-wind-farm-wake-2d-atlas-0.1]] — why this scenario, the alternatives screen, and the phased plan
- [[conservation-as-constraint-atlas-0.1]] — the general treatment; §7.2's stream-function result should be promoted to it
- [[expert-library-atlas-0.1]] — the cut rule tested by open question 2; the closed-form-expert precedent the disk follows
- [[edge-generation-atlas-0.1]] — Mechanism A, used for all 15 edges; Mechanism B unlocked by Phase F's yaw axis
- [[agent-definition-atlas-0.1]] — the granularity heuristic §3.1 applies
- [[impl-atlas-0.1-phase1-scaffold]] — the built scaffold this reuses; token budget matched deliberately
- [[case-study-rocket-ascent-2d-atlas-0.1]] — the blocked scenario this substitutes for, and the source of the $d\!-\!g$ interface-declaration lesson in §5.1
- [[noether-1.0-rbc]], [[decoder-1.1]] — the stream-function head §7.2 requires
