# Navier–Stokes on body-fitted grids: the flow solver, verified

*Case study, PoC 3. New 2026-09-15. Tier 61.*

---

# 0. The result, in one paragraph

**Tier 60 built the curved grids, the overlap between them and one pressure system across all of them. This tier marches an incompressible flow on them — `atlas.cases.overset_ns` — and checks it before any car part is put on it.** The benchmark holds: the circular cylinder at $\mathrm{Re} = 100$ sheds at Strouhal number $0.169$ with mean drag coefficient $1.383$ and lift amplitude $0.338$, against $0.164$, $1.325$ and $0.28$ for a body-conforming reference and $0.169$, $1.453$ and $0.339$ for an immersed-boundary one. It sheds with a period steady to $0.001\%$ over twenty cycles, leaks $2.2\times10^{-4}$ of the free-stream flux through a contour round the overlap, and moves by under $0.1\%$ when the time step is halved or the body grid refined twice each way. **Every prediction about the benchmark held; every prediction about convergence order failed, and the failures are the useful part.** The manufactured solution converges at **first order in time** where second was predicted. A diagnosis with one change per arm put that on the projection, not on advection, the rotational term or the overlap: the collocated pressure Laplacian and the divergence of the gradient it corrects with are different operators (W271). Building the pressure operator as the exact divergence-of-correction restores second order on one grid, and on the composite the march **blows up** from the overlap in thirty-odd steps. With the time error pushed two orders down, the velocity converges at second order in space ($2.52$ background, $1.81$ body). The pressure converges at $1.22$ and $1.39$, under the $1.5$ predicted. Two failures on the way were found and repaired before the registered run: a pinned pressure cell that turned a small incompatibility into a point source, and a first-order edge condition. The cost prediction was not measured: the laptop was on battery when it could have run, and the judge had to be corrected to say so.

---

# 1. What this tier is for

On 2026-09-14 the user decided that RaceLab's windows must follow the car ([[poc3-racelab-body-fitted-grids]], §1). Tier 60 laid the foundation and verified it on an elliptic problem. What it could not say is whether a *flow* stays sound on those grids — whether the overlap preserves a uniform stream, leaks mass, or drives the divergence up — and at what order a collocated scheme converges on them. This tier answers that on two problems with known answers, before Tier 62 puts the car on the grids.

- **A manufactured solution with a moving wall** (§4 to §6): an exact velocity and pressure, forced into the equations, marched on Tier 60's verification geometry at three resolutions and four time steps.
- **The circular cylinder at $\mathrm{Re} = 100$** (§7): a published benchmark, with two refinements run as separate processes.

Nothing here touches RaceLab's column, its records, its gate or the car.

---

# 2. The scheme

## 2.1 The step

Velocity and pressure live at every grid point that carries an equation, on every grid. With the extrapolated advecting velocity $\mathbf u^* = 2\mathbf u^k - \mathbf u^{k-1}$ and $a_0 = 3/2$, one step of second-order backward differences (BDF2; backward Euler on a first step) with the **rotational incremental pressure correction** (Timmermans, Minev and Van De Vosse; analysed with open boundaries by Guermond, Minev and Shen) is:

$$\frac{a_0\tilde{\mathbf u} - 2\mathbf u^k + \tfrac12\mathbf u^{k-1}}{\Delta t} + (\mathbf u^*\cdot\nabla)\tilde{\mathbf u} - \nu\nabla^2\tilde{\mathbf u} = -\nabla p^k + \mathbf f^{k+1},$$

$$\nabla^2\phi = \frac{a_0}{\Delta t}\nabla\cdot\tilde{\mathbf u},\qquad \mathbf u^{k+1} = \tilde{\mathbf u} - \frac{\Delta t}{a_0}\nabla\phi,\qquad p^{k+1} = p^k + \phi - \chi\,\nu\,\nabla\cdot\tilde{\mathbf u}.$$

The intermediate velocity $\tilde{\mathbf u}$ carries every velocity boundary condition; $\partial\phi/\partial n = 0$ where velocity is prescribed and $\phi = 0$ at an outflow. The analysis asks for $\chi < 2/d$ when part of the boundary is open; the cylinder uses $\chi = 0.5$, the all-Dirichlet manufactured solution $\chi = 1$.

**Advection is implicit**, linearised about $\mathbf u^*$. A body grid clusters its rows toward the wall and spaces its columns finely round a curve, so an explicit advective step would be limited far below the background's. The implicit viscous term already costs one linear solve per component per step, and an implicit step is what the certified mode the user chose will need. The momentum system is solved by BiCGSTAB with a Jacobi preconditioner. **The pressure is factored once** (SuperLU) and back-substituted every step, as Tier 60's price decided (W269).

## 2.2 Where the equations live

Every unknown belongs to exactly one row class, and a test asserts the partition:

| class | where | momentum row | pressure-increment row |
|---|---|---|---|
| discretisation | background interior, body-grid interior | the step above, central differences through the metrics | Tier 60's Laplacian (five-point, or nine-point curvilinear) |
| wall | row 0 of a body grid | the wall's velocity (moving, if given) | Tier 60's one-sided $\partial\phi/\partial n = 0$ |
| interpolation | Tier 60's fringe and outer rings | the width-3 interpolation equation | the same equation |
| Dirichlet edge | the background's outer ring on an inflow, wall or road face | the box's velocity at the cell centre | $3\phi_e - 4\phi_1 + \phi_2 = 0$ |
| outflow edge | the outer ring on an outflow face | $u_e = (4u_1 - u_2)/3$ | $\phi_e = 0$ |

The interpolation rows couple the grids **implicitly inside every solve**, momentum and pressure alike. The box's conditions sit at the edge cells' centres, half a cell inside the geometric face — exact for a manufactured solution evaluated there, immaterial for a far field.

**A uniform stream is an exact discrete solution** when every wall moves with it: the interpolation reproduces constants and every derivative of a constant is zero. Twenty steps change it by under $10^{-12}$ (a test). The force on a body in the linear pressure $p = x$ at rest is $-\pi r^2$ to $0.5\%$ (a test), which is what every drag sign below rests on:

$$\mathbf F = -\oint \boldsymbol\sigma\cdot\mathbf n\,\mathrm ds,\qquad \boldsymbol\sigma = -p\,\mathbf I + \nu\left(\nabla\mathbf u + \nabla\mathbf u^{\mathsf T}\right),$$

with $\mathbf n$ pointing out of the fluid, Tier 60's convention.

---

# 3. Two failures found before the registered run

**A pinned cell blew the march up.** With velocity prescribed on every face, the increment $\phi$ is fixed only up to a constant, and the first version pinned it to zero in one corner cell. The interpolation equations make that pure-Neumann problem *slightly incompatible*, and the pin turned the mismatch into a point source: the manufactured solution blew up from the corner within ten steps, the divergence growing fourfold a step. The same march was stable with an outflow face (no pin) and on the background alone. **The repair adds one unknown** — a uniform source $\lambda$ on every discretisation row that absorbs the mismatch — **and one row asking for a zero mean.** Thirty steps now stay within $5\times10^{-3}$ of the exact solution (a test).

**The edge condition was first order.** The background's Dirichlet-edge row for $\phi$ was first $\phi_e = \phi_1$, and with it the background pressure's max error converged at about first order. The second-order one-sided row above replaced it, and the outflow's zero gradient likewise.

Two smaller defects went the same way: a flux routine that tested a *velocity* for being negative where it meant an *index*, and a BiCGSTAB breakdown when the starting guess already solves the system (a uniform stream does) — the guess is now accepted, and a genuine breakdown falls back to GMRES.

---

# 4. The manufactured solution

## 4.1 The problem

From $\psi = A\sin(ax + b_0)\sin(cy + d_0)\cos\omega t$ the velocity $\mathbf u = (\psi_y, -\psi_x)$ is divergence-free, and with $p = B\cos(ex)\sin(gy)\sin\omega t$ a body force makes the pair an exact solution:

$$\mathbf f = \partial_t\mathbf u + (\mathbf u\cdot\nabla)\mathbf u + \nabla p - \nu\nabla^2\mathbf u,$$

with $A = 0.5$, $a = 1.1$, $b_0 = 0.3$, $c = 0.9$, $d_0 = -0.2$, $\omega = 2$, $B = 0.3$, $e = 0.8$, $g = 1.3$, $\nu = 0.01$. The geometry is Tier 60's: a $4\times3$ box, a circle of radius $0.3$ with an O-grid $0.45$ deep, the background at $h = 1/(16\cdot2^{L-1})$. **The wall moves with the exact velocity**, every box face carries it, and BDF2 starts from the exact fields at $t = 0$ and $t = -\Delta t$. The pressure is compared after removing its mean.

## 4.2 Space (S1 to S3), at $\Delta t = 0.002$ to $t = 0.4$

| | L2 | L3 | L4 | orders |
|---|---|---|---|---|
| unknowns | $12{,}888$ | $51{,}344$ | $204{,}996$ | |
| background velocity, max | $2.95\times10^{-4}$ | $8.80\times10^{-5}$ | $5.14\times10^{-5}$ | $1.75$, $\mathbf{0.78}$ |
| body velocity, max | $3.36\times10^{-4}$ | $8.45\times10^{-5}$ | $2.71\times10^{-5}$ | $1.99$, $1.64$ |
| background pressure, rms | $7.30\times10^{-4}$ | $2.33\times10^{-4}$ | $8.89\times10^{-5}$ | $1.65$, $1.39$ |
| body pressure, rms | $2.52\times10^{-4}$ | $6.85\times10^{-5}$ | $1.21\times10^{-5}$ | $1.88$, $2.50$ |
| background pressure, max off the edge cells | $2.29\times10^{-3}$ | $1.21\times10^{-3}$ | $8.08\times10^{-4}$ | $0.92$, $0.58$ |
| body divergence, max | $2.91\times10^{-5}$ | $9.44\times10^{-6}$ | $1.56\times10^{-5}$ | $1.62$, $-0.73$ |

**S1, S2 and S3 fail.** At L4 the background's largest velocity error sits beside the box's corner, $(0.020, 0.012)$, and **the background's divergence grows with refinement** — $4.7\times10^{-5}$, $8.9\times10^{-5}$, $1.6\times10^{-4}$ — which nothing here diagnoses (W272). §5 shows what holds the velocity orders down.

## 4.3 Time (T1, T2), on the L3 grid

Four steps, $\Delta t = 0.04, 0.02, 0.01, 0.005$, to $t = 0.4$; orders from successive differences, so the spatial error cancels.

| | pair 1 | pair 2 | pair 3 | orders |
|---|---|---|---|---|
| velocity, max | $6.50\times10^{-4}$ | $3.51\times10^{-4}$ | $1.83\times10^{-4}$ | $\mathbf{0.89}$, $\mathbf{0.94}$ |
| velocity, rms | $2.36\times10^{-4}$ | $1.11\times10^{-4}$ | $5.39\times10^{-5}$ | $1.09$, $1.04$ |
| pressure, rms | $3.39\times10^{-4}$ | $1.25\times10^{-4}$ | $6.86\times10^{-5}$ | $1.44$, $0.86$ |

**First order, where BDF2 with the rotational correction was predicted second.** T1 and T2 fail. At $\Delta t = 0.002$ the first-order temporal error is near $7\times10^{-5}$, the size of L4's spatial error, which is why S1 read a velocity order under one.

---

# 5. Why first order in time (W271)

Every arm below was first seen in an exploration and is recorded, with what was read before it, in `READ_BEFORE_DIAGNOSIS`. Each changes one thing; orders are from successive differences at $\Delta t = 0.02, 0.01, 0.005$ to $t = 0.4$.

| arm | grid | velocity order, max / rms |
|---|---|---|
| as built | background alone, L2 | $0.98$ / $1.00$ |
| without the rotational term ($\chi = 0$) | background alone | $0.95$ / $0.96$ |
| **BDF2 momentum with the exact pressure gradient, no projection** | background alone | $\mathbf{1.96}$ / $\mathbf{1.96}$ |
| **pressure operator = the exact divergence of the correction** | background alone | $1.59$ / $\mathbf{1.93}$ |
| as built | composite, L3 | $0.94$ / $1.04$ |
| the compact projection repeated five times a step | composite | $1.10$ / $1.23$ |
| the exact divergence-of-correction | composite | **blew up at step 38** |

The time integration is second order: with the exact pressure and no projection it reads $1.96$. The rotational term does not decide the order, and in the exploration neither did advection (Stokes read $0.94$). **The projection does.** On a collocated grid the increment is solved with the compact Laplacian $L$, and the correction applies the wide operator $D\,G$ — the central divergence of the central gradient. The two differ, so the discrete velocity is not the projection the analysis's second order assumes:

$$D\,\mathbf u^{k+1} = D\tilde{\mathbf u} - \frac{\Delta t}{a_0}D\,G\,\phi = \frac{\Delta t}{a_0}\left(L - D\,G\right)\phi \neq 0.$$

Building the pressure operator as $D\,G$ itself — the gradient at discretisation points, zero on walls and edges, re-interpolated at interpolation points — makes the projection exact and restores second order on the background. **On the composite that operator is unstable.** At L2 the velocity error grew eightfold every five steps from the overlap, at $(1.67, 1.14)$ — not as a checkerboard, whose alternating fraction stayed under $0.03$ — and the march blew up at step 31. At L3 it blew up at step 38.

> **[AI Inference]:** two grids each held exactly divergence-free, in the discrete sense of their own stencils, over the region they share are two constraints on one velocity. The interpolation cannot satisfy both, and the mismatch has nowhere to go. Overset solvers that reach second order in time — Henshaw's, for one — solve a pressure Poisson equation for $p$ itself with a wall condition derived from the momentum equation, and never project. That is the principled route, and it is not built here.

Repeating the compact projection within the step halves the error's constant at five passes and leaves the order at $1.10$: **an approximate projection applied again converges to its own fixed point, not to the exact one.**

**What stays, and why it is acceptable for now.** The default is the compact projection: stable, second order in space for velocity, first order in time. A steady flow's answer does not depend on $\Delta t$ at all: at a fixed point $\mathbf u^{k+1} = \mathbf u^k$, the pressure update gives $\phi = \chi\nu D\tilde{\mathbf u}$ and the increment equation $\big(L - a_0/(\chi\nu\Delta t)\big)\phi = 0$, whose operator is negative definite, so $\phi = 0$ and the steady discrete equations hold exactly. Transients and shedding are first order, and §7 measures how much that costs the cylinder.

---

# 6. Space, with the time error out of the way (S4)

Registered after T1, T2 and S1 had failed, and before this stage ran: march to $t = 0.05$ at $\Delta t = 0.001$ (L2) and $0.00025$ (L3, L4), putting the temporal error near $10^{-6}$.

| | L2 | L3 | L4 | L3→L4 |
|---|---|---|---|---|
| background velocity, max | $2.23\times10^{-4}$ | $1.14\times10^{-4}$ | $1.99\times10^{-5}$ | $\mathbf{2.52}$ |
| body velocity, max | $2.33\times10^{-4}$ | $7.32\times10^{-5}$ | $2.10\times10^{-5}$ | $\mathbf{1.81}$ |
| background pressure, rms | $3.61\times10^{-4}$ | $6.36\times10^{-5}$ | $2.73\times10^{-5}$ | $1.22$ |
| body pressure, rms | $9.67\times10^{-4}$ | $1.91\times10^{-4}$ | $7.30\times10^{-5}$ | $1.39$ |

**The velocity clause holds and the pressure clause does not, so S4 fails.** With the time error removed the velocity is second order in space on both grids — the $0.78$ of §4.2 was the temporal floor. The pressure converges at $1.22$ and $1.39$, and neither maximum sits on an edge cell any more. The cause is not diagnosed (W273).

> **[AI Inference]:** at L3 the step $\Delta t = 0.00025$ is equal to $h^2 = 0.000244$, the regime in which a collocated pressure correction's implicit stabilisation of the pressure fades (the exploration saw L1 degrade badly below $h^2$). If that is what holds the pressure order, an arm at $\Delta t = 2h^2$ on each level would show it. Unmeasured.

---

# 7. The cylinder at $\mathrm{Re} = 100$

## 7.1 The set-up

A circle of diameter $D = 0.4$ in a $12\times8$ box ($30D\times20D$, $10D$ upstream, blockage $5\%$), $\nu = 0.004$, free stream $1$: $\mathrm{Re} = 100$. The background is $384\times256$ at $h = 1/32$; the body grid is $256\times33$, $0.3$ deep, rows clustered by $\beta = 2.5$ (wall spacing $0.0022$). Uniform flow on the inlet, top and bottom, an outflow on the right. It starts impulsively, with a small asymmetric bump in $v$ behind the cylinder so that shedding starts in tens of time units rather than hundreds, and is marched to $t = 120$ at $\Delta t = 0.0125$. **Statistics are read over $t\in[70, 120]$**: the period from the lift's upward zero crossings over whole cycles, the drag's mean and the lift's half peak-to-peak over the same cycles. The half-step arm and the fine-body-grid arm ($512\times65$) ran as separate processes, each into its own file.

## 7.2 What it measured

| | base | $\Delta t/2$ | body grid $\times2$ each way |
|---|---|---|---|
| unknowns | $106{,}496$ | $106{,}496$ | $131{,}328$ |
| Strouhal number | $\mathbf{0.169}$ ($0.16908$) | $0.16899$ ($-0.057\%$) | $0.16906$ ($-0.011\%$) |
| mean drag coefficient | $\mathbf{1.383}$ ($1.38254$) | $1.38159$ ($-0.069\%$) | $1.38196$ ($-0.040\%$) |
| lift amplitude | $\mathbf{0.338}$ ($0.33784$) | $0.33592$ ($-0.57\%$) | $0.33615$ ($-0.50\%$) |
| drag amplitude | $0.0096$ | $0.0094$ | $0.0098$ |
| whole cycles, period spread | $20$, $2.7\times10^{-5}$ of $2.366$ | $20$, $1.1\times10^{-5}$ | $20$, $8.2\times10^{-5}$ |
| net flux out of a background rectangle round the body grid, per unit free-stream flux through its height | $2.2\times10^{-4}$ | $2.3\times10^{-4}$ | $1.9\times10^{-4}$ |
| net flux through the body grid's middle ring, per unit $U D_\text{ring}$ | $8.8\times10^{-6}$ | $7.6\times10^{-6}$ | $2.5\times10^{-5}$ |

The mean drag over successive twenty-unit windows reads $1.3820$, $1.3823$, $1.3825$, $1.3826$ from $t = 40$: the flow was stationary well before the statistics began. The pressure part of the drag is $1.031$, three quarters of it.

**The divergence grows when the body grid is refined.** Over the statistics window the largest divergence was $9.8\times10^{-3}$ on the base arm and $2.2\times10^{-2}$ on the fine body grid — the same direction as the manufactured solution's background (§4.2), and not diagnosed (W272). Halving the step halved it ($4.7\times10^{-3}$).

**Against the references** (read before the bands were written):

| | $C_D$ | $C_L'$ | $\mathrm{St}$ | method |
|---|---|---|---|---|
| Ding et al. (2004) | $1.325$ | $\pm0.28$ | $0.164$ | body-conforming |
| a virtual-interpolation-point method | $1.328$ | $\pm0.31$ | $0.164$ | body-conforming |
| Tseng and Ferziger (2003) | $1.42$ | $\pm0.29$ | $0.164$ | immersed boundary |
| Uhlmann (2005) | $1.453$ | $\pm0.339$ | $0.169$ | immersed boundary |
| **this solver** | $\mathbf{1.383}$ | $\pm\mathbf{0.338}$ | $\mathbf{0.169}$ | overset, body-fitted |

The first two are quoted from arXiv:1401.6513, Table 3, the last two from PMC9887058, Table 3. **The drag is $4\%$ above the body-conforming references and below the immersed-boundary ones; the frequency is $3\%$ above; the lift amplitude matches Uhlmann's and is $21\%$ above Ding's.** Refining time or the body grid moves none of the three by more than $0.6\%$, so the gap to Ding is not resolution of what was refined.

> **[AI Inference]:** a $5\%$ blockage raises drag and shedding frequency by a few percent — the classical corrections scale with $D/H$ — and that is the gap's likeliest single cause. The background was not refined, and the domain was not lengthened; either could also move the lift amplitude, which varies by $20\%$ across the published values themselves.

**The first-order projection costs the shedding little:** halving $\Delta t$ moved the frequency $0.06\%$ and the drag $0.07\%$. At about $190$ steps a period, a first-order temporal error in the phase is small.

## 7.3 What it cost

Setting up the base arm took $1.8$ s, of which the pressure factorisation was $1.7$ s; $9600$ steps took $2265$ s. **Those wall times were taken with three to five other Python processes running** — the other arms, the diagnosis and the user's own phone-remote server — so they are a ceiling, not a price. The momentum solve took $20$ BiCGSTAB iterations a component at the median, $42$ at most; the fine body grid needed about $45$.

---

# 8. The price at the car's size (P1)

**Not measured.** P1 claims that one flow step at Tier 60's car-size stand-in costs at most $1.0$ s **on mains**, with at most $40$ BiCGSTAB iterations a component; a dry run before P1 was written saw $0.80$ s and $41$ iterations, and P1 was left as written. The stage was to run alone once the cylinder arms had finished — and when they had, **the laptop was on battery**.

**The judge was wrong, and the stage was about to expose it.** The code that judges P1 read the step's cost and the iterations and never the power state: prose and code disagreeing in the permissive direction, as in [[poc3-racelab-outlet-and-start]]. It now returns `None` unless the machine state recorded before *and* after the price says mains. The price stays open (W274), and W269's re-measurement on mains with it.

---

# 9. The predictions

Registered at 03:03:04 before any stage ran (S1 to P1), and at 03:18:21 after T1, T2 and S1 had failed (S4).

| id | claim | outcome |
|---|---|---|
| S1 | manufactured solution, L3→L4: velocity max-error order $\ge 1.7$ on both grids | **failed** — background $0.78$, body $1.64$ |
| S2 | pressure rms order $\ge 1.5$ on both grids, and the pressure max away from the edge cells $\ge 1.5$ | **failed** — background $1.39$ and $0.58$ |
| S3 | the body grid's largest divergence falls $\ge 2\times$ per level | **failed** — $1.62$, then $-0.73$ |
| T1 | successive time-step differences: velocity order $\ge 1.8$ between the two finest pairs | **failed** — $0.89$, $0.94$ |
| T2 | pressure rms order $\ge 1.3$ between the two finest pairs | **failed** — $1.44$, $0.86$ |
| C1 | cylinder, $t\in[70,120]$: St in $[0.160, 0.175]$, $C_D$ in $[1.30, 1.42]$, $C_L'$ in $[0.28, 0.36]$ | **held** — $0.169$, $1.383$, $0.338$ |
| C2 | at least $15$ whole cycles with the period's spread under $1\%$ | **held** — $20$, $0.001\%$ |
| C3 | mean absolute net flux $\le 10^{-3}$ of its scale, through a background rectangle and through a body-grid ring | **held** — $2.2\times10^{-4}$, $8.8\times10^{-6}$ |
| C4 | half the time step: St moves $< 1\%$, $C_D$ $< 2\%$ | **held** — $0.057\%$, $0.069\%$ |
| C5 | a body grid twice as fine each way: St $< 1\%$, $C_D$ $< 2\%$ | **held** — $0.011\%$, $0.040\%$ |
| P1 | at the car-size stand-in, one step $\le 1.0$ s on mains, $\le 40$ iterations a component | **not measured** (§8) |
| S4 | time error pushed down: velocity order L3→L4 $\ge 1.7$ and pressure rms order $\ge 1.5$, both grids | **failed** — velocity $2.52$, $1.81$; pressure $1.22$, $1.39$ |

---

# 10. What this tier did NOT do, named

- **Second order in time was not reached** (W271). The exact projection is kept in the module as `projection="exact"` with its instability documented; a velocity–pressure formulation was not built.
- **The pressure's spatial order under $1.5$ (W273) and the background divergence growing with refinement (W272) are not diagnosed.**
- **The cylinder's background was not refined and its domain not lengthened**, so the $4\%$ drag gap to the body-conforming references is attributed to blockage by inference only.
- **P1 was not measured** (W274); no cost on this page was taken on an idle machine, and none of the cylinder's wall times is a price.
- **Nothing here has several bodies, a road, or a car** — Tier 62.
- **RaceLab's column, its records, the gate, the car, the demo and the bundle are unchanged.** Tier 60's `overset.py` gained a refactor that assembles the curvilinear Laplacian and the Neumann wall row in one place, so the flow solver reads the same code; its matrices are bitwise the ones Tier 60 recorded.
- **Nothing was downloaded or installed**, no machine was rented, the unlicensed structural checkpoint was not loaded, and nothing was pushed.

```
python scripts/tier61_overset_flow.py --out out/racelab10 --stages mms_space,mms_time
python scripts/tier61_overset_flow.py --out out/racelab10 --stages diagnose,space_small_dt
python scripts/tier61_overset_flow.py --out out/racelab10 --stages cylinder --arm base
python scripts/tier61_overset_flow.py --out out/racelab10 --stages cylinder --arm dt_half
python scripts/tier61_overset_flow.py --out out/racelab10 --stages cylinder --arm body_fine
python scripts/tier61_overset_flow.py --out out/racelab10 --stages price,summary
```

The record is `out/racelab10/racelab10.json`; each arm's full trajectory is in `out/racelab10/cylinder_<arm>.json`, which is not carried upstream.

---

## See Also

- [[poc3-racelab-body-fitted-grids]] — Tier 60, the grids and the pressure system this solver marches on.
- [[poc3-racelab-outlet-and-start]] — Tier 59, where a pinned outlet and a judge that disagreed with its prose were found the same way.
- [[navier-stokes-equations]] — the equations.
- [[schwarz-iteration-atlas-0.1]] — the decomposition an overset grid is, and why two exactly-projected overlapping grids over-constrain one velocity.
- [[composition-error-theory]] — where the overlap's interpolation error belongs.
- [[gap-worklist]] — W271 to W274.
