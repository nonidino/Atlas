# Body-fitted grids for RaceLab: the foundation

*Case study, PoC 3. New 2026-09-14. Tier 60.*

---

# 0. The result, in one paragraph

**The user decided that RaceLab's fluid windows must follow the car, and chose body-fitted grids — curved grids wrapped round each part, overlapping a Cartesian background, with the bodies as real walls. This tier built the foundation that choice needs and verified it before any car part is put on it: `atlas.cases.overset` generates the grids, cuts the background, finds every overlap point's donors, and assembles one pressure system across all the grids.** On a manufactured solution at four resolutions, every arm on an analytic grid with the default interpolation converges at second order between the two finest — pressure $1.99$ to $2.01$, gradient $1.92$ to $2.00$ — and the overlap costs a constant rather than an order: the composite's body-grid error is $1.82\times$ the same grid's alone. Linear fields interpolate exactly ($\le 1.1\times10^{-15}$), and a grid too thin for its overlap, a grid leaving the box and two intersecting bodies are all refused. **Fourteen predictions were registered before the run and thirteen held. The one that failed is the useful one:** on a *generated* grid round a thin ellipse the order fell to $1.74$, and a diagnosis with one change per arm put it on the generator's normal smoothing — too wide for a tip of radius $0.0375$ — not on the overlap, the solver or the outline's density. The smoothing default became $\kappa = 0.25$, and two predictions registered for it held: the same composite then converges at $1.99$. **At the car's size** — a $672\times240$ background and eight body grids, $204{,}599$ unknowns — SuperLU factors the pressure system in $3.8$ s and solves it in $0.10$ s, so by a rule written before the number existed the flow solver will solve the pressure **once per macro-step**, not every exchange. Those costs were taken on battery. Nothing here marches a flow, and nothing touches the car or its records.

---

# 1. What was decided, and what it replaces

On 2026-09-14, after reviewing what was left of PoC 3, the user decided four things, recorded in `POC3-RACELAB-REQUIREMENTS.md` §13:

| | decided | replaces |
|---|---|---|
| windows | **body-fitted grids** round each part of the car, overlapping a Cartesian background, bodies as walls | 14 boxes of $128\times128$ cells whose $x$ positions a dynamic program chooses, with every body a porous body force |
| certified mode | **live**, through an implicit fluid time step | refused as a march mode (W226) |
| parameters | powertrain and cooling knobs **and** geometry knobs | none on the page (W230) |
| 3-D | a **real Formula One CAD model**, after the 2-D work | the half-car in a box |

The windows decision is the foundational one: the certified mode and the geometry knobs both sit on the windows. The user chose body-fitted grids over two alternatives offered with their costs — shaped regions on the same lattice, and keeping the boxes for now — knowing that it needs a new solver, a grid generator and a new pressure solve, and that **Poseidon-T then runs only in background windows away from the car**, because it accepts nothing but a uniform $128\times128$ grid (W270).

**What stays.** The background is RaceLab's own lattice ($h = 1/64$, cell centres), so a background field and a RaceLab field are the same array. The devices — the radiator core and the recovery turbine — can stay body forces inside a grid, and the joins J1 to J3 are untouched.

> **[AI Inference]:** an overset grid is a Schwarz decomposition with curved subdomains — each grid solves with data interpolated from its neighbours, and the pressure couples them globally, as RaceLab's spectral projection couples its boxes today. That makes each body grid a natural *agent* and each overlap a natural *seam* for the framework's compiler ([[schwarz-iteration-atlas-0.1]]). No declaration for such a seam exists yet (W268).

---

# 2. The grids

## 2.1 The background and the body grids

A **body grid** is a structured curvilinear grid $\mathbf x(\xi,\eta)$ round one closed body: $\xi$ runs clockwise round it and is periodic, $\eta$ runs from the wall ($\eta = 0$) outward. That orientation makes the mapping right-handed,

$$J = x_\xi\,y_\eta - x_\eta\,y_\xi > 0,$$

which the constructor requires. With the contravariant metric $g^{ab} = \nabla\xi^a\cdot\nabla\xi^b$, the grid stores

$$a_{11} = J g^{11} = \frac{x_\eta^2 + y_\eta^2}{J},\qquad a_{12} = J g^{12} = -\frac{x_\xi x_\eta + y_\xi y_\eta}{J},\qquad a_{22} = J g^{22} = \frac{x_\xi^2 + y_\xi^2}{J},$$

from second-order central differences in computational space, one-sided at the two ends of $\eta$.

## 2.2 The generator, and the first version it replaced

`ogrid_from_outline` offsets a closed outline along a smoothed normal:

$$\mathbf x(s, d) = \mathbf P(s) + d\,\mathbf N(s,d),\qquad \mathbf N(s,d) = \frac{G_{\sigma(d)} * \mathbf n}{\lVert G_{\sigma(d)} * \mathbf n\rVert},\qquad \sigma(d) = \sigma_w + \kappa\,d,$$

where $\mathbf P(s)$ is the outline by arc length, $\mathbf n$ its outward normal, $G_\sigma$ a periodic Gaussian along the outline, and $d$ runs from $0$ to the grid's thickness with rows clustered toward the wall. Smoothing widens with distance from the wall, where orthogonality matters least and folding matters most. **A grid with any folded cell is refused, not repaired.**

**The mapping is defined by the outline alone**, evaluated on a fixed $16{,}384$-point resampling and then sampled. A first version marched layer by layer and smoothed each layer. It shrank a circle's grid by $1.1\times10^{-2}$ at $96\times13$, and its smoothing grew with the layer count, so **a finer grid was a different grid**, which is useless for a convergence study. The offset mapping reproduces the analytic circle's grid to $9.2\times10^{-8}$ at $512\times25$ — the sag of the $4000$-point input polygon — and a test asserts that a grid at twice the resolution, sampled at every other point, *is* the coarser grid.

---

# 3. The overlap

**Holes.** Every background point inside a body, or closer to it than a margin, is a hole. The background points beside a hole become **interpolation points**, and each body grid's outer ring is interpolation points too.

**Donors.** A background point is located in a body grid in two steps. A bilinear inversion over the cells round its nearest nodes finds the cell. Then Newton's method on the width-$w$ Lagrange map of the grid's own coordinates finds $(\xi,\eta)$. **The same stencil defines both the map Newton inverts and the weights returned**, so the weights reproduce $x$ and $y$ — hence any linear field — to round-off on a curved, twisted grid. Background donors for a body grid's outer ring are exact index arithmetic.

**Interpolation is explicit.** A donor must carry its own equation: a discretisation point, or a wall point on a body grid, never another interpolation point. When the overlap is too narrow for that, the points that cannot be served are **orphans**, and the constructor raises with the first few named. A background point tries the nearest body's grid first and then the others, because between two close bodies the grid that *contains* the point may be the other one.

**Refused, not built, in this tier:** a body grid that leaves the background box, and a body grid that reaches inside another body. The car needs both — its wheels touch the road and its parts nearly touch each other (W265, W266).

---

# 4. The pressure system

One sparse system over every point that carries an equation:

| point | equation |
|---|---|
| background, discretisation | five-point Laplacian; a box face carries Dirichlet data through the quadratic ghost $p_{\text{ghost}} = (8g - 6p_0 + p_1)/3$ |
| body grid, discretisation | $\displaystyle \Delta p = \frac{1}{J}\Big[\partial_\xi\big(a_{11}p_\xi + a_{12}p_\eta\big) + \partial_\eta\big(a_{12}p_\xi + a_{22}p_\eta\big)\Big]$, nine points, face-averaged coefficients |
| body grid, wall | Neumann, $\displaystyle \frac{\partial p}{\partial n} = -\frac{g^{12}p_\xi + g^{22}p_\eta}{\sqrt{g^{22}}}$ with $\mathbf n = -\nabla\eta/\lvert\nabla\eta\rvert$ pointing out of the fluid; $p_\eta$ one-sided second order |
| interpolation | $p_k - \sum_m w_{km}\,p_m = 0$ |

**The quadratic ghost is a correction found by measuring.** The first version used the linear ghost $2g - p_0$. That is second order for $p$, but it left the **gradient** at order $1.01$ and $1.00$ along the box, and the flow solver reads the gradient at its outlet. With the quadratic ghost the box's gradient converges at $1.99$ and $2.00$.

---

# 5. Verification

## 5.1 The manufactured solution, and its controls

$$p(x,y) = \sin(1.1x + 0.3)\cos(0.7y + 0.2) + 0.3\,e^{-2\left[(x-2)^2 + (y-1.5)^2\right]}$$

The solution sits in a $4\times3$ box with a circle of radius $0.3$ at its centre, and the body grid is $0.45$ thick. The Gaussian is centred on the body, so the solution varies where the grids overlap and at the wall. Resolutions are $h = 1/16$ to $1/128$, with the body grid refined with it. **Two single-grid controls run through the same assembly**: the box alone, and the body grid alone with Dirichlet data on its outer ring. Orders are $\log_2(e_\ell/e_{\ell+1})$ in the max norm.

| arm | pressure order, L3→L4 | gradient order, L3→L4 | L4 max error |
|---|---|---|---|
| box (control) | $2.000$ | $1.996$ | $1.01\times10^{-5}$ |
| body grid alone (control) | $1.990$ | $1.957$ | $7.56\times10^{-5}$ |
| body grid alone, twisted | $2.008$ | $1.961$ | $7.08\times10^{-5}$ |
| **composite**, width 3 | bg $1.986$, body $1.990$ | bg $1.956$, body $1.951$ | bg $1.14\times10^{-4}$, body $1.38\times10^{-4}$ |
| composite, twisted | bg $2.000$, body $2.006$ | bg $1.916$, body $1.959$ | bg $9.02\times10^{-5}$, body $1.12\times10^{-4}$ |
| composite, **width 2** | bg $1.963$, body $1.983$ | bg $\mathbf{0.793}$, body $1.950$ | bg $1.08\times10^{-4}$ |

The twisted grid rotates each row by up to $0.6$ rad, which leaves the Jacobian exactly as it was and makes $a_{12} \neq 0$. Every linear solve's relative residual is below $10^{-11}$.

**The overlap costs a constant, not an order.** At L4 the composite's body-grid error is $1.82\times$ the body grid's alone, against a registered allowance of $3\times$.

**Width 2 costs the background's gradient an order.** Bilinear interpolation is second order in value, but its error is not smooth from one fringe point to the next, so its gradient there is $O(h)$: $1.57$, $1.59$, then $0.79$. **The pressure gradient must never be taken through width-2 interpolation**, and width 3 is the default.

## 5.2 What must hold to round-off, and what must be refused

| property | measured |
|---|---|
| linear field through the interpolation, widths 2, 3, 4 | $7.8\times10^{-16}$, $1.1\times10^{-15}$, $9.4\times10^{-16}$ |
| Laplacian and wall rows annihilate constants | $3.7\times10^{-16}$ of the diagonal |
| a linear field's gradient on the twisted grid | $3.2\times10^{-14}$ |
| a body grid $0.12$ thick against a $0.15$ hole margin | **refused**, $180$ orphans |
| a body grid leaving the box | **refused** |
| two body grids each reaching into the other's body | **refused** |

---

# 6. The prediction that failed, and what it found

## 6.1 G4

The generator stage ran the composite manufactured solution on a *generated* grid round an ellipse four times as long as it is thick, turned $10°$, $0.45$ deep, with $\kappa = 1$. The prediction was order in $[1.85, 2.15]$ on both grids and a body-gradient order $\ge 1.75$. **Measured: background $1.92$, $1.88$, $1.79$; body $1.94$, $1.95$, $1.74$; body gradient $1.13$, $1.73$, $1.06$. Falsified.**

## 6.2 The diagnosis: one change per arm

Stage `diagnose` was written after the verdict, and an exploration outside the record had already run most of its arms. The record says so, and these are read as a diagnosis, not as tests.

| arm | the one change — from G4, or from the grid alone | pressure orders | where the largest error sits |
|---|---|---|---|
| G4 | — | body $1.94$, $1.95$, $1.74$ | on the tip |
| input polygon | from G4: $64{,}000$ points, not $4000$ | body $1.94$, $1.95$, $1.74$ — errors identical to four figures | on the tip |
| grid alone | from G4: no background | $2.02$, $1.88$, $1.67$, and $1.75$ at L4→L5 | on the tip |
| blunt tip | ellipse $0.45\times0.3$, tip radius $0.2$ not $0.0375$: from the grid alone, and from G4 | alone $1.95$, $1.97$, $1.99$; composite body $1.92$, $1.97$, $1.99$ | the upper side |
| thinner grid | from the grid alone: $0.15$ deep, not $0.45$ | $1.84$, $1.93$, $1.98$ | on the tip |
| gentler smoothing | from the grid alone: $\kappa = 0.25$ | $1.89$, $1.97$, $1.99$ | beside the tip |
| stronger smoothing | from the grid alone: $\kappa = 4$ | $1.14$, $1.56$, $1.01$ | the flank |

**Read together:**
- **The overlap is not the cause**: the grid alone fails the same way.
- **The outline's density is not the cause**: $64{,}000$ points change nothing.
- **The tip is where it fails**, and **the smoothing is what fails it**. On the grid alone, with nothing else changed, the order between the two finest levels follows $\kappa$ monotonically: $1.99$ at $0.25$, $1.67$ at $1$, $1.01$ at $4$. A normal field smoothed over a width comparable to a tight tip bends the grid lines hard enough there to hold the error off its asymptote, and a blunter tip or a thinner grid gives the smoothing less to bend.

## 6.3 The adopted default, and its own prediction

`overset.KAPPA = 0.25`. **Two predictions were registered for it before stage `diagnose` ran, and neither had been measured in the exploration:**

| id | prediction | measured | |
|---|---|---|---|
| D1 | the thin ellipse's composite at $\kappa = 0.25$ meets G4's own bounds | background $1.954$, $1.993$, $1.992$; body $2.052$, $2.001$, $1.994$; body gradient L3→L4 $1.915$ | **held** |
| D2 | the ellipse, plate and NACA 0012 still generate with zero folded cells, and the dent is still refused | zero folds; wall skew $0.17°$, $1.46°$, $3.96°$; dent refused ($356$ folds) | **held** |

**Every stage registered before the change still runs at $\kappa = 1$** — `tier60_body_fitted_grids.GEN_KAPPA`, and `manufactured_poisson`'s own default — so the record reproduces rather than describing a different generator. That is Tier 59's practice with the pinned outlet, one module on.

**Not measured:** how much smoothing a *concave* region of the car needs to stay unfolded. The dent folded at every $\kappa$ tried ($0$, $0.25$ and $1$), so smoothing does not rescue a tight concavity; whether $0.25$ is enough for a mild one is a question for the car's own shapes (W263).

---

# 7. The price at the car's size

Before any flow solver is designed, the pressure system is priced where it will run. **The stand-in is not the car**: two NACA sections, two wheels, a long body, a plate, a pod and a nose, kept apart so this tier's refusals do not fire. It is car-sized: a $672\times240$ background at $h = 1/64$, eight body grids $25$ rows deep and $256$ to $576$ points round, $0.3$ thick.

| | |
|---|---|
| unknowns | $204{,}599$ ($20{,}681$ background holes, $1519$ background interpolation points) |
| non-zeros | $1{,}277{,}086$ |
| SuperLU factorisation (COLAMD) | $\mathbf{3.79}$ **s**, fill $28.0$ M |
| one solve, five repeats | $0.092$–$0.114$ s, median $\mathbf{0.103}$ **s**; relative residual $3.9\times10^{-12}$ |
| smoothed-aggregation AMG + BiCGSTAB | setup $3.72$ s, $21$ iterations, $2.97$ s, residual $5.2\times10^{-11}$ |
| incomplete LU + GMRES | **failed**: *"Factor is exactly singular"* — not diagnosed |
| building the overlap | $46.8$ s; after a speed-up, $15.3$ s (median of three), point counts identical |
| power | **on battery** for all of it; a $1024^2$ matrix product took $0.023$ s before and $0.019$ s after |

**The rule, written before the number existed.** A RaceLab macro-step has four exchanges and a $0.5$ s ceiling. If one factored solve costs $\le 0.05$ s, the pressure could be solved every exchange; if $\le 0.2$ s, once per macro-step; above that, it needs an iterative warm-started solve or a smaller system. **At $0.103$ s: once per macro-step.** The factorisation is paid once per geometry, so a geometry knob pays $3.8$ s plus the overlap's rebuild.

**AMG is $29\times$ slower per solve than back-substitution** on a matrix that does not change, and `pyamg` is not a bundle dependency. It matters only if the geometry changes every step.

**The overlap's build was $47$ s, and $44$ of them were two edge loops.** The point-in-polygon test visited each body's $4000$ outline edges one at a time, and the distance test measured every candidate point, including those already inside a body. The speed-up vectorises the parity count, measures a distance only for points not already inside a body, and screens the pairwise body check by bounding box. A test pins it against the old loop **bitwise**, on points placed exactly on vertices and edges. What is left of the $15.3$ s is mostly distance-to-outline (W267).

---

# 8. What comes next, and what could stop it

| tier | builds | ends with |
|---|---|---|
| **61** | incompressible Navier–Stokes on overset grids: momentum on every grid, the pressure on the composite system with a wall condition that keeps the divergence under control, no-slip and moving walls | a cylinder at $\mathrm{Re} = 100$ against the published shedding frequency and drag, with the divergence and the mass through each overlap measured |
| **62** | the car as closed shapes with thickness; wheels as rotating walls; grids cut by each other and by the road | the drawn car marching on body-fitted grids, beside the immersed column as a comparison between two models, not a referent |
| **63** | the framework: body grids as agents, overlaps as seams, the devices and joins re-sited, background windows for the learned expert, the demo's overlay | a compile with verdicts, and the page drawing curved grids |
| then | the implicit step and live certified mode; the knobs; the gate re-run once; the bundle rebuilt and verified once; push when asked | the requirements' definition of done |
| after | the real Formula One CAD model in three dimensions | — |

**Risks, named before they bite:**
- **The pressure boundary condition.** A collocated velocity–pressure scheme on curved overlapping grids needs a wall condition for the pressure that does not let the divergence grow. This is well studied, but getting it wrong looks like a solver that works for a while.
- **Mass through an overlap.** Interpolation is not conservative, so fluid can be created or destroyed across a fringe. It will be measured, not assumed.
- **The car is drawn as plates of zero thickness.** A wall needs a closed outline. Where each part's thickness comes from is a declaration, and a drawing change (W264).
- **Wheels touch the road.** A grid cannot be wrapped round a point of contact. The options are a gap, a plinth, or cutting the wheel's grid by the road, and each is a modelling choice (W265).
- **Once per macro-step** fixes the time integration round the pressure (W269).

> **[AI Inference]:** the implicit step the certified mode needs may help the learned expert too. A step that is stable at a longer $\Delta t$ could ask Poseidon-T for something nearer the step it was trained on, instead of $1/32$ of it. Unmeasured, and it concerns the background windows only.

---

# 9. The predictions

| id | claim | outcome |
|---|---|---|
| P1 | box: pressure order in $[1.95, 2.05]$, gradient in $[1.9, 2.1]$ | **held** |
| P2 | body grid alone: pressure in $[1.9, 2.1]$, gradient in $[1.8, 2.1]$ | **held** |
| P3 | twisted body grid alone: the same | **held** |
| P4 | composite, width 3: pressure in $[1.9, 2.1]$ on both, background gradient $\ge 1.85$, body gradient in $[1.8, 2.1]$, body error $\le 3\times$ the control's | **held** |
| P5 | twisted composite: P4's bounds | **held** |
| P6 | composite, width 2: pressure $\ge 1.85$, background gradient $\le 1.7$ | **held** |
| X1 | exactness to $10^{-12}$ | **held** |
| X2 | three refusals | **held** |
| G1 | the generator reproduces the circle to $10^{-6}$, skew $\le 0.01°$ | **held** |
| G2 | ellipse, plate, NACA 0012 generate unfolded | **held** |
| G3 | the dent is refused at $\kappa = 0$ and $\kappa = 1$ | **held** |
| G4 | the generated ellipse's composite converges like the analytic one | **falsified** (§6) |
| T1 | SuperLU at the car's size: factor $\le 30$ s, solve $\le 0.3$ s | **held** |
| T2 | AMG-BiCGSTAB reaches $10^{-10}$ in $\le 100$ iterations | **held** |
| D1 | the adopted smoothing meets G4's bounds | **held** |
| D2 | the adopted smoothing still generates unfolded, and still refuses the dent | **held** |

---

# 10. What this tier did NOT do, named

- **No flow was marched.** Everything here is a scalar elliptic problem with a manufactured solution; the flow solver's pressure has other data and other boundaries.
- **No car part is on a grid.** The shapes are a circle, two ellipses, a plate and a NACA section; the price is on a stand-in.
- **Cutting a body grid by the box or by another body is refused, not built**, and the car needs both.
- **The incomplete-LU failure is not diagnosed**, and neither is what is left of the overlap's build time.
- **Every cost was taken on battery**, in one process with no other Python running. Re-measure on mains before designing to the millisecond.
- **The smoothing default was checked on convex shapes only.**
- **RaceLab's column, its records, its gate, the car and the demo are unchanged**, and so is every earlier page.
- **Nothing was downloaded or installed** — `pyamg`, `gmsh`, `shapely` and `numba` were found already on this machine; only `pyamg` was used, for pricing, and none of them is a runtime dependency. No machine was rented, the unlicensed structural checkpoint was not loaded, and nothing was pushed.

```
python scripts/tier60_body_fitted_grids.py --out out/racelab9 --stages poisson,properties,generator,price,summary
python scripts/tier60_body_fitted_grids.py --out out/racelab9 --stages diagnose,build_cost,summary
```

---

## See Also

- [[poc3-racelab-outlet-and-start]] — Tier 59, the column this one will replace near the car.
- [[case-study-racelab-graph-atlas-0.1]] — CS-19, the rectangular windows, W218's cut bodies and W220's invisible ones.
- [[poc3-racelab-3d]] — phase 4, where a projection that inverted one operator and applied another was caught the same way: by a measurement on the operator, not by a convergence study alone.
- [[poc3-racelab-demo]] — the dashboard the curved grids will be drawn on.
- [[schwarz-iteration-atlas-0.1]] — the decomposition an overset grid is.
- [[composition-error-theory]] — where an overlap's interpolation error belongs.
- [[navier-stokes-equations]] — the equations Tier 61 solves on these grids.
- [[gap-worklist]] — W263 to W270.
