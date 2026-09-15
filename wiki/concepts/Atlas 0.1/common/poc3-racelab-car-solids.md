# The drawn car as solids, on body-fitted grids

*Case study, PoC 3. New 2026-09-15. Tier 62.*

---

# 0. The result, in one paragraph

**The car the user drew now marches on body-fitted grids: its 27 plates turned into five welded solids and two rolling wheels by the rule the user chose, 381,698 unknowns on ten grids cut by each other and by the road, marched stably from the free stream to $t = 16$ at a median $1.4$ s a step.** Getting there needed three things Tier 60 refused — grids cut by other bodies, grids cut by the road, and road patches under the tyres — and they are verified before the car uses them: the subclass joins Tier 60's own grids bit for bit, every interpolation equation reproduces a linear field to $1.3\times10^{-15}$, and a manufactured pressure on a tyre two percent of its radius above a road patch, beside a panel and an aerofoil, converges at order $1.88$ to $2.10$ on every grid through a fifth level. The manufactured flow's velocity converges at $1.75$ to $2.83$; its pressure does not converge on the road patch ($0.22$, W277). **What the march says about the drawing is the finding.** With solid walls the pod the drawing encloses is fed only through the clearances left round the wheels, and **its duct flows backwards**: $-0.109$ at both device planes, against $+0.457$ forward in the porous column on the same drawing (W278). **The car makes lift, $+1.83$, not downforce**, and most of it comes from the front wing, whose drawn outline had to be filled almost solid before any grid could wrap it (W275, W279). Sixteen predictions were registered: thirteen held, and the three that failed are those two findings and the road patch's pressure.

---

# 1. What was decided, and what had to be built

On 2026-09-15 the user chose how the drawing becomes solids, and chose it editable:

| decided by the user | what it replaces |
|---|---|
| **wing elements are aerofoil-like**, about $12\%$ of their chord thick | porous line segments |
| **body and floor panels are a few cells thick** | the same |
| **shell plates that touch are welded into one closed body** | each plate on its own |
| **every thickness is data**, in `car_geometry.json`'s `solids` block | — |
| **the wheels sit a small gap above the road**, about $2\%$ of the radius, and **roll at road speed** | a ring of porous segments that could not roll (W219) |

Tier 60 had left three rows that stood between that decision and a march: **W264** (a wall needs a closed outline), **W265** (a grid cannot be wrapped round a point of contact) and **W266** (a grid cut by another body or by the road is refused, not built). This tier closes all three.

---

# 2. Several bodies and a road on one composite

`atlas.cases.overset_multi.MultiOverset` subclasses Tier 60's `Overset`, so the grids Tier 60 joins it still joins the same way — **V1 held bit for bit**: on Tier 60's own geometry at two levels the point statuses, the interpolation points and their stencils are identical, and the weights differ by exactly $0$.

## 2.1 How the grids are joined

1. **Boundary rows.** A body grid's row 0 is its wall and its outer row interpolation. A **road patch** — a new, non-periodic grid lying along the road under a wheel — has the road as its row 0 and interpolation on its two ends and its top.
2. **Holes.** A background point inside any body, or within the hole margin ($0.04$) of one; a grid's point inside any *other* body, or below the road. A background edge cell whose first or second cell inward is a hole becomes a hole too, because an edge cell carries the box's condition through exactly those two cells — allowed along the road, refused at the inlet, the outlet and the top.
3. **Fringe.** A background point with a hole among its four neighbours, or a body grid's with a hole in its $3\times3$ block (the curvilinear Laplacian is nine-point), becomes an interpolation point.
4. **Donors**, explicit as in Tier 60 — every node of a donor stencil carries its own equation. For a point next to a hole, the grid of whatever cut the hole is tried first (a road patch, for the road), then the other body grids nearest first, then the background; for a grid's own boundary, the background first. A stencil that touches a hole or an interpolation point is shifted one node each way, with Newton re-run on the shifted stencil, before a candidate is given up.

**One tolerance was added, and why.** A wall's polygon is what cuts, while its wall row's quadratic interpolant bulges slightly past the polygon's chords, so a point can be outside a tyre by the cut and just below $\eta = 0$ by the map. Two road-patch points were orphans for exactly that reason, and a stencil starting at a wall row may now reach a quarter of a row past it.

**V2 held**: at two levels every interpolation equation reproduces $0.7x - 1.3y + 2$ to $1.3\times10^{-15}$, and every donor node is a discretisation or wall point. **V3 held**: two bodies that overlap, a wheel on the road, a tyre closer to the road than two of its grid rows, and a grid leaving through the inlet are each refused with a message that names the point.

## 2.2 The tyre, the road and the gap

A wheel two percent of its radius above the road puts the bottom of its own grid below the road, and the background — a sixty-fourth of a unit per cell — cannot resolve a gap of half a cell. So the road cuts the wheel's grid, and a road patch $1.6$ radii long and a quarter of a radius tall, its rows clustered toward the road like the wheel's toward its wall, resolves the gap. On the verification composite at level 3 the road cuts $1466$ of the wheel grid's points, the wheel cuts $1747$ of the patch's, and each serves the other's fringe ($121$ and $239$ donors).

---

# 3. Three additions to the generator

`ogrid_from_outline` gained three options, each off by default so Tier 60's grids are unchanged. **V4 held**: with all of them on, a grid at twice the columns and rows, sampled at every other, *is* the coarser grid — difference exactly $0$ — so each option is a property of the outline, not of the resolution.

- **A rounded trailing edge** (`aerofoil_outline`): the NACA thickness gains $t_{te}/2$ times $x/c$ and a semicircle closes it.
- **Columns clustered where the outline turns**: the density $1 + c\,w\,\dot\theta_w$, with $\dot\theta_w$ the turning rate smoothed over a width $w$.
- **Room-limited thickness**: at each column the thickness is at most $0.4$ times the distance along the normal to the outline itself, never under a floor, taken as a running minimum and smoothed — a grid reaching into its own body's notch stops short of the far wall instead of folding.

**How the trailing-edge settings were chosen** — on the verification geometry itself, so the page says it. The manufactured pressure's max-error order from level 3 to 4:

| aerofoil grid | background | aerofoil |
|---|---|---|
| NACA trailing edge closed to a point | $1.14$ | $1.12$ |
| rounded to $1\%$ of the chord | $1.67$ | $1.72$ |
| $+$ columns clustered ($c = 4$) | $1.76$ | $1.82$ |
| $+$ normals smoothed at the wall ($\sigma_w = 0.02$, $\kappa = 0.5$) | $\mathbf{1.94}$ | $\mathbf{2.01}$ |

Behind a point, an offset grid's columns fan out and the error sits in the fan. Those settings became the car's.

---

# 4. The verification

## 4.1 A manufactured pressure (M1)

A $3\times1.2$ box with the road its lower face; a wheel of radius $0.3$ two percent above it with its road patch; a thick panel whose nose comes within $0.027$ of the tyre, under two background cells at level 3; an NACA aerofoil above the panel. Every grid is built with the car's settings, and level 3 is the car's own spacing, $h = 1/64$.

| max error | L3 ($46{,}613$) | L4 ($183{,}652$) | L5 ($728{,}356$) | orders |
|---|---|---|---|---|
| background | $1.37\times10^{-3}$ | $3.73\times10^{-4}$ | $9.32\times10^{-5}$ | $1.88$, $2.00$ |
| wheel | $1.04\times10^{-3}$ | $2.52\times10^{-4}$ | $6.20\times10^{-5}$ | $2.05$, $2.02$ |
| panel | $7.27\times10^{-4}$ | $1.76\times10^{-4}$ | $4.31\times10^{-5}$ | $2.05$, $2.03$ |
| aerofoil | $1.49\times10^{-3}$ | $3.76\times10^{-4}$ | $9.36\times10^{-5}$ | $1.99$, $2.01$ |
| road patch | $2.44\times10^{-4}$ | $5.71\times10^{-5}$ | $1.39\times10^{-5}$ | $2.09$, $2.04$ |

**Second order on every grid, including the fifth level no exploration had reached.** At level 5 SuperLU factored the $728{,}356$-unknown system in $67$ s.

## 4.2 A manufactured flow (M2, M3)

Tier 61's manufactured solution on the same composite, every wall — the tyre, the panel, the aerofoil and the road — moving with the exact velocity, at $\Delta t = 0.00025$ to $t = 0.05$ so the first-order time error (W271) is out of the way.

| L3→L4 order | velocity max | pressure rms |
|---|---|---|
| background | $1.90$ | $2.12$ |
| wheel | $1.75$ | $1.32$ |
| panel | $2.00$ | $1.34$ |
| aerofoil | $2.04$ | $1.54$ |
| road patch | $2.83$ | $\mathbf{0.22}$ |

**The velocity converges on every grid; the pressure does not converge on the road patch**, so M3 fails. Its largest pressure error sits at the patch's wall row next to its end — at both levels $x \approx 0.66$, $y = 0$, where the road passes from the patch to the background's road row. Two different one-sided conditions describe one road there, and the error does not shrink (W277).

> **[AI Inference]:** the background's road row writes $\partial\phi/\partial n = 0$ as $3\phi_e - 4\phi_1 + \phi_2 = 0$ at a cell centre on the road, while the patch writes the curvilinear wall row with its own tangential term, and the patch's end column is interpolated from rows the background's condition shaped. A pressure increment continuous across that seam but not smooth would read exactly like this. A patch whose ends run out onto the background's DISC region beyond the road row's influence — or a road patch that spans the whole road — would separate the two.

## 4.3 The car's own composite (M4)

On the car's ten grids a manufactured pressure's max error is at most $5.7\times10^{-3}$ on any grid, and the uniform stream with every wall moving with it stays uniform to $1.9\times10^{-14}$ over three steps. **M4 held.**

---

# 5. The car as solids

## 5.1 The rule, step by step

`atlas.cases.car_solids.car_solids`, in cells:

1. **Every plate becomes a polygon**: `FW_MAIN`, `FW_FLAP` and `RW_FLAP` as $12\%$ aerofoils with a trailing edge rounded to $1\%$ of the chord; every other plate as a panel $3$ cells thick with round ends.
2. **Wheels** are circles raised to $2\%$ of their radius above the road. A panel is **trimmed along its centreline** to stay $4$ cells from any wheel, then thickened, so a trimmed end is a round cap like any other.
3. **Welding.** The polygons are united and closed by $1.5$ cells: parts closer than that become one body.
4. **Fillets**, one shell at a time, from a ladder of radii ($0$, $1.5$, $3$, $4.5$, $6$, $8$, $10$ cells): each shell takes the smallest radius whose grid neither folds, nor skews past $70°$, nor leaves an orphan against the background alone.
5. A fluid pocket sealed inside a body is filled and a sliver is dropped (**none occurred**).

**Which of those numbers are the user's and which are this tier's.** The user chose aerofoil-like wing elements at about $12\%$, panels a few cells thick, welding, editable thicknesses, and a gap of about $2\%$. **This tier chose** the $1\%$ trailing edge (§3), $3$ cells for "a few", the $1.5$-cell weld, the $4$-cell clearance (§6), and the fillet ladder with its three tests. Each is a named field of `SOLIDS_DEFAULT` or `GRID`, and `car_geometry.json` can override any of the rule's.

## 5.2 What the rule made of the drawing

| solid | plates | what the ladder tried | fillet | area added |
|---|---|---|---|---|
| front wing and nose tip | `FW_MAIN`, `FW_UPPER`, `FW_FLAP`, `NOSE_TIP`, `FW_UNDER_TE`, `FW_TRAILING`, `FW_LEADING`, `FW_UNDER` | $0$ to $3$ fold; $4.5$ skews $84°$ | $\mathbf{6}$ | $427$ cells², **$+92\%$** |
| body, rear wing and diffuser | `NOSE`, `CHASSIS`, `COCKPIT`, `ROLL_HOOP`, `AIRBOX`, `POD_UP`, `COVER_TAIL`, `TAIL_DECK`, `RW_PYLON`, `RW_FLAP`, `RW_ENDPLATE`, `RW_ENDPLATE_LO`, `DIFF_EXIT`, `DIFF` | $0$ to $3$ fold; $4.5$ leaves $62$ orphans inside the rear wing | $\mathbf{6}$ | $83$ cells², $+5\%$ |
| floor | `FLOOR_LE`, `FLOOR_STEP`, `FLOOR` | $0$ passes | $0$ | — |
| duct, lower and upper | `DUCT_LO`; `DUCT_UP` | $0$ passes | $0$ | — |
| two wheels | — | — | — | — |

**The front wing is the largest departure from the drawing.** Its plates draw an endplate outline with the main plane inside it; welded, that outline and the main plane make one thin body with a deep notch and a slot, and no offset grid $0.1$ thick wraps the notch until a $6$-cell fillet has filled most of it. The solid that marches is closer to a wedge than to a wing (W275). Six panels were trimmed at the wheels' clearances — `NOSE_TIP`, `NOSE`, `FLOOR_LE`, `FLOOR`, `DIFF` and `FW_UNDER_TE`, each losing $15$ to $19$ cells² — and the drawn wheels overlapped the silhouette, which a centreline slice cannot have, so those trims are what the clearance *is* (W276).

## 5.3 The grids

| grid | points | detail |
|---|---|---|
| background | $672\times241$ | RaceLab's lattice, its first row on the road |
| front wing | $512\times25$ | skew $\le 63.6°$ |
| body shell | $3328\times25$ | skew $\le 51.1°$, thinned to $0.091$ in its narrowest corner |
| floor, two ducts | $1856\times25$, $1152\times25$ ×2 | skew $\le 30°$ |
| two wheels | $512\times33$ each, $0.15$ deep | rows clustered toward the tyre |
| two road patches | $224\times33$ each | $0.8$ radii either side of the contact, $0.25$ radii tall |

$381{,}698$ unknowns; the background loses $15{,}101$ points to holes, and every grid's holes are named by what cut them — the road cut $1701$ of each wheel's points.

---

# 6. Starting the car: two starts that failed

Tier 61's cylinder started impulsively. The car did not survive that, and the start it marches from is the third tried.

| start | what happened |
|---|---|
| **impulsive**: the stream everywhere but on the walls | the jump across a wall row $0.0014$ thick put the first step's divergence at $2.3\times10^{3}$ and its speed at $23$; the second momentum solve did not converge |
| **from rest**: stream, road and wheels ramped together | a pressure-correction step accelerates the interior only through the lagging pressure, so the road grew a spurious boundary layer that the road patches resolve and the background does not; the divergence at a patch's end grew from $0.5$ to $2.5$ in six steps |
| **walls ramped to rest from the stream** (adopted) | at $t = 0$ every wall moves with the stream, so the uniform stream is an exact state (§4.3); over $0.5$ each wall's velocity turns into its own, $(1-s)\,U + s\,\mathbf u_{\text{wall}}$ with $s = \tfrac12(1 - \cos\pi t/0.5)$ — rest for a shell, $\boldsymbol\omega\times\mathbf r$ for a wheel. Nothing far from the car ever changes |

Two geometry settings changed on the way, both recorded before the registered run. **Square corners** where the first version subtracted the clearance discs from the *thickened* panels put the first steps' largest divergence on the nose's and the diffuser's trimmed ends, growing from $3.5$ to $28$ in six steps; trimming the centreline instead leaves caps. **A $2$-cell clearance** put a divergence of $28$ in the slits between the rolling tyres and the caps by step 11; at $4$ cells, with a smaller road patch that no longer runs into the diffuser, the march below.

**The momentum solve needed a better preconditioner.** On the car's cells — up to $28$ times longer than wide, skewed up to $64°$, a wall spacing of $0.001$ — Jacobi-preconditioned BiCGSTAB took $100$ to $300$ iterations a component. An incomplete LU factor, refreshed every $20$ steps, took it to $1$ to $33$. SuperLU's first incomplete factor at a drop tolerance of $10^{-4}$ was *exactly singular*, undiagnosed (W280); $10^{-6}$ worked, and the solver falls back to it, then to Jacobi, and says which ran — in the registered march all $64$ refreshes fell back to $10^{-6}$.

---

# 7. The march

From the stream at $t = 0$ to $t = 16$, $\Delta t = 0.0125$ (RaceLab's macro-step), $\nu = 0.004$, free stream on the inlet and the top, the rolling road, an outflow. **Statistics over $t\in[12, 16]$.** On battery, one Python process, with the app holding the machine awake.

## 7.1 Stability, mass and cost

- **No blow-up** (K1): every momentum solve converged; the largest speed was $3.82$, during the ramp — a starting vortex shed from the rear wing's top, which drifted up and out through the outlet by $t \approx 8$.
- **Divergence** (K2): at most $7.8$ during the ramp, on the front wing's notch; $0.030$ over the window, on the rear road patch beside the road.
- **Mass** (K3, K5): the net flux out of a contour round the car is $0.14\%$ of the inflow through it; the duct's flux through its two device planes agrees to $2\times10^{-5}$ of itself.
- **Cost** (K8): at most $7$ BiCGSTAB iterations a component after $t = 1$; a median step of $1.39$ s; $1280$ steps in $47$ minutes.

## 7.2 The forces

Per unit span, in the tiling's units (free stream $1$, density $1$):

| | drag $F_x$ | vertical $F_y$ (positive up) |
|---|---|---|
| **the car** | $\mathbf{2.736}$ (spread $1.1\%$) | $\mathbf{+1.834}$ (spread $0.127$) |
| front wing and nose tip | $-0.246$ | $+1.419$ |
| body, rear wing, diffuser | $+0.701$ | $-0.901$ |
| floor | $-0.369$ | $+0.263$ |
| duct plates | $-0.036$ | $+0.271$ |
| front wheel | $+1.869$ | $+0.348$ |
| rear wheel | $+0.816$ | $+0.433$ |

**The front wheel carries $68\%$ of the car's drag. The car makes lift** (K6 failed), **and the front wing makes most of it**: by the plate each wall point belongs to, `FW_UNDER` — the filled wedge's flat underside, low over the road — carries $+2.14$. The body shell makes the only real downforce, $-0.90$.

> **[AI Inference]:** a wedge whose broad underside slopes up toward a wheel, over a road moving at the stream, stagnates the flow beneath it. The drawing's wing elements, porous in the old column, could make downforce by turning the flow; filled into the wedge by the fillet, they cannot. W275's generator, which would keep the notch open, is the obvious control — and the drawing itself, which draws an endplate a slice does not have, the other.

## 7.3 The duct flows backwards

**The mean streamwise velocity across the duct is $-0.109$ at the radiator core's plane and $-0.109$ at the turbine's** (K4 failed). In the porous column on the same drawing — the fingerprints match — the turbine's inflow at the settled release state is $+0.457$: the solid car's duct carries $-24\%$ of it.

The flow reversed early: $+0.03$ at $t = 0.8$, $-0.05$ by $t = 1.6$, $-0.10$ by $t = 7$. **K4's band was written from an exploration whose summary averaged from $t = 0$**, where the duct holds the uniform stream, so its $0.154$ was never the duct's own flow — a window chosen wrongly in the exploration, not a threshold moved; the registered stage reads its declared window.

With solid walls, the pod the drawing encloses is reached only through the clearances round the wheels. **The devices Tier 63 re-sites into that duct would see a weak reversed flow** (W278).

> **[AI Inference]:** the front clearances open into the front tyre's low-pressure wake, and the rear ones toward the higher pressure ahead of the rear tyre and under the diffuser, so air would enter at the back and leave at the front. The pressures at the openings were not measured, and that is the first measurement W278 needs.

---

# 8. The predictions

Registered at 13:40:32, before any stage ran.

| id | claim | outcome |
|---|---|---|
| V1 | on Tier 60's geometry at L2 and L3, statuses, points and stencils identical, weights within $10^{-15}$ | **held** — identical, difference $0$ |
| V2 | at L3 and L4, every interpolation equation exact for a linear field to $10^{-10}$, every donor node its own equation | **held** — $1.3\times10^{-15}$ |
| V3 | overlap, a wheel on the road, a gap under two rows and a grid through the inlet are refused | **held** |
| V4 | clustering and room-limited thickness are one mapping to $10^{-12}$ | **held** — $0$ |
| M1 | manufactured pressure: order $\ge 1.7$ L3→L4 and $\ge 1.8$ L4→L5 on every grid | **held** — $1.88$ to $2.10$ |
| M2 | manufactured flow: velocity max order L3→L4 $\ge 1.6$ on every grid | **held** — $1.75$ to $2.83$ |
| M3 | manufactured flow: pressure rms order L3→L4 $\ge 1.0$ on every grid | **failed** — road patch $0.22$ |
| M4 | the car's composite: manufactured pressure $\le 10^{-2}$, uniform stream to $10^{-12}$ | **held** — $5.7\times10^{-3}$, $1.9\times10^{-14}$ |
| K1 | the march reaches $t = 16$, every solve converged, speed $\le 4$ | **held** — $3.82$ |
| K2 | largest divergence over $[12, 16]$ $\le 0.2$ | **held** — $0.030$ |
| K3 | net flux through a contour round the car $\le 1\%$ of its inflow | **held** — $0.14\%$ |
| K4 | the duct's mean velocity at the turbine plane in $[0.08, 0.40]$ | **failed** — $-0.109$ |
| K5 | core-plane and turbine-plane flux agree to $2\%$ | **held** — $2\times10^{-5}$ |
| K6 | the car's mean vertical force is negative (downforce) | **failed** — $+1.834$ |
| K7 | the drag's spread over $[12, 16]$ $\le 10\%$ of its mean | **held** — $1.1\%$ |
| K8 | at most $60$ iterations a component after $t = 1$ | **held** — $7$ |

---

# 9. What this tier did NOT do, named

- **The front wing was not kept as drawn** (W275): the grid generator cannot wrap the welded endplate outline without filling its notch, and a hyperbolic or elliptic generator, or collar grids at the junctions, were not built.
- **The road patch's pressure was not diagnosed** (W277), and neither was the incomplete LU's singular factor (W280).
- **Nothing here is RaceLab's march.** The devices, the joins, the agents, the seams, the learned expert's background windows and the demo are Tier 63; the car's march here has no radiator core and no turbine in its duct.
- **The time accuracy is Tier 61's first order** (W271); the forces over $[12, 16]$ are settled to $1.1\%$ in drag, not shown independent of $\Delta t$ or the grids.
- **Every number is two-dimensional, at the tiling's Reynolds number of $250$ per unit length**, and nothing was compared with a wind tunnel or a published car.
- **No cost was taken on mains.**
- **RaceLab's column, its records, the gate, the demo and the bundle are unchanged**, and so is `car_geometry.json` — its `solids` block is optional and absent, so the rule's defaults are what ran.
- **Nothing was downloaded or installed** (`shapely` was already on this machine), no machine was rented, the unlicensed structural checkpoint was not loaded, and nothing was pushed.

```
python scripts/tier62_car_solids.py --out out/racelab11 --stages controls,poisson,flow,car
python scripts/tier62_car_solids.py --out out/racelab11 --stages march
python scripts/tier62_car_solids.py --out out/racelab11 --stages compare,summary
```

The record is `out/racelab11/racelab11.json`; the march's full trajectory, `out/racelab11/march.json`, is not carried upstream. The first `compare` read the march's summary under names an earlier draft used and failed; the error stays in the record's `stage_errors`, and the stage was re-run after the names were corrected.

---

## See Also

- [[poc3-racelab-overset-flow]] — Tier 61, the flow solver this march uses.
- [[poc3-racelab-body-fitted-grids]] — Tier 60, the grids, the overlap and the refusals this tier lifts.
- [[poc3-racelab-outlet-and-start]] — Tier 59, the porous column's settled release state the duct is compared with.
- [[case-study-racelab-graph-atlas-0.1]] — CS-19, where W218 to W220 were opened on the rectangular windows.
- [[navier-stokes-equations]] — the equations.
- [[gap-worklist]] — W275 to W280, and the rows this tier closes.
