# PoC 3 phase 4 — the half-car in three dimensions

*Case study, PoC 3 phase 4. New 2026-09-13.*

---

# 0. The result, in one paragraph

**Phase 4 is built: a half-car in a box with a symmetry plane, a 3-D window solver, and the dashboard toggle — classical only, with the learned switch greyed out and the reason on the screen.** The solver is the same explicit fractional step [[case-study-racelab-graph-atlas-0.1]]'s windows run, in three dimensions, with the Neumann pressure solved by a **DCT** rather than iterated. **The first version of it was silently first order and the measurement that caught it is the most useful thing this phase produced**: a projection applied to an analytically divergence-free field perturbed it by a relative $O(h^{0.96})$, which is the signature of a projection that inverts one operator and applies another. Staggering fixed it — the projection alone is now $O(h^{1.98})$ — and the whole step is declared at **the order it was measured at**, between first and second over the range tested, rather than rounded up to its interior operators'. **And the measurement CS-19 §7.4 deferred to this phase is made**: the wetted surface's panel count fits an exponent of $\mathbf{1.910}$ against a 2-D station count that does not move, confirming that page's `[AI Inference]` that a surface in three dimensions carries $O(n^2)$ where a curve carries $O(n)$.

---

# 1. What the requirements asked for

§3.2 asks for a **half-car in a box** with a symmetry plane rather than a body of revolution, *because a car is not axisymmetric*: body, front wing with endplate, floor, diffuser, one sidepod with its duct, one front and one rear wheel. §9 makes phase 4 **optional to ship** and **classical only**, and says it ends with *"the same dashboard with a 3-D toggle, and the learned switch greyed out in 3-D with the reason."*

§9.1 records the open decision and this phase does not take it: **Poseidon-T is 2-D**, so a learned 3-D window would have to be trained, which would also be the rung-5 expert [[case-study-ladder-to-f1]] has been waiting for. The requirements' own default is to ship classical-only and say so, *"because the data-generation budget cannot be estimated"* until this solver's cost per sample is known — which is what this phase measures.

---

# 2. The solver, and the defect it was built with

## 2.1 What it is

`racelab3d.WindowNS3D` is the 2-D expert's method in three dimensions: explicit fractional step, skew-symmetric centred advection, centred diffusion, body force in the right-hand side, a ring re-imposed after every stage, and a projection whose pressure carries homogeneous Neumann data on every face. The pressure is solved by a **discrete cosine transform**, which is the exact eigen-decomposition of the Neumann Laplacian on a cell-centred uniform grid — so the projection is exact to round-off rather than iterated to a tolerance.

## 2.2 The first version was first order, and its divergence looked healthy

The first arrangement kept all three velocity components at cell centres, as the 2-D expert does, and took the divergence with a **backward** difference and the pressure gradient with a **forward** one, so the two composed to exactly the compact Laplacian the DCT inverts. That much was verified directly: the finite-difference operator agreed with the spectral one to $4.5\times10^{-13}$.

**And the scheme converged at first order anyway.** The measurement that found it was not the convergence study — that only said the order was $1.1$ — but this:

| $n$ | a projection applied to an analytically divergence-free field changes it by |
|---|---|
| 16 | $5.560\times10^{-3}$ |
| 32 | $2.867\times10^{-3}$ |
| 64 | $1.455\times10^{-3}$ |

**Order $0.96$.** A projection should be nearly the identity on a solenoidal field and wrong by $O(h^2)$; this one was wrong by $O(h)$. The cause is that a one-sided difference is only first-order accurate **at the cell centre** even though two of them compose to a second-order Laplacian. On a collocated grid one cannot have both a second-order divergence and a compact exact Laplacian — that is the checkerboard problem, and it is structural rather than a coding slip.

## 2.3 Staggering, and what it bought

`u` now lives on the x-faces with shape $(n{+}1, n, n)$, `v` on the y-faces, `w` on the z-faces, and the pressure at the $n^3$ centres. The divergence of face-normal velocities is second-order accurate **at** the centre and compact; the pressure gradient is second-order **at** the face; and the two compose to exactly the operator the DCT inverts. All three at once, which the collocated arrangement cannot give.

| | collocated | staggered |
|---|---|---|
| projection on a solenoidal field, $n=16$ | $5.560\times10^{-3}$ | $\mathbf{3.891\times10^{-5}}$ |
| its order | $0.96$ | $\mathbf{1.98}$ |

**A factor of 143 at the same resolution, and the right exponent.**

## 2.4 The order is declared as measured, not as intended

The control is the **Ethier–Steinman Beltrami flow**, an exact solution of the 3-D incompressible equations whose nonlinear term is a pure gradient — so the pressure absorbs it and the field decays as $e^{-\nu d^2 t}$ with its shape unchanged. That puts advection, diffusion and projection under test **together**, which a decay test on a single Fourier mode does not: a solver can be right about diffusion and wrong about the nonlinearity and never know.

| $n$ | rel $L_2$ | divergence | order |
|---|---|---|---|
| 16 | $1.478\times10^{-3}$ | $1.30\times10^{-3}$ | — |
| 24 | $9.519\times10^{-4}$ | $5.76\times10^{-4}$ | $1.09$ |
| 32 | $6.548\times10^{-4}$ | $3.24\times10^{-4}$ | $1.30$ |
| 48 | $\mathbf{3.578\times10^{-4}}$ | $1.44\times10^{-4}$ | $\mathbf{1.49}$ |

**Refining $\mathrm{d}t$ alone at fixed $n$ moves the error by $1$–$2\%$**, so what remains is spatial and not the explicit time step. The order is still climbing at the finest grid measured. **It is recorded as what it is — between first and second order over this range — and `BELTRAMI_ORDER` carries the table so no page has to quote a number that was never taken.** A test asserts the recorded maximum is below $2.0$, so the table cannot quietly become a claim.

---

# 3. The half-car

The third dimension is a **width profile over the traced 2-D silhouette**, not a revolution of it, because a car is not axisymmetric. `HALF_WIDTH` is keyed to the traced car's own landmarks: the wings are the widest parts at $40$ and $36$ cells, the nose the narrowest at $7$, the sidepod $34$, the tail $12$. The symmetry plane is $z = 0$ and **no panel is placed in it** — a panel in the symmetry plane has no wetted area, because the flow does not cross it.

Wheels are cylinders whose axes lie across the car, sitting **outboard** at $z = 42$ to $60$. That is the thing a centreline slice cannot represent and the reason the 2-D car has to stop its floor short of them: in two dimensions the wheel and the floor compete for the same $x$.

At $n_s = n_z = 8$ the car carries **1248 panels** over a wetted area of $\approx 5.1\times10^{4}$ cells$^2$, and the area moves by under $1\%$ between the coarsest and finest panelisation — a panelisation whose area moved with the knob would be measuring the knob.

---

# 4. The measurement CS-19 deferred to this phase

[[case-study-racelab-graph-atlas-0.1]] §7.4 records an `[AI Inference]`: *the 3-D phase's cost is dominated by the same term, not by the solver, because a surface in three dimensions has $O(n^2)$ stations where a curve has $O(n)$* — and says **phase 4 is the place to measure it**. One refinement knob drives both directions, so both counts come off the same ladder:

| refinement | 3-D panels | 2-D stations |
|---|---|---|
| 2 | 128 | 440 |
| 4 | 416 | 440 |
| 8 | 1664 | 440 |
| 16 | 6656 | 440 |

**Fitted exponent $\mathbf{1.910}$ — CONFIRMED.** The 2-D station count is fixed by the expert and does not move with this knob at all, which is the other half of the claim. The exponent is fitted rather than asserted, and the test fails if the count comes out linear.

**[AI Inference]:** the inference's *consequence* — that the stamping rather than the solver dominates the wall time — is **not** yet established here. At the coarsening this phase uses, a macro-step is $\approx 0.04$ s over 5 windows and $153{,}600$ cells, and the panel and cell costs have not been separated. Measuring that separation is what would let §9.1's training-budget question be answered, and it is not done.

---

# 5. The dashboard

`RaceConfig.dims` selects `"2d"` or `"3d"`; the page carries a **dimensions** toggle above the switch. In 3-D the engine marches `racelab3d.March3D` a macro-step at a time and publishes a **z-slice** of the speed field, and:

- **every mode button, every preset and *measure windows* is `disabled`, not hidden** — greyed to $0.4$ opacity, so a reader can see that a control exists and has been withdrawn;
- the reason is printed in full beside them, in a warning band, naming the checkpoint, its fixed $128\times128$ resolution, and §9.1's open decision;
- the frame reports the panel count, the window count, the box size, the coarsening and the cost per macro-step, so the price of the third dimension is on the screen rather than in a log.

**The overlay says it is a slice.** A z-slice of a half-car is not the 2-D car and the note says so, because two pictures that look alike and mean different things is exactly the confusion a dashboard should not create.

## 5.1 Three things opening the page found, and none of them could be found any other way

The toggle passed every socket-level check before it was rendered: the engine marched, the frame carried `dims`, `learned_available` was `False`, and the reason was on the wire. **Opening it found three defects anyway**, which is [[poc3-racelab-demo]] §5.4's lesson arriving on schedule.

- **The 2-D vector overlay was drawn on top of the 3-D slice.** Fourteen $128\times128$ windows, the centreline bodies and the device planes, superimposed on a z-slice of the half-car — two different objects that happen to render alike, with the 2-D layout caption underneath describing a decomposition the picture was not of. **That is worse than W239 describes**, which is about two pictures resembling each other and not about them being drawn on top of one another. The overlay is hidden in 3-D and the caption replaced.
- **And the caption never came back.** It is set once from the meta message, so overwriting it in 3-D made the change permanent and the 2-D view then described the 3-D box. Caught by toggling **back**, which the first check did not do — a round trip is the test, not a one-way trip.
- **The demo was releasing the traced car from the hand-drawn car's settled field.** `out/racelab2/cache/settled.npz` was settled around bodies that had since moved, so the duct saw the wrong flow, the disk's induction pinned on its clamp, and the page stamped `OUTSIDE THE MODEL` from macro-step 6 — **correctly, which is how it was noticed.** A settled field belongs to a geometry. The cache now prefers Tier 54's, and the fallback says in its own words that it is the wrong car's.

**The stamp in three dimensions is not silence and not a stale 2-D decline.** A stamp carried over from the 2-D march would be reporting on a column that is not running, and an empty stamp would read as a clean bill of health. The page says instead that there is **no declared envelope in 3-D at all**, names W238, and calls it a gap.

---

# 6. What this phase did NOT do, named

- **No learned expert in three dimensions, and the decision is not taken.** §9.1 stays open. This phase measures the solver's cost so that it *can* be taken later; it does not take it.
- **The stamping-versus-solver split is not measured** (§4), so the inference's consequence is still an inference.
- **The 3-D lattice is coarsened $4\times$ from the 2-D one.** At the 2-D cell size the box holding this car is $9.8$ million cells and a macro-step is minutes, not the $0.5$ s §3.1 sets. The coarsening is declared in `COARSEN` and shown on the page.
- **The decomposition is one-dimensional** — windows along $x$ only — because the car is long and thin and the seams that matter run across it. A 3-D tiling in all three directions is not built.
- **No envelope is declared for the 3-D column.** The 2-D car's predicates are about its own experts and devices; nothing has been declared for this solver, and so nothing is checked. That is a gap and not a pass.
- **No join, no device, no powertrain in 3-D.** This is a fluid box with a car in it, not the union [[case-study-vehicle-march-atlas-0.1]] marches.
- **Nothing was downloaded for it, no machine was rented, and NeuberNet was not loaded.**

---

## See Also

- [[case-study-racelab-graph-atlas-0.1]] — phase 1, and §7.4's inference that this phase measures.
- [[case-study-racelab-switch-atlas-0.1]] — phase 2, the switch this phase greys out.
- [[poc3-racelab-demo]] — phase 3, the dashboard this phase adds a toggle to.
- [[gap-worklist]] — the rows this phase opens.
