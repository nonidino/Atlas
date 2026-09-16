# The body-fitted composite on one raster: the demo's field overlay

*Case study, PoC 3. New 2026-09-16. Tier 66.*

---

# 0. The result, in one paragraph

**The porous column's fluid is fourteen rectangles on one lattice, so a picture of it is an array slice. The body-fitted column is twelve overlapping grids and there is no array that holds the answer — so the demo could not draw it at all.** `atlas/cases/car_render.py` turns the composite into one rectangle of pixels, and the whole of this tier is the argument that the picture can be trusted: it is built from **the solver's own interpolation operator**, the one the devices' strips already read the composite with, asked over a raster instead of over a band. It is deliberately not a second implementation — a picture drawn by a second opinion can disagree with the solve, and this is the vault whose entire purpose is to not ship that. Three numbers say it works. A raster pixel is **the same number as a direct probe, to $0.0$ exactly** and not merely to rounding. A linear field laid on every node of every grid comes back through the raster at $3.55\times10^{-15}$, while a quadratic comes back at $4.5\times10^{-6}$ and **does not improve when the pixels are refined** — because that error belongs to the grid spacing, which is what makes the pair identify what is being measured rather than merely pass. And the masked pixels — the ones no grid can hold — are **exactly** the pixels inside the car's outlines, $2{,}785$ of $57{,}600$ at $320\times180$, with zero discrepancy in either direction at all three rasters tried. All eight registered predictions held.

---

# 1. Why drawing the background alone is not a picture of the car

The background is already a rectangle covering the whole domain, so the cheap thing is to colour it in and stop. That picture is blank exactly where the physics is.

The composite cuts holes through the background wherever a body sits, plus a margin, because the body-fitted grid wrapped round that body owns the answer there. Measured on the car's own composite at $320\times180$:

| | pixels |
|---|---|
| the raster | $57{,}600$ |
| masked — inside the car, nothing to draw | $2{,}785$ |
| **drawable** | $\mathbf{54{,}815}$ |
| of those, the background alone can draw | $51{,}102$ |
| **of those, the background alone leaves BLANK** | $\mathbf{3{,}713}$ *(6.8% of the picture)* |

Those $3{,}713$ pixels are the boundary layers, the wheel clearances and the cooling duct — the places the body-fitted column was built *for*, and the places Tiers 62–64 spent themselves resolving. A background-only overlay would not merely be coarse there; it would show nothing there.

**This was measured rather than asserted.** The demo that would have shipped without this tier is "draw the background", so it is run as a control: the same donor search, restricted to the background, over the same raster.

---

# 2. The rule is the solver's own

`car_union.probe_matrix` is the operator the radiator core and the recovery turbine already use to read the composite at arbitrary points: for each point, **the first body grid in `ov.comps` order that holds it with a usable stencil, else the background**. This tier asks that same operator over a raster.

What a raster has that a device's strip does not is points *inside the bodies*. A strip must reach every one of its points — a join applied where the composite cannot be evaluated is a join applied nowhere — so `probe_matrix` raises. A raster covers the car as well as the air, and the pixels on the car are not an error, they are the car.

So Tier 66 split the function rather than copying it:

- **`probe_matrix_masked`** is now the primitive. It returns the operator, a `missing` flag per point, and which grid drew each one.
- **`probe_matrix`** is a thin wrapper that raises when anything is missing, with its message and behaviour unchanged. Tier 64's tests pass untouched.

**That split is the point, not a tidy-up.** If the picture's interpolation and the solve's interpolation were written twice, they could come to disagree — and the failure mode would be a dashboard that looks right and is quietly not the thing being solved.

---

# 3. The mask is the car, exactly

The mask is defined by the **donor search**: "no grid holds this pixel". The car is defined by the **solids' outlines**, which the grids were generated from but which the donor search never consults. So their agreeing is a real check and not a tautology — it says the hole cutting, the grid generation and the donor search all describe the same car.

| raster | masked | inside the outlines | masked but outside | inside but drawable |
|---|---|---|---|---|
| $160\times90$ | $702$ | $702$ | $0$ | $0$ |
| $320\times180$ | $2{,}785$ | $2{,}785$ | $0$ | $0$ |
| $480\times270$ | $6{,}296$ | $6{,}296$ | $0$ | $0$ |

Set equality at every resolution. The masked fraction is $4.84$–$4.88\%$ and does not drift with the raster, which is what it should do for a set that is a region of the plane being sampled more finely.

The mask is also **state-independent and build-independent**, both checked rather than assumed: rendering a random vector instead of Tier 64's state gives the identical mask, and building the operator a second time gives bitwise identical `data`, `indices` and `indptr`. It is a function of the geometry alone.

---

# 4. The two controls, and why there are two

The donor weights reproduce linear fields by construction. So a plane $f(x,y) = ax + by + c$ laid on every live node of every grid must come back through the raster **exactly**:

$$\max_{\text{drawn}} \left\lvert (Mf)_k - f(x_k, y_k) \right\rvert \;=\; 3.55\times10^{-15}$$

at every raster tried. Almost any error in the stitching — a grid taken in the wrong order, a weight normalised wrongly, a stencil shifted and not re-solved — would show up as a kink in a plane.

**That control proves nothing on its own**, because a renderer that ignored the grids entirely and evaluated the plane analytically would pass it too. So a quadratic $f = x^2$ is rendered beside it and **must not** be exact:

| raster | linear | quadratic |
|---|---|---|
| $160\times90$ | $3.55\times10^{-15}$ | $5.42\times10^{-6}$ |
| $320\times180$ | $3.55\times10^{-15}$ | $4.46\times10^{-6}$ |
| $480\times270$ | $3.55\times10^{-15}$ | $5.76\times10^{-6}$ |

The quadratic error is $\sim10^{9}$ times the linear one, so the pair is not measuring rounding. And **it does not fall as the raster is refined** — it is set by the *grid* spacing, not the pixel spacing, so refining pixels cannot refine it. That flatness is what identifies which discretisation the error belongs to.

---

# 5. What a frame costs, and the step it is measured against

The operator splits into an expensive half paid once and a cheap half paid every frame.

| raster | build (once) | apply (per frame) |
|---|---|---|
| $160\times90$ | $15.8$ s | $0.00014$ s |
| $320\times180$ | $63.2$ s | $0.00031$ s |
| $480\times270$ | $143.5$ s | $0.00075$ s |

A frame is one sparse matrix-vector product, so the demo's frame rate is set by the solver, which is the right way round.

**The denominator had to be measured properly, and the first attempt got it wrong.** Timing one step and dividing by it gave a step of $3.24$ s — five times Tier 64's median — because `overset_ns` performs the incomplete-LU factorisation on the first step after a state is loaded (`_ilu` starts unset) and then reuses it for twenty steps. Dividing by that would have let the prediction pass by a factor of five it had not earned, and in the *permissive* direction, which is the direction a check never gets to be wrong in. The check now takes five steps and divides by the median of the ones that did **not** refactor:

| | seconds |
|---|---|
| first step, including the factorisation | $3.4355$ — the ILU is $\sim86\%$ of it |
| **the denominator: the median of the four that did not refactor** | $\mathbf{0.5851}$ |
| a frame at $320\times180$ | $0.00021$ |
| **frame / step** | $\mathbf{0.0364\%}$, about one part in $2700$ |

That median sits $6\%$ under Tier 64's recorded $0.6228$ s, on a machine that throttles between $1.4$ and $3.8$ GHz — close enough to say the two tiers measured the same solver, and far enough apart to say **the absolute number is not a property of the code**.

Which is the whole reason the claim is a *ratio* with both halves taken in one process, seconds apart. The same five steps were timed on three separate runs of this stage and gave step medians of $0.5684$, $0.5851$ and $0.6030$ s and ratios of $0.0400\%$, $0.0364\%$ and $0.0383\%$: **the seconds moved by $6\%$ between runs and the conclusion did not move at all.** Quoting either half alone would have been quoting the machine's mood. All three were measured beside the user's two `phone-remote` servers.

**W287, opened:** the operator depends on the geometry and the raster alone and **is not cached**, so the demo will pay $63$ s at every start. The cache key is already written down in `car_render.RASTER_CACHE_NOTE` — the geometry fingerprint, the grid settings, the hole margin, the stencil width and the raster — and nothing in it changes while the demo runs.

## 5.1 What opening the PNG found, and no assertion did

G7 asked that the car's pixels be exactly the mask's, in **a colour the field's map cannot produce**. It held: $2{,}785$ car pixels of $2{,}785$ masked, and the $256$-entry map contains nothing equal to the car's colour.

**Then the picture was opened and looked at, and the claim turned out to be nearly worthless.** The first car colour was $(38, 38, 42)$, a near-black grey; the map's lowest colour, the one it paints motionless air with, is $(0, 0, 89)$. Both are dark. The car and the wake read as the same thing — and the wake is the one place on this picture where confusing an object with slow air actually costs something.

"The map never produces this colour **exactly**" is satisfied by any colour whatsoever that is not one of $256$ entries. It says nothing about whether a viewer can tell the car from the air. The quantity that matters is the **distance to the nearest colour the map can make**, and measuring it is damning of the guess:

| car colour | margin |
|---|---|
| mid grey $(128,128,132)$ | $3.6$ — invisible as an object |
| slate $(96,104,112)$ | $6.2$ |
| the first choice, near-black $(38,38,42)$ | $56.8$ |
| **now: dark maroon $(92,34,44)$** | $\mathbf{79.1}$ |

`car_render.car_colour_margin` computes it and a test pins it above $60$, with a failing control — a mid grey must score under $10$ — so the test cannot pass by measuring nothing.

**The registered G7 was deliberately left as written and is still judged as written.** A pre-registered criterion that gets rewritten once the data are in is not a pre-registered criterion, and this vault has a row about exactly that drift. The verdict stands on the claim as made; the margin is recorded beside it, and the lesson is that *the predicate was too weak*, which is a finding about the prediction rather than about the renderer.

**[AI Inference]:** the same weakness likely sits in any "the encoding is unambiguous" check written as an equality — a mask index that never collides, a sentinel value that never occurs — because equality tests identity while the property wanted is separation. Where the consumer is an eye or a lossy channel, the check should be a distance with a floor.

## 5.2 Two more things visible only by looking

**The car is an outline, not a silhouette, and that is correct.** The masked set is $2{,}785$ pixels — under $5\%$ — while the car's bounding box covers nearly $29\%$ of the domain. The picture shows why: the solids are thin *plates* (`shell2` is $0.17$ tall over a span of $4.2$), so the mask is the plates' own footprint and nothing else. **The dark region enclosed by the car's shell is not a hole — it is background unknowns**, air the solver genuinely marches, nearly still because the duct's openings are its only ways in and out. Reading the picture as "the car is a solid block" would misread both the geometry and the physics.

**$4{,}146$ pixels are clipped.** The scale was fixed at $[0, 2]$ and the field reaches $2.803$, so $7.6\%$ of the drawn picture sits at the top of the colour map and **cannot be read as a measurement**. That is a legitimate choice for a screen — it keeps the free stream mid-scale where the wake is legible — but it is a choice, and the record now carries `field_max` and `clipped_pixels` so the page cannot quietly imply otherwise.

---

# 6. The predictions

Registered in code before the stages ran; the exploration that preceded them is in the record's `read_before_this_run`, and what it had *not* measured is named there too.

| id | claim | outcome |
|---|---|---|
| G1 | the mask is exactly the car's interior at every raster, not just the explored one | **held** — $0$ either way at all three |
| G2 | the background alone would leave at least $3{,}000$ drawable pixels blank | **held** — $3{,}713$ |
| G3 | the mask and the source map do not depend on the state | **held** — identical for a random vector |
| G4 | building the same raster twice gives a bitwise identical operator | **held** |
| G5 | a raster pixel is the same number as a direct probe, to $0.0$ exactly | **held** — $0.0$ |
| G6 | a frame costs under $1\%$ of a march step, both measured in this process | **held** — $0.0364\%$ (§5) |
| G7 | the PNG's car pixels are exactly the mask, in a colour the field cannot make | **held** — $2{,}785$ of $2{,}785$ — **but the claim was too weak** (§5.1) |
| G8 | at an untried raster the linear control holds and the quadratic does not improve | **held** — $3.55\times10^{-15}$ and $5.76\times10^{-6}$ |

Eight of eight — and the two things worth taking from the tier are both failures that a green row would have hidden. The **measurement** behind G6 divided by a step five times too large, in the permissive direction (§5). The **predicate** behind G7 was satisfiable by a picture in which the car is indistinguishable from the wake (§5.1). One was caught by asking what the number was made of; the other only by opening the file and looking at it.

---

# 7. What this tier did NOT do, named

- **The demo does not run on the body-fitted column.** This is the overlay and only the overlay: nothing marches it live, no engine was changed, and `atlas/demo_racelab` is untouched. The column's fluid is one expert (Tier 65), so the window inspector, the three-way mode switch and the learned column have no meaning on it yet — the requirements put the learned expert in the Cartesian background's rectangles, and nothing declares those.
- **The operator is not cached** (W287), so a demo start would pay $63$ s.
- **Only `speed` was drawn.** $u$, $v$, vorticity, pressure and the error-against-referent field of the requirements' §5.1 are all sampled by the same operator, but only speed was encoded and looked at; **vorticity is the one that is not just another `sample` call**, because it is a derivative and the composite's grids are curvilinear.
- **The picture has now been looked at, and looking found something no assertion had** — see §5.1. What is still unchecked is everything below the silhouette: whether the *flow* is right is Tier 64's question, not this one, and no feature of the rendered field has been compared with anything.
- **The body outlines, the grid boundaries and the window tints of the requirements' §5.1 are not drawn**, only the field and the car's silhouette.
- **Nothing was downloaded or installed**, no machine was rented, the unlicensed structural checkpoint was not loaded, and nothing was pushed.

```
python scripts/tier66_car_render.py --out out/racelab15 --stages raster,blank,state,summary
python -m pytest tests/test_tier66_car_render.py tests/test_tier64_car_union.py -p no:cacheprovider
python scripts/vault_scan.py wiki
```

About seven minutes, most of it building the operator at three resolutions on a composite that takes a minute to assemble. The record is `out/racelab15/racelab15.json` and the frame is `out/racelab15/overlay_speed.png`.

---

## See Also

- [[poc3-racelab-car-union]] — Tier 64, the march whose cached state this renders, and the probe operator this generalises.
- [[poc3-racelab-car-graph]] — Tier 65, why the composite's fluid is one expert and its overlap is not a seam.
- [[poc3-racelab-car-solids]] — Tier 62, the solids whose outlines the mask is checked against.
- [[gap-worklist]] — W287, and the rows this tier leaves standing.
