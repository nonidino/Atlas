# The body-fitted car as a demo column, and the cache that starts it

*Case study, PoC 3. New 2026-09-16. Tier 68.*

---

# 0. The result, in one paragraph

**Tiers 60–67 built a second column — real walls, an open duct, the radiator core and the recovery turbine marching in it, a declaration that compiles, an overlay, and a measured learned-expert territory — and none of it had ever been on a screen.** This puts it there, as a *second* column beside the porous one rather than in place of it (requirements 13.3), because the porous column is the only place §12's window-flipping criteria can be told over fourteen windows. Two things had to be built before it could start at all. **The probe operator is now cached** (W287, closed): $64.3$ s becomes $0.09$ s, a $715\times$ saving — and the cache **validates itself against the live composite instead of trusting its key**, which catches a mask wrong by a *single* pixel in either direction. And **the column had to be given a release state that belongs to its machine**: released from Tier 64's device-free prefix, the rotor sits at $1.292$ of its sizing inflow against a band ending at $1.1408$ and is outside its envelope on **every one of 60 steps**, closing at $0.0013$ a step — about three minutes of a demo stamped `OUTSIDE` before it says anything trustworthy. Settled once to $t = 12$ with the devices on and cached, it starts at $1.0348$ and **no step is outside**. Six of seven predictions held; the one that failed had been refuted by its own stated reason before any measurement (§5.1).

---

# 1. What this column can and cannot offer

It is worth being exact, because the honest offer is narrower than the porous column's and the narrowness is the interesting part.

**The fluid is one expert** (Tier 65). The composite is a single implicit solve whose interpolation rows sit in the same matrix as its momentum rows, so no grid can be stepped alone and there is nothing to flip *inside* it.

**A learned expert can reach at most a third of it** (Tier 67). Poseidon-T accepts nothing but a uniform $128\times128$ grid; the body-fitted grids hold $61.0\%$ of the unknowns; a uniform grid cannot accept a curvilinear patch in any arrangement. Four windows exist, they cover about a sixth of the unknowns, and none of them is near the car.

So what this column offers is the real physics drawn, the devices and joins live, and a **labelled** statement of the $61\%$ no learned expert can ever touch — in the shape §4.3 already uses for families with no learned option: greyed out *with the reason*, never hidden.

## 1.1 Three things are refused with a reason rather than left missing

| what | why it is not offered |
|---|---|
| **vorticity** | it is a *derivative*, and on twelve overlapping curvilinear grids that is an operator this column does not have — not another sample of the probe |
| **temperature** | the coolant loop is lumped and owns no region, so there is no thermal field over the fluid; return, block and disc temperatures are numbers in the telemetry, which is what §5.3 asks for |
| **the referent march** | with no window running a learned expert, **the all-classical march IS this march** — a referent would cost a second $0.6$ s solve per step to measure an error that is identically zero |

That last one is worth dwelling on. §5.3 asks for the global rms error against an all-classical referent and requirements 13.3 turned it off by default *for speed*. Writing the column showed the speed argument is the weaker one: until a learned window exists there is nothing for a referent to differ from, so running it would produce a confident-looking $0.0$ that measures only that the same arithmetic was done twice. §5.4's third rule is untouched — the per-window one-step error needs no referent and stays always on.

---

# 2. The cache, and why it validates instead of trusting its key

The operator is a function of the geometry and the raster alone — Tier 66 measured that it is bitwise identical on a second build and unmoved by the state — so it can be written to disk. The risk is the ordinary one: a key covers the inputs somebody remembered to hash, and a cache served because nobody thought of an input is exactly the silent wrongness this package refuses.

So a loaded operator is **put through the tier's own positive control** before it is used: a linear field laid on the *live* composite must come back through it at rounding, and a sample of masked pixels must still be unreachable by the donor search. Both are milliseconds against a $63$ s rebuild, so validating always is free.

| planted fault | caught by | verdict |
|---|---|---|
| a wrong key | the key check | refused |
| the operator's column indices rolled | the linear control, at $1.08$ | refused |
| the mask **one pixel** too large | re-probing sampled masked pixels | refused |
| the mask **one pixel** too small | the linear control (an unmasked pixel with an empty row returns $0$, which is not the plane) | refused |
| every pixel masked | the degenerate-mask check | refused |

Fixing that last case found a real defect in Tier 66's control: `linear_control` reduced over an empty selection and *raised* when nothing was drawn, so a raster that drew no pixels at all would have crashed rather than failing cleanly. It now returns $\infty$ — a picture of nothing passes no control.

---

# 3. A release state belongs to its machine

This is the third time this project has met the same defect, and the first time it was predicted rather than diagnosed.

Tier 64's cached prefix is at $t = 8$ and **device-free**: it was marched with the devices applying nothing. The machine is sized for $u_{\text{host}} = 0.0829$ and generates only for $0.9731 \le u/u_{\text{host}} \le 1.1408$ (W282). A device-free duct runs faster than a ducted one, so releasing there and switching the devices on puts the rotor at $1.292$ — outside, immediately.

| released from | ratio at step 1 | ratio at step 60 | steps outside |
|---|---|---|---|
| the device-free prefix at $t = 8$ | $1.2921$ | $1.2086$ | $\mathbf{60}$ of $60$ |
| **settled to $t = 12$ with the devices on** | $\mathbf{1.0348}$ | $1.0216$ | $\mathbf{0}$ of $60$ |

The transient closes at about $0.0013$ a step, so roughly $300$ steps — three minutes — would pass with every frame stamped `OUTSIDE THE MODEL` before the column said anything a viewer should believe. Tier 64 met the same transient and judged its arms over a window beginning at $t = 12$ for exactly this reason.

So the column **settles once** to $t = 12$ with its devices on, caches that state, and releases from it forever after. The settle costs about $540$ s on battery, paid once per car.

**[AI Inference]:** the general shape is that a cached state carries not just a field but the *configuration it was settled under*, and a demo that releases from someone else's settled state inherits their configuration silently. The fingerprint already guards the geometry (W250) and the outlet condition (W258); the machine's sizing is a third such axis and nothing currently hashes it.

---

# 4. What it costs to start, and to run

| | cold | warm |
|---|---|---|
| the composite | $66.0$ s | $90.1$ s — **not cached**, and the machine was slower (§5.1) |
| the probe operator | $64.3$ s | $\mathbf{0.12}$ s |
| the flow, the devices | $3.7$ s | $5.7$ s |
| the settle to $t = 12$ | — | $\mathbf{0}$ s, cached |
| **total** | $\mathbf{134.1}$ s | $\mathbf{96.0}$ s |

The two columns were measured minutes apart on a machine whose reference matmul moved from $0.0099$ s to $0.0155$ s between them, so the composite's $66.0$ and $90.1$ are the *same build* at two clock speeds and not a regression.

The composite is deliberately *not* cached: it is a live object graph of grids, donor searches and index maps, and pickling it would be a second definition of the car that could drift from the first. It is the remaining start-up cost and it is named rather than hidden.

Marching runs at $1.63$ steps a second on battery at $1.4$ GHz, and $1.84$ on mains — the same measurement, on a machine that throttles by $2.7\times$, which is why the page quotes both and neither alone.

---

# 5. The predictions

Registered in code before the stages ran; what the exploration had already measured, and what it had not, are both named in the record's `read_before_this_run`.

| id | claim | outcome |
|---|---|---|
| K1 | with both caches warm the column builds in under a quarter of the cold $134.1$ s | **failed** — $96.0$ s, $0.716$ of it (§5.1) |
| K2 | a warm build does zero settling and still starts inside the envelope | **held** — settle $0.0$ s, ratio $1.0348$ |
| K3 | released from the settled state, no step of the run is outside the envelope | **held** — $0$ of $30$ declined |
| K4 | every offered field draws the car in exactly as many pixels as the raster masks | **held** — $2{,}785$ of $2{,}785$, all four |
| K5 | the validator rejects a corrupted operator built on the **car**, not only on the verification composite | **held** — refused at $4.198$ |
| K6 | every field not offered, and the referent, carry a reason | **held** |
| K7 | the settled state reloads deterministically, to $10^{-9}$ | **held** — delta exactly $0.0$ |

## 5.1 K1 failed, and its own reason had already refuted it

K1 claimed a warm build under a quarter of the cold $134.1$ s. The *why* registered beside it reads: *"the raster's $64.3$ s becomes $0.09$ s and the settle is skipped entirely, leaving the composite, the flow and the devices, which came to $69.7$ s."*

$69.7 / 134.1 = 0.52$. **The justification refuted the claim before a single measurement was taken**, and nothing checked the two against each other — which is precisely the drift this vault has a row about, in its least excusable form, because here both halves were written in the same breath. The measurement came in at $96.0$ s, $0.716$.

The structural fact the prediction should have stated is the one §4 already gives: **the composite is not cached and it dominates.** Caching the operator removed $64.3$ s of $134.1$; nothing can remove the composite's $66$–$90$ s without caching a live object graph of grids, donor searches and index maps — which would be a second definition of the car, free to drift from the first. **W289** is opened for it.

**One caveat on the number itself.** The $0.716$ divides a numerator measured at $11{:}49$, on mains restored minutes earlier and still ramping (composite $90.1$ s, reference matmul $0.0155$ s), by a denominator measured earlier on warm mains (composite $66.0$ s, matmul $0.0099$ s). It is not a clean like-for-like ratio. It does not rescue the prediction — on the same machine state the figure would be about $0.52$, still twice the claim — but a ratio whose halves were taken in different machine states is exactly what this project keeps telling itself not to quote, and it is recorded here rather than tidied away.

---

# 6. What this tier did NOT do, named

- **There is no page.** This is the engine: the column builds, marches, draws and reports. The dashboard that switches between the two columns and shows the $61\%$ is the next tier, and `atlas/demo_racelab/static/index.html` is untouched.
- **No checkpoint is loaded and no window runs a learned expert**, so the mode switch, the presets and the referent have nothing to act on yet — which is why the referent refuses with a reason rather than running.
- **The composite is not cached**, so a start still costs about a minute.
- **No knob moves anything.** §3.3's parameters are not wired on this column; that is criterion 2 and its own tier.
- **The porous column is untouched** — its engine, its records, its gate and its bundle are exactly as they were.
- **Nothing was downloaded or installed**, no machine was rented, the unlicensed structural checkpoint was not loaded, and nothing was pushed.

```
python scripts/tier68_bodyfitted_demo.py --out out/racelab17 --stages build,settle,frames,summary
python -m pytest tests/test_tier68_bodyfitted_demo.py -p no:cacheprovider
python scripts/vault_scan.py wiki
```

The record is `out/racelab17/racelab17.json`.

---

## See Also

- [[poc3-racelab-car-render]] — Tier 66, the overlay this column draws with, and the operator now cached.
- [[poc3-racelab-car-windows]] — Tier 67, the $61\%$ this column has to label.
- [[poc3-racelab-car-union]] — Tier 64, the march, the machine's band and the window this release state sits in.
- [[gap-worklist]] — W287 closed here, and the rows this tier leaves standing.
