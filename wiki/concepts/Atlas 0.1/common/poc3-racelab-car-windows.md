# Where a learned expert can sit on the body-fitted column

*Case study, PoC 3. New 2026-09-16. Tier 67.*

---

# 0. The result, in one paragraph

**The requirements decided that the background keeps rectangular windows and the learned expert runs only there, because Poseidon-T accepts nothing but a uniform $128\times128$ grid. Nobody had measured what "only there" costs, and the answer is: the learned expert can reach at most a third of the body-fitted column, and none of the third is near the car.** There is no shortage of *places* — $18{,}109$ of $62{,}130$ possible placements ($29.15\%$) are free of holes. The binding constraint is somewhere else entirely: **the body-fitted grids hold $230{,}197$ of the composite's $377{,}267$ unknowns, $61.0\%$**, and a uniform window cannot accept a curvilinear grid at all. So the ceiling — the union of *every* admissible placement, which no tiling can pass — is $33.4\%$ of the composite, and a real greedy tiling of four windows covers $17.4\%$. The $61\%$ the learned expert can never touch is exactly the boundary layers, the wheel clearances and the cooling duct: the physics the body-fitted column was built to resolve. Seven of eight predictions held; the one that failed is in §3 and it failed because the effect it predicted is real but **three times smaller** than guessed.

---

# 1. The arithmetic that fixes the window, with nothing to choose

The body-fitted background is $241\times672$ at $h = 1/64$ — the porous column's own lattice, $240\times672$, plus a row — so `racelab.WX`, `racelab.WY` and the scaling `racelab_switch` derives from them carry over unchanged:

$$128\ \text{cells}\times\frac{1}{64} = 2.0\ \text{length units}$$

and there is no freedom in it. A learned window is a $2.0\times2.0$ square on a domain $10.500\times3.766$, with the car punched through the middle. That single fact drives everything below: the car spans $x\in[0.96, 8.11]$, so the gap upstream of it is only $0.95$ wide — **less than half a window** — and no learned window can ever sit in front of the car at all.

| the background | |
|---|---|
| shape, spacing | $241\times672$ at $h = 1/64$ |
| holes (the car) | $14{,}882$ cells, $9.2\%$ |
| live (carry an unknown) | $147{,}070$ |
| of those, `INTERP` (the fringe) | $2{,}153$ |
| owned (`DISC`, the background's own equation) | $144{,}917$ |

---

# 2. Where a window can go

Every $128\times128$ placement was tested with a summed-area table rather than a loop — $62{,}130$ candidates, each otherwise a $128\times128$ reduction — and the fast test was checked against the honest slow one at every placement on the verification composite and at $400$ sampled placements of the car's: **no disagreement anywhere**. The identity is exact in integers, so any disagreement would have been a bug and not a tolerance.

$\mathbf{18{,}109}$ **of** $\mathbf{62{,}130}$ **placements are admissible**, $29.15\%$. A greedy non-overlapping tiling of them, taken in row order, finds **four** windows:

| window | $x$ | $y$ | where it is |
|---|---|---|---|
| 1 | $8.164 \to 10.164$ | $0.000 \to 2.000$ | the wake, downstream of the car's tail at $8.11$ |
| 2 | $0.008 \to 2.008$ | $0.703 \to 2.703$ | upstream and above |
| 3 | $2.008 \to 4.008$ | $1.125 \to 3.125$ | over the front of the car |
| 4 | $5.680 \to 7.680$ | $1.375 \to 3.375$ | over the rear |

**The count depends on the order the greedy takes placements in**, which is why the order is declared rather than left to whatever `np.nonzero` happens to return: row gives $4$, and column, downstream and upstream each give $5$. Greedy set packing is order-dependent and this is not a maximum — it is a lower bound with its tie-break written down.

---

# 3. The stricter test, and the prediction that failed on it

"Hole-free" is the obvious admissibility test and it is **not the right one**. A window with no holes can still contain `INTERP` cells — the background's fringe, whose values are interpolated *from* the body-fitted grids rather than computed by the background's own equation. A learned expert handed those is handed numbers it does not own, and writing its answer back over them would overwrite the very coupling that makes the composite one implicit solve (Tier 65). So `require="owned"` demands every cell be a `DISC` cell.

**H1 predicted that owning would cost at least $1{,}000$ of the $18{,}109$ placements. It cost $647$** — the effect is real and it is three times smaller than the guess.

| | live | owned | |
|---|---|---|---|
| admissible placements | $18{,}109$ | $17{,}462$ | $-647$, $3.6\%$ |
| greedy tiling, four orders | $4, 5, 5, 5$ | $4, 5, 5, 5$ | **identical** |
| reach, of the composite | $33.41\%$ | $33.24\%$ | $-0.17$ pp |

**Why the guess was wrong, which is the useful part.** The reasoning behind H1 was that the fringe hugs the holes, so a window clearing a hole by a little would still straddle the fringe around it. That is true of *some* windows and irrelevant to most: to be hole-free at $128\times128$ at all, a window must already stand well clear of the car, and the fringe is a thin band of $2{,}153$ cells — $1.3\%$ of the background. The windows that survive the hole test have mostly left the fringe behind already. **The distinction between holding a cell and owning one is conceptually essential and numerically almost free here**, and the tiling does not notice it at all.

---

# 4. The ceiling, and the denominator that decides

A fraction of the *background's* live cells flatters the learned expert, because the background is only $39\%$ of the composite. Both denominators are reported, and the one that decides anything is the composite's:

| | of the live background | **of the composite** |
|---|---|---|
| the four-window tiling | $44.6\%$ | $\mathbf{17.4\%}$ |
| the reach — every admissible placement, unioned | $85.7\%$ | $\mathbf{33.4\%}$ |

**The reach is a ceiling.** No tiling, however clever, can cover a cell that no admissible window contains, so $33.4\%$ bounds every scheme — including one that overlaps windows or re-tiles every step.

And the reason it is a third and not more is not the hole pattern:

$$\frac{\text{body-fitted unknowns}}{\text{composite unknowns}} = \frac{230{,}197}{377{,}267} = 61.0\%$$

which a uniform $128\times128$ grid cannot accept in any arrangement whatsoever. **[AI Inference]:** this is a property of the *checkpoint's interface*, not of the car or of the tiling, so it will not improve with a better packing — only with a checkpoint that accepts a curvilinear patch, or with a body-fitted column whose near-wall region is a smaller share of the unknowns. The first is a different model; the second is a coarser boundary layer, which is the thing the column exists to resolve.

That number belongs beside the demo's expert switch, not in a footnote: on this column, "learned" can mean at most a third of the problem and never the part nearest the car. **W288** is opened for it.

---

# 5. One window, out and back

A window is a view of the composite, not a copy that can drift from it. Taken off the car's composite at $(j=0, i=523)$ — $x\in[8.180, 10.180]$, $y\in[0, 2]$, the wake:

- the block is dense $128\times128$ and **finite everywhere**;
- extracted and written straight back, the solution vector is **bitwise identical**;
- a block changed by $1$ and written back changes exactly $16{,}384 = 128^2$ entries and no others, so a scatter cannot write outside its own window;
- every cell is a `DISC` cell under the owned test.

A window that *does* contain a hole raises rather than returning whatever the index map holds — a checkpoint handed a hole has nothing to read there and no way of being told so.

---

# 6. The predictions

Registered in code before the stages ran; what the exploration had already measured, and what it had not, are both named in the record's `read_before_this_run`.

| id | claim | outcome |
|---|---|---|
| H1 | owning rather than merely holding costs at least $1{,}000$ placements | **failed** — $647$, three times smaller (§3) |
| H2 | the greedy count depends on the order | **held** — $4, 5, 5, 5$ |
| H3 | the ceiling is under $35\%$ of the composite's unknowns | **held** — $33.4\%$ |
| H4 | a window goes out and back bitwise, changing exactly $128^2$ entries | **held** |
| H5 | the summed-area test agrees with brute force everywhere sampled | **held** — $400/400$ |
| H6 | the owned test does not collapse the tiling below $3$ windows | **held** — $4$, unchanged |
| H7 | the owned reach is a subset of the live reach, cell for cell | **held** |
| H8 | a window off the car is dense, finite and all `DISC` | **held** |

Seven of eight. H1's failure cost nothing and bought something: it says the fringe is not the obstacle, which sharpens §4's claim that the obstacle is the body grids' $61\%$ and nothing else.

---

# 7. What this tier did NOT do, named

- **No checkpoint was loaded and nothing learned was run.** This is where a learned expert *could* sit, not a measurement of one sitting there. Poseidon-T's accuracy, its lead ratio and its cost on these windows are all untouched.
- **Nothing was marched.** The windows are placed on a geometry, and no window has been stepped, exchanged or blended.
- **The tilings are greedy, not maximal.** Four orders are declared and measured; no maximum-packing argument is made, and the counts are lower bounds.
- **The fringe question is measured, not settled.** `require="owned"` excludes `INTERP` cells from a window, but nothing here says what a learned expert should *do* at a window edge that abuts one — that is the seam question, and this tier does not declare a seam.
- **No graph, no agents, no compile.** Tier 65 declared the column with one fluid expert; splitting the background into window agents would change that declaration, and nothing here does.
- **The machine slept for about seven hours mid-run**, between the launch and the first stage. Nothing in this tier is timed — every prediction is combinatorial or exact — so the suspension corrupts nothing, and it is recorded here rather than left to be noticed later.
- **Nothing was downloaded or installed**, no machine was rented, the unlicensed structural checkpoint was not loaded, and nothing was pushed.

```
python scripts/tier67_car_windows.py --out out/racelab16 --stages territory,owned,window,summary
python -m pytest tests/test_tier67_car_windows.py -p no:cacheprovider
python scripts/vault_scan.py wiki
```

About two minutes, nearly all of it the composite the first stage builds. The record is `out/racelab16/racelab16.json`.

---

## See Also

- [[poc3-racelab-car-render]] — Tier 66, the overlay that draws this composite, and the same $61\%$ seen as pixels.
- [[poc3-racelab-car-graph]] — Tier 65, why the composite is one implicit solve and its fringe is not a seam.
- [[poc3-racelab-car-union]] — Tier 64, the march these windows would sit on.
- [[gap-worklist]] — W288, and the rows this tier annotates.
