# The Reuse Probe — is a certificate a property of the expert, or of the state?

**Type:** Concept page — **case-study summary** (folder: `Atlas 0.1/common/`)
**Status:** opened 2026-08-31. The short, readable companion to [[tier0-measurements]] §20, which is the full measurement record. Everything here is quoted from `out/w106/w106.json`; nothing is estimated.
**Code:** `atlas/cases/reuse_probe.py` (the declarations and the probe design), `scripts/w106_reuse_probe.py` (the driver), `tests/test_tier20_reuse_probe.py`
**Related:** [[tier0-measurements]] · [[gap-worklist]] · [[case-study-ladder-to-f1]] · [[case-study-scaling-ladder-atlas-0.1]] · [[expert-library-atlas-0.1]] · [[plug-in-composition-theorems]] · [[probed-dtn-coupling]] · [[composition-error-theory]] · [[f1-pathmap-and-end-goal]] · [[atlas-and-standard-dd-theory]]

---

## 1. The question, and why the economics turn on it

Atlas's promise is that a physics expert can be certified once and then plugged
in wherever its ports match. [[f1-pathmap-and-end-goal]] §1.2 prices the whole
programme on it: search a design space wide and cheap, because you are reusing
certified pieces rather than re-verifying a new simulation each time.

Every certificate in this vault has been issued at a **probe state**.
`probe.probe_block` perturbs an interface trace and watches the response, so for
a nonlinear expert the object it returns is the *tangent map at the state it was
poked in* — [[atlas-and-standard-dd-theory]] says this outright and adds that
every quantity read off it is local to that linearization. Nine tiers have
quoted $\beta$, $\Xi$, $\tau$ and a substitution verdict, **each measured once**,
and no page has ever asked what a second reading would say.

A twenty-agent car carries roughly sixty certificates. So:

> **If a certificate is a property of the expert**, you certify once and reuse
> across the design space, and the economic argument holds.
> **If it is a property of the state**, the plug-in claim is *per-design*, and
> every candidate design re-pays the whole certification cost.

## 2. How it is built

No new physics and no new graph. CS-7's ladder ([[case-study-scaling-ladder-atlas-0.1]])
already supplies the geometry and the trajectories; what CS-8 adds is a **probe
design** over them. One swap everywhere — `reference.WindowNS` $\to$ Poseidon-T
at one fluid agent — measured at $55$ cells.

**Four factors, and nothing else moves.**

| factor | what varies | held |
|---|---|---|
| **state** | macro-step $0, 10, 30, 60, 110$ of one trajectory | the seam, the graph, both experts |
| **seam** | the flow regime the seam sits in | the state, the graph, the port declaration |
| **geometry** | $6$ windows against $12$ | the seam ID, the state |
| **replicate** | two seams the *rule* calls the same regime | everything else |

**The regimes are a rule over the layout, not a list.** Count the rotors of a
seam's own window row lying upstream of it: zero is `near-freestream`, one is
`shallow-wake`, two or more is `deep-wake`, and a face a rotor plane splits is
`bypass-clean` or `bypass-wake` by the same count. That is decidable **before the
march runs**, which matters — a label assigned after looking at the answer is a
description of the answer.

**Every probe state comes off the one classical trajectory this vault trusts.**
[[tier0-measurements]] §19.6: the classical column with its elliptic part inside
the agents is not finite past macro-step $82$ at six windows, and a probe on a
diverging trajectory measures the divergence. So CS-8 marches
`wake_array.exposed_reference_solver` — elliptic part *out* of the agent — with
one `assembly.ProjectedAssembly` after the blend, and Tier 18's published
wake-array state is excluded for exactly that reason. Both marches reproduce
§19.6's column on an independently written driver, to three digits.

## 3. Both controls, and both had to pass first

**Control ZERO — the state where the answer is known.** At the freestream every
window holds the same uniform field and every port of a group is the same face
with the same modes, so seams the taxonomy calls *different regimes* must return
the *same operator*. They do: worst range **exactly $0.0$** over all eight
certificate quantities, in all three port groups, by `==` rather than to a
tolerance. Had it not, the regime labels would be reading the port rather than
the flow and §5 below would be uninterpretable.

**Control FLOOR — the same cell, twice.** State reloaded off disk, experts
constructed fresh, graphs rebuilt, operators re-probed. Four cells: **exactly
$0.0$** disagreement, every quantity. The pipeline is bit-reproducible.

That second result is good news and a problem, and the problem is worth stating
because it is a finding about the framework rather than about the expert.
`reuse_probe.travel_verdict` asks *did the movement exceed the floor* — and
against a floor of zero, a movement of $0.037\%$ and a movement of $155\%$ both
answer *yes*. **The test the framework brought to this question does not
discriminate** (**W106**). What replaces it is the quantity's own level and a
same-regime replicate.

## 4. The result

Median movement over each factor, as a fraction of the level the quantity sits
at. Read the columns against each other.

| quantity | **state** | **seam** | **replicate** | **geometry** |
|---|---|---|---|---|
| $\lVert S_i\rVert$ | $2.49\%$ | $1.99\%$ | $0.037\%$ | $0.199\%$ |
| $\lVert\Lambda^{\text{expert}} - \Lambda^{\text{ref}}\rVert$ | $3.63\%$ | $2.97\%$ | $0.228\%$ | $0.156\%$ |
| $\beta$ | $6.46\%$ | $3.99\%$ | $0.769\%$ | $0.340\%$ |
| $\Xi$, the composability index | $55.9\%$ | $53.9\%$ | $20.5\%$ | $1.61\%$ |
| `visible_above` | $155\%$ | $79.2\%$ | $3.56\%$ | $2.70\%$ |
| `fails_above` | $155\%$ | $96.0\%$ | $5.32\%$ | $3.66\%$ |

**Three orders of magnitude separate the most portable quantity from the least,
and they are all fields of one certificate** (**W107**). The norms travel to a
few percent. The two $\beta_{\min}$ thresholds move by *more than their own
size*.

**Why, and it is arithmetic rather than a property of this checkpoint.** Both
thresholds are $\beta$ minus a norm:

$$\texttt{visible\_above} = \beta - \lVert S_i\rVert,\qquad
\texttt{fails\_above} = \beta - \lVert\Delta\rVert$$

Here $\beta \approx 0.21$ and both norms are $\approx 0.22$, so each threshold is
$\sim 10^{-3}$ — two orders below the terms it is built from. A $3\%$ movement in
one against a $6\%$ movement in the other lands as $155\%$ in their difference.
**Any certificate decided near $\lVert\Delta\rVert \approx \beta$ inherits that
amplification, and that is exactly the regime a *good* substitution is in.**

## 5. Three things the factors say that the headline does not

**The dependence is local.** `x0r1_full` is near-freestream by construction — no
rotor upstream in its row — and over $110$ macro-steps its $\beta$ moves
$0.594\%$ and its $\Xi$ moves $4.77\%$. The deep-wake seam *in the same graph on
the same trajectory at the same steps* moves $26.1\%$ and $59.4\%$. A certificate
is state-dependent exactly to the extent that its own seam's flow is, which means
a seam in undisturbed flow keeps a portable certificate inside a graph whose
other seams do not.

**Most of the movement is over by macro-step 30.** At the deep-wake seam the
thresholds move by a factor of $44.8$ between the freestream and step $30$, and
by $1.6$ between step $30$ and step $110$. The state-dependence is overwhelmingly
*freestream versus developed*, not *developed versus developed* — the more
hopeful of the two shapes, and the one a library could exploit.

**Doubling the graph is twenty to forty times cheaper than developing the flow.**
Every quantity moves under $4\%$ across the two rungs. That is the wrong way
round for the library argument: reuse across *scenarios* is what the economics
need, and reuse across *sizes* is what the measurement gives.

## 6. The verdict travels — and that is the least informative true thing here

Swept over the whole admissible tolerance axis at all $55$ cells, the
substitution verdict is **`refuse`, everywhere**. Perfectly portable.

It is portable because it is **saturated**. Both thresholds are negative exactly
when $\lVert\Delta\rVert > \beta$, so on the non-negative axis the verdict is
`refuse` if and only if that ratio exceeds $1$. Measured:

$$\frac{\lVert\Delta\rVert}{\beta} \in [\,1.00254,\ 1.38952\,],\qquad
\text{never below } 1$$

The minimum is a **quarter of one percent** from the value at which the verdict
changes — and it occurs at the seam with the *smallest* state-dependence in the
study. The distance from that sign change varies by $153\times$ across the grid.
A stability claim read off a saturated verdict is not a stability claim
(**W109**).

The same shape appears in W81's *derived* tolerance:
$\varepsilon_{\text{tol}} = \min(\tau,\sigma)$ moves by $35\times$ at one seam
down one trajectory, $\tau$ by $465\times$, and at the freestream it does not
exist at all — `seam_defect_split` correctly refuses a relative defect where no
power crosses the reference interface. **W81 closed *where does the number come
from*; CS-8 opens *which state is it the number for*** (**W108**).

## 7. What it means for the expert library

| the claim | the verdict |
|---|---|
| a certificate is a property of the **expert** | **false as stated** — two of its eight quantities move by more than their own size with the probe state |
| a certificate is a property of the **design** | **also false**, and more surprisingly so: graph size costs $20$–$40\times$ less than flow development |
| the plug-in claim is **per-design** | **not shown** — the verdict is invariant everywhere, but only because it is saturated |
| a certificate can be **indexed by something cheap** | **supported, and it is the practical result** |

That last row is the one to carry forward. A regime label computed from the
layout with **no run at all** captures between $5$ and $53$ times the
seam-to-seam variation, on every quantity except $\Xi$. So a library would carry
neither one certificate per expert nor one per design, but **one per expert per
regime** — a cost that grows with the vocabulary of flow situations rather than
with the number of designs searched. Weaker than the pathmap's claim, and very
much cheaper than re-certification.

**The exception matters and is on the record.** $\Xi$ — the axis
[[expert-library-atlas-0.1]] and [[probed-dtn-coupling]] §4.5 propose to *rank* a
library by — disagrees by $20.5\%$ in the median and up to $49\%$ between two
seams the taxonomy calls the same regime. It is a ratio of two norms that each
moved a couple of percent in opposite directions, so it inherits both. The axis
is real; **its reproducibility had never been quoted**, and a ranking on it can
reorder two experts that are genuinely a fifth apart.

**[AI Inference]:** the shape of §5 — nearly all the movement before step $30$ —
suggests the right index is not *time* but *whether the seam's own flow has
developed*, which a probe already measures. A certificate could then carry an
admissibility envelope in probe-base coordinates rather than a scalar, and
`SeamOperator.probe_state_agrees` (W77, already derived rather than declared) is
where it would live. This is a design proposal fitted to one geometry at one
Reynolds number and nothing here tests it.

## 8. Honesty list

- **One expert pair, one governing family, one Reynolds number.** Every number
  above is `WindowNS` against Poseidon-T in 2-D incompressible flow at
  $\mathrm{Re}_D = 255$. Nothing here says how a different architecture's
  certificate travels.
- **Two graph sizes, not a ladder.** The geometry factor is $6$ against $12$
  windows. $N=24$ was never marched at this horizon (§19.6, wall time), so the
  geometry column is a difference of two points and not a trend.
- **The thresholds are negative at every cell**, so the informative window sits
  entirely at inadmissible tolerances and the verdict could not have moved on
  this expert pair. A pair with $\lVert\Delta\rVert$ nearer $\beta$ would test
  what this one cannot.
- **$\tau$ is measured on a cross, not the grid** — every seam at the developed
  state, two seams down the whole trajectory — because each split runs a
  `tight_couple` and a finite-difference Jacobian.
- **The divergence guard's ratio form never fired** (**W110**): its reference is
  the first snapshot, which is the freestream, where the divergence is
  identically zero. The absolute values were checked and are healthy; the ratio
  was inert.
- **The regime taxonomy is this geometry's.** It works because `rotor_motif` puts
  every rotor of a row in line, so a row has one wake. A layout with staggered
  rotors would need a different rule, and whether the *result* survives that is
  untested.

---

## See Also

- [[tier0-measurements]] — §20 is the full record, with every table and its provenance
- [[gap-worklist]] — Tier 20, and W106–W110 with their definitions of done
- [[case-study-ladder-to-f1]] — the plan CS-8 is the second row of, and §7's rung-4 criterion
- [[case-study-scaling-ladder-atlas-0.1]] — CS-7, which supplied the geometry, the trajectory and the two instruments this one re-reads
- [[expert-library-atlas-0.1]] — the two measured axes, one of which this tier prices
- [[plug-in-composition-theorems]] — the substitution guarantee whose discriminating power is what moved
- [[probed-dtn-coupling]] — §4.5's composability index, and the probe this tier ran fifty-five times
- [[atlas-and-standard-dd-theory]] — the nonlinear-substructuring reading that predicted this result and had never been tested
- [[f1-pathmap-and-end-goal]] — rung 4, *the one most likely to be skipped by accident*
