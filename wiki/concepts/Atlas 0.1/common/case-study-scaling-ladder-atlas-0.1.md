# The Scaling Ladder — does composition error grow with the number of interfaces?

**Type:** Concept page — **case-study summary** (folder: `Atlas 0.1/common/`)
**Status:** opened 2026-08-31; §5 rewritten the same day, when marching its own positive control to $120$ steps falsified it and W100 closed on a different repair. The short, readable companion to [[tier0-measurements]] §19, which is the full measurement record. Everything here is quoted from `out/w100/w100.json`; nothing is estimated.
**Code:** `atlas/cases/scaling_ladder.py` (the declarations), `scripts/w100_scaling_ladder.py` (the driver), `scripts/w100_timing.py`, `scripts/w100_frames.py` (the viewer's data), `tests/test_tier19_scaling_ladder.py`
**Related:** [[tier0-measurements]] · [[gap-worklist]] · [[case-study-ladder-to-f1]] · [[cs7-scaling-ladder-pickup]] · [[case-study-wake-array-atlas-0.1]] · [[f1-pathmap-and-end-goal]] · [[master-error-bound]] · [[composition-error-theory]] · [[atlas-implementation]]

---

## 1. The question, and why it is the one that decides

Every argument for composing physics experts rests on the same unmeasured claim:
that gluing more pieces together does not make the error grow faster than the
number of joints. [[f1-pathmap-and-end-goal]] names it **F1** and says the program
fails if composition error is *super-linear in interface count*. It calls rung 9 —
fifteen to twenty coupled agents — *the rung that decides everything*, and
schedules a cheap sweep as its early warning.

**That sweep had never been run at any size.** The largest real graph in this
vault was nine agents, and no page anywhere carried a number for how error scales.
Meanwhile [[case-study-ladder-to-f1]] schedules nine further case studies on the
answer, and says outright that the final one is not built if it comes back
super-linear.

So: take one geometry, grow it from **1 to 24 coupled windows**, hold everything
else still, and report an exponent with an error bar.

## 2. How it is built

The parent is [[case-study-wake-array-atlas-0.1]] — real turbines at $3.5\,D$ in
an L, $128$-cell Poseidon-T windows of $4\,D$ each, zero-parameter actuator disks,
and a `reference.WindowNS` column beside the checkpoint's. This is that geometry
with the **size** as the variable and everything else nailed down: the cell, the
macro-step, the $16$-cell overlap, the $8$-cell ramp, the referent's viscosity and
the disk model are identical at every rung.

| $N$ | tiling | domain | rotors | seams | overlapping window pairs |
|---|---|---|---|---|---|
| $1$ | $1\times1$ | $4.0 \times 4.0\ D$ | $0$ | $0$ | $0$ |
| $2$ | $2\times1$ | $7.5 \times 4.0\ D$ | $1$ | $3$ | $1$ |
| $6$ | $3\times2$ | $11.0 \times 7.5\ D$ | $3$ | $13$ | $11$ |
| $12$ | $4\times3$ | $14.5 \times 11.0\ D$ | $5$ | $27$ | $29$ |
| $24$ | $6\times4$ | $21.5 \times 14.5\ D$ | $12$ | $62$ | $68$ |

Two columns run at every rung, and the reason is not symmetry. A classical solver
has a **monolith** — the same discretization on the undivided domain — so its
composition error can be measured against the right answer. A checkpoint frozen at
$128\times128$ has no monolith at any resolution ever (**W95**), so its column is
measured reference-free. The classical column is what calibrates that surrogate;
it is load-bearing rather than a courtesy.

**Two of the five rungs are controls and both pass.**

- **$N=1$ has no interface at all.** The partition of unity is identically one,
  cutting and assembling are the identity, and the composed step *is* the
  monolith — so the composed defect must be **exactly zero**. It is, bit for bit,
  not to a tolerance. This is [[tier0-measurements]] §8's own lesson: the control
  that would have redirected the parent's whole search was the one nobody ran.
- **$N=6$ is the wake array.** Its array loss must come back at the published
  numbers or the ladder is measuring the harness rather than the physics. It does:
  $25.27\%$ and $14.62\%$, to the digit.

## 3. The answer

| series | exponent | 95% CI | reading |
|---|---|---|---|
| classical, reference-free | $\mathbf{+0.804}$ | $[+0.641,\ +0.967]$ | sub-linear |
| frozen checkpoint, reference-free | $\mathbf{+0.477}$ | $[+0.362,\ +0.593]$ | sub-linear |
| classical, against the monolith | $\mathbf{+0.851}$ | $[+0.748,\ +0.954]$ | sub-linear |

> **F1 is not falsified.** No column's interval lies above $1$ over the full
> $24\times$ range.

Read as F1 asks and no further. A criterion that was not falsified is not a
theorem: this is five rungs, one geometry, one Reynolds number, two experts, and
the largest graph is $24$ agents against rung 9's $15$–$20$. **It licenses
building Phase C. It does not establish Claim B.**

One thing in the table deserves suspicion rather than comfort. The classical
column's *local* slope over the last segment — $29$ to $68$ interfaces — is
$+1.005$, and the checkpoint's is $+0.537$. The fitted interval clears $1$; the
last segment does not.

## 4. The control that turned out to matter more than the answer

**A bigger array is not only more interfaces. It is harder physics.** At $N=24$
the flow reaches $u_{\min} = 0.183$ against $N=2$'s $0.509$ — deeper wakes,
stronger gradients. A defect growing with $N$ might be growing because the flow
got worse, not because the cut got longer, and §3's measurement cannot tell them
apart.

So the same three numbers were measured again with **one turbine running at every
rung**. The flow near it is then identical at every size and every window added
sits in near-freestream, so anything that still grows is growing on interface
count alone.

| | headline | physics held fixed |
|---|---|---|
| classical, reference-free | $+0.804$ | $\mathbf{+0.138}$ |
| checkpoint, reference-free | $+0.477$ | $\mathbf{+0.201}$ |
| classical, against the monolith | $+0.851$ | $\mathbf{+0.160}$ |

**A $68\times$ increase in interfaces buys a $2.2\times$ increase in composed
defect.** Most of the headline is the flow, not the cut.

This control was not in the brief and it should have been. Stated as the warning
it is: **a sub-linear exponent measured on a ladder whose physics grows with its
size is a weaker result than it reads as** — and no other case study in this vault
has separated the two.

## 5. What the run was not looking for

**The classical composed rollout is unstable.**

Continue the wake array's own march past the $60$ macro-steps it published. The
composed `WindowNS` column leaves the band between step $70$ and $80$ at $N=6$ and
is not finite by $82$. The monolith, from the same state under the same forcing,
holds $u_{\max}$ at $1.33$ throughout. The frozen checkpoint's column is stable to
$110$.

The mechanism is one line. Each window returns a field that is divergence-free
*on its own window*, and a partition-of-unity blend of two divergence-free fields
is not — on the overlap, where $\chi_1 + \chi_2 = 1$,

$$\nabla\cdot(\chi_1 u_1 + \chi_2 u_2) \;=\; \nabla\chi_1\cdot(u_1 - u_2)$$

**The assembly creates divergence exactly where the two local solves disagree,
and nothing in the classical column removes it.** The assembled divergence grows
with the graph — $0.0132 \to 0.0957 \to 0.1507 \to 0.2502$ — and the blow-up
arrives earlier as it does: macro-step $14$ at $N=6$, $9$ at $N=12$, $8$ at $N=24$.

**The repair is one operator, and it is one the framework already prescribes.**
Applying a single global Leray projection to the *assembled* field each
macro-step — which is what the checkpoint's column does, and what `R10b` says to
do with an exposed elliptic part — makes every rung stable and holds the
divergence flat or falling. Nothing else changes.

> So `L2/R10` — the rule that refuses to decompose an agent whose elliptic
> subsolve is `embedded` — is **vindicated by measurement for the first time.** It
> has refused the reference column of every graph in this vault since Tier 0, and
> nothing had ever shown the refusal was about anything real. It is: the graph it
> refuses goes unstable, the graph that exposes its elliptic part does not, and
> the difference is a single operator.

And one consequence that has to be said plainly: **Tier 18's published state sits
on a diverging trajectory.** Every number measured *at* that state stands — the
state exists and the instruments read what they read — but the trajectory does not
continue, and $60$ macro-steps is ten to twenty short of where it would have
shown.

**Marched to $120$ steps the same day, the repair above is falsified — and the
one that replaces it is a rule this framework already had.**

$20$ macro-steps from the developed state is what "makes every rung stable" was
measured over. At $120$, from the freestream, at $N=6$, reporting the step at
which $\lvert u\rvert$ leaves the band $3$:

| agents | composition layer | leaves the band at |
|---|---|---|
| embedded | nothing | $74$ (not finite by $82$) |
| embedded | one global **spectral** Leray projection | $\mathbf{51}$ |
| embedded | one global **Neumann** Leray projection | $\mathbf{33}$ |
| **exposed** | nothing | never — and $\lVert\nabla\cdot u\rVert = 1.25$ |
| **exposed** | one global spectral Leray projection | **never, to $120$** |

**Adding a projection to agents that already project makes it worse.** Each
`WindowNS` window has already answered the disk's momentum sink on its own
subdomain, with its own Neumann solve, inside its own sub-steps; a global solve
after the assembly answers it again, and the pressure is applied twice. From the
developed state the same control that §5 reports as stable over $20$ dies at
$\mathbf{38}$ — so the positive control *is* stable exactly as far as it was
marched, which is Tier 18's mistake happening again one level down.

**And the divergence is not the proximate cause.** The Neumann variant holds
$\lVert\nabla\cdot u\rVert_{\text{rms}}$ at $0.0089$ against the bare column's
$0.69$ — $78\times$ cleaner — and dies soonest of the three. The mechanism §5
states is real and the effect it was charged with is not its alone.

What survives is the arrangement **`L2/R10` has prescribed since Tier 0**, and
that no classical column in this vault had ever been built to satisfy: take the
elliptic part *out* of the agent and let the composition layer apply it **once**,
to the assembled field. `wake_array.exposed_reference_solver` is that agent —
`_project` replaced by the identity, the velocity update untouched, which is the
composition layer declining to use *part of* an expert rather than editing one.

> **It is the first classical column in this vault that `R10` does not refuse**,
> and it is the one that survives $120$ macro-steps. `R10` clears, `R10b` admits
> the cadence, `R12` admits the assembly, and the graph goes from `refuse` to
> `admit-uncertified`.

So the assembly is now a declared pair — `assembly.ProjectedAssembly` carries a
partition of unity **and** an `assembly.ConstraintProjection`, `L6/C2` is the
condition, `R12` decides it from the declaration, and
`emit.HarnessParameters.assembly_projection` puts it on every emitted defect
(**W54's fifth instance**). The statement generalizes past two windows: for a
linear $\mathcal C$ with $\mathcal C u_i = 0$ on every subdomain,

$$\mathcal C\Big(\sum_i \chi_i u_i\Big) = \sum_i [\mathcal C, \chi_i]\,u_i
= \sum_i \nabla\chi_i\cdot u_i = \sum_i \nabla\chi_i\cdot(u_i - w)$$

for **any** $w$, since $\sum_i \nabla\chi_i = \nabla(1) = 0$. The residual is a
functional of the *disagreement*, manufactured inside the overlap rather than
transported into it — which is why no halo width touches it, and why it is
nobody's agent defect. And at $N=1$, $\nabla\chi \equiv 0$: there is no
commutator, the two assemblies are *the same operator*, and **no case study run
at a single size could have found any of this**.

**One thing this leaves open, and it is new.** The control in the fourth row —
exposed agents with no global projection — runs all $120$ macro-steps without
leaving the band at $\lVert\nabla\cdot u\rVert = 1.25$. **Stable, and not
incompressible.** `R10` moves the elliptic part out, `R10b` fixes the cadence it
must be applied at and assumes it happens, `R12` is silent because L6/C2's
hypothesis genuinely fails on an all-exposed graph — so a graph can clear every
rule and never apply the operator. That is **W105**, and it is the first place
here where *a rollout that does not blow up* and *a rollout that solves the
equations* come apart.

## 6. What the three instrument repairs bought

The run could not be trusted until three defects in its own instruments were
fixed, and two of them turned out to be bigger than their descriptions.

**W58 — which bound was measured.** `cut_defect_bound` had two readings and the
record never said which. That is now a field, with four named forms and a
compile-time check of the geometric condition that makes two of them equal. Then
the measurement found a hypothesis nobody had stated: one reference-free form is a
**max over neighbour pairs**, the bound is a **norm over the whole grid**, and
those diverge as the tiling grows whatever the geometry does. Across the ladder
the correctly-aggregated surrogate holds at $0.765 \to 0.802$ of the bound while
the pairwise one falls to $\mathbf{0.234}$. **The hypothesis the framework named
moves the answer by four percent; the one it never named moves it by a factor of
three.**

It has teeth immediately: the checkpoint's column can only supply a reference-free
number, and the compile **admits** it at $N=2$, where the geometry is clean, and
**decertifies** it at $N=6$ and above — the same declaration losing its
certification at a graph size, with nothing about the checkpoint changing.

**W54 — a defect that carries its harness.** Four composition-layer errors have
worn an agent's label on this project and the depth tag caught none of them. An
emitted defect now carries the overlap, $\chi$'s shape, the exchange cadence and
the elliptic placement, derived from the graph so it cannot disagree with what
ran, and an artifact with a measured defect and no harness is **refused**. The
tier then produced a fifth instance unprompted: a construction that froze the disk
force diverged, and that instability belongs to the disk model and would have been
charged to the composition.

**W81 — $\beta_{\min}$.** The number that carries the entire discriminating power
of the plug-in guarantee had no definition anywhere. It now has both things the
gap asked for: a derived candidate ($\min(\tau,\sigma)$ over the terms with a
positive scale), and — where no tolerance is supplied — a certificate that reports
**the two thresholds** rather than inventing a verdict. Below one the test is
blind, above the other the swap is refused, and between them a pass means
something. And the misleading sibling default is gone: `schur_complement`'s
$10^{-12}$ guards a *different* quantity and is now named for it.

## 7. What it cannot say

- **One expert family, one Reynolds number, one geometry.** Sub-linear here says
  nothing about a different physics or a different partition.
- **Twenty-four agents, not two hundred.** Four points on a log–log fit with two
  degrees of freedom is a genuine error bar and a short lever.
- **Wall time answers Claim B’s other half in two different directions.** Per agent, from $N=2$ to $N=24$: the checkpoint gets $3.1\times$ *cheaper* (a batched forward pass amortizing, which is better than $O(1)$) and the classical solver $3.84\times$ dearer. The CFL is eliminated as the cause — $21$ sub-steps at every rung at an identical $u_{\max}$ — so the classical rise is the memory hierarchy and its *work* per agent is still $O(1)$. Neither number transfers off this host, and the instrument had to fail first (a $34\%$ spread, refused) before the $6.0\%$ one was worth believing.
- **The composed classical rollout is not a scheme anybody should ship** in the
  form this study measured it in, and the repair is not the one §5 first named:
  it is `R10`'s, the elliptic part *moved* out of the agent rather than a
  projection added on top. The arrangement that holds $120$ macro-steps exists
  now (`wake_array.exposed_reference_solver`) and **no shipped case study uses
  it yet** — and it clears every rule while still being able to run without the
  operator at all, which is W105.

---

## See Also

- [[tier0-measurements]] — §19 is the full record: every number, every control, and the two rejected constructions
- [[gap-worklist]] — Tier 19: W54, W58 and W81 closed; W100, W102, W103 and W104 opened
- [[case-study-ladder-to-f1]] — the plan this is step one of, and the schedule its result unblocks
- [[cs7-scaling-ladder-pickup]] — the brief this was built from, and what it did and did not anticipate
- [[case-study-wake-array-atlas-0.1]] — the parent, whose $N=6$ numbers are this run's reproduction control
- [[f1-pathmap-and-end-goal]] — F1 and F3, the two criteria reported against
- [[master-error-bound]] — §4.1's overlapping branch, whose $\Pi$ saturates along this ladder
- [[composition-error-theory]] — the assembly term, which §5 shows can end a rollout rather than merely degrade one
- [[atlas-implementation]] — the compiler, whose decertification count grows exactly linearly in seams and agents
