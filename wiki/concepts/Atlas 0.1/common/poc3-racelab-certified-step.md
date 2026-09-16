# PoC 3 — the fixed point the implicit step has, and the certified mode on it

**Tier 72, 2026-09-16.** Section 12's criterion 4 — *flip to certified and watch
the error go to the classical answer, and see what that cost* — has never been
startable. This tier makes it startable, on the body-fitted column, and prices
it honestly enough that the answer is "it does not pay, and here is by how
much."

Related: [[poc3-racelab-dashboard]], [[poc3-racelab-bodyfitted-demo]],
[[poc3-racelab-overset-flow]], [[poc3-racelab-car-windows]],
[[defect-correction-learned-operator]], [[gap-worklist]].

---

## 1. What W226 actually said, and what the user chose

Defect correction certifies a **fixed point**: `atlas.defect_correction` finds
$w$ with $\phi(w) = w$. The porous column's `WindowNS` step is **explicit** —
one call to $\phi$ already *is* the answer — so there is nothing for an
iteration to do inside a macro-step, and `racelab_switch.MixedRollout` raises
rather than quietly running something else.

W226 offered two exits. On 2026-09-14 the user chose the second: **an implicit
fluid time step**, so that a step *has* a fixed point to certify. Amending
§4.2 to a steady-state mode was explicitly rejected.

**The step they chose already existed and nobody had used it.**
`overset_ns.OversetFlow.step` — the body-fitted column's own solver — is
implicit in viscosity *and* in advection (linearised about
$u^{*} = 2u^{k} - u^{k-1}$), with BDF2 in time, and its docstring says so
outright: *the implicit advection "is the kind of step the live certified mode
the user chose will need."* So the certified mode belongs on the **body-fitted**
column, not the porous one — which is a relocation of §4.2's per-window mode,
not a weakening of it.

---

## 2. Where the fixed point is

The step's momentum system is

$$M x = b, \qquad
M = K_{\text{fixed}} + K_{\text{visc}} + \frac{a_0}{\Delta t} P_{\text{disc}}
  + u^{*} D_x + v^{*} D_y$$

and it is solved **iteratively**, by BiCGSTAB to `rtol`. Its solution
$x^{\star} = M^{-1} b$ is the fixed point of the preconditioned sweep

$$\phi_P(x) = x + P\,(b - Mx),$$

for **any** non-singular $P$, since $\phi_P(x) = x$ iff $Mx = b$. So the
certified mode's $\phi$ is one sweep, the state it certifies is the momentum
solve's own answer, and by `defect_correction`'s consistency (Theorem 1) the
limit is the **classical** answer whatever the cheap map does.

**The rest of the step is direct.** The projection is one factored solve
(`p_lu`), the interpolation one sparse apply; neither iterates. So certifying
the momentum solve certifies the *step* — which this tier measures rather than
assumes.

### Existence, measured

$\lVert \phi(x^{\star}) - x^{\star}\rVert / \lVert x^{\star}\rVert$, against
BiCGSTAB's own $x^{\star}$:

| sweep | cylinder (27,716 unknowns) | car (377,267 unknowns) |
|---|---|---|
| `picard` | $4.008\times10^{-11}$ | $3.753\times10^{-13}$ |
| `jacobi` | $7.188\times10^{-12}$ | $3.542\times10^{-11}$ |
| `ilu` | $1.401\times10^{-11}$ | $4.757\times10^{-11}$ |

**P1 held.** The implicit step has a fixed point and it is the classical answer.

---

## 3. Existence is not reachability, and the geometry decides which sweep

$P = (\Delta t/a_0) I$ — `picard` — is the tempting choice, because $\phi$ is
then literally *"evaluate the implicit right-hand side explicitly"*: a
time-advance map of the same shape as a learned expert's. **It diverges here,
and the reason is the geometry the body-fitted column exists for.** A body grid
clusters its rows toward the wall at about $0.0014$, and

$$\frac{\Delta t}{a_0}\,\frac{\nu}{h^{2}}
 = \frac{0.0125}{1.5}\cdot\frac{0.004}{0.0014^{2}} = 17.0$$

before the Laplacian's own stencil factor. Measured growth per sweep: **19.36**
on the cylinder and **327.8** on the car. The prediction's number was the
viscous term alone; the car's measured $327.8$ is about $19\times$ it, so the
stencil factor and the implicit advection dominate what the viscous estimate
only pointed at. The direction was right and the magnitude was not, and the
record carries both.

| sweep | cylinder rate/sweep | car rate/sweep | contracts |
|---|---|---|---|
| `picard` | $19.36$ | $\mathbf{327.8}$ | no — diverges |
| `jacobi` | $0.6241$ | $0.846$ | yes, slowly |
| `ilu` | $0.4433$ | $\mathbf{0.007719}$ | yes |

**P2 and P3 held.** An explicit evaluation cannot reach the fixed point it has;
the step's own incomplete factor can.

### The staleness that the fair instrument exposed

The first build of this module factored a **fresh** incomplete factor for every
call, and read the cylinder's `ilu` rate at $0.0066$. `step` does not do that:
it refreshes one every `ilu_every` $= 20$ steps and reuses it between. Given
the step's own three-step-stale factor the same sweep reads $\mathbf{0.4733}$ —
**72 times slower per sweep**, and 92 sweeps to round-off against 7.

**[AI Inference]:** this is a structural disadvantage of the certified route on
this column, not an accident of tuning. $\phi$ is plain preconditioned
Richardson, whose rate *is* $\lVert I - PM\rVert$; the classical solve is a
Krylov method, which adapts to a mediocre preconditioner instead of inheriting
it. The certified mode is therefore more sensitive to preconditioner staleness
than the mode it is being priced against, and the car's own $0.007719$ was read
on a freshly refreshed factor and will not hold across the refresh cycle.

---

## 4. What is certified, and what it costs

### The null element is the classical march, bitwise

A constant $\psi$ makes the inner problem's solution $\phi(w_k)$ after one cheap
call, so the iterates **are** the sweep march. Measured on the car: the defect
correction's state is bit-identical to `classical_march`'s, both at 3 $\phi$
calls. **P4 held** — `defect_correction`'s own fact 2, never before checked on
this column.

### Every cheap map lands on the classical answer

On the car, all against one $x^{\star}$:

| cheap map | status | outer | inner | error to classical |
|---|---|---|---|---|
| null | converged | $3$ | $4$ | $1.267\times10^{-12}$ |
| jacobi | fallback_converged | $6$ | $43$ | $1.267\times10^{-12}$ |
| detuned | **unavailable** | — | — | — |
| **wrong, on purpose** | fallback_converged | $6$ | $86$ | $6.151\times10^{-9}$ |

The deliberately wrong map — a fixed random rescaling of the state — lands
$6.151\times10^{-9}$ from the classical answer against a state of order one,
having cost $86$ inner calls against the null element's $4$. **Consistency did
not depend on the cheap map's accuracy.** That is the one accuracy claim in this
demo that rests on a proof rather than a measurement, and it survived the
adversarial map.

### The step, and the price

| | cylinder | car |
|---|---|---|
| certified-vs-classical state gap | $3.496\times10^{-9}$ | $\mathbf{1.163\times10^{-12}}$ |
| classical step | $0.122$ s | $0.454$ s |
| certified step (null) | $0.140$ s | $0.758$ s |
| cost ratio | $1.23$ | $\mathbf{1.67}$ |

**P6 and P7 held.** Certifying the momentum solve certifies the step, and it
costs $1.67\times$ the classical step to do it.

**Why it does not pay is more specific than "defect correction is slow."** On a
settled car the classical solve takes **one** BiCGSTAB iteration a component —
the extrapolated velocity $u^{*}$ is already almost the answer, so the residual
starts at the solver's tolerance. The certified mode is competing against a
solve that is very nearly free. §4.2 says RaceLab must not imply otherwise, and
this is the number that keeps it honest.

---

## 5. P5 failed, and the prediction asked the wrong question

**P5 as registered:** *a deliberately wrong cheap map lands on the same
classical answer as the null element, within a factor of 10, while costing at
least twice the inner calls.*

The cost half held ($86$ against $4$). The error half did not:
$6.151\times10^{-9}$ against $1.267\times10^{-12}$ is a ratio of
$\mathbf{4854}$.

**The verdict stands as judged, and the reason it was the wrong question is
recorded beside it rather than substituted for it.** Both errors are far below
any tolerance asked for, against a state of order one — *both arms reached the
classical answer*, which is exactly what Theorem 1 claims. A ratio between two
numbers that are both at their stopping floor is a reading of the floor, not a
measure of agreement; the same pair read $0.40$ on one cylinder resolution and
$84$ on another while nothing about Theorem 1 changed. The claim needed a
**threshold** on each error, not a **ratio** between them.

This is the vault's own *flat ratios are floors* row in a new dress, and it was
caught before the car ran — by the cylinder test, which was corrected to assert
each error against the state's scale. The prediction was deliberately left
alone. **[AI Inference]:** the general rule this points at is that a
prediction comparing two quantities should first say what floor each is
measured against, because two quantities below their common floor can be
ordered arbitrarily and the ordering will look like a finding.

---

## 6. Poseidon-T could not be called at all

**P8 is `None` — not measured, and not `False`.** The checkpoint loaded in
$1.1$ s and refused the call:

> lead time $0.0250$ is below the expert's native $0.1$. Sub-native steps are
> out of distribution [...] the responses are halo exchange, IQN-ILS, or a
> negative result, not a smaller step.

So the body-fitted column's time step is **four times below** the only step this
expert will accept. That is a second, independent block on top of W288's
interface one:

1. **Space (W288).** Poseidon-T accepts nothing but a uniform $128\times128$
   grid; the body-fitted grids hold $61.0\%$ of the unknowns and cannot be
   accepted in any arrangement, so the ceiling is $33.4\%$ of the composite.
2. **Time (new, W294).** The column marches at $\Delta t$ four times under the
   checkpoint's native step, and the wrapper refuses rather than extrapolating.

**[AI Inference]:** even with both lifted, the cheap map is being asked to
approximate the wrong operator. $\phi$ here is a linear-solve sweep with
Jacobian $I - PM$; Poseidon-T is a time-advance map. `defect_correction`'s fact
3 says the rate is the cheap map's *Jacobian fidelity*, and these two Jacobians
are not the same kind of object. A learned expert that helps this fixed point
would be a learned **preconditioner**, which is a different artefact from any
checkpoint this project holds. Nothing was forced through `step_unchecked`: a
number from an out-of-distribution call would look like a measurement and would
not be one.

---

## 7. The control that keeps the claim conditional

`MixedRollout` with a `CERTIFIED` window **still raises**, and still names the
explicit step as the reason. **P9 held.** Without it this tier would be
claiming a universal fact where it has measured a conditional one: the fixed
point is a property of **implicitness**, not of this tier's code.

---

## 8. What this tier did NOT do

- **Criterion 4 is not met.** There is no screen: nothing in
  `bodyfitted_server.py` or `bodyfitted.html` offers a certified mode, so nobody
  can yet flip a window and watch. This is the mechanism and its price, not the
  interaction.
- **No learned expert ran** (§6), so the certified mode has been exercised only
  with classical cheap maps and one adversarial one.
- **`certified_momentum` is per step, not per window.** Tier 65 made the
  body-fitted fluid **one** expert rather than fourteen windows, so "certified
  on window $k$" has no referent on this column and §5.2's per-window telemetry
  is unaddressed (W295).
- **The rate was read at one point in the refresh cycle.** The car's $0.007719$
  is on a freshly refreshed factor; §3 argues it will degrade across the cycle
  and that was not measured.
- **W271 is untouched.** The solver is still first order in time. That is a
  statement about the discretisation's accuracy against the PDE, not about the
  algebraic system's fixed point, and §5.4's rule 2 already carries it — but it
  means "the classical answer" here is the classical answer of a first-order
  scheme.
- Nothing downloaded or installed, no machine rented, the unlicensed structural
  checkpoint not loaded, nothing pushed.

---

## 9. Reproducing

```
python scripts/tier72_certified_step.py --out out/racelab21 \
    --stages cylinder,car,learned,control,summary
python -m pytest tests/test_tier72_certified_step.py -p no:cacheprovider
```

The car stage builds the composite in about $85$ s and needs the settled release
state at $t = 12$. Timings were taken on mains at $1.4$ GHz of $3.8$, with the
user's two `phone-remote` processes running; the cost ratios are the numbers to
quote, because both halves of each are measured in one process.
