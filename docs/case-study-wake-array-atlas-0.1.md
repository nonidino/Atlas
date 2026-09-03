# Wake Array — a three-turbine case study on a frozen neural operator

**Type:** Concept page — **case-study summary** (folder: `Atlas 0.1/case-study-wind-farm-wake/`)
**Status:** opened 2026-08-30. The short, readable companion to [[tier0-measurements]] §18, which is the full measurement record. Everything here is quoted from `out/w93/w93.json`; nothing is estimated.
**Code:** `atlas/cases/wake_array.py` (the declarations), `scripts/w93_wake_array.py` (the driver), `scripts/w93_frames.py` (the viewer's data), `tests/test_tier18_wake_array.py`
**Related:** [[tier0-measurements]] · [[gap-worklist]] · [[spec-wind-farm-wake-atlas-0.1]] · [[case-study-wind-farm-wake-2d-atlas-0.1]] · [[probed-dtn-coupling]] · [[port-algebra-atlas-0.1]] · [[master-error-bound]] · [[atlas-implementation]] · [[expert-library-atlas-0.1]]

---

## 1. What it is

Three wind turbines, $3.5\,D$ apart in an **L**, standing in a flow computed by a
**frozen pretrained neural operator**. The question it exists to answer is not
"what is the wake loss" — it is:

> **When a pretrained model is wrong inside a composed simulation, does the
> framework's certification machinery catch it, and by how much?**

This is the sixth *real* case study in the vault (one whose `boundary_response`
calls an actual solver or checkpoint rather than returning a matrix somebody
wrote down) and the **first whose geometry is real**. `wind_farm_real.py` has the
wind farm's topology with real experts and says of itself that its geometry is
schematic — six windows of one field standing in for a farm. This is the other
half.

**Why it is worth building:** wake steering and micro-siting move annual energy
yield by several percent at farm scale, so the accuracy of the wake model is
worth real money. That makes it a good place to ask what a certificate on a
learned expert is actually worth.

## 2. The pieces

### Two experts, one of them frozen

| agent | what it is | parameters |
|---|---|---|
| **six fluid windows** | **Poseidon-T** (`camlab-ethz/Poseidon-T`), a frozen scOT neural operator for 2-D incompressible flow, wrapped by `adapters.FrozenFluidExpert` | **20.8 M, none trained here** |
| **three rotors** | `disk.ActuatorDisk` — classical actuator-disk momentum theory in local-induction form, $C_T' = 4a/(1-a)$ | **zero fitted** |
| **the referent** | `reference.WindowNS` — an explicit fractional-step Navier–Stokes solver on the same 128-cell windows | zero fitted |

The reference solver is not ground truth. It is the **declared referent** the
learned expert's infidelity is measured *against*, which is what `lambda_ref` is
for, and W87 is the standing warning that a referent can be wrong.

### The geometry, which the checkpoint chose

Poseidon-T is fixed at $128\times128$ and `adapters.Scaling` fixes everything
else from it. One macro-step must be one *native* expert lead ($0.1$) and the
freestream must land inside the trained velocity distribution ($U_s = 2$), so a
window spanning $S$ rotor diameters forces

$$\Delta t = \frac{S}{20}, \qquad \mathrm{Re}_{\text{eff}} = \frac{1020}{S}.$$

Choosing the window chooses the macro-step **and** the Reynolds number **and**
how much lateral room a wake has to recover in — and the last two pull against
each other. This case spends the Reynolds band ($\mathrm{Re} = 255$, against the
spec's $[10^3,10^4]$) to buy the geometry, and **the turbine spacing is a whole
number of window strides by construction.**

| | |
|---|---|
| window | $4\,D$ per 128 cells, $\mathrm{d}x = \tfrac{1}{32}D$ |
| macro-step | $0.2$ — lead $0.1$, exactly native |
| domain | $11.0 \times 7.5\,D$, tiled by six windows ($3\times2$), overlap $0.5\,D$ |
| turbines | R1 $(3.75,\ 1.75)$, R2 $(7.25,\ 1.75)$, R3 $(7.25,\ 5.25)$, rotor $D = 1$ |

R1 and R2 are in line so R2 sits in R1's full wake; R3 is abreast of R2 one row
over, so it is the near-clean control. **The array carries its own controls.**

### How they interact — 13 seams, none of them hand-coded

Every connection is a typed port pairing derived from the layout; there is no
coupling code anywhere in `atlas/cases/wake_array.py`, which is the whole claim
the package exists to make true.

- **7 fluid–fluid `MECH` seams.** Two overlapping Poseidon windows exchange a
  normal velocity and return the Steklov–Poincaré flux $\nu\,\partial w/\partial n$.
- **6 field-to-lumped `MECH` seams.** Each rotor has an `up` and a `down` port:
  it reads its inflow from the ring upstream of its plane and imposes the
  traction $\tfrac12\rho C_T' u^2$ on the ring downstream. *In an overlapping
  decomposition each window's artificial ring lies inside its neighbour, so a
  disk in the overlap meets the **downstream** window's inflow ring on its
  **upstream** side.*
- **3 open `ROT` ports.** The extracted shaft power leaves the system through a
  port with a measurable flow, which is what the port algebra says an absent
  drivetrain is.

Two declarations are new here and both are forced by real geometry:

1. **A port is per face *segment*.** A rotor spans $1\,D$ of a $4\,D$ face, so the
   plane carries two ports on one ring — the disk and the open flow beside it.
2. **The trace is *absolute*.** Every earlier fluid case study writes it as a
   perturbation, which makes `probe_base` identically zero and hides the
   linearization point. Writing it absolutely is also a precondition for
   measuring anything in interface power, since a perturbation times a
   perturbation is not a power.

## 3. Results

> **These numbers replace an earlier set, and the reason is W98.** The first
> version of this page quoted an array loss of $46.79\%$ for the checkpoint. That
> figure was mostly the *coupling's* error, not Poseidon-T's: the march bought
> its transport and its pressure projection from a per-window API, both of which
> are periodic on the window, so a wake left a window's outflow edge and came
> back in at its own inflow edge. It manufactured a $25\%$ velocity deficit
> $3.5\,D$ **upstream** of a lone turbine, where nothing causes one. It was found
> by watching the viewer's animation beside the referent's rather than by reading
> any number, which is the argument for building the viewer. The repair is
> `wake_array.transport_and_project`; the pre-repair artifact is kept beside the
> new one as `out/w93/w93.pre-w98.json`. Everything below is post-repair, at the
> same $60$ macro-steps, and **the referent's column is unchanged to every digit**
> — the reference solver never had the defect, which is itself the control.

### The physics is sound before anything is certified

| control | Poseidon-T | `reference.WindowNS` |
|---|---|---|
| turbine-free run (the array's own noise floor) | $\langle U_d\rangle$ drifts $+0.345\%$ to $+0.481\%$; $u \in [0.9842, 1.0266]$ | exactly $1.000000$, spread $0$ |
| isolated turbine vs. momentum theory | $\langle U_d\rangle = 0.7763$, $+16.44\%$ | $\langle U_d\rangle = 0.8289$, $+24.33\%$ |
| the disk's momentum budget | $\sum f_x\,\mathrm{d}A = -T$ to floating point, by construction (`body_force_field`) | same disk, same budget |

$\langle U_d\rangle$ is sampled $0.25\,D$ upstream of the disk plane, so
$U_\infty(1-a) = 0.667$ is a *lower* bound both must clear: the gap above it
measures how much upstream induction each expert produces, not which is right.

**Two coupling choices are forced, not free**, and both follow from the expert
being frozen at a fixed native step. Measured over four macro-steps:

| | min $u$ |
|---|---|
| as shipped (derived thickness, projection on) | $+0.4313$ |
| `disk.py`'s default $\Delta_d = 0.1\,D$ | $+0.2073$ |
| **projection off** | $\mathbf{-0.7395}$ — the flow through the disk reverses |

The disk's smearing thickness must be $\Delta_d = \langle U_d\rangle\,\Delta t$ or
the body-force impulse exceeds what momentum theory allows; and the composition
layer must supply the pressure projection, because **the checkpoint's pressure
channel is a placeholder its loader pins to zero** — the field that mediates a
momentum sink is not among its outputs. W98 adds a third clause to that second
one: the projection the composition layer supplies has to be **global**, because
the graph declares `pressure` a `GlobalField` and an elliptic operator run per
window smears a disk's response over that window and wraps it round.

### The power, which is the number operators are paid in

Four runs per expert, 60 macro-steps each, with a turbine-free control and a
single-turbine control.

| | Poseidon-T | `reference.WindowNS` | difference |
|---|---|---|---|
| R1 — **blockage** (against R1 alone) | $+3.60\%$ | $+1.60\%$ | $+2.00$ pp |
| **R2 — the wake loss** (against R1 off) | $\mathbf{+74.98\%}$ | $\mathbf{+79.54\%}$ | $-4.56$ pp |
| R3 — the offset row | $-8.48\%$ | $+4.29\%$ | $-12.77$ pp |
| **array loss**, $1 - \sum P / 3P_{\text{iso}}$ | $\mathbf{14.62\%}$ | $\mathbf{25.27\%}$ | $\mathbf{-10.65}$ pp |

$10.65$ percentage points of array loss between the two fluid experts, on
identical geometry, disks, controls and macro-step — down from $21.5$ before
W98, which is the honest measure of how much of the original disagreement was
the framework's own. At farm scale this *is* the quantity wake steering is bought
to move, so a certificate that cannot see the remaining difference is still not
worth having.

Three honesties, without which the table should not be quoted:

1. Neither column is validated against data.
2. The two now **agree closely where the physics is strongest and disagree where
   it is weakest.** R2 sits squarely in a developed wake and the two experts are
   $4.6$ pp apart; R3 sits in the bypass flow beside one, and there they differ
   in *sign* — the checkpoint says R3 gains $8.5\%$ from its neighbour's
   blockage, the referent says it loses $4.3\%$. A wake model that is nearly
   right in the wake and wrong outside it is a specific, reportable failure, and
   it was invisible while the coupling was contributing more error than the
   expert.
3. Poseidon-T's turbine-free control drifts by up to $0.48\%$ where the reference
   solver is exact. That is the checkpoint's own noise floor and every number in
   its column carries it.

### The accuracy, attributed rather than assumed

Measured at the fluid-to-fluid seam beside R2, $3.5\,D$ downstream of R1 — where
the port's own $\nu$ appears on both sides and cancels exactly.

**The instrument is calibrated first**, on the reference pair at that seam:

| injected error | recovered $\tau$ | unswapped side |
|---|---|---|
| $\times 1.05$ | $\mathbf{0.050000}$ | $0$ |
| $\times 1.25$ | $\mathbf{0.250000}$ | $0$ |
| $\times 2.00$ | $\mathbf{1.000000}$ | $0$ |

**Then the measurement.** Poseidon-T substituted for the reference solver:

| what was swapped | $\tau$ on the swapped agent | $\tau$ on the other |
|---|---|---|
| the upstream window | $\boldsymbol{11.58}$ | $\mathbf{0}$ exactly |
| the downstream window | $\boldsymbol{1.12}$ | $\mathbf{0}$ exactly |

**The framework catches it and names the agent.** At the interface state the
reference pair converges to, the checkpoint's interface power is an order of
magnitude above the referent's on the upstream side, and swapping one side leaves
the other's $\tau$ identically zero. Note that $\tau$ *rose* when the coupling
was repaired — $10.74 \to 11.58$ — while the power tables converged. That is not
a contradiction and it is worth sitting with: $\tau$ is a defect in **interface
power at a probed state**, and the march's error had been partly masking the
checkpoint's by degrading both sides of the seam together. Fixing the composition
made the trajectory better *and* the attribution sharper, which is the behaviour
one wants from an instrument that is supposed to isolate the expert.

Corroborating, from the same seam:

| quantity | value |
|---|---|
| $\Xi = \lVert\Lambda_{\text{Poseidon}}\rVert / \lVert\Lambda_{\text{ref}}\rVert$ | $\mathbf{0.0182}$ — the checkpoint reproduces under $2\%$ of the reference's boundary response |
| $\beta$ on the assembled seam | $8.40\times10^{-6}$ against the reference's $4.54\times10^{-3}$ — $540\times$ worse conditioned |
| passivity defect $\pi$ | $2.37\times10^{-4}$ against the reference's exactly $0$, so **E7 fails because of the learned expert** |
| substitution certificate | **`refuse`** at every $\beta_{\min}$ tried |

And **how robust the number is**: the referent's own viscosity is
under-determined by $27\times$ in cell Reynolds number (the checkpoint's
viscosity *is not a number* — it behaves like an LES with a spectral cutoff), and
$\tau$ moves only $\mathbf{1.05\times}$ across that whole band, $11.58$ to
$11.03$. That is what licenses quoting the $11.58$ at all.

### Where it does not work, stated plainly

| finding | what it means |
|---|---|
| **W98** — the composition ran two global operators per window | *(opened and closed here.)* Transport and pressure both belong to the whole domain and the march bought both from `step_many`, which does them **per window and periodically**. A wake reaching a window's outflow edge re-entered its own inflow edge — one full circulation every $N / 6.4 = 20$ macro-steps — and a disk's elliptic response was smeared over its own $4\,D$ box. It manufactured a $25\%$ deficit $3.5\,D$ upstream of a lone turbine. **A wider halo does not touch it**: giving the wrapped strip exactly zero weight in the partition of unity moves the deficit by $0.012$, because W93's reach is the whole window |
| **W93** — the halo declaration was $32\times$ short | The record declares a 2-cell domain of dependence; poking a delta measures **64** — the response is nonzero in all 128 seam cells. A neural operator's receptive field is global by construction, so the rule now declines rather than believing the declaration. **This is the residual W98 could not remove**: after the repair, $\approx 5\%$ of spurious upstream deficit survives at ten macro-steps, and it belongs to the checkpoint |
| **W99** — `force` is dropped silently | `FrozenFluidExpert.step` and `.step_many` apply `force` only inside their `if galilean:` branch, so a caller that moves the translation up to the domain — which W98's repair must — gets a forward pass with no body force and no error. The same branch pins each window's output to zero mean, deleting the momentum deficit a disk just deposited |
| **W95** — no referent from the checkpoint alone | The tightly coupled pair $\tau$ is defined against **cannot be built** from Poseidon-T: 63 of 96 Jacobian directions sit below the probe's own reproducibility and Newton diverges under truncation *and* damping. A learned expert cannot be its own referent |
| **W97** — rotor seams are blind | $\nu\,\partial w/\partial n$ carries $\nu$ and a disk's traction does not, so a field-to-lumped seam is one-sided by the cell Reynolds number and its substitution certificate is **blind by construction** — it certifies any replacement of the fluid expert, including one that ignores its boundary data |
| **W94** — the disk has no bond | A surface bond needs a non-overlapping cut, which R2b refuses for a learned one-shot map; the overlapping alternative is two-way *volumetric* coupling, which the port algebra has no bond for |

**The compile lands at `admit-uncertified` with zero refusals and 50
decertifications** — and it *refuses* outright if `elliptic_subsolve` is declared
at the value the probe measures, which is the correct behaviour of R10 and a hard
result: **R10 refuses to decompose any expert whose response is global, and a
globally-receptive pretrained operator is nearly all of them.**

## 4. What to take from it

- **The certification machinery works on a real pretrained expert.** $\tau$
  calibrates exactly, localizes exactly, and is robust to the referent's own
  under-determination. On the fluid–fluid seam the certificate refuses the swap.
- **It is blind exactly where the physics is lumped**, and that is structural
  rather than a tuning failure.
- **A learned expert cannot be its own referent**, so the classical solver is not
  a convenience here — it is load-bearing.
- **The composition can contribute more error than the expert it is measuring,
  and every attribution number is quietly conditioned on it.** W98 was worth
  roughly twice the checkpoint's own disagreement in array loss, and it sat
  underneath a page of correct-looking tables: $\tau$ calibrated exactly, the
  certificate refused, the controls passed, and the march was still inventing a
  wake $3.5\,D$ upstream of a turbine. Nothing in the ledger was false — the
  instrument was measuring a graph that was not the one described.
- **It was found by looking, not by checking.** No assertion caught W98 and no
  reasonable assertion would have: the manufactured deficit is smooth, bounded,
  physically shaped and in the right units. What exposed it was an animation of
  the field beside a referent's, where a wake upstream of the only turbine is
  obvious in one second. A rendering of the state is a **measurement instrument**
  in this vault's sense, and this case study is the argument for building one per
  case rather than per publication.
- **Real geometry is what made all of this visible.** A schematic rotor hangs off
  any ring, so it never forces the bond question; a synthetic seam has a base of
  zero, so it never forces the linearization question; a solver you can run at
  any resolution has a monolith, so it never forces the referent question. And a
  domain with no distinguished upstream never forces W98: a wake has to have
  somewhere it is not allowed to be.

## 5. Reproducing it

```
python scripts/w93_wake_array.py --steps 60      # every number above
python scripts/w93_frames.py --steps 60          # the viewer's field snapshots
python scripts/w93_build_viewer.py               # inlines them into the page
python -m pytest tests/test_tier18_wake_array.py -q
```

Artifacts land in `out/w93/`. `w93.json` is the record this page quotes;
`w93.pre-w98.json` is the superseded one, kept so the size of the coupling's own
error stays checkable rather than remembered. The frame script reports the
velocity range over every frame of both experts and says so when the colour ramp
would clip, because a ramp that silently clips understates exactly the wake the
page is about.

## See Also

- [[tier0-measurements]] — §18 is the full record, with the derivations and the failed hypotheses this page leaves out
- [[gap-worklist]] — Tier 18: W93 and W98 closed, W94–W97 and W99 opened
- [[spec-wind-farm-wake-atlas-0.1]] — the specification this case study's rotor, ports and macro-step come from
- [[case-study-wind-farm-wake-2d-atlas-0.1]] — the original wind-farm case study, whose topology `wind_farm_real` carries and whose *geometry* this one supplies
- [[probed-dtn-coupling]] — §2.2's Steklov–Poincaré flux, the normative `MECH` effort every seam here returns
- [[port-algebra-atlas-0.1]] — the five port types, and the two-way volumetric gap W94 lands in
- [[expert-library-atlas-0.1]] — where a fixed-resolution frozen expert's constraints belong as a class rather than as this checkpoint's quirk
- [[atlas-implementation]] — the compiler that produced the verdict
