# Atlas — the generalized plug-in composition layer

The implementation of the Atlas specification: `end-to-end-architecture-spec`,
`interface-transfer-theory` and `plug-in-composition-theorems` in the research vault
(see that vault's `atlas-implementation` page for the knowledge-layer record). Nine layers on a seven-hypothesis envelope, with a
three-verdict compiler.

**The claim this package exists to make true:** a new case study is a graph of declared
agent capability records plus port connections, with **zero hand-written coupling code**.
The scheme is compiled from the declarations. There is no hook for case-specific coupling
logic anywhere, and adding one would defeat the purpose.

```bash
python -m atlas wind-farm
python -m atlas wind-farm --boundary-capable --declare-transfer --primal-cross-points
python -m atlas rocket --staging --json out/rocket.json
```

```python
from atlas import compile_scheme, rollout
from atlas.cases import wind_farm

graph = wind_farm.build()
result = compile_scheme(graph)
print(result.report())              # verdict, stamp, refusals, scheme
result.artifact.write("run.json")   # always safe, especially on a refusal

if result.runnable:                 # a refused scheme is not runnable, full stop
    steps, conservation, declined = rollout(result, graph, n_steps=10)
```

Code home is `src/atlas/` on the `atlas-0.1` branch of
`github.com/nonidino/physics-foundation-model`; this tree is drop-in for that path.
Dependencies: `numpy` (`torch` only for the two Poseidon-T case studies). Tests: `python -m pytest tests/ -q` (397 tests).

---

## The layer stack

| L | Layer | Module | Refuses when |
|---|---|---|---|
| **L1** | capability record | `capability.py` | a record is missing a field a downstream layer needs |
| **L1.5** | routing (G15) | `routing.py` | the feasible set is empty |
| **L2** | decomposition, cuts | `compiler.py` | motion, an unledgered topology event, untreated cross-points, **an embedded elliptic sub-solve (R10)**, **an overlap narrower than the agents' domain of dependence** |
| **L3** | port algebra | `ports.py`, `transfer.py`, `admissibility.py` | no declared interface space and prolongation; scale set incomplete or not power-preserving; **the two sides of a seam return different halves of the conjugate pair (C9)** |
| **L4** | transmission operator | `probe.py` | probed null space of the wrong dimension |
| **L5** | interface solve | `scheme.py`, `compiler.py`, `solve.py` | R6 on an empty operator; a degenerate axis |
| **L6** | assembly | `assembly.py` | the identity fails; **a partition of unity that is not convex (R11)** |
| **L7** | time integration | `compiler.py`, `solve.py` | R9 a POINTWISE flux match under multirate, clocks that do not nest, or a declared time-integrated match no agent can compute; R3; **two exposed agents at different sub-step cadences (R10b)** |
| **L8** | emit | `emit.py` | a run that cannot produce the envelope stamp |
| **L9** | typing of claims | `claims.py` | a trajectory metric quoted past its own horizon |

Plus closure and substitution (`composition.py`), the conformance suite
(`conformance.py`), and the five named holes with their required measurements
(`holes.py`).

## Three verdicts, not two

| verdict | meaning |
|---|---|
| `admit` | run, and the bound applies with all constants reported |
| `admit-uncertified` | run, but no bound is claimed; the claim names the unverified hypothesis |
| `refuse` | do not run; the rule number and the quantity that would have been silently wrong |

The split rule: **refuse the silent-wrongness class, decertify the unverified-hypothesis
class.** Every decision cites its rule (`L3/C5`, `R9`, `E7`, `InterfaceMotion`, `W1`), and
the record is built and serializable even when the verdict is a refusal — a refusal that
reports only that it refused is a refusal nobody can act on.

## What is deliberately not filled in

Where the specification left a named hole, this package refuses or decertifies and emits
**the number the missing rule will constrain**. `NamedHole.solve()` raises; it does not
guess.

| slot | today | field-5 measurement it still requires |
|---|---|---|
| `SeamReference` | decertify, tau `UNDEFINED` | the tau-undefined seam list; the one-sided operator surrogate |
| `InterfaceMotion` | refuse | re-probe count; operator drift; unaccounted power |
| `TopologyEvent` | refuse without a state map **and** a ledger | the ledger; the injected initial error |
| `AssemblyCertificate` | **CLOSED 2026-08-28** — `condition` is **L6/C1** and `R11` enforces it | — |
| `PortAmendment` | refuse a sixth port type | the ADVEC passenger-pressure count. **W70, 2026-08-29**: the first physics the five types cannot express is not a sixth port -- a thermoelastic body coupling is *volumetric* and every port is a surface bond |

Unmeasured constants are the same discipline applied to numbers. `holes.Unmeasured`
refuses to behave like a number — `float(L)` raises — so no bound can be quoted from a
constant nobody measured. `p` and the OOD graph-size threshold are still unmeasured; `L`, `sigma`, `tau`,
`norm_A` and `C_mu` are measured for `window_ns` and declared on its graph, and the
compile reports which ones each run lacks. **`cases/window_ns.py` in `split-step`
mode now compiles with an empty `unmeasured` list.**

## Six real case studies, and what each is for

`boundary_response` calls a solver rather than returning a matrix in six graphs
now, and they answer different questions.

| case | expert | what only it can say |
|---|---|---|
| `cases/window_ns.py` | `reference.WindowNS` | the whole Tier 0 stack; the only graph that reaches **`admit`** |
| `cases/channel_ns.py` | `reference.ChannelNS` | whether a rule survives **different internals** (W55) |
| `cases/poseidon.py` | **Poseidon-T**, frozen, 20.8M params | whether a rule survives having **no internals** (W55); the first measured probe floor |
| `cases/wind_farm_real.py` | six `WindowNS` + two `ActuatorDisk` | the **port algebra**: `ADVEC` passengers per face, open `ROT` ports, a field-to-lumped seam |
| `cases/thermal_seam.py` | `Compressible2D` + `ThermoStruct2D` | whether a rule survives **different equations on the two sides** -- the only exercise E3 and the `bc_channel` ladder have had against a genuine disagreement |
| `cases/wake_array.py` | six **Poseidon-T** windows + three `ActuatorDisk` | whether the attribution machinery works on a **real pretrained expert in a real wake** -- the only graph with turbine geometry, and the only one that produces a number an operator is paid in |

**Every one of them found something the others could not.** `ChannelNS` showed
that **R10 survives the absence of the compatibility violation it was measured
by** -- `project_outflow` has an outlet Dirichlet and therefore no compatibility
condition at all, and `tau` is still flat in `dt` to a ratio of 1.05 over a 25x
range. Poseidon-T found **W61**: R2b left the transmission rung lifted for an
agent declaring `time_discretization=unknown`, producing a *runnable* probed-DtN
scheme -- the construction measured at 8.9x worse than doing nothing. And the
wind-farm topology found **W66**: an `ADVEC` port whose callable returned the
power where its scale set says effort produced a **false E7 failure**, and
`check_scales` validates the declared scale set while nothing validates that the
callable returns the declared variable.

And the thermal seam **closed W66 by settling what could replace the missing
check, which is nothing**. `THERM` declares the bond `(T, q_n/T)` and every
thermal solver in the build repo returns `q_n`, so the wrong convention arrives
on its own — and it produces a **false pass** where `ADVEC` produced a false
alarm: verdict, stamp, null count and passivity defect all *numerically
identical*, because the two responses differ by a positive factor of `T` and
`sym(cS) = c sym(S)` keeps every eigenvalue's sign. The magnitude cannot separate
them either (nondimensionalize and `s_e = s_f = s_P = 1`; measured at 5% on
`ADVEC`), nor can a power balance against `storage` (0.473 against 0.499). One
statement covers all three: **the wrong half differs from the right one by an
O(1) factor once nondimensionalized, and no dimensionless diagnostic separates
O(1) from O(1).** Hence `ResponseHalf`, a declaration — and **L3/C9**, which
refuses a seam whose two sides return different halves. Before it existed,
`window_ns` compiled to **`admit`** with one side of every seam returning the
velocity where its declared pair says traction.

The same seam then found **W68**: `elliptic_signature` reads a *known* embedded
backward-Euler conduction solve as *"consistent with EXPOSED or NONE"*, at every
cadence over a 10,000x range. So `EllipticSubsolve.UNKNOWN` exists (**W60**) and
the gate W60 proposed does not.

**And the reason turned out to be one level below the first diagnosis, 2026-08-30.**
It is true that the statistic measures **non-normality** where **globality** was
wanted -- and that predicts that measuring the shell's *own block* rather than the
assembled seam would help. It does not: the block reads the same, because the
block **is** `5.0001 * I`, the shell's film coefficient, with the conduction
operator 300x underneath it. The statistic was **swamped, not blind**, and no
threshold could have recovered anything.

`probe.operator_content` is the quantity that separates the two situations --
`omega = ||S - cI|| / ||S||`, how much of a block is *not* a scalar -- and it
reads 0.53 and 0.047 on `window_ns`'s two calibration points against 2.0e-5 on
the thermal seam, a 2370x separation. `elliptic_signature` now has a
**precondition and a third outcome**, `NO OPERATOR RESOLVED`, and **L4** issues
`operator-content` on a seam carrying a thin block, naming `beta`: every bound
with `1/beta` in it is then scaled by a declared constant rather than a measured
operator. The closed form for when this happens is `omega ~ C Bi (k_max ell)^2`
(**W71**): conditioning at a seam is set by **locality**, not by whether the two
sides solve the same equations.

**W74, found while writing a test for something else.** The probe linearized
about the **zero trace**. For an affine expert that is free and correct; this
port's effort is a temperature in kelvin, so it was probing at 0 K, and `beta` on
that seam moves 4.8062 -> 0.3807 against the experts' own operating point while
the verdict and all seven decertifications stay identical. The cause is the
declared bond itself: `PORT_SPECS[THERM]` pairs `(T, q_n/T)` so that effort times
flow is a power -- the choice W66 rests on -- and dividing by `T` is exactly what
makes the response nonlinear. `ExpertCapabilities.probe_base` fixes it and
defaults to zeros, so every earlier measurement reproduces bit-for-bit.

**And the field was on the wrong object -- 2026-08-29, W74's class.** `probe_base`
is per EXPERT, so the two sides of a seam name their own and nothing compared
them. `thermal_seam`'s are 500 K apart: the gas linearizes at the wall
temperature it sees (400 K), the shell at the gas temperature it sees (900 K),
on one interface variable. `Lambda_M = sum_i P_i* Lambda_i P_i` is a sum of
Jacobians and that is a Jacobian only if the terms share a point -- so the
reported `beta = 0.3757` is attainable at **no** admissible interface state, the
whole range over the two reservoirs being `[0.4200, 1.7689]`. At the consistent
base `lambda* = 371.97 K` it is `1.23807`, a further 3.30x. `assemble_seam` now
takes a `seam_base`, `probe.base_disagreement` reports the spread, and
`L4/probe-base` decertifies on it. `probe.base_sensitivity` is the escape: below
`BASE_SENSITIVITY_FLOOR` the response is affine, the base is provably free, and
the finding goes away -- the one check here whose PASSING promotes.

**W75, and the elliptic gate finally works.** W60 wanted `elliptic_subsolve`
measured; W68 refused to promote `elliptic_signature` from a diagnosis. W75 asked
whether clearing W68's precondition would settle it, and the answer is no: on the
same shell under two solvers the statistic is identical at `dt = 1` and **ranked
backwards** at `dt = 100` (kappa: implicit 3.1965, explicit 9.6687). The reason
is the basis -- a Fourier mode is global by construction, so a block built from
smooth modes cannot report support, and `EllipticSubsolve` is defined by support.
**Poke a delta.** An implicit macro-step inverts a sparse SPD matrix and the
inverse is dense; an explicit march's response is exactly zero past its domain of
dependence. `probe.support_reach` reads 1.000 / 0.188 / 0.021 / 0.986 / 0.210 on
five known declarations and gets every one right, at two solves, with no
threshold on any spectral quantity -- and `conformance._test_elliptic_subsolve`
resolves `UNKNOWN` with it.

**W76, and the certificate that could not fail.** Per-block diagnostics were read
by no rule and `alpha_star` by nothing anywhere. On `window_ns`'s seam `sx0` the
worst block kappa is 4170.9 under an assembled 1.196 -- 3486x -- between two
instances of the same solver on a symmetric tiling. The consequence is derivable:
a substitution moves the assembled operator by at most the swapped agent's own
block, so when `||S_i|| < beta - beta_min` an under-responding replacement cannot
be caught. Replacing that seam's 16.7% agent with an expert that IGNORES its
boundary data is certified at every `beta_min` up to 0.20, while the same swap at
the balanced seam `sy0` is refused from 0.10. `SubstitutionCertificate.blind`
downgrades such a pass, and nothing in this framework derives `beta_min` (W81).

**W77 -- `probe_state` is derived now.** It read `"duct, T_hot=900 K,
T_wall=400 K"` on every `thermal_seam` certificate while the probe ran at 0 K. The
probe knows its base and the base IS the state, so it is computed:
`gas@400; shell@900 INCONSISTENT(spread 223.6)`. That removes a declaration
rather than adding a fourth to W69's three.

## Multiphysics, and the field that was written down and never read

**The standing reason no multiphysics graph here could be certified, removed
2026-08-29.** `E3` compares two `governing_family` strings and, when they differ,
the compiler emitted `tau` as UNDEFINED for both sides -- so error attribution,
and every rule downstream of it, was unavailable for the whole class.

Read what `tau` is measured as, in `scripts/tier0_window_ns.py`: the composed step
given the TRUE trace, against a reference trajectory. **Nothing in that mentions a
governing equation.** It needs a reference TRAJECTORY, and at a multiphysics seam
one is constructible from the agents themselves -- the tightly coupled solve, with
the interface converged inside the macro-step instead of lagged across it. That is
the exact analogue of the single-physics monolith, which is likewise not ground
truth but "the same expert applied without the cut".

So E3 was gating the wrong thing. Sharing a family is what lets two agents share a
MONOLITHIC reference; `tau` needs a reference PAIR -- and `lambda_ref`, on the
record since the end-to-end spec and documented as *"tau at a multiphysics seam"*,
was the declaration that one exists. It was checked for presence and never used.

Verified against surrogates carrying a KNOWN error, because that is the only way
to check that an attribution attributes: a gas replaced by the same solver with
its wall conductivity scaled by 1.05, 1.25 and 2.0 gives `tau = 0.0500, 0.2500,
1.0000` exactly, per agent, with the unswapped side identically zero.

**The second half is the norm.** The master bound sums `tau + sigma + gamma` as
scalars, and at a multiphysics seam there is no single norm -- conserved variables
against kelvin. Worse, an agent's own norm can be blind: the gas's state is
BIT-IDENTICAL at wall temperatures of 352, 400 and 450 K over one macro-step while
its flow moves 24%, because the isothermal wall enters through a ghost state the
interior has not felt. `PORT_SPECS` already pairs every port so effort times flow
is a power, so the defect is measured in **interface power** -- a unit neither
side owns and every port type has. It inherits `response_half` and refuses rather
than guessing when that is undeclared.

`thermal_seam` split-step at a matched clock now reaches `admit-uncertified` with
**0 refusals and 6 decertifications** (was 8), `tau_undefined_seams` empty for the
first time on a multiphysics graph, and `L1/E3` **admitting** -- while E3 the
hypothesis still fails, and always will. What its failure costs is the monolithic
reference and nothing else.

**What this does not fix**, and the ledger is in `gap-worklist` Tier 16: multirate
(native clocks at 500:1 still refuse on E4), interface motion, lumped-to-field at
`dim M = 1`, topology events, and two-way volumetric coupling between two agents.
One of four foundational blockers -- the one that made the class structurally
uncertifiable rather than merely unfinished.

**And the multirate row of that ledger was aimed at the wrong term -- 2026-08-30.**
`thermal_seam` at `clocks="native"` now compiles to `admit-uncertified` with zero
refusals, because R9's refusal always named its own exit -- *"until the scheme
declares time-integrated matching with each side's own substep quadrature"* --
and nothing could declare it. `graph.FluxMatching` can, and L7 requires three
things of the declaration rather than trusting it: the clocks must **nest**
(`DT/dt_i` a whole number, or the two sums have different endpoints and there is
no common integral), both sides must supply `boundary_response_integrated`
(`boundary_response` restarts from the agent's own state, so calling it n times
recomputes the first sub-step n times -- the integral is not recoverable from
it), and `solve._port_fluxes` must actually call it. At `n = 1` the integrated
response must equal the plain one exactly, so this is not a fifth unverifiable
declaration.

**The measurement is the finding, and it is a negative.** Put both multirate
terms in interface power over one exchange interval, on `thermal_seam`'s own
500:1 mismatch:

    R9's term   pointwise against integrated flux     5.9972e-05
    the lag     a stale trace over the whole interval  3.7431e-03   62.4x larger

R9 has refused every multirate graph for the life of this compiler over the term
**62x smaller** than the one it does not mention. The rule is still right -- over
an interval the conserved quantity IS the integral, and the leak is exactly zero
at a rate ratio of 1, which is the control -- but at 500:1 the pointwise match
still delivers 0.9999x the heat the gas actually transported. So `L7/R9/lag`
decertifies beside the admission and carries the 62.4, because admitting R9 in
silence would read as having fixed multirate. **W90** is the real blocker now,
and its candidate is W17's `W > 1`: a trace carried as a waveform is exactly a
trace that is not stale.

## Real turbine geometry, and what happens when the attribution machinery meets a checkpoint

`cases/wake_array.py` is the sixth real case study and the first whose *geometry*
is real: three turbines at 3.5 D in an L, six frozen Poseidon-T windows tiling
11.0 x 7.5 rotor diameters, three zero-parameter actuator disks, a wake, and a
power loss. `wind_farm_real.py` says of itself that its geometry is schematic and
that a number measured there is a number about the port algebra; this is the
other half. Run it with `python scripts/w93_wake_array.py --steps 60`.

**The checkpoint chooses the geometry, and the Reynolds band is what it costs.**
One macro-step must be one native lead, so a window spanning `S` rotor diameters
forces `dt = S/20` AND `Re = 1020/S`. The spec's band and the lateral room a wake
needs pull against each other; this graph spends the band (`Re = 255` at `S = 4`)
to buy the domain, and **the turbine spacing is a whole number of window strides
by construction**. Two declarations follow from the checkpoint's own measured
cutoff rather than from convention: `effective_resolution` is a *wavelength*
(33 modes on a 128-cell face, 9 on a 32-cell rotor face), and **a port is per
face SEGMENT** -- a rotor spans 1 D of a 4 D face, so the plane carries two ports
on one ring.

**W93 -- the halo rule reads a declaration and the probe measures the same
thing.** W69 corrected `required_halo` once, in these words: *"that product is an
EXPLICIT agent's domain of dependence."* It fixed the `implicit` branch and left
the branch where the record says nothing, which is what a frozen learned one-shot
map declares under R2b. Measured: the record says **2 cells**, `support_reach`
reads **64** -- nonzero in all 128 seam cells at every amplitude from 1 to 1e-3 --
and the graph's 16-cell overlap was passing on the 2. A neural operator's
receptive field is global by construction, so the repair is not a bigger integer:
`required_halo` returns `None` there now. The same measurement resolves
`elliptic_subsolve` to `embedded`, which `elliptic_signature` **independently
agrees with** on the first black box both instruments have had signal on -- and
declaring it makes L2/R10 refuse the whole graph.

**tau, with its controls, on a real pretrained expert.** Injected errors of 5%,
25% and 100% on the reference pair come back as `tau = 0.050000, 0.250000,
1.000000` exactly with the unswapped side identically zero. Then Poseidon-T
substituted for the reference solver at the wake seam: **`tau = 10.74`**, with
the unswapped side at exactly zero, so the attribution names an agent rather than
a graph. `Xi = 0.0382` corroborates -- the checkpoint reproduces under 4% of the
boundary response the reference does, which is **W59** getting a number. And the
referent is under-determined by 27x in cell Reynolds number while tau moves
1.05x, which is what lets the 10.74 be quoted at all.

**W95 -- the referent cannot be built from the checkpoint's own pair.** A 96x96
dense FD Jacobian, full rank at a 1e-8 cut, and the same `J` at a 2x smaller
probe step differs by 4.09e-2, so **65 of 96 singular directions sit below the
probe's own reproducibility**. Newton diverges from the first iteration;
truncating to the 31 resolved directions does not help and damping to 0.2 does
not help. W85's subspace, conditioning and step length are all eliminated by
measurement. The cost, though, inverts W85's worry: `seam_defect_split` converges
the *reference* pair and asks the checkpoint for four forward passes.

**W97 -- a rotor seam is blind by the cell Reynolds number.** Section 2.2's
normative MECH effort is `nu dw/dn` and an actuator disk's traction carries no
`nu`, so a field-to-lumped seam is one-sided by `Re_h` *before any expert is
chosen*. The fluid-fluid wake seam **refuses** the WindowNS -> Poseidon-T swap at
every `beta_min` tried; the rotor seam certifies it `blind` at every one, over a
range 81x the fluid block's own norm. W76 found the blind spot on one asymmetric
tiling; it is a property of a **port-type pairing**.

**W94 -- and the disk has no bond at all, by two routes.** A surface bond needs
the two fluid subdomains to abut, which R2b refuses for an agent declaring
`time_discretization=unknown` (W61); under the overlapping axis the checkpoint
does support, the disk sits inside the overlap and the coupling is **two-way
volumetric**, which is W70's shape and has no bond. This file takes a third
option that is neither, declares `geometrically_coincident=False` because the two
rings are 0.5 D apart, and reports what the compiler says.

**The consequence, in the units the case is about:** the same array, the same
disks, the same controls, two fluid experts, and **21.5 percentage points of
array loss** between them (46.79% against 25.27%). Neither column is validated
against data -- the classical solver is the *declared referent*, not ground truth,
which is exactly what W87 warns a referent can be wrong about.

**One convention change carries half of this.** `window_ns`, `channel_ns`,
`poseidon` and `wind_farm_real` all write the trace into the ring as a
*perturbation*, so `probe_base` is identically zero on every fluid seam in this
package and `base_disagreement` has read *consistent* on all of them while the
two sides sat at their own states. `wake_array` writes it **absolutely**, and
`L4/probe-base` fires on **10 of its 13 seams**, up to 123% of the base norm --
the three that agree being exactly the three constructed to agree. It is also a
precondition rather than a preference: interface power is `effort x flow`, and a
perturbation times a perturbation is not a power.

## The first real case study

`cases/window_ns.py` is the only graph here whose `boundary_response` is a solver
rather than a matrix: four `reference.WindowNS` windows from the build repo
(`src/atlas/cases/windfarm/reference.py`; set `ATLAS_BUILD_REPO` if it is not at
the default path), tiling a 255x255 domain against a monolith of the same class at
`n = 255`. Run it with `python scripts/tier0_window_ns.py`, which is the driver for
`gap-worklist` Tier 0.

It closed W1, W2, W3 and W30 on 2026-08-27, and the outcome is worth reading before
trusting any number this package prints. **Every diagnostic the construction proposes
for itself came back healthy** -- `kappa` around 20, `beta` three orders above the
probe floor, `mu > 0` so every seam is passive, `gamma` at machine zero, the finite
difference stable to seven digits across four decades of epsilon -- **while the
interface equation those diagnostics feed was the wrong one**: solving it exactly
made one composed step 8.9x worse than not solving it, because a flux-balance
condition is a steady condition and every expert this package can probe is a
one-step map. The full record is `wiki/concepts/Atlas 0.1/common/tier0-measurements.md`.

**Two rules came out of that, and they are the reason `build(mode=...)` has two
modes.** `mode="as-built"` declares what `WindowNS` is -- an agent with an embedded
pressure solve -- and **L2 refuses it under R10**: a projection method's Poisson
solve is global over whatever domain it runs on, so decomposing the domain
decomposes the operator, and the resulting error is elliptic (flat in distance from
the cut, flat in `dt`, untouched by any halo). `mode="split-step"` gives the elliptic
part to the composition layer and compiles to `admit-uncertified` with **zero
refusals**. **R2b** gates probed-DtN on `time_discretization`, because flux balance
is a boundary-value-problem condition and an explicit macro-step poses none.

Measured across the two: the composed macro-step improves **220x**, the interface
operator's conditioning goes from `kappa = 21.7` to `kappa = 1.20`, and the fitted
`L` goes from disagreeing with the monolith's to matching it to five decimal places.

**Three more rules came out of the second session (2026-08-28), and one number in the
paragraph above did not survive it.**

**R11 -- a partition of unity must be CONVEX**, not merely sum to one. The cellwise
bias-variance identity makes `chi >= 0` necessary *and* sufficient for the blend to be at
least as accurate as the local solves it blends, so `AssemblyCertificate.condition` is
`chi_min >= 0` and L6 refuses otherwise. It is the silent-wrongness class: the identity
check passes, both blend-defect definitions read as passing, and the composed step is
166x worse.

**R10b -- an exposed elliptic part must be applied at the AGENT's own sub-step cadence.**
R10 moves the elliptic solve into the composition layer, which applies it once per
exchange; a different cadence makes the composed step a different *splitting* of the same
equations. Measured, a factor-two mismatch costs two orders of magnitude in `tau` with
every other diagnostic healthy.

**And the 220x is a property of that cadence, not of the construction.** `SUBSTEPS = 10`
was tuned to `dt = 0.05`, where it happens to equal the reference's own internal sub-step
count; at `dt = 0.025` and `dt = 0.10` the improvement collapsed to 1.8x and 2.1x. With
the cadence matched it is **138x to 361x across a 4x range of dt** -- the claim survives,
the constant did not. Use `window_ns.substeps_at(dt)`, never `SUBSTEPS`, anywhere `dt`
varies.

**The sigma bound now has an overlapping branch.** `master-error-bound` 4.1:
`sigma <= C_mu * Pi * ||d_lambda||`, with `Pi` the weight the assembly gives to cells a
stale artificial-boundary datum can reach. `assembly.sigma_halo_bound` evaluates it, L6
emits `Pi` on every overlapping compile and decertifies when the partition does not
declare its contaminated cells. `C_mu = 1.2`, measured over 16 configurations.

## What the two fixtures produce

Fixtures, not targets. They exist so the compiler has concrete graphs to be tested
against; `tests/test_compiler.py` asserts these outcomes.

**Wind farm, as built** — `refuse`. Stamp `E1 holds, E2 holds, E3 holds, E4 holds, E5/E6/E7
unchecked`. L3 refuses for a missing declaration; the probe finds the interface problem
**empty** rather than ill-conditioned, so the word *coupled* is refused on the output while
the run itself is not; R6 refuses the direct and Krylov solvers; the claim types `UNTYPED`.
The finding was available at compile time, from one probe, with no rollout.

**Wind farm, boundary-capable** — R2 lifts the rung to `probed-DtN`, which switches the
decomposition axis to non-overlapping, which **introduces the cross-point difficulty the
overlapping scheme does not have**, and L2 refuses. Declare primal cross-point degrees of
freedom and it compiles to `admit-uncertified`: never `admit`, because `L` is unmeasured
and the assembly has no accuracy condition, and no amount of declaring fixes either.

**Rocket ascent** — `refuse`, and it trips three of the five named slots in its declared
scope with the fourth arriving with staging. E2, E3 and E4 all fail. The **rung choice
survives E3's failure** while tau does not: probing never mentions a governing equation, so
the transmission layer works at a seam where no single global evolution operator exists,
and it is the error *attribution* that has no meaning there.

## Where the implementation had to decide something the spec left ambiguous

Recorded here rather than buried, because a specification that does not separate its
decisions from its inventions is worse than no specification, and the same goes for an
implementation of one.

**Status, 2026-08-27:** items 1, 2, 3, 6 and 7 have since been **written into the theory
pages** (spec 1.1, 2.1, 3.4, 5.3; `interface-transfer-theory` 7; `probed-dtn-coupling`
2.1, 4.2), so they are no longer deviations — the pages and this code agree. They stay
listed because the reasoning is the reason the code looks the way it does, and because a
reader comparing against an older copy of the spec will find the difference here. Items 4
and 5 remain implementation choices the spec does not speak to.

1. **E2 and L3 admissibility are not the same test.** The envelope states E2 as
   *"interfaces static and geometrically coincident"*; the 2026-08-27 amendment changed the
   **connection rule**, not the hypothesis. So a seam can satisfy E2 — static, coincident —
   and still be refused at L3 for want of the declaration the amended rule requires. That
   is the wind farm exactly, and collapsing the two would misreport the stamp. A
   `geometrically_coincident` flag carries the coincidence half, and **nothing checks it**:
   it is a label like any other and the conformance argument applies to it.
2. **A missing declaration leaves E7 `unchecked`, not `failed`.** A measured non-adjoint
   pair fails it; an absent one establishes nothing either way.
3. **The transmission rung is decided before L2**, because the rung fixes the decomposition
   axis and the axis decides whether cross-points exist. Deciding it after L2 would let a
   compile check cross-points against the declared axis and then silently switch to the one
   that has them.
4. **A refused seam is still probed**, on a provisional interface space, marked as
   diagnosis-only. Refusing a connection and then declining to measure it would reproduce
   the exact failure the wind-farm finding is about.
5. **R6 fires per-seam, not globally.** One zero block makes the assembled system singular
   there; a direct solve over it is not defined.
6. **A passivity defect below the arithmetic noise floor is reported as zero**, with the raw
   eigenvalue kept alongside. Reporting roundoff as a physical defect would put a passivity
   failure on every symmetric operator the probe ever assembles.
7. **A null-space *deficit* is decertified, not refused.** The safe reading of the
   cross-point identity is *"any excess null direction is a defect"*; a deficit still
   indicts either the probe or the declaration and should not pass silently.

## What the implementation could not do, and why

- **The OOD graph-size threshold gate cannot fire, and the spec has retracted it.** L1 was
  specified to *refuse* any graph at or above the size at which the probability that some
  agent is out of distribution approaches 1, at a threshold the spec did not set. The
  compiler decertified and said so rather than inventing a number; spec 3.4 now makes the
  decertification the rule, reports `K` with it, and names `K* = ceil(log(1-eps) /
  log(1-p))` as what a measured declination rate `p` would reinstate.
- **`compile_scheme` returns `admit`, as of the second session of 2026-08-28**, and this
  entry is kept because the three reasons it previously could not are the history of the
  package. First `L` was unmeasured and the assembly accuracy condition did not exist —
  fixed by W45's ingest path and **L6/C1**. Then one decertification remained, `L2/G5/W16`:
  *"cut where the exact operator is closest to local and the conditioning is comfortable"*,
  adopted as policy with its **[AI Inference]** status preserved, and correctly described
  as a theory-status item that no run could close.

  **It was closed by theory, both ways at once.** The locality half is **derived** as
  `L2/C2`: the composed defect over one exchange interval is exactly
  `sum_i R_i^T chi_i D_i` with `D_i = E_i R_i - R_i E` the restriction defect, and L6/C1's
  convexity bounds it cellwise — so *"closest to local"* denotes a commutator rather than
  a mood. The conditioning half is **falsified**: the cut score ranks placements at
  **-0.853** against the measured composed defect, has a **361x** orbit under re-declaring
  the same interface space, and is constant to 1e-10 where the truth spreads 2.29x.
  `cut_score` is retired as a criterion and scoped to substructuring; see its docstring.

  `cases/window_ns.py` in `split-step` mode now compiles with **zero refusals, zero
  decertifications, zero unmeasured constants, and all seven envelope hypotheses `holds`**.
  Stated narrowly: the bound applies as a theorem and its value is measured and declared
  with provenance. **It does not mean the cut is optimal** — the criterion ranks cuts and
  this graph declares one — and it does not extend past this expert, state, cadence or
  topology. **W55 closed on 2026-08-29**: three experts for the "one expert" half, and a
  compressible gas against a thermoelastic shell for the "one governing family" half.

- **A non-overlapping graph has its own criterion since 2026-08-29, and this entry is kept
  because it said the opposite for a day.** L2/C2's hypotheses are a partition of unity and
  an assembly; substructuring has neither, so the branch decertified citing `C2/W57` with no
  criterion at all. **L2/C3** now stands there, and it is `master-error-bound` §4's own
  factorization stopped one step earlier than §4 stops it — before the operator-mismatch
  bound, where the chain is still an identity:

  `Q_sub(Gamma) = || S_M a* - chi_M || / beta`

  the residual the exact trace leaves in the approximate interface equation, over the
  interface solve's own conditioning. Measured on `tests/substructure_model.py` over 11
  configurations: **ranks the cut placements positively in 11 of 11** (+0.579 to +1.000),
  tight to a factor of 1.7, and following it costs at most **1.105x** the best available cut.

  **Two things that measurement settled beyond the criterion.** `1/beta` *helps* here and
  *inverted* the ranking on the overlapping branch (§10.2) — which is §4's own scoping box,
  measured rather than restated. And **§4's product form ranks NEGATIVE in 9 of 11**: it is a
  bound and must never be used as a criterion. That is also the shape the falsified `Q` had,
  and the cleanest available account of why that one ranked at -0.853.
