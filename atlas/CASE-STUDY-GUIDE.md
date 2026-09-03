# How to carry out a case study

This is the practical companion to `README.md`. That file says what the package
is; this one says how to add to it. Read it after `README.md`, before writing
any code.

## What a case study is, and what it is not

A case study is exactly one file: `atlas/cases/<name>.py`. It declares agents
(each carrying an `ExpertCapabilities` record) and connections (each a typed
port pairing between two agents), and returns a `CaseGraph`. **Nothing else.**
`graph.py`'s own docstring is the constraint stated as design: *"adding a case
study adds no code anywhere else in the package."* If you find yourself writing
an `if case_name == "my-case":` branch in `compiler.py`, `probe.py`, or
anywhere outside `atlas/cases/`, stop — that is hand-written coupling code, and
the whole point of the nine layers is that none is needed.

**Decide up front which of two things you're building, because they earn
different things:**

- **A fixture.** You write a synthetic `boundary_response` callable — a matrix,
  a closed-form law, `zero_response` — standing in for an agent. This is what
  `wind_farm.py` and `rocket.py` both are. It costs an afternoon, it exercises
  the compiler against a topology the existing fixtures don't cover (a real
  cross-point, multirate clocks, a moving interface, a `TopologyEvent`), and it
  is how W39–W44 were found: **new topology finds new gaps in the theory
  pages, not new physics.** It does *not* move any Tier 0 (`W1`–`W3`) or Tier 6
  (`W33`) row on `gap-worklist` — nothing about a hand-written response
  function measures anything about a real expert.
- **A real case study.** `boundary_response` calls an actual solver or trained
  checkpoint. This is the only kind of case study that can produce a real
  $\tilde\Lambda$ (W2), feed the conformance suite for real (W33), or supply a
  paired rollout to fit $L$ (W1). **There are nine**: `window_ns.py`,
  `channel_ns.py` (a second discretization), `poseidon.py` (a frozen neural
  operator), `wind_farm_real.py` (the port algebra), `thermal_seam.py` (a
  compressible gas against a thermoelastic shell), `wake_array.py` (real
  turbine geometry, and the attribution machinery pointed at a frozen checkpoint
  inside a real wake), `scaling_ladder.py` (one geometry at five sizes — the
  variable is the *size of the graph*), `reuse_probe.py` (the variable is the
  *state a certificate is measured at*) and `thermal_strain.py` (the variable is
  the *carrier of the bond*). **A tenth is worth writing when it can answer a
  question none of those nine can** — that is the test each of them passed, and
  it is a higher bar than "another expert".
  `thermal_strain.py` passed it by being the first **co-located** split: two
  agents on the same mesh over the same region, cut along the physics rather than
  the domain, with `ThermoStruct2D` unsplit as the free referent. What only it
  could say is that thermal strain **is a bond and is not a port** — the
  conjugate pair exists and its product is a power density, and the free energy
  carries a bilinear cross term measured at $2680\times$ the elastic energy, so
  $\mathcal R$'s premise that agent energies *add* fails before any of its
  arithmetic does. It exercised `PortAmendment` for the first time (**W32**),
  refused it on two of six fields, and found the procedure needs a **seventh**;
  it closed **W94** and reached **W70**'s conclusion from the other side. See
  the box on co-located splits below.
  `wake_array.py` passed it by being the first graph whose **geometry** is real:
  three turbines at $3.5\,D$ in an L, a wake that propagates through six
  Poseidon-T windows, and a power loss in the units an operator is paid in. What
  only it could say is that `probe.support_reach` reads a learned operator's
  receptive field as an *infinite domain of dependence* (**W93** — the
  declaration said 2 cells and the measurement says 64), that a rotor seam is
  one-sided by the **cell Reynolds number** so its substitution certificate is
  blind by construction (**W97**), and that the tightly coupled referent $\tau$ is
  defined against **cannot be built from the checkpoint's own pair at all**
  (**W95**).
  `thermal_seam.py` passed it by being the only place four rules have met a
  *genuine* disagreement rather than a fixture built to produce one: two
  different `governing_family` strings (E3), two different `bc_channel` values
  (R1's ladder, and the first real expert above `dirichlet`), an `IMPLICIT` time
  discretization, and an `EMBEDDED` elliptic part that is not a pressure
  projection. It closed W55, W60 and W66 and opened W68 through W71.
  **`cases/window_ns.py` is the first one** —
  four `reference.WindowNS` windows from the build repo — and it closed W1, W2,
  W3 and W30 on 2026-08-27. Read `wiki/.../tier0-measurements.md` before writing
  another: the numbers that came out are less interesting than the fact that
  **every diagnostic the construction proposes for itself came back healthy
  while the scheme built on it was solving the wrong equation**. A real case
  study should therefore check one thing no fixture can: that the *reference's
  own answer* reduces your interface residual. It costs one extra call.

Both are legitimate. Neither should be mistaken for the other. A fixture that
compiles to `admit-uncertified` has told you the plumbing works; it has told
you nothing about whether the wind farm's actual coupling error is small.

## The five things you declare

### 1. Capability records — one `ExpertCapabilities` per agent

`capability.py` is the schema. Required, or `missing_fields()` refuses the
record before any solve:

- `expert_id`, `ports` (at least one `PortDecl`)
- `dt_native` — the macro-step clock this agent runs at
- `governing_family` — a string. **This is what E3 checks by comparison, and
  it is load-bearing**: two ports of a seam that declare different families
  fail E3, and $\tau$ becomes `UNDEFINED` there. An algebraic closure *within*
  a continuum problem (an actuator disk inside an incompressible flow, say)
  should declare the *same* family as the flow it closes, not a family of its
  own — `wind_farm.py`'s `actuator_disk()` has a comment on exactly this, put
  there after getting it wrong once.
- `boundary_response` — signature `(port_name, trace) -> flux`. The only
  callable the probe needs, and it must not depend on which governing equation
  produced it: this is *why* probing survives E3's failure while $\tau$ does
  not.
- every port's `effective_resolution` — the expert's own measured spectral
  cutoff, not a chosen number. It sets the probe basis dimension.
- `elliptic_subsolve` — `none`, `embedded`, `exposed` or `unknown`. **Get this
  one right or nothing else matters.** An agent that solves a global problem on
  its own subdomain (a pressure projection, a Poisson solve, an implicit
  parabolic step, anything elliptic) is not decomposable: cutting the domain cuts
  the operator, and the error is flat in distance from the cut and flat in
  $\Delta t$. L2 refuses `embedded` under R10, and the refusal is worth more than
  it looks — measured, that defect was 99.8% of a composed step's error and was
  reading as agent infidelity.

  **`unknown` exists for a black box, and it decertifies** (`L2/R10/W60`). Do not
  reach for it to avoid thinking: it is for a checkpoint whose internals nobody
  can see, where `none` would assert something unmeasured *and switch R10 off*
  and `embedded` would assert something unmeasured *and make L2 refuse*. **And do
  not expect the probe to settle it for you.** `poseidon.elliptic_signature`
  reads a *known* embedded backward-Euler conduction solve as *"consistent with
  EXPOSED or NONE"* at every cadence over a 10,000$\times$ range. It is a
  diagnosis, it does not gate, and **W68** is why.

  **Since 2026-08-30 it declines rather than guessing.** The mechanism is not
  only that non-normality is not globality; it is that the block it was handed
  was `5.0001 * I` -- the shell's *film coefficient* -- with the conduction
  operator 300$\times$ underneath. So the statistic gets a **precondition**:
  below `probe.OPERATOR_CONTENT_FLOOR` it returns `NO OPERATOR RESOLVED`, which
  is neither calibration point and not a weaker form of either. If your seam
  gets that verdict, `elliptic_subsolve` is **unmeasurable at this cadence** --
  neither corroborated nor contradicted -- and `L4/operator-content` will
  decertify, because `beta` on such a seam is a boundary coefficient. Section
  14.2 of [[tier0-measurements]] says in closed form when to expect it:
  $\omega \approx C\,\mathrm{Bi}\,(k_{\max}\ell)^2$, so a small Biot number
  or a short exchange interval is enough on its own.

  **And since 2026-08-29 it can be MEASURED, by a different instrument (W75).**
  Clearing the precondition above does not rescue the spectral route: on one
  shell under two solvers `elliptic_signature` is identical at `dt = 1` and ranks
  the pair BACKWARDS at `dt = 100`. A Fourier mode is global by construction, so
  a block built from smooth modes cannot report support -- and this enum is
  *defined* by support. `probe.support_reach` pokes a delta instead: an implicit
  step inverts a sparse SPD matrix whose inverse is dense, an explicit march is
  exactly zero past `radius * substeps`, and the fractions separate 0.986 from
  0.210 on the two `window_ns` modes. **Two solves, no spectral threshold, and it
  resolves `unknown`** -- so declare `unknown` honestly and let
  `conformance._test_elliptic_subsolve` settle it. Only the GLOBAL reading is
  positive: a dense tail that underflows reads as compact, so the gate has an
  operating range in `dt` (**W79**) and the other direction is left uncertified.
- `probe_base` -- **declare it whenever your port's variable has an absolute
  origin.** `port -> the trace the probe linearizes about`, defaulting to zeros.
  For a perturbation variable -- a velocity, a wake deficit -- zero is both the
  origin and a state the expert is in constantly, and the default is right. For
  an absolute temperature it is 0 K, and on `thermal_seam` that moved `beta` by
  **12.6$\times$** while leaving the verdict and every decertification identical
  (**W74**). The tell is whether your `boundary_response` is affine in the trace:
  `PORT_SPECS[THERM]`'s declared bond $(T,\ q_n/T)$ is not, and it is the port
  algebra's own choice of the *true* bond that makes it so.

  **The base belongs to the SEAM, not to you -- corrected 2026-08-29 (W74's
  class).** This field is per expert, so both sides of a seam declare their own,
  and on `thermal_seam` they came out 500 K apart -- each side naming the
  temperature the OTHER presents to it, which is right alone and wrong summed.
  Pass `assemble_seam` a `seam_base` when you can; `L4/probe-base` decertifies
  when the two sides disagree, and `probe.base_sensitivity` clears it outright if
  your response turns out affine. Do not use `epsilon`-stability to check this:
  all four blocks here are stable to 1.0003x over four decades of probe step and
  every one of them is base-dependent. That measures curvature AT the base; this
  asks whether the base is the right place.

  **And the base can be hidden inside the state rather than declared — W74's
  class again, 2026-08-30.** `window_ns`, `channel_ns`, `poseidon` and
  `wind_farm_real` all write the trace into the ring as a **perturbation**
  (`ring + trace`), so `probe_base` is identically zero on every fluid seam in
  this package and `probe.base_disagreement` reads *consistent* on all of them —
  while the two sides are in fact linearized about their own stored states, which
  nothing compares. Write the trace **absolutely** (`ring = trace`) and declare
  `probe_base` as the ring's own physical value: the arithmetic at that point is
  identical and the point is now on the record. On `wake_array`'s wake seams,
  whose two rings sit $0.5\,D$ apart across a velocity gradient, `L4/probe-base`
  then fires for the first time on a fluid seam — the bases differ by up to
  **123% of the base norm**.

  **`multiphysics.seam_defect_split` needs the absolute convention.** It measures
  the defect in interface power, $\int_\Gamma e\,f$, and a perturbation times a
  perturbation is not a power. None of the four perturbation-convention case
  studies could have been measured in that norm.
- `time_discretization` — `explicit`, `implicit` or `unknown`. R2b gates the
  probed-DtN rung on it: flux balance is the interface condition of a
  *boundary-value problem*, and an explicit macro-step poses none.
- `stencil_radius` and `substeps_per_macro_step` — their product is how far an
  artificial boundary's influence travels before the next exchange corrects it,
  and the overlap must exceed it. Undeclared, the halo check decertifies rather
  than assuming you are safe.

  **And the product is an EXPLICIT agent's domain of dependence, which is the
  whole content of W69 and of W93.** If your record says `implicit` with a
  nonzero stencil, one macro-step inverts an operator coupling every cell to
  every other and the product is not your reach. If it says **`unknown`** —
  which is what a frozen learned one-shot map declares, under R2b — you have not
  said your step is explicit either, and since 2026-08-30 `required_halo()`
  returns `None` for that case too and the halo check decertifies. Measured on
  Poseidon-T at a real wake seam: the record declares `stencil_radius=2,
  substeps=1`, and `probe.support_reach` finds the response nonzero in **all 128
  seam cells at every amplitude from 1 to 1e-3**, 64 from the poke — a factor of
  32, on which a 16-cell overlap had been passing. **A neural operator's
  receptive field is global by construction**, which is a fact about the
  architecture and not about the physics, so do not declare a halo for one:
  measure it. **`substeps_per_macro_step` does a second job under
  R10b**: if the agent exposes its elliptic part, the composition layer applies
  that part once per exchange, and the exchange interval is set to
  `dt_native / substeps_per_macro_step` so the composed step is the same splitting
  as the agent's own. Get it wrong by a factor of two and `tau` moves two orders
  with every other diagnostic healthy — measured 2026-08-28. **It is a function of
  `dt_native`, not a constant**: `window_ns.substeps_at(dt)` exists because
  hard-coding it is exactly the bug that was found.

Declare when you have them, decertify (never fabricate) when you don't:

- `storage` — a callable $u \mapsto H(u)$. Its absence decertifies the $L\le1$
  branch (E7) and is the *only* certificate that survives an expert swap
  without re-earning (`plug-in-composition-theorems` §3) — it is worth
  declaring even on a fixture if the response is genuinely passive, because it
  is free once you know the matrix is SPD.
- `validity` — a callable `(state, cond) -> bool`. Its absence decertifies at
  every graph size $K$ (spec §3.4) rather than refusing — see the box below.
- `differentiable` — `NONE`, `JVP`, or `FD`. **If you declare `JVP` you must
  also supply `boundary_response_jvp`.** A record claiming a derivative with
  no way to compute one is a contradiction *inside the declaration*, and
  `missing_fields()` catches it before L4 ever tries to call it — that was one
  of the seven things this implementation found wrong the first time it was
  tried.
- `bc_channel` — `NONE` through `VENTCELL`. `NONE` means the expert has
  nowhere to put a boundary condition at all (a periodic-window checkpoint,
  say); its zero-probe response then doesn't depend on the imposed trace, so
  $\Lambda\equiv0$ and $\Xi=0$ — see the box on empty-vs-hard below.

> **The $K$-threshold, since you'll hit it on the first agent that skips
> `validity`.** A missing `validity` predicate doesn't refuse the compile — it
> decertifies at every graph size, with $K$ reported, per spec §3.4. The
> threshold at which that decertification would become a refusal needs a
> measured per-expert declination rate $p$ (W13); until then the honest
> statement is a decertification that names the missing measurement, not an
> invented number.

### 2. Ports — one `port_decl(...)` per interface an agent exposes

Five closed types: `MECH`, `ROT`, `THERM`, `ELEC`, `ADVEC` — a sixth needs a
`PortAmendment`, which is a named hole, not a thing you can just add. Every
port needs a complete, power-consistent scale set ($s_e \cdot s_f = s_P$),
checked by `check_scales()` from the port list alone.

**And every port needs `response_half`** — `ResponseHalf.EFFORT` or `.FLOW`,
naming which half of the conjugate pair your `boundary_response` *returns*; the
imposed trace is then the other half. It is not derivable from the port type:
`MECH` and `ADVEC` in this package return the **effort** (a traction, a specific
total enthalpy) against a flow trace, and `thermal_seam.py`'s `THERM` returns the
**flow** (an entropy flux) against a temperature trace. Both are legitimate; only
the *pairing* is constrained, and only you know which way round your callable
runs. L3/C9 refuses a seam whose two sides disagree and decertifies one that does
not declare. See mistake 7 for why nothing can check the declaration itself.

`ADVEC` is the one that bites. **The passenger list is per face, not per
agent.** If agent $f$ meets $e$ carrying `(h0, Y_k)` and meets $g$ carrying
only `(h0)`, declare two separate `ADVEC` port decls on $f$ — one per face —
not one list on the agent. `port-algebra-atlas-0.1.md` §3.2 has the worked
example (the rocket's plume) if you want to see why a single per-agent list
cannot express it.

**And real geometry takes it one step further: a port is per face SEGMENT.** A
rotor spans $1\,D$ of a $4\,D$ window face, so the plane it sits on carries two
ports on one ring — the disk and the open flow beside it — and nothing in the
port algebra says a port's $V_i$ has to be a whole face. Declare the segment's
own prolongation with the segment's own cell count; `wake_array.py` derives both
from the layout, and `derive_space=True` still works because both sides of each
seam then declare the same `effective_resolution`. **The cell set does not have
to be contiguous**: a bypass segment is the face minus the rotor, which is two
intervals, and an orthonormal basis on an index set is an orthonormal basis.

**A cutoff is a WAVELENGTH, not a mode count.** `effective_resolution` is the
number of modes, so carrying one expert's number onto a shorter face claims
resolution nobody measured — `window_ns`'s 16 modes on a 128-cell face is a
cutoff of about 18 cells, and putting 16 on a 32-cell rotor face would be
claiming four times the checkpoint's own. `wake_array.modes_for` derives it:
$m = 2n/\lambda_{\text{cut}} + 1$, with $\lambda_{\text{cut}}$ the expert's
measured cutoff in cells.

### 3. Connections — one `Connection` per seam

```python
Connection(
    seam_id="e1",
    a=("agent_a", "port_name_on_a"),
    b=("agent_b", "port_name_on_b"),
    port_type=PortType.MECH,
    geometrically_coincident=True,   # or False
    derive_space=True,               # opt in to M = min_i m_i_eff, identity P_i
    expected_null_dim=1,             # n_0(Gamma): see the table below
)
```

### 3b. The partition of unity, if the decomposition overlaps

`CaseGraph(partition_of_unity=...)` takes a `PartitionOfUnity` or, for a grid too
large for dense restrictions, a `GridPartitionOfUnity`. Two things are checked and
one is emitted:

- **`sum_i chi_ij = 1`** — the identity. L6 refuses on failure.
- **`chi_ij >= 0`** — **L6/C1, and R11 refuses on failure.** Convexity is
  necessary *and* sufficient for the blend to be at least as accurate as the local
  solves it blends, from the cellwise identity
  `|A(u)-u*|^2 = sum_i chi_i |u_i-u*|^2 - V_chi` whose chi-weighted variance
  `V_chi` is non-negative for all data exactly when `chi >= 0`. **Nothing else
  detects a signed partition**: the identity residual is machine zero, both
  blend-defect definitions read as passing, and the composed step is 166x worse.
  Extrapolatory blending is a real technique and this is the rule that refuses it.
- **`contaminated`** — optional, and it is geometry so only you can supply it.
  `contaminated[i]` marks, on subdomain `i`'s own cells, the ones an
  artificial-boundary datum can have reached within one exchange interval. From it
  L6 emits `Pi`, the constant `master-error-bound` 4.1's sigma bound needs for the
  overlapping branch. Undeclared, that bound decertifies rather than assuming your
  assembly is clean.

**The accuracy is not the admissibility, and both matter.** A convex partition that
gives full weight right up to each window's artificial edge is perfectly admissible
and measurably bad: on the four-window tiling it costs `tau = 5.0e-5` against
`2.5e-7` for one that ramps across the whole overlap, a factor of 200. R11 will not
save you from that; `Pi` will tell you about it.

**`geometrically_coincident` is a label nothing checks.** It sets the E2
coincidence stamp and nothing more — since the 2026-08-27 amendment, a
connection is admissible only with a declared interface space $M$ and
prolongation $P_i$ on each side, coincidence or not. `derive_space=True` is
the fast path: it hands the compiler `dim M = min_i m_i_eff` and the identity
prolongation, and is only legal in the conforming case. If your two sides are
at different resolutions, declare $M$ and $P_i$ yourself — that is what
non-conforming coupling *means* under interface-transfer-theory, and it's
still admissible, just not free.

**`expected_null_dim` is $n_0(\Gamma)$, a property of the *seam*, not of a
port type**, and there is no default:

| what meets at the seam | $n_0(\Gamma)$ |
|---|---|
| fluid–fluid, incompressible `MECH` | `1` |
| field ↔ lumped (e.g. rotor face) | `0` |
| conjugate heat transfer, `THERM` | `0` — a uniform temperature shift produces a uniform flux change, so no direction of the trace space is invisible to the response. Nothing constrains the trace the way incompressibility constrains a `MECH` one |
| solid–solid `MECH` on a **free** body | `1` per unconstrained rigid direction the trace can excite — and for a *different reason* than the fluid–fluid row's. A uniform normal velocity on the only constrained face of a free elastic body **is a rigid translation**: no strain, no reaction, so the probed operator cannot see it. Measured on `thermal_strain.py`'s seam: the null direction is the constant mode to $5.7\times10^{-13}$, at $\sigma_{\min}/\sigma_{\max}=2.4\times10^{-15}$. Not incompressibility, not lumpedness — kinematics |
| anything you haven't reasoned through | `None` — disables the check rather than guessing |

Get this wrong in the direction of *too large* and it silently masks a real
defect (a missing cross-point treatment); get it wrong *too small* and you
pay one refusal that inspection resolves in a minute. Leave it `None` if
you're not sure — a disabled check is honest, a wrong one is not.

### 4. The decomposition

`CaseGraph(decomposition=Decomposition.OVERLAPPING | NON_OVERLAPPING, ...)`.
**This is decided once, before you place a single connection**, because the
rung an agent's `bc_channel` and `differentiable` fields can support (R1
lifted by R2) fixes which axis is legal — non-overlapping is what `probed-DtN`
requires, and it's the axis where cross-points become a real problem. If your
graph has three or more subdomains meeting at a point under a non-overlapping
decomposition, either declare `cross_points` and `primal_cross_point_dofs`, or
expect L2 to refuse there — deliberately, per spec §2.1.

### 5. Global fields and topology events, only if you need them

`GlobalField` bypasses the port mechanism entirely — gravity, a uniform
external field — and enters each agent's update directly. `DeclaredTopologyEvent`
is a named hole (`TopologyEvent`): declaring one without both a state map
*and* a conservation ledger is refused outright, not defaulted.

## Running it

```bash
python -m atlas <case-name>              # your case must be registered in __main__.py's CASES dict
python -m atlas <case-name> --decisions  # every decision, not only failures
python -m atlas <case-name> --json out.json
```

Or directly:

```python
from atlas.compiler import compile_scheme
from atlas.cases import my_case

result = compile_scheme(my_case.build())
print(result.report())
```

## Reading the result

The verdict is one of three, and a `refuse` that tells you *why* is a
successful compile, not a failed one:

- **`refuse`** — silent-wrongness class. Something is missing a declaration it
  needs, or a measured quantity contradicts a hypothesis outright. Fix the
  declaration; do not work around the refusal.
- **`admit-uncertified`** — the compile succeeded but at least one of
  E5/E6/E7 is `unchecked` or `fails` on a decertifiable branch. This is where
  every graph in `atlas/` lands today, fixture or real, because $L$ is
  unmeasured (W1) and `AssemblyCertificate` doesn't exist.
- **`admit`** — reachable since the second session of 2026-08-28, and `window_ns` in
  `split-step` mode is the graph that reaches it: zero refusals, zero
  decertifications, zero unmeasured constants, all seven envelope hypotheses
  `holds`. What used to stand in the way was `L2/G5/W16`, the decomposition
  policy, whose **[AI Inference]** status was preserved because it had been
  identified and never derived. It is now derived (**L2/C2**) and the cut score
  that carried it is falsified.

  **What you have to do to get there, since it is a short list and none of it is
  optional.** An *overlapping* decomposition; a partition of unity that is convex
  (L6/C1) and declares its `contaminated` cells; an overlap that outruns the
  agents' domain of dependence; and **every bound constant measured and declared
  on `MeasuredConstants`** — `L`, `tau`, `sigma`, `gamma`, `C_mu`, `norm_A` and
  `cut_defect_bound`. A single missing one is enough: since **W56** a non-empty
  `unmeasured` forces `admit-uncertified`, because a bound quoted with a constant
  nobody measured is the silent-wrongness class.

  **Do not read `admit` as more than it is.** It says the bound applies and every
  constant in it was measured *on this graph, at this state, at this cadence*. It
  does not say the cut is optimal — L2/C2 ranks cuts and your graph declares one —
  and it does not transfer to another expert, which is what **W55** is about. A
  *non-overlapping* graph cannot reach it at all today: L2/C2's hypotheses are a
  partition of unity and an assembly, substructuring has neither, and no criterion
  is derived for that branch (**W57**).

The envelope stamp $(\text{E1}\dots\text{E7})$ is worth reading even on a
refusal — it tells you *which* hypothesis the refusal is about, and whether
that's a `fails` (something was measured and contradicts it) or an
`unchecked` (nothing was measured, because a declaration is missing). Only a
measurement writes `fails` — see spec §1.1 if the distinction is surprising.

## Common first-attempt mistakes

These are the ones the two existing fixtures made and the tests now guard
against — expect to hit at least one:

0. **Leaving `time_discretization` at its default.** It defaults to `unknown`,
   and since **W61** an undecided premise holds the transmission rung *down*
   rather than lifting it. Before W61 it lifted the rung and decertified beside
   it — which is how both shipped fixtures were getting probed-DtN, from an
   omission rather than a declaration. Declare it.
1. **Declaring `differentiable: jvp` without `boundary_response_jvp`.** Caught
   at L1 by `missing_fields()`, not three layers later at the probe.
2. **One `ADVEC` passenger list per agent instead of per face.** Two seams on
   the same agent that genuinely carry different passengers cannot both be
   satisfied by one list.
3. **Defaulting `expected_null_dim` to `1` everywhere.** True at a fluid–fluid
   seam, false at a field↔lumped one — a rotor face responds in exactly the
   direction incompressibility leaves open, so its true null space is `0`,
   and declaring `1` produces a false defect on every such seam.
4. **Choosing the decomposition axis after placing connections.** The rung
   (from R1/R2, decided from the capability records alone) fixes the axis;
   deciding the axis first and hoping the rung agrees is backwards.
5. **Hard-coding the sub-step count.** `substeps_per_macro_step` is a function of
   `dt_native`. Tuned at one macro-step and reused at another it silently turns
   the composed step into a different splitting from the reference, and the whole
   defect is charged to `tau` — measured at two orders of magnitude, with `sigma`
   and every probe diagnostic unchanged. **Three composition-layer defects on this
   project have now worn an agent's label**; this was the third.
7. **A port whose callable does not return its declared conjugate half.**
   `check_scales` validates the scale *set* ($s_e \cdot s_f = s_P$) and it never
   sees the callable. On an `ADVEC` port, whose conjugate pair is (specific total
   enthalpy, mass flux), returning the *power* against a velocity trace made E7
   fail with a passivity defect of $3.9\times10^{-2}$ that was measuring nothing.

   **Since 2026-08-29 you declare it: `response_half=ResponseHalf.EFFORT` or
   `.FLOW`, and L3/C9 checks that both sides of a seam agree** — different halves
   **refuse**, because $\tilde\Lambda_M=\sum_i P_i^*\Lambda_i P_i$ *adds* the two
   responses and a traction plus a velocity is not a quantity; undeclared
   **decertifies**. Leaving it at its default is mistake 0 in a new costume.

   **Do not expect any diagnostic to catch a wrong declaration, because none
   can.** Three were measured and all three fail. E7 is *exactly* invariant under
   a positive rescale ($\operatorname{sym}(cS)=c\operatorname{sym}(S)$), which is
   what the `THERM` pseudo-bond $q_n$ is against the declared $q_n/T$ — a **false
   pass**, where `ADVEC` gave a false alarm. Magnitude cannot help: C4 enforces
   $s_es_f=s_P$, so a properly nondimensionalized port declares all three halves
   at scale 1 and the correct and incorrect `ADVEC` responses differ by 5%. Nor
   can a power balance against `storage` (0.473 against 0.499 on the same port).
   **The wrong half differs from the right one by an $O(1)$ factor once
   nondimensionalized, and no dimensionless diagnostic separates $O(1)$ from
   $O(1)$** — so this is one of the two declarations in the architecture nobody
   can verify, beside `validity`. `MECH` hides the whole class because its
   trace-flux pair happens to line up by convention.
6. **A `bc_channel: none` agent expecting a nonzero coupling.** If nothing can
   accept a boundary datum, $\Lambda\equiv0$ and the interface problem is
   *empty*, not *hard* — the run is a legitimate ensemble of independent
   local solves, but the compiler refuses the word "coupled" on the output
   (spec §6.4(a)). That's not a bug to work around; it's the finding.

> **If your two agents share a REGION rather than a surface (new 2026-09-01).**
> `cases/thermal_strain.py` is the first graph in this package whose two agents
> occupy the same cells, and four things behave differently there. **Read this
> before declaring one, because three of the four are refusals you would
> otherwise read as being about your physics.**
>
> 1. **There is no port, and there is not going to be one.** The five types are
>    surface bonds; a co-located coupling has co-dimension 0. `ports.spec_for`
>    refuses a sixth name, `PortAmendment` was exercised on exactly this object
>    and **declined it**, and the reason is structural rather than a missing
>    entry in a table — see [[port-algebra-atlas-0.1]] §10. What you have is an
>    **operator splitting**; measure its splitting error and quote it with the
>    lag, exactly as `sigma` is quoted.
> 2. **`Decomposition` has no member for it.** `OVERLAPPING` and
>    `NON_OVERLAPPING` are both wrong; overlapping-with-`overlap`-equal-to-the-
>    domain is the closer of the two and `thermal_strain.build` declares it and
>    says so in a comment. Do the same rather than picking one silently.
> 3. **`L2/R10` will refuse you, and its own derivation does not reach you.**
>    R10's sentence is *"the graph decomposes the domain, so the decomposition
>    changes the operator rather than restricting it"* — and a co-located split
>    cuts no domain. It fires anyway, because the rule reads `elliptic_subsolve`
>    and never asks whether the decomposition cuts the agent (**W114**). There is
>    also no `split-step` escape on a *quasi-static* agent: no time derivative,
>    nothing to sub-step, so a structural agent is `EMBEDDED` or it is not an
>    agent.
> 4. **`GlobalField` is the trap, and since W117 it is a trap with a rule on it.**
>    Declare `produced_by`: `()` asserts the field is external, a full agent list
>    says it is a global operation the composition layer owns, and a proper subset
>    applied outside itself is refused at `L3/global-field`. Leaving it undeclared
>    is decertified rather than admitted -- silence is not a pass. It bypasses L3 entirely, so routing the
>    coupling through it gives the *exactly correct* answer with no scale set, no
>    prolongation, no adjoint, no null space, no response half, and no `tau`,
>    `sigma` or `beta` — measured, it turns E3 from `fails` to `holds` and drops
>    two constants from the unmeasured list. **Declaring a coupling out of the
>    port algebra improves its stamp.** If you reach for it, say in the
>    declaration that you are doing so and why.

## Where this fits in the project

A fixture case study is worth doing when you want to stress a topology the
wind farm and rocket don't cover, or when you're about to wire in a real
expert and want a template to copy first. It is **not** a substitute for the
Tier 0 measurement work — see `gap-worklist` — which needs a real trained
agent's `boundary_response`, not a declared one, and is the actual blocker on
this project reaching `admit` for anything.

## See also

- `README.md` — what the package is and the seven decisions the implementation had to make
- `atlas/cases/wind_farm.py`, `atlas/cases/rocket.py` — the two worked examples this guide describes
- `end-to-end-architecture-spec` — the layer-by-layer specification `compile_scheme` implements
- `gap-worklist` — what a real case study would actually unblock (W1, W2, W33)

## If your seam is multiphysics

**Declare `lambda_ref` on both sides.** It names the reference expert this agent
is scored against, and since 2026-08-29 it is what decides whether `tau` is
UNDEFINED at a seam whose two sides declare different `governing_family`. Before
that the family comparison decided it, which was the wrong question: `tau` is
measured against a reference TRAJECTORY, and at a multiphysics seam that is the
tightly coupled pair rather than a monolith neither side can run.

For a real solver the reference is **itself** -- its infidelity is zero by
construction -- and declaring that is not vacuous: it is what lets the compile
say `tau = 0` rather than `tau = UNDEFINED`, and it is what makes a later swap to
a learned expert MEASURE that expert instead of silencing the question.

- **Do not measure the defect in your agent's own state norm.** Use
  `multiphysics.seam_defect_split`, which measures in interface power. Two
  reasons, both measured: the two sides' norms are not commensurable (conserved
  variables against kelvin, and the master bound adds them as scalars), and an
  agent's own norm can be *blind* -- `thermal_seam`'s gas state is bit-identical
  across 100 K of wall temperature over one macro-step while its flow moves 24%.
- **Quote `sigma` with the lag it was measured at, in `sigma_lag` and not in
  prose.** It is a function of the lag, not a property of the seam: 1.0000 at the
  initial condition against 3.392$\times10^{-5}$ at the lag a real run carries.
  **And a scalar lag does not determine it (W86)** -- walking `thermal_seam`'s
  reference trajectory, two consecutive steps whose lag distances agree to 8%
  carry sigmas 1.48$\times$ apart, because sigma's argument is the lag *profile*.
  What transfers is the *slope*, $\mathrm{d}\sigma/\mathrm{d}(\text{uniform lag})$,
  constant to five digits over five steps; what does not is the lag, which moves
  3.43$\times$ across those same steps and which **nothing at compile time can
  know**, since nothing has stepped yet. So `solve.rollout` derives the run's own
  lag from its consecutive traces and `multiphysics.check_sigma_lag` compares it
  against your declaration. That comparison can falsify and never confirm.
- **Expect the referent to cost, and do not reach for the probed `S`.** Newton on
  a finite-difference Jacobian is $2(n+1)$ solves. A scalar secant is not a
  substitute, because it moves only a uniform shift of the trace and stalls where
  the residual varies along the seam. **Neither is `S` (W85)**: the coarse step
  $PS^{-1}Rr$ lives in $\operatorname{range}(P)$, and $\dim M = 16$ against
  $\dim V = 48$ leaves 32 directions untouchable -- measured over five operating
  points spanning 17.8$\times$ in `operator_content`, it never converges, and the
  plateau IS the orthogonal part of the residual to every digit. `seam_jacobian`
  exposes it as a **warm start** (4.8$\times$ fewer solves, landing
  5.2e-5 K away) and only at a *consistent seam base*; at the per-expert bases the
  same operator is 245$\times$ worse.
- **`lambda_ref` is tested now, and the test sees one thing (W87).**
  `conformance._test_lambda_ref` runs `tight_couple` on the pair you supply, so
  the declaration is falsifiable rather than free -- but of four deliberately
  wrong reference pairs, **four of five converge**. A monotone residual has a root
  whichever way its halves lean, so a blind reference and even a sign-flipped one
  balance at a different trace. What the check falsifies is that the interface
  problem is EMPTY, which is the claim `L1/E3` promotes on, and nothing more. A
  wrong referent that still converges corrupts tau's *magnitude* while leaving its
  *localization* correct.
- **`tau = 0` is a legitimate measurement and it breaks `min`.** A composition of
  exact solvers has no agent defect, which set `eps_tol = min(tau, sigma)` to zero
  until **W84**. If you add a rule combining defect terms, check what it does when
  one is identically zero.

## If your agents run at different clocks

**Declare `flux_matching=FluxMatching.TIME_INTEGRATED` and supply
`boundary_response_integrated` on every agent.** Without it L7/R9 refuses, and it
has refused every multirate graph since this compiler existed -- correctly, since
across two clocks a pointwise flux match is not conservative, but see the size
below before you conclude it was the thing standing in your way.

- **The clocks must nest.** `DT / dt_native` has to be a whole number of each
  agent's own steps, or the two sides sum over intervals with different endpoints
  and there is no common integral to match. L7 decides this from the declared
  clocks alone.
- **`boundary_response` cannot stand in for the integral.** It is one state in,
  one flux out, and it restarts from the agent's own state, so calling it n times
  recomputes the first sub-step n times rather than marching n. That is why
  `boundary_response_integrated` is a field: `(port, trace, n_substeps) -> the
  flux averaged over n of YOUR steps`. At `n = 1` it must equal
  `boundary_response` exactly, which is the test that keeps it from being another
  unverifiable declaration.
- **The trace is a value, not a waveform, and that is deliberate.** At `W = 1` the
  composition holds the trace constant across the interval and the flux still
  varies, because the agent's own state evolves under it. That variation is the
  whole of what R9 is about, and it is separable from W17's `W > 1`, which is
  about the trace varying too.
- **Know what declaring it buys, because it is smaller than it looks (W90).**
  Measured on `thermal_seam`'s own 500:1 mismatch, both terms in interface power
  over one exchange interval: R9's is $5.9972\times10^{-5}$ and the *stale trace*
  over that interval is $3.7431\times10^{-3}$, a factor of **62.4**. Matching the
  integral fixes the fast side's flux transient and does nothing about its trace
  being stale for 500 of its own steps. `L7/R9/lag` says so on every compile.
