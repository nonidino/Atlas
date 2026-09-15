# The devices in the body-fitted duct: the union's three joins on solid walls

*Case study, PoC 3. New 2026-09-15. Tier 64.*

---

# 0. The result, in one paragraph

**The radiator core and the recovery turbine are back in the car's duct — the body-fitted one, with solid walls — and the union's three joins close there better than they ever did on the porous column. J3's receiving balance, the one the gate is really about, reads a relative residual of $3.4\times10^{-4}$ against the porous column's $0.0292$: in a solid duct continuity holds the velocity at the disk's plane to the velocity on the ring it reads, so the $3\%$ gap CS-18 diagnosed as *ring-to-plane distance* is simply not there.** J2's first law closes to $9.8\times10^{-9}$ and P4's null still fails as it should; the devices slow the duct by $6.7\%$ and leave the car's drag and lift within $0.3\%$; the joins cost $0.75\%$ of a fluid step. Getting there needed the machine's sizing **iterated** — its generating band is only $2.7\%$ wide below the inflow it is sized for, and the first arm, sized for Tier 63's own duct, spent $196$ of the window's $321$ steps with the machine motoring. Two of sixteen registered predictions failed, and both were worth failing: the radiator's conductance reads $\langle|u|\rangle$ where the trace reports the signed mean, and at the core's ring **$8$ of $32$ cells are flowing backwards along the duct's roof**; and the pressure-jump check charged the whole of a non-uniform duct's friction to the core, which a device-free arm at the same times then measured and removed ($0.599 \to 1.092$).

---

# 1. What a device had to become

On the porous column a device was a strip of cells on **one** lattice: `disk.ActuatorDisk.body_force_field` weights each cell by its exact overlap with the strip, so $\sum f\,\mathrm{d}A = -T$ to floating point, and gate `P7` checks exactly that. The body-fitted car has no such lattice. Across the duct's $29$ open cells the background carries the middle, each wall's own grid carries the six rows nearest it, and the two overlap: there is no single quadrature the thrust can be exact on.

So the device is written once, as a **function of position**:

$$f_x(x, y) \;=\; -\,T\,\varphi(x - x_p)\,\frac{\chi_{[y_0,\,y_1]}(y)}{W},\qquad \varphi(s) \;=\; \frac{1 + \cos(\pi s / \Delta)}{2\Delta}\quad (|s| < \Delta),$$

with $\Delta = 2$ cells, $[y_0, y_1]$ the duct's open band and $W = y_1 - y_0$. Each grid samples it at its own discretisation points. What survives of the old identity is **measured rather than assumed**:

- on the background — whose columns are uniform — the shifted copies of a raised cosine are a partition of unity, so its samples integrate the force to one for *any* position of the plane: measured $1.0$ within $2\times10^{-16}$, of which $0.724$ sits on its own discretisation points and the rest on the rows the two wall grids carry ($207$ forced points each);
- on the quadrature lattice the work is read on ($4$ points a cell each way, interpolated from the composite by a fixed sparse matrix), the weights sum to one within $2.2\times10^{-16}$, and that matrix reproduces `car_solids.probe` to $1.3\times10^{-15}$ and linear fields to $2.7\times10^{-15}$.

**The ring is read $8$ cells upstream of each plane**, on $32$ points at the midpoints of $32$ equal parts of the open band — `disk.disk_average`'s own rule ("deliberately upstream of the strip: sampling inside it would read a velocity the disk's own body force has already slowed") at the porous column's distance ($7.5$ cells) and cell count.

**The device is the duct's open width, $29$ cells and not $32$.** The solids rule thickened `DUCT_LO` and `DUCT_UP` by half a panel each into the duct. Decision 2 (W199) says a device is its host's size, so the disk's area **and** the machine's scale follow the open width; decision 8 (W222) sizes the machine for the inflow the host delivers. `racelab.machine_for_host(u_host, scale=W)` applies both similarities at once, and at the reference inflow it reproduces `vehicle_march.machine_for_rotor` exactly.

---

# 2. The machine's band, and why the sizing has to be iterated

`powertrain`'s machine sits just above the battery's open-circuit voltage, so it generates only over a narrow range of shaft speeds, and the similarity carries that range with it. Measured by bisection (`car_union.machine_band`), a machine sized for an inflow $u_h$ has an operating point **and** generates only for

$$0.9731\,u_h \;\le\; u \;\le\; 1.1408\,u_h,$$

the same three figures at $u_h = 0.0876$, $0.08$ and $0.07$: it is a property of the powertrain's declaration, not of this duct. Below the band the machine motors — a legitimate mode, and not the one the graph declares; above it the disk's induction reaches its clamp $0.4$.

That is **$2.7\%$ of headroom below the sizing point**, against a duct whose flow was still falling when Tier 63 stopped (W281) and which the devices themselves slow further. RaceLab's rule is W228 — *size for the horizon's minimum, not for one instant* — and here it has to be **iterated**: arm 1 is sized for Tier 63's own minimum over $[12, 16]$, arm $k$ for arm $k-1$'s minimum over the same window, and the first arm that stays inside every declared envelope at every step of the window is the one the balances are read on.

---

# 3. The march

Every arm releases from **one** state: the car marched from the uniform stream to $t = 8$ with no devices, which is Tier 63's march repeated in a new process. **It is that march bitwise** — over all $640$ steps every force, the largest speed, the largest divergence and both iteration counts are identical, and so are all $80$ probes of the duct's two planes and the contour. The device-free arm that continues from the kept state to $t = 16$ is bitwise Tier 63 too, so **the restart costs nothing measurable**.

The devices switch on at $t = 8$ and each arm runs to $t = 16$, with statistics over $[12, 16]$ — Tiers 62 and 63's window, so the no-device reference is paired at the same times. The joins are **lagged**: each is re-read once per fluid step from the state at its start, the porous column's lagged cadence with one exchange per step. The coolant circuit is sub-cycled every $400$ fluid steps (`vehicle_march._coolant_step`, unchanged). The envelopes are consulted every step and **recorded rather than enforced**: a release at $t = 8$ is not a settled state for the machine, and what the check would have said is the measurement.

---

# 4. The arms, and the sizing that had to be iterated

| | arm 1 | **arm 2 (registered)** |
|---|---|---|
| sized for | Tier 63's own minimum, $0.087572$ | arm 1's minimum, $\mathbf{0.082938}$ |
| inflow over $[12, 16]$, as a fraction of the sizing | $0.947$–$0.989$ | $\mathbf{0.990}$–$\mathbf{1.035}$ |
| induction | $0.020$–$0.081$ | $0.082$–$0.202$ |
| **MGU declined** | $196$ of $321$ window steps | **none** |
| **ROTOR declined** | $230$ in the window, $63$ before it | $124$, **all before it** (last at $t = 9.55$) |
| duct at the turbine plane | $0.08443$ | $\mathbf{0.08351}$ |
| admitted | no | **yes** |

Arm 1 is the prediction U1 made and the reason the iteration exists: sized for the duct *without* devices, it put the machine below its band for most of the window — the core alone takes about $6\%$ off the flow — and the record says so in the envelope rather than in a footnote. Arm 2, sized for what arm 1 actually delivered, sits inside the band for the whole window; its only declines are in the settle between $t = 8$ and $t = 9.55$, where the duct still runs faster than its minimum and the disk's induction is at its clamp. **That is U3 exactly: every decline is the rotor's, before the window, and never the machine's.**

---

# 5. The three balances

Read by `vehicle_march.receiver_balances` — CS-18's own function, unchanged — on the registered arm's window.

| | body-fitted (this tier) | porous column (Tier 59, `out/racelab8`) |
|---|---|---|
| **J3** relative residual, with the join's term | $\mathbf{3.40\times10^{-4}}$ | $0.0292$ |
| $u$ at the plane over $u$ on the ring | $0.99960$ | $0.9678$ |
| the shaft charged against the **other** device's work | $1.96$ | $3.5$ |
| **J1** tracking residual (`receiver_balances`) | $0.00905$ | $9.8\times10^{-8}$ |
| **J2** block first law, with the mount term | $9.8\times10^{-9}$ | $1.1\times10^{-8}$ |
| the machine's heat | $0.0758$ W | $1.265$ W |

**J3 closes to a part in three thousand, and the reason is geometric.** The shaft claims $T\langle u_{\text{ring}}\rangle$ and the force does $T\langle u_{\text{plane}}\rangle$; in open flow the disk's own induction slows the air between the two, which is the whole of CS-18's $3\%$ (it measured $u_{\text{plane}}/u_{\text{ring}} = 0.968$ and attributed all of it to the $7.5$-cell ring-to-plane distance). **In a solid duct continuity forbids that**: the same flux crosses both stations, so the ratio is $0.99960$ and what remains is the $32$-point midpoint rule on a profile that vanishes at both walls, plus one step of lag. The null arm was not marched — withholding the force makes the term zero and the residual $1$ by construction — and the discriminating control that *was* measured is the shaft charged against the core's work instead: $1.96$, comfortably past the gate's $0.5$, though that margin narrows as the turbine's induction rises toward the core's loss coefficient.

**J2 behaves, and the machine is tiny.** The block's first law closes at $9.8\times10^{-9}$ on the march's one coolant step, and the slow recipe (`tier53_racelab_rerun.stage_slow`, imported unchanged) passes both of P4's clauses: $5.5\times10^{-9}$ with the mount term and $0.998$ with it removed. But the heat itself is $0.076$ W against the porous column's $1.265$ — the similarity carries $I^2R$ down with the cube of the inflow ratio, and this duct delivers a fifth of the porous column's air. The block's thermal time constant is $52.5$ s, $1050$ coolant steps: **J2 is a join whose receiver barely notices it here**, and that is the price of W281's weak duct, not a defect of the join.

**J1 is where a prediction broke** — section 8.

---

# 6. Does the fluid feel the forces

The registered check was the pressure jump across each device, net of the friction over an adjacent $16$-cell stretch of the same duct, against $T/W$. It **failed for the core** ($0.599$ where $1$ was claimed) and held for the turbine ($1.132$), and the reason is not the device: the duct is not uniform enough to be its own friction reference. The prefix had already measured the formula's floor with **no device at all** — $-0.0067$ at the core's station, the size of the jump the core was expected to make — so a device-free arm over the same window was added (judging nothing) to measure that floor where the arms read it.

| | core | turbine |
|---|---|---|
| jump measured, over $[12, 16]$ | $0.003901$ | $0.002483$ |
| the same formula on the device-free arm | $-0.003219$ | $+0.000320$ |
| difference | $0.007120$ | $0.002163$ |
| $T/W$ | $0.006518$ | $0.002195$ |
| **ratio, net of the device-free arm** | $\mathbf{1.092}$ | $\mathbf{0.986}$ |

**Net of the duct's own pressure field, each device's jump is its thrust over its width to $9\%$ and $1.4\%$.** The fluid feels the forces; the registered clause charged a duct's non-uniformity to them, and the number it produced is reported as it stands.

---

# 7. What the devices did to the car

| over $[12, 16]$ | no devices (Tier 63, and the device-free arm — bitwise equal) | **with the devices** |
|---|---|---|
| duct at the turbine plane | $0.08952$ | $\mathbf{0.08351}$ ($-6.7\%$) |
| drag $F_x$ | $2.9133$ | $2.9050$ ($-0.28\%$) |
| vertical $F_y$ | $+2.9402$ | $+2.9406$ ($+0.01\%$) |
| largest speed | $3.37$ | $2.80$ |
| largest divergence | $0.0309$ | $0.0309$ |
| contour's net flux | $0.159\%$ | $0.159\%$ |
| core-to-turbine flux mismatch | $9.5\times10^{-5}$ | $5.7\times10^{-5}$ |

The devices take $6.7\%$ off a duct that was already carrying a fifth of the porous column's air, and touch nothing else: the car's drag and lift move by less than a third of a percent, because the two thrusts together are about a thousandth of the drag and act inside the pod. The march is as sound as the one without them, at most two BiCGSTAB iterations a component. **The joins cost $0.75\%$ of a fluid step** — $4.6$ ms against $612$ ms — so the cadence question CS-18's P6 raised does not arise for a lagged column here.

---

# 8. The one the duct was hiding: the ring flows backwards along the roof

U5 claimed the radiator's conductance follows $\mathrm{UA} = \mathrm{UA_{RAD}}(u/u_{\text{ref}})^{0.8}$ in the air the trace reports, to round-off. It missed by $1.5\%$, and the `ring` stage (added after arm 1, judging nothing) says why. At $t = 8$, on the state every arm releases from:

| at the ring, $8$ cells upstream of the plane | core ($x = 272$) | turbine ($x = 384$) |
|---|---|---|
| signed mean $\langle u\rangle$ | $0.10718$ | $0.10716$ |
| mean of $\lvert u\rvert$ | $\mathbf{0.11630}$ | $0.10716$ |
| smallest sample | $\mathbf{-0.0272}$ | $+0.0090$ |
| largest sample | $0.2518$ | $0.1622$ |
| cells flowing backwards | $\mathbf{8 \text{ of } 32}$ | $0$ |

**At the core's station the duct runs fast along its floor and backwards along its roof**; by the turbine's station, $112$ cells downstream, the profile has healed into an ordinary channel flow with the same mean. `CoreRadiator.ua` reads $\langle|u|\rangle$ — reversed air still cools — while `JoinState.u_core`, which every balance and every page quotes as *"the air through the core"*, is the signed mean. The two differ by $8.5\%$ at the state the core's reference is fixed at, and by $0.7\%$ averaged over the window as the reversal heals ($1.085 \to 1.002$ from $t = 8$ to $t = 16$). So the join is evaluating its own declaration exactly; what failed is the identity **as the trace records it**, and the tracking residual `receiver_balances` reports ($0.00905$) is that mismatch, not the $1.3\times10^{-5}$ of averaging (Jensen) that a moving-but-forward air would have given.

---

# 9. The predictions

Registered at 18:33:08, before any march with a device in it. Five controls (C) and twelve claims about the marches (U).

| id | claim | outcome |
|---|---|---|
| C1 | the force integrates to one on the strip lattice and on the background | **held** — $2.2\times10^{-16}$, $1.0$ |
| C2 | the interpolation matrices are `car_solids.probe`, and exact on linear fields | **held** — $1.3\times10^{-15}$, $2.7\times10^{-15}$ |
| C3 | the duct's open band is $[19.5, 48.5]$ cells at every station read | **held** — eleven stations |
| C4 | the machine's band is the same at three sizing inflows; the similarity at the reference is exact | **held** |
| C5 | on the uniform stream the rings and the plane means read one | **held** — $2\times10^{-16}$ |
| U0 | the prefix repeats Tier 63 within $10^{-6}$ | **held** — bitwise, every step and every probe |
| U1 | arm 1 is not admitted, and its declines include the machine motoring | **held** — $196$ window steps |
| U2 | an arm is admitted by arm 3 at the latest | **held** — arm 2 |
| U3 | every decline in the admitted arm is the rotor's, before the window | **held** — $124$, last at $t = 9.55$ |
| U4 | J3's residual $\le 0.005$; the mis-charged control $\ge 0.5$ | **held** — $3.40\times10^{-4}$ and $1.96$ |
| U5 | J1's identity to $10^{-12}$; its tracking residual above $10^{-6}$ and equal to the Jensen gap | **failed** — $0.0146$: the conductance reads the mean of $\lvert u\rvert$ and the ring is partly reversed (§8) |
| U6 | J2's first law $\le 10^{-6}$, P4's two clauses pass, the heat below $1.265$ W | **held** — $9.8\times10^{-9}$, pass, $0.0758$ W |
| U7 | each device's pressure jump is $T/W$ within $20\%$ (core) and $40\%$ (turbine) | **failed** for the core — $0.599$; net of the device-free arm $1.092$ (§6) |
| U8 | the devices slow the duct by $3$–$20\%$ | **held** — $6.7\%$ |
| U9 | the arm marches soundly on every count | **held** — $2.80$, $0.0309$, $0.159\%$, $5.7\times10^{-5}$, $2$ |
| U10 | drag and vertical force within $2\%$ of Tier 63's | **held** — $-0.28\%$, $+0.01\%$ |
| U11 | the joins cost at most $5\%$ of a step | **held** — $0.75\%$ |

---

# 10. What this tier did NOT do, named

- **No tight coupling and no composition-error arms.** Every join is lagged, so G4/G5's tight-against-lagged comparison — and with it CS-18's finding about `q_machine`'s elasticity — is not made on this column.
- **No graph, no compile, no agent, no seam.** The devices are body forces and the joins are solved in `car_union`; nothing declares a `CaseGraph` on the body-fitted column, so `L7/R9`, the device seams and the overlap seams (W268) are untouched, and P1 is not re-run.
- **No null arm was marched.** J3's "without the term" is $1$ by construction and J1's null is a pinned `UA` the fluid cannot feel; neither was marched here.
- **The duct's flow is still the weak, unsettled one** (W281), by the user's decision, and the devices now take another $6.7\%$ off it. The machine's heat, $0.076$ W, is what that costs J2.
- **The registered arm's window is not a settled state**: the duct falls $3.5\%$ across it, which is why the sizing had to be iterated at all, and a longer horizon would move the sizing again.
- **The front wing is still Tier 62's filled wedge** (W275), and the car still makes lift (W279).
- **Nothing about the pressure jump was repaired**, only measured: the registered clause stands failed and the device-free arm's correction is reported beside it.
- **No cost was taken on battery**: every march here ran on mains beside the user's two phone-remote servers.
- **RaceLab's column, its records, the gate, the demo, the bundle and `car_geometry.json` are unchanged.**
- **Nothing was downloaded or installed**, no machine was rented, the unlicensed structural checkpoint was not loaded, and nothing was pushed.

```
python scripts/tier64_car_union.py --out out/racelab13 --stages controls,prefix
python scripts/tier64_car_union.py --out out/racelab13 --stages arms,null
python scripts/tier64_car_union.py --out out/racelab13 --stages ring,slow,compare,summary
```

About $44$ minutes on the development laptop, on mains: the prefix $775$ s for $640$ steps, each arm about $540$ s, the device-free arm $539$ s. The record is `out/racelab13/racelab13.json`; the trajectories (`prefix.json`, `arm1.json`, `arm2.json`, `null_arm.json`) and the kept state are not carried upstream.

---

## See Also

- [[poc3-racelab-duct-openings]] — Tier 63, the duct these devices sit in and the flow they see (W281).
- [[poc3-racelab-car-solids]] — Tier 62, the solids, the grids and the march this tier extends.
- [[case-study-vehicle-march-atlas-0.1]] — CS-18, the union's march, the seven decisions and the receiver balances reused here unchanged.
- [[poc3-racelab-results]] — the porous column's own arms, its gate and the numbers compared against.
- [[gap-worklist]] — W282, W283, W284 and the rows this tier annotates.
