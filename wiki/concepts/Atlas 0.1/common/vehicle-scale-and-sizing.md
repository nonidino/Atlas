# Which Vehicle This Is — the two decisions that are not framework work

**Type:** Concept page — **two declarations and their controls**: a length and a speed that put three unit systems on one clock (W201), and a rotor sized for its host with a machine sized for that rotor (W199) (folder: `Atlas 0.1/common/`)
**Status:** built 2026-09-12, Tier 49. `atlas/cases/vehicle_march.py` (`VehicleScale`, `VEHICLE`, `machine_for_rotor`, `SizedCircuitSolve`), `scripts/tier49_union_march.py` stage `decisions`, `out/tier49/tier49.json`, `tests/test_tier49_union_march.py`. Worklist rows **W201** and **W199** (closed as decisions), **W196** (annotated).
**Related:** [[rung9-gate-restated]] · [[case-study-vehicle-march-atlas-0.1]] · [[joining-seam-cost]] · [[f1-pathmap-and-end-goal]] · [[case-study-ladder-to-f1]] · [[gap-worklist]] · [[case-study-powertrain-atlas-0.1]] · [[case-study-wake-array-atlas-0.1]] · [[case-study-cooling-loop-atlas-0.1]] · [[poc2-frontwing-results]] · [[port-algebra-atlas-0.1]]

---

# 0. The result, in one paragraph

**Two numbers turn the vehicle graph from three machines into one, and both of them are choices rather than derivations.** [[rung9-gate-restated]] §6 named them as the first two of seven things a march of the union needs, and said they are *"not framework work at all — they are the choice of which vehicle this is."* They are made here. **The units (W201):** the tiling's length unit is $L_0 = 0.50$ m and its speed unit is $U_0 = 50$ m/s, so the front wing's chord is $0.25$ m and the time unit is $0.01$ s. Everything else follows, and what follows is the finding: the union's three clocks, which `L7/R9` compares as the bare numbers $0.0125$, $0.05$ and $0.2$ — **a spread of $16$** — are in seconds $1.25\times10^{-4}$, $10^{-3}$ and $5\times10^{-2}$, **a spread of $400$**. Tier 46's *"reconciled clocks"* set the three bare numbers equal, which puts the coolant circuit at $1/400$ of its own step. The spread is $4U_0/L_0$ and nothing else, so over any plausible F1 choice — chord $0.125$–$0.5$ m, speed $30$–$90$ m/s — it runs from $120$ to $1440$ and **is never below $100$**: the choice moves the number and not the conclusion. **The rotor and the machine (W199):** the disk is re-sized to its host face, $D = 0.5$ in the tiling's units and $0.25$ m, and the machine is re-sized to it by a **similarity** rather than a fit — $k_e = k_t$ scale with the width, every resistance inversely — which leaves the induction, the torque residual and the demand-over-supply ratio **bitwise what they were** at width $1$, while the shaft power and the machine's heat each fall by exactly $2$. Tier 47's un-resized machine demanded $31.15\times$ the largest torque a host-sized disk can deliver; the re-sized one demands $0.2385\times$, which is the width-$1$ control's number to twelve digits.

---

## 0.1 These are the user's calls, and they are cheap to overrule

> **Everything downstream of this page moves if you change two lines.**
> `VEHICLE = VehicleScale(l0_m=0.50, u0_ms=50.0)` and
> `HOST_ROTOR_WIDTH = WA.ROTOR_CELLS * W.DX`.
> They are in one place, as declarations, rather than as constants buried in a
> march. §1.4 records what choosing otherwise costs, and the answer is that it
> moves every number and no conclusion.

The vault's rule is that a decision made on the model's behalf is stated as a decision. Neither of these is derived from anything; both are the kind of thing a vehicle programme would be told rather than compute.

**Intuitively.** Three teams built three parts. One wrote its clock in "chord lengths over freestream", one in "rotor diameters over freestream", one in seconds. Nothing on any drawing says how big the chord is or how fast the car goes, so nobody can say whether the three clocks are the same clock. Pick the chord and the speed and every drawing becomes dimensional at once — and then the three clocks turn out to be $400$ apart, which nobody could have seen while they were bare numbers.

---

# 1. Decision 1 — the units (W201)

## 1.1 What had no answer

[[rung9-gate-restated]] §3 found that no declaration in the package carries a unit of time or length. `ExpertCapabilities.dt_native` and `L_native` are bare floats, and `compiler._clocks` compares them numerically. So the union's table read:

| subsystem | `dt_native` | `L_native` | written in |
|---|---|---|---|
| `front_wing` | $0.0125$ | $1.25$ | the tiling's convective units: $64$ cells to the length unit, $U_\infty = 1$ |
| `powertrain` | $0.2$ | $1.0$ | `wake_array`'s: the rotor diameter $D$, $U_\infty = \rho = 1$ |
| `cooling_loop` | $0.05$ | $0.2$ | SI: seconds, metres, watts |

and Tier 46's two ways to pay for three clocks — re-declare ten agents' `dt_native` at $0.0125$, or give all eighteen an integrated response — treated the first column as comparable numbers. **It is not a column of comparable numbers.**

## 1.2 The declaration

$$L_0 = 0.50\ \text{m}\quad(\text{the tiling's length unit, }64\text{ cells}),\qquad U_0 = 50\ \text{m/s}\quad(U_\infty = 1),$$

$$T_0 = \frac{L_0}{U_0} = 0.01\ \text{s}.$$

**Why these.** The plate spans $32$ cells of the $64$ to a length unit, so $L_0 = 0.50$ m puts the front wing's chord at $0.25$ m — a real front-wing element — and the domain at $1.625 \times 1.125$ m, which is a box you would actually mesh around one. $U_0 = 50$ m/s is $180$ km/h: a medium-speed corner, where a front wing is loaded and the car is not at its straight-line ride height. `wake_array`'s length unit is its rotor's diameter, which decision 2 sets to the host face, $0.5\,L_0 = 0.25$ m.

Derived, and each one checkable:

| quantity | value |
|---|---|
| cell | $7.8125$ mm |
| domain | $1.625 \times 1.125$ m |
| chord | $0.25$ m |
| window (`front_wing`'s $L_{\text{native}} = 1.25$) | $0.625$ m |
| rotor diameter (`powertrain`'s $L_{\text{native}} = 1.0$) | $0.25$ m |
| passage (`cooling_loop`'s $L_{\text{native}} = 0.2$) | $0.2$ m — already SI, unmoved |

## 1.3 The finding: the bare numbers understate the spread by $25\times$

| subsystem | `dt_native`, bare | in seconds | over the fastest clock |
|---|---|---|---|
| `front_wing` | $0.0125$ | $1.25\times10^{-4}$ | $\mathbf{1}$ |
| `powertrain` | $0.2$ | $1.0\times10^{-3}$ | $\mathbf{8}$ |
| `cooling_loop` | $0.05$ | $5.0\times10^{-2}$ | $\mathbf{400}$ |
| **spread** | $\mathbf{16}$ | — | $\mathbf{400}$ |

The bare column's spread is $16$; the real one is $400$. And the ordering changes: by bare number the coolant circuit is the *middle* clock and `wake_array` the slowest; in seconds the coolant circuit is slowest by a factor of $50$ over `wake_array`.

**What that does to Tier 46's reconciliation.** Re-declaring ten agents' `dt_native` at the tiling's $0.0125$ admitted the union at `L7/R9`. In seconds it asks the coolant circuit to run at $1.25\times10^{-4}$ s — $400$ times its own step — which is not a reconciliation of clocks but a $400\times$ over-resolution of one subsystem, silently. **`L7/R9` admitted a graph whose clocks it had not compared.** That is W201 stated as a consequence rather than as a gap.

**[AI Inference]:** the same is true of every `dt_native` comparison the compiler makes on any multi-subsystem graph, not only this one. Nothing has ever compared two clocks in one unit, because no field could carry one. The repair is a unit on the declaration, and this page is a workaround that lives outside the record rather than in it — the hole stays open.

## 1.4 What choosing otherwise costs

The pair $(L_0, U_0)$ moves exactly one thing downstream: the ratio of the coolant circuit's clock to the tiling's, because the circuit is the one subsystem already in seconds.

$$\frac{\Delta t_{\text{cool}}}{\Delta t_{\text{tiling}}\,L_0/U_0} \;=\; \frac{0.05}{0.0125}\cdot\frac{U_0}{L_0}\;=\;\frac{4U_0}{L_0}.$$

Over a plausible F1 box — chord $0.125$ to $0.5$ m, so $L_0 \in [0.25, 1.0]$ m, and $U_0 \in [30, 90]$ m/s:

| $L_0$ | $U_0$ | chord | ratio |
|---|---|---|---|
| $0.25$ | $30$ | $0.125$ | $480$ |
| $0.25$ | $90$ | $0.125$ | $1440$ |
| $1.0$ | $30$ | $0.5$ | $\mathbf{120}$ |
| $1.0$ | $90$ | $0.5$ | $360$ |
| **$0.50$** | **$50$** | **$0.25$** | $\mathbf{400}$ — the declaration |

**The ratio is never below $100$ anywhere in the box.** So the qualitative statement — *the vehicle's subsystems are hundreds of clock ticks apart, and one march cannot span them* — survives any choice a reasonable person would make, which is what makes it safe to make the choice here rather than stop and ask.

## 1.5 What the declaration does NOT buy, stated because it is easy to assume

**Kinematic similarity, yes. Dynamic similarity, no.**

The tiling marches at $\nu = 4\times10^{-3}$ in its own units, so its Reynolds number is

$$Re_{\text{model}} = \frac{U_\infty L}{\nu} = \frac{1 \times 1}{4\times10^{-3}} = 250,$$

while air at $50$ m/s over $0.5$ m is at

$$Re_{\text{vehicle}} = \frac{50 \times 0.5}{1.55\times10^{-5}} = 1.61\times10^{6}.$$

A factor of $6.45\times10^{3}$. Matching it would need $\nu \approx 6\times10^{-7}$ in code units, on a grid that cannot resolve it. **Fixing lengths, times and speeds does not fix $Re$**, so every second quoted downstream is a second of a two-dimensional laminar model of a vehicle and not of the vehicle. The units make the three subsystems comparable to each other; they do not make any of them the real thing. That distinction is the whole content of [[f1-pathmap-and-end-goal]] §1.2's *search wide and cheap, verify the shortlist classically*, and it is restated here because a page with metres and seconds on it invites the other reading.

---

# 2. Decision 2 — a rotor sized for its host, and a machine sized for that rotor (W199)

## 2.1 What Tier 47 found, in one line

`disk.ActuatorDisk` is declared at swept width $D = 1$. Its face in `front_wing`'s tiling is $32$ cells at $1/64$, so $0.5$. Tier 46 re-declared the rotor's *ports* for their host — the host's cutoff and cell measure — and left the *device* a one-diameter disk, so the shaft delivered exactly twice the power the flow gave up through the face. Re-sizing the disk closes that and **leaves the declared machine with no operating point at all**: it demands $31\times$ the largest torque a half-width disk can produce anywhere in its induction clamp.

## 2.2 Why it breaks, from the donor's own formulae

`disk.py`:

$$T = \tfrac12\rho A\,C_T'(a)\,U_d^2,\qquad r = \tfrac{A}{2},\qquad \omega = \frac{\lambda U_d}{r} = \frac{2\lambda U_d}{A},\qquad \tau_{\text{disk}} = \frac{T\,U_d}{\omega} = \frac{T\,r}{\lambda}.$$

So at fixed inflow and fixed induction, **halving the width halves the thrust, doubles the shaft speed, and quarters the torque**:

$$T \propto A,\qquad \omega \propto A^{-1},\qquad \tau_{\text{disk}} \propto A^{2}.$$

The machine, meanwhile, sits just above the battery's open-circuit voltage:

$$I = \frac{k_e\omega - V_{oc}}{R_{\text{total}}},\qquad \tau_{\text{machine}} = k_t I .$$

At width $1$, $k_e\omega = 0.10 \times 13.837 = 1.384$ against $V_{oc} = 1.34$, so the numerator is $0.044$ — **a small difference of two large numbers**. Doubling $\omega$ takes the numerator to $1.427$, a factor of $32$, and the torque demanded with it. That is the whole of Tier 47's $31\times$.

## 2.3 The choice: a similarity, not a fit

Ask that the re-sized machine reach the **same induction at the same inflow** — that the electrical solution be *similar* and only its size change. Two conditions decide it uniquely:

- **the back-EMF must not move.** $k_e\omega$ invariant with $\omega \propto A^{-1}$ gives $k_e \propto A$; and $k_e = k_t$ is the same air-gap flux linkage, so $k_t \propto A$ with it.
- **the torque must follow the disk's $A^2$.** With $k_t \propto A$ that needs $I \propto A$; the numerator $k_e\omega - V_{oc}$ is invariant, so $R \propto A^{-1}$.

$$\boxed{\;k_e = k_t \;\longrightarrow\; A\,k_t,\qquad R_i \;\longrightarrow\; R_i/A\ \text{ for every element},\qquad V_{oc}\ \text{unchanged}.\;}$$

At $A = 1/2$: $k_t = k_e = 0.05$, $R_{\text{MGU}} = 0.30$, $R_{\text{total}} = 0.60$.

## 2.4 Measured, with both controls

`SizedCircuitSolve` overrides exactly the two places `CircuitSolve` reads the disk at the donor's default width and nothing else, so at width $1$ it must reproduce the parent. At the settled rotor inflow, $\bar u = 0.9225$:

| | **control**: width $1$, the machine `powertrain` declares | **Tier 47's failure**: width $0.5$, that same machine | **the choice**: width $0.5$, the machine scaled by the width |
|---|---|---|---|
| $k_t = k_e$ | $0.10$ | $0.10$ | $\mathbf{0.05}$ |
| $R_{\text{MGU}}$, $R_{\text{total}}$ | $0.15$, $0.30$ | $0.15$, $0.30$ | $\mathbf{0.30}$, $\mathbf{0.60}$ |
| shaft speed $\omega$ | $13.837059$ | $27.674118$ | $27.674118$ |
| induction $a$ | $0.11379171$ | $0.400$ — the clamp edge | $\mathbf{0.11379171}$ |
| loop current $I$ | $0.1456863$ | $4.7580392$ | $0.0728431$ |
| torque demanded | $0.01456863$ | $0.4758039$ | $0.00364216$ |
| largest torque in the clamp | $0.0610940$ | $0.0152735$ | $0.0152735$ |
| **demand over largest supply** | $\mathbf{0.2384625}$ | $\mathbf{31.152258}$ | $\mathbf{0.2384625}$ |
| operating point exists | yes | **no** | yes |
| torque residual | $3.0\times10^{-15}$ | $0.968$ | $3.0\times10^{-15}$ |
| thrust | $0.2185294$ | $0.4582049$ | $0.1092647$ |
| shaft power | $0.2015869$ | $0.4226806$ | $\mathbf{0.1007935}$ |
| $q_{\text{machine}}$ at the held $p_{\text{ref}}$ | $\mathbf{134.31}$ W | $143{,}262$ W | $\mathbf{67.16}$ W |

**Three things are controls and not results.**

1. **The width-$1$ column reproduces Tier 46.** $\omega = 13.837$, $a = 0.114$, $I = 0.1457$, $q_{\text{machine}} = 134.31$ W — [[joining-seam-cost]] §2.4 and §8.1's numbers, to every digit they were published at. `SizedCircuitSolve` is therefore the parent at width $1$, and the difference between columns is the width and nothing the override did.
2. **The middle column reproduces Tier 47.** $\omega$ doubled, $a$ at the clamp edge, demand $31.15\times$ supply, `rotor_valid` false — [[rung9-gate-restated]] §4.3's row.
3. **The similarity is exact, not approximate.** The induction and the demand-over-supply ratio agree with the control to $\le 10^{-12}$, and the torque and balance residuals are the control's to the digit. Nothing was fitted; the scaling was derived in §2.3 and then checked.

## 2.5 What it costs: a smaller vehicle, not a different framework

$$\frac{P_{\text{shaft}}(A = \tfrac12)}{P_{\text{shaft}}(A = 1)} = \mathbf{0.5000},\qquad \frac{q_{\text{machine}}(A = \tfrac12)}{q_{\text{machine}}(A = 1)} = \mathbf{0.5000}.$$

Both exactly a half, because $P = T U_d \propto A$ and $I^2R \propto (A\,I)^2 (R/A)\cdot A^{-1}$... more plainly: $I \propto A$ and $R \propto A^{-1}$, so $I^2R \propto A$, and the machine's dissipation tracks its shaft power as it must. With Tier 46's power unit $p_{\text{ref}} = 42{,}187.5$ W **held** — the decision that made it a unit rather than a fit — the heat the machine puts into the block falls from $134.31$ W to $67.16$ W.

**One arithmetic trap, recorded because this tier fell into it.** The machine's casing heat is $I^2R_{\text{MGU}}$ and not $I^2R_{\text{total}}$: the bus, battery and inverter losses happen elsewhere and are not in the casing the block is bolted to, which is how `CooledMachine.heat_flux` and `integration_union` both write it. Charging the loop total doubles the number **exactly**, because $R_{\text{total}} = 2R_{\text{MGU}}$ on this circuit — and a factor of exactly two that is also exactly the similarity's own factor is invisible in any ratio. It was caught by the width-$1$ control failing to reproduce Tier 46's $134.31$ W, which is the only reason the control exists.

---

# 3. What these two decisions do NOT settle

- **No record carries a unit.** `ExpertCapabilities` is unchanged; `VehicleScale` lives outside the declaration and no rule reads it. `L7/R9` still compares bare numbers, so the compile is exactly what it was. **W201 is answered as a decision and open as a schema hole.**
- **No dissipation field was added and no domain boundary became a port** (W200). [[case-study-vehicle-march-atlas-0.1]] writes each receiver's balance by hand instead.
- **`CircuitSolve` is unchanged.** `SizedCircuitSolve` subclasses it; the parent still builds its disk at the default width and still cannot be told otherwise, so any other caller has Tier 47's bug.
- **`integration_union` is unchanged.** The re-sized rotor and machine live in `vehicle_march`, so Tier 46's compile and every number on [[joining-seam-cost]] are bitwise what they were. The union that *marches* and the union that *compiles* are therefore not the same object, and that is a gap this tier opens rather than closes.
- **The Reynolds number is not the vehicle's** (§1.5), and nothing here changes that.
- **The tip-speed ratio $\lambda = 7.5$ is the donor's and was not re-chosen.** A half-diameter rotor at the same $\lambda$ spins twice as fast, which is what forced the machine's re-sizing; choosing $\lambda$ for the host instead would have been a second design decision and is not made.

## See Also

- [[rung9-gate-restated]] §6 — the seven decisions a march needs; these are the first two.
- [[case-study-vehicle-march-atlas-0.1]] — the march these two decisions unblocked, and the other five.
- [[joining-seam-cost]] §2.4 and §8.1 — the width-$1$ operating point these controls reproduce.
- [[case-study-powertrain-atlas-0.1]] and [[case-study-wake-array-atlas-0.1]] — the machine and the rotor, unchanged.
- [[poc2-frontwing-results]] — the tiling whose length unit $L_0$ names.
- [[f1-pathmap-and-end-goal]] §1.2 — why a page with metres on it still is not the vehicle.
- [[gap-worklist]] Tier 49 — W201 and W199 as decisions, and what stays open.
