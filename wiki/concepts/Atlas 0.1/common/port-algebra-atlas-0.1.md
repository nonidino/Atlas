# Atlas: The Port Algebra — Effort–Flow Interface Types

**Type:** Core Concept — Model Portion, **binding interface contract** (folder: `Atlas 0.1/common/`)
**Status:** Adopted 2026-08-19, replacing the ad-hoc edge-type vocabulary (`heat`, `pressure`, `stress`, `fluid`, `momentum`, `mass`, `shear`, `conservation`) that had grown by accretion, one case study at a time. Motivated by the $O(K^2)$ interface-contract problem in [[prior-art-and-novelty-atlas-0.1]] §4.
**Related Concepts:** [[edge-generation-atlas-0.1]], [[conservation-as-constraint-atlas-0.1]], [[expert-library-atlas-0.1]], [[prior-art-and-novelty-atlas-0.1]], [[f1-pathmap-and-end-goal]], [[backbone-1.1]], [[agent-definition-atlas-0.1]]

---

# 1. Why the old vocabulary had to go

The previous edge types mixed four different categories:

| Old label | What it actually was |
|---|---|
| `momentum`, `mass` | conserved **quantities** |
| `heat`, `shear` | **phenomena** (each a mix of quantities and transport mechanisms) |
| `fluid` | a **medium** |
| `pressure`, `stress` | two names for **the same tensor**, split by convention |
| `conservation` | a **constraint**, not a quantity at all |

Two failures follow. The visible one: `pressure` and `stress` are the isotropic and deviatoric parts of one object, and `shear` is that same object with a different normal — three labels, one physical quantity. The structural one: **a vocabulary that grows per case study forces the interface contract to be defined per expert-pair, and $K$ expert families then need up to $K(K+1)/2$ contracts** — 10 at $K{=}4$, but 120–210 at the $K\approx15$–20 that [[f1-pathmap-and-end-goal]] requires. That integration burden, not accuracy, is what would end the project.

The fix is not a tidier list. It is a **closed vocabulary that cannot grow with the number of case studies**, because it is indexed by the physics of energy exchange rather than by the phenomena people happen to name.

---

# 2. The principle: an interface is a power bond

Bond-graph theory (Paynter, 1959) and its modern form, port-Hamiltonian systems theory, established the relevant fact: **every physical domain exchanges energy through a conjugate pair of an *effort* $e$ and a *flow* $f$, whose product is power.** The pair is domain-independent in structure and domain-specific only in units.

$$\boxed{\;P = e\cdot f\;}$$

For field interfaces the pair is per unit interface area (an intensity and a flux density), and the power crossing an interface $\Gamma$ is

$$P_\Gamma=\int_\Gamma e\,f\;ds .$$

Three consequences, and together they are the entire scaling argument:

1. **The contract is per *port type*, not per expert pair.** An expert declares which ports it exposes. Any two experts exposing the same port type can be connected without a bespoke adapter. **$O(K)$ port declarations replace $O(K^2)$ pairwise contracts.**
2. **Conservation becomes one global diagnostic.** Because the exchanged variables *are* a power bond, the energy the coupling creates or destroys is computable from the coupling values alone — §6.
3. **Composition inherits a theory.** Interconnection of port-Hamiltonian systems is port-Hamiltonian, and passivity composes. [[backbone-1.1]]'s symmetric+skew core is already this structure.

---

# 3. The closed port vocabulary

**Five port types. This list is closed** — a new case study may not add to it without an explicit amendment to this page, and the burden of proof is on the addition.

## 3.1 True power bonds

| Port | Effort $e$ | Flow $f$ | $e\cdot f$ | Where it appears |
|---|---|---|---|---|
| **`MECH`** | traction $\mathbf t=\boldsymbol\sigma\!\cdot\!\mathbf n$ [Pa] | velocity $\mathbf v$ [m s$^{-1}$] | W m$^{-2}$ | every solid–solid, solid–fluid and fluid–fluid interface |
| **`ROT`** | torque $\tau$ [N m] | angular velocity $\omega$ [s$^{-1}$] | W | lumped rotating machinery (shafts, rotors, gearboxes) |
| **`THERM`** | temperature $T$ [K] | entropy flux $\dot s_n=q_n/T$ [W m$^{-2}$K$^{-1}$] | W m$^{-2}$ | every interface across which heat conducts |
| **`ELEC`** | potential $\phi$ [V] | current density $\mathbf j\!\cdot\!\mathbf n$ [A m$^{-2}$] | W m$^{-2}$ | conductors, windings, cells, EM machines |

## 3.2 The convective multiport

| Port | Carrier | Passengers | Energy content |
|---|---|---|---|
| **`ADVEC`** | mass flux $\dot m=\rho\,\mathbf u\!\cdot\!\mathbf n$ [kg m$^{-2}$s$^{-1}$] | total enthalpy $h_0=h+\tfrac12\lVert\mathbf u\rVert^2$; species mass fractions $Y_k$; any other advected scalar | $h_0\dot m$ [W m$^{-2}$] |

`ADVEC` is deliberately **not** a simple effort–flow pair, and pretending otherwise would be wrong: a mass flux carries several conserved quantities simultaneously, so bond-graph practice represents it as a **multibond**. Its conjugate effort for the energy channel is $h_0$; for species $k$ it is the chemical potential $\mu_k$. **Any interface with $\mathbf u\!\cdot\!\mathbf n\neq0$ relative to the interface has an `ADVEC` port; a wall does not.**

**The passenger list is part of the type, and it is declared per face rather than per agent.** Two agents that both declare `ADVEC` but disagree about the passengers **are not declaring the same port**. Connecting them either drops a conserved quantity silently or invents one, which is an unnamed sixth port type in disguise and belongs in a `PortAmendment` ([[end-to-end-architecture-spec]] §12.5) rather than in a seam — so §5's connection rule requires the lists to match and refuses a disagreement.

**Per face is not a stylistic preference; a per-agent list cannot express §7.1.** Agent $f$ there meets $e$ across $\texttt{ADVEC}(h_0,Y_k)$ and $g$ across $\texttt{ADVEC}(h_0)$, so **no single list describes $f$** — declare one and one of its two seams is misdeclared, whichever list is chosen. This is the concrete, checkable form of §8's warning that *"species lists are the thing to watch"*: a caution turned into a connection condition.

## 3.3 `MECH` absorbs four old labels

This is the largest single simplification and it is exact, not a convenience:

$$\boldsymbol\sigma=\underbrace{-p\,\mathbf I}_{\text{`pressure'}}+\underbrace{\boldsymbol\tau}_{\text{`stress'},\ \text{`shear'}},\qquad \mathbf t=\boldsymbol\sigma\!\cdot\!\mathbf n .$$

Pressure is the isotropic part of the stress tensor; viscous and elastic stress are the deviatoric part; "shear" is the tangential component of the same traction. **`pressure`, `stress`, `shear`, and the momentum-flux content of `momentum` are one port.** $\mathbf t\cdot\mathbf v$ is the mechanical power per unit area across *any* interface, solid or fluid, normal or tangential.

Note also what this resolves: [[spec-wind-farm-wake-atlas-0.1]] open question 1 asked whether `shear` needs treatment distinct from `momentum`. **It does not** — they are the tangential and normal components of one traction vector, and the port formulation says so structurally rather than by argument.

## 3.4 The thermal convention, chosen deliberately

Two conventions exist and they are not equivalent:

- **True bond:** $(T,\;q_n/T)$. Product is $q_n$, the heat flux — a power. Dimensionally uniform with every other port, so it can enter a single global energy residual.
- **Pseudo-bond:** $(T,\;q_n)$. What most co-simulation codes actually exchange. Its product is not a power, so it cannot participate in §6's residual.

**Atlas uses the true bond.** The pseudo-bond is what an external solver will hand you, so the adapter converting between them is a one-line division and belongs in the port adapter, not in the physics. Entropy *generation* occurs inside agents, not at ports; at the interface crossing itself $T\cdot(q_n/T)=q_n$ exactly, so no irreversibility is hidden by this choice.

## 3.5 The trap this vocabulary exposes

At an interface where mass crosses, **heat is transported by two mechanisms**: conduction (the `THERM` port) and advection of enthalpy (a passenger on `ADVEC`). The old `heat` label conflated them, and any edge previously labelled `heat` across a mass-crossing interface was **under-specified — silently missing the advective half**. §7's migration table repairs several such edges in the rocket graph. This is a substantive correction, not a renaming.

---

# 4. `conservation` is not a port type

It was never a quantity. Under the port algebra it disappears entirely, because **conservation is the definition of a port connection, not a label applied to some of them**: what leaves agent $A$ through a port is what enters agent $B$ through the same port.

$$e_A\big|_\Gamma=e_B\big|_\Gamma,\qquad f_A\big|_\Gamma=-f_B\big|_\Gamma .$$

Every port carries this by construction. What the old `conservation` label was really marking — *"check flux matching here"* — becomes a per-port **residual reported everywhere**, not a type applied selectively:

$$r_\Gamma^{(\text{port})}=\frac{\big\lVert f_A+f_B\big\rVert_\Gamma}{\tfrac12\big(\lVert f_A\rVert+\lVert f_B\rVert\big)_\Gamma}.$$

Whether a given port is *enforced* to $r=0$ or merely *measured* remains a separate, per-port decision — see [[conservation-as-constraint-atlas-0.1]] — and it depends on whether either side is closed-form, not on what the edge was called.

---

# 5. What an expert declares

An expert's interface contract is now a short, machine-checkable declaration, independent of what it will be connected to:

```
expert: external_flow
  ports:
    MECH   : {faces: all,        direction: bidirectional}
    ADVEC  : {faces: open,       passengers: [h0]}
    THERM  : {faces: all,        direction: bidirectional}
  conditioning: [Re, Ma]
  native_dt: 1e-3
  nondim: {L: D, U: U_inf, rho: rho_inf}
```

**`passengers` is per face**, and the flat list above is shorthand for *"the same on every face this port occupies"* — correct for `external_flow`, and not general. Where the faces differ, they are named:

```
    ADVEC  : {faces: {inlet: [h0, Y_k], plume: [h0]}}
```

§3.2 gives the reason this cannot be simplified back: §7.1's agent $f$ has one face with species on it and one without, so no per-agent list is right for it.

**Connection rule:** two agents may share an edge for port type $P$ iff both experts declare $P$ **with the same passenger list** and their interface geometries coincide. The passenger clause is vacuous for the four simple types, whose lists are empty, and binding for `ADVEC` (§3.2). Nothing else pairwise is required. Adding the twentieth expert costs exactly what adding the fourth cost — one declaration.

> **Amended 2026-08-27 — the coincidence clause is superseded.** [[interface-transfer-theory]] §4.1 replaces *"their interface geometries coincide"* with: *a common interface space $M$ is declared for the seam, and each side declares a stable, implementable prolongation $P_i:M\to V_i$.* Geometric coincidence is the special case $M=V_A=V_B$, $P_i=\mathrm{id}$, so **nothing that works today stops working** — but the clause as written above is the line [[generalization-requirements]] G14 calls *"the single most restrictive line in the framework for plug-in use"*, and it was a property of this rule rather than of the coupling construction. **Read the revised rule as binding and this one as its special case.**

**The `nondim` block is the residual per-expert engineering cost**, and it is real: two experts trained on different reference scales will not agree at a port until their values are converted. That cost is $O(K)$, not $O(K^2)$, and it is mechanical rather than physical.

## 5.1 Unconnected ports are visible, and that is the point

An expert may declare a port that nothing is connected to. The wind-farm rotor declares a `ROT` port — the shaft — with no drivetrain attached, so the extracted power leaves the system there.

**This makes the model's incompleteness explicit rather than invisible.** Under the old vocabulary, "no generator" was simply an absence. Under the port algebra it is an *open port with a measurable power flow through it*, and Phase G's "add the generator" becomes literally *connect a port*. This is the mechanism by which [[f1-pathmap-and-end-goal]] grows: **each new subsystem is a port connection, not an integration project.**

---

# 6. The global power residual

Because every port carries a power, the whole graph admits one scalar diagnostic. For agent set $\mathcal A$ and port connections $\mathcal P$:

$$\boxed{\;\mathcal R(t)\;=\;\sum_{i\in\mathcal A}\frac{dE_i}{dt}\;+\;\sum_{\Gamma\in\mathcal P}\int_\Gamma\big(e\,f\big)_\Gamma ds\;-\;\underbrace{\sum_{i}\mathcal D_i}_{\text{declared dissipation}}\;-\;\underbrace{P_{\text{ext}}}_{\text{open ports}}\;}$$

$\mathcal R\neq0$ means **the coupling is creating or destroying energy**, and the per-port breakdown localizes where. This is the standard power-bond residual of the co-simulation literature, and it generalizes [[conservation-as-constraint-atlas-0.1]] from a per-edge special case into one number that works in every case study, present and future, without modification.

**Report $\mathcal R(t)$ every macro-step, in every case study, from now on.** It is the cheapest possible physical-plausibility monitor and it needs no reference data.

---

# 7. Migration of the existing case studies

## 7.1 Rocket ([[case-study-rocket-ascent-2d-atlas-0.1]])

| Edge | Old types | **New ports** | Note |
|---|---|---|---|
| $a\!-\!b$ | heat, pressure | `MECH`, `THERM`, **`ADVEC`**$(h_0,Y_k)$ | mass and species cross; the old label was missing both |
| $b\!-\!c$ | heat, stress | `MECH`, `THERM` | wall — no mass crossing. Already correct |
| $c\!-\!d$ | stress, heat, fluid | `MECH`, `THERM`, `ADVEC`$(h_0)$ | |
| $d\!-\!g$ | conservation of heat and fluid | `THERM`, `ADVEC`$(h_0)$, `MECH` | `conservation` was doing the work of naming the ports |
| $e\!-\!f$ | conservation | `MECH`, `THERM`, `ADVEC`$(h_0,Y_k)$ | the least specified edge in the old graph; now fully typed |
| $e\!-\!b$ | heat, pressure | `MECH`, `THERM`, `ADVEC`$(h_0,Y_k)$ | |
| $g\!-\!f$ | heat, fluid | `MECH`, `THERM`, `ADVEC`$(h_0)$ | |

**Four of seven rocket edges were under-specified** — they carried a `heat` label across an interface with mass crossing it and therefore silently omitted advected enthalpy. The migration is a correction.

## 7.2 Wind farm ([[spec-wind-farm-wake-atlas-0.1]])

| Edges | Old types | **New ports** |
|---|---|---|
| 1, 5, 11, 13 (disk faces) | momentum, mass, conservation | `MECH`, `ADVEC`$(h_0)$ |
| 2, 3, 4, 6, 12 (streamwise cuts) | momentum, mass | `MECH`, `ADVEC`$(h_0)$ |
| 7–10, 14, 15 (wake/bypass) | shear | **`MECH`** — same port, tangential component |
| $R_1,R_2$ shaft | *(did not exist)* | **`ROT`, unconnected** — the extracted power, now visible |

The isothermal incompressible setting means no `THERM` port anywhere, and `ADVEC` degenerates to volumetric flux since $\rho$ is constant. **The 15-edge list collapses from four type labels to two port types plus two open `ROT` ports.**

## 7.3 RBC control ([[case-study-rbc-decomposition-atlas-0.1]])

| Edge | Old types | **New ports** |
|---|---|---|
| $\alpha\!-\!\beta$, $\beta\!-\!\gamma$ | heat, stress, conservation | `MECH`, `THERM`, `ADVEC`$(h_0)$ |

The Boussinesq buoyancy coupling means the thermal and mechanical ports are not independent — a fact the old three-label list obscured and the port list makes checkable, since $\mathcal R$ must close across both.

---

# 8. What this does *not* fix

Recorded so the port algebra is not oversold:

- **Nondimensionalization mismatch** between independently trained experts. Real, unavoidable, $O(K)$.
- **Composition error accumulation.** A clean contract does not make composition accurate; that is the open question of [[prior-art-and-novelty-atlas-0.1]] §2.2 and is untouched here.
- **Regime coverage and abstention.** A port says what a coupling *is*, never whether the expert on the other side was trained anywhere near that operating point. With 20 experts the probability that at least one is out of distribution approaches 1, so this gets *worse* as the vocabulary gets cleaner.
- **`ADVEC` is genuinely harder than the true bonds.** It is a multibond with a variable passenger list, and the passenger list is where case-study-specific growth will reappear if it is not disciplined. **Species lists are the thing to watch.**

---

# 9. Implementation notes

1. **Delete the `conservation` edge type from the scaffold**; replace with a `conservative: bool` flag per port connection, default `true`, and always report $r_\Gamma$.
2. **Merge `pressure`/`stress`/`shear`/`momentum` handlers into one `MECH` handler.** In the built Phase-1 scaffold this reduces the typed-attention head count and should be verified as a no-op on the existing tests before the new physics is added.
3. **Adopt preCICE's conservative-vs-consistent mapping distinction per port.** An `ADVEC` flow must map **conservatively** (integral preserved); a `THERM` effort must map **consistently** (pointwise values preserved). Getting this backwards is a classic partitioned-coupling bug, and Atlas is currently making the choice implicitly, per edge, by accident. — **Superseded 2026-08-27:** [[interface-transfer-theory]] §2.2 shows the two mapping classes are the two halves of one adjoint pair, so the choice is **derived from the single declared prolongation rather than adopted as a per-port field**, and getting it backwards stops being expressible. No `mapping` field is needed.
4. **Add $\mathcal R(t)$ to the standard metric set** alongside per-port residuals.
5. **Port declarations live with the expert, not with the case study** — one file per expert in the library, so a case study only names agents, geometry, and connections.

---

## See Also

- [[composition-error-theory]] — why $\mathcal R(t)$ is a falsifier and not a bound, and the storage-function/dissipation-inequality upgrade that turns §2's one-line passivity claim into one
- [[prior-art-and-novelty-atlas-0.1]] — §4, which motivated this page; recommendation 1 is now discharged
- [[f1-pathmap-and-end-goal]] — the roadmap this vocabulary exists to make survivable
- [[edge-generation-atlas-0.1]] — Mechanism A now materializes ports rather than ad-hoc types
- [[conservation-as-constraint-atlas-0.1]] — per-port enforcement vs. measurement
- [[expert-library-atlas-0.1]] — experts are cut by governing family; ports are how the cuts talk
- [[backbone-1.1]] — the symmetric+skew port-Hamiltonian structure this connects to
- [[spec-wind-farm-wake-atlas-0.1]], [[case-study-rocket-ascent-2d-atlas-0.1]], [[case-study-rbc-decomposition-atlas-0.1]] — migrated in §7
- [[interface-transfer-theory]] — **amends §5's connection rule** (geometric coincidence becomes a declared prolongation), **derives §9.3's mapping choice** instead of declaring it, and adds the power constraint $s_e s_f=s_P$ that §8's nondimensionalization cost was missing. §2's power bond is what forces the reduction to be the adjoint of the prolongation
- [[plug-in-composition-theorems]] — §5.1's open ports become a composite's port list under nesting, and §8's *"with 20 experts P(at least one OOD) → 1"* becomes a **horizon**, $T_{\text{abs}}\approx\Delta t/(Kp)$, that degrades linearly in library size
- [[end-to-end-architecture-spec]] — turns §5's connection rule into an **eight-condition admissibility checklist** (L3), makes §9.3's mapping choice a compile-time refusal, and specifies the **`PortAmendment` procedure** §3's closed vocabulary needs before a sixth port type can be added — that procedure belongs on this page as a binding section (W32)
