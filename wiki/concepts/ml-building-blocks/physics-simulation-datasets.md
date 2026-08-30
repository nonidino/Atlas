# Physics Simulation Datasets — Landscape and Coverage Map

**Type:** Concept page — external resource landscape (folder: `concepts/ml-building-blocks/`)
**Related:** [[neural-surrogates]], [[transfer-learning-fine-tuning]], [[incremental-transfer-roadmap]], [[poseidon-pde-foundation-model]], [[walrus-paper]], [[gphyt-physics-foundation-model]]
**Written:** 2026-08-09, prompted by an audit of [[impl-atlas-0.1-phase0-scope-and-data]]'s claim that "Atlas 0.1 has no dataset."

> **Audit verdict up front:** that claim is **substantially wrong for per-expert pretraining and exactly right for the interface layer.** Public data covers three of Atlas's four governing families well. What it does not cover — anywhere, by anyone — is *labelled flux data on the interfaces between subdomains*, which is precisely the quantity Atlas exists to learn. §4 works this through.

---

# 1. Intuition

There is a persistent failure mode in surrogate-modelling projects: looking at a *scenario* (a rocket, a reactor, a turbine), failing to find a dataset of that scenario, and concluding that data must be generated from scratch. This is almost always an error, because **models learn governing equations, not scenarios.** The right question is never "is there a rocket dataset?" — it is "what equation families does this decompose into, and which of *those* have data?"

That reframing is exactly what [[expert-library-atlas-0.1]] already argues for *architecture* — cut along governing-equation families, not along physical labels — and it applies with equal force to data. An architecture organized by equation family should have a **data strategy organized the same way.** Atlas's spec cuts the architecture correctly and then, in the very next phase, evaluates data availability by scenario. That mismatch is the finding of this page.

The counterweight, and it is a real one: **"same governing family" is not the same as "usable."** A dataset can solve your equation and still be unusable because its boundary conditions, dimensionality, nondimensionalization, or resolution place it outside the regime you need. §3 is a coverage map; §3.5 is the domain-gap discount that must be applied to it.

---

# 2. Theory — decompose before you shop

## 2.1 Atlas 0.1's governing equations, stated compactly

Seven agents, but only **three PDE families plus one ODE** ([[impl-atlas-0.1-phase0-scope-and-data]] §2):

**(F1) Compressible reacting Navier–Stokes** — hyperbolic with viscous and reactive source terms. Agents `a`, `b`, `e` (confined, with heat release) and `d`, `f`, `g` (free boundaries, no reaction):

$$\partial_t U+\partial_z F(U)+\partial_y G(U)=\partial_z F_{\text{visc}}+\partial_y G_{\text{visc}}+S,\qquad
U=\begin{bmatrix}\rho\\\rho u\\\rho v\\E\end{bmatrix}$$

with ideal-gas closure $E=\frac{p}{\gamma-1}+\frac12\rho(u^2+v^2)$, and for agent `a` a single progress variable under a one-step Arrhenius law:

$$\partial_t(\rho Y)+\nabla\!\cdot\!(\rho\mathbf u Y)=\rho\,\omega,\qquad \omega=A(1-Y)\exp\!\left(-\frac{T_{\text{act}}}{T}\right),\qquad S_E=\rho\,\omega\,q_{\text{rxn}}$$

**Note carefully that agents `a`,`b`,`e` and `d`,`f`,`g` share this equation identically.** They are split into two experts by *boundary condition and regime* (confined + source vs. free), not by equation — a fact [[expert-library-atlas-0.1]] flags as an open question and which matters here, because it means **one body of public data serves both experts.**

**(F2) Transient conduction** — parabolic. Agent `c`:
$$\rho_s c_{p,s}\,\partial_t T=\nabla\!\cdot\!(k_s\nabla T)$$
with Robin (convective + linearized-radiative) conditions on both faces.

**(F3) Quasi-static plane-stress thermoelasticity** — elliptic. Agent `c`:
$$\nabla\!\cdot\!\boldsymbol\sigma+\mathbf b=0,\qquad
\boldsymbol\sigma=\frac{E_s}{1-\nu^2}\left[(1-\nu)\boldsymbol\varepsilon+\nu\,\mathrm{tr}(\boldsymbol\varepsilon)\mathbf I\right]-\frac{E_s\alpha_s\Delta T}{1-\nu}\mathbf I$$

**(F4) 3-DOF rigid-body ODE** — closed form, **not learned**, so it needs no data at all:
$$m\dot{\mathbf v}=\mathbf F_{\text{thrust}}+\mathbf F_{\text{aero}}+m\mathbf g(h),\qquad I(m)\dot\omega=\tau,\qquad \dot m=-\dot m_p$$

Plus the US Standard Atmosphere 1976, which is an algebraic table, not data to learn.

## 2.2 The two things Phase 2 conflates

Solo expert training ([[impl-atlas-0.1-phase2-experts]] §3.2) asks each expert to learn two separable things:

| | What it is | Data source |
|---|---|---|
| **(i) Interior dynamics** | evolve the governing family inside the agent | **public data works** — this is generic F1/F2/F3 |
| **(ii) Interface response** | respond correctly to prescribed boundary flux at a declared edge | **must be generated** — supervised by `/iface` |

The spec trains both from the same generated corpus. **They do not have to come from the same corpus,** and separating them is where the leverage is.

---

# 3. Coverage map — what actually exists, per family

## 3.1 F1a — compressible flow with shocks (no reaction)

**Abundant, and Atlas already depends on it indirectly.**

| Source | Content | Fit for Atlas |
|---|---|---|
| **PDEgym** ([[poseidon-pde-foundation-model]]) | 77,840 trajectories, 11 snapshots each, 6 operators — 4 compressible Euler (Riemann quadrants, curved Riemann interfaces, Kelvin–Helmholtz, Gaussian bumps) + 2 incompressible NS | Strong on shocks/gas dynamics. **All periodic BC, 2D Cartesian, inviscid Euler** |
| **The Well** (15 TB, 16 datasets) | includes Euler multi-quadrant (~5,000 trajectories), Rayleigh–Bénard, shear flow, turbulent radiative layer, supernova, MHD | Broad; the compressible subset is the relevant slice |
| **PDEBench** | compressible NS at a range of Mach and viscosity | Includes *viscous* compressible, which PDEgym lacks |

**This is already Atlas's plan, laundered through weights rather than data:** Poseidon *is* PDEgym, and [[impl-atlas-0.1-phase2-experts]] bootstraps two experts from it. The audit finding is that this route is available to the third expert too.

## 3.2 F1b — reacting flow / combustion

**Exists, and the spec does not mention it.** This is the clearest gap in the Phase 0 audit.

- **BLASTNet 2.0** — a community network-of-datasets for combustion ML: **2.2 TB, 744 full-domain samples from 34 high-fidelity DNS** of reacting *and* non-reacting compressible turbulent flow. Explicitly built to address "the current limited availability of 3D high-fidelity reacting and non-reacting compressible turbulent flow simulation data," and already used to benchmark 49 variants of five deep-learning approaches for 3D super-resolution, with neural-scaling analysis.
- The **TNF workshop / Sandia flame** series (D/E/F) — *experimental* turbulent nonpremixed flame data; validation-grade, not ML-scale.
- Chemical mechanisms (GRI-Mech, San Diego, Cantera) are **mechanisms, not data** — they generate, they don't supply.

**Discount to apply (§3.5):** BLASTNet is 3D DNS with detailed chemistry; Atlas is 2D planar FV with a *one-step progress variable*, deliberately ([[impl-atlas-0.1-phase0-scope-and-data]] §2.1 calls this "the specification, not a placeholder"). What transfers is a **turbulent reacting-structure prior** — heat-release-driven density and pressure coupling, flame-front morphology — not chemistry. Sample count is also modest: 744 full-domain samples is a pretraining set, not a large one.

## 3.3 F1c — external aerodynamics over bodies

**Exists, with a regime mismatch.**

| Source | Content | Fit |
|---|---|---|
| **AirfRANS** | steady-state RANS over 2D NACA 4/5-digit airfoils, **subsonic incompressible**, refined meshes, deliberately data-scarce; basis of the NeurIPS 2024 ML4CFD competition | Right *geometry* problem (flow over a body, non-Cartesian), wrong *regime* (incompressible, steady, subsonic) |
| **Poseidon's SE-AF task** | steady Euler over an airfoil, non-Cartesian handled by masking | Directly relevant: demonstrates the periodic-Cartesian → body-masked transfer Atlas's agent `d` needs |
| **Flowbench** | flow over complex geometries; part of [[walrus-paper]]'s 19-scenario pretraining set | Geometry diversity |
| DrivAerStar, CarBench | industrial 3D vehicle aerodynamics | Wrong dimensionality and regime; noted for completeness |

## 3.4 F2 + F3 — conduction and thermoelasticity

**Genuinely thin. Atlas's `thermostruct` expert is the one with no good public option.**

- **TFRD** (Temperature Field Reconstruction Dataset) — built explicitly because "there is no public dataset for research on reconstruction methods" for heat-source systems. But it is a *reconstruction* task (sensor → field), steady, and carries **no structural mechanics**.
- **DeepEDH / conjugate heat transfer** — 1,500 FEM simulations of a liquid-cooled battery cold plate, unstructured → structured remapping. A one-off application dataset, not a benchmark, and again **thermal only**.
- **MeshGraphNets' deforming-plate** and **SimJEB** — structural FEM, but *isothermal*. No thermal expansion coupling.
- **PLAID** — a unified data model for ML on heterogeneous physics simulations; infrastructure for exchanging such datasets rather than a corpus itself. Worth tracking as the format problem gets worse.

**No public corpus pairs transient conduction with quasi-static thermoelastic stress on a shell.** Poseidon's smooth elliptic/parabolic pretraining coverage is, as [[impl-atlas-0.1-phase2-experts]] §2.4 already argues, the best available proxy — and it is a proxy, not data.

## 3.5 The domain-gap discount

Every row above must be discounted by four axes before it counts as usable:

| Axis | Public data | Atlas needs | Severity |
|---|---|---|---|
| **Boundary conditions** | PDEgym: *all periodic*. The Well: mostly periodic/simple | walls, no-slip, choked throat, mass-flow inlet, freestream | **High** — walls are where Atlas's physics lives |
| **Nondimensionalization** | each corpus has its own scaling | per-agent refs frozen per episode ([[00-atlas-0.1-implementation-plan]]) | Medium — an adapter/renormalization problem, tractable |
| **Dimensionality** | BLASTNet 3D; PDEgym 2D | 2D planar | Low — [[walrus-paper]]'s 2D↔3D augmentation addresses exactly this |
| **Regime** | AirfRANS subsonic incompressible steady | transonic/supersonic unsteady | **High** for AirfRANS, low for PDEgym |

**The counter-evidence that the discount is survivable is Poseidon's own result:** its pretraining is *narrow* — two PDE families, all convection-dominated, all periodic BC, 2D Cartesian — and it still transfers to 15 out-of-distribution downstream tasks including elliptic, parabolic, non-periodic-BC, and non-Cartesian ones. [[transfer-learning-fine-tuning]]'s diversity principle and [[poseidon-pde-foundation-model]]'s "recombinable physical primitives" mechanism are the reason. Narrow-but-relevant pretraining beats no pretraining; it does not beat in-distribution data.

## 3.6 The genuine holes

Two things this audit could not find anywhere, and both are Atlas-specific:

1. **Confined nozzle flow through a choking throat.** NASA's validation archives carry *single validation cases* (transonic bumps, axisymmetric nozzles), not ML-scale corpora. This matters more than it looks: the **choked-throat check** is [[impl-atlas-0.1-phase2-experts]]'s most informative M4 gate — it tests whether the expert learned that a choked throat isolates the chamber from downstream conditions — and **no public dataset contains a choked throat at all.** That gate can only be trained and graded on generated data.
2. **Interface flux fields on labelled subdomain boundaries.** See §4.

---

# 4. Why the interface data genuinely cannot be found

[[impl-atlas-0.1-phase0-scope-and-data]] §3.4 marks this "Critical," and it is right:

> the saved record must include the **interface flux fields themselves**, not only the agent interiors. Those fluxes are the supervision signal for the edge behaviour and for the conservation constraint.

Every public PDE corpus stores **whole-domain state on a single grid**. None stores $\left.\left(\rho u_n,\ p,\ q_n,\ \boldsymbol\sigma\!\cdot\!\mathbf n\right)\right|_{\Gamma_{ij}}$ on a *declared* interface $\Gamma_{ij}$ between two named subdomains — because no other architecture has declared interfaces to store them for. Monolithic solvers have no interfaces; domain-decomposition solvers have them but discard them as an implementation detail rather than publishing them as labels.

**[AI Inference]:** this is not a temporary gap that a future release fills. It is structural. A dataset only carries the labels its producing architecture needed, and the wiki already recorded the general form of this in [[pfm-purpose-and-direction]]: *"need data for interaction interfaces between agent types, which no existing PDE dataset (The Well included) is organized around."* Atlas's central thesis is therefore **untestable on existing benchmarks by construction** — which is simultaneously the argument for the architecture's novelty and the reason its data cost is irreducible.

A corollary worth stating plainly: if the interface layer is the only part requiring generated data, then **generated episodes should be budgeted for interface learning, not interior learning** — the interiors can come from public pretraining. That reverses the default assumption behind [[impl-atlas-0.1-compute-and-training-budget]]'s corpus sizing.

---

# 5. Implementation — the revised three-tier data strategy for Atlas 0.1

| Tier | Source | Serves | Cost |
|---|---|---|---|
| **T1 — public pretraining** | PDEgym (via Poseidon weights, already planned) · **BLASTNet 2.0** (new) · The Well compressible subset | interior dynamics of F1 for both flow experts | free |
| **T2 — generated single-agent** | classical solvers, *uncoupled*, cheap short runs | the regime holes: choked throat, wall-bounded duct flow, thermoelastic shell | small |
| **T3 — generated coupled** | the Phase 0 coupled generator with `/iface` | interface response, conservation constraint, Phase 3 joint, Phase 4 grading | the irreducible cost |

Three concrete changes this implies:

1. **`reacting_flow` should not be "from scratch."** [[impl-atlas-0.1-phase2-experts]] §2.2 justifies from-scratch by "neither confirmed donor was pretrained on confined reacting duct flow" — true for *donor weights*, but it never asks about *donor data*. BLASTNet exists. **[AI Inference]:** pretraining `reacting_flow` on 2D slices of BLASTNet's reacting DNS, then fine-tuning on T2/T3, is strictly better than random init, and costs one download plus a normalization adapter.
2. **T2 is a new, cheap category the spec doesn't have.** Uncoupled single-agent runs need no macro-step synchronization, no interface bookkeeping, and can be run at whatever short horizon the regime requires — far cheaper per useful sample than a coupled episode. The choked-throat gate in particular wants *many short nozzle runs at varied back-pressure*, which is a T2 job, not a T3 one.
3. **T3 gets smaller.** **[AI Inference]:** with experts arriving pretrained on T1+T2, the coupled corpus serves interface response and joint fine-tuning rather than teaching three governing families from nothing. A reduction from the budget page's 144 episodes to roughly **60–90** looks defensible — but this is an inference from transfer-learning results on *other* problems, not a measurement, and it should be validated by an ablation (train one expert with and without T1 pretraining, compare the M4 gate) before the corpus size is cut.

**The larger win is not cost, it is the M4 gates.** They are measured on **held-out corner configs** — the most under-expanded and most over-expanded cases, deliberately chosen as extrapolation. Pretrained experts generalize off-distribution better than from-scratch ones; that is the single most replicated result in [[transfer-learning-fine-tuning]]. Saving compute is a side effect.

---

# 6. Pitfalls

- **Shopping by scenario instead of by equation family.** The error this page exists to correct. "Is there a rocket dataset?" has a useless answer; "is there compressible-reacting-flow data?" has a 2.2 TB one.
- **Treating "same governing family" as "usable."** Apply §3.5's four-axis discount. Periodic-BC data does not teach wall behaviour, and walls are where a nozzle's physics lives.
- **Assuming donor *weights* are the only transfer path.** Poseidon-the-weights and PDEgym-the-data are the same physics; a corpus you can fine-tune on directly is more flexible than a checkpoint you must adapt to, and does not carry the checkpoint's license question.
- **Expecting the interface gap to close.** It will not (§4). Budget for it permanently.
- **Cutting the generated corpus on the strength of §5's estimate.** It is an [AI Inference]. Run the with/without-pretraining ablation on one expert first.
- **Forgetting that some experts need no data at all.** `rigid_body` is closed-form and asserted bit-identical to `solvers/trajectory.py`. No amount of trajectory data trains it, so trajectory-rich episodes have far less value than they appear to.

---

## See Also

- [[impl-atlas-0.1-phase0-scope-and-data]] — the "no dataset" claim this page audits; §3.4's interface-flux requirement
- [[impl-atlas-0.1-phase2-experts]] — where T1 pretraining would change the from-scratch/bootstrap split
- [[impl-atlas-0.1-compute-and-training-budget]] — corpus sizing this page's §5 revises downward
- [[poseidon-pde-foundation-model]] — PDEgym; the narrow-pretraining-still-transfers evidence
- [[walrus-paper]] — The Well + Flowbench; 2D↔3D augmentation for the dimensionality gap
- [[gphyt-physics-foundation-model]] — precedent for *supplementing* The Well with custom-generated engineering scenarios
- [[transfer-learning-fine-tuning]] — the diversity principle and why held-out corners favour pretrained experts
- [[neural-surrogates]] — the wiki's prior, thinner list of benchmark datasets
- [[pfm-purpose-and-direction]] — item 8, which already predicted the interface-data gap
