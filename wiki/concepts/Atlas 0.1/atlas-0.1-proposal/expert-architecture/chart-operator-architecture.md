# The chart operator — a learned expert built for domain decomposition

**Type:** Concept page — **architecture proposal, version 0** (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** written 2026-09-30. **A design, not a build**: nothing here is implemented, trained or measured. Where the page states a theorem, it is a classical one with its source, or a statement that [[formal-proofs-plan]] schedules for proof. Everything else that goes beyond the vault's record is marked **[AI Inference]**.
**Superseded in part, 2026-10-01:** the owner has decided the chart layer, the operator's input, the cut rule, the chart certificate and the coupling stack; the decisions, their evidence and what they change here are in [[chart-operator-design-decisions]], with the measured coupling costs in [[coupling-cost-and-complexity]]. This page is rewritten as version 1 once the owner confirms the whole design.
**Hub:** [[00-proposal-workstreams]] · **Companions:** [[chart-operator-training-and-cost]] · [[dd-neural-prior-art-2026]]
**Answers to:** [[outcome-c5-requirements-for-dd-native-experts]] (S1–S12) · **Built on:** [[port-algebra-atlas-0.1]] · [[atlas-and-standard-dd-theory]] · [[composition-error-theory]] · [[probed-dtn-coupling]] · [[defect-correction-learned-operator]] · [[master-error-bound]]

---

## 0. The idea in one paragraph

A manifold is described by an atlas: charts, each a flat square mapped onto one patch, glued by transition maps. **The chart operator takes the metaphor literally.** The decomposition cuts the domain into pieces that are each a curved square. A classical, provably non-folding map charts each piece onto the unit square. The learned operator works **only on the unit square**, and receives the piece's geometry as the **metric of its chart**, the "generalised Jacobian" the owner proposed. Neighbouring charts exchange **wave variables** through Atlas's typed ports, so the coupled iteration converges at the optimized-Schwarz rate, and it provably converges whenever each expert passes a checkable bound, **however well or badly it was trained**. Three separations carry the design:

$$\boxed{\ \text{topology}\to\text{the decomposition}\qquad\text{geometry}\to\text{the chart}\qquad\text{physics}\to\text{the operator}\qquad\text{agreement}\to\text{the waves}\ }$$

Each separation gives the learned part less to learn, and gives the classical part something it can guarantee.

---

## 1. What it has to do

The owner's three goals, each against the measured specification it must meet:

| goal (the owner, 2026-09-30) | measured requirement ([[outcome-c5-requirements-for-dd-native-experts]]) | where this design meets it |
|---|---|---|
| **work on almost any geometry** | S1 (the frozen expert accepts only a uniform $128^2$ grid; $61\%$ of the car's unknowns were unreachable); S4 (a bounded, measurable domain of dependence) | §2: every piece is charted onto the square, so the operator never sees an irregular grid. Locality is the chart's own |
| **converge in few iterations inside a decomposition** | S3 (interface data as an input); S5, S5a (the right slow-mode response, along the whole approach); S6 (never closer to singular than the classical map) | §3.2 (interface data declared), §4 (wave variables with an optimized impedance; a coarse space; the contraction contract), §5 (training on the response) |
| **one standard, modular format** | S2 (runs at the coupling's step); S8 (a same-class reference); S10 (precision); S12 (a usable licence) | §6, the Expert Card: one manifest every expert carries, and that the compiler reads |
| (implied) **worth its slot** | S7, S7a (cheaper than the cheapest classical alternative in the same slot); S9, S11 (accurate, stable) | §7 names what the design cannot promise, and [[chart-operator-training-and-cost]] §6 is the gate |

---

## 2. Geometry: charts

### 2.1 The decomposition makes the charts

**Rule: every piece is a topological square.** The layout ([[demo-finish-plan]] §2.3; `atlas/workbench/layout.py`) already cuts a domain into pieces that are "rectangles in the domain's own coordinates". A piece has four **sides**. Each side is an interface to one neighbour, or a piece of physical boundary. Its four **corners** are where the cuts meet the walls or each other.

**Why topology belongs to the decomposition.** A hole, a fork or a loop makes a domain's topology. Cut well, every piece is simply connected and four-sided, whatever the whole is: a ring becomes arcs, a fork becomes three channels and a junction. **[AI Inference]:** the 2026 topology-generalisation benchmark found that neural operators' errors under a topology change track the input's mean frequency rather than the topology itself ([[dd-neural-prior-art-2026]] §1.3). An operator that never sees topology cannot be caught out by it.

### 2.2 The chart map

A chart is $F_i:\hat\Omega=[0,1]^2\to\Omega_i$, orientation-preserving, with $J=\det DF_i>0$, taking the square's sides to the piece's sides.

| map | how it is made | guarantee | cost | role |
|---|---|---|---|---|
| **conformal onto the square, up to scale** | the layout's own **along** and **across** coordinates: harmonic, along $=0,1$ on two opposite sides with no flux through the other two, across the reverse | see below | two sparse solves per piece, already made by `layout.py` | **default for scalar physics** |
| **Winslow (harmonic, Dirichlet)** | $\xi,\eta$ harmonic in $\Omega_i$, with boundary values taking $\partial\Omega_i$ onto $\partial\hat\Omega$ by arc length (Winslow, *J. Comput. Phys.* 1, 1966) | **injective by the Rado–Kneser–Choquet theorem**: a harmonic map onto a *convex* domain whose boundary map is a homeomorphism is a diffeomorphism. Discretely, Floater's convex-combination maps are one-to-one (*Math. Comp.* 72, 2003) | two sparse solves, then an inversion | **default when the conformal map's corners are poor** |
| **transfinite interpolation** | Gordon & Hall's Coons patch from the four side curves (*IJNME* 7, 1973) | none: it folds on a non-convex piece | explicit, no solve | fast path for near-rectangles, **checked by $\min J>0$** |
| conformal (Schwarz–Christoffel, Riemann) | classical conformal maps | injective, but **crowding**: on an elongated piece $\lvert F'\rvert$ varies exponentially | numerical | analysis only |
| learned (Geo-FNO) | trained end to end | none | inside the network | **not used**: a folded chart would be silent |

**The layout's coordinates are a conformal chart. [AI Inference], a continuum argument to be checked.** Let $Q$ be a piece with sides $A,B,C,D$ in order. The along coordinate $u$ ($u=0$ on $A$, $u=1$ on $C$, $\partial_nu=0$ on $B,D$) is, up to a factor, the real part of the conformal map of $Q$ onto a rectangle $[0,M]\times[0,1]$, where $M$ is $Q$'s conformal modulus. Its harmonic conjugate is constant on $B$ and $D$, and so is the across coordinate. Hence

$$x\;\mapsto\;(\text{along},\ \text{across})\;=\;\bigl(\operatorname{Re}f/M,\ \operatorname{Im}f\bigr)$$

is a conformal map composed with an axis scaling: a diffeomorphism onto the unit square whose metric is **isotropic up to that scaling**, $G=\lvert f'\rvert^2\operatorname{diag}(M^{-2},1)$.

Two consequences:
- **The decomposition controls the distortion.** Crowding comes from large modulus. The layout cuts at equal cell counts along the shape, so each piece's modulus stays near one. Registering a bound, for example $M\in[\tfrac12,2]$, turns chart quality into a rule the layout must meet.
- **The layout's cuts meet the walls at right angles** (the no-flux condition), so a chart's corners are right angles, where a conformal map is regular. A physical re-entrant corner stays singular, as the solution is there too.

### 2.3 Pulling the physics back

**Scalar, divergence form.** $-\nabla\!\cdot(\kappa\nabla u)=f$ on $\Omega_i$ becomes, with $\hat u=u\circ F$, $G=DF^{\top}DF$ and $J=\det DF$,

$$-\hat\nabla\cdot\bigl(\hat{\mathsf K}\,\hat\nabla\hat u\bigr)=J\,\hat f,\qquad \hat{\mathsf K}\;=\;J\,DF^{-1}\,\kappa\,DF^{-\top}\;=\;J\,\kappa\,G^{-1}\ \ (\kappa\text{ isotropic}).$$

**The flux through a side is unchanged** (Nanson's formula, $\mathbf n\,ds=J\,DF^{-\top}\hat{\mathbf n}\,d\hat s$):

$$\int_{\Gamma}\kappa\,\partial_n u\,ds\;=\;\int_{\hat\Gamma}\bigl(\hat{\mathsf K}\hat\nabla\hat u\bigr)\cdot\hat{\mathbf n}\,d\hat s .$$

So **a port's power is the same number in either frame**, and the port algebra's balance holds chart by chart without any conversion ([[port-algebra-atlas-0.1]]).

**Vector fields: the Piola transform.** A flux or velocity $\mathbf v$ pulls back as $\hat{\mathbf v}=J\,DF^{-1}\,(\mathbf v\circ F)$. Then

$$\nabla\!\cdot\mathbf v=J^{-1}\,\hat\nabla\cdot\hat{\mathbf v},\qquad \mathbf v\cdot\mathbf n\,ds=\hat{\mathbf v}\cdot\hat{\mathbf n}\,d\hat s .$$

**Divergence-free in the chart is divergence-free in the world, exactly.** A stream function $\hat\psi$ on the square gives $\hat{\mathbf v}=\hat\nabla^{\perp}\hat\psi$, divergence-free to round-off on a staggered grid, and $\mathbf v=J^{-1}DF\,\hat{\mathbf v}$.

**Elasticity and flow with Cartesian components.** Keep the physical components and transform only the derivatives, $\nabla=DF^{-\top}\hat\nabla$. Then the operator needs the rotation too. Write the polar decomposition $DF=R\,U$, with $R$ a rotation and $U=G^{1/2}$.

### 2.4 What the operator receives: the generalised Jacobian

| channel | 2-D count | meaning |
|---|---|---|
| $J\,G^{-1}$ | 3 (symmetric) | the metric density: all a scalar isotropic operator needs |
| $J$ | 1 | the volume factor, for sources, capacities and time derivatives |
| $\theta$, the angle of $R$ | 1, vector physics only | the chart's local rotation |

**Invariances bought by construction.**
- A scalar problem depends on the piece only through $J\kappa G^{-1}$ and $J$, so moving or rotating the piece changes nothing the operator sees. That is exact equivariance to rigid motions without averaging ([[symmetry-averaging-atlas-0.1]] gets it by averaging instead).
- For vector physics, rotating the components into the chart's frame ($\mathbf u_{\text{loc}}=R^{\top}\mathbf u$) gives the same.

### 2.5 Three dimensions

**Rado–Kneser–Choquet fails in three dimensions**: harmonic maps onto convex bodies need not be injective (Laugesen, *Complex Variables* 28, 1996). A 3-D chart operator needs one of three things:
1. a decomposition into topological cubes, which is the hexahedral-meshing problem, open in general;
2. charts from a map with injectivity enforced, for example foldover-free optimisation (Garanzha et al., *ACM TOG* 40, 2021);
3. a mesh- or point-based operator for the pieces no chart fits (GINO, Transolver; [[dd-neural-prior-art-2026]] §1.3), exchanging the same wave variables.

**[AI Inference]:** option 3 as a fallback keeps the coupling contract of §4 intact whatever the local operator is. That is the point of standardising the contract rather than the network.

---

## 3. The operator on the square

### 3.1 Inputs and outputs

| in | shape | notes |
|---|---|---|
| geometry | fields on the $n\times n$ reference grid | §2.4 |
| coefficients, sources | fields | pulled back |
| state $\hat u^n$ | fields | for a time-dependent family |
| $\Delta t$, physical parameters | scalars | conditioning, not channels (S2) |
| per side: its kind, impedance $Z$, incoming wave $a(\hat s)$ | a flag, a scalar and a function on $[0,1]$ | §4 (S3) |
| **out** | | |
| $\hat u^{n+1}$ | fields | |
| per side: outgoing wave $b(\hat s)$ | functions on $[0,1]$ | **computed from the field**, never from a separate head, so the field and what the neighbours receive cannot disagree |
| optionally: JVPs, a residual estimate | | for Krylov coupling (§4.5) and for the compiler |

### 3.2 From the boundary into the domain

**The lifting.** The owner's "boundary-to-domain mapping". The side data enter the interior through a fixed, exact extension, and the network learns only a correction that vanishes on the boundary:

$$\hat u\;=\;\mathcal T[g]\;+\;\beta(\xi,\eta)\,\mathcal N_\theta(\cdots),\qquad \beta=16\,\xi(1-\xi)\,\eta(1-\eta),$$

with $\mathcal T[g]$ the transfinite (Coons) interpolation of the Dirichlet data $g$ on the four sides. **Dirichlet data are then met exactly, whatever the network returns.** For Robin and wave data, which cannot be imposed this way, the lifting still supplies a field that carries the boundary information everywhere from the first layer. The side condition is met through the loss, and checked.

**[AI Inference]:** this is the physics-encoding spectrum's level 5 for boundary data, and it answers S3 at the architecture level. The frozen checkpoint was driven by overwriting a halo it was never trained to read.

### 3.3 A conservative head, for conservation laws

For a family in conservation form, the network outputs **fluxes on the reference grid's faces**, not states, and the state update is their exact discrete divergence:

$$J\,\frac{\hat u^{n+1}-\hat u^{n}}{\Delta t}\;=\;-\hat\nabla_h\cdot\hat{\mathbf F}_\theta\;+\;J\,\hat s .$$

Content is conserved **exactly** in every cell. **The flux a side sends its neighbour is the flux the chart used in its own balance**, so the assembled domain conserves exactly too. That is the port algebra's flux matching, now inside the expert ([[conservation-as-constraint-atlas-0.1]]).

### 3.4 The backbone

**Recommended: a U-shaped convolutional neural operator** on the square, in the manner of the Convolutional Neural Operator (Raonic et al., *NeurIPS* 2023):
- anti-aliased up- and down-sampling, so it is consistent across resolutions: trained at $32^2$, $64^2$ and $128^2$, it has same-class references at other resolutions (S8);
- time and parameters enter by adaptive normalisation;
- it is local and cheap on a CPU.

| alternative | why not first |
|---|---|
| FNO | its spectral layers assume periodicity. The square's sides are not periodic, and padding is a patch |
| a transformer (scOT, Poseidon's backbone) | its attention cost on a CPU. S7 prices every expert against the cheapest classical alternative, and the demo runs on a laptop. **But it is the path to fine-tuning Poseidon, DPOT or Walrus** ([[chart-operator-training-and-cost]] §4), so the interface must not rule it out |
| a graph operator | it needs no chart, which is its virtue and why it is §2.5's 3-D fallback, but it gives up the square's cheap structure |

---

## 4. Coupling: wave variables through typed ports

### 4.1 The variables

A port carries an effort $e$ and a flow $f$ whose product is power per unit length of side, with $f$ positive **into** the chart ([[port-algebra-atlas-0.1]]). For an impedance $Z>0$, define the incoming and outgoing waves

$$a=\frac{e+Zf}{2\sqrt Z},\qquad b=\frac{e-Zf}{2\sqrt Z},\qquad\text{so}\qquad ef=a^2-b^2 .$$

**Power in is incoming wave power minus outgoing.** On a shared side the flows are opposite ($f_j=-f_i$) and, at the solution, the efforts are equal. So the neighbour's outgoing wave **is** my incoming one:

$$a_i=b_j\quad\text{on }\Gamma_{ij}\qquad\Longleftrightarrow\qquad e_i+Zf_i=e_j-Zf_j,$$

which is a Robin transmission condition. **Optimized Schwarz is wave-variable exchange.** It is also the scattering formulation long used to couple passive systems robustly: Niemeyer & Slotine's wave variables in teleoperation (*IEEE J. Oceanic Eng.* 16, 1991), and Després's and Collino, Ghanemi & Joly's analyses of Robin-type DD for waves (*CMAME* 184, 2000).

**For `THERM`**, whose pair is $(T,\ q_n/T)$, the exchange uses the linearisation about the interface temperature. That gives the classical Robin condition in $(T,q_n)$. This is a detail for the build, not a change to the port algebra.

### 4.2 An agent is a scattering map

Chart $i$, with its sources and state fixed, maps incoming waves on its interface sides to outgoing ones, $b_i=S_i(a_i)$. For a linear, source-free local problem that absorbs power, $\int ef\,ds\ge0$, so

$$\lVert b\rVert^2=\lVert a\rVert^2-\int_{\partial\Omega_i}ef\,ds\;\le\;\lVert a\rVert^2 .$$

**A passive agent's scattering map is non-expansive.** For diffusion, let $\Lambda$ be the chart's Dirichlet-to-Neumann map, $f=\Lambda e$, positive semi-definite. Then $a=(I+Z\Lambda)e/(2\sqrt Z)$ and $b=(I-Z\Lambda)e/(2\sqrt Z)$, so $S$ is a Cayley transform:

$$S=(I-Z\Lambda)(I+Z\Lambda)^{-1},\qquad\text{eigenvalues}\ \ \frac{1-Z\lambda}{1+Z\lambda}\in(-1,1]\ \ \text{for}\ \lambda\ge0 .$$

**It contracts every mode, but not uniformly**: the factor tends to $-1$ as $\lambda\to\infty$ (the highest frequencies) and is $+1$ at $\lambda=0$ (a floating piece's constant). Its magnitude approaches one at both ends.

### 4.3 Choosing the impedance

The global iteration is $a^{k+1}=\Pi\,S(a^k)+c$, with $S=\operatorname{diag}(S_i)$ and $\Pi$ the swap that hands each side's outgoing wave to its neighbour, an isometry. For two pieces and a mode with DtN symbols $\lambda_1,\lambda_2$, the factor per round trip is $\frac{1-Z\lambda_1}{1+Z\lambda_1}\cdot\frac{1-Z\lambda_2}{1+Z\lambda_2}$. Minimising its worst case over the resolved band $[\lambda_{\min},\lambda_{\max}]$ gives the zeroth-order optimized impedance $Z^\star=1/\sqrt{\lambda_{\min}\lambda_{\max}}$, which is the Robin parameter $p^\star=1/Z^\star$ of the optimized-Schwarz literature. For the Laplacian, $\lambda\sim\lvert k\rvert\in[\pi/L,\pi/h]$, so $p^\star\sim h^{-1/2}$ and the contraction is $1-O(h^{1/2})$ against classical Schwarz's $1-O(h)$ (Gander, *SIAM J. Numer. Anal.* 44, 2006; with overlap and second-order conditions, $1-O(h^{1/5})$). **The host chooses $Z$ from the port's physics and passes it in**, so one trained expert serves every $Z$.

### 4.4 The contraction contract: convergence however the expert was trained

Let $\tilde S_i$ be a learned scattering map and $S_i$ the true one.

**Level 1, non-expansive.** If every $\operatorname{Lip}(\tilde S_i)\le1$ and a fixed point exists, the averaged iteration $a\leftarrow(1-\omega)a+\omega(\Pi\tilde S a+c)$, $0<\omega<1$, converges in finite dimensions (Krasnoselskii–Mann). There is no rate.

**Level 2, contractive on the fine space.** Split the interface space into a small **coarse space** $V_0$ (a few low modes per chart) and its complement. With overlap, a true Schwarz map contracts the complement's high modes uniformly: they decay across the overlap. The low modes that do not contract are $V_0$'s, which the host solves directly. If each learned map satisfies

$$\operatorname{Lip}\bigl(\tilde S_i\big|_{V_0^\perp}\bigr)\le\rho<1\qquad\text{and}\qquad\sup_a\bigl\lVert\tilde S_i(a)-S_i(a)\bigr\rVert\le\delta,$$

then the two-level iteration converges geometrically at rate $\rho$ **from any start**, to a limit within

$$\lVert\tilde a^\star-a^\star\rVert\;\le\;\frac{\delta}{1-\rho}$$

of the classical decomposition's. *Proof sketch:* Banach's theorem for existence and rate, then $\lVert\tilde a^\star-a^\star\rVert\le\lVert\tilde S(\tilde a^\star)-\tilde S(a^\star)\rVert+\lVert\tilde S(a^\star)-S(a^\star)\rVert\le\rho\lVert\tilde a^\star-a^\star\rVert+\delta$. This is **T1 with T8–T9** of [[formal-proofs-plan]], scheduled for a machine-checked proof.

**Why this matters.** Convergence then rests on a **checkable property** of each expert, its Lipschitz constant, and not on how well it was trained. Accuracy enters only the distance of the limit, $\delta/(1-\rho)$, which is [[master-error-bound]]'s $\tau$ carried to the interface. **[AI Inference]:** of the Tier-89 findings, S5a (weakly shrunk columns diverging along the approach) is the one this contract rules out by construction.

**How the bound is established.** A Lipschitz penalty during training keeps the map near the bound. After training, the bound is **certified**: by interval or linear-relaxation bound propagation on the trained network (TorchLean implements such methods in Lean 4; [[dd-neural-prior-art-2026]] §1.6), or by the cruder product of layer norms. **[AI Inference], the honest risk:** certified bounds on million-parameter networks are often loose. The fallback is architectural: the expert is a classical linear scattering map of known contraction plus a small learned correction whose bound is certified. The build decides between them.

### 4.5 Beyond fixed-point sweeps

- **The coarse space** of Level 2 is the classical remedy for Schwarz's slow low modes. Plain Schwarz on the bracket took 616 iterations for lack of one ([[showcase-gallery]] §3).
- **Interface Newton–Krylov.** The expert returns Jacobian–vector products by automatic differentiation, so the host can solve the interface equation $a=\Pi\tilde S(a)+c$ with GMRES. That gives convergence in a number of Krylov steps set by the spectrum's clustering, not by $\rho$. The compiler already derives a direct Schur interface for the linear examples (`K_accelerator = direct-schur`, [[gap-worklist]] W348); this is its matrix-free form.
- **The warm start.** Each macro-step starts from the last step's converged waves, extrapolated.

### 4.6 Cross-points: the open part

Where three or more charts meet, at chart corners, side-by-side wave exchange is known to break the energy argument in discrete optimized Schwarz (Gander & Kwok, DD20 proceedings, 2013). Remedies exist: a non-local exchange at cross-points (Claeys & Parolin, *Numer. Math.* 151, 2022), or letting the coarse space own the corners. **Open, and named:** the workbench already declares cross-points on every graph (W162) and sees them on every generated layout.

---

## 5. Training for the iteration count, in brief

The full plan, data and costs are in [[chart-operator-training-and-cost]]. The loss:

$$\mathcal L=\underbrace{\lVert\hat u_\theta-\hat u\rVert^2}_{\text{state}}+\lambda_b\underbrace{\lVert b_\theta-b\rVert^2}_{\text{outgoing waves}}+\lambda_J\,\mathbb E_{v\sim\text{low bands}}\underbrace{\lVert D_a\tilde S\,v-D_aS\,v\rVert^2}_{\text{response to interface data (S5)}}+\lambda_{sg}\underbrace{\lVert E_{2\Delta t}-E_{\Delta t}\!\circ\!E_{\Delta t}\rVert^2}_{\text{step consistency (S2)}}+\lambda_L\,\underbrace{\operatorname{pen}(\operatorname{Lip})}_{\S4.4}.$$

The Jacobian term is derivative-informed training (DINO; [[dd-neural-prior-art-2026]] §1.5), aimed at the derivative that sets the Schwarz rate. It is **evaluated along whole approaches** to the settled state, not only at it (S5a).

---

## 6. The Expert Card: one format for every expert

Every expert, learned or classical, ships a manifest the host and the compiler read. It extends the capability records the compiler already reads (`atlas/capability.py`).

```yaml
expert: chart-diffusion-2d
version: 0.1.0
licence: <the owner's choice>          # S12
family: diffusion                       # governing-equation family
ports: [THERM]                          # the port algebra's types
chart:
  reference: unit-square
  map_methods: [conformal-quad, winslow, tfi]
  inputs: [J, JGinv]                    # + theta for vector physics
  envelope: {modulus: [0.5, 2.0], J_ratio_max: 50}
resolution: [32, 64, 128]               # S8: same-class references exist
time: {dt_range: [1.0e-3, 1.0e-1]}      # S2
sides:
  kinds: [wave, dirichlet, neumann]
  impedance_range: [0.1, 100.0]
outputs: [state, outgoing_wave, jvp]
certificates:                           # measured, dated, with their method
  lipschitz_fine_space: {value: 0.93, method: crown, date: ...}
  defect_vs_classical: {value: 2.1e-3, norm: L2-side, date: ...}
  slow_band_response: {psi_slow: ..., phi_slow: ..., date: ...}   # S5
  conservation: exact                   # conservative head
precision: float64-capable              # S10
```

The values are placeholders showing the format. **The compiler's new rules read the card**:
- a chart outside the envelope is refused, with the quantity named;
- a $\Delta t$ outside `dt_range` is refused (S2's refusal, made declarative);
- a certified Lipschitz bound below 1 on the fine space admits the seam's convergence claim, citing the theorem;
- a missing certificate admits uncertified, as today.

---

## 7. What the design does not promise

1. **Speed against the cheapest classical alternative (S7).** Nothing in the architecture makes it cheaper than a coarse classical solver in the same slot. That is measured per family, and the gate ([[chart-operator-training-and-cost]] §6) can fail.
2. **Three dimensions** (§2.5): charts are not solved there.
3. **Shocks and strong nonlinearity**: the conservative head keeps conservation exact, but the contraction contract of §4.4 is a statement about maps near a regime. A shock crossing a seam is outside it.
4. **Moving and changing geometry**: a moving chart adds $\partial_tF$ terms (arbitrary Lagrangian–Eulerian). A topology change, staging or contact, is a new decomposition ([[global-fields-and-topology-atlas-0.1]]).
5. **Cross-points** (§4.6).
6. **Tight Lipschitz certificates at scale** (§4.4).

---

## 8. The owner's candidate list, evaluated

| the owner's idea | verdict | where |
|---|---|---|
| diffeomorphic neural operators | **adopted, and published before** (DIMON, DNO). What is added here: the diffeomorphism is **per chart** of a decomposition, computed classically with an injectivity guarantee, and the operator receives the metric explicitly | §2, [[dd-neural-prior-art-2026]] §1.2 |
| a generalised Jacobian passed alongside the geometry, boundary and initial conditions | **adopted** as $(J\,G^{-1},\ J,\ \theta)$. For a scalar problem it is *all* the geometry the operator can see, which is why rigid motions come for free | §2.3–§2.4 |
| transfinite interpolation, isoparametric maps | **adopted as the fast path** for near-rectangular charts, checked by $\min J>0$; and TFI of the **data** is the lifting | §2.2, §3.2 |
| harmonic parametrisation, conformal mapping | **harmonic (Winslow) adopted as the guaranteed map; the quadrilateral conformal map is the default**, and the workbench already computes it. Conformal maps' crowding is removed by the decomposition keeping every modulus near one | §2.2 |
| Geo-FNO | **not adopted as the map**: a learned map can fold silently. Kept as a baseline to compare against | §2.2 |
| optimized Schwarz methods for fast convergence | **adopted, as wave-variable exchange through the typed ports**, with a coarse space and a contraction contract that makes convergence checkable | §4 |
| boundary-to-domain mapping | **adopted as the lifting**, with exact Dirichlet enforcement | §3.2 |

---

## 9. Open decisions for the architecture chat

1. Read SNI, NEST, L-DDM, DIMON and DNO in full, and confirm or withdraw each "left to claim" row of [[dd-neural-prior-art-2026]] §3.
2. Check §2.2's conformal claim on the workbench's own pieces: compute $J$, the metric's anisotropy and $\min J$ on `s-channel` and `bend-3`.
3. Choose the first family and its reference solver (the recommendation is 2-D diffusion, [[chart-operator-training-and-cost]] §2).
4. Choose how §4.4's bound is certified: penalty plus certification, or the linear-plus-correction architecture.
5. **Decided (O8, the owner, 2026-09-30): the proposal document is an HTML page in arXiv's HTML style**, not a PDF. That means:
   - a single column, with a title block and an abstract;
   - a table of contents in a side rail;
   - numbered sections, equations, figures and tables;
   - typeset mathematics;
   - a numbered reference list.
   
   The website links to it ([[website-outline]] section 5), and it goes public with the site (O9).
6. **This page is version 0, not the settled design.** The owner's instruction for the architecture chat is to **brainstorm with the owner first, and confirm the design before documenting it** ([[proposal-chat-prompts]] §2). Every choice above is an input to that conversation, and the confirmed design replaces this version.

---

## See Also

- [[chart-operator-training-and-cost]] — data, training, stages, costs, fine-tuning against training from scratch
- [[dd-neural-prior-art-2026]] — what is published and what is left
- [[formal-proofs-plan]] — the theorems §4.4 rests on
- [[outcome-c5-requirements-for-dd-native-experts]] — S1–S12
- [[atlas-and-standard-dd-theory]] — the optimized-Schwarz hierarchy this design climbs
- [[probed-dtn-coupling]] — the Steklov–Poincaré operator that §4.2's Cayley transform is built from
- [[neural-operators]] — the vault's general page on the operator families
