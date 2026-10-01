# The chart operator — a learned expert built for domain decomposition (version 1)

**Type:** Concept page — **architecture, version 1, as confirmed by the owner** (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** rewritten 2026-10-01 after the owner confirmed the whole design ("Yes to the design and defaults", phase 2 of [[proposal-chat-prompts]] §2). It replaces version 0 of 2026-09-30, whose changes are listed in §1. **A design, not a build**: nothing is implemented or trained. The numbers in §5 are measured by named scripts with classical solvers or untrained networks; every estimate is labelled; anything beyond the record is marked **[AI Inference]**. The theorems of §3.6 are proved on paper here and in the proposal document; **none is machine-checked**.
**The document:** `proposal/architecture/index.html`, *Chart Operators: Learned Superelements for Domain Decomposition on Curved Geometry* (Naunidh Singh), the arXiv-style HTML page decided as O8. **Private until the website launches (O9).** This page is its wiki counterpart and the reference for the other chats.
**Hub:** [[00-proposal-workstreams]] · **Decision record:** [[chart-operator-design-decisions]] · **Measured costs:** [[coupling-cost-and-complexity]] · **Companions:** [[chart-operator-training-and-cost]] · [[dd-neural-prior-art-2026]]
**Answers to:** [[outcome-c5-requirements-for-dd-native-experts]] (S1–S12) · **Built on:** [[port-algebra-atlas-0.1]] · [[atlas-and-standard-dd-theory]] · [[probed-dtn-coupling]] · [[master-error-bound]]

---

## 0. The idea in one paragraph

A manifold is described by an atlas of charts; the chart operator takes the metaphor literally. **Topology** goes to the decomposition, which cuts the domain into four-sided pieces. **Geometry** goes to a classical chart of each piece onto the unit square, whose injectivity is certified exactly; by a change of variables the curved piece becomes the square carrying an anisotropic coefficient. **Physics** goes to the learned operator, which solves one problem only — diffusion on the unit square with values prescribed on its four sides — and answers with **constraint modes**: one field per interface mode, with that mode as its exact boundary values. **Agreement** goes to the host, which forms each piece's port matrix as the **energy Gram** of those modes and solves one interface system, with no iteration. The Gram is symmetric positive semidefinite **whatever the network returns**, its error is the energy of the field errors (quadratic and one-signed), and the coupled answer is the energy-best one the modes allow.

$$\boxed{\ \text{topology}\to\text{the decomposition}\qquad\text{geometry}\to\text{the chart}\qquad\text{physics}\to\text{the operator}\qquad\text{agreement}\to\text{the host's energy Gram}\ }$$

**Intuition for the last box.** Version 0 had the pieces negotiate by repeated exchange (Schwarz iteration), which is like assembling a truss by letting every member push on its neighbours until nothing moves. Structural engineers instead ask each member for its **stiffness at its joints**, assemble, and solve once. The constraint modes are each piece's stiffness at its joints, delivered in a form the host can check.

---

## 1. What changed from version 0, and why

| part | version 0 (2026-09-30) | version 1 (2026-10-01) | why |
|---|---|---|---|
| coupling | wave-variable (Robin) iteration with a coarse space and a contraction contract | **three tiers** (§3.7): tier 0 superelement with no iteration, tier 1 tangent (Newton), tier 2 the version-0 wave iteration for black-box experts | measured rounds on 8 × 8 pieces: traditional Schwarz 1,864, version 0's wave exchange 1,124, with GMRES 83, two-level 70; the superelement needs none ([[coupling-cost-and-complexity]]) |
| output | the state and the outgoing waves | **constraint modes and a particular field**; the host forms the port matrix as an energy Gram | measured: a network predicting the port matrix directly turns indefinite; the Gram cannot (§5, Theorem 1 in §3.6) |
| what the operator sees | $J\,G^{-1}$ (3 channels) and $J$ | $\bigl(\log J-\overline{\log J},\ \operatorname{Re}\mu,\ \operatorname{Im}\mu,\ \log\hat\kappa,\ \hat f\bigr)$ and the scalars $(\mathrm{Fo},\dots)$ | $\det(J\,G^{-1})=1$ in 2-D, so version 0 double-counted; $\mu$ is bounded by one |
| chart | conformal by default, Winslow as the guaranteed map, transfinite as a fast path | **a chart rule by corner type**, switched by the envelope (§3.2) | the corner bound (Proposition 2) and the measured trade-off: conformal $J$ ratio 291 against Winslow 3.2 and transfinite 4.1 on the s-channel's end piece |
| injectivity | Rado–Kneser–Choquet for Winslow; $\min J>0$ for transfinite | **an exact discrete certificate** for any chart (Proposition 3) | positivity of the bilinear Jacobian at cell corners plus a simple boundary polygon is a proof, in $O(n^2)$ |
| boundary data | Coons lifting of Dirichlet data plus a bubble correction; Robin data met through the loss | **analytic principal part of the plain square plus a bubble-multiplied learned correction**; every boundary condition is the host's linear relation on ports | one canonical question for every expert; the trace is exact by construction |
| backbone | CNO-style U-Net, "unlike FNO, no periodicity" | U-shaped convolutional operator with **explicit even/odd extension**, FiLM; sine–cosine spectral operator as the ablation; **1 M parameters** with a {0.3, 1, 3} M sweep | CNO's filters act on the periodic extension (withdrawn claim); forward cost measured (§5) |
| certification | a certified Lipschitz bound for every expert | **needed only at tier 2**; tier 0 is certified by construction | the Gram's guarantee holds for every output |
| training objective | state, waves, a Jacobian (response) term, step consistency, a Lipschitz penalty | **energy-norm error of the modes and particular field**; a label-free objective with the same gradients (Corollary 11); step consistency only for networks that approximate the exact flow map | at tier 0 the energy norm is what controls the coupled error (Theorems 1–2) |
| geometry source | (not specified) | **the drawn smooth outline**; the classical reference is solved in chart coordinates on the same grid | a staircase chart would give spurious corners; gives S8's same-class reference and gate E3's baseline |

Decisions (a)–(c) and 1–19 with their dates are in [[chart-operator-design-decisions]] §2. **Two refinements made while writing the document** (2026-10-01, reported to the owner):
1. **Step doubling (decision 14).** The tier-0 networks approximate the backward-Euler step, and $(I+2\Delta t\,L)^{-1}\ne(I+\Delta t\,L)^{-2}$, so a step-doubling penalty would push them away from their target. It applies only to networks that approximate the exact evolution over a step (Stage 2's flow windows). The tier-0 networks cover the declared Fourier-number range with exact examples.
2. **Interface data (decision 11) and the objective (decision 15)** follow decision 6: tier-0 interface data are the cosine trace coefficients, and the port matrix is derived, not supervised.

---

## 2. What it has to do

| the owner's goal (2026-09-30) | measured requirement ([[outcome-c5-requirements-for-dd-native-experts]]) | where version 1 meets it |
|---|---|---|
| **almost any geometry** | S1 (only a uniform $128^2$ grid; 61% of the car's unknowns unreachable), S4 (no bounded domain of dependence) | §3.1–§3.2: every piece is charted onto the square; a piece's response is computed inside the piece |
| **few iterations inside a decomposition** | S3 (interface data as input), S5, S5a, S6 (response and stability on the slow band) | §3.3 and §3.6: tier 0 has **no iteration**, takes boundary data by construction, and cannot make a piece singular or softer; tier 2 keeps the contraction contract |
| **one standard, modular format** | S2, S8, S10, S12 | §3.5 and §4, the Expert Card |
| (implied) **worth its slot** | S7, S7a, S9, S11 | §5 and §6: measured costs, and a gate that can fail |

---

## 3. The design

### 3.1 Topology

**Intuition.** Holes, forks and loops are the decomposition's business; every piece the network sees is a curved square.

**Rigorously.** Every piece is a quadrilateral (a Jordan domain with four marked corners). Within a channel-like block the layout's along coordinate $u$ and across coordinate $v$ are harmonic with Dirichlet data on two opposite sides and zero normal derivative on the other two, and $\Phi=Mu+iv$ is the conformal map onto $[0,M]\times[0,1]$, $M$ the block's conformal modulus, $M=1/\!\int|\nabla u|^2$.

> **Proposition 1 (equal-value cuts).** The piece $\{k/n\le u\le(k+1)/n,\ r/m\le v\le(r+1)/m\}$ has modulus $M'=Mm/n$; its chart has constant Beltrami coefficient $\mu=(M'-1)/(M'+1)$, so $J\,G^{-1}=\operatorname{diag}(1/M',M')$; all four corners of an interior piece are right angles.

The proof (uniqueness of the mixed problem gives $\operatorname{Re}\Phi=Mu$, $\operatorname{Im}\Phi=v$) is in the document. With $n=\operatorname{round}(Mm)$ every interior piece has $\mathsf A\approx\hat\kappa I$. **Measured** on the s-channel ($M=8.872$, 9 pieces): predicted modulus 0.986, own moduli 0.977–0.988. The block layout for forks and rings is **not built**.

### 3.2 Geometry: the pull-back, the corner bound, the certificate

**Geometry is anisotropy.** With $\hat u=u\circ F$,
$$\int_{\Omega_i}\bigl(\kappa\nabla u\cdot\nabla v+\sigma uv\bigr)dx=\int_{\hat\Omega}\bigl(\mathsf A\,\hat\nabla\hat u\cdot\hat\nabla\hat v+J\hat\sigma\,\hat u\hat v\bigr)d\hat x,\qquad \mathsf A=J\,DF^{-1}\hat\kappa\,DF^{-\top},$$
and with $\mu=F_{\bar z}/F_z$,
$$J\,G^{-1}=\frac{1}{1-|\mu|^2}\begin{pmatrix}|1-\mu|^2&-2\operatorname{Im}\mu\\-2\operatorname{Im}\mu&|1+\mu|^2\end{pmatrix},\qquad \det(J\,G^{-1})=1,$$
so for scalar $\kappa$ the coefficient is $\mathsf A=\hat\kappa\,J\,G^{-1}$, a function of $(\hat\kappa,\mu)$ alone; $J$ enters only the capacity and the source. The inputs $\bigl(\log J-\overline{\log J},\operatorname{Re}\mu,\operatorname{Im}\mu,\log\hat\kappa,\hat f\bigr)$ are exactly invariant to translating, rotating and uniformly scaling the piece (size enters through $\mathrm{Fo}$ with $L^2$ the area).

> **Proposition 2 (distortion at a corner).** If $F$ is $C^1$ and nondegenerate up to a corner whose image has interior angle $\alpha$, then $\alpha<\pi$ and $K\ge\max(\tan\tfrac\alpha2,\cot\tfrac\alpha2)$ there, with $K=(1+|\mu|)/(1-|\mu|)$.

*Proof sketch:* $K+1/K=\operatorname{tr}(L^\top L)/\sqrt{\det L^\top L}=(a^2+b^2)/(ab\sin\alpha)\ge2/\sin\alpha$ for $L=DF$ sending the corner's edges to lengths $a,b$ at angle $\alpha$. A conformal chart escapes only by being singular, $J\sim r^{2-\pi/\alpha}$. For the s-channel's $68^\circ$ and $148^\circ$ corners the bound is $K\ge1.48$ and $K\ge3.49$; the transfinite chart, smooth up to the corners, reaches 4.53.

**The chart rule:** conformal (the equal-value coordinates) when all corners are right angles and at re-entrant corners, where it turns the singular solution $r^{\pi/\alpha}$ into the polynomial $2\xi\eta$; Winslow with arc-length data at other convex corners; imported body-fitted grids as given; transfinite interpolation for geometry that moves every step, if it passes the certificate. The switch is made by the envelope, not by an angle threshold.

> **Proposition 3 (discrete certificate).** A bilinearly interpolated chart whose Jacobian is positive at the four corners of every cell, and whose boundary nodes form a simple positively oriented polygon, is a homeomorphism of the closed square onto the polygon's closed interior.

*Why:* the bilinear Jacobian is affine on a cell ($\mathbf d\times\mathbf d=0$); the four cell angles at an interior node are in $(0,\pi)$ and close up, so they sum to exactly $2\pi$; a local homeomorphism whose boundary winds once is a bijection (degree argument). Cost $O(n^2)$.

**The envelope** declared on the Expert Card: $\sup|\mu|\le k_\mu$, $\max\log J-\min\log J\le\ell_J$, $\lVert\hat\nabla\log J\rVert_\infty+\lVert\hat\nabla\mu\rVert_\infty\le s$, plus ranges of $\hat\kappa$ and $\mathrm{Fo}$. The host refuses a chart outside it, naming the quantity; the layout can cut again.

**Learned charts are not used** — not because they cannot be certified (Proposition 3 would certify one) but because the pull-back is exact for any valid chart, so nothing about the map needs learning.

### 3.3 The output: constraint modes

**One question for every expert:** given values on all four sides of the square, what is inside, and what flux leaves each side? Walls, Robin conditions and interfaces are the host's linear relations on port coordinates and fluxes.

**Rigorously.** On the reference grid with cell values $u\in\mathbb R^{n^2}$ and boundary-face values $\lambda\in\mathbb R^{4n}$ (face-hybridised cell-centred finite volumes), the piece's energy matrix is
$$M=\begin{pmatrix}A_I&-B\\-B^\top&D\end{pmatrix}\succeq0,$$
and with the cosine port basis $Q\in\mathbb R^{4n\times4m}$ the exact objects are
$$H=A_I^{-1}BQ,\qquad u_p=A_I^{-1}f,\qquad V=\begin{pmatrix}H\\Q\end{pmatrix},\qquad \Lambda=V^\top MV=Q^\top(D-B^\top A_I^{-1}B)Q .$$
The network returns
$$\hat H_k=H_k^{(0)}+\beta\odot N_{\theta,k}(z),\qquad \hat u_p=\beta\odot N_{\theta,p}(z),\qquad \beta=16\,\xi(1-\xi)\,\eta(1-\eta),$$
with $H^{(0)}_k$ the exact constraint mode of the plain square at the given $\mathrm{Fo}$, computed once per resolution by a fast sine transform. One forward pass returns all $4m+1=65$ fields at $m=16$. The trace is the separate vector $\lambda=Qe_k$, so it is exact whatever the network returns. D4 symmetry by augmentation at Stage 0 (a $90^\circ$ rotation of the reference coordinates sends $\mu\to-\mu$).

**[AI Inference], registered as comparison C3:** the principal part makes learning easier because equal-modulus cuts make the correction small.

**A caveat found while writing (the document's Remark 4):** cosine modes of adjacent sides jump at a corner; in the continuum such a trace has infinite energy, in the face-based discretisation the energy is finite but grows like $\log n$. Comparison C6 registers a conforming basis (sines plus corner functions).

### 3.4 The backbone

A U-shaped convolutional operator in the manner of CNO: four levels, two $3\times3$ convolutions per level, GELU, average-pool down, bilinear up, skip connections, FiLM on the scalars; **even extension for inputs, odd for outputs**; registered ablation: a sine–cosine spectral operator. About $10^6$ parameters (between the measured 0.49 M and 1.95 M configurations), sweep $\{0.3,1,3\}\times10^6$. Reference grid $n=64$, trained on 32, 64, 128. Float32 training, float64-capable inference (S10). Tier 0 needs no certified Lipschitz constant of the network.

### 3.5 Ports, conservation, time

- **Interface data:** at tier 0, 16 cosine coefficients per side; the host transfers them between two charts' parametrisations of a shared side by an $L^2$ projection in physical arc length (a mortar), and each expert carries about 20 modes per side internally (**an estimate**, to check at Stage 0). Tier 2 exchanges Robin/wave coordinates computed from the same output.
- **Conservation:** a face-flux output format, $J(\hat u^{\text{new}}-\hat u^{\text{old}})/\Delta t=-\hat\nabla_h\cdot\hat{\mathbf F}+J\hat s$, required for conservation-law families from Stage 1, optional for diffusion.
- **Time:** the step enters as the piece's Fourier number; the card declares its range and the host refuses a call outside it (S2 as a declared refusal). At fixed step and geometry the modes are formed once; each later step costs one forward pass (the new particular field) and a back-substitution.

### 3.6 The guarantee (tier 0)

The host computes $\tilde\Lambda_i=\tilde V_i^\top M_i\tilde V_i$, $\tilde b_i=\hat H_i^\top f_i-\tilde V_i^\top M_i(\hat u_{p,i},0)^\top$, assembles $\tilde{\mathsf S}=\sum_iR_i^\top\tilde\Lambda_iR_i$, solves $\tilde{\mathsf S}c=\tilde\chi$ and reconstructs $\tilde u_i=\hat u_{p,i}+\hat H_iR_ic$ — no further network call.

> **Theorem 1 (the document's Theorem 5).** For any network output, with $E_i=\hat H_i-H_i$: (i) $\tilde\Lambda_i\succeq0$; (ii) $\tilde\Lambda_i-\Lambda_i=E_i^\top A_{I,i}E_i\succeq0$, so $\lVert\tilde\Lambda_i-\Lambda_i\rVert\le\lVert A_{I,i}\rVert\lVert E_i\rVert^2$ and $\operatorname{tr}\tilde\Lambda_i-\operatorname{tr}\Lambda_i=\sum_k\lVert E_{i,k}\rVert^2_{A_{I,i}}$; (iii) $\tilde{\mathsf S}\succeq\mathsf S$, so if the exact port-reduced system is positive definite the learned one is too, with $\lambda_{\min}(\tilde{\mathsf S})\ge\lambda_{\min}(\mathsf S)$.

*Proof of (ii):* the cross terms are $E_i^\top(A_{I,i}H_i-B_iQ_i)=0$ by the definition of $H_i$ — the discrete Dirichlet principle.

> **Theorem 2 (the document's Theorem 6).** If shared sides are parametrised identically and boundary data lie in the port span, $\lVert u-\tilde u\rVert_M=\min_{w\in\tilde W}\lVert u-w\rVert_M$ (Céa with constant one), and $\lVert u-\tilde u\rVert_M\le\lVert u-u_m\rVert_M+\bigl(\sum_i\lVert e_{p,i}+E_iR_ic^\star\rVert^2_{A_{I,i}}\bigr)^{1/2}$.

> **Proposition (the document's Proposition 7; draft T24).** $\lVert\tilde c-c\rVert\le\bigl(\lVert\mathsf S-\tilde{\mathsf S}\rVert\lVert c\rVert+\lVert\chi-\tilde\chi\rVert\bigr)/\lambda_{\min}(\tilde{\mathsf S})$, with $\lambda_{\min}(\tilde{\mathsf S})\ge\lambda_{\min}(\mathsf S)$ by Theorem 1.

> **Corollary (the document's Corollary 11).** $\mathcal L_{\text{en}}(\theta)=\operatorname{tr}\tilde\Lambda(\theta)+\hat u_p^\top A_I\hat u_p-2f^\top\hat u_p$ differs from the supervised energy-norm loss $\sum_k\lVert\hat H_k-H_k\rVert^2_{A_I}+\lVert\hat u_p-u_p\rVert^2_{A_I}$ by a constant independent of $\theta$: **the same gradients, with no solved examples.**

**Not covered:** the mortar (non-matching) case is nonconforming, and Theorem 2 does not cover it.

### 3.7 The coupling tiers

| tier | when | how | rounds | guarantee |
|---|---|---|---|---|
| **0, superelement** | linear physics; constraint-mode output | energy Gram; one direct interface solve in 2-D, preconditioned CG in 3-D | **none** | §3.6, for every network output |
| **1, tangent** | nonlinear or implicit physics | Newton on the interface equation with tangent constraint modes, assembled by the same Gram | a few Newton steps | local; the tangent Gram is PSD when the problem derives from a convex energy **[AI Inference]** |
| **2, black box** | the expert exposes only boundary responses (a donor, a classical code) | Robin/wave exchange, a coarse space from 16 pieces, GMRES with JVPs, Anderson without | tens; nearly flat in the number of pieces with a coarse space | conditional: a certified contraction factor (T1 with T8–T9) |

A tier-0 expert also serves at tier 2: $\mathcal S=(I-Z\tilde\Lambda)(I+Z\tilde\Lambda)^{-1}$ is non-expansive because $\tilde\Lambda\succeq0$. Ordered sweeps along channels: tier 2 only, never where symmetry averaging is used. Transient: reuse port matrices, warm-start, energy-safe early termination at tier 2 (Wei et al., arXiv 2603.16424).

**The SNI counterexample.** SNI's Theorem 1 (Huang et al., ICLR 2026) concludes convergence from "the classical map contracts and the learned solver is within $c$". One subdomain, $F(u)=(1-\tau)u$, learned solve $-c\tanh(ku)$, $\tau=\tfrac12$, $c=1$, $k=10$: both hypotheses hold and the iterates settle on the two-cycle $\pm0.332471$ (`scripts/arch_sni_counterexample.py`). The missing hypothesis is $\operatorname{Lip}(\tilde F)<1$, T1's.

---

## 4. The Expert Card

The card every expert ships, as specified in the document's Appendix A: identity and licence (S12); family and port type (`THERM`); chart methods, certificate, inputs and **envelope**; `time: {scheme: implicit-euler, fourier_range}` (S2); resolution (S8); `output: {format: constraint-modes, port_basis: cosine, port_modes_per_side: 16, internal_modes_per_side: 20, principal_part: plain-square, particular_field: true}`; `tiers: [0, 2]`; certificates (Gram identity residual for SE1, energy error by $|\mu|$ bin for E1, the contraction factor for tier 2 only); precision (S10). **Host rules:** refuse a chart outside the envelope or a Fourier number outside the range, naming the quantity; admit tier 0 for constraint-mode output on linear physics with no further certificate; admit a tier-2 convergence claim only with a certified contraction below one; report a missing certificate, never assume it.

---

## 5. Measured (every number from a named script)

| quantity | value | script |
|---|---|---|
| rounds to $10^{-8}$, steady, 8 × 8 pieces of $32^2$ | traditional RAS 1,864; Robin (OSM) 1,124; OSM + Anderson 152; OSM + GMRES 83; two-level 70; superelement 0 | `arch_coupling_cost.py` |
| wall time, 8 × 8, as a fraction of RAS | OSM + GMRES 0.070; two-level 0.16; Port-8 0.12; **monolithic LU 0.036** | same |
| port truncation, steady | $m=4$: 1.4–5.9%; $m=8$: 0.22–1.05%; $m=16$: 0.017–0.14% | same |
| Gram against a direct matrix head, $\varepsilon=0.01$ | solution error 0.27% against 9.2% (steady); 0.12% against 1.9% ($\mathrm{Fo}=0.1$); direct pieces indefinite at every $\varepsilon$ when steady; identity to $2\times10^{-15}$ | `arch_superelement_gram.py` |
| forward cost at $n=64$, per piece, alone / batch of 16 | 0.49 M: 6.0 / 3.8 ms; 1.95 M: 16.9 / 7.7 ms; 6.2 M: 30 / 25 ms | `arch_backbone_timing.py` |
| classical recomputation of one port matrix, $n=64$, 16 modes | about 21 ms factorisation + 47 ms solves = 68 ms | `arch_coupling_cost.py` |
| charts of the s-channel's end piece | $J_{\max}/J_{\min}$: conformal 291, Winslow 3.17, transfinite 4.08 | `arch_chart_maps.py` |

**The S7 reading.** One forward pass is 9–20 times a classical local solve with a stored factorisation, but a tier-0 solve needs one pass per piece against $70\times64$ calls for the two-level iteration. On a fixed 2-D geometry stored classical modes, and a monolithic direct solve, beat any network. **The learned superelement pays only where classical work cannot be reused or does not scale**: changing geometry or material between solves (4–9 times cheaper than recomputing the port matrix), expensive local physics, three dimensions. **Estimate** (labelled in the document): 64 pieces of $32^2$ with the 1.95 M network, about 0.2–0.4 s per tier-0 solve on the container's CPU, against 0.8 s for the monolithic LU.

---

## 6. What the design does not promise

1. Speed against classical solvers in 2-D (S7); gate E5 can fail.
2. Learnability of the general input class at a useful forward cost (comparisons C1, C5).
3. Three dimensions: no injectivity theorem, hexahedral decomposition open, PCG on the 3-D port system untested; the fallback is a point- or mesh-based expert with the same port data.
4. Nonlinear and hyperbolic families: tier 1 is local; shocks crossing a seam are outside scope.
5. The mortar case (non-matching parametrisations): nonconforming, unanalysed.
6. Corners of the cosine port modes (§3.3); cross-points of vertex-based tier-2 schemes (the face-based formulation has none).
7. The block layout for arbitrary topology is not built.
8. Measurements from one container, and S1–S12 from one network.

---

## 7. For the proofs chat

**The owner relays this; the architecture chat cannot reach the proofs chat.** What the confirmed design changes in [[formal-proofs-plan]]:

1. **T1, T8, T9, T9′ keep their statements** and now apply to **tier 2 only**. T8 gains a use: a tier-0 expert's Gram is PSD, so its Cayley transform is non-expansive for every impedance.
2. **T22's** incomplete-solve term $\rho^k$ applies only at tier 2; a tier-0 step has no iteration, so its per-step defect is the Galerkin error of Theorem 2.
3. **New, ours, Tier A (finite-dimensional linear algebra):**
   - **T24 — perturbation of the interface solve** (§3.6, the document's Proposition 7).
   - **T25 — the Gram superelement** (the document's Theorems 5 and 6): PSD for every output; $\tilde\Lambda-\Lambda=E^\top A_IE$; $\tilde{\mathsf S}\succeq\mathsf S$ and $\lambda_{\min}$ monotone; Céa with constant one over the learned trial space; the bound by truncation plus field errors. Hypotheses to state precisely: face-hybridised energy matrices $M_i\succeq0$, $A_{I,i}\succ0$, $Q_i$ with orthonormal columns, conforming shared sides, boundary data in the port span, $\lVert\cdot\rVert_M$ a norm on $U_0$.
   - **T26 — the label-free objective** (the document's Corollary 11): a two-line consequence of T25(ii) and completing the square.
   - **Propositions 1–3** of §3.1–§3.2 (equal-value moduli, the corner bound, the discrete certificate) are classical analysis; the corner bound and the certificate are short and may be worth formalising; the modulus statement needs conformal-map theory and should be **cited**.
4. **A remark for T1:** the SNI counterexample shows the Lipschitz hypothesis on the learned map cannot be weakened to accuracy plus classical contraction.
5. **The headline sentence** of [[formal-proofs-plan]] §3 is about tier 2. For tier 0 the honest sentence is: *whatever the network returns, the coupled system is solvable and its answer is the best the network's modes can represent in the energy norm.*

Status in the document: every proof is given in full, and the document says plainly that none is machine-checked.

---

## See Also

- [[chart-operator-design-decisions]] — the owner's decisions, their dates and evidence
- [[coupling-cost-and-complexity]] — the measured cost of every coupling strategy
- [[chart-operator-training-and-cost]] — data, objective, stages, gates and costs
- [[dd-neural-prior-art-2026]] — the prior art read in full, and claims N1–N7
- [[formal-proofs-plan]] — T1, T8, T9, T22, and the drafts T24–T26 of §7
- [[outcome-c5-requirements-for-dd-native-experts]] — S1–S12
- [[probed-dtn-coupling]] — the Steklov–Poincaré operator that tier 0 assembles from learned modes
- [[atlas-and-standard-dd-theory]] — the classical theory the tiers rest on
- [[neural-operators]] — the vault's page on operator families
