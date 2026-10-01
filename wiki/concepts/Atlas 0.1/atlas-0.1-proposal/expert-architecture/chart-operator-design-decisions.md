# The chart operator's design conversation — the decisions as they are made

**Type:** Concept page — **running decision record** (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** opened 2026-10-01 by the architecture chat, during the design conversation with the owner (phase 2 of [[proposal-chat-prompts]] §2). **It records what the owner has decided, why, and on what evidence; it is not the design document.** [[chart-operator-architecture]] stays at version 0 until the owner confirms the whole design, and is then rewritten as version 1 from this page. Every number here is measured by a named probe; anything beyond the record is marked **[AI Inference]**.
**Hub:** [[00-proposal-workstreams]] · **Feeds:** [[chart-operator-architecture]] · [[chart-operator-training-and-cost]] · [[dd-neural-prior-art-2026]] · **Measured inputs:** [[coupling-cost-and-complexity]]

---

## 0. Where the conversation stands

| | items |
|---|---|
| **decided** | (a) scope; (b) the chart rule; (c) the reference domain and topology; 1 the operator's input; 2 the cut rule; 3 the chart certificate and envelope; 4 the geometry source; 5 the coupling stack; 6 the expert's output format; 7 the tier-2 accelerator; 8 ordered sweeps; 9 the transient policy; 10 the backbone and its size |
| **under discussion** | 17 the training route; 18 the first family and its gate; 19 the name |
| **not yet answered** | 11 interface data and lifting; 12 the conservation head; 13 certification; 14 time; 15 the training objective; 16 the Expert Card; 20 the proposal's author line |

---

## 1. What the reading and the probes changed before any decision

1. **The prior art was read in full** (arXiv reachable from 2026-09-30): the fifteen papers [[dd-neural-prior-art-2026]] listed, and seven more. The record, paper by paper, is `out/arch/prior-art-reading.md`. Of the "left to claim" rows, **N1, N4, N5 and N6 stand; N2 and N3 narrow**:
   - **N2 narrows.** Convergence whatever the training is published by another route: wrap the network in a classical residual loop (HINTS; Mao & Fan, arXiv 2509.03622). Wave-coordinate coupling with a checkable firm non-expansiveness condition is published for classical lumped port-Hamiltonian systems (Wei et al., arXiv 2603.16424). What is left is a certified bound on the **learned map itself**, giving convergence at network speed with no classical residual in the loop.
   - **N3 narrows.** Training aimed at the iteration count is published for learned **interface conditions** (Taghibakhshi et al., NeurIPS 2022: a loss estimating the spectral radius) and learned **preconditioners** (Mao & Fan). Not found: a learned **local solver** trained on its response to interface data.
2. **SNI's convergence theorem needs one more hypothesis.** Its Theorem 1 (ICLR 2026) concludes convergence from "the classical iteration contracts and the learned solver is within $c$". The step does not follow: a one-subdomain counterexample satisfies both hypotheses and cycles at $\pm0.3325$ for ever (`scripts/arch_sni_counterexample.py`). The missing hypothesis is a bound on the learned map, $\operatorname{Lip}(\tilde F)<1$, which is T1 of [[formal-proofs-plan]].
3. **Published, so not to be claimed:** Robin transmission with learned subdomain solvers (SNAP-DDM, ICML 2024; DD-DeepONet 2025; Mao & Fan 2025); coarse spaces built from learned solvers by probing their Robin-to-Robin maps (Mao & Fan); exact Dirichlet lifting $u=G+\phi\,\mathcal N$ (Mosaic Flows §4.4); iteration-aware training in general; the harmonic map as the reference map (DNO); learned local Dirichlet-to-Neumann maps assembled in a substructured Newton solve (Boutilier, Brenner & Miguez, arXiv 2405.04433) and DtN maps learned linear in the boundary data across geometries (PPDNO, arXiv 2606.25952); networks returning local compressed stiffness matrices (Kroepfl, Maier & Peterseim, *Adv. Contin. Discrete Models* 2022); and the classical component version, static condensation with port reduction (Huynh, Knezevic & Patera, 2013; Eftang & Patera, 2013).
4. **Version 0 corrections.** The Convolutional Neural Operator's anti-aliasing filters act on the *periodic extension* of the signal, so "CNO, unlike FNO, does not assume periodicity" is withdrawn; both need an explicit extension on the square. And $J\,G^{-1}$ has determinant one in 2-D, so version 0's "3 channels plus $J$" double-counts (§3.4).

---

## 2. The decisions

| # | decision | the owner, date | evidence |
|---|---|---|---|
| (a) | **Two dimensions first, with a named path to three**: hexahedral charts checked by $\min\det DF>0$, charts taken from existing body-fitted grids, and a point- or mesh-based fallback expert behind the same port contract | A2, 2026-09-30 | harmonic maps can fold in 3-D (Laugesen 1996); hex meshing is open |
| (b) | **A chart rule, not one map** (§3.2) | B2, 2026-09-30 | `out/arch/chart_maps.txt` |
| (c) | **One reference domain, the unit square; the decomposition owns topology** | C1, 2026-09-30 | the topology benchmark (arXiv 2609.05860): "a diffeomorphism cannot create a tunnel or enclosed cavity" |
| 1 | **The operator accepts general geometry**: the fields $(\log J,\ \operatorname{Re}\mu,\ \operatorname{Im}\mu,\ \log\kappa\ [,\ \theta])$ on the reference grid, and scalars (Fourier number, parameters) | as recommended, 2026-10-01 | §3.4 |
| 2 | **Equal-value cuts to modulus about one**; the layout chooses the along count from the domain's modulus | as recommended, 2026-10-01 | §3.5 |
| 3 | **The chart certificate and the envelope** | as recommended, 2026-10-01 | §3.6 |
| 4 | **Chart the drawn smooth outline, not the cell staircase**; the classical reference is solved in chart coordinates | as recommended, 2026-10-01 | §3.7 |
| 5 | **A three-tier coupling stack** chosen per seam by the compiler | as recommended, 2026-10-01 | [[coupling-cost-and-complexity]] |
| 7 | **Tier 2's accelerator**: GMRES when the expert gives Jacobian–vector products, Anderson otherwise; a coarse space whenever there are 16 or more pieces | as recommended, 2026-10-01 | §3.9 |
| 8 | **Ordered (multiplicative) sweeps along channels are allowed in tier 2 only**; rule R5's additive ordering is kept wherever symmetry averaging is used | as recommended, 2026-10-01 | §3.10 |
| 9 | **The transient policy**: reuse port matrices at a fixed step, warm-start from the previous step, terminate early only where the coupling is energy-safe | as recommended, 2026-10-01 | §3.11 |
| 6 | **The output format F3**: every expert answers the all-Dirichlet problem on the unit square by returning one **constraint mode** per port mode (an exact analytic principal part plus a bubble-multiplied learned correction) and a particular field; the host forms the port matrix as their **energy Gram**. 16 cosine port modes per side (about 20 inside the expert); D4 symmetry by augmentation at Stage 0; the label-free energy objective registered as a Stage 0 comparison. The wave map (F1) stays in the standard for tier-2 donors | 6a–6e as recommended, 2026-10-01 | §3.12 |
| 10 | **The backbone**: a U-shaped convolutional operator with explicit odd/even boundary extension, anti-aliased resampling and FiLM conditioning; the sine-spectral operator as the registered ablation; **1 M parameters by default, with a {0.3 M, 1 M, 3 M} sweep** in the Stage 0 gate; $n=64$, trained on 32, 64 and 128 | 10a–10b as recommended, 2026-10-01 | §3.13 |

---

## 3. Each decision, explained

### 3.1 Scope (a)

**Intuition.** In the plane a harmonic map onto a convex target cannot fold, and every planar domain with holes can be cut into four-sided pieces. In space neither holds. So the proposal builds and proves in 2-D, and names how 3-D is reached without pretending the 2-D guarantees carry over.

**The 3-D path, rigorously.** Injectivity is **checked, not guaranteed by theorem**: a chart sampled on the reference grid is certified by positive Jacobian bounds per element and a boundary map that covers the cube's boundary once. For trilinear hexahedra positivity at the eight corners is not enough, and the bound must come from the Jacobian's Bernstein coefficients, as for curved finite elements (Johnen, Remacle & Geuzaine, *J. Comput. Phys.* 233, 2013). Pieces no chart fits get a point- or mesh-based expert (GINO, Transolver) exchanging the **same port data**.

### 3.2 The chart rule (b)

**All harmonic charts solve one equation; they differ in the boundary correspondence.** The layout's along and across coordinates are harmonic with Dirichlet data on two opposite sides and no flux through the other two; the sides slide, and the map is **conformal**. Winslow's map fixes arc-length data on all four sides. Transfinite interpolation blends the four side curves explicitly.

**The corner trade-off.** At a corner of interior angle $\alpha$ the square's right angle must become $\alpha$. With $\mu=F_{\bar z}/F_z$ the Beltrami coefficient and $K=(1+|\mu|)/(1-|\mu|)$ the dilatation, a chart that is $C^1$ and non-degenerate up to the corner has
$$K\ \ge\ \max\!\left(\tan\tfrac\alpha2,\ \cot\tfrac\alpha2\right)$$
there (from the polar decomposition of $DF$, the two edge directions being sent to directions an angle $\alpha$ apart). A conformal chart avoids it only by being singular: its forward Jacobian behaves like $J\sim r^{\,2-\pi/\alpha}$. **[AI Inference]** as a statement in this form; the ingredients are classical.

**Measured** (`scripts/arch_chart_maps.py`, `out/arch/chart_maps.txt`), on the s-channel's end piece with corners of 68 and 148 degrees:

| chart | $J_{\max}/J_{\min}$ | $\hat K_{11}$, $\hat K_{22}$ (p5–p95) | $\lvert\hat K_{12}\rvert$ p95 | folds |
|---|---|---|---|---|
| conformal | 291 | 1.02–1.07, 0.94–0.98 | 0.05 | 0 |
| Winslow, arc length | 3.2 | 0.89–1.52, 0.79–1.42 | 0.81 | 0 |
| transfinite | 4.1 | 0.81–1.89, 0.83–1.40 | 0.95 | 0 |

On a middle piece all three are mild ($J$ ratio 1.1–1.8, $|\hat K_{12}|\le0.15$).

**The rule.**

| situation | chart | why |
|---|---|---|
| every corner a right angle (every interior cut: the layout's cuts meet walls orthogonally) | conformal, the layout's own pair restricted to the piece | free, $\mu\approx0$, the metric is a constant |
| a re-entrant corner, $\alpha>180°$ | conformal | the physical solution is singular there ($u\sim r^{\pi/\alpha}$); the conformal pull-back makes $\hat u$ smooth and puts the singularity into an integrable $J$ |
| a convex corner that is not a right angle | Winslow with arc-length data | bounded $J$ at the price of the unavoidable shear |
| an imported body-fitted grid | as given | answers S1 directly |
| re-charting every step (moving geometry) | transfinite, if it passes the certificate | explicit, no solve |

The switch is made by the envelope (§3.6), not by an angle threshold: when the conformal chart's $\log J$ range exceeds the declared bound, Winslow is used.

### 3.3 The reference domain and topology (c)

Every piece is a topological quadrilateral. A ring becomes arcs; a fork becomes channels and a junction cut into quadrilaterals. That needs a **block layout** (channels and junctions) before the along/across cuts, which is the well-studied quad-layout problem (cross-field methods; the survey of Bommes et al., *Comput. Graph. Forum* 32, 2013) and **is not built**. Cross-points stay a named open part ([[formal-proofs-plan]] T20), though they do not arise in the face-hybridised coupling of [[coupling-cost-and-complexity]].

### 3.4 What the operator sees (decision 1)

**Geometry is anisotropy.** Pulling $-\nabla\!\cdot(\kappa\nabla u)=f$ back through any chart gives
$$-\hat\nabla\!\cdot\!\left(A\,\hat\nabla\hat u\right)=J\hat f,\qquad A=J\,DF^{-1}\,\kappa\,DF^{-\top}.$$
A square of anisotropic material and a curved piece of isotropic material are indistinguishable to the operator. This is classical: a conductivity and its push-forward by a boundary-fixing diffeomorphism have the same Dirichlet-to-Neumann map (Kohn & Vogelius, *Comm. Pure Appl. Math.* 37, 1984; Sylvester, *Comm. Pure Appl. Math.* 43, 1990), which is also the basis of transformation optics (Pendry, Schurig & Smith, *Science* 312, 2006). So the expert's job is defined once: **anisotropic, heterogeneous diffusion on the unit square**, with every piece of every shape, and every anisotropic material, entering through $A$.

**The metric's shape is the Beltrami coefficient.** With $z=\xi+i\eta$ and $\mu=F_{\bar z}/F_z$,
$$J\,G^{-1}=\frac{1}{1-|\mu|^2}\begin{pmatrix}|1-\mu|^2 & -2\operatorname{Im}\mu\\ -2\operatorname{Im}\mu & |1+\mu|^2\end{pmatrix},\qquad \det\!\left(J\,G^{-1}\right)=1 .$$
$|\mu|<1$ everywhere is exactly orientation-preserving with $J>0$. A conformal chart of modulus $M$ has the constant $\mu=(M-1)/(M+1)$, giving $\mathrm{diag}(1/M,M)$.

**The channels.** $\log J$ minus its mean (the piece's size enters as the dimensionless Fourier number $\mathrm{Fo}=\kappa\Delta t/\text{area}$); $\operatorname{Re}\mu$ and $\operatorname{Im}\mu$, bounded by one, which is better conditioned than raw tensor entries **[AI Inference]**; $\log\kappa$; and for vector physics the rotation angle $\theta$ of $DF=RU$. Three dimensions has no conformal charts except Moebius maps (Liouville's theorem), so a conformal-only operator would have been a 2-D dead end.

**Open, to measure at Stage 0:** how much harder the general input class is to learn than the near-conformal one ($A\approx\kappa I$ after equal-modulus cuts). Registered as a comparison: accuracy per $|\mu|$ bin and iteration counts on the workbench's shapes.

### 3.5 The cut rule (decision 2)

Cutting at equal along and across **values**, not equal cell counts, gives every piece the modulus $M\,\Delta u/\Delta v$, because the pair is conformal up to an axis scaling. **Measured** (`out/arch/chart_maps.txt`): s-channel ($M=8.87$) cut 9 × 1, predicted 0.986, each piece's own solve 0.977–0.988; cut 18 × 2, 0.98–1.03; bend-3 cut 5 × 3, predicted 1.039, own 1.038–1.067. The metric is then $\mathrm{diag}(1.03,0.96)$ to a few per cent. The user chooses pieces across (or a target size); the layout sets the along count $n=\operatorname{round}(Mm)$. Equal modulus is not equal area (398 to 698 cells on the s-channel), but every piece costs one call on the same reference grid.

### 3.6 The certificate and the envelope (decision 3)

**The certificate.** $F$ is sampled on the reference grid and interpolated bilinearly. A bilinear quadrilateral's Jacobian is affine in each variable, so a positive $J$ at a cell's four corners makes it positive on the cell; and a local homeomorphism of the square whose boundary is traced once, positively, is injective (a degree argument). So **$J>0$ at every node and a simple, positively oriented boundary polygon prove the discrete chart injective**, exactly and cheaply.

**The envelope** is where an expert is trusted, declared on its Expert Card: $\sup|\mu|\le k$; $\max\log J-\min\log J\le\ell$; and smoothness bounds on $\hat\nabla\log J$ and $\hat\nabla\mu$, because the input's frequency, not its range, is what drives error (the topology benchmark's finding). The compiler measures each chart and refuses one outside the envelope, naming the quantity; the layout can then cut again (bend-3's conformal $J$ ratio falls from 10 to 2.7 from 2 × 1 to 5 × 3 pieces).

**Learned maps are left out because they are not needed, not because they cannot be certified**: the same check would certify one. The pull-back is exact for any valid chart, so nothing about the map has to be learned, and a learned map would move the expert's inputs whenever the network is retrained. Solution-adaptive charts (clustering resolution in a boundary layer) are the place a learned map could later earn a role.

### 3.7 The geometry source (decision 4)

The workbench holds the domain as a cell mask. Charting the staircase would hand the expert a staircase. Instead the chart is built from the **drawn smooth outline**, and the classical reference is solved **in chart coordinates on the reference grid** — the same pulled-back equation by finite volumes. That gives, for free, two things the record lacked: **a same-class reference at any resolution** (S8, whose absence blocked 47 tiers of the frozen-checkpoint campaign) and **the like-for-like classical local solver that gate E3 needs**. A build item for the workbench.

### 3.8 The coupling stack (decision 5)

**Intuition: exchange responses, not values.** Schwarz iteration is like assembling a truss by letting each member push and adjust over and over; structural engineers instead report each member's stiffness at its joints, assemble, and solve once (the direct stiffness method). A neighbour's *value* says where it is; its *response* says how it will react to anything, so one exchange suffices.

| tier | when | how | rounds | certificate |
|---|---|---|---|---|
| **0, superelement** | linear physics; the expert supplies its port data | assemble the pieces' port matrices through the port algebra's connections; one direct (2-D) or preconditioned-CG (3-D) interface solve | **0** | the port matrices' symmetry and positivity (by construction: the energy Gram of decision 6, §3.12); the interface solve's error bound in §5 |
| **1, tangent** | nonlinear or implicit | Newton–Schur with the expert's tangent port matrices, or mode-matched impedance plus Anderson | a few Newton steps | local: tangent coercivity, a trust region |
| **2, black box** | the expert exposes waves only (a donor, a classical code, a fallback) | optimized Robin waves, a coarse space, GMRES or Anderson; Douglas–Rachford where energy safety matters | $O(1)$ in the number of pieces with a coarse space | a Lipschitz or firm-non-expansiveness bound (T1, T8, T9) |

**Measured** ([[coupling-cost-and-complexity]]), steady, 8 × 8 pieces, exact local solves: the traditional scheme 1,864 rounds; version 0's wave exchange 1,124; with GMRES 83; with a coarse space 70; the superelement no rounds and $4m$ local solves per piece to form its port matrix (a learned one: about two calls, **[AI Inference]**). Port truncation at 16 modes per side: 0.017–0.14 %. **A monolithic direct solve is faster than every decomposition at these 2-D sizes on one machine**; the superelement's case is the expert-call count where classical work cannot be reused (§6 of that page).

A tier-0 expert is also a tier-2 expert: from its port matrix $\Lambda$ the host forms the scattering map $S=(I-Z\Lambda)(I+Z\Lambda)^{-1}$ for any impedance, so superelements remain compatible with version 0's wave contract.

### 3.9 Tier 2's accelerator (decision 7)

**Measured**, 8 × 8, steady: Anderson (depth 5) cuts the wave exchange from 1,124 rounds to 152 with no derivative information; GMRES on the interface fixed point to 83, but needs Jacobian–vector products of the expert, which a black-box donor may not give; the coarse space flattens the growth in the number of pieces (exponent 0.14–0.29 against 0.88–0.96). Hence: GMRES with JVPs, Anderson without, and a coarse space from 16 pieces.

### 3.10 Ordered sweeps (decision 8)

A chain of pieces along a channel can be solved exactly in two ordered sweeps (block elimination along the chain, the idea behind sweeping preconditioners: Engquist & Ying, *Comm. Pure Appl. Math.* 64, 2011). Ordered sweeps are multiplicative, which rule R5 of [[general-coupling-scheme]] forbids so that symmetry averaging stays exact ([[symmetry-averaging-atlas-0.1]]). Decided: **allowed in tier 2 only, never where symmetry averaging is used.**

### 3.11 The transient policy (decision 9)

1. **Reuse.** At a fixed step and fixed geometry a tier-0 piece's port matrix does not change, so it is formed once; each later step costs about one or two expert calls per piece and a back-substitution.
2. **Warm start** from the previous step's interface data, extrapolated.
3. **Energy-safe early termination.** Where the coupling is passive for any finite number of exchanges (Douglas–Rachford in wave coordinates with firmly non-expansive ports: Wei et al., arXiv 2603.16424), a step may stop after one or two exchanges without risking instability; the unconverged remainder is the $\rho^k$ term of T22 in [[formal-proofs-plan]], an accuracy term, reported.

**Measured** context: small steps localise the coupling physically; at $\mathrm{Fo}=0.01$ every scheme needs only 13–40 rounds.

### 3.12 The output format (decision 6)

**One canonical question for every expert.** Given values on all four sides of the unit square, what is inside, and what flux leaves each side? The all-Dirichlet Dirichlet-to-Neumann map holds all of a piece's linear boundary behaviour, so whether a side is a wall, an inlet, a Robin condition or a seam is the **host's** business (a linear constraint on side values and fluxes), as are impedances and neighbours. No expert can develop its own quirks about boundaries.

**The answer: constraint modes and an energy Gram.** The expert returns, for each port mode $k$, the field $\hat H_k$ with that mode as its exact trace, and a particular field $\hat u_p$ with zero trace:
$$\hat H_k=\hat H_k^{(0)}+\beta\odot N_{\theta,k}(\log J,\mu,\log\kappa,\mathrm{Fo}),\qquad \beta=16\,\xi(1-\xi)\,\eta(1-\eta),$$
with $\hat H_k^{(0)}$ the exact response of a plain unit square (cosine-times-sinh series, shifted for a time step) and $N_{\theta,k}$ the learned correction. The host forms
$$\Lambda_{jk}=v_j^{\top}M\,v_k,\qquad v_k=(\hat H_k,\ e_k),\qquad b_k=\hat H_k^{\top}f-v_k^{\top}M(\hat u_p,0),$$
with $M$ the piece's own discrete energy form on the reference grid.

**What it guarantees, by construction.** $\Lambda$ is symmetric and positive semidefinite whatever the network returns. Because the exact modes are energy-orthogonal to every trace-free perturbation (the Dirichlet principle),
$$\Lambda_{\text{learned}}-\Lambda_{\text{true}}=E^{\top}A\,E\ \succeq0,\qquad E=\hat H-\hat H_{\text{true}},$$
so **the learned piece is never softer than the true one, and its port error is quadratic in its field error**. The coupled answer is a Ritz–Galerkin solution in the span of the learned modes, the energy-best by Céa's lemma: coupling cannot amplify an expert's error. And $\operatorname{tr}\Lambda_{\text{learned}}-\operatorname{tr}\Lambda_{\text{true}}=\sum_k\lVert E_k\rVert_A^2$, so minimising the modes' own energy minimises their error with no labels (**[AI Inference]**, registered as a Stage 0 comparison against supervised training).

**Measured** (`scripts/arch_superelement_gram.py`, `out/arch/superelement_gram.txt`; 4 × 4 pieces of $32^2$ cells, 16 modes per side; exact modes and matrices perturbed by the same relative amount $\varepsilon$):

| $\varepsilon$ | port error, Gram | port error, direct head | direct pieces indefinite | solution error, Gram | solution error, direct |
|---|---|---|---|---|---|
| 0.30 | 8.1 % | 30 % | 16 of 16 | 64 % | 153 % (assembled indefinite) |
| 0.10 | 0.90 % | 10 % | 12 | 18 % | 1,380 % (assembled indefinite) |
| 0.03 | 0.081 % | 3 % | 3 | 2.0 % | 58 % |
| 0.01 | 0.009 % | 1 % | 3 | 0.27 % | 9.2 % |

Steady; at $\mathrm{Fo}=0.1$ the same pattern (0.12 % against 1.9 % at $\varepsilon=0.01$). The identity holds to $2\times10^{-15}$. The direct head was given its best case (exact fields and loads), and its steady interior pieces turn indefinite at any $\varepsilon$ because their true port matrix is singular (a uniform temperature carries no flux).

**Prior art.** Constraint modes are classical: Craig–Bampton component mode synthesis (*AIAA J.* 6, 1968), multiscale finite elements (Hou & Wu, *J. Comput. Phys.* 134, 1997), static condensation with ports (Huynh, Knezevic & Patera, 2013). Networks predicting multiscale basis functions are published (Wang, Cheung, Chung, Efendiev & Wang, *Deep multiscale model learning*, *J. Comput. Phys.* 2020; CNN-predicted reduced bases, arXiv 2406.16328); so is the principal-part split for DtN maps (PPDNO, arXiv 2606.25952). **Not found [AI Inference]:** per-chart constraint modes with exact traces whose host-computed energy Gram serves as the **coupling certificate** (positive by construction, error from above, Céa-optimal coupling), with typed multiphysics ports.

**Details decided:** 16 cosine port modes per side, about 20 reference modes inside the expert to absorb the side's reparametrisation to physical arc length (a mortar projection the host performs); the square's D4 symmetry by data augmentation at Stage 0 (group-equivariant convolutions or rotated passes as later options; a 90 degree rotation sends $\mu\to-\mu$); at a fixed step the modes are formed once and reused; tier 1 uses the same format with tangent modes; vector physics adds per-component modes and $\theta$. The host's Gram costs about $2(4m)^2n^2\approx34$ MFLOP per piece at $m=16$, $n=64$ (an operation count, not a measurement), a tenth of the smallest backbone's forward pass.

### 3.13 The backbone and its size (decision 10)

**What it must learn:** fields on the $n\times n$ square to $4m$ correction fields and a particular field; an elliptic extension, so a piece-wide receptive field. Tier 0 needs no certified Lipschitz constant of the network (the Gram does the certifying), which frees the choice.

**The trunk:** a U-shaped convolutional operator in the manner of the Convolutional Neural Operator (Raonic et al., NeurIPS 2023), with explicit odd (for the zero-trace corrections) and even (for inputs) boundary extension in place of the periodic one, anti-aliased resampling, and FiLM conditioning on the Fourier number and parameters. **The registered ablation:** a spectral operator in the sine and cosine bases, whose sine series vanish on the boundary exactly as the corrections must. A U-Net with a spectral bottleneck is the Stage 1 fallback; a transformer appears only in the Stage 2 fine-tuning comparison; graph and point operators stay the 3-D fallback.

**Measured forward cost** (`scripts/arch_backbone_timing.py`, `out/arch/backbone_timing.txt`; untrained, so cost only; torch 2.14.1 on the container's CPU, 4 threads, float32 — a proxy for the owner's laptop):

| size | parameters | GFLOP per piece, $n=64$ | ms per piece, alone | ms per piece, batched by 16 | × a classical local solve (0.85 ms) |
|---|---|---|---|---|---|
| small | 0.49 M | 0.35 | 6.0 | 3.8 | 4–7 |
| medium | 1.95 M | 1.35 | 16.9 | 7.7 | 9–20 |
| large | 6.2 M | 5.0 | 30 | 25 | 30–36 |

**The S7 arithmetic for 2-D diffusion on a CPU.** Recomputing one piece's port matrix classically at $n=64$ (4 × 4 pieces, 16 modes per side) was **measured** in the coupling probe at about 21 ms of factorisation plus 47 ms of local solves, about 68 ms per piece; a 1–2 M-parameter expert costs 8–17 ms. **So on a CPU the learned superelement is 4–9 times cheaper than recomputing classically, and only when geometry or material changes**; on a fixed geometry the stored classical modes are cheaper still. The decisive cases remain a GPU, three dimensions, and expensive local physics ([[coupling-cost-and-complexity]] §6).

**Size:** 1 M parameters by default; the sweep {0.3 M, 1 M, 3 M} is registered in the Stage 0 gate against accuracy per $|\mu|$ bin and the S7 bar measured on the owner's laptop and a GPU. Reference grid $n=64$, trained on a mixture of 32, 64 and 128; float32 training, float64-capable inference for the probes (S10).

---

## 4. The open decisions

| # | question | the current proposal | status |
|---|---|---|---|
| 11 | interface data and lifting | Robin or wave modal coefficients; Dirichlet imposed exactly by lifting | not yet answered |
| 12 | the conservation head | fluxes on faces, exact discrete conservation, required for conservation-law families | not yet answered |
| 13 | certification | tier 0 by construction and eigenvalues; tiers 1–2 by bound propagation, with a linear-plus-correction fallback | not yet answered |
| 14 | time | the step as the Fourier number; a step-doubling consistency loss; a declared range | not yet answered |
| 15 | the training objective | state, port data (the response term, supervised directly), step consistency; optionally an iteration-aware loss | not yet answered |
| 16 | the Expert Card | envelope, Fourier range, port modes, certificates, precision, licence; compiler rules including the tier | not yet answered |
| 17 | the training route | from scratch for Stages 0–1; a controlled fine-tuning comparison at Stage 2 (DPOT, Walrus or GPhyT; Poseidon excluded by its licence) | under discussion |
| 18 | the first family and its gate | 2-D diffusion (`THERM`); E1–E6 plus superelement rows; the expert's forward time measured first | under discussion |
| 19 | the name | to choose | under discussion |
| 20 | the proposal's author line | to ask | not yet answered |

---

## 5. For the proofs chat — a draft, to be finalised in version 1

The decided coupling stack changes what the headline theorem of [[formal-proofs-plan]] §3 must cover. **Tiers 1 and 2 keep T1, T8, T9 and T9′ as stated.** Tier 0 has no iteration, so its guarantee is a statement about one linear solve. A draft, **ours**, Tier A in Lean (finite-dimensional linear algebra):

> **T24 (draft) — the superelement interface solve.** Let each piece $i$ supply a symmetric positive semidefinite port matrix $\tilde\Lambda_i$ and a load $\tilde b_i$, and let the assembled interface matrix $\tilde{\mathsf S}=\sum_iR_i^{\top}\tilde\Lambda_iR_i$ be positive definite on the free port space, with smallest eigenvalue $\tilde\beta>0$. Then the interface system $\tilde{\mathsf S}\tilde\lambda=\tilde\chi$ has a unique solution, and against the exact system $\mathsf S\lambda=\chi$
> $$\lVert\tilde\lambda-\lambda\rVert\ \le\ \frac{1}{\tilde\beta}\Bigl(\lVert\mathsf S-\tilde{\mathsf S}\rVert\,\lVert\lambda\rVert+\lVert\chi-\tilde\chi\rVert\Bigr).$$

It is the master bound's transmission term ([[master-error-bound]] §4) with its inf-sup constant **guaranteed** rather than measured. Decision 6 adopted the energy-Gram construction (§3.12), which allows a stronger companion, also **ours** and also Tier A:

> **T25 (draft) — the Gram superelement is a Ritz–Galerkin method.** Let $A$ be symmetric positive definite on the global grid, let each piece's learned constraint modes $\hat H_{i,k}$ carry the exact port traces, and assemble $\tilde{\mathsf S}$ from the Grams $\Lambda_i=V_i^{\top}A_iV_i$. Then (i) every $\Lambda_i$ is symmetric positive semidefinite; (ii) $\Lambda_i-\Lambda_i^{\text{true}}=E_i^{\top}A_iE_i\succeq0$ with $E_i=\hat H_i-\hat H_i^{\text{true}}$; (iii) the coupled field $\tilde u$ satisfies $\lVert u-\tilde u\rVert_A=\min_{w\in W}\lVert u-w\rVert_A$ over the span $W$ of the learned modes and particular fields (Céa's lemma with constant 1).

Item (ii) is checked numerically to $2\times10^{-15}$ by `scripts/arch_superelement_gram.py`; that is a test, not a proof.

**One more item for the proofs chat:** the SNI counterexample (§1, item 2) is a clean, checkable example that T1's Lipschitz hypothesis cannot be weakened to "the classical map contracts and the learned one is close". It may be worth formalising as a remark beside T1.

**The owner relays this; the architecture chat cannot reach the proofs chat directly.**

---

## 6. What this page does not do

1. It does not replace version 0. The wiki's design pages are rewritten once the owner confirms the whole design.
2. Nothing learned was trained. The probes use classical solves; the one learned measurement is the forward cost of untrained backbones (§3.13), which says nothing about accuracy.
3. The block layout for forks and rings (§3.3), the smooth-outline chart (§3.7) and every tier are designs, not builds.

---

## See Also

- [[coupling-cost-and-complexity]] — the measured cost of each coupling strategy
- [[chart-operator-architecture]] — version 0, the starting point
- [[chart-operator-training-and-cost]] — training and costs, to be updated with decisions 15–18
- [[dd-neural-prior-art-2026]] — the prior art, to be updated with the verdicts of §1
- [[formal-proofs-plan]] — T1, T8, T9, T22, and the draft T24 of §5
- [[outcome-c5-requirements-for-dd-native-experts]] — S1–S12, which every decision answers to
