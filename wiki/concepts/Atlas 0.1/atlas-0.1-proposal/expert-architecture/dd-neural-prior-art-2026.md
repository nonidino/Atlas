# Neural operators inside domain decomposition — the prior art as of 2026-09, and what is left to claim

**Type:** Concept page — **positioning audit** (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** written 2026-09-30 from web searches made that day. **arXiv, the three reference websites and most publishers were blocked by the planning session's network**, so every entry below is from a search result's summary, not from the paper. **Each row is marked for the architecture chat to read in full before the proposal cites it.** The vault's earlier audit, [[prior-art-and-novelty-atlas-0.1]], predates every 2025–2026 entry here.
**Hub:** [[00-proposal-workstreams]] · **Used by:** [[chart-operator-architecture]] · [[website-evidence-and-citations]]

---

## 0. The finding, in one paragraph

**"Train a neural operator on small pieces, then run it inside a Schwarz iteration to solve on any geometry" is published, several times, and recently.** Mosaic Flows (2021–22) did it with physics-informed subdomain networks. *Operator Learning with Domain Decomposition* (ICLR 2026) did it with neural operators and proves a convergence rate and an error bound. NEST (May 2026) does it on $3\times3\times3$ voxel patches for 3-D hyperelasticity. A learning-based DDM (2025, *CMAME* 2026) uses one pretrained operator as the universal local solver. **Mapping each domain to a reference domain and learning there is published too**: Geo-FNO learns the map, and DIMON and the Diffeomorphism Neural Operator compute it (by LDDMM, and by harmonic and volume parametrisations). **Learning the DD method's own parts is published**: interface conditions (NeurIPS 2022), coarse spaces, and neural operators blended with relaxation (HINTS, 2024). **So the proposal must not claim any of these as new.** What is not found in these searches is §3: a **multiphysics typed-port contract**, a **convergence certificate that does not depend on how well the expert was trained**, and **training aimed at the iteration count**, backed by a measured failure record and machine-checked proofs. **[AI Inference]:** that is a narrower claim than "neural DD on any geometry", and a stronger one.

---

## 1. The closest work

### 1.1 Learned local solvers composed by Schwarz iteration

| work | what it does, from its summary | what to check in full |
|---|---|---|
| **Mosaic Flows** — Wang, Planas, Chandramowlishwaran & Bostanabad, *CMAME* 389 (2022), arXiv 2104.10873 | a physics-informed network solves a boundary-value problem on a small square ("genome") for arbitrary boundary conditions; an iterative predictor assembles its inferences on large unseen domains. Reports **1–3 orders of magnitude** speed-up over a state-of-the-art PINN on Laplace and Navier–Stokes | the baseline is a PINN, not a classical solver |
| **Operator Learning with Domain Decomposition for Geometry Generalization** (Schwarz Neural Inference, SNI) — Huang et al., **ICLR 2026**, arXiv 2504.00510 | a local neural operator trained on **randomly generated shapes**, using the PDE's symmetries to augment the data; SNI partitions a domain into subdomains and stitches local solves. **Theoretical convergence rate and error bound**; linear and nonlinear PDEs | **read first**: its transmission condition, whether subdomains are shaped or boxes, its theorem's hypotheses, and its speed against a classical solver |
| **A Learning-based Domain Decomposition Method** (L-DDM) — arXiv 2507.17328, *CMAME* 2026 | one pretrained neural operator, trained on simple domains, used as the universal local surrogate inside a classical iterative DD; elliptic PDEs with discontinuous microstructures; **resolution invariance** and generalisation to unseen microstructures | the classical iteration used, and the cost comparison |
| **Neural-Schwarz Tiling (NEST)** — Secchi et al., arXiv 2605.12343 (May 2026) | a neural operator on **minimal $3\times3\times3$ voxel patches** with diverse local geometry and boundary or interface data; an unseen voxelised domain is tiled into overlapping patches and made consistent by **Schwarz iteration with partition-of-unity assembly**; compressible neo-Hookean solids in 3-D, far beyond the training patches' scale | iteration counts at scale, and any coarse level |
| **Non-overlapping Schwarz hybrid FE–neural operator for solid mechanics on irregular domains** — arXiv 2606.08796 (2026) | a finite-element region coupled to a neural-operator region by non-overlapping DD; a **Point-DeepONet** takes unstructured FE point clouds directly, so the learned subdomain may be non-convex | the transmission condition, and convergence |
| **DD-DeepONet** — arXiv 2508.02717 (2025) | domain decomposition with DeepONet in three application scenarios | scope |
| **Interfacing finite elements with deep neural operators** — Yin, Zhang, Yu & Karniadakis, *CMAME* 402 (2022) | FE and DeepONet coupled by Schwarz for multiscale mechanics | the reported speed-up |

### 1.2 Geometry through a map to a reference domain

| work | what it does | relation to the chart operator |
|---|---|---|
| **Geo-FNO** — Li, Huang, Liu & Anandkumar, *JMLR* 24 (2023) | **learns** a deformation from the physical domain to a latent uniform grid, then applies an FNO | the map is learned, with no injectivity guarantee; the chart operator computes its map classically ([[chart-operator-architecture]] §2) |
| **DIMON** — Yin, Charon, Brody, Lu, Trayanova & Maggioni, *Nature Computational Science* (2024) | learns geometry-dependent solution operators on a **template** domain with respect to the **pulled-back** PDE operator; the shape enters as a parameter encoding the **diffeomorphism**; Laplace, reaction–diffusion, cardiac electrophysiology | the closest idea to the owner's "diffeomorphic neural operator": **this is published**. DIMON maps a whole domain; the chart operator maps each subdomain of a decomposition, and passes the metric explicitly |
| **Diffeomorphism Neural Operator (DNO)** — Zhao et al., *Communications Physics* (2024), arXiv 2402.12475 | learns on a **generic domain** reached by a diffeomorphism, using **harmonic parametrisation in 2-D and volume parametrisation in 3-D**; Darcy, pipe, airfoil and mechanics; a geometric-similarity measure predicts generalisation | **the harmonic map is published as the map**. What it passes to the operator must be read |
| **Diffeomorphic Latent Neural Operators** — arXiv 2411.18014 | a latent-space variant | to read |

### 1.3 Geometry without a map

| work | what it does |
|---|---|
| **GINO** — Li et al., *NeurIPS* 2023 | signed distance function and point cloud; graph operator to and from a regular latent grid, FNO in between. On car aerodynamics, **"a 26,000× speed-up compared to optimized GPU-based CFD simulators on computing the drag coefficient"**, at about 3% relative error, trained on 500 samples |
| **Transolver** — Wu et al., *ICML* 2024 | physics-attention over learned slices of a general mesh (MIT licence; [[expert-donor-survey]]) |
| **Beyond Arbitrary Geometry: Topology Generalization in Neural PDE Operators** — Chen, Xu, Nie, Fan & Wang, arXiv 2609.05860 (Sept. 2026) | a benchmark (TopoBox-3D) in which tunnels and cavities vary the topology. Its finding: out-of-distribution error under a topology change is driven **by the input field's mean frequency (its Rayleigh quotient)**, not by the dimension of the harmonic kernel. **[AI Inference]:** this supports keeping topology out of the learned operator, which is the chart operator's first separation |

### 1.4 Learning the decomposition method's own parts

| work | what it learns |
|---|---|
| **Learning Interface Conditions in Domain Decomposition Solvers** — Taghibakhshi, Nytko, Zaman, MacLachlan, Olson & West, *NeurIPS* 2022 | **optimized-Schwarz interface conditions on unstructured grids** with graph networks, trained unsupervised on small problems, robust on arbitrarily large ones at cost linear in size |
| **Learned adaptive coarse spaces** — Heinlein, Klawonn, Lanser & Weber (FETI-DP, from 2019); *Learning Adaptive Coarse Spaces Using Transferable Neural Network Models for Linear and Nonlinear Overlapping DD Methods*, arXiv 2607.06261 (2026) | which coarse constraints or basis functions to add |
| **HINTS** — Zhang, Kahana, Turkel, Zhang & Karniadakis, *Nature Machine Intelligence* 6, 1303–1313 (2024) | a DeepONet interleaved with relaxation: the network's spectral bias handles the **low** modes relaxation stalls on, relaxation the high ones, for a **uniform convergence rate**; usable as a Krylov preconditioner. **[AI Inference]:** the same slow-mode story as this vault's S5 and its defect-correction slot ([[defect-correction-learned-operator]]) |
| **DeepONet-based preconditioning** — Kopanicakova & Karniadakis, *SIAM J. Sci. Comput.* (2025) | neural-operator preconditioners for parametric linear systems |
| **Local neural operators for equation-free system-level analysis** — Fabiani, Vandecasteele & Goswami, *Nature Machine Intelligence* 8, 1127–1141 (2026) | local neural operators wrapped in matrix-free Krylov methods for fixed points, stability and bifurcation analysis |
| **Survey:** Klawonn, Lanser & Weber, *Machine learning and domain decomposition methods — a survey*, *Computational Science and Engineering* 1 (2024), arXiv 2312.14050 | the map of this field; read before writing any positioning sentence |

### 1.5 Training on derivatives

**DINO** — O'Leary-Roseberry, Chen, Villa & Ghattas, *J. Comput. Phys.* 496 (2024): a neural operator trained on its **Fréchet derivative** as well as its values. The chart operator's Jacobian-fidelity loss ([[chart-operator-architecture]] §5) is this idea aimed at a different derivative: the response to **interface** data, because that sets the Schwarz rate.

### 1.6 Formal proofs next door

- **EllipticPDE** (Soto Franco; arXiv 2609.32561, Sept. 2026): Lean 4 on Mathlib, with Sobolev spaces built independently, Poincaré, Lax–Milgram weak solutions, Rellich–Kondrachov, Fredholm, spectral theorem, interior regularity, **no `sorry`**. Repository `github.com/alejandro-soto-franco/EllipticPDE`.
- **A Galerkin construction for 2-D Navier–Stokes in Lean**, arXiv 2609.33033 (2026).
- **TorchLean**, arXiv 2602.22631 (2026): neural networks in Lean 4, with interval-bound and CROWN-style verification.
- **No machine-checked domain decomposition theory was found.** [[formal-proofs-plan]] builds on this gap.

---

## 2. What the proposal must not claim

1. That solving on arbitrary geometry by composing learned local solvers with Schwarz iteration is new (Mosaic Flows, SNI, L-DDM, NEST).
2. That learning on a reference domain through a diffeomorphism, harmonic or otherwise, is new (DIMON, DNO, Geo-FNO).
3. That learned interface conditions, learned coarse spaces or neural-operator-accelerated iterations are new (Taghibakhshi et al., Heinlein et al., HINTS).
4. That neural operators are fast on complex geometry. That is GINO's, Transolver's and many others' result, and the website should cite it as theirs ([[website-evidence-and-citations]]).

---

## 3. What is left, and is the proposal's

**[AI Inference]** throughout: each item was **not found in the searches of 2026-09-30**, which were summaries, not full reads. The architecture chat confirms or withdraws each one against the full papers of §1.1–§1.4.

| # | the claim | why it matters | what backs it in this vault |
|---|---|---|---|
| N1 | **One expert contract across physics**: typed effort–flow ports whose product is power, so a learned fluid expert, a learned solid expert and a classical circuit couple through one interface, and a compiler admits or refuses each seam with a cited rule | every work in §1.1 is one PDE family with its own coupling code | [[port-algebra-atlas-0.1]] · [[outcome-c4-modular-multiphysics]] · the compiler's verdicts ([[showcase-gallery]] §4) |
| N2 | **A convergence certificate independent of training quality**: experts exchange **wave variables** (impedance-weighted Robin data), each expert's scattering map carries a **certified Lipschitz bound below 1**, and the coupled iteration then converges geometrically from any start, **however inaccurate the expert**. Its limit is within a stated distance of the classical answer | SNI proves a rate *given* an accurate local operator; a certificate that holds for any expert passing a checkable bound is a different kind of guarantee | [[chart-operator-architecture]] §4; theorems T8–T9 in [[formal-proofs-plan]]; the vault's standing theme that "agreement buys nothing" without the right hypotheses ([[composition-error-theory]]) |
| N3 | **Training aimed at the iteration count**: a loss on the expert's response to low-frequency interface perturbations, because that response, not accuracy, sets the Schwarz and defect-correction rates | measured here: per-band response ordered twenty cheap maps' classical-call counts at rank correlation $+0.83$ against accuracy's $+0.69$ | [[corrupted-checkpoint-and-jacobian-fidelity]] · [[defect-correction-learned-operator]] |
| N4 | **Charts with the metric passed explicitly**: the decomposition makes every piece a topological square, a classical injective map charts it, and the operator receives the **pulled-back coefficients** $J\,DF^{-1}\kappa\,DF^{-\top}$ rather than a shape code | DIMON and DNO map whole domains; NEST and SNI learn local pieces without a chart. **Incremental**, and should be presented as a design choice, not as the novelty | [[chart-operator-architecture]] §2–§3 |
| N5 | **A measured specification of failure**: 89 tiers recording, as numbers, why a frozen foundation model fails in each DD slot | no published work reports what a DD-native expert must satisfy from the failures of one that is not | [[outcome-c5-requirements-for-dd-native-experts]] |
| N6 | **Machine-checked convergence and error theorems** for coupling with learned experts | no formalised DD theory was found | [[formal-proofs-plan]] |

---

## See Also

- [[prior-art-and-novelty-atlas-0.1]] — the vault's first audit (framework-level, 2026-08); this page extends it to learned experts in DD
- [[chart-operator-architecture]] — the design positioned here
- [[atlas-and-standard-dd-theory]] — the classical home of every DD mechanism named
- [[expert-donor-survey]] — licences and inputs of the downloadable models
- [[website-evidence-and-citations]] — which of these the website cites, and for what
