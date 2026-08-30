# Open Architectural Problems (initial model — critical self-audit)

**Type:** Concept — initial model audit (folder: Noether 1.0)
**Status:** Critical evaluation of the initial-model design *aside from training data*. Honest accounting of what stands between the current design and a complete, implementable architecture. Each problem lists the gap, why it matters, any conflict with the stated commitments, and resolution directions. **As of 2026-07-01 all 25 tracked problems are DECIDED** — see Addendum 4 (bottom); the design is now specified end-to-end for 1D Burgers ([[noether-1.0]]) and mechanism-complete for the first multiphysics target, 2D Rayleigh–Bénard convection. Remaining work is implementation, not open design (with particle-coupling implementations, Problem 9's equivariance-cost ablation, and Problem 22 moving boundaries the only explicitly deferred items).
**Related Concepts:** [[00-initial-model-overview]], [[initial-model-architecture]], [[graph-tokenizer]], [[symmetric-attention-physics]], [[skew-symmetric-attention]], [[normalization-scheme]], [[pipeline-contract]], [[training-curriculum]], [[multihead-attention]], [[intelligent-patching]], [[noether-1.0]], [[smoke-test-bringup]]
**Related Summaries:** [[dynami-cal-graphnet]], [[hamiltonian-neural-networks]], [[latent-diffusion-physics]], [[poseidon-pde-foundation-model]]

---

## Framing

The conceptual skeleton of the initial model is sound — the tokenizer topology story, the conditioning interface, the per-layer attention math, and the *conceptual* layering of conservation. The gaps are concentrated in two places: **turning per-layer/conceptual properties into trajectory-level guarantees**, and **the hard-constraint / lossy-autoencoder plumbing** that a working model needs. In several places the design currently conflates things that are not the same, or calls "conditioned" what would need to be "enforced." This page is deliberately critical.

Two recurring root causes worth naming up front:
- **Depth-dynamics vs. time-dynamics conflation** — properties proven across the $L$ layers of one forward pass are quietly treated as properties of the physical rollout.
- **"Conditioned" vs. "enforced"** — feeding the model a code/parameter makes it *aware* of a constraint; it does not make the output *comply*.

---

## Tier 1 — gaps that threaten the core claims

### Problem 1: Depth-conservation ≠ time-conservation (and it conflicts with the direct-state commitment)
**Status: DECIDED** — S1.1 adopted (2026-06-30).

**The gap.** The conservation story in [[initial-model-architecture]] (skew-symmetric channel, force-style aggregation) is the A-DGN result about dynamics *across the $L$ layers of one forward pass* ([[anti-symmetric-dgn]], [[skew-symmetric-attention]]). That is **not** the same as the decoded physical trajectory conserving energy/momentum *across the autoregressive rollout* $t\to t+1\to t+2$.

**Why it matters / the conflict.** The clean mechanisms this story was borrowed from — [[dynami-cal-graphnet]] (exact momentum), [[hamiltonian-neural-networks]] (exact energy) — obtain conservation by predicting **forces/derivatives** and tying the architectural antisymmetry to physics **through an explicit symplectic / integrator step**. The initial model then chose **direct state prediction with no integrator** ([[00-initial-model-overview]] commitment #2). That choice *severs the very link* that converts architectural antisymmetry into physical conservation. So commitment #4 ("inbuilt long-horizon stability") is currently **aspirational** — it leans on machinery that commitment #2 removed. Conserving the hidden-state aggregate across depth says nothing rigorous about the decoded $\|u\|^2$ from step to step; the decoder is a separate learned map that can break it.

**Resolution directions.**
- **Hybrid symplectic/projection step** — predict most of the state directly, but route the *conserved subspace* (total momentum, energy, mass) through a structure-preserving update so those invariants advance correctly in time.
- **Mandatory projection head** — make the currently-optional projection (predict invariants, project the output onto the manifold that satisfies them) a required stage, not an ablation.
- **Accept soft conservation + correction** — drop the "exact/inbuilt" framing, enforce conservation via a penalty + the diffusion/PINN corrector, and measure the residual drift honestly.

This is the deepest problem and is entangled with Problems 2 and 5 (all reduce to *enforce vs. learn*).

#### Candidate solutions (scored /10)

*Scoring rubric (used throughout this page): **leverage on the gap × implementation feasibility × strength of evidence**, 1–10, for the **initial** model specifically — not for the mature system.*

**S1.1 — Hard projection head onto conserved totals, hosted at the top supernode — 8/10.**
*Description.* After the residual head emits $\hat x_{t+1}$, read its global integrals off the **coarsest supernode** already in the hierarchy — mass $\int\rho\,dV$, momentum $\int\rho u\,dV$, energy $\int e\,dV$ — and project the fine field onto the affine manifold where those integrals equal their required targets (previous value $+$ known boundary flux). Linear invariants (mass, momentum) give a closed-form rank-$k$ correction; energy (quadratic) is a rescaling or small constrained solve. This is the corrected form of the "one graph per conserved quantity" idea: the conserved quantity is a *global functional read at the top of the existing tree*, not a parallel graph.
*Pros.* Converts a per-step prediction into one that satisfies global conservation **exactly**, turning conservation from a depth property into a trajectory property; reuses the supernode hierarchy (no new structure); cheap for linear invariants.
*Cons.* Pins only a handful of global scalars — necessary but weak (the field can be locally wrong while totals are exact); the energy projection is nonlinear and can be non-unique (which DOF absorb the correction?); requires knowing boundary fluxes to set targets; does nothing for local fidelity.

**S1.2 — Hybrid symplectic update on the conserved subspace — 7/10.**
*Description.* Split the state into a low-dimensional conserved/slow subspace and the fast remainder; regress the remainder directly but advance the conserved subspace with a structure-preserving (symplectic / energy-conserving) integrator, reintroducing on that subspace only the integrator link that direct-state prediction severed ([[hamiltonian-neural-networks]], [[dynami-cal-graphnet]]).
*Pros.* Restores the exact mechanism that makes architectural antisymmetry $\to$ physical conservation; theoretically the most principled; a *partial* integrator commitment that keeps the backbone equation-agnostic elsewhere.
*Cons.* Requires identifying a Hamiltonian/symplectic structure, which is unknown or nonexistent for dissipative/open systems; partially conflicts with commitment #2 (no integrator); unproven at foundation-model scale.

**S1.3 — Learned invariants (Noether-style) + projection — 6/10.**
*Description.* For systems whose invariants aren't known a priori, learn candidate conserved quantities from data (functionals that stay constant along training trajectories), then project onto the learned manifold as in S1.1.
*Pros.* Generalizes beyond the hand-listed mass/momentum/energy to arbitrary multiphysics regimes; linearly probeable; fits the breadth goal.
*Cons.* **[AI Inference]:** learned invariants are only approximate, so "hard" projection onto a *soft* manifold can inject error; identifiability is hard (many near-constant quantities exist); risk of projecting onto spurious invariants that fail OOD.

**S1.4 — Soft conservation penalty + corrector + measured drift — 5/10.**
*Description.* Drop the "exact/inbuilt" framing; add a conservation penalty to the loss, let the diffusion/PINN corrector clean residual violation, and honestly report drift over long rollouts.
*Pros.* Simplest; fully equation-agnostic; a realistic always-available baseline.
*Cons.* No guarantee; penalty weighting is fragile; contradicts the headline "architectural conservation" claim; drift returns at long horizon — treats the symptom.

**S1.5 — Push-forward / multi-step rollout training only — 5/10.**
*Description.* No hard enforcement; train on unrolled trajectories (push-forward, context-noise injection, MTP horizon $k$) so stability is learned implicitly ([[gns-graph-network-simulators]], [[walrus-paper]]).
*Pros.* Architecture-agnostic; empirically improves rollout stability; already planned.
*Cons.* Learns approximate stability, not conservation; expensive (backprop through rollout); no invariant guarantee.

**Recommended stack:** S1.1 (mandatory projection) + S1.5 (training), with S1.2 wherever a symplectic structure genuinely exists. **[AI Inference]:** S1.1 also becomes the natural host for Problem 2's divergence-free and boundary-condition projections — one projection stage, several constraints — which is why the "enforce the conserved/constrained subspace" decision collapses Problems 1 and 2 together.

**Decision (2026-06-30): adopt S1.1**, refined by two design clarifications:
1. **Project once on the final latent, not per-block.** Projecting between attention blocks would harden *depth*-conservation — the property that was never at risk — and over-constrain intermediate computation. The projection instead acts on the final latent $h^L$ via a *trained conserved-readout* $\hat C(h)=W_C h$ (validated against the physical integral read at the coarsest supernode), enforcing *time*-conservation without a decode round-trip. It must be the **last magnitude-affecting op** — after any output gain ([[normalization-scheme]]) and after Stage-7 diffusion, then re-project ([[initial-model-architecture]] Stage 5b).
2. **Tiered invariants — the model helps decide.** Known universal invariants (mass, momentum, energy) are hard-enforced with exact analytic forms; system-specific/hidden invariants (enstrophy, helicity, integrable charges) are **discovered** as minimum-variance, non-degenerate latent probes across the rollout, enforced softly, and promoted to hard projection only after validation. This folds **S1.3 into S1.1 as its discovery-half** (re-scored 6→7 in that role) and answers "do we or the model pick the invariants?" as *both, tiered*.

**Complement adopted (2026-06-30):** S1.5 push-forward training with a **defined curriculum horizon schedule** ($1\to2\to4\to8$ steps) as the trained-stability complement to the hard projection.

### Problem 2: Hard constraints are conditioned, not enforced
**Status: DECIDED** — P2.1–P2.4 decision-ready recommendation adopted; implemented concretely in [[noether-1.0]] §3.7 for the $k{=}1$ case.

**The gap.** BC codes and field-type codes enter as *conditioning* ([[pfm-interface-design]], [[initial-model-architecture]]). Conditioning makes the model *aware* of a no-slip wall or incompressibility; it does not make the output *comply*.

**Why it matters.** Two cases bite on the very first smoke test (2D incompressible Navier–Stokes):
- **Incompressibility** $\nabla\cdot u = 0$ is a *global elliptic* constraint requiring a pressure projection every step. "The supernode hierarchy can *represent* elliptic coupling" is not the same as *enforcing* divergence-free output ([[structure-preserving-tokens]] has the projection as an option only).
- **Boundary conditions** need ghost nodes / masking / projection to be satisfied exactly; conditioning on a BC code does not guarantee $u=0$ on the wall.

As specified, the architecture cannot guarantee a physically admissible fluid state.

**Resolution directions.** A divergence-free projection head (Leray / stream-function, [[pc-deeponet-cfd]]); ghost-node boundary handling integrated into the graph ([[dynami-cal-graphnet]]); make constraint enforcement a *selectable per-regime projection module* ([[structure-preserving-tokens]]) rather than a conditioning code.

#### Candidate solutions (scored /10)

*How a projection works (shared machinery). Given a raw prediction $\hat x$ and a constraint set $\mathcal C$, the projected output is the **nearest admissible state** $x^\* = \arg\min_{x\in\mathcal C}\|x-\hat x\|_M^2$. The structure of $\mathcal C$ sets the cost:*
- ***Linear equality** $Ax=b$ → closed form $x^\* = \hat x - M^{-1}A^\top\,(A M^{-1} A^\top)^{-1}(A\hat x - b)$: measure the violation $A\hat x-b$, distribute the correction back along $A^\top$; the metric $M$ chooses which DOF absorb it. **Conservation** ($k$ global rows) and **Dirichlet BCs** (local rows fixing boundary DOF, e.g. $u_i=0$ on a wall) are *both* of this form, so they stack into one system.*
- ***Divergence-free** $\nabla\cdot u=0$ is linear but **elliptic** ($A=\nabla\cdot$ has one row per node), so its projection is the **Leray step**: solve $\nabla^2\phi = \nabla\cdot u$, set $u^\* = u-\nabla\phi$ — a Poisson solve, which the supernode hierarchy (a neural multigrid) already can run.*
- ***Energy** is quadratic; project with a one-scalar Newton/Lagrange step $x^\*=(I+\lambda Q)^{-1}\hat x$.*

**P2.1 — Leray/Helmholtz divergence-free projection (Poisson solve on the supernode hierarchy) — 8/10.**
*Description.* Enforce $\nabla\cdot u=0$ via $u^\*=u-\nabla\phi,\ \nabla^2\phi=\nabla\cdot u$, running the elliptic solve as a V-cycle on the existing supernode hierarchy — the model's neural analog of the classical CFD pressure projection.
*Pros.* Exact incompressibility; reuses the hierarchy (no new machinery); textbook-understood; the same solve provides pressure.
*Cons.* Elliptic solve cost every step (aggravates Problem 11); needs consistent BC on $\phi$; is a **physical-space, post-decode** op (no clean latent form).

**P2.2 — Joint KKT projection: conservation + linear BCs + divergence in one system — 8/10.**
*Description.* Stack all linear constraint rows — global conservation, Dirichlet boundary DOF, discrete divergence — into one $Ax=b$ and project onto the **intersection** in a single KKT solve. This is the literal "force the BCs *inside* the projection" idea.
*Pros.* Enforces conservation, BCs, and incompressibility **simultaneously and consistently**; the mathematically correct answer — sequential projection onto one constraint *violates* the others, so they must be solved jointly (or iterated, Dykstra-style).
*Cons.* Requires a **compatibility condition** on $b$ (e.g. net boundary flux $=$ rate of mass change, or the system is infeasible); the divergence rows make it an elliptic solve, not a rank-$k$ correction; tangential no-slip inherits the classical projection-method splitting-error subtlety.

**P2.3 — Ghost-node / immersed-boundary BC handling in the graph — 7/10.**
*Description.* Represent BCs with phantom nodes just outside the domain whose values are set (reflection/interpolation) to enforce the wall, integrated into graph construction ([[dynami-cal-graphnet]]).
*Pros.* Handles irregular/curved/particle geometry the algebraic projection alone does not; local and cheap; composes with S4.1 typed edges and S4.2 immersed-boundary coupling.
*Cons.* Adds boundary bookkeeping to an already-dynamic graph (Problem 7); approximate for curved/moving boundaries.

**P2.4 — Divergence-free by construction (stream / vector potential) — 7/10.**
*Description.* Predict a stream function $\psi$ (2D) or vector potential $\mathbf A$ (3D) and output $u=\nabla\times\psi$ — automatically divergence-free, **no projection needed** ([[structure-preserving-tokens]]).
*Pros.* Incompressibility exact by construction, zero solve cost; elegant.
*Cons.* Clean only in 2D (3D vector potential carries gauge freedom); doesn't generalize to compressible or non-div constraints; changes the output representation; BCs on $\psi$ are less intuitive; enforces *only* div-free, not conservation or BCs.

**Decision-ready recommendation:** **P2.4** where it applies (2D incompressible) for exactness with no solve; otherwise **P2.2** (joint projection folding conservation + BCs) with **P2.1**'s Leray solve on the hierarchy for the elliptic part and **P2.3** ghost nodes for irregular geometry. All live in the S1.1 projection stage but **in physical space, post-decode** (elliptic/BC constraints have no clean latent form) — refining the pipeline order to: *latent global-scalar projection → decode → physical-space joint div-free + BC projection → diffuse residual → re-project*.

### Problem 3: The tokenizer is a lossy autoencoder with no specified training or adequacy guarantee
**Status: DECIDED** — S3.1 + complements (S3.3, S3.4) adopted (2026-06-30).

**The gap.** The query-compression encoder discards sub-patch detail; the coordinate-implicit decoder reconstructs it ([[graph-tokenizer]]). Unspecified: is the latent *adequate* (encode→decode fidelity)? Is the tokenizer **pretrained separately** (VQ-VAE/autoencoder style) or **end-to-end** with the backbone?

**Why it matters.** Without an answer, fine-scale turbulence/shocks are destroyed at tokenization and the decoder cannot recover them — and that loss **compounds every rollout step** ([[autoregressive-rollout-stability]]). The whole resolution-free story assumes the latent is sufficient; that assumption is untested and unspecified.

**Resolution directions.** A reconstruction objective with a measured fidelity floor; adaptive $N_q$ (more queries for turbulent inputs); a residual high-detail path ([[wave-particle-dual-tokens]]); decide explicitly between separate tokenizer pretraining vs. joint end-to-end training (a real fork with different stability/quality trade-offs).

#### Candidate solutions (scored /10)

*First, the reframing that makes the problem well-posed: **strictly lossless tokenization of an arbitrary continuous field is impossible** — an infinite-dimensional function space cannot inject into a finite latent (a counting argument). But physical fields are band-limited (a smallest active scale set by viscosity/Kolmogorov) and live on low-dimensional inertial manifolds, so they have **finite effective DOF**. The achievable target is therefore **"lossless above the physical cutoff $k_{\max}$, with bounded and measured loss below it."** The solutions below pursue that target.*

**S3.1 — Dual-path encoding: smooth latent + explicit high-frequency residual channel — 8/10.**
*Description.* Encode each patch as a low-rank smooth latent **plus** a separate residual stream carrying high-$k$ / sharp-feature content ([[wave-particle-dual-tokens]]). The smooth path handles the bulk; the residual path protects shocks and fronts that generic query-compression averages away.
*Pros.* Directly protects the features whose loss drives rollout error; the residual channel is a clean handoff to the Problem-5 generative path (**[AI Inference]:** the sub-grid content diffusion must generate is exactly what this channel stores); degrades gracefully to the smooth baseline if dropped.
*Cons.* Doubles token bookkeeping; the capacity split between paths is a new hyperparameter; the residual stream can be noisy and unstable to train.

**S3.2 — Spectral / wavelet patch basis with bounded truncation — 7/10.**
*Description.* Encode each patch in a basis that diagonalizes smooth fields (Fourier / Chebyshev / wavelet) rather than a generic learned-query set. Truncation error becomes explicit — the discarded high-$k$ tail — and boundable a priori.
*Pros.* Converts unbounded, unmeasured loss into a controllable, bounded error; wavelets localize sharp features well; interpretable coefficients.
*Cons.* A fixed basis may underperform a learned one on complex multiphysics; per-field-type basis choice; Gibbs ringing near true discontinuities demands many coefficients.

**S3.3 — Capacity matching: adaptive $N_q$ + intelligent patching — 7/10.**
*Description.* Match latent DOF to local effective DOF: more queries / finer patches where local spectral energy above $k_\text{cut}$ is high ([[intelligent-patching]]), keeping the latent above the loss threshold exactly where content lives.
*Pros.* Makes the capacity$\to$loss link explicit and controllable; token-efficient; reuses the refinement tree; can be driven by model uncertainty.
*Cons.* Dynamic token count complicates batching; needs a monitor function; refinement lag on fast features; tree-construction overhead.

**S3.4 — Reconstruction pretraining with a measured fidelity floor — 8/10.**
*Description.* Pretrain the tokenizer autoencoder-style and require encode$\to$decode error **below the data's own discretization error** on held-out turbulent/shock data — the operational definition of "lossless relative to the data." Resolves the separate-vs-joint fork toward *separate-then-finetune*.
*Pros.* The only achievable, testable standard of losslessness; de-risks the tokenizer before any dynamics are trained (the audit's top recommendation); gives a hard go/no-go gate.
*Cons.* A separately-trained latent may not be ideal for dynamics; a measured floor is not a guarantee for unseen regimes; adds a training stage.

**S3.5 — Invertible / bijective tokenizer — 4/10.**
*Description.* A normalizing-flow-style encoder with $\dim(\text{latent}) = \dim(\text{input})$, giving exact reconstruction.
*Pros.* Provably lossless — zero reconstruction error.
*Cons.* Abandons compression (defeats the purpose of tokenization); expensive; fits the variable-resolution graph setting poorly; you surrender exactly the intrinsic-DOF savings that motivated tokenizing.

**Required complement (not standalone): non-spectrally-biased decoder.** Any of the above is wasted if the coordinate-implicit decoder cannot *emit* high frequencies — coordinate-MLPs have a well-known spectral bias. Use SIREN / Fourier-feature decoders so decode-side loss doesn't reintroduce what the encoder preserved.

**Recommended stack:** S3.1 + S3.3 + S3.4 (residual dual-path, capacity-matched, reconstruction-gated) with the non-spectrally-biased decoder assumed throughout.

**Decision (2026-06-30): adopt S3.1** (dual-path residual) as the committed mechanism, with the non-spectrally-biased decoder assumed. **Complements adopted (2026-06-30):** S3.4 (reconstruction fidelity floor — pretrain the tokenizer and gate on encode→decode error below the data's discretization error, before any dynamics) and S3.3 (capacity matching via adaptive $N_q$ / intelligent patching). With S3.4 the latent-adequacy guarantee that S3.1 alone lacked is now supplied — Problem 3 is closed pending the measured floor passing on turbulent data.

### Problem 4: The continuum–particle bridge is solved only for the homogeneous case
**Status: DECIDED (fully)** — S4.1 + S4.3 + S4.4 adopted (2026-06-30) for the coupling *slot*; the cross-type message *form* is now decided too (2026-07-01): P4.3 nondimensional scale-bridge + P4.1 IBM adjoint spread/interpolate + P4.4 conserved-flux antisymmetry, with P4.2 learned-residual fallback. Implementation deferred to the S4.4 settling-particles benchmark (not exercised by the Rayleigh–Bénard field-only target), but the mechanism is fully specified. See the "Cross-type message form" subsection below.

**The gap.** The claim that the bridge is "solved at the tokenizer" holds only when a system is *all particles* OR *all field*. The hard problem is **coupled** systems — fluid–structure interaction, suspended particles, plasma — which need **heterogeneous node types in one graph with edges between a field-centroid node and a particle node** ([[graph-tokenizer]] open problem #4).

**Why it matters.** This is overselling: the headline capability is unsolved for exactly the regime that motivates it. The edge encoder for heterogeneous node types — how a field node and a particle node exchange a physically meaningful message — does not exist in the current design or in published work.

**Resolution directions.** A typed-edge encoder with per-node-type embeddings; a shared latent interface so both node types expose the same message protocol; start with a single concrete coupled benchmark (settling particles in a fluid) to force the design rather than leaving it abstract.

#### Candidate solutions (scored /10)

*Keep two sub-problems distinct: the **interface** (one encoder, one latent for both regimes — largely already the design) versus the **coupling** (heterogeneous field$\leftrightarrow$particle edges — the actual unsolved gap). Solving the interface does not solve the coupling.*

**S4.1 — Typed-edge encoder with per-node-type embeddings + shared message protocol — 8/10.**
*Description.* Give each node a type embedding (field / particle / …); the edge encoder $\phi_e$ consumes both endpoint types and emits messages in a **shared latent message space**, so a field-centroid node and a particle node exchange a physically meaningful, momentum-consistent message. Antisymmetric construction ($\mathbf m_{ij} = -\mathbf m_{ji}$) still enforces Newton's third law *across* types.
*Pros.* Directly targets the true gap; keeps one backbone; type embeddings are cheap; preserves the tokenizer-level momentum seed across heterogeneous edges.
*Cons.* Unproven in published work; making a field$\leftrightarrow$particle message *physically* meaningful (mismatched units/scales) is genuinely hard; more edge types $\to$ more parameters; needs coupled data to validate.

**S4.2 — Immersed-boundary-style coupling (physics-structured message) — 7/10.**
*Description.* Borrow the immersed boundary method: particles spread localized forcing onto surrounding field nodes and interpolate the field velocity for their own advection, via spread/interpolate operators between particle positions and field centroids — giving the message function a principled physical form instead of learning it from scratch.
*Pros.* Decades-validated fluid–particle/structure coupling; sample-efficient; a strong physical prior.
*Cons.* Bakes in a specific coupling (partly against the equation-agnostic goal); IBM assumptions (e.g. no-slip particles) may not transfer to plasma/EM coupling; ties the tokenizer to a physics choice.

**S4.3 — Single shared encoder + committed discrete type code — 8/10 (interface half).**
*Description.* One set-encoder for both regimes (already the design); carry node type as a **discrete code**, supplied when known and inferred-then-*committed* via a classifier only when unlabeled — because the decoder needs a hard type to pick its output head.
*Pros.* Cheap, exact, robust; resolves the decode-ambiguity that a soft/inferred type creates (you cannot half-decode a vortex into a particle); ~90% the current design already.
*Cons.* Solves the *interface*, not the *coupling* — S4.1 is still required; the classifier can err on genuinely ambiguous data.

**S4.4 — Concrete coupled benchmark first (settling particles in a fluid) — 6/10.**
*Description.* A methodology, not an architecture: force the coupling design against one real system before generalizing.
*Pros.* De-risks and prevents overselling; produces the data needed to validate S4.1/S4.2; cheap insurance.
*Cons.* Not itself a solution; risks overfitting the design to one coupling regime; defers the general answer.

**Recommended stack:** S4.3 (interface) + S4.1 (coupling), de-risked via S4.4; use S4.2's physical prior wherever immersed-boundary physics applies.

**Decision (2026-06-30): adopt S4.1** (typed-edge encoder) as the coupling mechanism; S4.3 (shared encoder + committed type code) is already the interface. **Complement adopted (2026-06-30):** S4.4 (concrete coupled benchmark — settling particles in a fluid — run right after the NS smoke test to force and validate the typed-edge design).

#### Cross-type message *form* — the genuine research object, now decided (scored /10)

*The gap S4.1 left open: it gives a typed-edge **slot** but not the message **form** — how a field-centroid node and a particle node exchange a *physically meaningful, dimensionally consistent, conservation-preserving* message across mismatched scales and units. This is the one genuinely novel research object in Problem 4. Scored below and decided 2026-07-01. (Not exercised by Rayleigh–Bénard, which is field-only; decided now per the "close everything" scope, implementation deferred to S4.4.)*

**P4.1 — Immersed-boundary adjoint spread/interpolate as the message kernel (promotes S4.2's physical prior to the message form) — 8/10.**
*Description.* Give the message an IBM structure: **particle→field** spreads the particle's force/momentum onto nearby field nodes via a regularized kernel $\delta_h$; **field→particle** interpolates field velocity to the particle position via the *same* kernel. The learned part is the closure coefficient; the *structure* is physical, so units are consistent by construction (spread maps force→force-density over the kernel's support; interpolate maps field→pointwise value).
*Pros.* Solves the units/scale mismatch *structurally* rather than hoping data teaches it; decades-validated for fluid–particle/structure coupling; antisymmetry (Newton's third law) is automatic because spread and interpolate are an **adjoint pair** sharing one kernel — momentum lost by the particle is exactly gained by the field.
*Cons.* Bakes in a specific coupling physics (partly against the equation-agnostic goal); IBM's assumptions (regularized-delta, no-slip particle surface) may not transfer to EM/plasma coupling; ties the message to a length-scale $h$ that must be set.

**P4.3 — Nondimensional scale-bridge: map both endpoints into shared characteristic scales before messaging (reuses S20.1) — 8/10.**
*Description.* Before any message (learned or IBM), express both endpoints in the *same* nondimensional frame using the per-region characteristic scales already adopted in S20.1 $(L_0,U_0,T_0,\rho_0)$. A particle's momentum and a field cell's momentum-*density* become commensurable once both are in shared units; then P4.1/P4.2 operate on dimensionless quantities.
*Pros.* Directly targets the crux — the "mismatched scales/units" that is *the* hard part of P4 — rather than the coupling mechanism around it; reuses S20.1's per-region scales (extend, don't fork); makes P4.1/P4.2 well-posed instead of fighting raw-unit disparity.
*Cons.* Requires the S20.1 hierarchy to span both node types in a shared region (so a particle and its surrounding field cells draw the *same* characteristic scales); a *precondition* for the other options, not a standalone message.

**P4.4 — Exchange conserved fluxes only, so antisymmetry makes coupling exactly conservative — 7/10.**
*Description.* Restrict the cross-type edge to carry a **conserved-quantity flux** (momentum, energy) computed on each side via the S1.1 readout, rather than raw latent features. Antisymmetric construction then makes the exchange exactly conservative: whatever momentum leaves the particle enters the field.
*Pros.* Guarantees the coupling *cannot violate* total momentum/energy conservation — folds directly into the S1.1 projection (the coupling is conservative by construction, not corrected after); a physically-typed flux is the *right* thing to exchange across a heterogeneous interface; reuses the S1.1 readout.
*Cons.* Restricts message expressiveness to conserved channels — misses genuinely non-conservative couplings (drag heating) unless energy is explicitly included; needs each node type to expose a momentum/energy readout at the edge.

**P4.2 — Fully-learned typed-edge message in a shared dimensionless latent (pure S4.1) — 6/10 (fallback).**
*Description.* Both node types project into a shared dimensionless message space; the message is entirely learned, keyed by the ordered type pair, with $m_{ij}=-m_{ji}$ enforced.
*Pros.* Fully equation-agnostic (best fit to core philosophy); one mechanism for *all* coupling regimes including EM/plasma where no IBM-like kernel exists.
*Cons.* The scale/unit mismatch must be learned from data with no physical prior — sample-hungry and the actual unsolved risk; no guarantee the learned message is physically meaningful. Best as the fallback where no physical kernel is known.

**Decision (2026-07-01): the cross-type message form is a nondimensionalized, conserved-flux, IBM-adjoint spread/interpolate pair with a learned residual.** Concretely: **P4.3** first (put both endpoints in shared S20.1 scales), then **P4.1** (IBM adjoint spread/interpolate as the default physical message form), constrained by **P4.4** (carry conserved fluxes so antisymmetry makes it exactly conservative and it folds into S1.1), with **P4.2** (fully-learned) as the residual/fallback for coupling regimes lacking a known physical kernel (plasma/EM). This answers the research object concretely rather than deferring it. **Implementation deferred to the S4.4 settling-particles benchmark** — Rayleigh–Bénard is field-only and does not exercise cross-type edges. **[AI Inference]:** every ingredient is a reuse — S20.1 scales, the S1.1 conserved readout, S4.2's IBM prior, S4.1's typed-edge slot — so even the "one genuine research object" in the folder turns out to be assembled from committed pieces rather than a new primitive; the only genuinely novel content is the *composition order* (nondimensionalize → conservative IBM kernel → learned residual), which is the message form itself.

### Problem 5: Deterministic core vs. chaotic physics
**Status: DECIDED (fully)** — S5.1 + S5.2 adopted (2026-06-30) for the scale-split mechanism; the guided-diffusion constraint-projection mechanism is now specified too (2026-07-01): P5.2 $x_0$-projection + P5.5 conserved-orthogonal channel + P5.3 stream-function-for-div-free, with P5.4 guidance as a cheap complement. See the "Constraint-projection mechanism" subsection below.

**The gap.** The core is deterministic regression; the diffusion corrector is "optional, off the critical path" ([[initial-model-architecture]] Stage 7).

**Why it matters.** For turbulent/chaotic systems a deterministic model provably blurs toward the conditional mean ([[latent-diffusion-physics]], [[diffusion-models-physics]]). So for the systems we most want to model, the **generative path may be load-bearing, not optional**, and it is entangled with the direct-state choice. The determinism/stochasticity fork was deferred but is central to fluids.

**Resolution directions.** Decide whether the core is deterministic-with-optional-refinement or generative-at-the-core; possibly route by regime (deterministic for laminar/smooth, generative for chaotic/turbulent) via an in-context regime classifier; treat the diffusion corrector as a candidate *core*, not a bolt-on, for chaotic regimes.

#### Candidate solutions (scored /10)

*The failure to fix is that a deterministic map collapses to the conditional mean under chaos; the right target is **correct statistics** (energy spectrum, structure functions), evaluated spectrally — not pointwise MSE past the Lyapunov horizon.*

**S5.1 — LES-style scale split: deterministic resolved core + physics-guided latent diffusion on the sub-grid residual — 8/10.**
*Description.* The transformer regresses the large, energy-containing scales; a **physics-guided latent-diffusion** model generates the chaotic sub-grid detail on the Problem-3 residual channel, projected to satisfy constraints ([[pisd-physics-informed-spectral-diffusion]]). Diffusion becomes the *core* for fine scales, not a bolt-on — the disciplined form of "predict coarse structure, let diffusion fill the (spectral) gaps." Note the reframe: the gaps are **spectral, not spatial** — the implicit decoder already fills within-patch space; what's missing is the high-$k$ content the lossy+deterministic path dropped.
*Pros.* Matches the physics (large eddies deterministic, sub-grid stochastic); cures conditional-mean blur exactly where it occurs; latent + few-step diffusion controls cost; reuses the residual channel; produces correct small-scale statistics with free ensemble uncertainty.
*Cons.* Must be constraint-projected or it hallucinates plausible-but-unphysical detail; demands spectral/statistical evaluation; conditional-diffusion training adds complexity; risk of the corrector perturbing the conserved aggregate (mitigate by confining it to the residual channel).

**S5.2 — Regime-routed hybrid (deterministic ↔ generative via in-context classifier) — 7/10.**
*Description.* An in-context regime classifier routes laminar/smooth states to the deterministic core and chaotic/turbulent states to the generative core.
*Pros.* Pays the generative cost only where needed; reuses the existing in-context inference machinery; interpretable; best-of-both.
*Cons.* The routing boundary is fuzzy (transitional flows); two paths to train and maintain; misrouting can fail silently; laminar/turbulent is not binary.

**S5.3 — Full generative core (diffusion everywhere) — 6/10.**
*Description.* Make the entire next-state predictor generative.
*Pros.* Uniform; the provably correct target for chaos; no routing; ensemble uncertainty everywhere.
*Cons.* Overkill and expensive for smooth regimes where deterministic is cheaper and adequate; harder to enforce conservation through full diffusion; slow inference (aggravates Problem 11).

**S5.4 — Deterministic core + probabilistic output head — 6/10.**
*Description.* Keep the regressor but output distribution parameters or a small ensemble per node (e.g. Gaussian/mixture), sampling for fine scales.
*Pros.* Much cheaper than diffusion; gives uncertainty; simple add-on.
*Cons.* Parametric families under-model turbulent multimodality; still blurs if the family is too simple; far less expressive than diffusion for rich small-scale structure.

**S5.5 — Spectral loss on the deterministic core (no generative path) — 4/10.**
*Description.* Keep the model deterministic but add a spectral / high-$k$ loss penalizing spectrum blurring.
*Pros.* Cheap; preserves spectral content somewhat; worth adding regardless.
*Cons.* Does not fix the root cause — a deterministic map still collapses to the conditional mean for genuinely chaotic dynamics; treats the symptom.

**Recommended stack:** S5.1 as the core mechanism, gated by S5.2 routing so smooth regimes stay cheap; S5.5 as a cheap always-on complement.

**Decision (2026-06-30): adopt S5.1** (LES scale split; diffusion generates the constrained sub-grid residual on the S3.1 channel). **Complement adopted (2026-06-30):** S5.2 (regime routing — an in-context classifier reading trajectory statistics, the same mechanism that infers Re, routes smooth→deterministic and chaotic→generative so diffusion cost is paid only where determinism fails).

#### Constraint-projection mechanism — how generated detail is made div-free/conservative, now decided (scored /10)

*The gap S5.1 left "to spec": an unconstrained diffusion corrector hallucinates plausible-but-unphysical sub-grid detail — violating incompressibility, conservation, and BCs. How is the **generated** detail constrained? Rayleigh–Bénard makes this live (unlike Burgers, where diffusion was deferred): at high Rayleigh number RBC is turbulent, so per S5's own logic the generative path becomes **load-bearing**, and its output must be div-free (incompressible), heat-conserving, and BC-respecting. Scored below, decided 2026-07-01.*

**P5.2 — $x_0$-projection: project the model's *clean-sample estimate* at each denoising step, not the noisy sample — 9/10.**
*Description.* At each reverse-diffusion step, form the Tweedie clean-sample estimate $\hat x_0$, project *that* onto the constraint manifold (S1.1 conserved totals, P2.2 BCs, div-free), then re-noise for the next step. The final step yields a constraint-satisfying sample. Reuses the exact P2/S1.1 projection machinery.
*Pros.* Projects the **dimensionally correct object** — the constraint applies to clean data, and $\hat x_0$ is a clean-data estimate, unlike the naive "project the noisy iterate" which mixes a clean-data manifold with a noised sample (P5.1's flaw); reuses the projection stack already built; standard in constrained/guided diffusion (ΠGDM / DPS-with-projection); guarantees the final sample satisfies constraints.
*Cons.* Pays a projection per denoising step (the elliptic part aggravates Problem 11 — mitigated by P5.3 below removing the Leray solve); $\hat x_0$ is inaccurate early in sampling, so early projections are weak (harmless — they tighten as sampling proceeds); the quadratic energy projection inside the loop adds mild nonlinearity.

**P5.5 — Confine diffusion to the conserved-*orthogonal* residual channel, so conservation needs no in-loop projection — 8/10.**
*Description.* Architecturally restrict the diffusion output to the high-$k$ residual channel (S3.1) constructed to be **orthogonal to the conserved functionals** (which live in the coarse/global modes the deterministic core owns). Then generated detail *cannot* perturb conserved totals by construction — exactly the LES property that sub-grid stresses carry zero net resolved momentum/heat.
*Pros.* Makes conservation-preservation **structural**, not corrected — the sub-grid channel carries zero net mass/momentum/heat by design, so no per-step conservation projection is needed at all (only div-free + BC remain for the projection); the cheapest possible conservation guarantee; matches the physical LES decomposition exactly; protects Problem 1's conserved aggregate from the corrector (the S5.1 con flagged).
*Cons.* Requires the S3.1 residual basis to be genuinely orthogonal to the conserved functionals — a real design constraint on the dual-path encoder (the smooth path must own the conserved modes); div-free and BCs still need P5.2/P5.3.

**P5.3 — Generate the div-free part as a stream function / vector potential (reuses P2.4), removing the per-step Leray solve — 8/10.**
*Description.* Have the diffusion model generate the incompressible sub-grid velocity detail as a stream function $\psi$ (2D) so $u'=\nabla\times\psi$ is div-free by construction — no Poisson solve inside the sampling loop.
*Pros.* Incompressibility exact and **free** during sampling (kills the per-step Leray-solve cost that would otherwise make P5.2 expensive — the single biggest inference-cost win here); reuses P2.4; the residual channel is exactly where a potential representation is cleanest; ideal for RBC's incompressible velocity fluctuations.
*Cons.* Handles div-free *only* (conservation via P5.5, BCs/temperature via P5.2 still needed); 2D-clean, 3D vector potential carries gauge freedom; changes what the velocity residual channel stores (a potential increment, not a velocity increment) — must reconcile with S3.1's representation.

**P5.4 — Constraint-gradient guidance added to the diffusion score (DPS-style) — 5/10 (cheap complement).**
*Description.* Steer sampling with a $\nabla_x\|\text{constraint violation}\|^2$ term added to the score, no hard projection.
*Pros.* Cheap, differentiable, no solve in the loop; a useful always-on nudge that keeps intermediate samples near-admissible so the final projection has less to correct.
*Cons.* Soft — no exact guarantee (the "conditioned not enforced" failure mode); fragile guidance weight; the project consistently rejects soft-only as *sole* mechanism (S1.4, S6.6, S25.5, P9.5). Complement, not mechanism.

**Decision (2026-07-01): the generated sub-grid detail is constrained by construction wherever possible, projected where not.** Concretely, three composed pieces: **P5.5** (confine diffusion to the conserved-orthogonal channel → conservation is free, no in-loop projection), **P5.3** (generate the div-free part as a stream function → incompressibility is free, no Leray solve in the loop), and **P5.2** ($x_0$-projection via the shared P2.2 machinery each step, and finally, for the *remaining* constraints — BCs and the temperature field's admissibility). **P5.4 guidance** kept as a cheap always-on complement to keep iterates near-admissible; **rejected as sole mechanism.** This closes S5.1's "still to spec." Net: the corrector can only ever write to a channel that (i) can't touch conserved totals (P5.5) and (ii) is div-free by construction (P5.3), with a light per-step projection (P5.2) for BCs — so the expensive elliptic solve is avoided entirely inside the sampling loop. **RBC note:** the generative path activates only for turbulent (high-Ra) RBC via S5.2 routing; a first laminar/steady-convection RBC pass can defer diffusion exactly as Burgers did ([[noether-1.0]] §3.9). **[AI Inference]:** the pattern mirrors P25's "reparameterize for admissibility-by-construction, project only the residual" — both prefer baking the constraint into *what is generated* over correcting *what was generated*, and both fall back to projection only for constraints with no by-construction form; the diffusion corrector and the deterministic head thus share one constraint philosophy, not two.

**[AI Inference]:** Problems 1, 3, and 5 compose into one coherent three-tier stance rather than three independent patches — **enforce** the global conserved/constrained totals (S1.1), **regress** the resolved scales (the existing backbone), **generate** the constrained sub-grid detail (S5.1) — all scaffolded by the single supernode hierarchy (global node / mid levels / residual channel). This is the concrete realization of the page's closing claim that settling *enforce vs. learn* collapses most of Tier 1 into one decision.

### Problem 12: No per-node nonlinear computation — attention alone risks rank collapse and lacks constitutive-relation capacity
**Status: DECIDED** — F1 adopted, F3 as mandatory ablation, F2 as the equivariance upgrade path (2026-07-01).

**The gap.** Surfaced during Problem 6 implementation planning: the backbone equation ([[initial-model-architecture]] Stage 4) has exactly one per-node nonlinearity — a single elementwise activation $\sigma$ wrapping the sum of the linear self-term $(W-W^\top-\gamma I)h_i$ and the linear neighbor-aggregation term $\sum_j A_{ij}(V_j-V_i)$. There is no feedforward (FFN/MLP) sub-layer — no hidden-dimension expansion, no independent per-node nonlinear feature transform of the kind every standard transformer block includes alongside attention.

**Why it matters.** Attention is fundamentally a **communication** mechanism — a (nonlinearly-weighted) linear recombination of value vectors *across* tokens/nodes. The **computation** — combining a node's own features in new nonlinear ways — is standardly the FFN's job, and empirically holds most of a transformer's representational capacity (FFNs read/write the residual stream's "knowledge"; attention only moves information between positions). Two concrete risks follow:
- **Rank collapse / over-smoothing.** Stacks of attention-only layers (no FFN, or an FFN too weak) are known to lose rank with depth — token/node representations drift toward a shared, low-rank subspace, destroying the node-local distinctions the model needs (the physics failure mode: neighboring fluid/particle nodes becoming indistinguishable deep in the backbone).
- **Constitutive-relation capacity.** Multiphysics is full of *local, nonlinear, per-point* relations — equations of state, non-Newtonian stress laws, reaction rates, turbulence closures. A single elementwise activation after a linear combination cannot approximate these nearly as well as a wide two-layer MLP (universal approximation needs the hidden-layer width); this directly threatens the multiphysics-breadth commitment (#1), not just an efficiency nicety.

**The complication.** A generic (unconstrained) FFN is not a free fix: it doesn't touch cross-node interactions, so it does *not* reintroduce the $W_O$-style conservation problem flagged in [[multihead-attention]]. But it *can* silently break the skew-symmetric channel's norm-preservation story ([[normalization-scheme]]) that Problem 1's conservation claim depends on — an arbitrary FFN amplifies/shrinks node norms uncontrollably — and, for vector/tensor-valued channels (velocity components etc.), a plain per-channel FFN breaks rotation equivariance by mixing components arbitrarily.

**Resolution directions.** A norm-bounded per-node nonlinear block (semi-orthogonal expand/contract), so it earns FFN-style capacity without unbounded amplitude drift; or, if equivariance (Problem 9) is committed, an equivariant gated nonlinearity that gives the scalar/invariant sub-channels full FFN width while only ever *rescaling* (never rotating) the vector/tensor channels.

#### Candidate solutions (scored /10)

**F1 — Semi-orthogonal (Stiefel-constrained) expand→contract FFN, norm-bounded — 8/10.**
*Description.* A second Euler sub-step per layer, after the attention sub-step: $h_i \mathrel{+{=}} \epsilon\,\mathrm{FFN}(h_i)$ with $\mathrm{FFN}(x) = W_2\,\phi(W_1 x)$, where $W_1\in\mathbb R^{d_{ff}\times d}$ satisfies $W_1^\top W_1 = I_d$ (an isometric embedding into the wider hidden space, so $\|W_1x\|=\|x\|$ even though $d_{ff}>d$), $\phi$ is an elementwise nonlinearity applied then rescaled to preserve pre-activation norm, and $W_2\in\mathbb R^{d\times d_{ff}}$ satisfies $W_2W_2^\top=I_d$ (non-expansive: $\|W_2z\|\le\|z\|$). Net effect: a wide, expressive, per-node nonlinear map whose output norm is *bounded* by the input norm (not exactly preserved, but controllably so — no unbounded drift).
*Pros.* Gives real FFN-style capacity (hidden-width nonlinear feature mixing) while keeping amplitude under analytic control, protecting the Problem-1 conservation story without waiting on an equivariance decision; fits naturally as a second sub-step alongside the existing attention Euler step, so the validated attention equation is untouched.
*Cons.* Semi-orthogonal constraint costs some raw expressivity vs. an unconstrained FFN; only *bounds* (doesn't exactly preserve) norm through the nonlinearity, so the S1.1 projection is still needed as the authoritative fix, not this; adds a Stiefel-manifold parameterization to implement and optimize.

**F2 — Equivariant gated nonlinearity as the FFN (e3nn-style) — 7/10.**
*Description.* For vector/tensor channels, compute rotation-invariant scalars (norms) as a side channel; run a **freely wide, unconstrained** MLP on the invariant scalars (no equivariance constraint applies to scalars); use its output as multiplicative gates that rescale — never rotate — the original vector/tensor channels. Exactly preserves equivariance while giving full FFN width on the invariant sub-space; already the natural extension of the merged-norm idea cited in [[normalization-scheme]] and the irrep heads in [[multihead-attention]].
*Pros.* Exact equivariance preservation (not just approximate norm-boundedness); reuses machinery already gestured at elsewhere in the design; the right long-term answer if Problem 9 resolves to hard equivariance.
*Cons.* Only pays off once irrep-typed (vector/tensor) features are actually in play; more machinery than the initial smoke test needs if Problem 9's default stays "no hard equivariance" (see Problem 9 below); tensor-product bookkeeping.

**F3 — No FFN; rely on $\sigma$ only (current spec, as baseline) — 4/10.**
*Description.* Ship as currently specified — single elementwise activation, no hidden-width expansion.
*Pros.* Zero new machinery; cheapest; **necessary as an ablation control** regardless of which fix is adopted, to empirically justify that F1/F2 are earning their cost.
*Cons.* Exposed to rank collapse with depth; likely under-fits local constitutive nonlinearities central to the multiphysics-breadth claim; should not be the shipped default.

**F4 — Unconstrained dense FFN (standard transformer block) — 3/10.**
*Description.* Ordinary $W_2\,\phi(W_1x+b_1)+b_2$ with no norm or equivariance constraint.
*Pros.* Maximum raw expressivity; simplest to implement; cheapest to train; **[AI Inference]:** survivable *for conservation specifically* because the downstream S1.1/P2.2 projection ([[pipeline-contract]] Stage 7/10) re-projects onto the conserved manifold regardless of what the FFN did to amplitude — the projection sandwich is a safety net here.
*Cons.* The projection sandwich rescues *conservation*, not *equivariance* — those are different failure modes with different remedies, and an unconstrained FFN still silently breaks equivariance if vector/tensor channels are present; leaves the amplitude free to swing wildly pre-projection, which can still destabilize training (Problem 10) even if the final output is corrected.

**Decision (2026-07-01): adopt F1** for the initial smoke-test model (norm-bounded semi-orthogonal FFN sub-step), with **F3 retained as a mandatory ablation** to empirically justify the addition. **F2 is the designated upgrade path** if/when Problem 9 resolves to hard equivariance — F1 and F2 are not mutually exclusive; F2 can absorb F1's role on the invariant sub-channels once irrep-typed features exist. **[AI Inference]:** this closes the loop with Problem 1 — the FFN sub-step is *also* a magnitude-affecting operation, so it inherits the [[pipeline-contract]] ordering invariant (no uncontrolled amplitude changes after the final conservation projection); F1's bound keeps it compatible with that invariant by construction rather than by correction.

---

## Tier 2 — real but normal specification gaps

### Problem 6: The loss function is unspecified beyond "residual regression"
**Status: DECIDED** — S6.1 + S6.3 + S6.4 + S6.5 adopted (2026-07-01); full spec in [[training-curriculum]].
Needs per-field weighting (fields differ by orders of magnitude), a **spectral loss** (or high-$k$ structure blurs out — a known emulator failure), and a defined rollout-horizon schedule for push-forward training. Conservation penalties likely needed *because* Problem 1 leaves conservation not fully architectural.

#### Candidate solutions (scored /10)

*Delta from the paragraph above: it predates the S1.1 decision. Tier-1 **known** invariants (mass/momentum/energy) are now enforced by the architectural projection ([[pipeline-contract]] Stage 7/10), not a loss term — they need no weight. Only tier-2 **discovered** invariants still need a loss. The objective below is organized into four dependency-ordered groups; full spec (phase schedule, exact loss forms) lives in the companion page [[training-curriculum]].*

**S6.1 — Homoscedastic per-channel uncertainty weighting (Kendall-style) for the core state-fit terms — 8/10.**
*Description.* Learn $\log\sigma_c^2$ per field channel $c$ (and per domain: physical-space residual vs. spectral loss); weight $=1/(2\sigma_c^2)$ with a $\log\sigma_c$ regularizer preventing collapse to zero weight. Automatically balances velocity vs. pressure vs. whatever field set a given trajectory carries.
*Pros.* Principled, automatic, well-established; directly answers "fields differ by orders of magnitude"; scales to arbitrary field sets without redesign — fits the breadth-from-day-one commitment better than hand tuning.
*Cons.* Noisy early in training, needs warmup; only balances at the channel level (homoscedastic), not spatially/temporally within a channel.

**S6.2 — GradNorm dynamic gradient-magnitude balancing across loss groups — 7/10.**
*Description.* Rebalance group-level weights during training so each group's gradient norm (at a shared reference layer) tracks a common target rate.
*Pros.* Handles cross-*group* imbalance (residual vs. spectral vs. discovery vs. denoising) that within-group uncertainty weighting doesn't reach; useful when terms live in genuinely different spaces.
*Cons.* Extra backward passes; more hyperparameters; heavier to debug — better as a diagnostic/fallback than the default mechanism.

**S6.3 — Staged curriculum: introduce loss groups in dependency order, not jointly from step 0 — 8/10.**
*Description.* Core state fit first; add multi-step/MTP consistency once single-step is stable; add conservation-discovery once trajectories are meaningful to probe; add generative denoising once the backbone/latent is trustworthy (full phase table in [[training-curriculum]]).
*Pros.* Matches the actual dependency structure (MTP is meaningless before single-step is decent; discovery needs real trajectories; denoising needs a trustworthy residual channel); avoids destabilizing a nonstandard backbone with five competing objectives from random init (entangled with Problem 10).
*Cons.* Adds a scheduling hyperparameter (when to switch phases); risk of a shock to a converged region when a new group turns on.

**S6.4 — Slow-Feature-Analysis-style loss for tier-2 conservation discovery — 7/10.**
*Description.* **[AI Inference]:** treat "discover a hidden invariant" as a slowness-learning problem — minimize a candidate readout's *within-trajectory* variance over time while maximizing its *across-trajectory* variance (so it can't collapse to a trivial global constant), the classic Slow Feature Analysis framing, extended with the non-degeneracy term the second-pass audit (§B.4) flagged as missing.
*Pros.* A concrete, implementable answer to "does the model help pick invariants?"; borrows an established technique rather than inventing one; naturally yields *multiple* candidate invariants (top-$k$ slowest/most-discriminative probes), matching how real systems (2D turbulence: enstrophy; MHD: helicity) carry more than one hidden invariant.
*Cons.* Statistical, not certified — still needs the promotion-to-hard-projection validation step from S1.1; needs long/diverse trajectories to separate true invariants from merely-slow modes.

**S6.5 — Two-phase training: converge the backbone first, then train the diffusion head on the (lightly fine-tuned) latent — 8/10.**
*Description.* Mirrors standard latent-diffusion practice — train the deterministic backbone to convergence, then train the generative head in its latent space — rather than jointly training a regression head and a stochastic generative head from scratch on a shared, still-shifting representation.
*Pros.* Avoids the known instability of jointly optimizing deterministic + generative objectives on shared representations; decouples two very different optimization dynamics; the residual channel only needs to be "good enough to generate into" once phase 1 provides it.
*Cons.* Two training phases to manage; a fully frozen backbone can't be corrected by the diffusion head's gradients (mitigate with light fine-tuning, not a hard freeze).

**S6.6 — Fixed, hand-tuned loss weights — 4/10.**
*Description.* Constant scalar weights, tuned by hand on the smoke-test problem.
*Pros.* Simplest possible baseline; fine for a single-PDE smoke test.
*Cons.* Brittle and directly conflicts with commitment #1 (breadth) — weights tuned for one field set don't transfer to a new field type/regime without re-tuning.

**Decision (2026-07-01): adopt S6.1 + S6.3 + S6.4 + S6.5** (within-group uncertainty weighting, staged cross-group curriculum, SFA-style discovery loss, phase-separated diffusion training); **S6.2 held in reserve** as a diagnostic if staged introduction alone doesn't balance groups sufficiently; **S6.6 rejected as the sole mechanism** but implicitly present as each phase's initial constant before its within-group weighting stabilizes. Full phase schedule and loss forms: [[training-curriculum]].

### Problem 7: Dynamic graph topology over rollout
**Status: DECIDED (fully)** — adaptive-patching sub-case resolved via [[noether-1.0]] §3.2/§3.7 (2026-07-01); persistent-node-identity sub-case now decided too (2026-07-01): P7.1 persistent-nodes/ephemeral-edges + P7.2 particle-sum anchoring + P7.4 smooth cutoff. Implementation deferred to the S4.4 settling-particles benchmark (not exercised by Rayleigh–Bénard). See the "Persistent-identity sub-case" subsection below. Particle creation/destruction (fragmentation) flagged as a narrow remaining edge case.
Particles move and adaptive patches re-refine ([[intelligent-patching]]), so the graph **changes every step**: neighbor-list rebuild cost, batching of variable-topology graphs, and — critically — whether antisymmetric-edge momentum conservation survives edges *appearing and disappearing* between steps (edge churn can inject/remove momentum, feeding back into Problem 1).

**Partial resolution (2026-07-01), adaptive-patching sub-case:** [[noether-1.0]] §3.2/§3.7 turns this from a deferred concern into an active, tested design on the 1D Burgers benchmark. Two properties are argued (and will be empirically checked) to make momentum/mass conservation robust to this specific kind of topology churn: (1) the graph is **ephemeral** — rebuilt fresh each step from the canonical physical state, never persisted or matched across steps, so there is no node identity for churn to corrupt; (2) global conserved-quantity checks are anchored to a **fixed canonical quadrature** (the underlying grid points), never to the current adaptive node partition, so an unweighted average over variable-width patches — which would be wrong — never enters the computation. **This does not resolve the harder sub-case:** a *persistent-identity* topology change (e.g. moving particles, where "particle 5" must remain particle 5 across steps rather than being freshly re-derived) has no such escape hatch — resolved below.

#### Persistent-identity sub-case (moving particles) — now decided (scored /10)

*The hard sub-case: Lagrangian particles carry genuine, physically-meaningful identity across steps (particle 5 stays particle 5), so "re-derive everything fresh" — the adaptive-patching escape hatch — is unavailable. Edge churn (neighbors entering/leaving the cutoff) can then inject or remove momentum, feeding back into Problem 1's conservation. Scored below, decided 2026-07-01. Not exercised by field-only Rayleigh–Bénard; decided now per the "close everything" scope, implementation deferred to S4.4.*

**P7.1 — Persistent identity on *nodes*, disposability on *edges*: split what must persist from what can be rebuilt — 9/10.**
*Description.* Decompose the problem. Particle **nodes** carry persistent identity + state (position, velocity, mass), advected across steps by a light integrator. The **edges** (neighbor lists) are still rebuilt fresh every step — ephemeral, exactly like the field graph. Momentum lives on *nodes*, and cross-node exchange is an **antisymmetric conserved-flux** message (P4.4), so however the neighbor list changes, each pairwise message conserves momentum and edge churn merely *moves* momentum between existing particles, never creating or destroying it.
*Pros.* Cleanly separates persistent (identity/state) from disposable (topology) — the right decomposition, and it **dissolves the churn worry**: because momentum is stored on nodes and exchanged by antisymmetric pairwise flux, appearing/disappearing *edges* have no momentum to inject (there is no per-edge momentum store); reuses the validated GNS / [[dynami-cal-graphnet]] pattern (persistent particles, rebuilt neighbor graphs); the antisymmetric-flux message is exactly P4.4, already decided.
*Cons.* Advecting particle state across steps reintroduces a *localized* integrator (mild tension with commitment #2, but confined to particles, like S1.2's partial-integrator stance); per-step neighbor-list rebuild cost (Problem 11) — standard for particle methods, not new.

**P7.2 — Anchor global conservation to the particle-*sum*, not the topology (generalizes §3.7's canonical-anchoring to Lagrangian) — 8/10.**
*Description.* Just as the field case anchors conservation to the fixed canonical grid, the particle case anchors to the **fixed total over all persistent particles** $\sum_p m_p v_p$ — a quantity indifferent to which edges exist — and projects (S1.1) onto that total each step.
*Pros.* Directly generalizes the *already-decided* §3.7 anchoring recipe (the problem's own resolution pattern) to particles — the anchor "sum over the persistent particle set" is well-defined *precisely because* identity persists; edge churn cannot perturb a node-sum; reuses S1.1.
*Cons.* Pins only the global total (the standard S1.1 weakness — globally exact, locally free); assumes a fixed particle cardinality — creation/destruction (fragmentation) is a further edge case (flagged below).

**P7.4 — Message on *relative* quantities with a smooth radial cutoff, so edge birth/death is continuous — 7/10.**
*Description.* Make every edge message depend only on relative position/velocity ($x_i-x_j$, $v_i-v_j$) through a smooth radial cutoff kernel that →0 at the neighbor radius, so a particle crossing the cutoff contributes a message that vanishes *smoothly* rather than switching on/off discontinuously.
*Pros.* Kills the *discontinuity* at edge appearance/disappearance — the specific mechanism by which churn injects momentum — a clean, standard GNN trick (SchNet/PaiNN smooth cutoffs); translation/Galilean-invariant by construction; composes with P7.1.
*Cons.* A smooth cutoff is a *continuity* mitigation, not a hard conservation guarantee on its own (needs P7.1's antisymmetry for that); slightly reduces effective interaction range near the cutoff.

**P7.3 — Soft cross-step re-identification (optimal-transport / Hungarian matching) — 4/10 (rejected).**
*Description.* Instead of hard persistent IDs, softly match nodes between steps by feature/position similarity.
*Pros.* Handles ambiguous identity or merge/split cases.
*Cons.* Soft-matching errors accumulate over rollout; expensive; *unnecessary* when true IDs exist (a particle simulation knows which particle is which) — reintroduces exactly the cross-step-matching instability the ephemeral-graph design was built to avoid.

**Decision (2026-07-01): adopt P7.1 + P7.2 + P7.4.** Persistent identity lives on nodes with ephemeral rebuilt edges (P7.1); momentum is exchanged by antisymmetric conserved flux so churn cannot inject it (P7.1 × P4.4); edge messages use relative quantities with a smooth radial cutoff so churn is *continuous* (P7.4); and global conservation is anchored to the fixed particle-sum, generalizing §3.7 (P7.2). **P7.3 rejected** — soft re-identification is strictly worse than using the real IDs a particle system already has. **Remaining narrow open case:** particle *creation/destruction* (fragmentation, phase change) changes the particle-sum's cardinality and needs a source/sink term in P7.2's anchor — flagged, not solved, since no near-term benchmark requires it. **Implementation deferred to the S4.4 settling-particles benchmark.** **[AI Inference]:** the unifying insight across *both* sub-cases of Problem 7 is that "dynamic topology" was never the real risk — the risk was anything *load-bearing* being keyed to a specific topology. The adaptive-patching case escaped by making *everything* disposable (re-derive fresh); the particle case can't do that (identity is physical), so it instead makes the *load-bearing* quantity (momentum) live on the persistent objects (nodes) and flow through the disposable ones (edges) only as antisymmetric conserved flux — two different routes to the same principle: *never key conservation to something that churns.*

### Problem 8: Variable field cardinality per node
**Status: DECIDED** — S8.1 + S8.5 adopted, S8.4 as the zero-shot upgrade path (2026-07-01). This was the last fully-open problem and the #1 blocker to multi-field training; it is resolved here for the **Rayleigh–Bénard convection** benchmark (velocity vector + temperature scalar + pressure), the first multiphysics target.
How one node latent represents 2 vs. 5 fields, and how a field type never seen in training is handled. We gestured at AION-1 modality-specific tokenization ([[scalar-cdf-tokens]], [[vector-quantized-tokens]]) but never specified the mechanism for heterogeneous, variable field sets — core to the multiphysics-breadth claim.

**Why it now bites (and why Burgers dodged it).** Burgers carries exactly one field ($u$), so [[noether-1.0]] never had to answer this — its encoder $d_u=1$ is hard-wired. Rayleigh–Bénard carries a **velocity vector** ($u_x,u_y$), a **temperature scalar** $T$, and a diagnostic **pressure** $p$, coupled by Boussinesq buoyancy ($+\beta g\,T\,\hat z$ in the momentum equation) and thermal advection ($u\cdot\nabla T$). The encoder must map a *variable-arity, heterogeneous-unit, irrep-mixed* field set into the single $d$-vector the backbone consumes — and must not silently break when a later system adds a magnetic field or a species concentration it never saw.

**The two sub-requirements, kept distinct.** (a) **Cardinality invariance** — the same architecture ingests 1, 2, or 5 fields without reshaping weights. (b) **Open-vocabulary generalization** — a field *type* absent from training is still processable. (a) is an engineering requirement; (b) is the genuinely hard, foundation-model-defining one.

#### Candidate solutions (scored /10)

**S8.1 — Per-field-type typed encoder → shared *additive* node latent, keyed by the P16 irrep/role tag — 9/10.**
*Description.* Each physical channel carries a field-type code (already in [[pfm-interface-design]], extended with the irrep tag from S16.1 / S17.1). A small per-type encoder maps that channel's raw value(s) into the shared $\mathbb R^d$ latent; the node latent is the **sum over present fields** of (typed-encoder output $+$ a learned field-type embedding). Absent fields contribute nothing; cardinality invariance is automatic (a sum over a variable index set). An **unseen** field type falls through to a **generic by-arity encoder** selected by its irrep tag (scalar / vector / rank-2 tensor, S17.1) — a never-seen scalar is still encoded by the shared scalar pathway, just without a type-specific embedding. Decode mirrors this: per-field-type decoder heads read the shared latent and each emits its own field, run only for the fields actually requested (variable-cardinality output = run a variable subset of heads).
*Pros.* Reuses machinery already committed to — the conditioning interface (S16.1/S24.1) and the irrep taxonomy (S17.1) — rather than a new component (the recurring "extend, don't fork" pattern); additive aggregation is *exactly* Deep Sets / MPP-style field embedding, a validated design; cardinality invariance and open-vocabulary fallback drop out of the same construction; the type embedding disambiguates which summand is which, so superposition doesn't lose field identity.
*Cons.* Additive superposition of many fields into one $d$-vector risks capacity collision when the field count is large (mitigated by $d$ sizing and the type embeddings acting as near-orthogonal keys); the fallback encoder gives a *structurally valid* but *semantically blind* embedding for a truly novel field (it knows "this is a vector" but nothing field-specific) — real zero-shot quality needs S8.4; requires the decode-side head set to be extensible as new fields are added.

**S8.5 — Universal per-field normalization (CDF/quantile transform, AION-style) as the required commensurability complement — 8/10.**
*Description.* Before S8.1's typed encoder, push each channel through a per-field-type normalizing transform (learned CDF / quantile map, or the [[normalization-scheme]]'s nondimensionalization where characteristic scales exist) so temperature-in-Kelvin and velocity-in-m/s land in a **common normalized token range** regardless of native units/scale ([[scalar-cdf-tokens]]). Vector/tensor fields are normalized component-wise within their irrep block so the transform doesn't break equivariance.
*Pros.* Directly kills the "fields differ by orders of magnitude" heterogeneity that would otherwise make the additive sum in S8.1 dominated by whichever field has the largest raw magnitude; established (AION-1); makes disparate fields genuinely commensurable, which is the precondition for one shared encoder to work at all; composes cleanly with the S6.1 per-channel loss weighting (normalize on the way in, weight on the way out).
*Cons.* The CDF/quantile map needs per-field statistics (a fitting pass, or online estimation) — a mild pipeline addition; a purely statistical normalization can distort a field with a physically meaningful zero (temperature deviations) unless anchored; it is a *complement to*, not a *substitute for*, S8.1 — it solves scale, not cardinality.

**S8.2 — Fields-as-tokens: each (location × field) is its own token, cross-field coupling via attention — 6/10.**
*Description.* Rather than merging all fields into one node latent, emit one token per field per location; the backbone attends across both space and field, so buoyancy coupling ($u_y \leftarrow T$) emerges as learned cross-field attention.
*Pros.* Cleanest possible cardinality handling (more fields = more tokens, nothing reshapes); zero superposition interference; cross-field physics is explicit and inspectable.
*Cons.* **Forks the one-node-per-location invariant the entire architecture rests on** — the supernode hierarchy, radius graph, S1.1 projection, and §3.7's canonical-grid anchoring all assume a single node per spatial site; multiplying token count by field count directly aggravates Problems 11/21 (cost) and would force the hierarchy to be rebuilt per-field. A genuine architectural fork, not an extension — fails Tier 3's "extend, don't fork" test.

**S8.3 — Fixed maximal field superset with presence masking — 4/10.**
*Description.* Predefine a maximal channel set (velocity, pressure, temperature, density, $B$-field, …); every node carries a fixed-width raw slot vector with absent fields zero-masked plus a presence bitmask.
*Pros.* Trivial static shapes; easy batching; the standard pad-and-mask trick.
*Cons.* **Cannot handle a genuinely unseen field type** — the headline requirement — without retraining to widen the superset; wastes capacity on absent fields; the "maximal set" is a closed vocabulary that directly contradicts open-ended breadth (commitment #1); scales badly as the field zoo grows.

**S8.4 — Field-type hypernetwork: generate the encoder/decoder weights from a field descriptor — 7/10 (designated upgrade path).**
*Description.* A hypernetwork consumes a field's *semantic descriptor* (irrep tag + physical role + governing-quantity metadata: "conserved density", "intensive scalar", "flux") and emits the encoder projection for that field. A novel field arriving with a descriptor gets a **zero-shot generated encoder**, no retraining.
*Pros.* The strongest answer to sub-requirement (b) — true open-vocabulary generalization, which S8.1's fallback only approximates; the right long-term mechanism for a genuine physics *foundation* model that meets fields it was never trained on; the descriptor is exactly the metadata S16.1/S17.1 already maintain.
*Cons.* Hypernetworks are notoriously unstable and sample-hungry to train — overkill and risky for the first multiphysics step; zero-shot quality is only as good as the descriptor's expressiveness (a novel field with a poor descriptor still fails); heavier machinery than Rayleigh–Bénard's three known fields need.

**Decision (2026-07-01): adopt S8.1 + S8.5** — a per-field-type typed encoder summing into a shared, irrep-tagged node latent (S8.1), fed by universal per-field normalization for scale-commensurability (S8.5). Together they deliver cardinality invariance *and* a structurally-valid unseen-field fallback while reusing the existing conditioning/irrep machinery. **S8.4 (hypernetwork) is the designated upgrade path** for aggressive zero-shot to truly novel fields, adopted once open-vocabulary generalization is a measured need rather than a hypothetical one — the same staged posture used for equivariance (P9) and sparse attention (P21). **S8.2 rejected** as a fork of the one-node-per-location invariant every downstream component depends on; **S8.3 rejected** as a closed vocabulary incompatible with the breadth commitment. **[AI Inference]:** S8.1's additive-typed-sum is the field-space analogue of the *spatial* supernode aggregation already trusted throughout — both are permutation/cardinality-invariant sums over a variable index set disambiguated by learned keys (type embeddings for fields, tree position for space), which is why the same design instinct that made the hierarchy work should make heterogeneous field sets work.

**Rayleigh–Bénard instantiation.** Field-type codes $\{$velocity (vector, $L{=}1$), temperature (scalar, $L{=}0$)$\}$; pressure is decoded diagnostically from the S18/P2.1 pressure-projection, not a primary encoded field. Node latent $\tilde h_i = E_{\text{vec}}(u_i)+e_{\text{vel}} + E_{\text{sc}}(T_i)+e_{\text{temp}} + z_{\text{param}} + z_{\Delta t}$, each field pre-normalized by S8.5, with buoyancy entering as a covariant conditioning vector $\hat g$ (see Problem 9's P9.2). Positivity of any bounded field (not $T$, which is signed here, but density/concentration in later systems) is handled at decode by S25.1 reparameterization.

### Problem 9: Equivariance — commit or not
**Status: DECIDED** — full scored pass complete (2026-07-01): the dilemma is *dissolved* by P9.2 (symmetry-breaking fields become covariant conditioning), realized via P9.3 (channel-typed selective equivariance) and gated by the S15.2 ablation; P9.1 augmentation is the working baseline for the first pass.
Referenced as an "option" throughout ([[structure-preserving-tokens]], [[multihead-attention]]) but never decided. It is a major architecture choice: large sample-efficiency / OOD gain ([[equivariant-gnns]]) vs. real cost ([[equiformer-v3]] tensor products). Leaving it undecided leaves the backbone underspecified.

**Prior provisional default (2026-07-01, superseded below):** no hard equivariance; data augmentation as a soft proxy. This was justified *specifically* because 1D Burgers has no meaningful rotation group. The move to 2D Rayleigh–Bénard forces the full pass — but with a twist the provisional framing missed (below).

**The Rayleigh–Bénard subtlety that reframes the whole problem.** RBC is **not** $SO(2)$-invariant: gravity picks out a vertical direction, and the buoyancy term $+\beta g\,T\,\hat z$ is a fixed external vector that breaks full rotational symmetry. So "make the backbone $SO(2)$-equivariant" would be enforcing a symmetry the *physics does not have* — actively wrong, not merely expensive. This dissolves the original "commit or not" dilemma: the right question is not *whether* to be equivariant but *to which group*, and the answer is **the full group acting jointly on the state and the symmetry-breaking external fields** — under which RBC genuinely *is* invariant (rotate the fields *and* gravity together and the physics is unchanged).

#### Candidate solutions (scored /10)

**P9.2 — Symmetry-breaking fields as covariant conditioning: build equivariance to the full group acting jointly on state + external vectors — 9/10.**
*Description.* Feed every symmetry-breaking external field as an explicit **covariant vector input** — gravity $\hat g$ for buoyancy, frame angular velocity $\Omega$ (already S24.1) for rotation, background $\mathbf E/\mathbf B$ for EM — and construct the backbone equivariant to $SO(2)/SO(3)$ acting simultaneously on the physical fields *and* these conditioning vectors. Then anisotropic systems (RBC, turbomachinery, magnetized plasma) are handled by *one* equivariant architecture, distinguished only by which external vectors are conditioned on.
*Pros.* The **physically correct** framing — restores *exact* equivariance by making the broken direction an explicit covariant input, so equivariance is never "wrong physics" again; unifies isotropic and anisotropic systems under one backbone (the difference is data, not architecture); it is the exact generalization of a pattern the project *already adopted* — S24.1 feeds $\Omega$ as a covariant conditioning vector for Coriolis, and gravity is identical in kind; matches the founding "the backbone learns the force given context" philosophy.
*Cons.* Requires committing to equivariant machinery (irrep typing P16/P17, tensor products) with its $O(L_{\max}^4)$/edge cost — though S17.2's $L_{\max}\le2$ cap bounds it; the covariant-conditioning path must be threaded through the interface (a clean extension of S16.1/S24.1, but real work); more than the *first* RBC pass strictly needs if a plain augmented baseline already fits.

**P9.3 — Selective/channel-typed equivariance: F2 gated nonlinearity on vector/tensor channels, F1 on scalars (= S15.3) — 8/10.**
*Description.* Once P16 tags channels by irrep, apply F2's equivariant gated treatment only to vector/tensor fields (velocity, stress, $B$-field) while scalar fields (temperature, pressure, density) stay on the cheaper F1 pathway.
*Pros.* Captures most of the sample-efficiency/OOD benefit at a fraction of full-equivariance cost; the cleanest "extend, don't fork" realization — F1 and F2 already coexist by design (Problem 12), this just runs both on different channels; the natural *implementation* of P9.2's framing (P9.2 says *why* equivariance is correct; P9.3 says *how* to pay for it selectively).
*Cons.* Depends on P16/P17 (adopted); the mixed equivariant/non-equivariant interface — e.g. a pressure-gradient scalar-channel term feeding a velocity vector-channel — needs a defined covariant coupling; RBC's buoyancy is exactly such a scalar($T$)→vector($u$) coupling, so this interface must be gotten right, not hand-waved.

**P9.1 — No hard equivariance; augmentation over the *true residual* symmetry group — 7/10 (first-pass baseline).**
*Description.* Keep the plain backbone; augment training with the symmetries that *actually* hold for the target system — for RBC: horizontal translation + horizontal reflection + the joint (fields, $\hat g$) rotation of P9.2's group — rather than naive full-plane rotations that RBC does not obey.
*Pros.* Cheapest; correct once the augmentation group is chosen to match the real physics (the provisional default's error was augmenting over a group RBC doesn't have); reversible — augmentation-trained weights transfer if P9.3 is later adopted; a legitimate, evidence-producing baseline for the S15.2 ablation.
*Cons.* Statistical, not exact — no OOD guarantee at test-time symmetries; a strictly weaker proxy than P9.3 for the isotropic sub-systems where rotation is a true continuous symmetry; if the ablation shows a real gap, a benchmark cycle confirmed what the literature already predicts.

**P9.4 — Full hard equivariance everywhere (irrep-typed attention throughout, = S15.1) — 6/10.**
*Description.* Adopt $SO(2)/SO(3)$-equivariant treatment across the *entire* backbone, all channels, as soon as 2D work begins.
*Pros.* Maximal sample efficiency and OOD generalization; the correct target if data is scarce and OOD-rotation robustness is the priority.
*Cons.* Full tensor-product cost on *every* channel including scalars that don't need it; overkill for RBC's reduced symmetry; commits heavy machinery before the non-equivariant core is even validated in 2D — the same "commit before evidence" the project has consistently declined.

**P9.5 — Soft equivariance training penalty (= S15.4) — 5/10.**
*Description.* Add a loss term penalizing $f(Rx)\ne Rf(x)$ over sampled group elements, distinct from augmentation.
*Pros.* More targeted than plain augmentation; composes with [[training-curriculum]]'s loss-group structure.
*Cons.* Still only statistical, not exact; one more term in an already multi-term objective (Problem 6); untested whether it beats P9.1's simpler baseline.

**Decision (2026-07-01): the dilemma is dissolved by P9.2's reframing** — symmetry-breaking external fields (gravity, $\Omega$, EM) become **covariant conditioning inputs**, making hard equivariance *always physically correct* and reducing the open question to pure cost, answered empirically. **Realize it via P9.3** (channel-typed selective equivariance: F2 on vector/tensor, F1 on scalars), **run P9.1 (augmentation over the true joint symmetry group) as the first-pass baseline**, and **let the S15.2 ablation on the first RBC benchmark decide** whether P9.3's exactness beats P9.1's cheapness — promoting to P9.3+P9.2 if it does. **P9.4 reserved** only if P9.3 proves insufficient; **P9.5 held as a cheap complement**, rejected as sole mechanism. This unifies Problem 9 with Problems 15/16/17 and S24.1 into one stance. **[AI Inference]:** the deepest content here is that "commit to equivariance or not" was a *false* binary all along — the real axis is *which group*, and once symmetry-breaking fields are made covariant inputs, every physical system (isotropic or not) is exactly equivariant under the appropriate joint group, so equivariance stops being a per-system judgment call and becomes a uniform architectural property whose *cost* (not correctness) is the only tunable. RBC is the ideal first test precisely because its broken symmetry forces this realization that a rotationally-isotropic first benchmark would have hidden.

### Problem 10: Training stability of the nonstandard backbone
**Status: DECIDED** — full scored pass complete (2026-07-01): P10.1 (analytic spectral-norm control) + P10.4 (ReZero/bounded per-layer gain) as the load-bearing stability guarantee, P10.3 (init+warmup+clip+AdamW bundle) as the standard-practice complement, P10.5 (fused kernel) deferred to post-correctness, P10.2 (implicit integrator) reserved as fallback. Promotes [[noether-1.0]] §5/§6.1 from provisional default to decided.
No LayerNorm, $\tanh$ symmetric scores, skew-symmetric channel, forward-Euler step $\epsilon < 2/\|J\|_2$ that **drifts during training** ([[skew-symmetric-attention]], [[anti-symmetric-dgn]]). Init scheme, optimizer, schedule, and a fused kernel for the symmetric score are all unspecified, and the wiki itself flags this combination as empirically unproven.

**Prior provisional default (2026-07-01, now subsumed into P10.1 below):** bound $\|J\|_2$ analytically rather than measuring it post-hoc. The full pass below keeps that as the core, adds the *normalization-free depth-stability* piece the provisional default omitted (P10.4), and commits the practical training recipe.

**Why Rayleigh–Bénard raises the stakes.** Burgers-base is shallow ($L{=}4$) and single-field. RBC needs a deeper/wider backbone (more fields, chaotic dynamics, elliptic pressure coupling), and depth is exactly where the two flagged risks — over-smoothing/rank-collapse (Problem 12) and forward-Euler drift — compound. The provisional default bounded the *step*; it did not address *signal growth through depth without LayerNorm*, which a deeper RBC model makes acute.

#### Candidate solutions (scored /10)

**P10.1 — Hard analytic spectral-norm control by constrained parameterization (Cayley/Householder skew term + Stiefel F1) — 9/10.**
*Description.* Parameterize the skew self-term so $\|W_{\text{skew}}-W_{\text{skew}}^\top\|_2$ is *bounded by construction* — Cayley transform of a skew generator (yields an orthogonal factor, norm 1), or spectral normalization — and keep F1's semi-orthogonal (Stiefel) bound from Problem 12. Then the stable forward-Euler bound $\epsilon<2/\|J\|_2$ is computable *per sub-step, per layer* and holds continuously through training rather than being measured and chased post-hoc.
*Pros.* Converts stability from empirical tuning into an **analytic guarantee** — the entire point of the problem; composes exactly with F1's existing norm-bound (same Stiefel machinery); no drift, because the bound is enforced every step, not sampled; the [[noether-1.0]] §5 init already sits inside this bound, so it is a promotion of an existing recipe, not a new one.
*Cons.* Constrained parameterization costs some raw expressivity vs. an unconstrained $W$; the Cayley/QR retraction has a per-step cost (small — §5 already re-orthogonalizes every 100 steps); guarantees *stability*, not *accuracy* — forward-Euler is still first-order (see P10.2).

**P10.4 — Normalization-free residual scaling (ReZero / bounded per-layer gain $\alpha_\ell$) to control depth signal growth without LayerNorm — 8/10.**
*Description.* Since the design forbids LayerNorm (to preserve the skew-channel norm-conservation story of [[normalization-scheme]]), recover deep-network trainability with a ReZero-style learned per-layer gain initialized at (or near) zero on the residual branch, or a bounded $\alpha_\ell$ schedule, so the residual stream's norm is controlled through depth by construction.
*Pros.* Directly targets the "no LayerNorm at depth" risk the provisional default left open; ReZero/DeepNorm/SkipInit are *proven* for deep transformers; a scalar gain on the residual branch does **not** break the skew channel's norm-preservation (it only scales an already norm-controlled update, staying compatible with P10.1's bound and Problem 1's conservation); makes the deep RBC model trainable where a shallow Burgers model got away without it.
*Cons.* ReZero-at-0 slows early training (the branch starts inert); adds a per-layer schedule/knob that interacts with $\epsilon$ (both scale the residual branch — must be jointly budgeted so their product stays under the P10.1 bound); another thing to ablate.

**P10.3 — Standard-practice bundle: careful init + $\epsilon$ warmup + gradient clip + AdamW-with-exclusions — 8/10.**
*Description.* Commit the recipe already drafted in [[noether-1.0]] §5/§6.2 as the default: Xavier/orthogonal init with the RMT-based $\|W_{\text{skew}}\|_2\lesssim4$ bound, $\epsilon$ warmup $0.1\to0.3$, global-norm-1.0 gradient clipping, AdamW excluding $W_{\text{skew}}$ and F1's Stiefel factors from weight decay, cosine LR with 5% warmup, fp32 for projection arithmetic.
*Pros.* Cheap, standard, already ~fully specified — this pass just *commits* it; empirically robust for stiff/deep nets; the exclusions matter (decaying a constrained factor fights its constraint).
*Cons.* Mitigation, not proof — on its own it does not *guarantee* stability, which is why it rides on top of P10.1's bound rather than replacing it; several constants (warmup length, peak LR) are Burgers-tuned estimates that RBC will need re-checked.

**P10.2 — Higher-order / implicit integrator for the depth step (removes the $\epsilon$ knife-edge) — 6/10 (reserved fallback).**
*Description.* Replace the forward-Euler depth update with implicit Euler (unconditionally stable for any $\epsilon$) or Heun/RK2 (second-order accuracy).
*Pros.* Removes the stability-vs-step-size tradeoff entirely (implicit Euler stable for any step); higher per-layer accuracy; principled if the explicit bound proves too restrictive at RBC depth.
*Cons.* Implicit needs a solve per layer (real cost); **changes the clean A-DGN forward-Euler story the whole backbone derivation rests on** ([[anti-symmetric-dgn]]) — a nontrivial fork of the validated equation; RK2 doubles per-layer cost. Hold in reserve, not a default.

**P10.5 — Fused symmetric-score kernel + mixed-precision-safe arithmetic (the systems half) — 7/10 (deferred).**
*Description.* Specify a fused kernel for the tied-$Q{=}K$ symmetric $\tanh$ score (compute $q_i\!\cdot\!k_j+q_j\!\cdot\!k_i$ once, symmetrized), and keep the fp32 island around the projection (already §6.2).
*Pros.* Addresses the explicitly-flagged "fused kernel unspecified"; performance win + numerical safety for the symmetric score.
*Cons.* An optimization, not a stability *guarantee* — correctness-first plain implementation ships before any fused kernel; kernel work is premature before the architecture is validated. Defer.

**Decision (2026-07-01): adopt P10.1 + P10.4 as the load-bearing guarantee** — analytic spectral-norm control (P10.1) makes the Euler step provably stable, and normalization-free residual gain control (P10.4) makes the *deep* model trainable without reintroducing the LayerNorm that would break Problem 1's conservation — **with P10.3 (the standard-practice bundle) committed on top** as the practical recipe. **P10.5's fused kernel is deferred** until after a correct reference implementation; **P10.2's implicit integrator is reserved** as the fallback if the explicit $\epsilon$-bound proves too restrictive at RBC depth (deeper than Burgers). Net posture: **stability by construction, not by tuning** — the bound (P10.1) and the gain control (P10.4) are structural, the schedule (P10.3) is the standard finishing recipe. **[AI Inference]:** P10.4 is the piece the 1D provisional default could omit and the 2D model cannot — a 4-layer Burgers backbone is shallow enough that uncontrolled residual growth stays benign, but the depth RBC's chaotic multi-field dynamics demand is exactly the regime where normalization-free training silently diverges; committing ReZero-style gain control now is the difference between "worked at $L{=}4$" and "trains at all at $L{=}10{+}$."

### Problem 11: No sizing, compute, eval protocol, or inference-cost analysis
**Status: DECIDED** — resolved in full via [[noether-1.0]] (2026-07-01).
Parameter allocation across tokenizer/backbone/decoder; concrete success criteria and conservation diagnostics; and an inference-cost analysis — which sits awkwardly against the "faster than classical solvers" goal, since per-step graph construction + hierarchy + diffusion refinement could be slow.

**Prerequisite chain (2026-07-01):** parameter sizing cannot proceed until Problem 9 (equivariant vs. plain layers scale parameters/compute very differently) and Problem 12 (whether an FFN sub-layer exists — typically the majority of a standard transformer's parameters) are settled. Both now have provisional defaults (no hard equivariance; norm-bounded FFN F1), so sizing is unblocked pending those defaults holding up. This page's own sizing/compute/eval-protocol content remains open.

**Resolved (2026-07-01): [[noether-1.0]].** The named, sized model — nano/base/large configs (≈118K / 847K / 8.1M params) mapped onto [[possible-architectures]]'s existing 100K→1M→10M sweep, full backbone parameter derivation ($12Ld^2$, identical to a vanilla transformer block despite the physics constraints), graph/hierarchy hyperparameters for the 1D Burgers smoke test, $\epsilon/\gamma$ stability bounds instantiating Problem 10's default, optimizer/curriculum schedule tied to [[training-curriculum]]'s phases, and a full eval protocol including an **honest (not falsely precise) inference-cost finding**: "faster than classical solvers" is not testable at this scale and shouldn't be claimed from it. All numbers are design estimates awaiting empirical validation, not tuned results.

---

## Tier 3 — general 2D/3D geometry (surfaced 2026-07-01)

**Framing.** Everything above was designed and validated against 1D periodic Burgers — uniform grid, no boundary to enforce, one field, one node type. General 2D/3D geometry (curved boundaries, unstructured/scattered input, multiple simultaneous BC types, vector/tensor fields) is exactly the case several pages already deferred (graph-tokenizer's own open problem #1 on unstructured patches; P2.3's unspecified ghost-node mechanism; Problem 9's equivariance default, whose justification was 1D-specific). This tier audits what's missing to go there.

**Evaluation lens for every solution below: does it extend the existing architecture, or fork it?** A solution that only works for curved geometry, leaving the uniform-grid path as separate code, fails this test even if it's otherwise sound — the goal is one architecture viable from 1D Burgers through 3D multiphysics, not a family of special cases.

### Problem 13: Unstructured patch/mesh definition for general 2D/3D geometry
**Status: DECIDED** — S13.4 (extend [[intelligent-patching]]'s octree) + S13.2 (mesh-native when available) adopted (2026-07-01).

**The gap.** [[graph-tokenizer]]'s own open problem #1 — for scattered/unstructured input, patches must be defined by spatial clustering, with the clustering algorithm and patch count left unspecified. This is Stage 2 of [[pipeline-contract]], the very first step; nothing downstream runs without an answer, and general 2D/3D geometry is exactly the case that was deferred.

**Why it matters.** 1D Burgers used a fixed uniform patch size on a periodic grid — trivial. Curved boundaries, unstructured meshes, and scattered sensor data have no such uniform structure to exploit.

#### Candidate solutions (scored /10)

**S13.1 — Generic k-means / Voronoi clustering on sample points — 6/10.**
*Description.* Cluster raw points into $N_{\text{patch}}$ groups (k-means, or a Voronoi tessellation from farthest-point/Poisson-disk-sampled seeds); each cluster becomes one node.
*Pros.* Works for any input format with zero geometry-specific logic — maximally general.
*Cons.* Not stable frame-to-frame (cluster assignment can flip between similar snapshots, a form of Problem 7's topology churn); ignores mesh-native connectivity or curved-boundary structure entirely; a full re-cluster every step is real cost.

**S13.2 — Native mesh hierarchy, when a mesh already exists — 8/10.**
*Description.* When input carries an explicit mesh (FEM/FVM cells, CAD triangulation), use its own cells/elements as patches and face-adjacency as edges directly — no clustering step.
*Pros.* Exact; stable across timesteps (no re-clustering); respects the mesh's own resolution grading (fine near walls, coarse in the bulk) for free.
*Cons.* Only applies when a mesh is given — doesn't cover scattered/gridless input; creates a second tokenizer code path unless composed carefully with a geometry-agnostic fallback (see recommendation).

**S13.3 — Learned/differentiable soft clustering — 6/10.**
*Description.* Replace hard k-means with a differentiable slot-attention or optimal-transport-based assignment, trained jointly with the S3.4 fidelity objective.
*Pros.* Smooth, frame-to-frame stable by construction; patch placement optimizes directly for reconstruction quality.
*Cons.* Real added training cost and complexity; soft assignments still need a hardening step before the radius-graph/hierarchy can consume them.

**S13.4 — Universal hierarchical partitioning: extend the existing octree/quadtree from [[intelligent-patching]] to serve as the patch definer everywhere — 9/10.**
*Description.* The refinement tree already built for adaptive resolution is purely geometric (bounding-box subdivision) and doesn't care about input format: on a uniform grid it reduces to the current uniform-patch design exactly; on scattered/curved-domain input, seed it from the point cloud's bounding volume and subdivide by point density plus the existing refinement indicator $\eta$ (augmented with a geometric distance-to-boundary term, once Problem 14's boundary representation exists).
*Pros.* **The strongest fit for "one architecture throughout"** — no new machinery, an extension of a component already committed to; stable if the tree is only rebuilt when the monitor function crosses a threshold, not every step; naturally handles boundary-layer anisotropy via anisotropic box splitting.
*Cons.* Needs a geometric (not just physics-gradient) refinement criterion for boundary regions; axis-aligned box splitting is a poor fit for genuinely diagonal/curved boundaries unless boxes are allowed to be clipped (a tractable extension, analogous to classical cut-cell methods).

**Decision: adopt S13.4** as the universal default (it is literally [[intelligent-patching]]'s existing tree, extended, not a new mechanism), **composed with S13.2** whenever an explicit mesh is available (use the mesh's exact adjacency for edges even when S13.4's tree defines patches — these compose rather than compete), and **S13.1 as a cheap fallback** for pure scattered data with no mesh and no tuned geometric monitor yet. S13.3 is a valuable future upgrade once frame-instability is a measured (not hypothetical) problem.

### Problem 14: Curved/multi-type boundary enforcement has no concrete mechanism
**Status: DECIDED** — S14.3 (mandatory) + S14.1 (default) + S14.2 (mesh-native fallback) adopted (2026-07-01).

**The gap.** P2.3 (ghost-node/immersed-boundary handling) names a resolution direction but specifies no algorithm — no normal estimation, no ghost-node placement rule for curved boundaries, and no handling of *multiple simultaneous* BC types (Dirichlet inlet + no-slip wall + Neumann outflow), which P2.2's joint projection implicitly assumed was one homogeneous block.

**Why it matters.** This blocks the first real 2D/3D benchmark outright — even a simple case (flow around a cylinder in a channel) needs three BC types active on one mesh simultaneously.

#### Candidate solutions (scored /10)

**S14.1 — Signed-distance-field (SDF) boundary representation + analytic normals — 8/10.**
*Description.* Represent the boundary as $\phi_{\text{boundary}}(x)$ (positive inside, $|\nabla\phi|=1$); place/reflect ghost nodes along $\nabla\phi_{\text{boundary}}$; label the zero level-set into Dirichlet/Neumann/periodic *regions* as metadata, so P2.2 gets one constraint row-block per region instead of one global block.
*Pros.* General, differentiable, well-precedented (ghost-fluid/immersed-boundary methods); normals come free from the gradient; composes cleanly with S13.4 (the SDF value doubles as a geometric refinement criterion for the octree).
*Cons.* Requires converting CAD/mesh geometry to an SDF (standard but an extra step); the per-region BC labeling must be threaded through [[pfm-interface-design]]'s "boundary-condition spec" input, which needs to become a *list* of (region, type) pairs, not a single code.

**S14.2 — Mesh-native boundary faces, when a mesh exists — 8/10.**
*Description.* When a mesh is available (S13.2), boundary faces already carry BC tags in standard formats; derive ghost values/normals directly from face geometry, no SDF needed.
*Pros.* Exact, zero extra geometric machinery, matches standard CFD conventions directly.
*Cons.* Only covers the normal/placement half of the problem — still needs the multi-type projection fix (S14.3) independent of how normals are obtained; only applies when a mesh exists.

**S14.3 — Generalize P2.2 to multi-block constraint stacking by labeled region — 9/10.**
*Description.* Regardless of how normals are obtained, build the joint KKT system's constraint matrix as a *union* of typed row-blocks — {conservation} $\cup$ {Dirichlet rows on region $R_1$} $\cup$ {Neumann rows on region $R_2$} $\cup$ {periodic-pairing rows on region $R_3$} — each with its own right-hand side.
*Pros.* This is the actual required fix, and it is a direct **extension of the already-adopted P2.2 mechanism**, not a new one — exactly the "extend, don't fork" answer the architecture needs.
*Cons.* The existing compatibility-condition caveat on $b$ (net flux consistency) gets more bookkeeping with more region types — a global cross-region feasibility check is now needed, not just a single-block one.

**S14.4 — Ghost-node placement via local k-NN plane-fitting (no SDF, no mesh) — 6/10.**
*Description.* For bare labeled point clouds, estimate local boundary normals via PCA/plane-fitting on nearby boundary-tagged points.
*Pros.* Fewest input assumptions of any option.
*Cons.* Noisy/unstable on sparse or noisy point clouds; least accurate — a fallback, not a default.

**Decision: adopt S14.3 as mandatory** (required regardless of geometry source), **S14.1 as the default normal/placement mechanism** for its generality and clean composition with S13.4, falling back to **S14.2** whenever an explicit mesh exists (prefer exact information over a derived SDF), with **S14.4** as the point-cloud-only fallback.

### Problem 15: The "no hard equivariance" default's justification doesn't survive 2D/3D
**Status: DECIDED (process only)** — S15.2 adopted: run the ablation first (2026-07-01); the substantive equivariance question itself stays open pending that result.

**The gap.** Problem 9's provisional default was justified *specifically* because 1D Burgers has no meaningful rotation group to test against. $SO(2)$/$SO(3)$ are real continuous symmetries in 2D/3D with documented sample-efficiency and OOD benefits from hard equivariance ([[equivariant-gnns]]). Silently carrying the 1D default forward would be an omission, not a decision.

**Why it matters.** This is a scope-defining choice, not a detail — it changes parameter/compute scaling (per the original Problem 9/11 discussion) and is gated on Problem 16 (below) existing before it's even executable.

#### Candidate solutions (scored /10)

**S15.1 — Commit to full hard equivariance now (F2 + irrep-typed attention throughout) — 6/10.**
*Description.* Adopt SO(2)/SO(3)-equivariant treatment across the backbone as soon as 2D/3D work begins — irrep-typed features, F2's gated FFN, equivariant heads per [[multihead-attention]]'s irrep-head option.
*Pros.* Maximal sample efficiency and OOD generalization; directly resolves the justification gap; F2 was explicitly designed as this upgrade path.
*Cons.* Real tensor-product cost ($O(L_{\max}^4)$/edge, per [[possible-architectures]]); gated on Problem 16 existing first; commits real architecture change before the non-equivariant core is validated at all in 2D/3D.

**S15.2 — Explicit, scored re-validation ablation before committing either way — 8/10.**
*Description.* Keep augmentation as the working mechanism, but run an explicit comparison (augmented non-equivariant vs. a lightweight equivariant baseline) on the first 2D benchmark as a scored ablation, rather than silently inheriting or blindly committing.
*Pros.* Cheapest next step; doesn't block starting 2D work; produces the *evidence* this project has consistently preferred over guessing (the same "hard to pick without evidence" stance the founding brainstorm took); turns a silent inheritance into a considered, falsifiable decision.
*Cons.* Augmentation is a strictly weaker proxy than true equivariance (statistical, not exact; doesn't help at OOD test-time rotations) — if the ablation shows a real gap, a benchmark cycle was spent confirming what the literature already predicts.

**S15.3 — Selective/channel-typed equivariance: F2 only on vector/tensor channels, F1 stays on scalars — 8/10.**
*Description.* Once Problem 16 tags channels by irrep type, apply F2's gated treatment only where it matters (velocity, stress) while scalar channels (pressure, density) stay on the cheaper F1 pathway.
*Pros.* Captures most of the sample-efficiency benefit where it's most valuable, at a fraction of S15.1's cost; **the cleanest "extend, don't fork" answer** — F1 and F2 already coexist by design (Problem 12), this just runs both simultaneously on different channels rather than switching wholesale.
*Cons.* Depends on Problem 16; mixed equivariant/non-equivariant channels need a defined interaction interface (e.g. a pressure-gradient term touching a velocity channel) — bounded but real.

**S15.4 — Soft equivariance training penalty (not augmentation, not architecture) — 5/10.**
*Description.* Add an explicit loss term penalizing $f(Rx)\ne Rf(x)$ for sampled rotations, distinct from training on rotated copies.
*Pros.* More targeted than plain augmentation; composes with [[training-curriculum]]'s existing loss-group structure.
*Cons.* Still only statistical, not exact; one more term in an already multi-term objective (Problem 6); untested whether it beats S15.2's simpler baseline.

**Decision: adopt S15.2 first** — an explicit ablation, not a silent inheritance or an uncommitted leap — with **S15.3 as the likely destination** if the ablation favors equivariance, reserving full S15.1 commitment only if S15.3 proves insufficient.

### Problem 16: No irrep-typing mechanism exists to feed equivariant treatment
**Status: DECIDED** — S16.1 (default) + S16.2 (fallback) adopted (2026-07-01).

**The gap.** F2 and S15.3 both assume node/edge features arrive pre-labeled by irrep type (scalar/vector/tensor); nothing upstream produces this label today — a multi-component field becomes one undifferentiated latent vector.

#### Candidate solutions (scored /10)

**S16.1 — Extend the existing field-type conditioning codes with an irrep tag — 9/10.**
*Description.* [[pfm-interface-design]] already has field-type codes distinguishing velocity/pressure/$B$-field channels; add an explicit irrep tag (scalar/vector/tensor/pseudo-variants, extended further in Problem 17) to this *existing* mechanism, and route per-channel encoder treatment by the tag.
*Pros.* Reuses existing machinery directly — no new conditioning channel; the tag is known a priori from physics (you always know which raw channel is velocity-$x$), avoiding an inference failure mode entirely.
*Cons.* Requires maintaining a type taxonomy as new field types are added (ties to Problem 8's already-open variable-cardinality gap); doesn't cover a genuinely unlabeled channel at inference (falls through to S16.2).

**S16.2 — Learned irrep-type inference, committed via classifier when unlabeled — 7/10.**
*Description.* Mirror S4.3's already-adopted pattern exactly: infer a channel's type from its transformation behavior under sampled rotations when unlabeled, then *commit* via argmax before downstream processing (equivariant treatment, like field/particle decoding, needs a hard choice, not a soft blend).
*Pros.* Reuses an identical design pattern already validated for a structurally analogous problem (node type, Problem 4) — strong internal consistency.
*Cons.* Needs multiple orientations to observe, not always available from one static snapshot; a mis-typed channel silently corrupts equivariant processing.

**Decision: adopt S16.1** as the default (exact, reuses the existing interface directly), **S16.2 as the fallback** for unlabeled data — the same known-vs-inferred-then-committed shape already used for S4.1/S4.3, applied to a structurally identical new problem.

### Problem 17: 3D-specific representation distinctions unaddressed
**Status: DECIDED** — S17.1 + S17.2 adopted together (2026-07-01).

**The gap.** 2D vorticity is a pseudoscalar; 3D vorticity is a genuine axial (pseudo-)vector; turbulence closures introduce rank-2 tensors (Reynolds stress). None of these distinctions exist in the current design, and equivariant tensor-product cost at real scale is unbudgeted.

#### Candidate solutions (scored /10)

**S17.1 — Extend Problem 16's tag taxonomy to include parity and rank, not just arity — 8/10.**
*Description.* Broaden the tag set to {scalar, pseudoscalar, vector, pseudovector, rank-2 tensor, …} so F2's gating respects parity under reflection, not just rotation (getting compound quantities right, e.g. helicity $u\cdot\omega$ is a pseudoscalar — a vector dotted with a pseudovector).
*Pros.* A minimal, direct extension of S16.1 — same mechanism, richer taxonomy; gets reflection physics right where a naive vector-only treatment would silently fail.
*Cons.* Assigning parity correctly for compound/derived quantities needs real domain-knowledge care — bounded, not open-research, but not free either.

**S17.2 — Bound tensor-product cost via low-$L_{\max}$ truncation — 8/10.**
*Description.* Cap the equivariant representation's maximum angular order at $L_{\max}=1$ or $2$ (enough for vectors and rank-2 tensors) rather than the high-$L_{\max}$ regime atomistic modeling needs, keeping the $O(L_{\max}^4)$ cost proportionate to what CFD-scale physics actually requires.
*Pros.* Directly controls the flagged cost risk; tailored to the actual physics (CFD tensors rarely exceed rank 2, unlike molecular-orbital modeling); keeps S15.1/S15.3 tractable at Noether-1.0 scale.
*Cons.* Bounds the risk, doesn't eliminate it; the right cap is an empirical question this page can't answer.

**Decision: adopt S17.1 + S17.2 together** — correct taxonomy, deliberately bounded cost — so whichever way Problem 15 resolves, the equivariant path stays proportionate to CFD-scale tensors rather than open-ended.

### Problem 18: The elliptic solve (P2.1) was sized only for the trivial periodic/uniform case
**Status: DECIDED** — S18.1 + S18.2 adopted as a package (2026-07-01).

**The gap.** P2.1's Poisson/Leray projection assumed an FFT-friendly or simple-multigrid-friendly domain. Curved geometry needs a correctly-posed Neumann condition on the pressure correction at walls and a multigrid scheme that respects irregular/anisotropic geometry (agglomeration multigrid) — neither exists in the current design.

#### Candidate solutions (scored /10)

**S18.1 — Agglomeration multigrid on the same supernode hierarchy already built for global coupling — 8/10.**
*Description.* Use the identical learned restriction/prolongation hierarchy already constructed for elliptic/global coupling ([[graph-tokenizer]], [[pipeline-contract]]) to also run the agglomeration-multigrid V-cycle — the coarse-graining that groups irregular/anisotropic fine cells for long-range reach is the same operation agglomeration multigrid needs for irregular meshes.
*Pros.* **The cleanest possible "same mechanism, reused" answer** — literally zero new structure, matching the wiki's own repeated stance that this hierarchy should be "one structure, reused throughout."
*Cons.* Agglomeration multigrid's convergence on genuinely bad-aspect-ratio meshes is a real numerical-methods concern that a hierarchy built primarily for physics-locality reasons may not automatically satisfy well — needs empirical validation, not just assertion.

**S18.2 — Correct Neumann BC on the pressure correction, derived from Problem 14's region labels — 8/10.**
*Description.* Use S14.3's per-region BC labels to automatically derive the standard classical-CFD Neumann condition on $\phi$ at each labeled wall/inlet/outlet segment (homogeneous Neumann at a no-slip wall, from the momentum equation evaluated there).
*Pros.* Well-understood, standard part of the fix, not a research question — made concrete by hooking directly into Problem 14's labels rather than a separate mechanism.
*Cons.* Depends on Problem 14 being resolved first; getting the sign/derivation right per boundary type (inlet vs. outlet vs. wall) is an implementation-correctness risk, not a design one.

**Decision: adopt S18.1 + S18.2 as a package** — the multigrid mechanism (reused hierarchy) plus the boundary condition feeding it (reused region labels) — extending P2.1 to curved geometry by wiring existing pieces together, adding no genuinely new component. Validate S18.1's convergence empirically before trusting it on production-quality meshes.

### Problem 19: Is flat Euclidean coordinate encoding adequate for curved/wall-bounded geometry?
**Status: DECIDED** — S19.1 adopted (2026-07-01).

**The gap.** The coordinate-implicit decoder's $\gamma(y-x_i)$ uses raw Euclidean displacement. Boundary-layer physics is more naturally described in wall-normal/wall-tangential coordinates.

#### Candidate solutions (scored /10)

**S19.1 — Keep Euclidean $\gamma(\cdot)$; add the SDF value (from S14.1) as an extra decoder input, not a replacement — 8/10.**
*Description.* Append $\phi_{\text{boundary}}(y)$ as one more scalar into the decoder alongside the existing Euclidean encoding.
*Pros.* Minimal, additive, reuses the SDF that Problem 14 already introduces; gives the decoder wall-distance information without redesigning the encoding.
*Cons.* Doesn't fully replace Euclidean structure with true wall-fitted coordinates — very thin boundary layers may still be harder to fit than a dedicated curvilinear system would allow.

**S19.2 — Full curvilinear/geodesic encoding near boundaries, replacing Euclidean locally — 6/10.**
*Description.* Within some distance of a labeled boundary, encode $(\text{signed distance}, \text{arc-length along boundary})$ instead of raw offset.
*Pros.* The physically "correct" representation for boundary-layer flows.
*Cons.* A genuine architectural fork — different coordinate logic near vs. far from boundaries, with blending machinery needed at the transition — in real tension with the "one architecture" goal for a benefit S19.1 mostly already captures.

**Decision: adopt S19.1** — a strict additive extension; reserve S19.2 only if S19.1 proves empirically insufficient on very thin boundary layers.

### Problem 20: Global single-scale nondimensionalization breaks for embedded-object/multi-scale domains
**Status: DECIDED** — S20.1 adopted (2026-07-01).

**The gap.** Stage 1 picks one global $(L_0,U_0,T_0)$. A domain with hugely disparate local scales (a small object in a large enclosure) has no single good choice.

#### Candidate solutions (scored /10)

**S20.1 — Per-supernode-region characteristic scales, tied to the existing hierarchy — 8/10.**
*Description.* Compute characteristic scales per coarse supernode region (reusing the hierarchy already built for everything else) so $z_{\text{param}}$ becomes a per-region signal rather than one global one.
*Pros.* Reuses existing machinery again rather than adding new — directly targets the stated failure mode.
*Cons.* Needs a rule for estimating per-region scales and smoothly blending them at region boundaries to avoid discontinuities.

**S20.2 — Keep one global scale; rely on existing in-context regime inference to compensate implicitly — 5/10.**
*Description.* Leave Stage 1 unchanged; trust the "infer unknown parameters from trajectory statistics" mechanism to handle local variation implicitly.
*Pros.* Zero new machinery.
*Cons.* That mechanism was validated for whole-trajectory regime inference, not spatially-varying regimes within one snapshot — likely to underperform on genuinely large scale-disparity cases; closer to hoping it works than a principled fix.

**Decision: adopt S20.1**; S20.2 acceptable only as a stopgap before S20.1 is implemented.

### Problem 21: Sparse attention (Index Share) moves from a deferred option to a mandatory requirement at realistic 3D scale, never actually exercised
**Status: DECIDED** — S21.1 adopted as a scheduled milestone, S21.2 as parallel insurance (2026-07-01).

**The gap.** "Dense attention is fine, defer Index Share" was explicitly justified by Burgers/small-2D-NS token counts. Realistic 3D DNS/LES grids ($10^6$–$10^9$ points) break that assumption outright, and the sparse-attention machinery it's deferred to has never been validated.

#### Candidate solutions (scored /10)

**S21.1 — Treat this as a scheduled validation milestone, not a new design problem — 8/10.**
*Description.* [[index-share-sparse-attention]] and [[hierarchical-query-attention]] already fully specify the mechanism — this was a *deferred scaling option*, not an unspecified one. Schedule its activation and benchmarking once node counts exceed the crossover threshold already implied by [[initial-model-architecture]]'s own reasoning, rather than treating it as a new gap.
*Pros.* Correctly identifies no new design work is needed — the cheapest possible resolution; keeps the architecture unified exactly as already planned.
*Cons.* Doesn't eliminate the real risk that Index Share's claimed symmetry/conservation-preservation ([[symmetric-attention-physics]]) hasn't actually been checked at the scale where it would matter.

**S21.2 — Pre-emptively validate on a synthetic large-$N$ benchmark, decoupled from any specific PDE — 6/10.**
*Description.* Build a synthetic large symmetric-attention graph with known ground-truth conservation properties purely to stress-test Index Share at scale.
*Pros.* Isolates validation from PDE-specific confounds; catches bugs earlier and cheaper than mid-3D-benchmark.
*Cons.* Extra upfront harness-building effort; some risk it doesn't represent real 3D graph statistics (degree distribution, hierarchy depth) closely enough to be predictive.

**Decision: adopt S21.1** as the primary plan, with **S21.2** as cheap parallel insurance once resources allow.

### Problem 22: Moving/deforming domain boundaries (ALE, free surfaces, FSI) — a genuinely new regime
**Status: DECIDED (scoped out)** — S22.2 adopted: explicitly out of scope for now (2026-07-01); S22.1 is the plan whenever it's picked back up.

**The gap.** Not covered by Problem 4 (heterogeneous node *types*) or Problem 7 (particles moving / patches re-refining) — this is the domain's *own boundary* deforming over time (free surfaces, fluid-structure interaction), with no design anywhere yet.

#### Candidate solutions (scored /10)

**S22.1 — Arbitrary Lagrangian-Eulerian (ALE) mesh motion, layered onto Problem 7's existing dynamic-topology handling — 7/10.**
*Description.* Add a mesh-velocity field advecting patch centroids near the moving boundary, reusing the SAME re-tokenization-every-step machinery already built for moving particles, rather than a separate deforming-mesh system.
*Pros.* Extends already-acknowledged machinery (Problem 7) rather than adding a parallel one; ALE itself is well-precedented classically.
*Cons.* A genuinely new failure mode: boundary-adjacent nodes moving means Problem 14's region labels must also update every step, compounding Problem 7's already-flagged edge-churn/momentum-injection risk.

**S22.2 — Explicitly scope this out of the near-term roadmap — 7/10.**
*Description.* Treat static-boundary domains as the scope for the foreseeable plan; flag ALE/FSI as future work rather than solving it now.
*Pros.* Keeps the immediate roadmap focused and achievable — consistent with this project's own established discipline (Burgers-first, narrow-then-expand); a legitimate scoping decision, not a weak solution.
*Cons.* Doesn't solve anything; free surfaces and FSI are common enough in real multiphysics that this can't be deferred indefinitely.

**Decision: adopt S22.2 for now** — explicitly out of scope for the next 2D/3D benchmark — with **S22.1 as the concrete plan** once it's picked back up, since it correctly reuses Problem 7's machinery rather than forking.

### Problem 23: Mesh-native connectivity vs. radius graph for real unstructured meshes
**Status: DECIDED** — S23.1 adopted (2026-07-01).

**The gap.** Real 3D meshes carry explicit face/cell adjacency; a pure Euclidean-distance cutoff can misrepresent coupling on stretched/sliver elements.

#### Candidate solutions (scored /10)

**S23.1 — Prefer mesh-native face-adjacency edges whenever a mesh exists; radius graph only as a meshless fallback — 8/10.**
*Description.* The same "use the mesh when you have one" pattern already adopted twice (S13.2, S14.2), extended explicitly to edges: build from shared mesh faces when available, radius-graph only for scattered/meshless input.
*Pros.* Directly reuses an already-established pattern for the same underlying reason — strong internal consistency, and simply more correct when exact information exists.
*Cons.* The graph-construction code path genuinely branches on input type — real but already-precedented complexity from S13.2/S14.2, not new.

**S23.2 — Always use radius graph uniformly, regardless of mesh availability — 4/10.**
*Description.* Ignore native connectivity even when available, for implementation simplicity.
*Pros.* One code path.
*Cons.* Throws away free, exact information and risks the sliver/anisotropic-element miscoupling that motivated this problem — no remaining simplicity argument once mesh-aware paths exist elsewhere anyway (S13.2, S14.2).

**Decision: adopt S23.1** — consistent with the pattern already adopted twice in this same tier.

### Problem 24: Rotating reference frames (turbomachinery) — missing conditioning slot
**Status: DECIDED** — S24.1 adopted (2026-07-01).

**The gap.** No slot exists in [[pfm-interface-design]]'s 5-input interface for frame angular velocity, needed for Coriolis/centrifugal terms in turbomachinery-class 3D problems.

#### Candidate solutions (scored /10)

**S24.1 — Add angular velocity $\Omega$ as a sixth conditioning input, injected exactly like the existing dimensionless parameters — 8/10.**
*Description.* Extend the interface with one more conditioning signal, entering via the identical (additive + AdaLN-gain) mechanism already used for $\mathrm{Re}/\mathrm{Ma}/\mathrm{Pr}$.
*Pros.* Trivial additive extension of an existing, well-specified mechanism; unblocks turbomachinery problems without touching the backbone — the Coriolis/centrifugal terms become something the backbone *learns to apply given context*, exactly like every other force law per [[graph-tokenizer]]'s "the backbone learns the force, not the tokenizer."
*Cons.* Doesn't hard-encode the force structure — relies on in-context learning generalizing to it, the same trust already extended everywhere else, but worth stating explicitly.

**S24.2 — Hard-code Coriolis/centrifugal terms as an explicit analytic force contribution — 4/10.**
*Description.* Add a fixed, non-learned term for $-2\Omega\times v$ and $-\Omega\times(\Omega\times r)$ using the S24.1 conditioning.
*Pros.* Guarantees correct rotating-frame physics regardless of training data coverage.
*Cons.* **Directly contradicts the project's foundational equation-agnostic philosophy** ("you never hardcode the interaction... let the processor learn it," [[graph-tokenizer]]) — a real tension with core philosophy, not a minor style question, and a slippery slope toward hardcoding every other known force law too.

**Decision: adopt S24.1** — both for triviality and because S24.2 would break the project's own stated architectural philosophy.

---

## [AI Inference] — the throughline across Problems 13–24

**[AI Inference]:** Almost every recommended solution above is a **reuse or metadata extension of machinery already committed to elsewhere in this folder**, not a new component: S13.4 extends [[intelligent-patching]]'s octree; S14.3 extends the already-adopted P2.2; S16.1/S24.1 extend the existing conditioning-injection interface; S18.1 reuses the supernode hierarchy a third time (tokenization, global coupling, now curved-geometry multigrid); S16.2 reuses S4.3's exact known-vs-inferred-then-committed pattern; S13.2/S14.2/S23.1 share one "prefer the mesh when you have one" decision, made three times for the same underlying reason. The **one genuinely new primitive introduced across all twelve problems is the SDF boundary representation** (S14.1) — and even that gets reused three more times (S13.4's geometric refinement criterion, S18.2's Neumann BC derivation, S19.1's decoder input). This is strong, direct evidence that **the architecture is viable the entire way through by extension, not by forking** — the 2D/3D gap was mostly missing *plumbing between existing pieces*, not a missing new architecture.

### Problem 25: The projection machinery is equality-only — inequality/positivity/realizability constraints are unaddressed

**Status: DECIDED** — S25.1 (default) + S25.3 (complement when combined with an equality constraint) adopted (2026-07-01).

**Framing note:** unlike Problems 13–24, this is not a 2D/3D-geometry gap — it applies at any dimensionality, including 1D — so it doesn't fit Tier 3's frame, but it's tracked alongside it since it surfaced from the same "is Noether 1.0 viable for general physics" audit. Severity is **Tier-1-adjacent**: it threatens whether the model can guarantee *physically admissible* output at all, for any field beyond a signed scalar.

**The gap.** Every hard-constraint projection adopted so far — S1.1's conserved-total projection, P2.1–P2.4's divergence-free/BC projections — is onto an **affine equality set** $\{x:Ax=b\}$, solved in closed form (or via a Poisson solve for the elliptic case). General physical fields carry **inequality** constraints that are just as physically fundamental: density $\rho\ge0$, temperature/pressure/concentration positivity, turbulent kinetic energy $k\ge0$, phase fractions in $[0,1]$ — and **realizability** constraints like a Reynolds-stress tensor needing to be positive semi-definite (a matrix inequality, not even a simple scalar one). Nothing in the current design prevents the model from outputting negative density or an unrealizable stress tensor.

**Why it matters.** 1D Burgers dodges this entirely — $u$ is a signed velocity, negative values are physically fine — so it never surfaced during any Burgers-focused audit ([[noether-1.0]] included). But almost every other physical system this model targets has some positivity- or bound-constrained field. Without a mechanism, "physically admissible" is not actually guaranteed even after S1.1/P2.2's fixes — those handle conservation/BCs but not admissibility. This directly threatens the project's headline "hard architectural constraint" (level 5 on the physics-encoding spectrum) framing for any field with a sign or bound constraint — which is most of general multiphysics.

**The complication.** Combining an inequality constraint (a half-space, box, or PSD cone) with an *existing* equality constraint (like S1.1's mass conservation) is no longer a single closed-form linear solve — the intersection of a cone/box and an affine subspace generally has no closed form and needs an iterative method, unlike the pure-equality case.

#### Candidate solutions (scored /10)

**S25.1 — Reparameterize outputs to be admissible by construction (exp/softplus for positivity, Cholesky factor for PSD tensors) — 9/10.**
*Description.* Instead of predicting the raw quantity and projecting/clipping afterward, predict a transformed quantity whose inverse-transform is automatically admissible: $\log\rho$ (or a softplus/exp output activation) so $\rho=\exp(\cdot)>0$ always; a sigmoid output for phase fractions in $[0,1]$; a Cholesky factor $L$ (unconstrained) for PSD tensor fields, reconstructing the tensor as $LL^\top$, PSD by construction regardless of what $L$ is.
*Pros.* Admissibility is free — zero runtime projection cost, differentiable everywhere, no non-smooth clipping; matches this project's consistent preference for baking structure into the parameterization rather than correcting after the fact (the same philosophy behind the skew-symmetric channel and F1's semi-orthogonal FFN); standard, well-established technique across physics-ML, not exploratory.
*Cons.* Changes what "the residual" means for that field (a log-space/multiplicative update, not the uniform additive-residual framing used elsewhere) — needs a per-field-type branch, not one unified rule; can distort the loss landscape near zero for very small values; doesn't handle constraints that couple *multiple* fields jointly (an equation-of-state linking pressure/density/temperature) — those still need real projection.

**S25.2 — Explicit clipping / box projection for scalar bounds — 6/10.**
*Description.* After decode, clamp out-of-bound scalar fields directly (ReLU-style for positivity, clamp to $[0,1]$ for fractions).
*Pros.* Trivial to implement, zero added cost, works for any scalar bound without redesigning the output head.
*Cons.* Not the minimum-norm correction (unlike S1.1's provably-optimal rank-$k$ projection) — just truncates, can silently discard information; non-differentiable at the boundary; doesn't handle joint/coupled constraints at all.

**S25.3 — Dykstra's alternating projection, to combine an inequality constraint with an existing equality constraint (S1.1) — 7/10.**
*Description.* When a field needs *both* an equality projection (e.g. conserved total mass) *and* an inequality constraint (positivity) simultaneously, iteratively project onto each set in turn (with the standard Dykstra correction term carried between iterations), converging to the closest point in the intersection.
*Pros.* The mathematically correct general answer when constraints must combine — handles exactly the case S25.1/S25.2 don't (e.g. "density conserved in total *and* everywhere positive"); a textbook algorithm, not novel machinery; generalizes as more constraint types accumulate with multiphysics breadth.
*Cons.* Iterative, not one-shot closed-form — real per-step compute cost, unlike S1.1's cheap rank-$k$ correction; convergence/iteration-count needs tuning; more complex to implement correctly than a single clip or reparameterization.

**S25.4 — Eigenvalue clipping / nearest-PSD projection for tensor realizability — 6/10.**
*Description.* For tensor fields requiring PSD realizability, eigendecompose the predicted tensor, clip negative eigenvalues to zero (or a small floor), reconstruct — the standard "nearest PSD matrix" projection.
*Pros.* Direct, well-understood fix for the PSD sub-case specifically; useful if a raw (non-Cholesky) tensor output is needed for some other reason (e.g. composing with a pretrained head).
*Cons.* Per-tensor, per-node, per-step eigendecomposition is real compute cost, strictly worse than S25.1's free-by-construction guarantee; largely redundant with S25.1 once adopted — a fallback, not a primary recommendation.

**S25.5 — Soft penalty (hinge/barrier loss) only, no hard enforcement — 4/10.**
*Description.* Add a loss term penalizing violation (e.g. hinge loss on negativity), with no architectural enforcement.
*Pros.* Simplest, always available, no architecture change.
*Cons.* No guarantee — exactly the "conditioned not enforced" failure mode Problem 2 exists to fix; contradicts the project's level-5 hard-constraint ambitions; the project has consistently rejected soft-penalty-only as the *sole* mechanism elsewhere (S1.4, S6.6) — adopting it here would be inconsistent with its own established stance.

**Decision (2026-07-01): adopt S25.1** as the default (reparameterize for free-by-construction admissibility, matching the project's general architectural-not-corrective preference), with **S25.3 (Dykstra)** as the necessary complement whenever an inequality constraint must combine with an existing equality projection (S1.1) or a joint/coupled multi-field constraint (S25.1 alone doesn't cover those). **S25.4** is a fallback where Cholesky-reparameterization isn't available; **S25.2** (clipping) acceptable only as a cheap placeholder before S25.1 is implemented; **S25.5 rejected as the sole mechanism**, consistent with the project's stance on Problems 1 and 6.

---

## What is actually solid (so the audit stays balanced)

- **Tokenizer topology** — the reachability-vs-range argument and "hierarchy = neural FMM/Barnes–Hut" ([[graph-tokenizer]]).
- **Conditioning interface** — Re/Ma/Pr injection and in-context inference ([[pfm-interface-design]], [[initial-model-architecture]]).
- **Per-layer attention math** — the symmetric + skew-symmetric update ([[symmetric-attention-physics]], [[skew-symmetric-attention]]).
- **Conceptual conservation layering** — correct in spirit; the gap is making it *trajectory-true* (Problem 1), not the concept itself.

---

## Priority ordering

1. **Problem 1** (depth-vs-time conservation) — deepest; undercuts the headline goal; forces clarity on Problems 2 and 5.
2. **Problem 2** (enforce vs. condition) — blocking for the first NS smoke test.
3. **Problem 3** (tokenizer autoencoder adequacy/training) — foundational; everything downstream depends on the latent being sufficient.
4. **Problem 5** (determinism vs. chaos) — load-bearing for fluids; entangled with #1/#2.
5. **Problem 4** (coupled multiphysics) — the actual bridge; can be deferred behind a concrete coupled benchmark but should not be forgotten or oversold.
6. Tier 2 — resolve during implementation; normal engineering.

The unifying theme across the top four is a single decision the project keeps deferring: **enforce vs. learn**. Settling that — which invariants/constraints are hard (projected/integrated) vs. soft (penalized/corrected) — collapses most of Tier 1 into one coherent stance.

**Addendum (2026-07-01):** Problem 12 (no per-node nonlinear/FFN block, newly surfaced) inserts alongside Problem 1 in severity — expressive capacity is co-equal with conservation as a threat to the core claims, since an under-expressive backbone cannot represent the local constitutive nonlinearities multiphysics requires, and a naively-added FFN can silently undo the very conservation Problem 1 fixes. It does not reorder the original five; it runs in parallel, resolved via a norm-bounded FFN (F1) compatible with the S1.1 projection. Problem 6 (loss) is now resolved (see Candidate solutions above + [[training-curriculum]]) and drops out of the open list; Problems 9 and 10 gain provisional defaults sufficient to unblock parameter sizing (Problem 11) but remain open for a full scored pass. *(Superseded by Addendum 4: both received full scored passes and are now DECIDED.)*

**Addendum 2 (2026-07-01) — Tier 3 priority, within general 2D/3D geometry:** 1. **Problem 13** (patch/mesh definition) — most foundational; blocks graph construction entirely, so it gates everything else in this tier. 2. **Problem 14** (boundary enforcement) — blocks the first BC-bearing 2D/3D benchmark. 3. **Problem 15** (equivariance re-decision) — a scope-defining choice that Problem 16 is gated on. 4. **Problem 18** (elliptic solve at curved geometry) — depends on 13/14, needed specifically for incompressible NS. 5. **Problem 16** (irrep typing) — needed to make 15's decision executable. 6. **Problem 17** (3D-specific distinctions) — matters once 15/16 resolve toward equivariance, more acutely in 3D. 7. **Problem 23** (mesh-native edges) — bundled with 13/14's mesh-vs-generic pattern, low incremental risk. 8–11. **Problems 19, 20, 21, 24** — refinements/scheduled milestones, not blocking. 12. **Problem 22** (moving boundaries) — explicitly scoped out for now (its own recommended solution). None of Tier 3 reorders Tier 1/2's priority; it is the next tier once a curved-geometry benchmark is chosen.

**Addendum 3 (2026-07-01) — Problem 25 and overall status:** Problem 25 (inequality/positivity/realizability constraints) sits **outside Tier 3's dimensionality frame** (it applies even in 1D) but is **Tier-1-adjacent in severity** — it threatens physical admissibility of output for any field beyond a signed scalar, which is most of general multiphysics. It does not block the current 1D Burgers benchmark (where $u$ has no sign constraint) but was, alongside **Problem 8** (variable field cardinality), a direct blocker to training on anything beyond a single scalar field.

**Addendum 4 (2026-07-01) — every open problem now closed; Rayleigh–Bénard set as the first multiphysics target.** A dedicated "make Noether 1.0 trainable on general physics" pass closed the six remaining non-fully-decided problems, with **2D Rayleigh–Bénard convection** (coupled velocity + temperature + pressure, Boussinesq buoyancy, no particles) chosen as the first multiphysics benchmark after 1D Burgers:
- **Problem 8** (field cardinality) — **DECIDED**: S8.1 typed-encoder additive latent + S8.5 universal normalization, S8.4 hypernetwork as zero-shot upgrade. The #1 multiphysics blocker; instantiated for RBC's velocity/temperature field set.
- **Problem 9** (equivariance) — **DECIDED** (full pass): the "commit or not" binary is *dissolved* — symmetry-breaking fields (gravity, $\Omega$, EM) become covariant conditioning (P9.2), making hard equivariance always physically correct; realized selectively via P9.3, gated by the S15.2 ablation, with P9.1 augmentation as the first-pass baseline. RBC's gravity-broken symmetry forced this reframing.
- **Problem 10** (training stability) — **DECIDED** (full pass): P10.1 analytic spectral-norm control + P10.4 ReZero gain (normalization-free depth stability) + P10.3 recipe; promotes [[noether-1.0]] §5/§6 from provisional to decided.
- **Problem 4** (cross-type message *form*) — **DECIDED**: nondimensional scale-bridge (P4.3) → IBM adjoint spread/interpolate (P4.1) → conserved-flux antisymmetry (P4.4) + learned residual (P4.2). Implementation deferred to the S4.4 particle benchmark.
- **Problem 5** (diffusion constraint-projection) — **DECIDED**: conserved-orthogonal channel (P5.5) + stream-function div-free (P5.3) + $x_0$-projection (P5.2); load-bearing for turbulent RBC, deferrable for laminar.
- **Problem 7** (persistent identity) — **DECIDED**: persistent nodes / ephemeral edges (P7.1) + particle-sum anchoring (P7.2) + smooth cutoff (P7.4). Fragmentation flagged as a narrow remaining edge case. Implementation deferred to S4.4.

**Current overall status:** **all 25 tracked problems are DECIDED.** Caveats/deferrals that remain, each noted in its own header: Problems 4 and 7's particle-coupling *implementations* are deferred to the S4.4 settling-particles benchmark (mechanisms fully specified); Problem 9's equivariance *cost* choice is settled by an ablation on the first 2D run (framing decided); Problem 22 (moving boundaries) stays deliberately scoped out; Problem 7's particle *fragmentation* is a flagged narrow edge case. **Direct blockers to the Rayleigh–Bénard target specifically:** Problem 8 (now decided), Problem 2's div-free/BC projection (decided, needs building for 2D), Problem 18's elliptic solve (decided, needs building), and Problem 25's admissibility (decided) — all mechanisms exist; what remains for RBC is *implementation*, not open design. A dedicated `noether-1.0-rbc` config page (the RBC analogue of [[noether-1.0]]) is the natural next artifact.

---

## [AI Inference]

**[AI Inference]:** The strongest evidence that Problem 1 is the crux is that it is *causally upstream* of three others: choosing a hard symplectic/projection time-step (enforce) simultaneously fixes time-conservation (1), provides the natural home for the divergence-free / BC projections (2), and pins down whether the core can stay deterministic (5, since exact-conservation rollouts behave very differently under chaos than free regression). A single "enforce-the-conserved-subspace" decision is the highest-leverage move available; the rest of Tier 1 reorganizes around it.

**[AI Inference]:** Problems 3 and 4 together suggest the tokenizer — not the backbone — is where the initial model is least complete. The backbone (attention) is heavily specified and well-grounded; the tokenizer carries two unsolved load-bearing assumptions (latent adequacy, heterogeneous-node coupling). This inverts the usual intuition that the "interesting" research is in the attention; for *this* model, the riskiest, least-specified component is the graph tokenizer, and that is where a prototype should be stress-tested first (encode→decode fidelity on turbulent data before any dynamics are trained).

---

## Second-pass audit — after the S1.1 / S3.1 / S4.1 / S5.1 decisions (2026-06-30)

**Framing.** The four decisions pick *mechanisms* for Problems 1, 3, 4, 5. They do **not** *close* those problems: each mechanism (a) only partially covers its problem, (b) was chosen without its validation-complement, and (c) introduces new coupling holes. Meanwhile Problems 2 and 6–11 are untouched, and several are now *aggravated*. Honest accounting below.

### A. How much each decision actually closes

- **S1.1 (projection) — closes the aggregate half of Problem 1.** Makes *global totals* trajectory-conserved (kills energy drift / blow-up). Does **not** give *local field fidelity* (a few scalars pinned against millions of DOF), and enforces only *known* invariants exactly (discovered ones stay soft until validated). Depth→time is fixed for the aggregate, not the field.
- **S3.1 (dual-path) — closes the mechanism half of Problem 3, not the guarantee.** Preserves sharp-feature *magnitude*, but S3.4 (measured fidelity floor) was not selected, so "is the latent adequate?" — the *original* Problem-3 gap — is still unanswered.
- **S4.1 (typed edges) — closes the *slot*, not the *physics*, of Problem 4.** Gives heterogeneous nodes a shared, antisymmetry-preserving message channel. The load-bearing part — a *physically meaningful* field↔particle message across mismatched scales/units — is asserted, not designed. Still research-grade.
- **S5.1 (LES diffusion) — closes the *where*, not the *when/how*, of Problem 5.** Puts the generative path where blur happens. But *when* it engages (S5.2 routing) is uncommitted, *how* the generated detail is constraint-projected is a direction not a mechanism, and it worsens the eval-metric and cost problems.

### B. New holes introduced by the decisions

1. **Pipeline ordering is now load-bearing and under-specified.** Path: encode(dual-path) → backbone → residual head → **project (S1.1)** → decode → **diffuse residual (S5.1)** → **re-project** → re-dimensionalize. Two ordering constraints are now hard: projection *after* the output gain (energy), and projection *after* diffusion (generated detail must be re-projected). Wrong order silently breaks conservation.
2. **Residual channel: encode-deterministic but predict-stochastic (S3.1 × S5.1).** At tokenization the high-$k$ residual is reconstructed exactly from the input; at prediction it is *generated* as a sample. Whether one representation serves both — and whether feeding a *generated* residual back as context accumulates rollout error — is unresolved.
3. **Conserved-readout fidelity + energy-projection non-uniqueness (S1.1).** Hard-projecting via a learned readout is only as safe as the readout's accuracy; the quadratic energy projection has a null space (which DOF absorb the correction) needing a canonical choice (min-norm / spectral).
4. **Discovered-invariant identifiability (S1.1 tier 2).** Min-variance probes collapse to trivial constants without a non-degeneracy regularizer; promotion criteria (how constant, over what horizon) are unspecified.
5. **Cross-type message physics (S4.1).** The one genuinely novel research object; no published precedent.

### C. Original problems still open — several now aggravated

- **Problem 2 (enforce vs. condition local constraints) — now has a scored solution set (P2.1–P2.4, above).** The joint KKT projection (P2.2) folds BCs + conservation + divergence into one physical-space solve; the elliptic part is a Leray/Poisson solve on the supernode hierarchy (P2.1); div-free-by-construction (P2.4) removes the solve entirely in 2D. Remaining work is the elliptic-solve *implementation* on the hierarchy and the constraint-compatibility bookkeeping — engineering, not open research. This unblocks the 2D NS smoke test.
- **Problem 6 (loss) is now multi-term and unbalanced.** The decisions *demand* specific losses: spectral/statistical (S5.1), reconstruction (S3.1), conservation penalty for discovered invariants (S1.1). Balancing {residual, spectral, denoising, conservation, reconstruction} is unsolved and now unavoidable.
- **Problem 7 (dynamic topology) worsened by S4.1.** Typed edges between moving particles and field centroids churn every step; edge appearance/disappearance can inject/remove momentum, feeding back into the S1.1 conservation it was meant to protect.
- **Problem 8 (variable field cardinality) untouched.** S3.1/S4.1 don't address how one latent carries 2 vs 5 fields or an unseen field type.
- **Problem 9 (equivariance) still undecided — now with more surfaces.** Should the typed-edge message (S4.1) and the residual channel (S3.1) be equivariant? Leaving it open leaves both under-specified.
- **Problem 10 (training stability) materially worse.** We added a differentiable projection, a dual-path encoder, and a diffusion model atop an already-nonstandard (no-LayerNorm, skew-symmetric, drifting Euler-step) backbone. Joint training of deterministic core + diffusion is notoriously unstable; separate-then-couple is likely required but unspecified.
- **Problem 11 (compute / inference cost) materially worse.** Iterative diffusion per rollout step + typed-edge messages + dual-path + a projection solve all push against the "faster than classical solvers" goal. No cost budget exists.

### D. Validation complements — ADOPTED (2026-06-30)

All four complements are now committed, so the picks are falsifiable: **S3.4** (fidelity floor, gating tokenizer pretraining) for S3.1; **S4.4** (settling-particles coupled benchmark) for S4.1; **S5.2** (in-context regime routing) for S5.1; **S1.5** (push-forward with a $1\to2\to4\to8$ curriculum) for S1.1. What remains is not *whether* to measure but the measurement thresholds themselves (fidelity floor value, routing boundary, horizon schedule tuning) — set during implementation.

**[AI Inference]:** The decisions moved the project from "which mechanism?" to "does the mechanism *work and compose*?" — and the risk has concentrated at **interfaces**, not components. The highest-leverage next step is not another mechanism but a **specified pipeline contract**: the exact stage order and the interface each tier exposes (global-node correction → fine nodes; constraint spec → diffusion; deterministic encode ↔ stochastic predict on the residual). After that, **Problem 2** is the concrete blocker for the first smoke test.

### E. Problem 6 resolved; Problem 12 discovered (2026-07-01 update)

Following the pipeline-contract work, two more items moved: **Problem 6 (loss)** is resolved — four dependency-ordered loss groups (core fit, temporal consistency, conservation-discovery, generative), within-group uncertainty weighting, cross-group staged curriculum, full spec in the new [[training-curriculum]] page. **Problem 12 (newly discovered)** — the backbone as specified has no FFN/per-node nonlinear block, only a single elementwise activation; resolved via a norm-bounded semi-orthogonal FFN sub-step (F1), with the equivariant-gated version (F2) as the upgrade path if Problem 9 resolves to hard equivariance. **Problems 9 and 10** gain provisional (not fully scored) defaults — no hard equivariance (augmentation instead) and analytically-bounded stability step — sufficient to unblock **Problem 11**'s parameter sizing, which is now the concrete next step. **Remaining fully open:** Problem 4's cross-type message form, Problem 7 (dynamic topology/edge churn), Problem 8 (variable field cardinality), and full scored passes on Problems 9 and 10 if more rigor is wanted before sizing.

---

## See Also

- [[initial-model-architecture]] — the design this audits; conservation table and ablations
- [[00-initial-model-overview]] — the commitments some problems conflict with (esp. #2 vs. #4)
- [[graph-tokenizer]] — Problems 3 and 4 live here
- [[symmetric-attention-physics]] / [[skew-symmetric-attention]] — the depth-conservation properties that Problem 1 shows are not time-conservation
- [[dynami-cal-graphnet]] / [[hamiltonian-neural-networks]] — the integrator-based conservation the direct-state choice gave up
- [[normalization-scheme]] — relevant to Problem 10 (training stability)
- [[latent-diffusion-physics]] / [[diffusion-models-physics]] — relevant to Problem 5 (generative core)
- [[possible-architectures]] — the smoke-test benchmark where Problems 2 and 3 first bite
- [[pipeline-contract]] — the forward-pass ordering Problem 12's FFN sub-step and Problem 1's projection both live inside
- [[training-curriculum]] — the training-time resolution of Problem 6
- [[multihead-attention]] — Problem 12's F2 equivariant-gate upgrade path, and Problem 17's tensor-typed heads
- [[intelligent-patching]] — Problem 13's octree, extended to general geometry
- [[noether-1.0]] / [[smoke-test-bringup]] — the 1D Burgers instantiation Tier 3 generalizes beyond
- [[noether-1.0-rbc]] — the 2D Rayleigh–Bénard config instantiating the P8/P9/P10/P2.4/S14 decisions (the first multiphysics target)
- [[equivariant-gnns]] / [[equiformer-v3]] — the sample-efficiency case and tensor-product cost behind Problems 15/17
