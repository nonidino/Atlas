# Data Augmentation for Physics Surrogates

**Type:** Concept page — method (folder: `concepts/ml-building-blocks/`)
**Related:** [[physics-simulation-datasets]], [[autoregressive-rollout-stability]], [[poseidon-pde-foundation-model]], [[walrus-paper]], [[transfer-learning-fine-tuning]], [[impl-atlas-0.1-corpus-completion-plan]]
**Written:** 2026-08-09, answering "does data augmentation not work?" against Atlas 0.1's corpus cost.

> **Verdict up front:** augmentation works, and one form of it — **all2all / semi-group pair amplification** — is worth roughly $100\times$ on Atlas's coupled corpus and is *already licensed by the architecture*, which conditions on $\Delta t$. But the classical symmetry augmentations mostly do **not** apply, for reasons specific to Atlas's geometry and nondimensionalization. And no augmentation of any kind generates new *interface* configurations, which is precisely where the data is scarce.

---

# 1. Intuition

## 1.1 The governing principle

**Augmentation adds zero new physics.** It multiplies samples along directions where you *already know* the answer, by asserting an invariance or equivariance you are confident holds. It is a way of telling the model something you know, using data as the medium.

That makes the useful question not "does augmentation work?" but:

> **Along which directions do I know the answer without computing it — and are those the directions my model is starved in?**

Augmentation pays when those two sets overlap. The recurring disappointment in physics surrogates is that they often don't: you are rich in interior samples and poor in boundary/interface ones, and the symmetry groups act on interiors.

## 1.2 The three families, ranked by what they actually buy

| Family | Mechanism | What it buys |
|---|---|---|
| **Symmetry augmentation** | apply a Lie point symmetry of the PDE (translation, rotation, Galilean boost, scaling, reflection) and transform the target accordingly | new samples, *if* the symmetry survives your boundary conditions |
| **Semi-group amplification** | train on all $(t_i \to t_j)$ pairs with lead-time conditioning, not just consecutive ones | $\mathcal O(K^2)$ pairs from $K$ snapshots — **usually the largest single win** |
| **Perturbation / robustness** | input noise, patch jittering, dropout | rollout stability, **not** sample count |

The third is often mislabelled augmentation. Training noise and patch jittering do not teach new physics or add effective samples; they regularize against a specific failure mode (error accumulation, patch aliasing). Worth doing, but do not budget them as data.

---

# 2. Theory

## 2.1 Symmetry augmentation and why boundaries kill it

The Lie point symmetry group of the compressible Navier–Stokes equations includes space and time translation, rotation, Galilean boost, reflection, and a scaling family. Applying $g$ in that group to a solution $u$ yields another exact solution $g\cdot u$ — genuinely free data, and the basis of Lie-point-symmetry augmentation for neural PDE solvers.

**The catch is that symmetries are properties of the *equation*, and training data is a property of the *boundary-value problem*.** A symmetry survives augmentation only if it maps the domain and boundary conditions to themselves. Against a fixed engineering geometry, most do not:

| Symmetry | Survives a fixed nozzle/airframe domain? |
|---|---|
| Space translation | **No** — the injector face, throat, and exit plane are at fixed $z$ |
| Rotation | **No** — the geometry is not rotationally symmetric |
| Galilean boost | **No** — walls are at rest in the body frame; boosting makes them move |
| Time translation | **Yes**, if the problem is autonomous (§2.2) |
| Reflection about the symmetry plane | **Yes**, but see §3.2 — usually already exploited or already swept |
| Scaling | **Yes in principle** — but see §2.3, which is the subtle one |

This is a general lesson, not an Atlas quirk: **periodic-box benchmarks are far more augmentable than engineering geometries**, which is part of why augmentation results reported on PDEBench-style data transfer poorly to applied surrogates. It is the same asymmetry noted in [[physics-simulation-datasets]] §3.5 — periodic-BC data does not teach wall behaviour — seen from the other side.

## 2.2 Semi-group amplification — the one that pays

For an autonomous PDE the solution operator forms a one-parameter semi-group:

$$\mathcal S(t)\circ\mathcal S(s)=\mathcal S(t+s),\qquad u(t)=\mathcal S(t)\,u(0)$$

So **every ordered pair of snapshots in a trajectory is a valid training example**, not just consecutive ones, provided the model is conditioned on the lead time $\tau = t_j - t_i$. From $K$ snapshots this gives

$$\binom{K}{2}=\frac{K(K-1)}{2}\quad\text{pairs instead of}\quad K-1$$

an amplification of $K/2$. This is [[poseidon-pde-foundation-model]]'s **all2all training**: 77,840 trajectories $\times$ 11 snapshots become ~5.11 M training pairs. [[pfm-architecture-approaches]] already records it as "architecture-agnostic and applicable to any time-dependent-PDE PFM."

**Two conditions must hold**, and both are checkable:

1. **The model must take lead time as an input**, not bake it into the architecture.
2. **The problem must be autonomous** — the operator cannot depend on absolute time. A system whose external forcing drifts over the episode is non-autonomous and the semi-group property fails *unless* the drifting quantity is itself part of the conditioned state.

## 2.3 The subtlety: nondimensionalization has already eaten the scaling symmetry

Compressible flow has a scaling symmetry: rescale lengths, velocities, and densities consistently and you get another solution, at a different Reynolds number. It looks like free augmentation.

It usually isn't, and the reason is worth internalizing. **Nondimensionalization is a quotient by exactly that group.** If a model consumes $\hat\rho, \hat u, \hat p, \hat T$ together with $(\mathrm{Ma}, \mathrm{Re}, \mathrm{Pr}, \gamma)$, then two dimensionally-different flows with identical dimensionless numbers are *the same input*. Rescaling produces a duplicate sample, and augmenting along the direction the representation already quotiented out adds nothing but training time.

**[AI Inference]:** this generalizes into a useful design check — **enumerate the symmetries your representation has already quotiented out, and cross them off the augmentation list first.** Whatever a well-designed nondimensionalization buys you in sample efficiency, it correspondingly removes from the augmentation menu. The two are the same saving, counted once.

## 2.4 The hard limit: augmentation cannot manufacture interfaces

Every mechanism above acts on a *solution field*. Interface supervision is a pair — a boundary flux and the response it induces in the neighbouring subdomain — arising from the **coupled** state of two agents.

To generate a new valid interface sample by transformation, the transformation must be a symmetry of the *coupled system*, including both subdomains' geometries and the interface itself. The geometry breaks nearly all of them (§2.1), and the ones that survive are already swept.

So the general shape of the result:

> **Augmentation multiplies data where you are already rich (interiors) and does nothing where you are poor (interfaces).**

Which is exactly the gap [[physics-simulation-datasets]] §4 identifies as structural and permanent.

---

# 3. Implementation — what applies to Atlas 0.1

## 3.1 ✅ All2all pair amplification — the recommendation

**Atlas is already architecturally licensed for this and the specs never invoke it.** [[impl-atlas-0.1-phase2-experts]] §2.1 states the expert interface as

$$\mathcal E_\theta:\ (Z,\ \mathbf z_{\text{cond}},\ \Delta t)\ \longmapsto\ Z'$$

with the explicit note that "$\Delta t$ enters as a conditioning input (log-scaled), never as a fixed architectural constant." That is condition 1 of §2.2, satisfied by design.

**The arithmetic on the B′ corpus:** 200 snapshots per episode gives $\binom{200}{2} = 19{,}900$ ordered pairs against 199 consecutive ones — a **$\sim\!100\times$ amplification**, on the corpus that costs ~1,000 core-hours to produce.

**Condition 2 is satisfied better under B′ than it would have been under the original spec**, which is a genuine and unplanned synergy. The coupled system is non-autonomous in principle — the vehicle climbs, the freestream changes — but (i) altitude, Mach and Reynolds enter through the FiLM conditioning, so the extended state *is* autonomous, and (ii) over B′'s $T_{\text{ep}} = 0.2$ s the vehicle moves ~0.4 m and the drift is negligible regardless. **The short-horizon choice made for cost also makes the semi-group assumption more nearly exact.**

Three implementation cautions:

- **Restrict the lead-time range for the rollout curriculum.** All2all pairs span $\tau$ from $10^{-3}$ s to 0.199 s — up to $200\times$ the design $\Delta t_{\text{model}}$. Use the full range for *pretraining the operator*, then train the push-forward curriculum ([[impl-atlas-0.1-phase2-experts]] §3.2) at the fixed design $\Delta t$, because that is what the multi-rate stepper actually executes.
- **Sample $\tau$ log-uniformly**, not uniformly. Uniform sampling over pairs concentrates mass at large $\tau$ (there are many more distant pairs than near ones), starving exactly the step size the model must be good at.
- **$\mathcal L_{\text{iface}}$ still works pairwise** — the interface flux at $t_j$ is a valid target for a jump from $t_i$. The *conservation projection*, however, is a per-step operation at inference; do not apply it across a large-$\tau$ training pair.

## 3.2 ⚠️ Reflection — free but nearly worthless here

Reflection about $y=0$ is a real symmetry of the planar geometry. But the sweep runs $\alpha \in \{0°, 2°, 5°\}$ and the mirror BC is *already* enabled at $\alpha = 0$ ([[impl-atlas-0.1-phase0-scope-and-data]] §2.1), so:

- At $\alpha = 0$: the flow is already symmetric. The reflected sample is **bit-identical**. Zero gain.
- At $\alpha \neq 0$: reflection yields the $-\alpha$ case. Real, but it doubles only the two-thirds of the sweep with nonzero incidence, and the model arguably should learn $\pm\alpha$ symmetry rather than be handed it.

Take it — it costs one line — but do not budget it as a corpus reduction.

## 3.3 ❌ Patch jittering — specifically unavailable, for a recorded reason

[[walrus-paper]]'s patch jittering dithers the patch grid to spread tokenization aliasing across frequencies, improving long-horizon stability in 89% of pretraining scenarios. It is one of the wiki's most-cited stability techniques ([[arch-autoregressive-transformer]], [[autoregressive-rollout-stability]]).

**It cannot be used in Atlas as built.** [[atlas-0.1-implementation-log]] (2026-08-09, `b`/`e` body-fitting) records that `geometry/domains.py` now derives the model's cells *from* `solvers/grid.build_blocks`, so that "no interpolation between solver and model" is true by construction — and states the reason plainly: **interface fluxes do not survive interpolation.** Jittering the patch grid reintroduces exactly that interpolation at exactly the tokens where it is least acceptable.

**[AI Inference]:** if patch-aliasing artifacts do show up in Phase 4 rollouts, the Atlas-compatible substitute is jittering *only within agent interiors* while pinning boundary-token patches to the solver cells — preserving the interface correspondence at the cost of a discontinuity in patch phase near the boundary. Untested, and a Phase 5 concern at the earliest.

## 3.4 ✅ 2D slicing of 3D data — already in the plan

Extracting planar slices from 3D DNS volumes is augmentation, and it is the single largest sample multiplier available: [[physics-simulation-datasets]] §3.1 estimates BLASTNet's 744 full-domain samples yield $\mathcal O(10^5\!-\!10^6)$ 2D slices. [[walrus-paper]]'s 2D↔3D augmentation is the precedent. Already tier T1.

## 3.5 ⚠️ Noise injection — robustness, not data, and it must be one-sided

Input-noise training is a standard mitigation for autoregressive error accumulation ([[autoregressive-rollout-stability]]) and Atlas should use it. But two constraints:

- It adds **no effective samples.** Do not count it against the corpus budget.
- **Perturb inputs only; leave targets untouched.** Perturbing an $(\text{input}, \text{target})$ pair jointly produces a state that violates the flux matching the corpus was validated for (gate G5 in [[impl-atlas-0.1-corpus-completion-plan]]), training the model to reproduce a conservation violation. The correct formulation — perturb the input, predict the *unperturbed* next state — is also what makes it a contraction toward the data manifold rather than a smearing of it.

## 3.6 Summary for Atlas

| Technique | Applies? | Gain |
|---|---|---|
| **All2all pair amplification** | ✅ already licensed by the $\Delta t$-conditioned interface | **~$100\times$ on the coupled corpus** |
| 2D slicing of BLASTNet | ✅ tier T1 | $\mathcal O(10^3)$ per volume |
| Reflection ($\alpha\neq0$) | ⚠️ real but marginal | $<2\times$ on part of the sweep |
| Scaling / nondimensional | ❌ already quotiented out (§2.3) | none |
| Translation, rotation, Galilean | ❌ broken by fixed geometry | none |
| Patch jittering | ❌ breaks solver↔token cell correspondence | (stability, not data) |
| Input noise | ✅ but one-sided | stability, not data |

**The honest bottom line:** all2all is a large and real win that should be implemented in the training harness, and it makes the corpus-size question less pressing. It does **not** remove the need to generate coupled episodes, because it amplifies *time pairs within* episodes rather than producing new interface configurations — and the number of distinct physical regimes the corpus covers is set by the sweep, not by the pair count.

---

# 4. Pitfalls

- **Counting augmented samples as independent ones.** $\binom{K}{2}$ pairs from one trajectory are heavily correlated; they buy far less than $\binom{K}{2}$ independent samples would. They are still worth having — Poseidon's results are the evidence — but corpus-size decisions should be made on *distinct physical configurations*, not on the amplified count.
- **Augmenting along a direction the representation already quotiented out.** §2.3. Costs training time, returns duplicates.
- **Assuming a PDE's symmetry group survives your boundary conditions.** It usually doesn't in engineering geometries. Check the domain and BCs, not the equation.
- **Uniform sampling of all2all lead times.** Distant pairs vastly outnumber near ones; sample $\log\tau$ uniformly or the design step size is starved.
- **Treating noise injection and patch jittering as data.** They are regularizers against specific failure modes.
- **Perturbing inputs and targets together.** Trains the model to reproduce a conservation violation.
- **Expecting augmentation to fill a boundary/interface gap.** It acts on fields; it cannot manufacture the coupled two-subdomain states that interface supervision consists of (§2.4).

---

## See Also

- [[physics-simulation-datasets]] — what public data covers; the permanent interface gap this page cannot close
- [[impl-atlas-0.1-corpus-completion-plan]] — the corpus work all2all amplifies
- [[poseidon-pde-foundation-model]] — all2all / semi-group training, and the 5.11 M-pair result
- [[walrus-paper]] — patch jittering and 2D↔3D augmentation
- [[autoregressive-rollout-stability]] — where noise injection and jittering actually belong
- [[impl-atlas-0.1-phase2-experts]] — the $\Delta t$-conditioned expert interface that licenses all2all
- [[pfm-architecture-approaches]] — records all2all as architecture-agnostic
