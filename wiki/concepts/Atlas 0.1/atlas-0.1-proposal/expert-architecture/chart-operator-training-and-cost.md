# Training the chart operator — sources, stages, costs, and fine-tuning against training from scratch

**Type:** Concept page — **training plan and cost model** (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** written 2026-09-30. **No cost on this page is measured.** Each estimate is **[AI Inference]**, shown with the arithmetic and assumptions behind it, so that the first measurement can replace it line by line. The vault's rule is that a number is not reported until it is measured. These are planning figures for deciding what to measure first, and none should reach the website.
**Updated 2026-10-01 for the confirmed design** ([[chart-operator-architecture]] version 1; decisions 15–18 of [[chart-operator-design-decisions]]): the network now returns constraint modes, so §3's objective is the energy-norm error of the modes, with a label-free objective that has the same gradients; the step-doubling term applies to flow networks only; §5 gains **measured** cost components (classical solves and untrained forward passes); §6's gate is replaced by gates E1–E6, SE1–SE3 and comparisons C1–C6. Estimates stay labelled.
**Hub:** [[00-proposal-workstreams]] · **Architecture:** [[chart-operator-architecture]] · **Prior art:** [[dd-neural-prior-art-2026]]
**Earlier plans this replaces for learned experts:** [[impl-atlas-0.1-phase2-experts]] and [[impl-atlas-0.1-compute-and-training-budget]], written 2026-08-09 for the rocket before any of S1–S12 was measured, and not executed

---

## 0. The short version

**Train small, from scratch, on the vault's own classical solvers, one family at a time, and measure the iteration count, not only the error.**
- **Stage 0**, 2-D diffusion on random charts, is cheap enough to run on the owner's laptop plus a few rented GPU-hours. It tests every architectural claim of [[chart-operator-architecture]].
- **Fine-tuning a foundation model** is a controlled comparison at Stage 2, not the starting point. A chart turns any piece into the uniform grid those models accept, which is new leverage. But their size works against the cost bar S7, and the one checkpoint measured here had the wrong slow-mode response (S5).
- **The funded scale**, three dimensions and several families, is Stage 3. Its cost is estimated here as an order of magnitude, to be priced from Stages 0–2's measured throughput.

---

## 1. Where the training data comes from

### 1.1 The vault's own solvers: the primary source

**Why primary:** unlimited, licence-clean, and matched exactly to the expert contract. Every sample can carry the pulled-back metric, the side waves at any impedance, and the **classical Jacobian–vector products** the loss needs (§3). No public dataset has the last two.

**The random chart generator.**
- Pieces are drawn as random closed spline shapes (`atlas/workbench/shapes.py` already draws centripetal Catmull–Rom curves), cut into four sides.
- They are charted by §2.2 of the architecture page and **binned by conformal modulus** and by $\max J/\min J$, so coverage of the chart envelope can be measured.
- Coefficient fields are Gaussian random fields; each material jump is a random spline interface.
- Side data are random waves on each side, drawn **with a controlled mean frequency**. The 2026 topology benchmark found that frequency, not topology, drove out-of-distribution error ([[dd-neural-prior-art-2026]] §1.3).
- The impedance is log-uniform over the Expert Card's range, and $\Delta t$ log-uniform over its range.

**Per family, the solver that makes the samples:**

| family | classical solver in the repo | what a sample costs |
|---|---|---|
| diffusion (THERM) | `atlas/workbench/fv.py` on the chart's cells, or a finite-difference solve on the reference square with the pulled-back coefficients | one sparse factorization; every JVP direction reuses it, so a Jacobian sample is two triangular solves |
| advection–diffusion (ADVEC) | `fv.py` with `flow.py`'s potential flow | the same, plus the flow |
| plane elasticity (MECH) | `atlas/workbench/fe.py` | one factorization; JVPs as above |
| current (ELEC) | `fv.py` | as diffusion |
| incompressible flow (MECH) | the wake array's `WindowNS`; the overset solver verified on the cylinder at $\mathrm{Re}=100$ ([[poc3-racelab-overset-flow]]) | a march per sample; JVPs by finite differences at float64 (S10) |

### 1.2 Public datasets: pretraining and outside evaluation

**Every licence here is to be verified before use.** None was read in the planning session. The owner's rule on downloads applies: ask first.

| dataset | what it is | possible use |
|---|---|---|
| **The Well** (Polymathic AI; Ohana et al., *NeurIPS* 2024 Datasets) | about 15 TB across 16 physics simulation sets, uniform grids; Walrus was trained on it ([[walrus-paper]]) | pretraining for Stage 2; outside evaluation |
| **PDEBench** (Takamoto et al., *NeurIPS* 2022 Datasets) | standard 1–3-D benchmark PDEs | a sanity baseline against published models |
| **PDEArena** (Gupta & Brandstetter, 2022) | 2-D fluid sets at scale | Stage 2 pretraining |
| **AirfRANS** (Bonnet et al., *NeurIPS* 2022 Datasets) | 2-D RANS around airfoils on irregular meshes | outside evaluation of irregular geometry |
| **DrivAerNet++**, **DrivAerML**, **AhmedML**, **WindsorML** (car aerodynamics, 2024) | 3-D automotive CFD at thousands of designs | Stage 3, and the F1 vision ([[vision-scenarios-and-image-prompts]]); **several of these carry non-commercial terms** — check each |

---

## 2. The stages

| stage | family and scope | what it proves | gate |
|---|---|---|---|
| **0** | 2-D **diffusion**, steady and transient, random charts, THERM ports | geometry through the chart; the constraint-mode output; the tier-0 Gram coupling; the forward cost against the classical alternatives (version 0 listed the lifting, wave-variable convergence and the contraction certificate; those now belong to tier 2) | §6, on the workbench's own drawn examples: `bend-3`, `s-channel`, `insert-round` |
| **1** | 2-D **advection–diffusion** (ADVEC), **plane elasticity** (MECH), **current** (ELEC) | the contract across port types; a multiphysics seam (the cooled block's) with two learned experts | §6, per family, plus the cooled block's seam |
| **2** | 2-D **incompressible flow** windows | a nonlinear family; the demo's learned case generalised from rectangles to charts ([[demo-learned-case-plan]]); **the fine-tuning comparison** (§4) | §6, on the wind farm and the cylinder |
| **3** | **three dimensions** (hexahedral charts where they exist, a point-based fallback elsewhere; [[chart-operator-architecture]] §2.5); compressible and reacting families | the vision's scale: a car, a fire front, a plume on regolith ([[vision-scenarios-and-image-prompts]]) | set after Stage 2 |

---

## 3. The objective and the data it needs

*Version 0's table (state, outgoing waves, a response term, step consistency, a Lipschitz penalty) is superseded by the constraint-mode output; it remains the plan for tier-2 experts, which exchange waves.*

**Intuition.** At tier 0 the coupled error is controlled by the energy of the network's field errors (Theorem 1 of [[chart-operator-architecture]] §3.6), so the loss measures exactly that.

**The supervised objective** (the document's equation 32):
$$\mathcal L_{\text{sup}}(\theta)=\mathbb E\Bigl[\sum_{k=1}^{4m}\lVert\hat H_k(\theta)-H_k\rVert_{A_I}^2+\lVert\hat u_p(\theta)-u_p\rVert_{A_I}^2\Bigr],$$
over random charts, coefficients, sources, Fourier numbers and the eight symmetries of the square.

**The label-free objective** (the document's Corollary 11):
$$\mathcal L_{\text{en}}(\theta)=\operatorname{tr}\tilde\Lambda(\theta)+\hat u_p^\top A_I\hat u_p-2f^\top\hat u_p=\mathcal L_{\text{sup}}\text{'s summand}+\underbrace{\operatorname{tr}\Lambda-u_p^\top A_Iu_p}_{\text{independent of }\theta}.$$
*Proof:* $\operatorname{tr}\tilde\Lambda-\operatorname{tr}\Lambda=\sum_k\lVert E_k\rVert^2_{A_I}$ by the Gram identity, and completing the square for the particular field. **The two have the same gradients**, so training needs the energy matrix $M$ of each example, not its solution; labelled examples are still needed for validation, because $\mathcal L_{\text{en}}$'s value carries the unknown constant. It is the discrete Ritz principle (Deep Ritz: E & Yu, *Commun. Math. Stat.* 6, 2018). In floating point the gradient involves the cancellation $A_I\hat H-BQ$, so which trains better is comparison C2.

| term | needs, per example | extra data cost |
|---|---|---|
| modes and particular field, supervised | $H=A_I^{-1}BQ$ and $u_p=A_I^{-1}f$ | one sparse factorisation of $A_I$ and $4m+1$ pairs of triangular solves |
| the same, label-free | the matrix $M$ | assembly only |
| step doubling | two steps of $\Delta t$ against one of $2\Delta t$ | **flow networks only** (Stage 2): the tier-0 networks approximate the backward-Euler step, and $(I+2\Delta t\,L)^{-1}\ne(I+\Delta t\,L)^{-2}$, so the penalty would pull them off their target |
| tier-2 response and Lipschitz terms | as version 0 | for experts used at tier 2 only |

---

## 4. Fine-tuning against training from scratch

**The new leverage.** A chart maps any piece onto the uniform grid that every uniform-grid foundation model accepts. So S1, the frozen checkpoint's *"accepts only a uniform $128\times128$ grid"*, stops being a wall for those models: the geometry arrives as extra channels on the grid they already read. **[AI Inference]:** this is the strongest argument for trying fine-tuning at all.

| | **from scratch, small** | **fine-tune a foundation model** |
|---|---|---|
| candidates | a U-shaped convolutional operator, about $10^5$–$10^7$ parameters | **DPOT** (Apache-2.0), **Walrus** (MIT), **GPhyT** (MIT); **Poseidon excluded** by its CC-BY-NC-4.0 weights, a term that carries into anything fine-tuned from them ([[expert-donor-survey]], S12) |
| new inputs (metric, side waves, impedance, $\Delta t$) | native | new input encoders; the pretrained trunk kept, adapted by low-rank updates |
| data needed | more | less: the documented benefit of pretraining ([[transfer-learning-fine-tuning]]; Poseidon's paper reports gains in sample efficiency, [[poseidon-pde-foundation-model]]) |
| inference cost (S7) | small by design, on a CPU | large. Poseidon-T, the smallest checkpoint measured here, took $86.1$ ms per agent per step ([[outcome-c3-learned-speed-at-scale]] §2), and the others are larger. **Distillation into a small model** is the way back under the bar |
| the slow-mode response (S5) | trained for it from the start | Poseidon-T's measured slow-band response was wrong ($\psi_{\text{slow}}=0.558$ against the classical $0.478$; [[corrupted-checkpoint-and-jacobian-fidelity]]). Whether fine-tuning with §3's term repairs a pretrained trunk is **unmeasured** |
| licence | the owner's | the donor's terms, which propagate |
| risk | under-training on narrow data | negative transfer, since pretraining never saw a metric or a Robin side; and cost |

**Recommendation.**
- **From scratch for Stages 0–1.** The families are linear or nearly so, the data are cheap, and small models are what S7 needs.
- **At Stage 2, a controlled comparison at matched GPU-hours:** (i) from scratch, (ii) DPOT or Walrus fine-tuned with new encoders and low-rank adaptation, and (iii) (ii) distilled into (i)'s size. Each is judged by §6's gate, not by training loss.
- **F9's discipline applies:** price the fine-tune by a timed step before running it ([[learned-contribution-kill-tests]] §4.7).

---

## 5. The cost model

**Data**, on CPU cores:

$$C_{\text{data}}=\frac{N_s\,\bigl(t_{\text{solve}}+m\,t_{\text{JVP}}\bigr)}{n_{\text{cores}}}.$$

**Training**, on a GPU:

$$C_{\text{train}}=\frac{E\,N_s\,t_{\text{fb}}}{B}\quad\text{GPU-seconds},$$

with $E$ epochs, batch $B$ and $t_{\text{fb}}$ the forward-and-backward time of one batch, including the Jacobian term's extra passes. **Each of $t_{\text{solve}}$, $t_{\text{JVP}}$ and $t_{\text{fb}}$ is one timing to take before a stage starts.**

**Measured components (2026-10-01).** In the cloud container (4 cores; ratios matter, not seconds):

| component | value | script |
|---|---|---|
| one sparse factorisation of a $64^2$ piece | about 21 ms | `scripts/arch_coupling_cost.py` (4 × 4 pieces, $n=64$) |
| one pair of triangular solves at $n=64$ | about 0.94 ms (47 ms for 50) | same |
| forward pass of the 0.49 M / 1.95 M / 6.2 M network at $n=64$, batch of 16 | 3.8 / 7.7 / 25 ms per piece | `scripts/arch_backbone_timing.py` |

**From these, [AI Inference]:** one labelled example at $n=64$ costs about $21+65\times0.94\approx82$ ms, so $10^5$ examples need about 2.3 core-hours, and none with the label-free objective. Training the 1.95 M network on the container's CPU, with a forward-and-backward pass taken as three forward passes, would need about $3\times7.7$ ms per example, roughly 64 hours for 100 passes over $10^5$ examples; hence Stage 0 on a rented GPU.

**Planning estimates, [AI Inference], each assumption named:**

| stage | assumptions | data | training | at the \$0.70/h cap |
|---|---|---|---|---|
| **0** | $N_s=10^5$ charts at $64^2$–$128^2$; $t_{\text{solve}}\sim10$–$50$ ms and $m=4$ cheap JVPs; 20 cores; a $10^6$–$5\times10^6$-parameter network, 100 epochs | **under an hour** on the laptop | **a few GPU-hours** | **a few dollars** |
| **1** | three families, each like Stage 0 | a few hours | $\sim10$ GPU-hours | $\lesssim\$10$ |
| **2** | $10^5$ flow windows from classical marches at seconds each; the three-arm fine-tuning comparison | **days of CPU**, or a rented CPU box, as the rocket's tiers used at \$0.17–\$2.34 each ([[vast-ai-windfarm-runbook]]) | $10$–$10^2$ GPU-hours across the three arms | $\$10$–$\$70$ |
| **3** | three dimensions, several families, $10^5$–$10^6$ samples of $10^5$–$10^6$ unknowns each | $10^4$–$10^5$ core-hours: a cluster or a cloud budget | $10^3$–$10^4$ GPU-hours on data-centre cards | **thousands to tens of thousands of dollars at market rates**, which exceed the \$0.70/h cap. **This is the funded scale** |

**The long-run rule applies from Stage 1 on:** anything expected to take over about two hours is announced first and saves its state, especially on failure. GPU instances are destroyed after use, as in [[vast-ai-windfarm-runbook]].

---

## 6. The gate each stage must pass

**Registered in code before the stage's first training run**, with numerical bars, so the results cannot shape the criteria (decision 18; the document's Tables 11–12). Benchmarks: the workbench's s-channel, bend-3 and insert-round, excluded from training.

| # | measure | bar |
|---|---|---|
| E1 | energy-norm error of the modes and particular field on held-out pieces, **by $\sup\lvert\mu\rvert$ bin and by input frequency** | registered per family |
| E2 | tier 2 only: the learned scattering map's contraction factor on the fine space, certified and estimated | certified below one in the envelope, or the seam admitted without a convergence claim |
| E3 | tier 0: coupled error against the classical superelement **with the same port modes** (isolates the network); tier 2: rounds to tolerance against classical local solves at the same transmission condition | tier 0: within Theorem 2's bound evaluated with the measured field errors; tier 2: at most $1.5\times$ the classical count |
| E4 | the coupled answer against the full-domain classical solution on the benchmarks | registered per family |
| E5 | wall time against the classical superelement with stored modes, the classical decomposition, a monolithic direct solve and a coarse classical solver, **on the owner's laptop and one GPU** | reported whatever it says; "faster" claimed only where measured faster |
| E6 | S1–S12 of [[outcome-c5-requirements-for-dd-native-experts]], row by row | each answered, or named as not met |
| SE1 | $\lambda_{\min}(\tilde{\mathsf S})\ge\lambda_{\min}(\mathsf S)$ and the Gram identity on every case | to rounding (an implementation test) |
| SE2 | network calls per piece per linear solve | one |
| SE3 | 16-mode truncation error on the benchmarks | below the network's E1 error |

| # | registered comparison (Stage 0) | question |
|---|---|---|
| C1 | general charts against near-conformal pieces only | how much harder the general input class is |
| C2 | label-free against supervised objective, equal compute | whether the exact equivalence survives floating point |
| C3 | principal part plus correction against direct prediction of the modes | whether the principal part helps |
| C4 | U-shaped convolutional trunk against the sine–cosine spectral operator | which backbone suits the square |
| C5 | 0.3, 1 and 3 M parameters | accuracy per $\lvert\mu\rvert$ bin against forward cost (S7) |
| C6 | cosine port modes against sines plus corner functions | the corner effect of [[chart-operator-architecture]] §3.3 |

**[AI Inference]:** E3 at tier 0, against the classical superelement with the same modes, isolates exactly what the network adds; no published work in [[dd-neural-prior-art-2026]] reports that comparison.

---

## 7. Risks

1. **The cost bar S7.** In linear families a classical local solve with a stored factorization is a pair of triangular solves, cheaper than any network. **[AI Inference]:** a learned expert pays in linear families only where it replaces something expensive: a global coupling iteration (fewer Schwarz sweeps, E3), or a factorization that cannot be stored (a changing geometry or coefficient in a design loop). Stage 0 must measure this, not assume it.
2. **Loose certificates**, now a tier-2 risk only ([[chart-operator-architecture]] §3.7); tier 0 is certified by construction.
   - **Measured since (2026-10-01):** on a fixed 2-D geometry a monolithic direct solve and stored classical modes beat any network; the learned superelement is 4–9 times cheaper than recomputing a port matrix classically, which pays only when geometry or material change ([[chart-operator-architecture]] §5).
3. **The chart envelope** excludes pieces the layout makes. The layout then has to cut again, which is a compiler rule to write.
4. **Negative transfer** in the fine-tuning arm.
5. **Three dimensions** ([[chart-operator-architecture]] §2.5).

---

## See Also

- [[chart-operator-architecture]] — the design being trained
- [[demo-learned-case-plan]] — the demo's single learned case, a rectangular-chart instance of Stage 2
- [[expert-donor-survey]] — licences and inputs of every downloadable model
- [[learned-contribution-kill-tests]] — how this vault prices and kills a learned contribution
- [[vast-ai-windfarm-runbook]] — the rented-compute workflow
- [[outcome-c5-requirements-for-dd-native-experts]] — the specification the gate checks
