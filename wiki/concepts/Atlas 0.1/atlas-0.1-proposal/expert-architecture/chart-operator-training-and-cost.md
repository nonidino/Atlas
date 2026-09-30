# Training the chart operator — sources, stages, costs, and fine-tuning against training from scratch

**Type:** Concept page — **training plan and cost model** (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** written 2026-09-30. **No cost on this page is measured.** Each estimate is **[AI Inference]**, shown with the arithmetic and assumptions behind it, so that the first measurement can replace it line by line. The vault's rule is that a number is not reported until it is measured. These are planning figures for deciding what to measure first, and none should reach the website.
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
| **0** | 2-D **diffusion**, steady and transient, random charts, THERM ports | geometry through the metric; the lifting; wave-variable DD convergence with learned experts; the contraction certificate | §6, on the workbench's own drawn examples: `bend-3`, `s-channel`, `insert-round` |
| **1** | 2-D **advection–diffusion** (ADVEC), **plane elasticity** (MECH), **current** (ELEC) | the contract across port types; a multiphysics seam (the cooled block's) with two learned experts | §6, per family, plus the cooled block's seam |
| **2** | 2-D **incompressible flow** windows | a nonlinear family; the demo's learned case generalised from rectangles to charts ([[demo-learned-case-plan]]); **the fine-tuning comparison** (§4) | §6, on the wind farm and the cylinder |
| **3** | **three dimensions** (hexahedral charts where they exist, a point-based fallback elsewhere; [[chart-operator-architecture]] §2.5); compressible and reacting families | the vision's scale: a car, a fire front, a plume on regolith ([[vision-scenarios-and-image-prompts]]) | set after Stage 2 |

---

## 3. The loss and the data it needs

From [[chart-operator-architecture]] §5, with what each term costs to supply:

| term | needs, per sample | extra data cost |
|---|---|---|
| state | the solved field | the solve |
| outgoing waves | the side traces and fluxes | none: read off the solve |
| **response to interface data** | $D_aS\,v$ for $m$ random low-band directions $v$ | $m$ JVPs. For a linear family with a stored factorization, $2m$ triangular solves; for flow, $m$ extra marches or one adjoint |
| step consistency | the expert's own two half steps | none: a training-time cost |
| Lipschitz penalty | a power iteration on the expert's Jacobian | none: a training-time cost |

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

**Registered in code before the stage's first training run.**

| # | measure | bar (draft) |
|---|---|---|
| E1 | one chart's accuracy on held-out shapes, **reported by modulus bin and by input frequency** | registered per family |
| E2 | the scattering map's Lipschitz constant on the fine space, **certified** and estimated | certified $<1$ in the envelope, or the seam is admitted uncertified |
| E3 | **Schwarz iterations to tolerance** on held-out domains (the workbench's drawn examples), against the same scheme with classical local solves at the same impedance | no more than $1.5\times$ the classical count |
| E4 | the assembled answer against the full domain | registered per family; for a linear family, within E1's error times the bound of [[chart-operator-architecture]] §4.4 |
| E5 | wall time against the classical decomposition, the full domain, and **the coarse classical competitor** (S7) | faster than the classical decomposition; the coarse comparison reported whatever it says |
| E6 | the S1–S12 checklist of [[outcome-c5-requirements-for-dd-native-experts]], row by row | each row answered, or named as not met |

**[AI Inference]:** E3 is the measure no published work in [[dd-neural-prior-art-2026]] §1.1 reports against a like-for-like classical local solver. Making it the headline is how the proposal's claim N3 is tested rather than asserted.

---

## 7. Risks

1. **The cost bar S7.** In linear families a classical local solve with a stored factorization is a pair of triangular solves, cheaper than any network. **[AI Inference]:** a learned expert pays in linear families only where it replaces something expensive: a global coupling iteration (fewer Schwarz sweeps, E3), or a factorization that cannot be stored (a changing geometry or coefficient in a design loop). Stage 0 must measure this, not assume it.
2. **Loose certificates** ([[chart-operator-architecture]] §4.4).
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
