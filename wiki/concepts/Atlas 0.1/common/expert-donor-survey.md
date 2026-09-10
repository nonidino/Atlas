# Atlas 0.1: Expert Donor Survey — Public Pretrained Weights by Governing Family

**Type:** Core Concept — Model Portion (build input, survey)
**Related Concepts:** [[expert-library-atlas-0.1]], [[training-and-bootstrap-atlas-0.1]], [[incremental-transfer-roadmap]], [[plug-in-composition-theorems]], [[interface-transfer-theory]], [[composition-error-theory]], [[probed-dtn-coupling]], [[physics-simulation-datasets]], [[code-and-papers]]
**Survey date:** 2026-08-31. Every "confirmed" line below was checked against the primary repository or model card on that date; everything else is marked unverified.

---

## Why this page exists

[[expert-library-atlas-0.1]] settles *how* to cut experts — along governing-equation boundaries, not field-quantity labels — and [[training-and-bootstrap-atlas-0.1]] settles *how* to start each one, deferring to [[incremental-transfer-roadmap]]'s donor table. That table was compiled from what this vault had already ingested, and it found exactly two donors with public weights: Poseidon and Walrus. Both are fluid models. Every Atlas case study built since has used a single expert family, 2-D incompressible Navier–Stokes, which means **the donor table has never actually been tested against the question multiphysics asks: what can you download for the *other* governing families?**

That is the question this page answers. It surveys the public literature and model hubs for pretrained neural operators and physics foundation models with **downloadable weights** across seven families: compressible/transonic flow, heat conduction and thermal diffusion, solid mechanics and linear elasticity, thermoelastic coupling, electromagnetics, reacting flow, and multiphase/free-surface flow.

The short version, before the detail: **the donor market is enormously better stocked for compressible fluids than for anything else, and it thins to nothing at the couplings.** Two of the seven families have no downloadable donor at all. And the single property that decides whether a donor can be *certified* rather than merely *used* — bounded receptive field — is satisfied by almost none of them, for the reason [[case-study-ladder-to-f1]] already names.

## What a donor record must carry

A checkpoint is not a library entry. [[plug-in-composition-theorems]] is explicit that an entry carries a **conformance certificate bound to the weight hash and the probe state**, and [[interface-transfer-theory]] adds a per-seam interface declaration. Nothing surveyed here ships either. So the fields recorded below are the *inputs* to a certificate, not a substitute for one — they are what you need to know before spending compute on a probe run:

| Field | Why the composition layer needs it |
|---|---|
| Weights URL + license | Whether the entry can exist at all, and whether the project may use it downstream |
| Architecture family | Predicts the receptive-field answer before you measure it |
| Spatial dimension + native resolution | Fixes the prolongation $P_i$ of [[interface-transfer-theory]]; a resolution mismatch at a seam is a mapping-class question, not a resampling detail |
| **Boundary conditions as input?** | Where the donor sits on [[composition-error-theory]]'s Dirichlet-to-Neumann ladder. A donor that cannot be told its boundary data cannot be driven by a neighbour at all |
| **Gradients (JVP/VJP)?** | [[probed-dtn-coupling]] probes the operator by poking it; adjointness in [[port-algebra-atlas-0.1]] needs the transpose |
| Native timestep | Whether $\Delta t$ is an argument or a baked constant — the difference between a schedulable agent and a fixed-cadence one |
| Training distribution | The honest support of the entry; regime coverage claims outside it are extrapolation |
| Frozen vs. transfer donor | Whether the weights are the product or the initialization |

## The flag that decides everything: global receptivity

W93 (recorded in [[gap-worklist]] and [[case-study-wake-array-atlas-0.1]]) measured the domain of dependence of a Poseidon-T checkpoint directly, by poking a delta at a seam and reading `probe.support_reach`. The capability record declared **2 cells**. The measurement returned **64** — nonzero response in all 128 seam cells, at every amplitude from $1$ to $10^{-3}$. The halo rule had been passing a 16-cell overlap on the strength of the declared 2.

The conclusion is not that Poseidon is badly documented. It is structural:

> **A neural operator's receptive field is global by construction.** There is no larger integer that fixes this — a spectral mixing layer, a full-attention block, or a U-Net bottleneck each couple every point to every point in one application.

So each record below carries an explicit **locality** line. Three verdicts are possible:

- **Globally receptive (architectural)** — the network couples the whole domain. Cannot be certified as a local decomposition today. This is nearly every entry.
- **Globally coupled (physical)** — the *governing equation itself* is elliptic, so even a correct local expert has an unbounded domain of dependence at fixed time. An overlap halo is the wrong instrument; the seam wants a Dirichlet-to-Neumann/impedance interface ([[probed-dtn-coupling]]).
- **Bounded** — a finite, declarable domain of dependence. Two entries in this whole survey qualify, and neither is a continuum fluid model.

---

## §1 — Compressible and transonic flow

The best-stocked family by a wide margin. Every general-purpose physics foundation model with public weights covers compressible Euler or compressible Navier–Stokes somewhere in its pretraining mixture, because PDEBench and The Well both ship it.

### Poseidon-T / B / L

- **arXiv:** [2405.19101](https://arxiv.org/abs/2405.19101) · **Code:** [camlab-ethz/poseidon](https://github.com/camlab-ethz/poseidon) · **Weights:** [camlab-ethz/Poseidon-{T,B,L}](https://huggingface.co/camlab-ethz) — 21M / 158M / 629M params
- **License:** weights are **CC-BY-NC-4.0** (stated on the model card). This resolves open question 1 of [[incremental-transfer-roadmap]] in the restrictive direction: **research use is fine, any commercial or production use of Poseidon-derived weights is not.**
- **Architecture:** scOT — a hierarchical (Swin) vision transformer in a U-Net, with time-conditioned layer norm $\alpha(t) = \alpha t + \bar\alpha$, $\beta(t) = \beta t + \bar\beta$.
- **Dimension / resolution:** 2-D, $128 \times 128$, domain $[0,1]^2 \times [0,1]$ in space-time.
- **Boundary conditions as input:** **No.** Pretraining is entirely periodic. Non-Cartesian geometry is reached downstream by *masking*, not by a boundary-data channel.
- **Gradients:** yes — PyTorch, VJP via autograd, JVP via `torch.func.jvp`.
- **Native timestep:** none baked in. Lead time is a continuous conditioning input, so $\Delta t$ is an argument. This is the single most Atlas-friendly property any donor in this survey has.
- **Training distribution:** 77,840 trajectories over 6 operators — 4 compressible Euler (CE-RP, CE-KH, CE-CRP, CE-Gauss) and 2 incompressible NS (NS-Sines, NS-Gauss). Periodic BCs, smooth and Riemann initial data. 4 channels: $\rho, u, v, p$.
- **Locality:** **globally receptive (architectural) — measured, not inferred.** W93's 64-cell reading is on this exact checkpoint family.
- **Judgment:** transfer donor for the external-compressible-flow expert; the non-commercial license makes it unusable as a frozen production component.

### Walrus

- **arXiv:** [2511.15684](https://arxiv.org/abs/2511.15684) · **Code:** [PolymathicAI/walrus](https://github.com/PolymathicAI/walrus) · **Weights:** [polymathic-ai/walrus](https://huggingface.co/polymathic-ai/walrus) — 1.3B params, plus 8 released fine-tunes
- **License:** **MIT.** The only broad-coverage donor in this survey under a permissive license.
- **Architecture:** encoder–processor–decoder; hMLP patch embedding with *stride modulation* holding internal resolution near 32/33 per dim in 2-D and 16/17 in 3-D; processor blocks use factorized space and time attention.
- **Dimension / resolution:** 2-D **and** 3-D, resolution-adaptive by construction.
- **Boundary conditions as input:** no explicit channel. Diverse BCs appear in the training mixture only.
- **Gradients:** yes (PyTorch).
- **Native timestep:** autoregressive residual, $u(t{+}1) \approx u(t) + M(U(t))$ over a short history $U(t) = [u(t{-}\tau{+}1), \ldots, u(t)]$, trained with random stride 1–5, which buys some $\Delta t$ flexibility without making it an explicit argument.
- **Training distribution:** 19 scenarios / 63 physical variables, drawn mostly from The Well — astrophysics, geoscience, rheology, plasma, acoustics, classical fluids.
- **Locality:** **globally receptive (architectural).** Factorizing attention into space and time axes does not bound it — each axis still spans the full domain.
- **Judgment:** best frozen generalist fallback *and* a transfer donor; the license is what makes it the default first choice over Poseidon for anything that might outlive research.

### DPOT-{Ti, S, M, L, H}

- **arXiv:** [2403.03542](https://arxiv.org/abs/2403.03542) · **Code:** [HaoZhongkai/DPOT](https://github.com/HaoZhongkai/DPOT) · **Weights:** [hzk17/DPOT](https://huggingface.co/hzk17/DPOT) — 7M / 30M / 122M / 509M / 1.03B
- **License:** **Apache-2.0.**
- **Architecture:** Fourier-attention transformer with an auto-regressive *denoising* pretraining objective.
- **Dimension / resolution:** 2-D, $128 \times 128$.
- **Boundary conditions as input:** no.
- **Gradients:** yes (PyTorch).
- **Native timestep:** autoregressive, 10 input frames → 1 output frame, fixed stride.
- **Training distribution:** FNO datasets, PDEBench (including *viscous* compressible NS across Mach and viscosity ranges — the coverage [[physics-simulation-datasets]] notes PDEgym lacks), PDEArena, CFDBench.
- **Locality:** **globally receptive (architectural)** — Fourier mixing takes an FFT over the whole domain; this is the most unambiguously global entry in the survey.
- **Judgment:** transfer donor. **[AI Inference]:** the 7M Tiny checkpoint under Apache-2.0 makes DPOT the cheapest donor on which to run the substitution-campaign probes of [[case-study-ladder-to-f1]] — the same experiment costs an order of magnitude less here than on Poseidon-B, and the license does not constrain what you do with the result.

### GPhyT (General Physics Transformer) — S / M / L / XL

- **arXiv:** [2509.13805](https://arxiv.org/abs/2509.13805) · **Code:** [FloWsnr/General-Physics-Transformer](https://github.com/FloWsnr/General-Physics-Transformer) · **Weights:** [flwi/Physics-Foundation-Model](https://huggingface.co/flwi/Physics-Foundation-Model)
- **License:** **MIT** (model card).
- **Architecture:** transformer **neural differentiator** predicting $\partial X/\partial t$, followed by an explicit numerical integrator, $X_{t+1} = f(X_t, \partial X/\partial t|_t, \Delta t)$ — forward Euler by default, RK4 available. Tubelet patch tokenization; first-order spatial and temporal derivatives are computed by central differences and concatenated to the input fields. Attention is deliberately **unified**, not factorized, "for maximum expressivity" on shock interactions.
- **Dimension / resolution:** 2-D, $256 \times 128$.
- **Boundary conditions as input:** not an explicit channel — obstacles and walls enter through the state snapshots, and new BCs are handled *in-context* from the prompt frames. This is the closest thing in the survey to zero-shot BC generalization, and it is also unauditable: nothing declares what BC the model thinks it is enforcing.
- **Gradients:** yes (PyTorch).
- **Native timestep:** $N_{\text{in}} = 4$ history frames, and $\Delta t$ is an explicit argument to the integrator. Trained with variable time increments so the differentiator is sampling-frequency invariant.
- **Training distribution:** 1.8 TB over 8 systems — incompressible shear flow, multiquadrant Euler (shocks), Rayleigh–Bénard, incompressible flow with obstacles, thermal channel flow, Rayleigh–Bénard with obstacles, two-phase porous flow, supersonic flow.
- **Locality:** **globally receptive (architectural)**, by explicit design choice.
- **Judgment:** transfer donor, and structurally the closest match to Atlas's needs of any entry here. The derivative-head-plus-integrator split is the one [[training-and-bootstrap-atlas-0.1]] already flagged as worth borrowing — this survey upgrades that from "structural pattern to reimplement" to **"weights to load"**, which is a direct correction to [[incremental-transfer-roadmap]]'s donor table.

### MPP-AViT-{Ti, S, B, L}

- **arXiv:** [2310.02994](https://arxiv.org/abs/2310.02994) · **Code + weights:** [PolymathicAI/multiple_physics_pretraining](https://github.com/PolymathicAI/multiple_physics_pretraining) (checkpoints via a linked Google Drive folder) · **License:** MIT
- **Architecture:** axial ViT with a shared normalization/projection scheme across heterogeneous systems (`n_states = 12` in the released weights). 2-D, inflatable to 3-D.
- **Boundary conditions as input:** no. **Gradients:** yes. **Native timestep:** autoregressive, fixed history.
- **Training distribution:** PDEBench + PDEArena — compressible NS, incompressible NS, shallow water, diffusion-reaction.
- **Locality:** globally receptive (architectural).
- **Judgment:** superseded by Walrus on coverage from the same lab; keep as a small, well-documented ablation baseline rather than a production donor.

### PhysiX

- **arXiv:** [2506.17774](https://arxiv.org/abs/2506.17774) · **Code:** [arshka/PhysiX](https://github.com/arshka/PhysiX), Apache-2.0. 4.5B params: VQ-VAE tokenizer + Cosmos-1.0 autoregressive transformer + refinement module; 2-D, $256\times256$, 33-frame sequences, 9-frame inference context; trained on the 2-D subset of The Well.
- **Weights:** the paper states checkpoints will be released; a public checkpoint URL was **not confirmed** in this pass.
- **Judgment:** **not a donor. [AI Inference]:** the discrete tokenizer is disqualifying independent of weight availability — quantizing the state destroys the differentiable state-to-state map that [[probed-dtn-coupling]] probes and [[port-algebra-atlas-0.1]] needs an adjoint of. A 4.5B model you cannot take a clean directional derivative through is not a compositional expert.

### Geometry-conditioned steady aero surrogates (DoMINO, AB-UPT, Transolver)

A distinct sub-class worth separating: these predict a **steady** field over an arbitrary geometry rather than advancing a state in time. They are the industrial state of the art for external aerodynamics and they are the wrong shape for Atlas's time-stepped graph, but they are the only entries trained on realistic 3-D engineering geometry.

| Model | Weights + license | Notes |
|---|---|---|
| **DoMINO** (NVIDIA PhysicsNeMo) | Framework [NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo) Apache-2.0; **checkpoint gated behind an authenticated NGC account** (DoMINO-Automotive-Aero NIM) | Steady, 3-D, mesh-free point evaluation. Inputs: STL geometry + global params (inlet velocity, density). Outputs: surface pressure/wall shear, volume velocity/pressure. Trained on DrivAerML (500 morphed notchback cars, low-Mach automotive — **not transonic**). Distributed as an inference container, so **no gradient access**. Gated weights cannot be a library entry: [[plug-in-composition-theorems]] binds a certificate to a weight hash you must be able to obtain. |
| **AB-UPT** (Emmi AI) | [2502.09692](https://arxiv.org/abs/2502.09692), [2510.15808](https://arxiv.org/abs/2510.15808); [Emmi-AI/noether](https://github.com/Emmi-AI/noether) under the **Emmi Open Research / Non-Production License** (MNPL-derived: research free, commercial requires a separate license). Downloadable pretrained weights **not confirmed** in this pass — the repo ships training recipes and a DrivAerML tutorial | Multi-branch transformer separating geometry / surface / volume with cross-attention, anchored neural-field decoders, divergence-free formulation; scales to 9M surface and 140M volume cells. DrivAerML, AhmedML, ShapeNet-Car, SHIFT-SUV, SHIFT-Wing. Globally receptive (cross-branch attention). |
| **Transolver / ++ / -3** | [thuml/Transolver](https://github.com/thuml/Transolver), MIT. Checkpoints confirmed for DrivAerML surface/volume tasks in the Transolver++ repo; **checkpoints for the Elasticity/Plasticity/Darcy standard benchmarks not confirmed** | Physics-attention over learned "slices"; strong on the Elasticity benchmark, which is why it also appears in §3. Steady. |

### What is *not* available for this family

No public checkpoint was found for **transonic external aerodynamics specifically** — the shock-on-wing regime. The datasets exist (GAOT's release spans NACA0012/2412 and RAE2822 across subsonic, transonic and supersonic; SuperWing is a transonic wing dataset), and [camlab-ethz/GAOT](https://github.com/camlab-ethz/GAOT) publishes the architecture, but a downloadable transonic checkpoint was not confirmed. The compressible coverage that *does* ship as weights is periodic-box Euler and PDEBench compressible NS — shocks in a box, not shocks on a body.

---

## §2 — Heat conduction and thermal diffusion

### Therm-FM — T / B / L

- **arXiv:** [2605.22663](https://arxiv.org/abs/2605.22663) · **Code + weights + datasets:** [haiyangxin/Therm-FM](https://github.com/haiyangxin/Therm-FM) (checkpoints on Google Drive at 21M / 158M / 629M — the Poseidon size ladder, because it *is* Poseidon)
- **License:** the repo carries an Apache-2.0 badge while the LICENSE text reads MIT — **inconsistent, resolve before use.** More seriously: **[AI Inference]:** these weights are fine-tuned from Poseidon, whose weights are CC-BY-NC-4.0, so the non-commercial term plausibly propagates to the derivative regardless of what the Therm-FM repo declares. Treat as non-commercial until the authors clarify.
- **Architecture:** the scOT backbone and codebase of Poseidon, fine-tuned with a thermal-equivalent multi-fidelity strategy.
- **Dimension / resolution:** steady 2-D maps at $55^2$, $88^2$, $85^2$, $101^2$, $151^2$; transient at $88\times88\times9$, $64\times64\times5$, $151\times151\times9$ (the third axis being time frames). 3-D chip stacks are handled through the layer structure, not a volumetric grid.
- **Boundary conditions as input:** **partially — the strongest in the survey so far.** A material/structure map $M$ and a power-density map enter as input channels. But the thermal BCs proper (adiabatic lateral walls, convective top) are fixed problem specifications, not variable inputs.
- **Gradients:** yes (PyTorch, inherits scOT).
- **Native timestep:** steady mode has none; transient mode learns a fixed frame stride.
- **Training distribution:** 3-D IC thermal simulation — heterogeneous materials, dense TSV/microbump interconnects, package boundary conditions, HotSpot and industrial benchmarks.
- **Locality:** globally receptive (architectural), inherited from scOT.
- **Judgment:** the **only** confirmed pretrained-weight donor for conduction found in this survey. Its real value to Atlas is less the weights than the demonstration: it is direct evidence that the **Poseidon → parabolic-family transfer path works**, which is precisely the bet [[training-and-bootstrap-atlas-0.1]] placed on the thermal-structural expert without evidence. Its training distribution (chip packaging at millimetre scale) has essentially no overlap with a rocket chamber wall or a nacelle, so it is a *method* citation for the bootstrap, not a checkpoint to load.

### PDEformer-2 — the one donor that takes boundary conditions as an argument

- **arXiv:** [2507.15409](https://arxiv.org/abs/2507.15409) · **Code:** [functoreality/pdeformer-2](https://github.com/functoreality/pdeformer-2) · **Weights:** Gitee AI — base (82.65M), fast (71.07M), small (27.75M) · **License:** Apache-2.0
- **Architecture:** the symbolic form of the PDE is encoded as a **computation graph** and consumed by a graph transformer; numeric fields are embedded by separate encoders; **boundary conditions are represented by signed distance functions and carried in that graph.** Framework is **MindSpore**, not PyTorch.
- **Dimension / resolution:** 2-D, arbitrary domains; solutions queried at arbitrary space-time points (inference examples use spatial resolution 32).
- **Boundary conditions as input:** **yes — explicitly, and varying.** This is the only entry in the survey where a neighbour could *tell* the expert its boundary data through the model's declared interface rather than by overwriting a halo.
- **Gradients:** MindSpore autodiff; **no `torch.func` equivalent**, so every probe, adjoint, and JVP in the `atlas/` composition layer would need a MindSpore path or a bridge. This is a real integration cost, not a footnote.
- **Native timestep:** none — solutions are queried at arbitrary $t$, closer to Poseidon's lead-time conditioning than to autoregression.
- **Training distribution:** ~40 TB of generated 2-D PDEs — conservation laws, shallow water, tracer equations, varying coefficients, domains and BCs. Generic scalar/system form rather than one named physics.
- **Locality:** globally receptive (architectural).
- **Judgment:** **[AI Inference]:** listed under this family because diffusion is inside its generated coverage, but its importance to Atlas is orthogonal to any one family — it is the only public demonstration that a foundation model can accept a **declared boundary condition as a typed input**. [[composition-error-theory]] sharpens BC flexibility into a selection axis and everything else in this survey scores at the bottom of it. PDEformer-2 scores at the top and pays for it in framework lock-in.

### Not available

No dedicated conduction or conjugate-heat-transfer foundation model with public weights beyond Therm-FM. GPhyT's thermal channel flow and Rayleigh–Bénard cover *convection*, not pure conduction. TFRD and DeepEDH (already catalogued in [[physics-simulation-datasets]]) are datasets, not checkpoints. Emmi's Noether ships a SIMSHIFT heatsink recipe, but that is conjugate heat transfer in a fluid, and the weights were not confirmed.

---

## §3 — Solid mechanics and linear elasticity

**There is no pretrained linear-elasticity foundation model with downloadable weights.** This is the survey's second-most consequential finding, and it is a genuine market gap rather than a search failure: the solid-mechanics ML literature is large, but it publishes architectures, benchmarks and one-off case studies, not checkpoints. Three partial exceptions are worth recording.

### NeuberNet

- **Paper:** Grossi, Beghini & Benedetti, *Communications Engineering*, 2025 ([s44172-025-00549-5](https://www.nature.com/articles/s44172-025-00549-5)) · **Code + weights:** [grossIt/neubernet](https://github.com/grossIt/neubernet) — the paper states the final inference weights are included in the repo; dataset on Zenodo
- **Architecture:** a Multi-Task Nonlinear Manifold Decoder — a neural operator mapping **far-field displacement boundary conditions taken from a cheap linear-elastic simulation** to high-resolution elastic-plastic stress and strain fields at a V-notch.
- **Dimension:** axisymmetric solid mechanics; small-scale plasticity with bilinear isotropic hardening.
- **Boundary conditions as input:** **yes — the boundary data is the entire input.** The model *is* a boundary-to-interior operator.
- **Gradients:** PyTorch (unverified in detail). **Native timestep:** none — quasi-static.
- **Locality:** **bounded (physically and by construction).** The operator acts on a notch patch driven by its own far-field boundary data. Its domain of dependence is the patch boundary, declared and finite.
- **Judgment:** **[AI Inference]:** narrow to the point of being a single component rather than a family, but it is the **shape** Atlas wants and almost nothing else in this survey has: a local sub-domain expert whose interface is its boundary condition, i.e. a Dirichlet-to-Neumann map with weights. If the substitution campaign of [[case-study-ladder-to-f1]] needs one learned agent that could plausibly pass the halo rule as shipped, this is the first candidate to try — not because notches matter to Atlas, but because it would be the first learned expert to survive certification at all.

> **Measured 2026-09-10 (Tier 41, CS-S2 — [[case-study-bounded-donor-atlas-0.1]]).** This entry was written from the paper and the repository's description. The weights have now been run, beside linear elasticity solved classically on the same disc, and three of its lines change.
> - **License: verified absent.** GitHub reports `license: null` for `grossIt/neubernet`, so nothing beyond reading the code is granted; only the Zenodo dataset is CC BY 4.0, and the article is CC BY-NC-ND 4.0. CS-S2 used the weights locally with the user's approval, outside the repository, and commits only derived numbers.
> - **"Boundary conditions as input: yes" is true, and wearing it took three corrections.** The flux has to be the Galerkin dual of the hat trace — sampled pointwise at the sensors, the classical control's operator moved $36$–$42\%$ between two meshes; the trace has to be continued across the notch opening the way the authors' pipeline fills it; and the hoop input's sign is the opposite of $u_\theta/\rho$ on the classical axes, which only a base carrying torsion could show.
> - **"Locality: bounded" is true of the patch and false of the response.** On the donor's own ring of $29$ material sensors, band-truncating its $u_x$ port leaves $0.81$ of the operator's norm outside $r=2$ and $0.69$ outside $r=4$, against $0.23$ and $0.14$ for linear elasticity on the same disc; its $u_x$ port gets under $10\%$ at no radius short of the whole ring and its $u_y$ port at $r=24$, where the classical ports get there at $r=9$ and $r=5$. Into the disc its response to one sensor does not decay at all: from $0.5$ to $2\,R_n$ deep it keeps $95$–$139\%$ of itself, where the classical patch keeps $12$–$17\%$.
> - **And the response it has is its training manifold's.** Along the tension far field it matches linear elasticity to $8\%$ at cosine $0.997$, and along the torsion far field to $12\%$ where both loads are signed; one sensor's worth of boundary data gets $6\%$ of the classical flux, orthogonal to it. On its in-plane ports its operator differs from the classical one by $100$–$110\%$ of the classical norm — the null-replacement signature [[substitution-campaign-checkpoint]] measured on Poseidon-T, arriving on a second architecture family.
> - **[AI Inference]:** *bounded* in the judgment above meant a bounded **domain**, which every sub-model has; what a halo rule or a certificate consumes is a bounded **response**. The first is not evidence for the second, and on this donor they point opposite ways.

### Transolver (Elasticity benchmark)

MIT-licensed, state of the art on the standard Elasticity and Plasticity benchmarks, and the architecture is the strongest available for irregular meshes. But benchmark checkpoints were **not confirmed** as released — only the DrivAerML aero checkpoints were. Judgment: **structural pattern to reimplement**, in [[incremental-transfer-roadmap]]'s vocabulary, not a weight-transfer path.

### MACE-MP-0 family — the locality existence proof

- **arXiv:** [2401.00096](https://arxiv.org/abs/2401.00096) · **Code + weights:** [ACEsuit/mace-mp](https://github.com/ACEsuit/mace-mp), [huggingface.co/mace-foundations](https://huggingface.co/mace-foundations)
- **License:** **MIT** for the MP variants (MACE-MP-0a/0b/0b2/0b3, MPA-0); Academic Software License for OMAT-0, OMOL-0, MATPES and MH variants.
- **Architecture:** equivariant message-passing graph tensor network over atoms (atomic cluster expansion + equivariant GNN). ~3.85M parameters for the 89-element foundation model; small model radial cutoff $r_{\text{cut}} = 6.0$ Å; variants at message-passing equivariance $L = 0, 1, 2$.
- **Locality:** **bounded — and this is the point.** The domain of dependence is exactly $r_{\text{cut}} \times (\text{number of message-passing layers})$: a finite number, derivable from the config, verifiable by the same delta-poke `probe.support_reach` that caught Poseidon. Forces are analytic gradients of a learned energy, so the model is differentiable by construction rather than by framework accident.
- **Judgment:** **not an elasticity expert** — it is an atomistic interatomic potential and cannot produce a continuum stress field over an engineering part. It earns its place here because it is the counterexample to the sentence that governs this entire survey. [[case-study-ladder-to-f1]] states that the framework cannot certify a decomposition whose agents are globally-receptive pretrained operators, "and that is nearly all of them." MACE is a large, widely-used, publicly-weighted physics foundation model whose receptive field is **finite, declared, and checkable**. **[AI Inference]:** that makes finite-cutoff message-passing architectures — not transformers or spectral operators — the architecture class in which Atlas's locality certification is actually reachable, and it suggests the constrained expert class that [[case-study-ladder-to-f1]]'s "bounded-receptive-field operators exist and are usable" rung is waiting on already exists in a neighbouring field.

---

## §4 — Thermoelastic / thermo-structural coupling

**Nothing. No pretrained model with downloadable weights was found for coupled thermo-mechanical or thermoelastic problems.** The literature is real but uniformly one-off: sequential DeepONets for thermomechanical steel solidification, coupled-multiphysics DeepONet studies ([2507.03660](https://arxiv.org/abs/2507.03660)), ANN thermo-mechanical constitutive models for shape-memory alloys, reduced-order models for one-way-coupled steady thermomechanics. None publish checkpoints; several publish neither code nor data.

This is worth stating carefully, because it is the family [[expert-library-atlas-0.1]] singles out. That page's rocket table lists a single **thermal-structural** expert governing "conduction + stress + thermal expansion," and argues from physics that the $b\!-\!c$ edge legitimately wants *two* experts exchanging a coupling term, since conduction and quasi-static elasticity are different governing equations weakly coupled through thermal expansion.

**The donor market independently confirms that cut.** There is a conduction donor (§2) and there is, at best, an elasticity architecture (§3). There is no coupled donor and no prospect of one. Any Atlas thermal-structural expert must therefore be assembled as **two experts exchanging the thermal-expansion term explicitly** — which is what the physics argument already concluded, now also forced by what exists to download.

---

## §5 — Electromagnetics

Pretrained EM weights exist, but only for **integrated photonics** — frequency-domain Maxwell inside optical devices at micron scale.

### NeurOLight

- **arXiv:** [2209.10098](https://arxiv.org/abs/2209.10098) · **Code:** [JeremieMelo/NeurOLight](https://github.com/JeremieMelo/NeurOLight) · **License:** MIT
- **Weights:** confirmed — a 12-layer model on tunable MMIs and a 16-layer model on etched MMIs, distributed via a SharePoint link in the repo's usage section (a fragile hosting choice for a library entry).
- **Architecture:** an FNO-family operator learning a *family* of frequency-domain Maxwell PDEs, with "masked source modeling" encoding the incident light. 2-D device cross-sections.
- **Boundary conditions as input:** the permittivity distribution and the source mask are inputs; the domain truncation (PML) is fixed.
- **Native timestep:** **none — this is a steady frequency-domain solve**, not a time-stepper.
- **Training distribution:** multimode interferometers (tunable and etched), parametric device geometries, a fixed wavelength band.
- **Locality:** **globally coupled (physical)** *and* globally receptive (architectural). Frequency-domain Maxwell is a Helmholtz problem — elliptic — so the true solution operator has unbounded domain of dependence at fixed frequency.
- **Judgment:** frozen, and only if a case study is literally photonics.

### PACE / PACE-Light

- **arXiv:** [2411.03527](https://arxiv.org/abs/2411.03527) · **Code:** [zhuhanqing/PACE-Light](https://github.com/zhuhanqing/PACE-Light) · NeurIPS 2024. Successor to NeurOLight; reported 73% error reduction on complex real-world photonic devices where NeurOLight reaches only ~0.38 normalized MAE. Checkpoint availability and license were **not confirmed** in this pass.

### What this family really tells Atlas

**[AI Inference]:** the EM finding that matters is not the checkpoint scarcity, it is the locality verdict. An elliptic governing family is globally coupled *as physics*, so no amount of architectural discipline produces a bounded halo for a fixed-frequency EM expert. The overlap-halo instrument is simply the wrong one here, and the seam must be a Dirichlet-to-Neumann / impedance interface of the kind [[probed-dtn-coupling]] already builds. The same argument applies to quasi-static elasticity (§3) and is the reason [[impl-atlas-0.1-phase2-experts]] flagged "the elliptic subtlety" for the shell. Elliptic families are where Atlas's DtN machinery is not an optimization but the only correct coupling.

Nothing was found for the *engineering* EM regimes a rocket or wind-farm case would need — low-frequency eddy currents, generator and machine electromagnetics, antenna radiation.

---

## §6 — Reacting and combusting flow

No spatial reacting-flow operator with public weights exists. But the family splits cleanly in two, and **one half is solved**.

### DeepFlame DNN chemistry integrators — the best-shaped donor in this survey

- **Code:** [deepmodeling/deepflame-dev](https://github.com/deepmodeling/deepflame-dev) · **Paper:** [2210.07094](https://arxiv.org/abs/2210.07094) · **Weights:** hosted on [AIS Square](https://www.aissquare.com/) (e.g. `HE04_Hydrogen_ESH2_GMS_sub_20221101`); the newer DF-ODENet models ship from v1.3, with DFODE-Kit for training your own. **License not verified** in this pass — a LICENSE file is present in the repo but its terms were not read; the model artifacts are distributed separately through AIS Square under that platform's terms.
- **Architecture:** an MLP replacing the stiff chemical ODE integration. It maps a **point** thermochemical state $(T, p, Y_i)$ to that point's state after one chemistry substep — a learned $\exp(\Delta t_{\text{chem}} \mathcal{L}_{\text{chem}})$.
- **Dimension / resolution:** **zero-dimensional.** There is no grid.
- **Boundary conditions as input:** not applicable — a pointwise operator has no boundary.
- **Gradients:** yes (Torch; DeepFlame couples OpenFOAM, Cantera and libtorch).
- **Native timestep:** the chemistry substep is baked into the trained map — a genuine fixed constant, and the one field where this donor is *less* flexible than the fluid models.
- **Training distribution:** thermochemical states sampled from canonical laminar flame and ignition configurations; hydrogen, hydrocarbons, ammonia; premixed flames, detonation, ignition.
- **Locality:** **bounded — trivially. The domain of dependence is a single point.**
- **Judgment:** **frozen, and the most Atlas-compatible donor found anywhere in this survey.** A pointwise operator passes any halo rule with room to spare, needs no interface transfer, and cannot be decertified by W93's argument because it has no receptive field to be global. **[AI Inference]:** this changes the reacting-flow plan in [[training-and-bootstrap-atlas-0.1]] concretely. That page says "train from scratch" for the reacting/internal compressible flow expert on the grounds that no donor covers confined reacting duct flow. That remains true of the *transport* half. But the expert factors: chemistry (pointwise, stiff, donor available and certifiable) and confined compressible transport (spatial, no donor). Only the second half needs a from-scratch build, and the split is exactly the operator-splitting decomposition reacting-flow solvers already use — so it costs nothing in physics fidelity to adopt.

### Not available

No neural operator for spatially-resolved reacting flow with public weights. BLASTNet ([[physics-simulation-datasets]]) supplies 2.2 TB of reacting and non-reacting compressible DNS but ships as a dataset and benchmark, not as checkpoints. Extended-FNO work on stiff chemical kinetics and adaptive physics-informed operators for non-equilibrium flows publish methods without weights. GPhyT and Walrus both include Gray–Scott-style reaction-diffusion in their lineage, which is chemistry as a toy, not combustion.

---

## §7 — Multiphase and free-surface flow

The second-best-stocked family, entirely because of boiling.

### Bubbleformer

- **arXiv:** [2507.21244](https://arxiv.org/abs/2507.21244) · **Code + weights:** [HPCForge/Bubbleformer](https://github.com/HPCForge/Bubbleformer) — "weights for all the benchmark models" in the repo's model zoo · **Dataset:** BubbleML 2.0 on Hugging Face · **License:** MIT
- **Architecture:** FiLM-AViT — an axial vision transformer with factored space-time blocks and a **FiLM layer conditioning on fluid thermophysical parameters** for cross-liquid generalization.
- **Dimension / resolution:** 2-D; spatial and temporal resolution vary by fluid according to characteristic scales, with AMR where needed.
- **Boundary conditions as input:** constant-heat-flux wall conditions and double-sided heater configurations are part of the simulation setup; the model conditions on thermophysical parameters rather than on a boundary-data channel.
- **Gradients:** yes (PyTorch). **Native timestep:** autoregressive forecasting; stride not confirmed.
- **Training distribution:** BubbleML 2.0 — 160+ high-resolution 2-D simulations across pool and flow boiling, three working fluids (FC-72 dielectric, R-515B refrigerant, LN₂ cryogen), saturated / subcooled / single-bubble nucleation, and the full flow-regime ladder from bubbly through slug to annular and dryout.
- **Locality:** globally receptive (architectural).
- **Judgment:** transfer donor for any phase-change expert. **[AI Inference]:** its FiLM conditioning head is the closest existing analogue to Atlas's dimensionless-conditioning contract ([[pfm-interface-design]]) — a donor that already accepts a physical-parameter vector separate from the field state is markedly cheaper to wrap in a Stage-1 adapter than one where the regime has to be inferred from the field.

### Walrus fine-tune: `walrus_ft_bubbleML_poolBoil`

MIT-licensed, on the Hugging Face hub, already fine-tuned from the Walrus trunk onto BubbleML pool boiling. Judgment: the zero-effort frozen multiphase entry — worth benchmarking against Bubbleformer before committing to either, since one is a 1.3B generalist adapted to boiling and the other is a specialist trained on it.

### GNS / *Learning to Simulate* — a correction to the existing catalogue

[deepmind-research/learning_to_simulate](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate) is the canonical particle free-surface/granular simulator, and [[code-and-papers]] already lists it. **Verified in this pass: the repository distributes datasets only** — WaterDrop, Water, Sand, Goop, MultiMaterial, WaterRamps, SandRamps, FluidShake and 3-D variants, via `storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/...` — **and no pretrained checkpoints.** This confirms rather than contradicts [[incremental-transfer-roadmap]]'s "structural pattern only" classification, and the code-and-papers row is annotated accordingly.

Note the irony for Atlas: GNS is a message-passing simulator with a **finite connectivity radius**, i.e. exactly the bounded-receptive-field architecture class §3 identifies as the one where certification is reachable — and it is the one whose weights nobody published.

### Not available

No pretrained VOF/level-set free-surface operator (sloshing, wave impact, ship hydrodynamics, tank draining) was found. Coverage that ships as weights is boiling (Bubbleformer, Walrus-ft), Rayleigh–Taylor (Walrus, via The Well), and two-phase porous-media flow (GPhyT).

---

## Summary table

Locality column: **G-arch** = globally receptive by architecture · **G-phys** = globally coupled because the governing equation is elliptic · **Bounded** = finite, declarable domain of dependence.

| Family | Model | Weights | License | Dim / native res | BC as input | $\Delta t$ | Locality | Verdict |
|---|---|---|---|---|---|---|---|---|
| Compressible | **Poseidon** T/B/L | HF `camlab-ethz` | **CC-BY-NC-4.0** | 2-D, $128^2$ | No (periodic) | Continuous lead time | **G-arch (measured)** | Transfer donor; NC blocks production |
| Compressible | **Walrus** 1.3B | HF `polymathic-ai/walrus` | MIT | 2-D + 3-D, adaptive | No | AR, stride 1–5 | G-arch | Frozen fallback + transfer donor |
| Compressible | **DPOT** 7M–1B | HF `hzk17/DPOT` | Apache-2.0 | 2-D, $128^2$ | No | AR, 10→1 | G-arch (Fourier) | Cheapest probe target |
| Compressible | **GPhyT** S–XL | HF `flwi/…` | MIT | 2-D, $256{\times}128$ | In-context only | **Explicit $\Delta t$** | G-arch | Closest structural match; upgrade from "pattern" to "weights" |
| Compressible | **MPP-AViT** Ti–L | Google Drive | MIT | 2-D (→3-D) | No | AR fixed | G-arch | Ablation baseline |
| Compressible | PhysiX 4.5B | not confirmed | Apache-2.0 (code) | 2-D, $256^2$ | No | AR, 9-frame | G-arch | **Not a donor** — discrete tokens |
| Compressible | DoMINO | NGC, gated | NVIDIA terms | 3-D point cloud | Global params | Steady | G-arch | Gated ⇒ cannot certify |
| Compressible | AB-UPT / Noether | not confirmed | Research-only | 3-D, 140M cells | Geometry | Steady | G-arch | License-blocked |
| Compressible | Transolver | Aero only | MIT | Mesh | Geometry | Steady | G-arch | Reimplement |
| Heat | **Therm-FM** T/B/L | GitHub + Drive | Apache/MIT conflict; **NC likely inherited** | 2-D + frames, $55^2$–$151^2$ | Material + power maps | Steady / fixed stride | G-arch | Only conduction donor; value is the proven transfer path |
| Heat (generic) | **PDEformer-2** | Gitee AI | Apache-2.0 | 2-D, arbitrary domains | **Yes — SDF in the PDE graph** | Arbitrary $t$ | G-arch | **Only donor with declared BCs**; MindSpore lock-in |
| Elasticity | **NeuberNet** | in-repo | not verified **[2026-09-10: verified — none]** | Axisymmetric notch | **Yes — BC is the input** | Quasi-static | **Bounded** **[measured 2026-09-10: the patch is, the response is not]** | First plausible halo-rule survivor **[CS-S2: not compact, and no same-class reference pair]** |
| Elasticity | Transolver (bench) | not confirmed | MIT | Mesh | Geometry | Steady | G-arch | Reimplement |
| Elasticity (atomistic) | **MACE-MP-0** family | GitHub + HF | MIT / ASL | Atomistic, $r_{\text{cut}}=6$ Å | n/a | MD step | **Bounded** | Not an expert — the **locality existence proof** |
| **Thermoelastic** | — | **none** | — | — | — | — | — | **No donor exists** |
| EM | **NeurOLight** | SharePoint | MIT | 2-D photonic | Permittivity + source | Steady (freq. domain) | G-arch + **G-phys** | Frozen, photonics only |
| EM | PACE | not confirmed | not verified | 2-D photonic | Permittivity + source | Steady | G-arch + G-phys | Successor to above |
| **Reacting** | **DeepFlame DNN** | AIS Square | not verified | **0-D pointwise** | n/a | Fixed chem substep | **Bounded (a point)** | **Best-shaped donor in the survey** |
| Reacting (spatial) | — | **none** | — | — | — | — | — | **No donor exists** |
| Multiphase | **Bubbleformer** | in-repo zoo | MIT | 2-D, per-fluid | Heat flux in setup; **FiLM params** | AR | G-arch | Transfer donor; FiLM ≈ conditioning contract |
| Multiphase | **`walrus_ft_bubbleML_poolBoil`** | HF | MIT | 2-D | No | AR | G-arch | Zero-effort frozen entry |
| Free surface | GNS | **datasets only** | — | Particles | n/a | Fixed | (bounded arch., no weights) | Reimplement |

---

## Five cross-cutting findings

**1. Licensing is a first-class selection criterion, and Poseidon fails it.** Poseidon's weights are CC-BY-NC-4.0. [[incremental-transfer-roadmap]] listed "verify current license terms" as its first unstarted action item and built two of its four bootstrap stages on Poseidon; the answer is that Poseidon is a research-only donor. Walrus (MIT), DPOT (Apache-2.0) and GPhyT (MIT) are the permissively-licensed alternatives, all with comparable or better compressible coverage. **[AI Inference]:** the non-commercial term also propagates to anything fine-tuned from those weights, which puts Therm-FM — the only conduction donor — under the same cloud, and means Atlas's most likely thermal path is legally encumbered at both ends.

**2. Almost nothing accepts a boundary condition as an input.** [[composition-error-theory]] makes BC flexibility a selection axis running from Dirichlet to Neumann. Measured against real donors, the axis mostly collapses: Poseidon, Walrus, DPOT, MPP, Bubbleformer and PhysiX have no boundary-data channel at all — they were trained on periodic or fixed-BC data and expect the domain to be self-contained. GPhyT handles new BCs *in context*, which is impressive and unauditable. Only **PDEformer-2** (SDF-encoded BCs in a symbolic PDE graph) and **NeuberNet** (boundary displacement is literally the input) take boundary data as a declared argument. A frozen donor with no BC input cannot be driven by a neighbour except by overwriting a halo — which is exactly the mechanism W93 decertified.

**3. Gradients are nearly free, with three exceptions.** Every PyTorch donor gives VJP through autograd and JVP through `torch.func.jvp`, so [[probed-dtn-coupling]]'s probing and [[port-algebra-atlas-0.1]]'s adjointness are cheap for most of this list. The exceptions matter: **DoMINO** ships as an inference container with no gradient access; **PDEformer-2** is MindSpore-only and needs a bridge; **PhysiX** quantizes the state, breaking the differentiable state-to-state map regardless of framework.

**4. The locality verdict is close to unanimous — and its two counterexamples share an architecture.** Twenty-odd entries; three bounded. **DeepFlame's chemistry MLP** (pointwise, zero spatial extent), **NeuberNet** (a boundary-driven patch), and **MACE** (finite cutoff × finite layer count). GNS would be a fourth if anyone had published its weights. **[AI Inference]:** the pattern is that bounded receptive field comes from *local operators and finite-radius message passing*, never from transformers or spectral mixing — and that the continuum-fluid foundation-model community has converged unanimously on the architectures Atlas cannot certify. This is the sharpest available statement of what [[case-study-ladder-to-f1]] calls the substitution campaign: the campaign is not a search for a better-documented transformer, it is a bet on a different architecture class.

**5. Elliptic families need DtN, not halos, as a matter of physics.** For frequency-domain EM (§5) and quasi-static elasticity (§3), the exact solution operator is globally coupled independent of any network. Even a perfect local expert has an unbounded domain of dependence. **[AI Inference]:** for these families the halo rule should not merely be relaxed, it should be declared inapplicable, with the seam certified through the DtN/impedance route instead — otherwise Atlas will keep decertifying experts for a property their governing equation requires them to have.

> **Findings 4 and 5 disagreed about elasticity, and CS-S2 measured which half of each holds (2026-09-10, [[case-study-bounded-donor-atlas-0.1]]).** Finding 4 lists NeuberNet as bounded; finding 5 says quasi-static elasticity is globally coupled whatever the network. Linear elasticity solved classically on NeuberNet's own disc sides with finding 5 **along the boundary**: a poke at one of its $29$ ring sensors moves all $29$, and band-truncating the ring operator still leaves $0.037$–$0.046$ of its in-plane norms outside $r=14$. It does not side with finding 5 **across** the boundary: poked at one sensor, its interior response falls to $12$–$17\%$ between $0.5$ and $2\,R_n$ deep. NeuberNet is more global than that physics in both directions. So finding 4's *bounded* is right about the domain and silent about the response, and finding 5's *DtN, not halos* is right for the boundary operator. **[AI Inference]:** the two findings describe different objects — a domain of dependence and a response — and a donor record needs a column for each.

## What this changes upstream

Three concrete revisions fall out, none of which this page applies unilaterally to the pages that own those decisions:

- **[[training-and-bootstrap-atlas-0.1]]'s reacting-flow row** says "train from scratch, no weight-transfer path." §6 splits that expert: the chemistry half has a downloadable, pointwise, trivially-certifiable donor; only the confined-transport half is from scratch.
- **The same page's thermal-structural row** says "bootstrap from Poseidon." §2 supplies direct evidence the path works (Therm-FM is that fine-tune, published) and §4 confirms the expert must nevertheless be built as two coupled experts, since no coupled donor exists.
- **[[incremental-transfer-roadmap]]'s donor table** lists GP$_{\text{hy}}$T as "structural pattern only — checkpoint unconfirmed." Its weights are public and MIT-licensed. That row is now wrong, and the row's conclusion — "only two donors support weight transfer today" — has grown to at least six.

## What was searched for and not found

Recorded so the next survey does not repeat the search: pretrained public weights for **coupled thermoelasticity** (nothing); **spatially-resolved reacting flow** (nothing — BLASTNet is data); **VOF/level-set free-surface flow** (nothing); **transonic external aerodynamics** (datasets yes — GAOT's RAE2822 span, SuperWing — checkpoints no); **engineering-regime electromagnetics** outside integrated photonics (nothing); **general linear elasticity** (nothing beyond the notch-specific NeuberNet).

## Open questions

1. Does the CC-BY-NC term on Poseidon's weights propagate to Therm-FM's fine-tuned checkpoints? The Therm-FM repo declares Apache-2.0 in its badge and MIT in its LICENSE text, and neither is compatible with an NC parent if the term does propagate. Unresolved, and it gates the only conduction donor.
2. Are the Transolver Elasticity/Plasticity benchmark checkpoints released anywhere? If yes, §3 gains its first continuum-elasticity weight-transfer path and the family stops being empty.
3. What is the actual measured `support_reach` of a **finite-cutoff** donor — MACE, or a GNS-style simulator retrained locally? §3 argues from the config that it is bounded; nobody has run the same delta-poke on it that produced W93's 64. Until that measurement exists, "bounded" is a derivation, not a fact, and the distinction is precisely the one W93 was about.
   > **Partly answered 2026-09-10, and not for the donor this question names.** CS-S2 ([[case-study-bounded-donor-atlas-0.1]]) ran the measurement on **NeuberNet**, whose boundedness is a bounded *patch* rather than a finite *cutoff*, and it came back not compact: bounded in its domain, global in its response. So the distinction this question draws is sharper than it reads — *derivation versus fact* was the right worry, and on the one donor measured the derivation was about the wrong object. For a finite-cutoff donor — MACE, or a GNS-style simulator — the question stands exactly as written.
4. Is the MindSpore integration cost for PDEformer-2 worth paying for the one donor that accepts declared boundary conditions? This trades a real engineering cost against the only escape from finding 2.

---

## See Also

- [[expert-library-atlas-0.1]] — the cut this survey shops for; §4 confirms its thermal-structural argument from the donor side
- [[training-and-bootstrap-atlas-0.1]] — the per-expert bootstrap decisions this survey revises
- [[incremental-transfer-roadmap]] — the donor table this survey supersedes
- [[plug-in-composition-theorems]] — why a checkpoint is not a library entry
- [[interface-transfer-theory]] — the per-seam declaration a donor would have to publish
- [[composition-error-theory]] — the Dirichlet-to-Neumann selection axis that finding 2 measures against
- [[probed-dtn-coupling]] — the coupling route finding 5 says elliptic families require
- [[case-study-ladder-to-f1]] — the substitution campaign this survey supplies candidates for
- [[physics-simulation-datasets]] — the data-side companion to this weights-side survey
- [[code-and-papers]] — where every link here is catalogued
