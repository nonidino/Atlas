# Code & Papers — External Resources

Central index of code repositories and source links for everything in the wiki. Keeps implementation one click away from the research.

---

## 🔗 Project repository (the code home)

**[github.com/nonidino/physics-foundation-model](https://github.com/nonidino/physics-foundation-model)** *(private)*

> *"The development for an actual physics foundation model, being for physics what LLMs are for language."*

This is where the **actual framework/model implementation** will live. The wiki (this Obsidian vault) is the **research & design layer**; the GitHub repo is the **build layer**. As the framework is implemented, link specific modules/commits back into the relevant concept pages (e.g. a `arch-physics-mamba` implementation links from [[arch-physics-mamba]]).

| | |
|---|---|
| **Repo** | `nonidino/physics-foundation-model` |
| **Branch** | `main` (README + LICENSE only) |
| **Branches — build** | `noether-1.0` (frozen 1.0 reference) · `noether-1.1` (active Noether track, [[implementation-log]]) · **`atlas-0.1`** (Atlas track, cut from `noether-1.1` 2026-08-09, package `src/atlas/`, Phase 1 built — [[atlas-0.1-implementation-log]]) · **`atlas-0.1-windfarm`** (wind-farm case study, cut from `atlas-0.1` 2026-08-20; `src/atlas/cases/windfarm/` + `src/atlas/ports/`; **gates W0–W3 pass, nothing trained** — [[results-w0-w3-wind-farm]], [[wind-farm-implementation-log]]) |
| **Composition layer** | **`atlas/` — the generalized plug-in composition layer, built 2026-08-27** ([[atlas-implementation]]). 23 modules, 120 tests, `numpy` only; drop-in for `src/atlas/` on the `atlas-0.1` branch. Implements [[end-to-end-architecture-spec]], [[interface-transfer-theory]] and [[plug-in-composition-theorems]] end to end: the coupling compiler, the capability-record schema and its conformance suite, the port algebra with the derived mapping and adjointness, the probed-DtN operator on a common interface space, the envelope stamp, the three-verdict gate and the emit contract. **A new case study is declarations only** |
| **Status** | Two parallel architecture tracks under active build |
| **Role** | Implementation home for architectures designed in [[possible-architectures]] and the `benchmark-architectures/` specs |

**[AI Inference]:** A natural layout for the repo as it grows: `models/` (one module per benchmark architecture), `benchmarks/` (the Burgers' suite from [[possible-architectures]]), `data/` (Genesis-generated trajectories, see [[genesis-physics-engine]]), `notebooks/`. Each model module should cross-link to its wiki spec page so research ↔ code stays synchronized.

---

## Reference implementations (per source paper)

Official / canonical code for the papers summarized in `summaries/`. Fill `arXiv` and `Code` as confirmed — placeholders marked *(find)* still need verification.

### Foundation models
| Wiki page | arXiv | Code |
|---|---|---|
| [[poseidon-pde-foundation-model]] | [2405.19101](https://arxiv.org/abs/2405.19101) | [camlab-ethz/poseidon](https://github.com/camlab-ethz/poseidon) · models/data: [huggingface.co/camlab-ethz](https://huggingface.co/camlab-ethz) (PDEgym) |
| [[gphyt-physics-foundation-model]] | [2509.13805](https://arxiv.org/abs/2509.13805) | *(find)* |
| [[walrus-paper]] | [2511.15684](https://arxiv.org/abs/2511.15684) | [PolymathicAI/walrus](https://github.com/PolymathicAI/walrus) · [hf: polymathic-ai/walrus](https://huggingface.co/polymathic-ai/walrus) |
| [[aion-1-astronomy]] | *(find)* | *(find)* |
| [[multimodal-transformers-survey]] | [2311.13165](https://arxiv.org/abs/2311.13165) | — (survey) |

### Transformers / theory
| Wiki page | arXiv | Code |
|---|---|---|
| [[pde-transformer-paper]] | *(find)* | *(find)* |
| [[transformer-mathematical-framework]] | [2510.03989](https://arxiv.org/abs/2510.03989) | — (theory) |
| [[transformers-particle-systems-clustering]] | [2312.10794](https://arxiv.org/abs/2312.10794) | — (theory) |
| [[kepler-newton-inductive-biases]] | [2602.06923](https://arxiv.org/abs/2602.06923) | *(find)* |
| [[physicsformer-pinn-ns]] | [2601.03613](https://arxiv.org/abs/2601.03613) | *(find)* |
| [[conditional-memory-engram]] | [2601.07372](https://arxiv.org/abs/2601.07372) | *(find)* |
| [[glm-index-share-attention]] | — (vendor blog; Zhipu AI GLM 5.2) · [mindstudio.ai writeup](https://www.mindstudio.ai/blog/glm-5-2-architecture-index-share-sparse-attention) · cf. Routing Transformers [2003.05997](https://arxiv.org/abs/2003.05997) | *(find)* |
| [[q-former-architecture]] | BLIP-2 [2301.12597](https://arxiv.org/abs/2301.12597) · HierarQ [2503.08585](https://arxiv.org/abs/2503.08585) · DisenQ [2507.07262](https://arxiv.org/abs/2507.07262) · Q-Former PEFT [2410.09489](https://arxiv.org/abs/2410.09489) | [salesforce/LAVIS (BLIP-2)](https://github.com/salesforce/LAVIS) |

### Neural operators
| Wiki page | arXiv | Code |
|---|---|---|
| [[deeponet-multi-operator]] | *(find)* | *(find)* |
| [[pc-deeponet-cfd]] | *(find)* | *(find)* |
| [[varmion-viscous-flows]] | *(find)* | *(find)* |

### Graph networks / equivariant GNNs
| Wiki page | arXiv | Code |
|---|---|---|
| [[gns-graph-network-simulators]] | [2002.09405](https://arxiv.org/abs/2002.09405) | [deepmind/deepmind-research](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate) |
| [[dynami-cal-graphnet]] | [2501.07373](https://arxiv.org/abs/2501.07373) | *(find)* |
| [[equiformer-v3]] | [2604.09130](https://arxiv.org/abs/2604.09130) | *(find)* |
| [[eqnn-hep-lhc]] | *(find)* | *(find)* |
| [[multipole-graph-neural-operator]] | [2006.09535](https://arxiv.org/abs/2006.09535) | [neuraloperator/graph-pde](https://github.com/neuraloperator/graph-pde) |
| [[anti-symmetric-dgn]] | [2210.09789](https://arxiv.org/abs/2210.09789) | *(find)* |

### Hamiltonian / Lagrangian / action-based
| Wiki page | arXiv | Code |
|---|---|---|
| [[hamiltonian-neural-networks]] | [1906.01563](https://arxiv.org/abs/1906.01563) | [greydanus/hamiltonian-nn](https://github.com/greydanus/hamiltonian-nn) |
| [[dissipative-hamiltonian-neural-networks]] | [2201.10085](https://arxiv.org/abs/2201.10085) | *(find)* |
| [[lagrangian-neural-networks]] | [2003.04630](https://arxiv.org/abs/2003.04630) | [MilesCranmer/lagrangian_nns](https://github.com/MilesCranmer/lagrangian_nns) |
| [[noether-networks]] | [2112.03321](https://arxiv.org/abs/2112.03321) | *(find)* |

### Diffusion / generative
| Wiki page | arXiv | Code |
|---|---|---|
| [[latent-diffusion-physics]] | *(find)* | Polymathic AI *(find)* |
| [[multiscale-diffusion-solar]] | *(find)* | *(find)* |
| [[pisd-physics-informed-spectral-diffusion]] | [2602.09708](https://arxiv.org/abs/2602.09708) | *(find)* |

### Solvers & simulation
| Wiki page | arXiv / site | Code |
|---|---|---|
| [[navier-stokes-nonuniform-grids]] | *(find)* | *(find)* |
| [[genesis-physics-engine]] | [genesis-embodied-ai.github.io](https://genesis-embodied-ai.github.io/) | [Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis) |
| [[rl-for-flow-control-and-coordination]] *(assessment, not a summary — paper not yet ingested)* | HydroGym: [2512.17534](https://arxiv.org/abs/2512.17534) · [Nature s41586-026-10917-6](https://www.nature.com/articles/s41586-026-10917-6) *(paywalled)* · [PMLR v283](https://proceedings.mlr.press/v283/lagemann25a.html) · [docs](https://dynamicslab.github.io/hydrogym/) | [dynamicslab/hydrogym](https://github.com/dynamicslab/hydrogym) |

### Other methods
| Wiki page | arXiv | Code |
|---|---|---|
| [[message-passing-cyclicity]] | *(find)* | *(find)* |
| [[deep-memory-dissipative]] | *(find)* | *(find)* |
| [[gaussian-process-regression]] | [2009.10862](https://arxiv.org/abs/2009.10862) | — (tutorial) |

---

## 📊 Public simulation datasets

Catalogued and coverage-mapped in [[physics-simulation-datasets]]. Listed by governing-equation family, since that — not scenario — is what determines usability.

| Dataset | Family / content | Link |
|---|---|---|
| **The Well** | 15 TB, 16 datasets — fluids, MHD, acoustics, active matter, supernova; includes compressible Euler | [arXiv 2412.00568](https://arxiv.org/abs/2412.00568) · [PolymathicAI/the_well](https://github.com/PolymathicAI/the_well) · [HF collection](https://huggingface.co/collections/polymathic-ai/the-well) |
| **PDEgym** | 77,840 trajectories, 4 compressible Euler + 2 incompressible NS operators; Poseidon's pretraining corpus | [huggingface.co/camlab-ethz](https://huggingface.co/camlab-ethz) · [[poseidon-pde-foundation-model]] |
| **BLASTNet 2.0** | **2.2 TB, 744 full-domain samples from 34 DNS** — reacting *and* non-reacting compressible turbulence. The combustion gap-filler Atlas's Phase 2 overlooked | [blastnet.github.io](https://blastnet.github.io/pub) · [orig. paper](https://www.sciencedirect.com/science/article/pii/S2666352X22000309) · [2.0 benchmark](https://openreview.net/forum?id=ugRnHKMK95) |
| **AirfRANS** | steady incompressible subsonic RANS over 2D NACA airfoils; NeurIPS 2024 ML4CFD competition basis | [airfrans.readthedocs.io](https://airfrans.readthedocs.io/en/latest/notes/introduction.html) |
| **PDEBench** | compressible NS across Mach/viscosity ranges; includes *viscous* compressible, which PDEgym lacks | *(find)* |
| **Flowbench** | flow over complex geometries; part of Walrus's 19-scenario set | [[walrus-paper]] |
| **TFRD** | temperature-field reconstruction for heat-source systems (steady, thermal only, no mechanics) | [arXiv 2108.08298](https://arxiv.org/pdf/2108.08298) |
| **DeepEDH (CHT)** | 1,500 FEM conjugate-heat-transfer sims, battery cold plate; one-off, thermal only | [arXiv 2311.17068](https://arxiv.org/pdf/2311.17068) |
| **PLAID** | unified data model for ML on heterogeneous physics simulations — infrastructure, not a corpus | [arXiv 2505.02974](https://arxiv.org/pdf/2505.02974) |

> **Not found anywhere** (see [[physics-simulation-datasets]] §3.6): confined nozzle flow through a **choking throat**, and **interface flux fields on labelled subdomain boundaries** — the latter structurally, since no other architecture declares interfaces to label.

---

*Maintenance: when ingesting a new source, add its row here with arXiv + code link. When the project repo gains a module implementing a wiki architecture, cross-link the module from that architecture's concept page.*
