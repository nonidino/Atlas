# PFM Interface Design: Communication Without Text

**Type:** Core Concept  
**Date:** 2026-06-26  
**Related Concepts:** [[physics-foundation-models]], [[pfm-concept-overview]], [[in-context-learning-physics]], [[world-models-physics-ai]], [[autoregressive-rollout-stability]], [[multimodal-tokenization]]  
**Related Summaries:** [[gphyt-physics-foundation-model]], [[pisd-physics-informed-spectral-diffusion]], [[kepler-newton-inductive-biases]]

---

## The Interface Problem

A PFM is not trained on text. Its native language is physical states — tensor fields defined on spatial domains. This raises three design questions that are distinct from the architectural questions addressed elsewhere:

1. **Task specification:** How does the model know what task to perform (forward simulation, inverse problem, uncertainty estimation) without a text instruction?
2. **Timescale handling:** Physics is continuous; model predictions are discrete. How does the model handle the 30-order-of-magnitude range from quantum ($\sim 10^{-15}$ s) to planetary ($\sim 10^{15}$ s) timescales?
3. **Regime specification:** How does the model adapt to different physical regimes (fluid vs. plasma vs. quantum) without text prompts?

The answer to all three is the same principle: **the interface is physical metadata, and the task is the computational pattern, not an instruction.**

---

## Principle 1: Tasks Are Computational Patterns, Not Instructions

A classical PDE solver doesn't need to be told "do forward simulation." You just call `solver.step(u0, dt)`. The task is determined by how the solver is invoked — the orchestration layer, not the solver itself. A PFM works the same way.

Each task maps to a different invocation pattern with purely physical inputs and outputs:

| Task | How to invoke the PFM | Input format | Output format |
|---|---|---|---|
| Forward simulation | Feed context; take next-state output | $[x_{t-n}, \ldots, x_t]$ | $\hat{x}_{t+1}$ |
| Long-horizon rollout | Autoregressively feed each output as new input | $[x_{t-n}, \ldots, x_t]$ (initially GT, then predicted) | $\hat{x}_{t+1}, \hat{x}_{t+2}, \ldots$ |
| Inverse problem | Feed target state; use DPS to find source | $x_T$ (target) + physics residual guidance | $\hat{x}_0$ (initial condition) |
| Uncertainty quantification | Sample multiple outputs from generative backbone | $[x_{t-n}, \ldots, x_t]$ | Distribution $p(\hat{x}_{t+1} \mid x_{t-n:t})$ |
| Multi-physics coupling | Multi-channel input with field type codes | $[\text{vel}, \text{pres}, \text{temp}, \ldots]$ as separate channels | Next-state multi-channel output |
| Parameter inference | Optimize input parameters to minimize prediction error on observed trajectory | Parametric family + observed trajectory | $\hat{\theta}$ (inferred parameters) |

None of these require text. The "instruction" is implicit in the computational pattern. The PFM doesn't need to know which task it is doing — it just predicts the next physically consistent state given its context.

### What text would actually add

Text training would enable:
- **Natural language specification of boundary conditions:** "no-slip walls with temperature fixed at 300K" instead of a spatial mask + discrete BC code + parameter value.
- **Symbolic law extraction:** Outputting equations like $\nabla \cdot u = 0$ or $F \propto 1/r^2$ directly.
- **Scientific report generation:** Summarizing simulation results in human-readable form.

These are valuable but constitute a **separate capability layer** on top of the core physics model — not part of the core PFM. Adding text capability is a full additional training challenge (joint tokenization of physics fields and text) and would make more sense as fine-tuning on a frozen physics-capable backbone, analogous to LLaVA freezing a vision encoder and training only a projector to a text LLM.

**Scientific law discovery** in particular cannot be a native PFM output if the model outputs physical fields. Extracting symbolic laws requires a post-processing pipeline: symbolic regression (e.g., PySR, AI Feynman) on the PFM's latent representations, or linear probing to check what quantities are linearly encoded in the hidden state. This is a downstream analysis tool, not a core model capability.

---

## Principle 2: Nondimensionalization Unifies Timescales

### The Core Idea

Physics problems span 30 orders of magnitude in time. A model trained on dimensional data would need to explicitly learn that "1 femtosecond of quantum dynamics" and "1 millennium of orbital mechanics" are both "one step." This is intractable.

The physicists' solution: **nondimensionalize before passing data to the model.** Divide every quantity by its characteristic scale so the model always sees dimensionless numbers of order $O(1)$.

For any physical system, the user identifies:
- Characteristic length $L_0$
- Characteristic time $T_0$ (e.g., orbital period, plasma oscillation period, $\hbar/E$ for quantum)
- Characteristic velocity $U_0 = L_0 / T_0$
- Characteristic field value $\phi_0$ (pressure, temperature, field strength)

All inputs to the model are rescaled:
$$\hat{x} = x / \phi_0, \quad \hat{\ell} = \ell / L_0, \quad \hat{t} = t / T_0, \quad \hat{\Delta t} = \Delta t / T_0$$

After this rescaling, a quantum simulation with $\Delta t = 10^{-15}$ s and $T_0 = 10^{-14}$ s produces $\hat{\Delta t} = 0.1$. A planetary simulation with $\Delta t = 10^6$ s and $T_0 = 10^7$ s also produces $\hat{\Delta t} = 0.1$. **The model sees identical dimensionless inputs** and is correctly applied to both.

### The Dimensionless Parameter Set

Nondimensionalization produces a small set of **dimensionless governing parameters** that characterize the physical regime. For fluid dynamics:

$$\text{Re} = \frac{U_0 L_0}{\nu} \quad (\text{Reynolds}), \qquad \text{Ma} = \frac{U_0}{c_s} \quad (\text{Mach}), \qquad \text{Pr} = \frac{\nu}{\alpha} \quad (\text{Prandtl})$$

These numbers, together with the dimensionless field data, fully characterize the flow. A model trained on a distribution of $(\text{Re}, \text{Ma}, \text{Pr})$ values learns a map from dimensionless inputs to dimensionless outputs that transfers across all physical scales.

### The $\hat{\Delta t}$ Token

The dimensionless timestep $\hat{\Delta t}$ is fed as an **explicit input conditioning token**, not a fixed architectural parameter. This is the key distinction:

- **Hyperparameter approach (wrong):** The model is trained at $\hat{\Delta t} = 0.01$; different $\hat{\Delta t}$ requires retraining.
- **Conditioning approach (right):** The model is conditioned on $\hat{\Delta t}$, which it receives as input. Different $\hat{\Delta t}$ at inference time is handled automatically.

$\hat{\Delta t}$ is embedded via a small MLP or sinusoidal encoding and added to the token embeddings (analogous to how positional encodings are added to patch tokens):

$$z_{\text{time}} = \text{MLP}(\log \hat{\Delta t}), \qquad \tilde{z}_i = z_i + z_{\text{time}} \quad \forall i$$

Using $\log \hat{\Delta t}$ rather than $\hat{\Delta t}$ handles the large dynamic range — the model sees a log-scale input, which is geometrically uniform across the range $[10^{-4}, 1]$.

### Alternative: Derivative Prediction (GP$_{\text{hy}}$T approach)

Instead of predicting $\hat{x}_{t+1}$ directly, the model predicts $\partial_{\hat{t}} \hat{x}$ (the dimensionless time derivative). An external integrator then computes:

$$\hat{x}_{t+\hat{\Delta t}} = \hat{x}_t + \hat{\Delta t} \cdot \partial_{\hat{t}} \hat{x}(\hat{x}_t) + O(\hat{\Delta t}^2)$$

The model's output is **$\hat{\Delta t}$-agnostic by construction** — it outputs a rate, not a next state. The user applies whatever timestep they choose. This is the most principled approach for adaptive timestepping (where $\hat{\Delta t}$ changes during a single rollout to resolve shocks or rapid transitions) and is the approach used by GP$_{\text{hy}}$T (7× SOTA).

---

## Principle 3: The Minimal Physical Interface

Five inputs fully specify a physics simulation problem, with no text required:

### Input 1: Field Data (the context trajectory)

$$\mathcal{X}_\text{context} = [\hat{x}_{t-n}, \hat{x}_{t-n+1}, \ldots, \hat{x}_t], \quad \hat{x}_\tau \in \mathbb{R}^{N_\text{spatial} \times C}$$

where $N_\text{spatial}$ is the number of spatial locations (grid points or particles) and $C$ is the number of field channels (velocity components, pressure, temperature, etc.). Each field value is dimensionless.

Minimum context length = order of the governing PDE (see [[world-models-physics-ai]]):
- First-order PDEs (heat equation): $n = 1$ sufficient in principle
- Second-order PDEs (Navier-Stokes, wave equation): $n \geq 2$ required for world-model mode
- In practice: $n = 4$–$8$ recommended for stability

### Input 2: $\hat{\Delta t}$ Token (the temporal resolution)

A single scalar, log-embedded and added to all token representations. Informs the model of the temporal resolution relative to the system's characteristic timescale. The model learns to predict dynamics at any $\hat{\Delta t} \in [10^{-4}, 1]$ without retraining.

### Input 3: Field Type Codes (what each channel represents)

A discrete code per input channel from a vocabulary of ~50–100 physical field types:

| Code | Meaning | Code | Meaning |
|---|---|---|---|
| `vel_x`, `vel_y`, `vel_z` | Velocity components | `B_x`, `B_y`, `B_z` | Magnetic field |
| `pressure` | Scalar pressure | `E_x`, `E_y`, `E_z` | Electric field |
| `temperature` | Scalar temperature | `density` | Mass density |
| `vorticity_z` | Out-of-plane vorticity | `psi` | Wave function (real) |

The model uses these codes to understand which physical transformation laws apply to each channel (e.g., velocity components transform under rotation differently from scalar pressure — relevant for equivariant processing).

**[AI Inference]:** Field type codes serve the same role as modality-specific tokens in AION-1 [[aion-1-astronomy]] — they tell the model what kind of data it is receiving so it can apply the appropriate learned representation. A physics vocabulary of ~100 field types would cover all standard continuum and particle physics fields.

### Input 4: Dimensionless Physical Parameters

A small set of governing dimensionless numbers, embedded as conditioning tokens alongside the field data:

$$\text{params} = [\log \text{Re},\ \log \text{Ma},\ \text{Pr},\ \text{Ro},\ \ldots]$$

Using log-scale for parameters spanning many orders of magnitude. The model learns that $\text{Re} = 10^2$ (laminar), $\text{Re} = 10^4$ (transitional), $\text{Re} = 10^6$ (turbulent) are distinct dynamical regimes. Unknown parameters can be left out and inferred from context (this is the in-context learning regime).

**[AI Inference]:** Leaving governing parameters unspecified and letting the model infer them from context is the most powerful form of in-context learning for physics — analogous to how a good physicist can estimate Reynolds number from looking at a flow visualization. A model trained on labeled (field data, Re, Ma, Pr) pairs will develop internal representations of these parameters even when they are not provided, which can be probed via linear regression on the latent space.

### Input 5: Boundary Condition Specification

Domain geometry + boundary condition type, encoded compactly:

- **Domain mask:** A spatial binary mask indicating which grid points are interior vs. boundary. Shape: $N_\text{spatial}$ boolean values.
- **BC type codes:** For each boundary region, a discrete code: `periodic`, `no-slip`, `free-slip`, `Dirichlet`, `Neumann`, `absorbing`, `inflow`, `outflow`.
- **BC values:** For Dirichlet BCs, the fixed field values at the boundary (encoded as additional channels at boundary locations).

For irregular geometry (turbine blades, biological tissue), the ghost-node approach from [[dynami-cal-graphnet]] handles arbitrary boundaries via reflection/extension, encoding the geometry implicitly in the graph structure rather than requiring explicit domain masks.

---

## The Complete Interface

$$\text{Input to PFM} = \underbrace{(\hat{x}_{t-n:t},\ \hat{\Delta t},\ \text{field codes},\ \text{params},\ \text{BCs})}_{\text{fully specifies physics without text}}$$

$$\text{Output from PFM} = \underbrace{\hat{x}_{t+1}}_{\text{or}\ \partial_{\hat{t}}\hat{x} \text{ for derivative prediction}}$$

This interface is:
- **Complete:** Fully specifies any classical physics simulation problem
- **Physical:** Every input is a physically meaningful quantity, not a neural network hyperparameter
- **Compact:** $\sim 5$ inputs total (vs. dozens of hyperparameters in a numerical solver configuration)
- **Text-free:** No natural language at any step
- **Scale-agnostic:** Nondimensionalization handles any physical scale range

The model is **robust to missing inputs**: if dimensionless parameters are not provided, the model infers them from context. If the $\hat{\Delta t}$ range is wide, the log-embedding handles it. If boundary conditions are approximate, the model uses physical plausibility to fill in. Robustness to the exact configuration is a learned property, not an architectural guarantee.

---

## What In-Context Inference Handles Automatically

Much of what might seem like "configuration" is actually handled by the in-context learning mechanism:

- **What physical system is this?** — Inferred from the trajectory. Compressible flow looks different from incompressible; plasma dynamics from neutral fluid dynamics. The model has seen all of these and can recognize the signatures.
- **What are the governing parameters?** — Inferred from the trajectory statistics. A turbulent trajectory (many active scales, chaotic mixing) implies high Re. The model doesn't need to be told.
- **What physical regime is this?** — Inferred from the input field type codes + trajectory context together.

This is exactly the **Newtonian / world-model** mode of operation from [[kepler-newton-inductive-biases]]: short context + right architecture → model infers local causal laws. The goal is for the model to internalize the connection between trajectory patterns and governing physics, so explicit parameter specification becomes optional (not required).

---

## Future Text Capability as a Separate Layer

If text communication becomes desirable (for natural-language problem specification, symbolic law output, or scientific report generation), the right architecture is a **two-layer system**:

1. **Physics backbone** (PFM): Takes physical state inputs → produces physical state outputs. No text. Trained on simulation data.
2. **Language bridge** (frozen physics backbone + trainable connector + LLM): A lightweight projector aligns PFM latent representations to a text LLM embedding space. Trained on (simulation, description) pairs. The physics backbone remains frozen.

This is exactly the BLIP-2 / LLaVA pattern [[multimodal-transformers-survey]]: freeze the vision encoder (= physics backbone), train only the projector (= field-to-language bridge), leverage an existing LLM for text generation. The PFM core remains physics-specialized; language understanding is added without contaminating the physics representation.

**[AI Inference]:** This two-layer approach means text training and physics training can proceed independently, solving the otherwise-intractable joint tokenization problem. A physics backbone trained first to simulate well can subsequently have language capability attached via a small trained connector — analogous to how LLaVA attaches language to a frozen ViT. The physics backbone is not penalized for being unable to describe what it simulates.

---

## Open Questions

1. **How many field type codes are sufficient?** A vocabulary of 50–100 covers standard continuum and particle physics. Exotic fields (gravitational tensor, spinor fields, lattice QCD) may require new codes — is the vocabulary open or closed?
2. **How should dimensionless parameters be embedded when unknown?** A learned "unknown" token that the model treats as "infer this from context" — analogous to a [MASK] token in BERT.
3. **Is nondimensionalization always possible?** Strongly nonlinear systems near criticality (phase transitions, turbulence onset) may not have clean characteristic scales. The interface needs to handle near-critical and multiscale regimes where $L_0, T_0$ are ill-defined.
4. **Can the model be robustly conditioned on $\hat{\Delta t}$ across 4+ orders of magnitude?** This requires $\hat{\Delta t}$ to appear in training data at all scales. Curriculum training (start at fixed $\hat{\Delta t}$, then randomize) may be necessary.

---

## See Also

- [[physics-foundation-models]] — capability requirements that the interface must support
- [[pfm-concept-overview]] — high-level vision; note that "scientific law discovery" requires post-processing, not native text output
- [[in-context-learning-physics]] — how trajectory context replaces explicit problem specification
- [[world-models-physics-ai]] — context length / ODE order relationship; minimum context requirements
- [[multimodal-tokenization]] — how physical fields are tokenized; field type codes
- [[autoregressive-rollout-stability]] — context quality degrades during rollout; relates to context length choice
- [[possible-architectures]] — Physics-Mamba uses $\Delta t$-as-input explicitly
- [[gphyt-physics-foundation-model]] — derivative prediction approach; $\Delta t$-agnostic output
- [[pisd-physics-informed-spectral-diffusion]] — DPS guidance for inverse problems; test-time constraint enforcement
