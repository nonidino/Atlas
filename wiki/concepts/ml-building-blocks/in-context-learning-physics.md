# In-Context Learning for Physical Systems

**Type:** Core Concept  
**Related Sources:** GP$_{\text{hy}}$T, Walrus, AION-1

---

## Definition

**In-context learning (ICL)** refers to the ability of a model to perform new tasks by conditioning on a "prompt" — example inputs and outputs in the context window — without any parameter updates. In LLMs, this manifests as few-shot learning from examples in the prompt.

**In-context learning for physics** is the analogous capability in physical simulation models: a model infers the governing dynamics of an unknown physical system purely from a sequence of observed states (the "physics prompt"), without being told the equations, parameters, or boundary conditions.

---

## The Physics Prompt

For a physics model, the "prompt" is a **short trajectory** of system snapshots:
$$\text{Prompt} = [x_{t-n}, x_{t-n+1}, \ldots, x_t]$$

From this prompt, the model must infer:
- The governing equations (NS, Euler, MHD, etc.)
- The Reynolds number / viscosity / other parameters
- The boundary condition type (periodic, Dirichlet, Neumann, …)
- The spatial scale and temporal scale

Then predict:
$$\hat{x}_{t+1} = f(\text{Prompt}) \approx x_{t+1}$$

---

## Why ICL is Necessary for a PFM

A PFM that requires explicit equation specification defeats the purpose — the user would need to specify the governing equations for every new application, which is:
- Infeasible for experimental data (no known exact equations).
- Expensive for complex multi-physics scenarios.
- Unnecessary if the model can infer dynamics from observations.

ICL enables **one model → many applications** without fine-tuning.

---

## Evidence from GP$_{\text{hy}}$T

GP$_{\text{hy}}$T demonstrates emergent ICL for physics:

1. **New boundary conditions (zero-shot):** Presented with flows using boundary conditions never seen in training, the model successfully infers the correct behavior from the prompt alone.
2. **Entirely novel physics (zero-shot):** For supersonic flow and turbulent radiative layers, GP$_{\text{hy}}$T remains below NMSE = 1 (the trivial mean-image baseline) — the **only** model to do so. All other specialized models diverge.
3. **Context length matters:** Performance scales with prompt length. $N_\text{input} = 1$ (no temporal context) → worst. $N_\text{input} = 4$ → good accuracy/efficiency tradeoff.

---

## Context Length vs. Accuracy

| $N_\text{input}$ | Interpretation | NMSE |
|---|---|---|
| 1 | No temporal context; only spatial snapshot | Highest |
| 2 | Minimal velocity estimate | Large improvement |
| 4 | ~1 physical timescale | Optimal tradeoff |
| 8+ | More context | Diminishing returns |

The improvement from 1→2 snapshots is the largest, because it provides the first estimate of $\partial x/\partial t$ (the dynamics).

---

## ICL for Physics vs. ICL for Language

| Aspect | Language ICL | Physics ICL |
|---|---|---|
| Prompt type | Text examples (input-output pairs) | Temporal snapshots of physical state |
| What is inferred | Task format, answer style | Governing equations, parameters, BCs |
| Evaluation | Accuracy on held-out task | NMSE on next-step / rollout |
| Emergent at scale? | Yes (GPT-3 ~175B) | Emerging (GP$_{\text{hy}}$T at 385M) |

---

## Enabling Conditions

ICL for physics emerges when:
1. **Diverse training data:** Model must see many different physical systems to develop a generalizable dynamics inferencer.
2. **Prompt includes sufficient temporal context:** At least 2 snapshots to estimate derivatives.
3. **No system-specific biases in architecture:** Minimal inductive biases (general transformers outperform physics-specific architectures here).
4. **Variable time increments in training:** Forces the model to infer temporal scale from dynamics, not from fixed $\Delta t$.

---

## Relationship to In-Context Operator Learning (ICON)

ICON (Yang et al., 2023) pioneered ICL for 1D PDEs and ODEs by framing the problem as: "given example input-output function pairs as context, predict the output for a new input." GP$_{\text{hy}}$T extends this to 2D multi-physics systems by using temporal trajectories as context.

---

## Relevance to PFM

ICL is arguably the **most important capability** for a true PFM. A model that requires fine-tuning for each new physical system is a multi-task model; a model that adapts to new systems from context alone is a genuine foundation model. The path from current models to a true PFM runs directly through scaling ICL capabilities.

**[AI Inference]:** ICL for physics may require reasoning not just about dynamics but about physics *structure* — recognizing patterns like "these snapshots show a conserved quantity" or "this exhibits time-reversal symmetry." Encoding such meta-physical reasoning (conservation laws, symmetries, scaling laws) into the context representation could dramatically improve ICL accuracy and generalization range.

---

## See Also

- [[physics-foundation-models]]
- [[transformer-architectures]]
- [[gphyt-physics-foundation-model]]
- [[walrus-paper]]
- [[poseidon-pde-foundation-model]] — few-shot transfer via compositional reuse of pretraining primitives (the finetuning analog of ICL)
