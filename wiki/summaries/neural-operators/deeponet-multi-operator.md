# Summary: DeepONet as a Multi-Operator Extrapolation Model

**Source:** `raw/DeepONet as a Multi-Operator Extrapolation Model Distributed Pretraining with Physics-Informed Fine-Tuning.md`  
**Authors:** Zecheng Zhang, Christian Moya, Lu Lu, Guang Lin, Hayden Schaeffer  
**arXiv:** 2411.07239v1  
**Date Ingested:** 2026-04-11

---

## Overview

Proposes a framework for **multi-operator learning (MOL)** via **distributed pretraining** (D2NO/MODNO) followed by **physics-informed zero-shot fine-tuning**. The central insight: a well-initialized neural operator (pretrained on diverse operators) can be adapted to new downstream operators using only physics-informed loss — **no labeled downstream data required**.

---

## Background: DeepONet Architecture

A DeepONet approximates an operator $G: \mathcal{U} \to \mathcal{V}$ as:

$$G[u](x) \approx G_\theta[\hat{u}](x) = \sum_{k=1}^{K} p_k(\hat{u})\, b_k(x)$$

- **Branch network** $p_k$: encodes the input function discretized at $m$ sensor points.
- **Trunk network** $b_k$: encodes the output function's spatial/temporal coordinates.
- Training minimizes: $\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N \|v_i - G_\theta[u_i](x_i)\|^2$

---

## D2NO / MODNO: Distributed Multi-Operator Learning

**D2NO** partitions data into $C$ sub-datasets with local parameters $\alpha_c$ (per-client/per-operator branch) and shared global parameters $\beta$ (trunk). Training alternates:

1. **Local update:** $\alpha_c \leftarrow \alpha_c - \eta_c \nabla_{\alpha_c} L_c(\alpha_c;\beta)$
2. **Global sync:** $\beta \leftarrow \beta - \eta \nabla_\beta L(\beta;\alpha)$ where $L(\beta;\alpha) = \sum_c L_c(\alpha_c;\beta)$

After pretraining, local parameters are averaged:
$$\alpha = \frac{1}{C}\sum_{c=1}^C \alpha_c$$
This averaging preserves diversity of operator responses and provides a strong warm-start initialization.

---

## Fine-Tuning Methods

### Full Fine-Tuning
All weights updated: $\tilde{W} = W + \Delta W$. Direct gradient descent on physics-informed loss.

### LoRA Fine-Tuning
Low-rank adaptation: introduces $A \in \mathbb{R}^{m\times r}$, $B \in \mathbb{R}^{r\times n}$ with $r \ll \min(m,n)$:
$$\tilde{W} = W + AB$$
Only $A$ and $B$ are trained; $W$ is frozen. Dramatically reduces trainable parameters while preserving pretrained knowledge.

---

## Physics-Informed (PI) Fine-Tuning

For a PDE like $u_t - u_{xx} = f$ with periodic boundary conditions, the PI loss minimizes:
$$\omega_1 \sum_{i,j} |\mathcal{AD}_t\{G_\theta[u_i]\} - \mathcal{AD}_{xx}\{G_\theta[u_i]\} - f|^2 + \omega_2 \sum_{i,j} |G_\theta[u_i](x,0) - u_i(x)|^2 + \omega_3 \sum_{i,j} |G_\theta[u_i](0,t) - G_\theta[u_i](1,t)|^2$$

$\mathcal{AD}$ denotes automatic differentiation. This allows **zero-shot fine-tuning** — no labeled solution data needed for the target operator.

---

## Key Results

- D2NO-initialized PI fine-tuning significantly outperforms random initialization + PI training.
- LoRA achieves comparable accuracy with fewer trainable parameters.
- Full fine-tuning with D2NO init achieves best accuracy for complex nonlinear operators.
- Demonstrates that **diverse pretraining provides a universally better starting point** for operator fine-tuning.

---

## Relevance to Physics Foundation Model Goal

This work directly addresses the **transfer learning** problem in neural operator learning. For a physics foundation model, the ability to zero-shot adapt to new PDE systems using only known physics (not labeled data) is enormously valuable. The D2NO/MODNO framework provides a rigorous formulation of how to train a single shared operator backbone that can specialize efficiently.

**[AI Inference]:** The LoRA technique from NLP has proved immediately applicable to operator learning — suggesting that the operator-learning community can systematically import PEFT (parameter-efficient fine-tuning) advances from LLM research. A natural extension would be combining D2NO pretraining with soft-prompt or prefix-tuning approaches adapted to operator conditioning.

---

## Links

- [[neural-operators]] — DeepONet fundamentals
- [[transfer-learning-fine-tuning]] — core theme of this paper
- [[partial-differential-equations]] — the operators being learned
- [[pc-deeponet-cfd]] — related physics-constrained DeepONet work
- [[varmion-viscous-flows]] — related operator network variant
- [[physics-foundation-models]] — broader goal
