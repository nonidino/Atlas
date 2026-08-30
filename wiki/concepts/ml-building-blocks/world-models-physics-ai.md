# World Models in Physics AI

**Type:** Core Concept  
**Related Sources:** From Kepler to Newton (ICML), GP$_{\text{hy}}$T  
**Related Concepts:** [[in-context-learning-physics]], [[transformer-architectures]], [[autoregressive-rollout-stability]], [[physics-foundation-models]]  
**Related Summaries:** [[kepler-newton-inductive-biases]], [[gphyt-physics-foundation-model]]

---

## Definition

A **world model** is an internal representation or computation within a neural network that captures the **governing mechanisms** (causal, dynamical laws) of a system — not merely its input-output statistics. A world model enables:

1. **Mechanistic understanding:** The model "knows" the governing equation (e.g., $F = ma$, $F \propto 1/r^2$), not just the trajectory shapes.
2. **Radical OOD generalization:** Because it decouples the causal rule from training history, a mechanistic model can correctly predict behavior in strictly out-of-distribution scenarios — "pink elephants" it has never seen.
3. **Scientific discovery:** The model's internal representations can be extracted (e.g., via linear probing or symbolic regression) to recover explicit symbolic laws.

---

## World Model vs. Curve-Fitting

The distinction is illustrated by the Kepler/Newton dichotomy in [[kepler-newton-inductive-biases]]:

| Aspect | Keplerian (Curve-Fitting) | Newtonian (World Model) |
|---|---|---|
| Internal representation | Orbital parameters ($a$, $b$, $\vec{A}$) | Forces ($F_x$, $F_y$, $\lvert F\rvert $) |
| Strategy | Fit global geometric curves | Compute local causal mechanism |
| Context length | Long (needs full trajectory) | Short (needs only 2 past states) |
| Prediction accuracy | Higher (smoother) | Lower (noisier) |
| OOD generalization | Poor (bound to training distribution) | Good (decouples rule from history) |
| Scientific insight | None | Recovers $F \propto 1/r^2$ |

**Key insight:** High prediction accuracy does not imply a world model. A system can achieve low loss by memorizing trajectory distributions while encoding no causal physical law.

---

## Requirements for World Models in Transformers

Research by Liu et al. (ICML, [[kepler-newton-inductive-biases]]) identifies three **minimal inductive biases** needed:

### 1. Spatial Smoothness
Continuous or smoothly-embedded inputs that preserve metric structure. Tokenization with large vocabularies breaks spatial locality — nearby coordinates are treated as unrelated.

**Solution:** Use continuous coordinates with regression loss, or small vocabulary sizes with sufficient training data.

Scaling law for spatial map emergence:
$$1 - R^2 \approx A\, D^{-\alpha_D} V^{\alpha_V}, \quad \alpha_D = 1.15,\; \alpha_V = 1.33$$

### 2. Spatial Stability
Robust training against autoregressive error accumulation. Continuous regression is prone to error growth; discrete tokenization implicitly corrects via finite vocabulary.

**Solution:** Noisy context learning — add noise $\sigma$ to input contexts during training:
$$L = \sum_i \|r_{i+1} - f_\theta(r_i + \sigma\epsilon_i,\, \ldots)\|^2$$

Optimal noise level $\sigma \approx 0.1$ for planetary motion.

### 3. Temporal Locality
Restricting attention to a short context window forces the model to discover local dynamical laws rather than global geometric patterns.

For a second-order ODE (Newtonian mechanics), **context length = 2** is sufficient and necessary. For an $n$-th order ODE, context length $n$ is the natural minimal window.

**Phase transition:** Increasing context length from 2 to 100 causes a monotonic phase transition from Newtonian to Keplerian world model (as measured by $R^2$ linear probing for force vs. orbital parameters).

---

## Probing for World Models

**Linear probing** is the standard method: fit a linear map from hidden representations to a target physical quantity. If $R^2 \approx 1$, the quantity is linearly encoded in the model.

Examples:
- **Newtonian model:** Probe for $F_x$, $F_y$, $|F|$, $r$, $1/r^2$
- **Keplerian model:** Probe for semi-major axis $a$, semi-minor axis $b$, Laplace-Runge-Lenz vector $\vec{A}$

**Limitation:** Probing requires knowing what to look for. An autonomous "AI Physicist" would need a secondary mechanism (symbolic regression head) to extract laws without supervision.

---

## Relationship to In-Context Learning for Physics

There is a tension between world-model discovery and in-context learning:

- **Long context** (ICL regime): Improves prediction accuracy; supports Keplerian fitting; enables few-shot adaptation to new physics via trajectory history.
- **Short context** (world model regime): Recovers local dynamical laws; enables OOD generalization; resembles Newtonian mechanics.

**[AI Inference]:** A PFM may need **separate operating modes** for (1) prediction accuracy (long context, Keplerian) and (2) scientific discovery (short context, Newtonian). This could be implemented as multi-head attention with mixed window sizes, or a hierarchical architecture that uses short-context local dynamics plus long-context global context for conditioning.

---

## Implications for PFM Architecture

1. **Physics encoding:** True world-model capability requires more than soft constraints — it requires architectural biases that force temporal locality.
2. **Short-context layers:** Adding local-context attention heads (context = 2–4) alongside long-context heads could give a PFM access to both dynamical laws and trajectory context.
3. **Symbolic interface:** For scientific discovery applications, a PFM needs an output layer that extracts symbolic laws from latent representations — beyond standard regression heads.
4. **Training regime:** Noisy context learning (patch jittering in Walrus, context noise here) is a general best practice for autoregressive physical simulation models.

---

## Connection to AI Physicist Approaches

Prior "AI Physicist" models (SINDy, AI Feynman, Rediscovering Orbital Mechanics) successfully recover symbolic laws by incorporating strong structural priors (sparsity, symmetry, graph structure). Liu et al. show that **minimal architectural biases** — without domain-specific priors — are sufficient for general-purpose transformers to recover world models, marking a step toward general AI scientists.

---

## Cross-Links

- [[in-context-learning-physics]] — long context ICL vs. short context world models
- [[autoregressive-rollout-stability]] — noisy context learning connection
- [[transformer-architectures]] — context length and attention window design
- [[physics-foundation-models]] — world-model capability as a PFM goal
- [[kepler-newton-inductive-biases]] — primary source
- [[gphyt-physics-foundation-model]] — ICL evidence in GP$_{\text{hy}}$T
