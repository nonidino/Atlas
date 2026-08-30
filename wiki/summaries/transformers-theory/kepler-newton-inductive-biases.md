# From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers

**Source:** arxiv:2602.06923  
**Authors:** Ziming Liu, Sophia Sanborn, Surya Ganguli, Andreas Tolias  
**Venue:** ICML (Machine Learning)  
**Related Concepts:** [[transformer-architectures]], [[in-context-learning-physics]], [[autoregressive-rollout-stability]], [[world-models-physics-ai]]  
**Related Summaries:** [[gphyt-physics-foundation-model]], [[pde-transformer-paper]]

---

## Overview

This paper investigates why generic transformers fail to acquire "world models" — causal representations that capture governing physical laws — even when trained on synthetic planetary motion data with high predictive accuracy. Using Newtonian/Keplerian motion as a controlled testbed, it identifies three minimal inductive biases that determine whether a transformer becomes a **physicist** (discovers Newton's law $\vec{F} = -GM\vec{r}/|\vec{r}|^3$) or a **curve-fitter** (fits ellipses via Kepler's geometric model).

---

## Key Problem

Prior work (Vafa et al., 2025) showed that a GPT-2-scale transformer trained on planetary trajectories achieves high prediction accuracy without encoding gravitational force in its internal representations. The paper explains **why** this happens and **how to fix it**.

---

## Three Inductive Biases

### Inductive Bias 1: Spatial Smoothness

**Failure mode:** Tokenization discretizes continuous spatial coordinates $(x, y)$ into $V = 7000$ bins with randomly initialized embeddings. Neighboring bins are unrelated before training. The embedding "spatial map" only weakly emerges ($R^2 \approx 0.86$), distorting local structure.

**Solution:** Reduce vocabulary size $V$ (smaller $V$ requires less data to cover) or use **continuous coordinates** (regression formulation).

**Scaling law for spatial map quality:**
$$1 - R^2 \approx A D^{-\alpha_D} V^{\alpha_V}, \quad A = 0.52,\ \alpha_D = 1.15,\ \alpha_V = 1.33$$

Since $\alpha_V > \alpha_D$, training data $D$ must grow at least as fast as vocabulary $V$ to maintain map quality. Embedding dimension $N$ has a critical value $N_c \approx 8$, beyond which it helps little.

---

### Inductive Bias 2: Spatial Stability (Noisy Context Learning)

**Failure mode:** Continuous coordinate regression suffers from catastrophic error accumulation during autoregressive rollout. Errors compound unboundedly, often sending the planet into the sun.

**Solution:** **Noisy context learning** — add Gaussian noise $\sigma$ to input contexts during training:
$$L = \sum_{i=1}^T \left\|\vec{r}_{i+1} - f_\theta(\vec{r}_i + \sigma\epsilon_i, \ldots, \vec{r}_0 + \sigma\epsilon_0)\right\|^2$$

With $\sigma = 0.1$ (optimal intermediate noise), regression consistently outperforms classification (tokenized) across all data scales when hyperparameters are tuned. Discrete tokenization provides an implicit "error correction" via finite vocabulary, explaining why the prior work preferred it naively.

---

### Inductive Bias 3: Temporal Locality

**The critical insight:** Newtonian mechanics is a second-order ODE — the next state depends only on the current state and the immediately previous state ($\Delta t$ resolution). This motivates **context length = 2**.

**Key finding:** Context length controls which world model emerges:

| Context Length | World Model | Internal Representation | $R^2$ for Force | Prediction Error |
|---|---|---|---|---|
| 2 (short) | **Newtonian** — local force computation | Force $F_x, F_y, \|F\|$ | $\approx 0.999$ | Higher (noisier) |
| 100 (long) | **Keplerian** — global curve fitting | Orbital parameters $a, b, \vec{A}$ | $\approx 0.9$ | Lower (smoother) |

The transition is monotonic: increasing context length smoothly shifts the model from Newtonian to Keplerian. There is a **phase transition** between the two regimes.

**Implication:** A Newtonian model understands causation ($F = ma$) and can generalize out-of-distribution. A Keplerian model memorizes trajectory shapes and fails on novel orbital configurations ("pink elephants").

---

## Key Equations

Gravitational equation of motion (ground truth):
$$\frac{d^2\vec{r}}{dt^2} = -GM\frac{\vec{r}}{|\vec{r}|^3}$$

Linear probing $R^2$ measures how well force or orbital variables are linearly encoded in hidden representations.

---

## Main Results

1. **Regression > Classification** when context noise is tuned: regression with $\sigma = 0.1$ consistently beats tokenized classification.
2. **Short context → Newtonian world model** with $R^2 \approx 0.999$ for force representation.
3. **Long context → Keplerian world model** with $R^2 \approx 0.998$ for orbital parameters.
4. Long-context (Keplerian) models have *lower prediction error* despite encoding less physical insight — global fitting is more stable than local force computation.

---

## Relevance to Physics Foundation Models

This paper is a critical benchmark for PFM design:

1. **World models vs. curve-fitting:** A PFM that merely curve-fits trajectories will fail OOD. The three inductive biases are design choices that push toward genuine physical understanding.
2. **Continuous coordinates:** PFM inputs should use continuous field representations, not tokenized bins, to preserve spatial smoothness.
3. **Context length and temporal locality:** For governing-law discovery, restricting context window size forces the model to learn local dynamics — relevant to PDE-based architectures (Neural ODE, RK4 integrators). However, for *prediction accuracy*, longer context improves rollout stability.
4. **Noisy training:** Patch jittering in Walrus / noisy context learning in this paper are both forms of the same bias — robust training against error accumulation. This already appears in our Arch 1 (AR Transformer) spec.

**[AI Inference]:** The Kepler/Newton dichotomy maps directly onto two regimes in physics FM design: (1) data-driven interpolation (Keplerian) vs. (2) physics-law discovery (Newtonian). A PFM trained with very long context on diverse trajectories may be an excellent interpolator but a poor physical law extractor. For scientific discovery rather than simulation, temporal locality constraints may be essential architectural choices.

**[AI Inference]:** The phase transition in world model type with context length suggests that transformer-based PFMs should use **variable context windows** or **multi-scale attention** — long context for prediction, short context for physical law extraction. This could be implemented as a multi-head attention with mixed window sizes.

---

## Cross-Links

- [[world-models-physics-ai]] — theoretical framework for this paper's contribution
- [[autoregressive-rollout-stability]] — noisy context learning connection
- [[in-context-learning-physics]] — contrasts with ICL: here short context → law discovery
- [[transformer-architectures]] — context length as architectural design parameter
- [[gphyt-physics-foundation-model]] — context length scaling in GP$_{\text{hy}}$T
