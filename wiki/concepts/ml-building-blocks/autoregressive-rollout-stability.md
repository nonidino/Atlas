# Autoregressive Rollout Stability

**Type:** Core Concept  
**Related Sources:** GP$_{\text{hy}}$T, Walrus, Lost in Latent Space, Multiscale Diffusion Solar, PDE-Transformer

---

## The Problem

**Autoregressive rollout** refers to iteratively applying a one-step predictor:
$$\hat{x}_{t+k} = f(\hat{x}_{t+k-1}) = f^k(x_t)$$

The critical challenge: **small errors compound**. With one-step defect $\epsilon$ and $L=\mathrm{Lip}(f_\theta)$ in the relevant norm, the standard recursion $\lVert e^{k+1}\rVert\le L\lVert e^k\rVert+\epsilon$ gives

$$\lVert \hat x_{t+k}-x_{t+k}\rVert \;\le\; \frac{L^k-1}{L-1}\,\epsilon \;=\;\begin{cases} \dfrac{\epsilon}{1-L}, & L<1 \quad\text{(contractive — \textbf{bounded uniformly in }}k\text{)}\\[6pt] k\,\epsilon, & L=1 \quad\text{(non-expansive — \textbf{linear})}\\[6pt] \dfrac{L^k}{L-1}\,\epsilon, & L>1 \quad\text{(\textbf{exponential})} \end{cases}$$

so for chaotic systems, with $\lambda$ the largest Lyapunov exponent and $L\approx e^{\lambda\Delta t}$:

$$\lVert \hat x_{t+k}-x_{t+k}\rVert \;\sim\; \epsilon\, e^{\lambda k \Delta t}$$

**Corrected 2026-08-26.** This paragraph previously read *"the error grows approximately as $\epsilon^k$ for stable systems"*, which is wrong in the direction that matters — for $\epsilon<1$ that expression **decays**, and it would say a stable system forgets its own accumulated error faster than it commits new error. The governing quantity is $L$, not $\epsilon$: **only $L>1$ gives exponential growth, and $L\le1$ is a property a model can be built to have.** See [[temporal-error-accumulation]], where this recursion is the whole argument, and [[composition-error-theory]] §2.3, which states it correctly.

---

## Sources of Error

1. **Model error:** The learned predictor $f_\theta$ is not identical to the true dynamics $f^*$.
2. **Tokenization artifacts:** Patching/tokenization breaks translation equivariance; artifacts are locked onto by the model and amplified.
3. **Distribution shift:** After $k$ steps, the model is predicting from states it was not trained on (training distribution ≠ inference distribution at step $k$).
4. **Spectral bias:** Neural networks preferentially learn low-frequency components; high-frequency errors (shockwaves, sharp boundaries) accumulate faster.

---

## Walrus: Patch Jittering

Root cause: fixed tokenization creates grid artifacts that the model learns to rely on, leading to exponential error growth when the grid pattern doesn't match training.

**Fix:** Before tokenization, randomly "jitter" (shift) the input data by a small random offset at each step. This prevents the model from locking onto specific grid patterns.

**Theoretical basis:** Derived from harmonic analysis of aliasing in patch-based compression. Analogous to stochastic data augmentation but motivated by frequency-domain analysis.

**Result:** Reduces long-horizon error in 89% of pretraining scenarios.

---

## GP$_{\text{hy}}$T: Neural Differentiator

Root cause: predicting $x_{t+1}$ directly from $x_t$ conflates learning dynamics with learning integration, making the problem scale-dependent.

**Fix:** Predict the time derivative $\partial x/\partial t$ and use numerical integration:
$$x_{t+1} = x_t + \int_t^{t+\Delta t} f_\theta(x)\, dt$$

This decouples dynamics learning from integration, enabling generalization across different $\Delta t$ values. In ablation: ~10× lower NMSE in long-horizon rollouts compared to direct next-state prediction.

---

## Diffusion Emulators

Root cause: deterministic predictors produce point estimates that diverge from the true distribution in chaotic systems — the "mean trajectory" is unphysical.

**Fix:** Use generative (diffusion) models that sample from $p(x_{t+1} \mid x_t)$ rather than predicting $\mathbb{E}[x_{t+1}\mid x_t]$. The ensemble of samples stays near the true distribution even when individual trajectories diverge.

**Temporal bundling:** Predict $n$ steps at once, reducing the number of autoregressive calls (and thus error accumulation opportunities) by factor $n$.

---

## Multiscale Inference

Root cause: standard autoregressive rollout discards distant past information, losing long-range temporal context needed for systems with long memory.

**Fix:** Multiscale templates condition on both fine-grained recent frames and coarse-grained past frames simultaneously, allowing the model to correct long-range drift.

---

## Quantitative Perspective

| Method | Long-Horizon Error Growth | Key Mechanism |
|---|---|---|
| Standard AR | Exponential (diverges) | Error lock-in |
| GP$_{\text{hy}}$T | Sub-exponential | Derivative prediction |
| Walrus + patch jitter | Reduced but still grows | Artifact elimination |
| Diffusion (ensemble) | Statistical validity maintained | Distribution sampling |
| Latent diffusion + bundling | Slowest growth | Compression + bundling |

No current method can maintain high-fidelity predictions indefinitely — **all models exhibit error accumulation**. The question is how slowly.

---

## Engineering Reality

Current models are far from matching numerical solver accuracy for engineering applications. The NMSE at step 24 for the best models (GP$_{\text{hy}}$T, Poseidon) is still orders of magnitude above what's acceptable for, e.g., aerospace design optimization. Achieving practical engineering accuracy will require:
- Orders of magnitude improvement in accuracy and stability.
- Possibly hybrid approaches combining neural predictors with periodic correction from numerical solvers.
- Better physical conservation enforcement (PC-DeepONet style) in end-to-end models.

---

## See Also

- [[diffusion-models-physics]]
- [[physics-foundation-models]]
- [[navier-stokes-equations]]
- [[poseidon-pde-foundation-model]] — continuous-in-time operator eval avoids rollout; optional autoregressive mode
- [[walrus-paper]]
- [[gphyt-physics-foundation-model]]
- [[latent-diffusion-physics]]
- [[multiscale-diffusion-solar]]
