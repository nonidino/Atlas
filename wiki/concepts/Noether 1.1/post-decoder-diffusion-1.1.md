# Post-Decoder Diffusion — Generative Gap-Fill (Noether 1.1)

**Type:** Concept — model portion, NEW dedicated page in 1.1 (folder: Noether 1.1)
**Status:** Design. A generative stage that runs **after** the deterministic decode ([[decoder-1.1]]) to fill in the high-wavenumber, chaotic content the deterministic path necessarily averages away. Realizes the diffusion path [[noether-1.0]] §3.9 deferred and [[noether-1.0-rbc]] §7 left unbuilt, and closes the specific gap [[noether-1.0-rbc]] §6.6 diagnosed: the deterministic model reproduces the *spectrum's shape* but loses the fine, transport-carrying structure (plumes), because averaging is the optimal deterministic answer to a chaotic target.
**Related Concepts:** [[00-noether-1.1-overview]], [[decoder-1.1]], [[graph-tokenizer-1.1]], [[iterative-refinement-pfm]], [[autoregressive-rollout-stability]]
**Related Summaries:** [[latent-diffusion-physics]], [[pisd-physics-informed-spectral-diffusion]], [[diffusion-models-physics]], [[pde-transformer-paper]]

---

## What it does — intuitively

A deterministic next-state predictor, trained on a chaotic system with MSE, is *rewarded for blurring* — when many fine-scale futures are equally likely, their average has the lowest error, so the model outputs the mean and the sharp features die. That average is exactly the smoother the RBC build became. The fix is not more deterministic capacity (which §6.6 showed does not help and can hurt); it is to make the fine scales **generative**: predict the resolved, deterministic part as before, then *sample* the unresolved part from a learned conditional distribution instead of averaging it.

Concretely: the deterministic decode produces a smooth next-state; a diffusion model, conditioned on that smooth state (and the tokens), **generates the high-frequency residual** — a physically plausible plume-scale texture consistent with the coarse prediction — rather than leaving a blurred gap. It fills the band the tokenizer's high-frequency path ([[graph-tokenizer-1.1]]) marked as present-but-chaotic.

## What it does — mathematically

Let $\bar v_{t+1}$ be the deterministic decode ([[decoder-1.1]]) and $r = v_{t+1}-\bar v_{t+1}$ the residual (dominant in the near-cutoff band, roughly the same band [[graph-tokenizer-1.1]]'s $h^{\text{hi}}$ path carries on the encode side). Train a conditional diffusion model $p_\theta(r\mid \bar v_{t+1}, \{h_i\}, z_{\text{cond}})$ by denoising:
$$\mathcal L_{\text{diff}} = \mathbb E_{r, \tau, \epsilon}\big\|\epsilon - \epsilon_\theta(r_\tau,\ \tau,\ \bar v_{t+1},\ \{h_i\},\ z_{\text{cond}})\big\|^2,\qquad r_\tau=\sqrt{\bar\alpha_\tau}\,r+\sqrt{1-\bar\alpha_\tau}\,\epsilon.$$

**$\tau$ is a diffusion timestep, unrelated to the physical timestep $t$/$\Delta t$** — an artificial noise-level index $\tau=0,\dots,T_{\text{diff}}$ internal to the generative process, easy to conflate with the trajectory clock but orthogonal to it. $\bar\alpha_\tau$ is a **fixed** (not learned) schedule decreasing from near 1 ($\tau=0$: $r_0=r$, clean) to near 0 ($\tau=T_{\text{diff}}$: $r_\tau\approx\epsilon$, pure noise) — $r_\tau$ is a literal interpolation between the true residual and Gaussian noise. This is how one clean $r$ becomes unlimited training pairs: sample a random $\tau$ and fresh $\epsilon$, corrupt $r$ to $r_\tau$, and train $\epsilon_\theta$ to recover exactly the $\epsilon$ that was injected (an $\epsilon$-prediction target, chosen over predicting $r$ directly because it keeps the training signal comparably scaled across every noise level). The three conditioning arguments each do a distinct job: $\bar v_{t+1}$ anchors generation to the coarse skeleton (add texture *consistent with this prediction*, not an unrelated one); $\{h_i\}$ gives the denoiser the tokenized physics state, so structure is placed where the dynamics support it; $z_{\text{cond}}$ selects which regime's statistics to sample from (turbulent vs. laminar texture) from the one shared head.

**Sampling — the reverse process, run only at inference** (the mechanism behind "sample $r$ at inference," made explicit): start from pure noise and iteratively denoise,
$$r_{T_{\text{diff}}}\sim\mathcal N(0,I);\qquad \text{for }\tau=T_{\text{diff}},\dots,1:\quad \hat\epsilon=\epsilon_\theta(r_\tau,\tau,\bar v_{t+1},\{h_i\},z_{\text{cond}}),\ \ \hat r_0=\frac{r_\tau-\sqrt{1-\bar\alpha_\tau}\,\hat\epsilon}{\sqrt{\bar\alpha_\tau}},$$
$$r_{\tau-1} = (\text{schedule-weighted blend of }\hat r_0\text{ and }r_\tau) + \sigma_\tau z,\ \ z\sim\mathcal N(0,I)\ (\text{omitted at the final step}),$$
terminating at $r_0$, then $v_{t+1}=\bar v_{t+1}+r_0$. Each step re-estimates the clean residual from the current noisy guess, nudges toward it, and injects a little fresh randomness — many small corrections rather than one deterministic leap, which is what produces plausible physical texture instead of a blurred average. **Few-step samplers** (DDIM / consistency-model style) collapse this loop to a handful of steps at some quality cost — the concrete mitigation for the "cost of sampling" risk below.

Design commitments:

- **Residual/latent band only.** The diffusion generates the *high-frequency residual*, not the whole field — because $r$ occupies a narrow band and small dynamic range, both $\epsilon_\theta$ and the number of reverse steps can stay small, versus diffusing the entire state from scratch ([[latent-diffusion-physics]], [[pisd-physics-informed-spectral-diffusion]]). Deterministic-encode / stochastic-predict is the intended split.
- **Runs after conservation, specifically in this order.** Placing it *after* [[discovered-conservation-1.1]] means the generated residual can be re-projected onto discovered constraints. This composition is safe *because* the projection is provably the minimal-disturbance correction under its metric (derived there) — it nudges a completed sample by the smallest edit needed to satisfy a confidently-held invariant, rather than fighting an intermediate step the sampler would otherwise have to work around.
- **Phase-separated training.** The deterministic stack trains first and is frozen; the diffusion head trains on its frozen residuals, per [[noether-1.0]]'s deferred-diffusion plan and [[training-curriculum]]'s phase separation. Training jointly would have the diffusion head chasing a residual defined against a nonstationary, low-quality $\bar v_{t+1}$ early in training — the same reasoning latent-diffusion pipelines use to freeze an autoencoder before training diffusion in its latent space ([[latent-diffusion-physics]]), with the deterministic decoder's residual space playing that role here.
- **Optional at inference.** "Low-temperature" scales down the injected noise $\sigma_\tau z$ at each reverse step (or truncates the loop), trading diversity for closeness to $\bar v_{t+1}$; "skip it" sets $r=0$ outright — useful for a stable reference rollout. For attractor-*statistics* fidelity (spectra, Nu) full sampling is what restores the missing plume energy.

## Physics relevance

- **Chaos needs a distribution, not a point.** Beyond the Lyapunov horizon a specific trajectory is unpredictable but its *statistics* are not; a generative head targets the correct object (the attractor measure) where a deterministic head targets an average that is not itself a physical state — the central lesson of [[latent-diffusion-physics]].
- **Directly addresses the RBC diagnosis.** §6.6 localized the remaining error to heat-transport / thermal-plume representation with the spectrum otherwise matching — precisely the sharp, near-cutoff, chaotic content this stage generates rather than blurs.
- **Rollout stability, honestly.** Regenerating fine structure each step avoids the deterministic high-frequency runaway *and* the opposite failure (progressive blurring to a flat field) — the two ends [[autoregressive-rollout-stability]] describes; the generative fill-in is the principled middle.

## How fundamental constants are included

- **Conditioned, never hard-coded.** The denoiser receives $z_{\text{cond}}$ (dimensionless groups) and the covariant channels, so the generated texture is regime-appropriate — high-$\mathrm{Ra}$ turbulent fine structure vs. low-$\mathrm{Ra}$ laminar — from the *same* head.
- **Cutoff constants set the band.** The wavenumber cutoff $k_{\max}$ (a viscous/Kolmogorov scale built from dimensional constants, [[conditioning-and-constants-1.1]]) defines which band is deterministic vs. generated — the diffusion targets $[k_{\text{det}}, k_{\max}]$, itself a function of the constants.
- **Constraints stay in charge.** Because generation is followed by re-projection onto discovered invariants, no constant-implied conservation law can be broken by sampling.

## Open items / risks
- **Cost of sampling** at every rollout step — mitigated by few-step samplers and the residual-only (small) target.
- **Distribution vs. accuracy metric.** Standard rel-$L^2$ *penalizes* a correct sample (it is not the mean); evaluation must move to attractor statistics (rollout-mean Nu, spectra), as [[noether-1.0-rbc]] §6.6 already began.
- **Interaction with discovered-hard constraints** — a hard-projected field plus a sampled residual must be re-projected, adding a projection per sample; ordering matters (§"runs after conservation").

## See Also
- [[decoder-1.1]] — produces the deterministic mean this corrects
- [[graph-tokenizer-1.1]] — the high-frequency residual path this generates into
- [[discovered-conservation-1.1]] — re-projection that keeps generation admissible
- [[latent-diffusion-physics]] / [[pisd-physics-informed-spectral-diffusion]] — the generative-emulator evidence and spectral-latent approach
