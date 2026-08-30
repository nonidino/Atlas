# Summary: Predicting Partially Observable Dynamical Systems via Diffusion Models with a Multiscale Inference Scheme

**Source:** `raw/Predicting partially observable dynamical systems via diffusion models with a multiscale inference scheme.md`  
**Authors:** Rudy Morel et al. — The Polymathic AI Collaboration  
**arXiv:** 2511.19390v1  
**Date Ingested:** 2026-04-11

---

## Overview

Addresses **probabilistic prediction of partially observable, long-memory dynamical systems** using conditional diffusion models with a novel **multiscale inference scheme**. Motivated by solar physics — where the Sun's evolution is driven by unobservable internal processes — but applicable to any partially observable chaotic system. The core contribution is a multiscale temporal conditioning strategy that enables capturing long-range dependencies without increased computational cost.

---

## Problem Setting

At time $t=0$, generate future trajectories $\mathbf{x}_{1:T}$ given past observations $\mathbf{x}_{t\leq 0}$:
$$p(\mathbf{x}_{1:T} \mid \mathbf{x}_{t\leq 0})$$
For a fixed-length model generating $K$ new steps per call with $K+1$ conditioning steps, generating a trajectory of length $T$ requires at least $\lceil T/K\rceil$ applications:
$$p(\mathbf{x}_{1:T}\mid\mathbf{x}_{t\leq 0}) \approx \prod_{n=1}^N p(\mathbf{x}_{I_n}\mid\mathbf{x}_{C_n})$$

---

## Standard Autoregressive Failure Mode

Standard autoregressive rollout conditions only on the **most recent** frames, discarding distant past information. For systems with long memory (e.g., solar dynamics), this leads to:
- **Distribution bias:** predictions drift from the true distribution.
- **Rollout instability:** errors compound rapidly without long-range corrections.

---

## Multiscale Inference Scheme

**Key idea:** Generate a future time step $t=9$ in a single diffusion model call while conditioning on both **fine-grained recent frames** and **coarse-grained distant past frames** — the "multiscale template".

The multiscale templates create a pyramid structure in time:
- Distant past: coarse resolution (every $2^k$ steps).
- Recent past: fine resolution (every step).
- Each new generation call always has access to a portion of the full temporal history.

This avoids the distribution error accumulation of pure autoregressive schemes because **distant future frames are generated in one step from past observations**, not iteratively propagated through many intermediate predictions.

---

## Solar Dataset

First multi-modal $8.5$ TB dataset of $512\times512$ solar region videos:
- 12 fields: magnetic vector field + solar atmosphere measurements.
- Multiple modalities (different instruments: HMI, AIA).
- $2\times$ spatial downsampling (matching optical resolution); 1h temporal sampling.
- For $\sim 1000$ solar active regions over years of SDO observations.

---

## Key Results

- Multiscale inference **significantly reduces prediction bias** vs. standard autoregressive rollout.
- **Improves rollout stability**: generates coherent long-horizon trajectories where standard methods diverge.
- First multi-modal diffusion model trained to predict high-resolution solar videos.
- Conditions more frequently on a larger portion of past data compared to competing inference schemes.

---

## Relevance to Physics Foundation Model Goal

This work represents a key contribution to the problem of **partially observable physical systems** — a regime that a universal PFM must handle. Most physical systems in practice have unobservable latent variables (internal temperatures, subsurface flows, chemical composition). The multiscale inference framework is a general tool for incorporating long-range temporal context into any fixed-length diffusion model.

The solar dataset itself is a valuable benchmark for testing a PFM on a real-world partially observable system with genuine physical complexity.

**[AI Inference]:** The multiscale template strategy is analogous to **hierarchical positional encoding** in transformers — representing information at multiple temporal resolutions simultaneously. This could be directly incorporated into a transformer-based PFM as a learnable multiscale temporal attention bias. There may also be a connection to wavelet representations of physical fields, where coarse-scale structure governs long-range behavior and fine-scale structure governs local dynamics.

---

## Links

- [[latent-diffusion-physics]] — closely related work from Polymathic AI (latent diffusion for physics)
- [[walrus-paper]] — same group, different (non-generative) approach
- [[diffusion-models-physics]] — core methodology
- [[autoregressive-rollout-stability]] — the core problem being solved
- [[physics-foundation-models]] — broader context
