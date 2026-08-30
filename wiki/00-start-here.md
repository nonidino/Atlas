# Start Here — Physics Foundation Model Wiki

> **What this is:** an LLM-maintained research wiki accumulating everything needed to build a physics foundation model (PFM) — "what LLMs are for language, but for physics." This is **preliminary research**. Framework/implementation pages will be added later as their own files.
>
> **Code home:** 🔗 **[github.com/nonidino/physics-foundation-model](https://github.com/nonidino/physics-foundation-model)** — the repo where the actual model will be built. See [[code-and-papers]] for this + reference implementations of every paper below.

---

## The two anchors

- **[[index]]** — full catalog of every page, grouped by theme. Read this to find a specific page.
- **[[log]]** — chronological build log (what was ingested/synthesized and when).

If you only open one page to navigate, open [[index]]. This page (`00-start-here`) is the *conceptual* on-ramp; the index is the *exhaustive* map.

---

## Reading path (goes from "what/why" → "how" → "what we'd build")

Follow this order to build understanding from the ground up. Each bullet links the page to read.

### 1. The goal — what a PFM even is
- [[physics-foundation-models]] — the central goal, challenges, current landscape
- [[pfm-concept-overview]] — what a PFM *is*, its 7 target capabilities, and the 8 gaps blocking it
- [[pfm-interface-design]] — how you'd actually talk to a PFM (without text): timescales, regimes, field codes

### 2. The physics being modeled
- [[partial-differential-equations]] — the PDE taxonomy (NS, Euler, MHD, Stokes, Burgers)
- [[navier-stokes-equations]] — the flagship hard problem; turbulence, Reynolds number
- [[message-passing-belief-propagation]] — statistical-physics / graph view

### 3. The ML building blocks
Start with the representation question, then the model families:
- [[multimodal-tokenization]] — how physical fields become tokens
- [[neural-operators]] · [[neural-surrogates]] — operator learning & surrogate taxonomy
- [[transformer-architectures]] — ViT/DiT/axial attention + the deep theory (PDE-discretization, token clustering)
- [[diffusion-models-physics]] — why generative > deterministic for chaotic systems
- [[equivariant-gnns]] — symmetry- and conservation-aware graph networks
- [[gaussian-processes]] · [[mixture-of-experts]] · [[memory-augmented-physics-models]] — supporting machinery
- [[in-context-learning-physics]] · [[world-models-physics-ai]] · [[autoregressive-rollout-stability]] · [[transfer-learning-fine-tuning]] — how a single model generalizes & stays stable

### 4. How to combine them — the paradigm map
- [[pfm-architecture-approaches]] — **the key comparison**: 7+ paradigms with pros/cons and a hybrid roadmap
- [[possible-architectures]] — the 5-way benchmark plan (the concrete next experiment)

### 5. The candidate architectures (implementation-ready specs)
Five detailed specs designed to be buildable on the Burgers' benchmark:
- [[arch-autoregressive-transformer]] · [[arch-diffusion-backbone]] · [[arch-neural-differentiator]] · [[arch-physics-mamba]] · [[arch-gnn-physics-bottleneck]]

### 6. Where it goes next — extension proposals
Rigorously-grounded ideas beyond the benchmark five:
- [[multiscale-hierarchical-gnn]] · [[hamiltonian-message-passing]] · [[action-based-noether-enforcement]] · [[antisymmetric-signed-attention-transformer]]

---

## The one framework to keep in your head: the Physics Encoding Spectrum

Every approach in this wiki sits somewhere on this axis (weak → strong physics enforcement):

1. **Data-driven only** — Walrus, GP$_{\text{hy}}$T backbone
2. **Soft loss constraints** — PI-DeepONet
3. **Test-time physics guidance** — PISD (DPS), Noether Networks tailoring
4. **Architectural soft biases** — VarMiON, GNN-PB energy bottleneck, multipole hierarchy, A-DGN
5. **Hard architectural constraints** — PC-DeepONet (divergence-free), Dynami-CAL (exact momentum), HNN (exact energy), LNN (Noether-derived)

The grand question of this wiki: *how far up this spectrum can you go without losing the generality that makes a foundation model a foundation model?*

---

## How the wiki is laid out (folders)

```
wiki/
  00-start-here.md          ← you are here (conceptual on-ramp)
  index.md                  ← exhaustive catalog
  log.md                    ← chronological build log
  resources/
    code-and-papers.md      ← GitHub repo + per-paper code/arXiv links
  summaries/                ← one page per source paper, grouped by theme
    foundation-models/  transformers-theory/  neural-operators/
    graph-networks/  hamiltonian-lagrangian/  diffusion-generative/
    solvers-and-simulation/  other-methods/
  concepts/                 ← synthesized topic pages, grouped by theme
    00-pfm-core/  physics-foundations/  ml-building-blocks/
    benchmark-architectures/  extension-architectures/
```

**Links are folder-independent.** All page links use bare `[[page-name]]` form (Obsidian resolves by filename), so pages can be moved between folders freely without breaking anything.

---

*Conventions, AI-inference marking, and maintenance workflow live in [[index]] and the root `CLAUDE.md`.*
