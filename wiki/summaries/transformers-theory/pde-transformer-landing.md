# Summary: PDE-Transformer (Landing Page / ICML Overview)

**Source:** `raw/PDE-Transformer Efficient and Flexible Network Architecture for Learning PDEs from Data.md`  
**URL:** https://tum-pbs.github.io/pde-transformer/landing.html  
**Authors:** Benjamin Holzschuh, Qiang Liu, Georg Kohl, Nils Thuerey (TU Munich)  
**Venue:** ICML 2025  
**Date Ingested:** 2026-04-11

---

## Overview

This is the project landing page for PDE-Transformer, providing an accessible summary of the architecture and results. It supplements the full paper with visual comparisons and capability tables.

---

## Key Points (Landing Page Highlights)

- **Mixed vs. Separate Channel representation:**
  - **MC (Mixed Channel):** All channels embedded in same token → computationally efficient, less flexible for transfer.
  - **SC (Separate Channel):** Channels embedded independently, interact only via channel-axis attention → better for transfer learning.
- **Model trained on 16 different PDE dynamics** simultaneously, without knowledge of simulation parameters.
- Performance comparison table shows PDE-Transformer is the only model combining multi-scale, scalability, probabilistic outputs, non-square domains, optional periodic BC, and advanced conditioning.

---

## Architecture Capabilities Table

| Architecture | Multi-scale | Scalable | Probabilistic | Non-square Domains | Periodic BC | Advanced Conditioning |
|---|---|---|---|---|---|---|
| FactFormer | ✗ | ✗ | ✗ | ✗ | ✓ optional | ✗ |
| UNet | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| scOT | ✓ | ✓ | ✗ | ✗ | ✓ required | ✗ |
| U-DiT | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| **PDE-Transformer** | ✓ | ✓ | ✓ | ✓ | ✓ optional | ✓ |

---

## Downstream Fine-tuning

Pre-trained PDE-Transformer fine-tuned on The Well datasets (active matter, Rayleigh-Bénard, shear flow) shows consistent improvements over training from scratch, with SC version benefiting most.

---

## Links

- [[pde-transformer-paper]] — full technical paper with all derivations
- [[partial-differential-equations]] — the equations
- [[transformer-architectures]] — architecture context
