# EquiformerV3: Scaling SE(3)-Equivariant Graph Attention Transformers

**Source:** Liao et al., arXiv 2604.09130v1, ICML  
**File:** `new/EquiformerV3...md`  
**Related Concepts:** [[equivariant-gnns]], [[transformer-architectures]], [[pfm-architecture-approaches]]  
**Related Summaries:** [[dynami-cal-graphnet]], [[gns-graph-network-simulators]]

---

## Overview

EquiformerV3 is the third generation of the SE(3)-equivariant graph attention Transformer (Equiformer family), designed for 3D atomistic modeling — predicting energy and forces from atomic positions. The primary domain is **molecular/materials science** (catalyst design, molecular dynamics, materials discovery), not continuum PDEs, but the architectural ideas are broadly relevant to any SE(3)-symmetric physical system.

Three advances over EquiformerV2:
1. **1.75× implementation speedup** via fused operations and `torch.compile`
2. **SwiGLU-S² activations** for many-body interactions and stricter equivariance
3. **Smooth radius cutoff attention** for continuous, differentiable potential energy surfaces

SOTA on OC20, OMat24, and Matbench Discovery benchmarks. On OMat24, achieves comparable accuracy to UMA-L while being 23× smaller.

---

## Key Equations

### SE(3) Equivariance via Irreps

Features are represented as irreducible representations (irreps) of SO(3). A type-$L$ vector $f^{(L)} \in \mathbb{R}^{(2L+1)}$ transforms under rotation $R$ via Wigner-$D$ matrices:
$$f^{(L)} \mapsto D^{(L)}(R) f^{(L)}$$

Tensor products couple type-$L_1$ and type-$L_2$ features to produce type-$L_3$ output:
$$h^{(L_3)}_{m_3} = \sum_{m_1, m_2} C^{(L_3, m_3)}_{(L_1, m_1)(L_2, m_2)} f^{(L_1)}_{m_1} g^{(L_2)}_{m_2}$$

where $C$ are Clebsch-Gordan coefficients and $|L_1 - L_2| \leq L_3 \leq L_1 + L_2$.

### Equivariant Merged Layer Normalization
Shared RMS normalization across all degrees (key improvement over EquiformerV2's separable normalization):
$$\sigma = \sqrt{\frac{1}{L_{\max}+1} \sum_{L=0}^{L_{\max}} \left(\sigma^{(L)}\right)^2}, \quad y^{(L)} = \gamma^{(L)} \circ \frac{x^{(L)}}{\sigma} \quad (L > 0)$$

This preserves relative magnitudes between degrees, which separable normalization destroyed.

### SwiGLU-S² Activation
Projects features onto the unit sphere $S^2$, applies SwiGLU (nonlinearity + gating), then projects back. The projection-multiplication is a fast tensor product via elementwise multiplication in grid space:
$$x^{\text{grid}}(\phi, \theta) = \sum_{L,m} Y^{(L)}_m(\phi, \theta) \cdot x^{(L)}_m$$
$$z^{\text{grid}} = x^{\text{grid}} \odot y^{\text{grid}} \quad \Rightarrow \quad z = x \otimes y$$

Reduces tensor product complexity from $\mathcal{O}(L_{\max}^6)$ to $\mathcal{O}(L_{\max}^4)$.

### Smooth Radius Cutoff Attention
Envelope function applied both to messages and within the softmax to guarantee smooth potential energy surfaces:
$$a_{ij} = \frac{\text{env}(\|\vec{r}_{ij}\|) \cdot \exp(z_{ij})}{\sum_{k \in \mathcal{N}(i)} \text{env}(\|\vec{r}_{ik}\|) \cdot \exp(z_{ik})}$$
$$m_{ij} = a_{ij} \times \left(\text{env}(\|\vec{r}_{ij}\|) \cdot v_{ij}\right)$$

Without the envelope in the denominator, atoms entering/leaving the cutoff radius cause discontinuities in $a_{ij}$ — breaking energy conservation in molecular dynamics simulations.

---

## Key Results

| Benchmark | Metric | EquiformerV3 vs. EquiformerV2 |
|---|---|---|
| OC20 S2EF-2M | Training efficiency | **5.9× speedup** |
| OMat24 | Force MAE | Comparable to EquiformerV2 + UMA-L at **23× smaller** model |
| Matbench Discovery (thermal conductivity) | Task accuracy | **18–31% improvement** vs. eSEN |
| Combined Performance Score (CPS) | All metrics | Best results with **22.6× less training time** vs. UMA-M-1.1 |

---

## Relevance to PFM Goal

EquiformerV3 is primarily a molecular/materials science model, not a continuum PDE model. Its direct relevance to the PFM goal is **architectural** rather than empirical:

1. **SE(3) equivariance at scale:** EquiformerV3 demonstrates that SE(3)-equivariant models can scale efficiently to large datasets (OC20 has 460M training examples). The bottleneck is tensor product cost — the eSCN decomposition ($O(L_{\max}^4)$ vs. $O(L_{\max}^6)$) is a critical algorithmic advance.

2. **Smooth PES as a physics constraint:** The smooth radius cutoff ensures energy-conservative dynamics. This is the molecular analog of requiring PDE solutions to satisfy conservation laws — the model's outputs must respect physical consistency, not just minimize training error.

3. **Transferable design patterns:** Equivariant merged layer normalization and SwiGLU-S² activations are architecture-level improvements independent of domain. Any GNN for SE(3)-symmetric physics (including continuum GNNs for Dynami-CAL-style particle systems) could benefit from these.

**[AI Inference]:** The bridge from EquiformerV3 (atomistic) to continuum PDE models is the **coarse-graining** problem. EquiformerV3 operates at the Å scale with explicit atomic positions; a PFM for continuum mechanics operates at the mm–m scale with field variables on meshes or particles. The two regimes require different inductive biases, but EquiformerV3's efficient equivariant architecture may be a blueprint for SE(3)-equivariant continuum field models — particularly for structural mechanics and molecular simulation at the nano–micro scale interface.

---

## **[AI Inference]**

**[AI Inference]:** EquiformerV3's smooth radius cutoff requirement reveals a **general principle**: any learned simulator used for energy-conserving dynamics must produce outputs that vary smoothly with inputs. This is stronger than the usual "train to minimize MSE" objective — it requires the learned function to lie in a smooth function class. This is analogous to the Sobolev regularity requirement in PISD ([[pisd-physics-informed-spectral-diffusion]]): both impose regularity constraints beyond pointwise accuracy, and both arise from the requirement that the model integrates well in a dynamical system.

**[AI Inference]:** SwiGLU-S² activations incorporate many-body interactions (tensor products via sphere projection) at the cost of $O(L_{\max}^4)$ complexity. For a continuum PFM that needs to handle both short-range contact forces and long-range pressure fields, a hybrid architecture with S²-activation GNN layers for local interactions and global attention for long-range coupling could be near-optimal.

---

## See Also

- [[equivariant-gnns]] — broader context on SE(3)-equivariant GNNs
- [[transformer-architectures]] — Transformer adaptation to equivariant settings
- [[dynami-cal-graphnet]] — equivariant GNN for continuum mechanics (vs. atomistic)
- [[pfm-architecture-approaches]] — where equivariant GNNs fit in the PFM landscape
