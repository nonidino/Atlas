# Equivariant Quantum Neural Networks for High Energy Physics at the LHC

**Source:** Lazaro R. Diaz Lievano, GSoC 2024 / ML4SCI, Medium post, 2024-07-23  
**File:** `new/Equivariant Quantum Neural Networks for High Energy Physics Analysis at the LHC.md`  
**Related Concepts:** [[equivariant-gnns]], [[physics-foundation-models]]  
**Related Summaries:** [[equiformer-v3]]

---

## Overview

This work presents **Equivariant Quantum Convolutional Neural Networks (EQCNNs)** for image classification in high-energy particle physics, developed as a Google Summer of Code 2024 project under ML4SCI/QMLHEP. The model encodes **roto-reflection symmetry (p4m group: 90° rotations + reflections)** directly into a quantum neural network by constructing equivariant embedding, ansatz, and measurement components. The motivation is two-fold: (1) reduce the number of trainable parameters by leveraging the symmetry of the data, and (2) improve generalization with small datasets — a key challenge in HEP analysis.

**Primary domain:** High-energy physics (LHC) event classification via jet images and particle physics observables. **Less directly relevant** to the PFM goal of continuum mechanics / field simulation than GNS or EquiformerV3.

---

## Key Concepts

### Symmetry and Equivariance
A function $f$ is **equivariant** to a group $G$ if:
$$f(g \cdot x) = g \cdot f(x) \quad \forall g \in G, x \in \mathcal{X}$$

For image classification, $f$ should be **invariant** (the special case where the output transformation is trivial):
$$f(g \cdot x) = f(x)$$

The p4m group has 8 elements: identity, three 90° rotations, and four reflections (x-axis, y-axis, two diagonals).

### Quantum Neural Network (QNN) Structure

A QNN consists of:
1. **Quantum feature map** $\psi: X \to \mathcal{H}$: embeds classical data $x$ into Hilbert space $|{\psi(x)}\rangle$
2. **Variational quantum circuit** $U(\theta)$: parameterized by angles $\theta$, optimized classically
3. **Measurement**: expectation value of observable $O$:

$$\hat{y}(x; \theta) = \langle \psi(x) | U^\dagger(\theta)\, O\, U(\theta) | \psi(x) \rangle$$

### Equivariant Quantum Embedding (Coordinate-Aware Amplitude, CAA)
For a 2D image, the first $n$ qubits encode the x-coordinate and the second $n$ qubits encode the y-coordinate. Reflection operators $V_x$, $V_y$ and rotation operator $V_r$ are induced representations of p4m acting on this qubit register.

### Equivariant Ansatz via Twirling
Given symmetry group $G$ with representation $V_s$, a gate $R$ is G-equivariant iff:
$$R_G \cdot V_s[g] = V_s[g] \cdot R_G \quad \forall g \in G$$

The **Twirling Method** projects any gate $U$ onto the G-equivariant subspace:
$$R_G = \frac{1}{|G|} \sum_{g \in G} V_s[g]^\dagger U V_s[g]$$

For p4m acting on the CAA embedding, the equivariant gateset is $\{Y_1 Y_2,\ Z_1 Z_2,\ X_1,\ X_2\}$.

### Invariant Measurement
Observable $O$ is G-invariant iff $[O, V_s[g]] = 0\ \forall g$. Measurement uses $\langle Z_1 Z_2 \rangle$ (product of Pauli-Z on each qubit) as the invariant observable.

---

## Key Results

Equivariant QCNN (EQCNN) vs. generic QCNN (GQCNN) benchmarks on standard HEP image classification datasets:
- EQCNN achieves comparable or better accuracy with fewer trainable parameters
- EQCNN benefits of: reduced barren plateau risk, better generalization from small datasets, symmetry-consistent predictions
- Evaluated at the NISQ (noisy intermediate-scale quantum) regime — limited qubit counts

---

## Relevance to PFM Goal

**Indirect relevance.** This work is primarily about quantum computing applied to particle physics image classification — a domain (HEP jet tagging) that is separate from the continuum mechanics / PDE-solving focus of the PFM goal.

However, two conceptual contributions carry over:

1. **Noether's theorem as inductive bias:** Symmetry → conservation law → constrained architecture. This is the same design philosophy driving Dynami-CAL GraphNet, VarMiON, and PC-DeepONet. The EQNN demonstrates this principle in a quantum circuit formalism.

2. **Equivariant design methodology:** The three-step EQNN construction (equivariant embedding → equivariant ansatz via twirling → invariant measurement) is a general recipe for any group $G$. In principle, this recipe could be applied to physical symmetry groups relevant to PFM: SO(3) (3D rotations), Poincaré group (special relativity), or gauge symmetries.

**Quantum ML for physics:** The quantum advantage claim for QNNs on NISQ devices remains unproven. Classical equivariant GNNs (EquiformerV3, Dynami-CAL) are the dominant paradigm for symmetry-constrained physics ML at present.

---

## **[AI Inference]**

**[AI Inference]:** The Twirling Method for constructing equivariant quantum gates is structurally equivalent to the **Reynolds operator** in classical representation theory. The classical analog for constructing equivariant neural network layers (e.g., group-equivariant convolutions) uses the same symmetrization over the group orbit. This suggests that any equivariant architecture design technique from classical deep learning — Clebsch-Gordan products, harmonic analysis, representation decomposition — can, in principle, be "quantized" by replacing classical feature vectors with quantum states and classical operations with unitary gates. The open question is whether this yields any computational advantage.

**[AI Inference]:** For a PFM that needs to model fundamental particle physics (rather than continuum mechanics), gauge symmetry (local SU(3) × SU(2) × U(1) in the Standard Model) is the dominant symmetry, not SE(3). The EQNN work, combined with the EquiformerV3 SE(3) framework, suggests a path toward gauge-equivariant neural networks as a future direction for physics foundation models beyond classical mechanics.

---

## See Also

- [[equivariant-gnns]] — classical equivariant GNNs for physics
- [[equiformer-v3]] — SE(3)-equivariant Transformer for atomistic physics
- [[dynami-cal-graphnet]] — symmetry + conservation in classical mechanics GNNs
- [[physics-foundation-models]] — broader PFM landscape
