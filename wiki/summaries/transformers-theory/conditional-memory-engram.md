# Conditional Memory via Scalable Lookup: Engram for Sparse Transformers

**Source:** Cheng et al., arXiv 2601.07372, January 2026  
**File:** `new/Conditional Memory via Scalable Lookup...md`  
**Related Concepts:** [[mixture-of-experts]], [[transformer-architectures]], [[neural-surrogates]], [[pfm-architecture-approaches]]  
**Related Summaries:** [[walrus-paper]], [[gphyt-physics-foundation-model]], [[deep-memory-dissipative]]

---

## Overview

This paper introduces **Engram**, a module that pairs dynamic neural computation (e.g., Mixture of Experts) with static **conditional memory** via O(1) lookup. Rather than computing all token features end-to-end, Engram retrieves pre-computed n-gram embeddings when applicable, freeing the backbone to focus on complex reasoning. The key contribution is identifying a **U-shaped scaling law** for sparsity allocation: the optimal trade-off between memory parameters (static lookup table) and compute parameters (MoE) exhibits a characteristic shape that guides architecture design.

**Domain:** Large language models (MMLU, BBH, HumanEval, MATH, long-context retrieval). **Relevance to PFM:** A potential blueprint for hybrid static-dynamic memory in physics models.

---

## Core Mechanism

### Engram: Modernized N-gram Embedding

A static lookup table mapping n-gram context to dense embeddings, accessed in O(1) time:
$$\text{Engram}(c_1, \ldots, c_n) = \mathbf{e}_{c_1 \cdots c_n} \quad \text{(table lookup, no computation)}$$

For sequences where n-gram context suffices (high-frequency patterns, local dependencies), Engram directly returns a pre-computed vector. The backbone transformer can then skip redundant computation on these "static reconstruction" tasks.

### Sparsity Allocation Problem

The total model is:
$$\hat{y}_i = \text{Router}(x_i) \begin{cases} \text{Engram}(c_i) & \text{(memory path)} \\ \text{MoE}_j(x_i) & \text{(compute path)} \end{cases}$$

**Scaling law:** Model size $N = N_{\text{mem}} + N_{\text{compute}}$ is fixed; the question is how to split between:
- Static memory (Engram parameters): lookup tables, no FLOPs per token
- Neural compute (MoE parameters): sparse experts, conditional FLOPs

The empirical finding is a **U-shaped curve**: allocating too little memory forces the backbone to recompute frequent patterns (inefficient); allocating too much memory leaves compute capacity idle. The minimum lies at an optimal balance — Engram uses 27B parameters, achieving better results than a 27B MoE-only baseline at the same total FLOPs.

### Mechanistic Benefits

1. **Early-layer relief:** Engram absorbs static reconstruction tasks (local patterns, frequent tokens), allowing early layers to skip these and preserve capacity for deeper reasoning
2. **Attention reallocation:** By delegating local dependencies to lookup, attention heads are freed to track global context over longer distances
3. **Infrastructure efficiency:** Deterministic addressing enables OS-level prefetching from host memory; negligible runtime overhead

---

## Key Results

| Benchmark | Improvement (Engram vs. MoE-only) |
|---|---|
| MMLU | +3.4% |
| CMMLU | +4.0% |
| BBH (general reasoning) | +5.0% |
| ARC-Challenge | +3.7% |
| HumanEval | +3.0% |
| MATH | +2.4% |
| Multi-Query NIAH (long-context) | 84.2 → 97.0 |

---

## Relevance to Physics Foundation Models

**[AI Inference]:** The Engram mechanism suggests an architectural principle for physics models: **bifurcate computation into static memory (known physics) and dynamic learning (unknown interactions)**.

In a physics PFM context, this could manifest as:

1. **Static physics module (lookup):** Pre-computed basis functions, Green's functions, analytical solutions to canonical PDEs (Poisson, heat, Burgers at standard parameters). Retrieve via deterministic addressing based on problem parameters.
   
2. **Dynamic neural module (MoE):** Learn corrections, heterogeneous interactions, and parameters outside the canonical regime. Use sparse experts for different material types, boundary conditions, or equation families.

3. **Sparsity allocation law for physics:** Analogous to Engram's U-shaped curve, there should be an optimal split between:
   - **Memory cost:** Storage of precomputed operators and basis functions ($\sim O(L^2)$ for $L$ basis functions)
   - **Compute cost:** Neural approximation of corrections and non-linear couplings ($\sim$ expert parameters)

The paper's mechanistic insight — that delegating routine tasks (local patterns in LLMs) to memory frees global reasoning capacity — maps to physics as: **delegating canonical/linear terms to static lookup frees the neural backbone to model singularities, shocks, and multi-scale coupling**.

### Comparison to Existing PFM Paradigms

| Paradigm | Memory | Compute | Sparsity Axis |
|---|---|---|---|
| Pure neural (Walrus, GNS) | None | All learned | None; dense computation |
| Neural operators (FNO, DeepONet) | Operator family (fixed) | Learned coefficients | Family selection + coefficient learning |
| **Engram-style hybrid** | Pre-computed operators (static) | MoE experts (sparse) | Both memory (lookup table size) and compute (expert routing) |
| Hard constraints (PC-DeepONet) | Constraint matrix (fixed) | Residual learning | Binary: exact constraint vs. learned residual |

Engram offers a **continuous dial** between memory and compute, rather than discrete choices. The U-shaped scaling law provides guidance on where to set it.

---

## **[AI Inference]**

**[AI Inference]:** The paper's finding that delegating local dependencies to static lookup allows better long-context attention (NIAH 84.2 → 97.0) suggests a **physics analog for multi-scale coupling**: if a PFM delegates short-range local interactions (contact, surface tension) to a pre-computed kernel matrix (memory), the transformer's attention can focus on long-range effects (pressure waves, gravitational coupling). This is particularly relevant for hierarchical PFM designs that must operate across scale ranges 10⁻⁶ m (molecules) to 10⁶ m (astrophysics).

**[AI Inference]:** Engram's O(1) lookup is infrastructure-efficient because addressing is deterministic (no branching, prefetchable). In a physics PFM, deterministic memory access to pre-computed operators could similarly enable GPU/TPU prefetching and lower latency, making real-time physics simulation feasible for applications like robotics control or surgical planning.

**[AI Inference]:** The router mechanism in Engram (deciding memory vs. compute path per token) is structurally equivalent to the sparse expert routing in MoE but operates on a different axis. For physics, an analogous router could decide per-region (or per-patch) whether to use static lookup (canonical regime) or dynamic neural computation (heterogeneous/nonlinear regime). This "problem-adaptive routing" could significantly improve efficiency on mixed-physics problems.

---

## See Also

- [[mixture-of-experts]] — conditional computation via expert routing; Engram adds a second axis (memory vs. compute)
- [[transformer-architectures]] — backbone architecture modified by Engram insertion
- [[walrus-paper]] — pure learned physics transformer; contrast with memory-augmented approach
- [[deep-memory-dissipative]] — cognitive memory systems; potential connection to physics-specific memory design
- [[pfm-architecture-approaches]] — where memory-augmented physics transformers fit in the landscape
