# Message Passing and Belief Propagation

**Type:** Core Concept  
**Related Sources:** Message Passing and Cyclicity Transition

---

## Overview

**Message passing** (also called **belief propagation**) is a framework for probabilistic inference on graphical models. Each node iteratively aggregates and distributes information to its neighbors, enabling global inferences from local computations. It has broad applications in statistical physics, network science, coding theory, and machine learning.

---

## Basic Equations

On a directed graph $G = (V, E)$, the simplest message passing equation is:
$$x_{j\to i} = \prod_{k\in\partial_{j\to i}} x_{k\to j}$$
where $\partial_{j\to i} = \partial_j \setminus \{i\}$ (neighbors of $j$ excluding $i$, to prevent echo).

For **bond percolation** (edge retained with probability $q$):
$$x_{j\to i} = \prod_{k\in\partial_{j\to i}} [1-q+q\, x_{k\to j}]$$

For **site percolation** (node retained with probability $q$):
$$x_{j\to i} = 1-q+q\prod_{k\in\partial_{j\to i}} x_{k\to j}$$

Node marginals:
$$y_i = \prod_{j\in\partial_i} [1-q+q\, x_{j\to i}]$$

---

## Reinterpretation: Cyclicity, Not Giant Component

**Classical interpretation:** $x_{j\to i}$ = probability that node $j$ does **not** belong to the giant component in the absence of node $i$.

**New interpretation (Hiraoka, 2026):** $x_{j\to i}$ = probability that node $j$ is **not reachable from any non-reciprocal cycle** (length > 2) in the graph.

The node marginal $y_i$ tracks whether the component containing $i$ is:
- **Acyclic** ($y_i \to 1$): tree component.
- **Unicyclic** ($y_i$ non-convergent): exactly one cycle.
- **Multicyclic** ($y_i \to 0$): multiple cycles (giant component regime for sparse random graphs).

---

## Two Distinct Phase Transitions

The paper identifies that **two structural transitions** occur in percolation:

1. **Giant component transition:** An extensive (order $N$) connected component emerges.
2. **Cyclicity transition:** A component with multiple cycles ($\geq 2$) emerges.

For **Erdős–Rényi and configuration model graphs**, these transitions coincide asymptotically — which is why the conflation persisted. For **geometrically embedded networks** (e.g., random geometric graphs on a torus), the transitions decouple: many small multicyclic components can co-exist before a giant component forms.

---

## The Dual Graph $M_G$

The dual graph $M_G$ has:
- **Nodes:** Messages $x_{j\to i}$.
- **Directed edges:** $x_{k\to j} \to x_{j\to i}$ when $k \in \partial_{j\to i}$.

Convergence behavior of messages maps to cycle structure of $M_G$:
- **Root removal** eliminates messages that are trivially 1 (in-degree 0 in $M_G$).
- A message converges to **non-trivial value** iff it is on or downstream from a cycle in $M_G$ that has at least one incoming edge from outside the cycle (i.e., reachable from multiple upstream cycles).

---

## Applications

| Domain | Application |
|---|---|
| **Network science** | Percolation, community detection, epidemic spreading |
| **Statistical physics** | Ising/Potts models, spin glasses, random satisfiability |
| **Information theory** | LDPC codes, turbo decoding |
| **Machine learning** | Loopy belief propagation in Bayesian networks, GNN message passing |
| **Optimization** | Survey propagation for constraint satisfaction |

---

## Connection to Graph Neural Networks

**Message passing GNNs** (MPNN) are the computational analog of belief propagation:
$$\mathbf{h}_i^{(l+1)} = \phi\!\left(\mathbf{h}_i^{(l)},\; \bigoplus_{j\in\mathcal{N}(i)} \psi\!\left(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, \mathbf{e}_{ij}\right)\right)$$

The Hiraoka result implies that MPNNs are inherently sensitive to the **cycle structure** of their input graphs. This has implications for:
- Physics simulations on unstructured meshes (where cycle structure reflects mesh quality).
- Social network analysis (where cyclicity correlates with community structure).
- Molecular property prediction (where ring systems are chemically significant).

---

## Limitations of Message Passing

1. **Exact on trees only:** BP is exact for tree-structured graphs; for loopy graphs, it is an approximation.
2. **Non-convergence:** On graphs with exactly one cycle reachable from a message, iteration does not converge.
3. **Confusion of transitions:** In non-treelike graphs (e.g., geometric graphs, dense networks), message passing identifies cyclicity rather than giant component size.
4. **Computational cost:** $O(|E|)$ per iteration, but may require many iterations near phase transitions.

---

## Relevance to PFM

**[AI Inference]:** Understanding what GNN message passing truly computes (cyclicity, not connectivity) is fundamental for designing graph-based physics solvers that need to resolve specific physical phenomena:

1. **Vortex rings in fluid dynamics** are cyclic structures. GNNs may naturally be sensitive to their formation/breakdown.
2. **Shock waves** are locally tree-like (planar wave fronts). GNNs may struggle to resolve shock topology.
3. **Mesh quality** in FEM is often characterized by element aspect ratios and angles — geometric quantities that correlate with local cycle structure.

A PFM using graph-based representations should encode **explicit cycle indicators** or use architectures (like spectral methods) that bypass the message-passing cyclicity limitation.

---

## See Also

- [[partial-differential-equations]]
- [[message-passing-cyclicity]]
- [[navier-stokes-nonuniform-grids]]
