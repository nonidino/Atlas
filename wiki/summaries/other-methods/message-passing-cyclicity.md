# Summary: Message Passing and Cyclicity Transition

**Source:** `raw/Message passing and cyclicity transition.md`  
**Author:** Takayuki Hiraoka (Aalto University)  
**arXiv:** 2604.01201v2  
**Date Ingested:** 2026-04-11

---

## Overview

Resolves a long-standing conceptual ambiguity in message passing (belief propagation) for percolation: **the messages identify reachability from cycles, not from the giant component.** This is a precise theoretical result in network science / statistical physics that has implications for the foundations of graph-based ML.

---

## Core Message Passing Equation

The simplest message passing equation on a directed graph $G = (V, E)$:
$$x_{j\to i} = \prod_{k \in \partial_{j\to i}} x_{k\to j}$$
where $\partial_{j\to i} = \partial_j \setminus \{i\}$ (neighbors of $j$ excluding $i$).

For **bond percolation** (each edge retained with probability $q$):
$$x_{j\to i} = \prod_{k\in\partial_{j\to i}} [1-q+q\, x_{k\to j}]$$

For **site percolation** (each node retained with probability $q$):
$$x_{j\to i} = 1-q+q\prod_{k\in\partial_{j\to i}} x_{k\to j}$$

---

## Main Result

**Theorem (informal):** The solution $x_{j\to i}$ of the message passing equations represents the probability that node $j$ is **not reachable from any non-reciprocal cycle** in graph $G$, not (as conventionally assumed) the probability of not belonging to the giant component.

Equivalently, the node marginal:
$$y_i = \prod_{j \in \partial_i} [1-q + q\, x_{j\to i}]$$
converges to:
- $y_i \to 1$ if $i$ is not reachable from any non-reciprocal cycle (component is acyclic).
- $y_i \to 0$ if $i$ is reachable from multiple non-reciprocal cycles (component is multicyclic).
- Non-convergence if $i$ is reachable from exactly one non-reciprocal cycle (component is unicyclic).

---

## Dual Graph Formalism

The author introduces the **dual graph** $M_G$: nodes are messages $x_{j\to i}$; directed edge from $x_{k\to j}$ to $x_{j\to i}$ when $k \in \partial_{j\to i}$. The convergence behavior of messages maps exactly to the cycle structure of $M_G$:
- Root removal in $M_G$ (removing nodes with in-degree 0) identifies trivially-one messages.
- Non-trivial convergence only occurs when $M_G$ has cycles with incoming edges (i.e., reachable from multiple upstream cycles).

---

## Key Distinction

Two structural transitions are generally **distinct**:
1. **Cyclicity transition:** emergence of components with multiple cycles.
2. **Giant component transition:** emergence of an extensive connected component.

These coincide asymptotically for Erdős–Rényi and configuration model graphs — which is why the conflation went unnoticed — but **diverge for geometrically embedded networks** (e.g., random geometric graphs) where many small multicyclic components can exist alongside a large unicyclic component.

---

## Empirical Validation

- Verified on Erdős–Rényi graphs ($n=1000$, $p=0.006$): excellent agreement between $y_i$ and $\hat{p}^\text{A}_i$ (acyclic probability), strong correlation with $\hat{p}^\text{M}_i$ (multicyclic).
- Random geometric graphs: $y_i$ matches cyclicity but not giant-component membership.
- 43 real-world networks (27 undirected, 16 directed): message-passing is consistently closer to cyclicity empirics than giant-component empirics.

---

## Relevance to Physics Foundation Model Goal

**Direct connection:** Message passing is the computational backbone of Graph Neural Networks (GNNs), which are used in mesh-based physics simulations and irregular-grid PDE solvers. Understanding exactly what structural property message passing computes is fundamental to understanding GNN expressivity and failure modes.

**Indirect connection:** Percolation theory has a deep connection to phase transitions in statistical physics. The cyclicity transition (identified here) is a distinct universality class from the giant-component transition. This may have implications for how ML models trained on physical systems near phase transitions behave — e.g., whether they can distinguish the two types of transitions.

**[AI Inference]:** This result implies that standard GNN message passing is fundamentally sensitive to the *cycle structure* of the input graph, not just connectivity. For physics simulations on unstructured meshes (where numerical accuracy depends on local grid topology), this could explain why GNNs struggle near mesh irregularities: they may be implicitly tracking cyclicity rather than physical locality. A physics foundation model using graph-based representations should be designed with this in mind.

---

## Links

- [[message-passing-belief-propagation]] — core concept
- [[partial-differential-equations]] — mesh-based PDE solvers use GNN-like message passing
- [[navier-stokes-nonuniform-grids]] — non-uniform grid CFD (where mesh topology matters)
