# Architecture: Graph Neural Network with Physics Bottleneck (GNN-PB)

**Type:** Architecture Specification  
**Status:** New Proposal (now better-supported by GNS + Dynami-CAL evidence)  
**Date:** 2026-05-14  
**Related Concepts:** [[equivariant-gnns]], [[message-passing-belief-propagation]], [[physics-foundation-models]], [[mixture-of-experts]], [[possible-architectures]]  
**Related Summaries:** [[gns-graph-network-simulators]], [[dynami-cal-graphnet]], [[equiformer-v3]], [[pc-deeponet-cfd]]

---

## Conceptual Overview

All three preceding architectures (AR Transformer, Diffusion Backbone, Neural Differentiator) operate on **regular grids**: fixed $(H \times W)$ arrays of field values. This is sufficient for academic benchmarks on periodic domains, but fails for the most physically important real-world geometries — turbines, blood vessels, aircraft surfaces, fracture networks — which are irregular and boundary-dominated.

GNN-PB addresses this gap. It operates on a **particle graph** where each node is a fluid parcel, mesh point, or physical particle, and edges encode proximity or interaction. The graph adapts to the geometry naturally — resolution can be concentrated near boundaries, shocks, or regions of interest without any fixed-grid overhead.

The novel element beyond existing GNN simulators (GNS, Dynami-CAL): a **physics bottleneck layer** that routes information through a low-dimensional subspace spanned by conserved physical quantities before expanding back to the full hidden dimension. The bottleneck does not just penalize conservation violations (like a loss term) — it forces the information flow through physically meaningful coordinates. Combined with Dynami-CAL's exact momentum conservation, this gives a model with both **hard conservation** (linear and angular momentum, by construction) and **soft conservation** (energy, through the bottleneck).

**Benchmark hypothesis:** GNN-PB should outperform transformer baselines on irregular-geometry Burgers' (non-uniform grid) while underperforming on the standard uniform-grid benchmark due to overhead from graph construction.

---

## Internal Architecture

The GNN-PB follows the GNS encode-process-decode pipeline, extended with:
1. **Learned adaptive graph construction** — nodes and edges from particle positions
2. **Antisymmetric edge-local message passing** — exact momentum conservation (Dynami-CAL)
3. **Physics bottleneck layer** — soft energy routing through conserved-quantity subspace
4. **Ghost-node boundary treatment** — unified body-body and body-wall interactions
5. **Autoregressive rollout via Euler/RK4 integrator**

---

### Stage 1: Graph Construction

**Nodes:**  
Each node $i$ represents a physical particle or mesh point with state:

$$\mathbf{x}_i = [\mathbf{p}_i,\ \dot{\mathbf{p}}_i^{t-C+1:t},\ m_i,\ \rho_i,\ \text{material\_type}_i] \in \mathbb{R}^{d_\text{node}}$$

- $\mathbf{p}_i \in \mathbb{R}^{d_\text{spatial}}$: current position
- $\dot{\mathbf{p}}_i^{t-C+1:t}$: last $C = 5$ velocity snapshots (concatenated)
- $m_i$: mass
- $\rho_i$: local density (estimated from neighbor count within radius $R$)
- material\_type: one-hot encoding of material (fluid, solid, boundary, etc.)

**Edges — radius graph:**  
Connect nodes $i$ and $j$ if $\|\mathbf{p}_i - \mathbf{p}_j\| \leq R$, where $R$ is a fixed connectivity radius. Edge features:

$$\mathbf{r}_{ij} = [\mathbf{p}_i - \mathbf{p}_j,\ \|\mathbf{p}_i - \mathbf{p}_j\|] \in \mathbb{R}^{d_\text{spatial}+1}$$

(Relative displacement vector and its magnitude — translation-invariant by construction.)

**Adaptive resolution:**  
For continuum simulations, nodes are distributed non-uniformly: denser near boundaries, shocks, or regions of high gradient. The graph automatically allocates more message-passing capacity to these regions because they have more edges (higher local connectivity). This adaptive resolution is a structural advantage over fixed-grid methods.

**Ghost nodes for boundary conditions:**  
For each physical node $i$ within distance $R_\text{wall}$ of a wall, reflect $i$ across the wall's outward normal to create a ghost node $i'$:

$$\mathbf{p}_{i'} = \mathbf{p}_i - 2(\mathbf{p}_i \cdot \hat{n} - d_\text{wall})\hat{n}$$

Ghost nodes have the same properties as their parent (same mass, material type) but zero velocity (for stationary walls). The edge $(i, i')$ is treated identically to any body-body edge — no special-case code needed. This elegant construction (from Dynami-CAL) makes boundary interactions structurally identical to particle interactions, unifying the message-passing scheme.

---

### Stage 2: Encoder (Node and Edge Embedding)

**Node encoder:**  
$$\mathbf{v}_i^0 = \text{MLP}_v(\mathbf{x}_i) \in \mathbb{R}^{d_\text{hidden}}$$

**Edge encoder:**  
$$\mathbf{e}_{ij}^0 = \text{MLP}_e(\mathbf{r}_{ij}) \in \mathbb{R}^{d_\text{hidden}}$$

Both MLPs are 2-layer, with layer norm and SiLU activation. Latent dimension $d_\text{hidden} = 128$ (from GNS).

---

### Stage 3: Processor — Conservation-Constrained Message Passing

This is the core component. The processor runs $M$ rounds of message passing, with each round alternating between:
1. **Antisymmetric edge update** — enforces momentum conservation
2. **Physics bottleneck node update** — routes through conserved-quantity subspace

#### 3a. Antisymmetric Edge-Local Reference Frame (Dynami-CAL)

For each edge $(i, j)$, construct an orthonormal basis $\{\hat{e}_1, \hat{e}_2, \hat{e}_3\}_{ij}$ with the antisymmetry property:

$$\hat{e}_k^{ij} = -\hat{e}_k^{ji} \quad \forall k$$

**Construction:**  
Using the relative position $\delta\mathbf{p}_{ij} = \mathbf{p}_i - \mathbf{p}_j$:

$$\hat{e}_1^{ij} = \frac{\delta\mathbf{p}_{ij}}{\|\delta\mathbf{p}_{ij}\|}$$

The second and third basis vectors $\hat{e}_2, \hat{e}_3$ are constructed via Gram-Schmidt from the cross-product with a reference axis, ensuring $\hat{e}_k^{ij} = -\hat{e}_k^{ji}$ (since swapping $i \leftrightarrow j$ negates $\delta\mathbf{p}_{ij}$).

**Symmetric edge embedding:**  
The scalar edge message $m_{ij}$ must be symmetric under node interchange:

$$m_{ij} = \text{MLP}_m\!\left(\mathbf{v}_i^\ell + \mathbf{v}_j^\ell,\ \|\delta\mathbf{p}_{ij}\|,\ \mathbf{e}_{ij}^\ell\right) \in \mathbb{R}^3$$

The symmetric functions $(+$ for node features, scalar for distance) ensure $m_{ij} = m_{ji}$.

**Force vectorization:**  
$$\vec{F}_{ij} = \sum_{k=1}^{3} m_{ij,k} \cdot \hat{e}_k^{ij}$$

Because $m_{ij,k} = m_{ji,k}$ (symmetric) and $\hat{e}_k^{ij} = -\hat{e}_k^{ji}$ (antisymmetric):

$$\vec{F}_{ij} = -\vec{F}_{ji} \quad \Rightarrow \quad \text{linear momentum conserved exactly at every edge}$$

**Angular momentum conservation:**  
Total edge angular momentum exchange is split into orbital and spin components:

$$\vec{A}_{ij} = \text{MLP}_A(m_{ij}) \in \mathbb{R}^3, \quad \vec{A}_{ij} = -\vec{A}_{ji}$$

Spin torque on node $i$:
$$\boldsymbol{\tau}_{ij} = \vec{A}_{ij} - (\mathbf{p}_i - \mathbf{x}_{0,ij}) \times \vec{F}_{ij}$$

where $\mathbf{x}_{0,ij} = \mathbf{x}_{0,ji}$ is a predicted symmetric force-application point. This ensures global angular momentum is conserved: $\sum_i \boldsymbol{\tau}_{\text{total},i} = 0$.

**Edge memory (temporal tracking):**  
The edge embedding is updated incorporating the previous-step embedding:

$$\mathbf{e}_{ij}^{\ell+1} = \phi_e\!\left(\mathbf{v}_i^\ell, \mathbf{v}_j^\ell, \mathbf{e}_{ij}^\ell\right) = \text{MLP}_e\!\left(\text{concat}[\mathbf{v}_i^\ell, \mathbf{v}_j^\ell, \mathbf{e}_{ij}^\ell]\right)$$

Edge memory is the GNN analog of the KV-cache: it tracks the history of each pairwise interaction across message-passing sub-steps within a single prediction step, giving the model spatiotemporal memory without increasing context window size.

#### 3b. Physics Bottleneck Node Update

**Standard aggregation:**  
Aggregate incoming force vectors to each node:

$$\tilde{\mathbf{v}}_i^{\ell+1} = \sum_{j \in \mathcal{N}(i)} \vec{F}_{ij}$$

This is a vector in $\mathbb{R}^3$ (physical space), not a hidden-dim feature vector. Expand to hidden dimension:

$$\mathbf{h}_i^{\ell+1} = \text{MLP}_\text{pre}\!\left(\text{concat}[\mathbf{v}_i^\ell, \tilde{\mathbf{v}}_i^{\ell+1}]\right) \in \mathbb{R}^{d_\text{hidden}}$$

**Physics bottleneck:**  
Project the hidden state through a low-dimensional physically-motivated subspace before expanding back:

$$z_{\text{bn},i} = \Pi_{\text{conserved}}\, \mathbf{h}_i^{\ell+1} \in \mathbb{R}^{d_\text{bn}}$$

$$\mathbf{v}_i^{\ell+1} = \text{MLP}_\text{post}(z_{\text{bn},i}) + \mathbf{v}_i^\ell \quad \text{(residual)}$$

**Construction of $\Pi_\text{conserved}$:**  
The projection matrix $\Pi_\text{conserved} \in \mathbb{R}^{d_\text{bn} \times d_\text{hidden}}$ maps the hidden state to a bottleneck space that includes explicit coordinates for:

| Coordinate | Physical Quantity | Dimension |
|---|---|---|
| $v_x, v_y, v_z$ | Linear momentum $\mathbf{p}_i$ | 3 |
| $\omega_x, \omega_y, \omega_z$ | Angular velocity $\boldsymbol{\omega}_i$ | 3 |
| $E_k$ | Kinetic energy $\frac{1}{2}m_i v_i^2$ | 1 |
| $E_p$ | Potential energy $m_i g z_i$ | 1 |
| Learned coordinates | Latent physics | $d_\text{bn} - 8$ |

The first 8 coordinates are initialized from interpretable physics quantities (linear/angular momentum, kinetic/potential energy); the remaining $d_\text{bn} - 8$ are learned. The full $\Pi_\text{conserved}$ is learnable but initialized near this physically-motivated basis.

**Why this works:**  
The bottleneck forces all information to flow through a low-dimensional physically-interpretable subspace. This is a **level 4** physics encoding (architectural soft bias from CLAUDE.md): it does not hard-enforce energy conservation, but it forces the network to represent state changes in terms of energy exchanges rather than arbitrary hidden features. The learned coordinates can capture quantities that matter for the specific physics being modeled but are not known a priori.

**Energy bottleneck only (hybrid hard+soft conservation):**  
Since linear and angular momentum are exactly conserved by the antisymmetric edge frame (hard constraint, level 5), the bottleneck can be restricted to **energy only**:

$$z_{\text{bn},i} = \Pi_E\, \mathbf{h}_i^{\ell+1} \in \mathbb{R}^{d_E}$$

where $d_E \ll d_\text{hidden}$ and $\Pi_E$ is initialized to project onto kinetic + potential energy coordinates. This is the recommended design: exact momentum conservation (Dynami-CAL) + soft energy bottleneck = levels 5 + 4 simultaneously.

---

### Stage 4: Decoder

**Acceleration decoding:**  
$$\hat{\ddot{\mathbf{p}}}_i = \text{MLP}_\text{decode}(\mathbf{v}_i^M) \in \mathbb{R}^{d_\text{spatial}}$$

(Predicts acceleration — the second derivative of position — analogous to GNS.)

**Euler integrator:**  
$$\hat{\dot{\mathbf{p}}}_i^{t+\Delta t} = \dot{\mathbf{p}}_i^t + \Delta t \cdot \hat{\ddot{\mathbf{p}}}_i$$
$$\hat{\mathbf{p}}_i^{t+\Delta t} = \mathbf{p}_i^t + \Delta t \cdot \hat{\dot{\mathbf{p}}}_i^{t+\Delta t}$$

For continuum field simulations (not particle-based), the decoder maps node features to field quantities instead:

$$\hat{u}(\mathbf{p}_i) = \text{MLP}_\text{field}(\mathbf{v}_i^M)$$

This is evaluated at each node $i$ to give the field value at that node position.

---

### Stage 5: Full Prediction Pipeline

```
Given: particle positions p_i, last C velocity histories, material types

1. Ghost-node boundary augmentation:
   For each node within R_wall of a wall:
     Create ghost node with reflected position, zero velocity

2. Build radius graph G with connectivity R
   For each edge (i,j): compute relative displacement r_ij

3. Encode: v_i^0 = MLP_v(x_i), e_{ij}^0 = MLP_e(r_ij)

4. Processor (M = 10 rounds):
   For each round ℓ = 0, ..., M-1:
     a. Construct antisymmetric edge frame {ê_k^ij} from p_i, p_j
     b. Compute symmetric edge embedding m_ij = MLP_m(v_i^ℓ + v_j^ℓ, ||r_ij||, e_ij^ℓ)
     c. Vectorize: F_ij = Σ_k m_{ij,k} · ê_k^{ij}
     d. Angular: τ_ij = A_ij - (p_i - x_{0,ij}) × F_ij
     e. Aggregate: ṽ_i = Σ_{j∈N(i)} F_ij, τ̃_i = Σ_{j∈N(i)} τ_ij
     f. Pre-MLP: h_i = MLP_pre(concat[v_i^ℓ, ṽ_i, τ̃_i])
     g. Bottleneck: z_{bn,i} = Π_E · h_i
     h. Post-MLP + residual: v_i^{ℓ+1} = MLP_post(z_{bn,i}) + v_i^ℓ
     i. Edge memory update: e_{ij}^{ℓ+1} = MLP_e(v_i^ℓ, v_j^ℓ, e_ij^ℓ)

5. Decode:
   p̈_i = MLP_decode(v_i^M)     (or field value for continuum)

6. Euler integrator:
   ṗ_i^{t+Δt} = ṗ_i^t + Δt · p̈_i
   p_i^{t+Δt} = p_i^t + Δt · ṗ_i^{t+Δt}

7. Strip ghost nodes from output
```

---

## Training

**Supervised loss on accelerations:**  
$$\mathcal{L}(\theta) = \mathbb{E}\!\left[\sum_i \|\hat{\ddot{\mathbf{p}}}_i - \ddot{\mathbf{p}}_i^{\text{true}}\|_2^2\right]$$

Training on one-step predictions from **ground-truth inputs** (teacher forcing).

**Noise injection for rollout stability (GNS technique):**  
Corrupt training velocity inputs with random-walk noise:

$$\tilde{\dot{\mathbf{p}}}_i^t = \dot{\mathbf{p}}_i^t + \sum_{k=0}^{t} \epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0, \sigma_v^2 I), \quad \sigma_v = 3 \times 10^{-4}$$

This matches the inference distribution (where inputs are model outputs with accumulated errors) to the training distribution.

---

## Benchmark Adaptation (1D Burgers' on Non-Uniform Grid)

GNN-PB's structural advantage: the non-uniform grid benchmark, where the grid has 5× refinement near $x = 0.5$.

**Representation:**  
- Each grid point is a node with position $x_i$ and state $u(x_i)$
- Edges connect adjacent nodes and nodes within radius $R = 3 \cdot \min(\Delta x)$
- Non-uniform spacing is handled naturally — no interpolation or coordinate transformation needed

**Encoder:** $\mathbf{v}_i^0 = \text{MLP}_v([u(x_i), x_i]) \in \mathbb{R}^{64}$

**Reduced architecture (benchmark scale):**
```
d_hidden: 64
Bottleneck dim d_E: 8  (energy coordinates)
Message-passing rounds M: 6
MLP depth: 2 layers
Total params: ~500K
```

**Conservation monitoring:**  
At benchmark time, explicitly track:
- Total linear momentum: $\sum_i m_i \dot{p}_i$ (should be exactly conserved)
- Total angular momentum: $\sum_i \mathbf{p}_i \times m_i \dot{\mathbf{p}}_i$ (should be exactly conserved)
- Total kinetic energy: $\sum_i \frac{1}{2} m_i \dot{p}_i^2$ (monitored, not conserved — dissipates physically)

Exact conservation of momentum with measurable energy dissipation is the expected behavior.

---

## Key Design Parameters

| Parameter | GNS Baseline | GNN-PB (Proposed) |
|---|---|---|
| Hidden dim $d_\text{hidden}$ | 128 | 128 |
| Message-passing rounds $M$ | 10 | 10 |
| Bottleneck dim $d_\text{bn}$ | None | 8–16 |
| Edge frame | Symmetric (no antisymmetry) | Antisymmetric (Dynami-CAL) |
| Momentum conservation | Approx. (learned) | **Exact (architectural)** |
| Energy conservation | Approx. (learned) | Soft (bottleneck) |
| Boundary treatment | Distance features | Ghost nodes |
| Edge memory | No | Yes (sub-step $K = 3$) |
| Total params | ~1M | ~2M |

---

## Connection to Mixture of Experts

**[AI Inference]:** The physics bottleneck is structurally analogous to MoE routing (from [[deep-memory-dissipative]]): both force information through a low-dimensional structured intermediate before expanding. MoE routes information through sparse expert gates; the physics bottleneck routes through sparse physically-meaningful axes. The connection suggests an implementation: the bottleneck layer could be implemented as a **physics-aware MoE** where each expert corresponds to one conserved quantity (momentum expert, energy expert, vorticity expert, etc.), with learned soft-routing weights. This would give the bottleneck adaptive per-physics expressivity beyond the fixed projection matrix.

**[AI Inference]:** EquiformerV3's SE(3)-equivariant irreps features (from [[equiformer-v3]]) could replace the standard MLP encoder in GNN-PB. High-$L_\text{max}$ irreps representations would give the edge features richer angular expressivity — important for non-spherical particles, oriented boundaries, or anisotropic materials. Cost: $O(L_\text{max}^4)$ tensor product complexity per edge. For $L_\text{max} = 2$ (quadrupole terms), this adds $16\times$ cost per edge operation but may substantially improve accuracy on oriented physics.

---

## Key Risks and Mitigations

**Risk 1: GNN scaling limits (over-smoothing)**  
At $M > 10$ message-passing rounds, node features converge to a local mean, losing distinguishing information ("over-smoothing"). At $M \leq 10$, performance is dominated by $M$; increasing $M$ linearly increases inference cost.

*Mitigation:* Residual connections (present in GNS and GNN-PB) slow over-smoothing. The physics bottleneck may also help by constraining the smoothing to happen in a physically-structured way. Graph rewiring (adding long-range edges to the graph) is an alternative.

**Risk 2: Learned graph adds optimization complexity**  
If the graph connectivity radius $R$ is learned rather than fixed, the graph construction is an additional optimization variable. This can lead to degenerate solutions (all nodes connected = $O(N^2)$ cost) or collapsed solutions (no nodes connected = no information propagation).

*Mitigation:* Fix $R$ based on physical length scales for the benchmark. Dynamic graph learning is deferred to the full model.

**Risk 3: Ghost-node explosion near complex boundaries**  
For complex geometries (curved walls, re-entrant corners), the ghost-node reflection may create non-physical ghost positions or too many ghosts per node.

*Mitigation:* Limit ghosts to one per wall surface normal. For benchmark (1D problem), only two wall ghosts are needed.

---

## See Also

- [[possible-architectures]] — benchmark plan; this is Architecture 5
- [[equivariant-gnns]] — equivariant GNNs and conservation constraints
- [[message-passing-belief-propagation]] — theoretical foundation for message passing
- [[mixture-of-experts]] — connection to physics bottleneck design
- [[gns-graph-network-simulators]] — empirical backbone; GNS provides the encode-process-decode framework
- [[dynami-cal-graphnet]] — antisymmetric edge frames; exact momentum conservation
- [[equiformer-v3]] — SE(3)-equivariant GNN patterns
- [[pc-deeponet-cfd]] — analogous hard constraint approach for continuum PDEs
- [[arch-autoregressive-transformer]] — Architecture 1: regular-grid alternative
- [[arch-physics-mamba]] — Architecture 4: alternative backbone for long sequences
