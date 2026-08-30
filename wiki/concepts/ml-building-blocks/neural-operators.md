# Neural Operators

**Type:** Core Concept  
**Related Sources:** DeepONet multi-operator, PC-DeepONet, VarMiON, Navier-Stokes non-uniform grids

---

## Definition

A **neural operator** is a neural network that learns a mapping between function spaces rather than between finite-dimensional vectors. For a PDE parameterized by some input function $u$ (e.g., initial condition, boundary condition, coefficient), a neural operator $G_\theta$ approximates the solution operator:
$$G: \mathcal{U} \to \mathcal{V}, \quad G[u] = v$$
where $u \in \mathcal{U}$ and $v \in \mathcal{V}$ are functions, not vectors. This makes neural operators **discretization-invariant** — once trained, they can be evaluated at arbitrary query points.

---

## Key Architecture: DeepONet

The **Deep Operator Network (DeepONet)** (Lu et al., 2019) approximates $G[u](x)$ via:

$$G[u](x) \approx G_\theta[\hat{u}](x) = \sum_{k=1}^{K} p_k(\hat{u})\, b_k(x)$$

- **Branch network** $p_k(\hat{u})$: takes the input function discretized at $m$ sensor points $\hat{u} = [u(x_1),\ldots,u(x_m)]$; outputs $K$ coefficients.
- **Trunk network** $b_k(x)$: takes a query coordinate $x \in \mathbb{R}^d$; outputs $K$ basis function values.
- Output = dot product of branch and trunk outputs.

Training loss:
$$\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N \left\|v_i - G_\theta[u_i](x_i)\right\|^2$$

---

## Key Architecture: Fourier Neural Operator (FNO)

FNO (Li et al., 2020) applies convolution in Fourier space, exploiting that convolution operators are diagonal in frequency space:
$$(\mathcal{K}(a; \phi)v)(x) = \mathcal{F}^{-1}(R_\phi \cdot \mathcal{F}(v))(x)$$
where $R_\phi$ is a learnable complex-valued weight tensor in frequency space. FNOs are naturally periodic and resolution-invariant.

---

## Variants and Extensions

| Variant | Key Feature |
|---|---|
| **PI-DeepONet** | Physics-informed training; loss includes PDE residual |
| **PC-DeepONet** | Hard-enforced divergence-free constraint via skew-symmetric Jacobian |
| **D2NO / MODNO** | Distributed multi-operator learning with shared trunk, local branches |
| **VarMiON** | Branch architecture determined by variational (weak) form of PDE |
| **HyDEA** | DeepONet used within hybrid deep-learning + classical-iterative solver |

---

## Physics-Informed Neural Operators

Physics can be incorporated into neural operators two ways:

**Soft constraint (loss penalty):**
$$\mathcal{L}_\text{total} = \mathcal{L}_\text{data} + \lambda \mathcal{L}_\text{physics}$$
where $\mathcal{L}_\text{physics}$ penalizes PDE residual, boundary condition violations, etc.

**Hard constraint (architectural enforcement):**  
Design the output map so that physical laws are satisfied exactly — e.g., PC-DeepONet's divergence-free velocity field:
$$\mathbf{v} = \text{div}(J_b - J_b^\top) \implies \nabla\cdot\mathbf{v} = 0 \text{ exactly}$$

---

## Multi-Operator Learning

The core challenge for a PFM using neural operators: a single model must approximate many different operators $G_1, G_2, \ldots, G_C$ corresponding to different physical systems.

**D2NO/MODNO** approach: shared trunk (output basis functions shared across operators) + per-operator branches (operator-specific coefficient encoding). After distributed pretraining, operator-averaged initialization enables rapid PI fine-tuning to new operators with zero labeled data.

---

## Discretization Invariance

Unlike standard neural networks trained on fixed grids, neural operators can in principle evaluate at any query point after training. This is crucial for:
- Variable resolution inference.
- Transfer between simulation grids.
- Query-adaptive evaluation (finer resolution where gradients are steep).

In practice, FNOs require fixed-resolution training while DeepONets maintain full discretization invariance.

---

## Relevance to PFM

Neural operators are a natural building block for a PFM because:
1. They directly address the operator learning problem (mapping from parameters/ICs to solutions).
2. Physics can be encoded at multiple levels (loss, architecture, pre-conditioning).
3. Multi-operator pretraining frameworks (D2NO) provide structured pretraining strategies.
4. Combination with transformers (as in large foundation models) extends their expressive capacity.

**[AI Inference]:** Transformer-based neural operators (treating operator kernels as learned attention mechanisms) may unify the neural operator and foundation model paradigms. The Galerkin transformer, for instance, interprets attention weights as discretized integral kernels, connecting self-attention to classical function-space methods.

---

## See Also

- [[poseidon-pde-foundation-model]] — operator learning scaled to foundation-model size (scOT + all2all)
- [[partial-differential-equations]]
- [[transfer-learning-fine-tuning]]
- [[physics-foundation-models]]
- [[branch-trunk-operator-tokens]] / [[spectral-fourier-tokens]] — operator tokenizations
- [[deeponet-multi-operator]]
- [[pc-deeponet-cfd]]
- [[varmion-viscous-flows]]
- [[navier-stokes-nonuniform-grids]]
