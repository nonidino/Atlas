# Skew-Symmetric Dynamics & Attention (the conservative channel)

**Type:** Concept (folder: adjusted-attention-mechanism)
**Status:** Foundational building-block page. The mathematical core of the initial model's *conservative channel*; referenced by [[symmetric-attention-physics]] and [[initial-model-architecture]].
**Related Concepts:** [[symmetric-attention-physics]], [[antisymmetric-signed-attention-transformer]], [[00-attention-overview]], [[normalization-scheme]], [[equivariant-gnns]]
**Related Summaries:** [[anti-symmetric-dgn]], [[hamiltonian-neural-networks]], [[transformers-particle-systems-clustering]], [[dynami-cal-graphnet]]

---

## Why a dedicated page

"Symmetric attention" and "skew-symmetric channel" sound like the same idea but are **opposite operators** doing complementary jobs ([[symmetric-attention-physics]] makes the distinction; this page is the math behind the conservative half). A symmetric operator *stretches* (dissipates or amplifies); a skew-symmetric operator *rotates* (preserves). Long-horizon stability and energy conservation — the initial model's headline goal — come from the skew-symmetric (rotational) part. This page collects the intuition, the mathematics, the trade-offs, and the prior art, including the specific question: *has anyone built skew-symmetric attention?*

---

## Intuition: rotation vs. stretching

Picture how a linear map $M$ acts on a state vector $h$ as you stack layers ($h^{\ell+1} \approx h^\ell + \epsilon M h^\ell$, i.e. the flow $\dot h = Mh$):

- A **symmetric** $M=M^\top$ has *real* eigenvalues. It stretches along its eigen-axes: positive eigenvalues blow the state up (exploding activations/gradients), negative ones shrink it toward zero (vanishing gradients, over-smoothing, energy leak). Either way **magnitude is not preserved** — the operator is dissipative or explosive.
- A **skew-symmetric** $M=-M^\top$ has *purely imaginary* eigenvalues. It generates a **rotation**: the state precesses without changing length. Energy and information are preserved indefinitely; the dynamics oscillate rather than decay or diverge.

This is exactly the difference physics cares about. A conservative (Hamiltonian) system rotates through phase space preserving energy; a dissipative system spirals inward. To make a deep network *behave like a conservative physical system*, make its depth-dynamics generator skew-symmetric.

---

## Mathematical background

### Definition and spectrum
A real matrix is **skew-symmetric** (antisymmetric) if $A = -A^\top$. Consequences:
- Eigenvalues are purely imaginary or zero: $A x = \lambda x \Rightarrow \lambda \in i\mathbb{R}$.
- The generated flow is **orthogonal** (norm-preserving):
$$h(t) = e^{At}h(0), \qquad (e^{At})^\top e^{At} = I \ \Rightarrow\ \|h(t)\| = \|h(0)\|\ \ \forall t.$$
- Quadratic "energy" is conserved: $\tfrac{d}{dt}\tfrac12\|h\|^2 = h^\top A h = 0$ because $h^\top A h = -(h^\top A h)$.

### Connection to Hamiltonian / symplectic structure
The symplectic form $J=\begin{psmallmatrix}0&I\\-I&0\end{psmallmatrix}$ is skew-symmetric ($J^\top=-J$). Hamiltonian dynamics $\dot z = J\,\nabla H(z)$ conserve $H$ and (Liouville) phase-space volume. Skew-symmetric linear dynamics are the *linear special case* ($H=\tfrac12 z^\top S z$), so a skew-symmetric channel is the simplest architectural way to import Hamiltonian energy-preservation ([[hamiltonian-neural-networks]]).

### The A-DGN parameterization (used by the initial model)
A learnable matrix $W$ is projected to skew-symmetric and given a tunable damping ([[anti-symmetric-dgn]]):
$$A = W - W^\top - \gamma I.$$
$W-W^\top$ is exactly skew (imaginary spectrum, conservative); $-\gamma I$ shifts the spectrum left by $\gamma$ for *controlled* dissipation. $\gamma=0$ is exactly energy-preserving; small $\gamma>0$ gives long-but-finite memory and lets the model **match the system's real physical dissipation** (viscosity, drag, resistivity). This is the **port-Hamiltonian** split: conservative part $+$ explicit dissipative part.

### Getting orthogonal weights from skew-symmetric ones
The matrix exponential and the Cayley transform map skew-symmetric $A$ to an orthogonal $Q$:
$$Q = e^{A}\quad\text{or}\quad Q=(I-A)(I+A)^{-1},$$
the trick behind orthogonal/unitary RNNs. Useful if one wants an *exactly* norm-preserving layer rather than an approximately-conservative Euler step.

### Two distinct "(anti)symmetries" in attention — do not conflate
| Object | Symmetry used | Effect |
|---|---|---|
| Attention **score matrix** $A_{ij}$ | **symmetric** $A_{ij}=A_{ji}$ | reciprocity (Newton's 3rd law); but diffusion $\Rightarrow$ dissipative |
| Per-token **linear channel** $Mh_i$ | **skew-symmetric** $M=-M^\top$ | rotation $\Rightarrow$ energy/information preserved |

The initial model uses **both at once**: symmetric *scores* for reciprocal interaction, a skew-symmetric *linear channel* for conservation ([[symmetric-attention-physics]], [[antisymmetric-signed-attention-transformer]]). They are not redundant — one conserves momentum, the other conserves energy.

---

## Pros

- **No vanishing/exploding gradients** — imaginary spectrum keeps gradient norms bounded; stable training at 20+ layers (A-DGN's headline result).
- **Information preserved across depth** — no over-smoothing / rank collapse; fine token distinctions survive, the property [[transformers-particle-systems-clustering]] shows standard attention destroys.
- **Energy conservation by construction** — long-horizon rollouts do not leak or inject energy (with $\gamma=0$); the inbuilt stability mechanism.
- **Tunable dissipation** — $\gamma$ (or a learned per-channel damping) places the model anywhere on the conservative↔dissipative spectrum, so ideal *and* viscous physics share one structure.
- **Reversibility** — norm-preserving, invertible dynamics enable memory-efficient reversible backprop (RevNet/Reformer idea).

## Cons

- **Restricts expressivity** — the linear part can only rotate, not select/contract; the constrained DOF must be compensated by the nonlinearity and the interaction term.
- **Cannot model dissipation without an explicit damping term** — pure skew-symmetry is *too* conservative for real (viscous, resistive) systems; needs $-\gamma I$ or a port-Hamiltonian dissipation channel.
- **Discretization sensitivity** — forward Euler needs a small step $\epsilon < 2/\|J\|_2$ for stability; symplectic/implicit integrators are more robust but costlier.
- **Clashes with standard normalization** — LayerNorm/√-variance normalization destroys the norm the channel preserves ([[normalization-scheme]]); must be dropped inside the conservative block.
- **Parameterization overhead** — enforcing $W-W^\top$ halves effective linear DOF; multi-head combination must avoid reintroducing asymmetry through $W_O$ ([[multihead-attention]]).

---

## Have models used skew-symmetric attention?

**The building block is well-established; its use *inside attention* is nascent.**

**Established skew-symmetric / orthogonal dynamics (not attention):**
- **AntisymmetricRNN** (Chang et al., ICLR 2019) — skew-symmetric recurrent weight; the original "stable deep dynamics via antisymmetry" result.
- **A-DGN** (Gravina et al., ICLR 2023) — skew-symmetric *graph message passing*; the wiki's rigorous foundation ([[anti-symmetric-dgn]]).
- **LEM** (Rusch et al., ICLR 2022) — multiscale antisymmetric ODE for long-memory RNNs.
- **Orthogonal/unitary RNNs** (uRNN, Arjovsky et al. 2016; expRNN, Lezcano-Casado & Martínez-Rubio 2019) — orthogonal recurrent weights via the skew-symmetric exponential map.
- **Hamiltonian/symplectic nets** — [[hamiltonian-neural-networks]], Symplectic ODE-Net (Zhong et al. 2020), SympNets; RevNet (Gomez et al. 2017) and **Reformer** (Kitaev et al. 2020) use Hamiltonian-style reversible coupling.
- **Graph-side antisymmetry for physics** — [[dynami-cal-graphnet]] uses antisymmetric *edge frames* (a different antisymmetry — per-edge $\mathbf m_{ij}=-\mathbf m_{ji}$ — for exact momentum).

**Closest things to skew-symmetric / conservative *attention* specifically:**
- **Sinkformers** (Sander et al., AISTATS 2022) — Sinkhorn-normalized *doubly-stochastic* attention; symmetric-leaning coupling with a Wasserstein-gradient-flow interpretation. Symmetric scores, not skew-symmetric channel.
- **Modern Hopfield / "Hopfield Networks is All You Need"** (Ramsauer et al. 2020) and **Energy Transformer** (Hoover et al., NeurIPS 2023) — attention *derived from an energy function* with symmetric coupling weights; energy-based, so closer to conservative, but again symmetric-not-skew.
- **The wiki's own [[antisymmetric-signed-attention-transformer]]** — the proposal that combines a symmetric score with a skew-symmetric linear channel.

**Bottom line:** as of the knowledge cutoff, **no published transformer puts a skew-symmetric channel inside attention** (the existing [[antisymmetric-signed-attention-transformer]] page states the same). The nearest art makes attention *symmetric/energy-based* (Sinkformers, Hopfield/Energy Transformers) but stops short of the rotational skew-symmetric channel. This gap is part of what makes the initial model's attention a genuine research bet rather than a re-implementation.

---

## [AI Inference]

**[AI Inference]:** The reason skew-symmetric *attention* is unexplored while skew-symmetric *RNNs/GNNs* are mature is historical, not fundamental: attention research optimized for language, where dissipative clustering (= semantic abstraction) is desirable, so there was no pressure to make attention conservative. Physics inverts the incentive — conservation is the goal — so porting the well-understood AntisymmetricRNN/A-DGN machinery into attention is low-risk on the math side and high-value on the physics side. The open work is engineering (multi-head $W_O$ structure, a fused kernel for the $\tanh$ symmetric score), not theory.

**[AI Inference]:** Energy-based attention (Hopfield/Energy Transformer) and skew-symmetric attention are two halves of a port-Hamiltonian view: the energy function supplies a conservative *gradient* structure (the symmetric/potential part) while the skew-symmetric channel supplies the *symplectic rotation*. A transformer that combined a learned energy (for the interaction) with a skew-symmetric channel (for the flow) would be a literal discretized port-Hamiltonian field model — the cleanest possible realization of "attention as conservative physical dynamics," and a natural next synthesis page.

---

## See Also

- [[symmetric-attention-physics]] — the symmetric-score half; why both halves are needed
- [[antisymmetric-signed-attention-transformer]] — the full proposed combination
- [[anti-symmetric-dgn]] — the rigorous stability theorem (skew-symmetric ODE)
- [[hamiltonian-neural-networks]] — the symplectic/energy-conservation cousin
- [[normalization-scheme]] — why LayerNorm must be dropped to keep norm preservation
- [[multihead-attention]] — preserving skew-symmetry through head combination
- [[00-attention-overview]] — where this sits in the attention design space
- [[transformers-particle-systems-clustering]] — proof that standard attention is the dissipative opposite
- [[dynami-cal-graphnet]] — a different antisymmetry (edge frames) for momentum
