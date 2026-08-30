# Conditioning & Fundamental Constants (Noether 1.1)

**Type:** Concept — model portion (folder: Noether 1.1)
**Status:** Design. Consolidates how *every* physical constant — scalar or vector, dimensional or dimensionless — enters the model, into one contract other portion pages reference. Generalizes [[noether-1.0]] §3.4 (single-scalar $\nu$ conditioning) and [[noether-1.0-rbc]] §4.4 ($\log\mathrm{Ra},\log\Pr$).
**Related Concepts:** [[00-noether-1.1-overview]], [[normalization-scheme]], [[pfm-interface-design]], [[graph-tokenizer-1.1]], [[backbone-1.1]], [[discovered-conservation-1.1]]

---

## What it does — intuitively

The same fluid at two viscosities is two different physics; the same $r^{-2}$ law with two different $G$ is two different orbits. Constants are **what selects which member of a family of dynamics the model should produce.** Noether 1.1 treats a constant as an input with three possible jobs, and a given constant may hold more than one:

1. **A ruler** — it sets the characteristic scale used to nondimensionalize the state (so $128^2$ water and $128^2$ honey look comparable to the encoder).
2. **A dial** — its value, injected as a token, tells the backbone *which* regime of the learned dynamics to run (a Reynolds-analog, a coupling strength).
3. **An arrow** — a *vector* constant (gravity $\mathbf g$, an external $\mathbf B$-field) additionally breaks isotropy and must enter as a direction the model can rotate with the frame, not a magnitude alone.

The design goal is that **no constant is ever hard-coded into a module.** A plasma, a self-gravitating cloud, and a convection cell differ only in which constants are non-zero and what values they take — never in the code path.

## What it does — mathematically

Let the input carry a set of physical constants $\{c_k\}$, each tagged scalar or vector.

**Nondimensionalization (the ruler).** Characteristic scales $L_0, U_0, T_0, \Phi_0$ are built from the constants and domain per [[normalization-scheme]]; the state is rescaled $\hat u = u/\Phi_0$, $\hat x = x/L_0$, $\hat t = t/T_0$ before tokenization. This is the *only* place dimensional constants act multiplicatively on the field.

**Scalar conditioning tokens (the dial).** Dimensionless groups $\pi_k$ (Reynolds, Rayleigh, Prandtl, Mach, …) formed from the $c_k$ are log-scaled and embedded:
$$z_{\text{param}} = \mathrm{MLP}\big(\big[\log \pi_1,\ \dots,\ \log \pi_m\big]\big)\in\mathbb R^{d},\qquad z_{\Delta t}=\mathrm{MLP}\big(\log(\Delta t/T_0)\big),$$
added to **every** token of every field stream (both/all context frames). A missing group is masked, not zero-filled, so field cardinality and constant cardinality vary independently.

**Vector conditioning / covariant channels (the arrow).** A vector constant $\mathbf c$ enters as a per-node steerable feature $\gamma_{\text{vec}}(\mathbf c, x_i)$ carried alongside the scalar latent, so that under a global rotation $R$ the conditioning transforms as $\mathbf c \mapsto R\mathbf c$ (P9.2 covariant conditioning, [[open-architectural-problems]]). This is what lets gravity break vertical isotropy *without* hard-coding "down," and is togglable per the equivariance ablation.

**Concretely, $\gamma_{\text{vec}}$ is a linear (bias-free) map on the raw vector, optionally gated by an invariant scalar:**
$$\gamma_{\text{vec}}(\mathbf c, x_i) = \sigma\big(\mathrm{MLP}(\|\mathbf c\|,\ z_{\text{cond}})\big)\cdot W_{\text{vec}}\,\mathbf c,$$
kept as a **separate vector-valued channel**, never passed through an elementwise nonlinearity on its own — that would break the rotation identity $\mathbf c\mapsto R\mathbf c$ (a bias term would too, for the same reason: it doesn't transform under $R$). Only linear maps and gates built from *rotation-invariant* scalars like $\|\mathbf c\|$ preserve equivariance; this is the standard gated-equivariant-nonlinearity construction from the equivariant GNN literature ([[equivariant-gnns]]), instantiated here for a conditioning constant rather than a node feature. The channel enters additively alongside the scalar latent, so any downstream operation that must stay equivariant (e.g. [[backbone-1.1]]'s messages) touches it only through the same rotation-safe operation class.

**Where this channel actually couples into the dynamics** is specified in [[backbone-1.1]]: as **rotation-invariant contractions** ($\langle\gamma_{\text{vec}},x_j-x_i\rangle$, $\|\gamma_{\text{vec}}\|$) that modulate attention scores/gates — the source of buoyancy-like anisotropy — and as an **additive vector source term** on the forced stream's value channel (a body force). The latter is the same place the antisymmetry gate $g_\tau$ opens below $1$, since an external body force is genuinely non-momentum-conserving: the arrow and the open gate are one site. This covariant construction is the **hard-equivariant** path; in the default no-hard-equivariance setting ([[open-architectural-problems]] Problem 9) the vector instead enters as ordinary component channels on the stream's tokens, with equivariance learned by augmentation. Either way it is never a backbone weight — it rides the token.

## Physics relevance

- **Buckingham $\pi$ done by the model.** Feeding dimensionless groups (not raw dimensional constants) is the learned analog of the Buckingham $\pi$ theorem: the dynamics depend only on the $\pi_k$, so the model that conditions on them generalizes across all dimensional realizations of the same $\pi_k$ — the mechanism behind "train at one scale, deploy at another."
- **Symmetry bookkeeping.** Splitting scalar (invariant) from vector (covariant) conditioning is exactly Noether bookkeeping: scalar constants preserve all spatial symmetries; a vector constant *breaks* the ones not fixing its direction, and the covariant channel is how that broken symmetry is represented rather than baked in.
- **Regime selection.** The dial is what a foundation model needs to be equation-agnostic: the same weights run laminar or turbulent, weakly or strongly coupled, depending on the injected $\pi_k$ — [[in-context-learning-physics]] made explicit through conditioning rather than context alone.

## How fundamental constants are included

This *is* the constants page, so the contract is stated in full here and referenced elsewhere:

| Constant kind               | Example                                        | Enters as                                                  | Which stage sees it                       |
| --------------------------- | ---------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------- |
| Dimensional scale-setter    | $\nu$, $\kappa$, $c$ (light), $G$              | characteristic scale $L_0,U_0,T_0$ → nondimensionalization | tokenizer input, decoder output           |
| Dimensionless group         | $\mathrm{Re},\mathrm{Ra},\Pr,\mathrm{Ma}$      | log-scaled scalar conditioning token $z_{\text{param}}$    | every backbone token                      |
| Vector field constant       | $\mathbf g$, external $\mathbf B$, $\mathbf E$ | covariant steerable channel $\gamma_{\text{vec}}$          | backbone (breaks isotropy), edge features |
| Interaction-range constant  | Debye length, cutoff radius                    | sets edge radius $r$ / candidate envelope $R$              | [[edge-generation-1.1]]                   |
| Invariant-defining constant | which symmetry a $c_k$ implies                 | gates which $C_k$ are candidate invariants                 | [[discovered-conservation-1.1]]           |

The last two rows are new in 1.1 and important: a constant does not only condition the *dynamics*, it can condition the *graph* (a screening length literally sets who talks to whom) and the *conservation set* (a symmetry a constant respects tells the invariant-discovery search where to look). Constants are therefore threaded through **structure**, not just through a bias vector.

**[AI Inference]:** Making interaction-range and invariant-set *functions of the constants* is what keeps the model from silently assuming a regime. If the Debye length is an input to edge construction, the same code handles a collisionless and a collisional plasma; if it were a fixed radius, the model would quietly be one or the other.

## See Also
- [[normalization-scheme]] — the ruler, in full
- [[pfm-interface-design]] — the 5-input interface this conditioning realizes
- [[edge-generation-1.1]] — where range-constants act on topology
- [[discovered-conservation-1.1]] — where symmetry-constants gate the invariant search
