# Phase 1 — Native Scaffold

**Type:** Implementation spec — agent task (folder: Atlas 0.1 / Atlas 0.1 implementation)
**Phase of:** [[00-atlas-0.1-implementation-plan]]. **Depends on:** nothing (buildable against random tensors). **Unblocks:** [[impl-atlas-0.1-phase2-experts]], [[impl-atlas-0.1-phase3-integration]].
**Design pages:** [[graph-tokenizer-atlas-0.1]], [[edge-generation-atlas-0.1]], [[unet-hierarchy-atlas-0.1]], [[agent-definition-atlas-0.1]].
**Milestones:** M0 (repo + config), M3 (end-to-end forward pass with identity experts).
**Status: BUILT 2026-08-09** — branch `atlas-0.1`, package `src/atlas/`, 32 tests, M0 and M3 green. Three places where the build diverged from this page are corrected inline below and recorded in full in [[atlas-0.1-implementation-log]]; §6 summarizes them.

---

# 1. Intuition

**Build the plumbing before the physics.** This phase produces a complete, runnable Atlas forward pass in which every expert is the identity function. Nothing it outputs is physically meaningful. That is the point: if the tensor shapes, the graph construction, the pooling, the decoding, and the gradient flow are all correct *before* any expert exists, then when Phase 2 drops real experts in, any failure is attributable to the expert — not to a silently mismatched interface three layers away.

The scaffold answers four structural questions:

1. **How does a physical region become tokens?** Each agent's field arrays are cut into patches, each patch becomes one token, and each token carries its own position, its own derived features (gradients, vorticity), and its agent's conditioning scalars.
2. **Which tokens are allowed to talk to which?** Only tokens on *both sides of a declared interface*. Two agents that share no edge in the declared list have no path between them at this level, by construction. The graph is not learned — it is *materialized* from geometry ([[edge-generation-atlas-0.1]], Mechanism A).
3. **How does whole-vehicle information reach a token?** Through the bottleneck. Local token-level physics cannot know the vehicle's altitude or acceleration; pooling to a single vehicle-level vector and broadcasting back is what carries that.
4. **How do tokens become fields again?** The decoder, which is the tokenizer run backwards.

A design note worth internalizing: **the tokenizer is per-agent and completely independent across agents.** There is no cross-agent mixing until the edge layer. This is what makes agents genuinely separable computations and what will later make it possible to swap one expert without touching another.

---

# 2. Theory

## 2.1 Tokenization as local function-space encoding

An agent's state is a field $x_a:\Omega_a\to\mathbb R^{C_a}$. Partition $\Omega_a$ into patches $\{\Omega_a^{(i)}\}$ and encode each patch into a latent vector:

$$Z_a^{(i)}=\phi_\theta\!\left(x_a\big|_{\Omega_a^{(i)}},\ \nabla x_a\big|_{\Omega_a^{(i)}},\ \mathbf r^{(i)},\ \mathbf z_{\text{cond},a}\right)\in\mathbb R^{d}$$

Three deliberate choices, inherited from [[graph-tokenizer-1.1]] via [[graph-tokenizer-atlas-0.1]]:

**Redundant derived features.** Gradients, divergence, and vorticity are computed *before* tokenization and concatenated as extra channels, even though they are in principle recoverable from the raw field. A patch embedding is a linear map over an $8\times8$ stencil; asking it to also learn a finite-difference operator wastes capacity on something we can supply exactly.

$$\nabla\!\cdot\!\mathbf u=\partial_z u+\partial_y v,\qquad \omega_x=\partial_z v-\partial_y u,\qquad \|\nabla T\|$$

**Fourier positional features** of the patch centroid $\mathbf r^{(i)}=(z_i,y_i)$ in the body frame:

$$\gamma(\mathbf r)=\left[\sin(2^k\pi \mathbf r),\ \cos(2^k\pi \mathbf r)\right]_{k=0}^{K-1},\qquad K=6$$

Absolute body-frame position matters physically here (the throat is *at* $z{=}0.40$), unlike in a homogeneous turbulence box where only relative position matters — so absolute encoding is correct, not a shortcut.

**FiLM conditioning** on the agent's dimensionless scalars:

$$Z\leftarrow \gamma_{\text{film}}(\mathbf z_{\text{cond}})\odot Z+\beta_{\text{film}}(\mathbf z_{\text{cond}})$$

This is the smooth-crossover mechanism from [[regime-moe-architecture]]'s finding 2: Mach number and Reynolds number vary continuously, so they modulate features continuously rather than switching a discrete branch.

## 2.2 Edge instantiation as interface matching

For a declared edge $(a\!\to\!b,\ \text{types }\mathcal T)$ with shared interface curve $\Gamma_{ab}$:

1. **Boundary token identification.** Token $i$ of agent $a$ is a boundary token for this edge iff $\mathrm{dist}\!\left(\Omega_a^{(i)},\Gamma_{ab}\right)<\epsilon_{\text{tol}}$ (default $\epsilon_{\text{tol}}=$ one patch diagonal).
2. **Arc-length projection.** Parameterize $\Gamma_{ab}$ by arc length $s\in[0,L_\Gamma]$. Project each boundary token centroid to its nearest point on $\Gamma$, giving $s_i$.
3. **Matching.** Connect $i$ (side $a$) to the $K=2$ nearest $j$ (side $b$) in $s$. $K=2$ rather than 1 so a resolution mismatch across the interface (the gas grid is finer than the shell grid) does not leave dangling tokens.
4. **Edge attributes.**
$$\mathbf a_{ij}=\left[\Delta s_{ij},\ \mathbf n_{ij},\ \ell_{ij}/L_\Gamma,\ \text{onehot}(\mathcal T)\right]$$
where $\mathbf n_{ij}$ is the interface outward normal and $\ell_{ij}$ the shared interface length assigned to the pair (needed later so a flux can be *integrated*, not merely evaluated — see [[impl-atlas-0.1-phase3-integration]]).

**Edges are bidirectional but not symmetric:** both $(i\!\to\!j)$ and $(j\!\to\!i)$ are stored, with $\mathbf n$ flipped, because heat flowing from chamber to wall is not the same operation as the wall's temperature constraining the gas.

## 2.3 Typed message passing

Per edge type $\tau\in\{\text{heat},\text{pressure},\text{stress},\text{fluid},\text{conservation}\}$, a separate projection triple, with messages summed over types:

$$m_i=\sum_{\tau}\ \sum_{j\in\mathcal N_\tau(i)}\alpha^{(\tau)}_{ij}\,W_V^{(\tau)}\!\left[h_j\,\|\,\mathbf a_{ij}\right],\qquad
\alpha^{(\tau)}_{ij}=\mathrm{softmax}_j\!\left(\frac{\left(W_Q^{(\tau)}h_i\right)^{\!\top}\!\left(W_K^{(\tau)}h_j\right)}{\sqrt{d_{\text{head}}}}\right)$$

This is the same typed-multigraph mechanism as [[backbone-1.1]], with one simplification: the type is **declared**, so there is no learned gate deciding which type is active. An edge either has type $\tau$ or it does not.

**Empty-type skipping matters for performance.** Most edges carry only 1–2 of the 5 types. Precompute the per-type edge slices **once per graph** rather than masking every layer — a `.any()` check per type per layer is a host–device synchronization on GPU and costs more than the computation it guards. (This exact issue cost measurable throughput in the parallel Noether track; see [[implementation-log]] 2026-07-19.)

## 2.4 The 2-level hierarchy

Per [[unet-hierarchy-atlas-0.1]]'s Stage 1 scope reduction — **2 levels, not 4**:

$$\underbrace{\{Z_a\}_{a=1}^{7}}_{\text{token level, }\sim1.1\text{k tokens}}\ \xrightarrow{\ \text{attention pool}\ }\ \underbrace{\{\bar Z_a\}}_{7\text{ agent vectors}}\ \xrightarrow{\ \text{aggregate}\ }\ \underbrace{\zeta}_{\text{1 bottleneck vector}}$$

Pooling is attention-weighted, not mean, so a small high-gradient region (the throat, a shock) can dominate its agent's summary:

$$\bar Z_a=\sum_i \mathrm{softmax}_i\!\left(w^{\!\top}Z_a^{(i)}\right)Z_a^{(i)}$$

The bottleneck vector $\zeta$ is concatenated with the **rigid-body state** and gravity (per [[global-fields-and-topology-atlas-0.1]], gravity enters here, not as an edge) and is where the non-learned trajectory expert reads and writes in Phase 3.

Unpooling broadcasts $\zeta$ back to every token by FiLM, plus a **skip connection** carrying the pre-pool token state:

$$Z_a^{(i)}\leftarrow Z_a^{(i)}+\mathrm{MLP}\!\left(\gamma_\zeta(\zeta)\odot Z_a^{(i)}+\beta_\zeta(\zeta)\right)$$

The skip is what stops the global correction from erasing local detail — a whole-vehicle acceleration should shift a boundary-layer profile, not overwrite it.

---

# 3. Implementation

## 3.1 Per-agent grid and token budget

| Agent | Grid $N_z\times N_y$ | Patch | Tokens $P_a$ | Channels $C_a$ (raw + derived) |
|---|---|---|---|---|
| `a` | $48\times80$ | $8\times8$ | 60 | 6 + 8 |
| `b` | $112\times80$ | $8\times8$ | 140 | 5 + 7 |
| `e` | $120\times96$ | $8\times8$ | 180 | 5 + 7 |
| `c` | $2\times(232\times8)$ | $8\times4$ | 116 | 6 + 4 |
| `d` | $184\times96$ (wall-stretched) | $8\times8$ | 276 | 5 + 7 |
| `f` | $152\times80$ | $8\times8$ | 190 | 5 + 7 |
| `g` | $152\times96$ minus plume | $8\times8$ | ~150 | 5 + 7 |
| **Total** | | | **~1,112** | |

*As built:* **1,114** — `g` is 19×12 = 228 patches minus the 4×19 that lie entirely inside the plume. Every other agent matches this table exactly. Each agent additionally carries **one structural active-mask channel** (not counted above, because it is not a physical field): `b` and `e` are bounding-box grids with the nozzle wall inside them, and a token must be able to distinguish "this cell is zero" from "this cell is not my domain." Measured mean active fraction: `b` 0.89, `e` 0.67, `g` 0.91, rest 1.00.

Agent `d` uses a wall-normal **stretched** grid (geometric ratio 1.08 from the wall) so the boundary layer is resolved without paying uniform-fine cost across the whole 3 m span. The patch layout is computed in *index* space, so stretching does not complicate tokenization — but it does mean patch physical areas vary, and `layout.cell_area` must be carried for any flux integration downstream.

## 3.2 `geometry/domains.py`

```python
@dataclass(frozen=True)
class AgentSpec:
    name: str                       # 'a'...'g'
    kind: str                       # 'reacting'|'confined_gas'|'solid'|'external_gas'|'free_jet'
    fields: tuple[str, ...]
    grid: GridSpec                  # Nz, Ny, node coords, cell areas, stretching
    dt_model: float
    expert: str                     # 'reacting_flow'|'external_flow'|'thermostruct'|'rigid_body'

@dataclass(frozen=True)
class EdgeSpec:
    src: str; dst: str
    types: tuple[str, ...]          # subset of the 5 declared types
    curve: InterfaceCurve           # polyline in body frame, arc-length parameterized

ATLAS_01_AGENTS: tuple[AgentSpec, ...]   # the 7, per 00-atlas-0.1-implementation-plan
ATLAS_01_EDGES:  tuple[EdgeSpec, ...]    # the 7, b-g absent
```

**Build-time assertions (run at import, not in tests only):**
- Agent domains are pairwise non-overlapping and their union covers the modelled region.
- Every `EdgeSpec.curve` lies within $\epsilon_{\text{tol}}$ of both agents' boundaries.
- Every edge's curve has non-zero length.
- `b-g` is **not** present (guard against reintroduction — [[edge-generation-atlas-0.1]]).

## 3.3 `geometry/layout.py`

```python
class PatchLayout:
    centroids:   Tensor  # [P, 2]  body-frame (z, y)
    cell_area:   Tensor  # [P]     physical area of the patch
    agent_id:    Tensor  # [P]     which agent each token belongs to
    boundary_of: dict[int, Tensor]  # edge_idx -> token indices on that interface
    arc_s:       dict[int, Tensor]  # edge_idx -> arc-length coordinate per boundary token
```

`PatchLayout` is **static for a given config** — it depends only on geometry, never on field values. Build it once and memoize it; do not recompute per forward pass. (Same lesson as [[implementation-log]] 2026-07-19: recomputing static geometry every step was a measurable cost in the Noether track.)

## 3.4 `model/tokenizer.py`

```python
class AgentTokenizer(nn.Module):
    def __init__(self, spec: AgentSpec, d_model: int, n_fourier: int = 6): ...
    def forward(self, fields: Tensor, cond: Tensor) -> Tensor:
        # fields [B, C_raw, Nz, Ny]; cond [B, n_cond]  ->  tokens [B, P_a, d_model]
```

Pipeline: derived-feature computation (central differences, one-sided at boundaries) → concat → `nn.Unfold` into patches → flatten → `Linear(patch_cells * C_total, d_model)` → add Fourier positional embedding → FiLM by `cond`.

One tokenizer instance **per agent kind**, not per agent — agents `d`, `f`, `g` share an `external_gas` tokenizer (same fields, same physics family), which is both a parameter saving and a mild inductive bias that these three are the same kind of thing.

## 3.5 `model/edges.py`

```python
def build_graph(layout: PatchLayout, edges: tuple[EdgeSpec, ...]) -> Graph:
    """Deterministic. No learned component. Memoized on layout identity."""

@dataclass
class Graph:
    edge_index: Tensor   # [2, E]  global token indices
    edge_type:  Tensor   # [E]     index into the 5 declared types
    edge_attr:  Tensor   # [E, d_edge]
    type_slices: dict[int, Tensor]   # precomputed per-type index slices
```

Note there is **no** KNN, no scorer, no Gumbel sampling here — that machinery belongs to [[edge-generation-1.1]]'s discovery setting and to Mechanism B, which is deferred ([[edge-generation-atlas-0.1]]).

## 3.6 `model/hierarchy.py`, `model/decoder.py`, `model/atlas.py`

```python
class TwoLevelHierarchy(nn.Module):
    def pool(self, tokens, layout)  -> tuple[Tensor, Tensor]:   # agent_vecs [B,7,d], bottleneck [B,d]
    def unpool(self, tokens, bottleneck) -> Tensor              # FiLM broadcast + skip

class AgentDecoder(nn.Module):
    def forward(self, tokens: Tensor) -> Tensor:  # [B,P_a,d] -> [B, C_raw, Nz, Ny]
```

The decoder predicts a **field increment**, not an absolute state:

$$\hat x_{t+\Delta t}=x_t+\Delta t\cdot \mathcal D_\theta(Z)$$

Increment prediction is the single most load-bearing choice for rollout stability: it makes the identity map the zero-output default, so an untrained model degrades to "nothing changes" rather than to noise, and it keeps target magnitudes $O(1)$ after nondimensionalization.

```python
class Atlas(nn.Module):
    def forward(self, state: AtlasState, dt: float) -> AtlasState:
        tokens   = {a: self.tok[a](state.fields[a], state.cond[a]) for a in agents}
        graph    = build_graph(self.layout, EDGES)              # memoized
        tokens   = self.message_passing(tokens, graph)          # typed, L layers
        tokens   = {a: self.experts[spec.expert](tokens[a], ...) for a in agents}
        _, zeta  = self.hier.pool(tokens, self.layout)
        zeta     = self.experts['rigid_body'](zeta, state.rigid, ...)   # Phase 3
        tokens   = self.hier.unpool(tokens, zeta)
        fields   = {a: state.fields[a] + dt * self.dec[a](tokens[a]) for a in agents}
        return AtlasState(fields=fields, ...)
```

For **Phase 1, every entry in `self.experts` is `IdentityExpert`** (returns its input unchanged). The conservation constraint and multi-rate stepping are absent — both are Phase 3.

## 3.7 Config (`config/atlas_0_1.yaml`)

Single source of truth. Everything numeric in this document appears there, nothing is hardcoded in modules. Note `n_layers` (per-expert backbone depth, Phase 2) and `n_mp_layers` (typed message-passing depth) are **different numbers**: MP depth defaults to **4**, the diameter of the declared agent graph ($a\!-\!b\!-\!c\!-\!d\!-\!g$ and $a\!-\!b\!-\!e\!-\!f\!-\!g$), which is the smallest depth at which every agent can reach every other exactly once.

```yaml
model:  {d_model: 256, n_layers: 6, n_mp_layers: 4, n_heads: 8, n_fourier: 6, d_edge: 16}
patch:  {gas: [8, 8], solid: [8, 4], tol_patch_diag: 1.0}
dt_model: {a: 1.0e-3, b: 1.0e-3, e: 1.0e-3, d: 5.0e-3, f: 5.0e-3, g: 5.0e-3, c: 5.0e-2}
dt_macro: 5.0e-2
edges:  {K_match: 2}
train:  {lr: 3.0e-4, weight_decay: 0.01, warmup: 500, batch_size: 8, grad_clip: 1.0}
```

---

# 4. Acceptance tests (M0, M3)

**Structural (M0):**
- `ATLAS_01_AGENTS` / `ATLAS_01_EDGES` import-time assertions all pass; a test explicitly asserts `('b','g')` and `('g','b')` are absent from the edge list.
- Every declared edge produces $\ge 4$ matched token pairs (no near-degenerate interface).
- Every boundary token on an interface is matched at least once (no orphans) — this is the test that catches a resolution mismatch across the chamber-wall interface.
- `PatchLayout` is bit-identical across two constructions with the same config (determinism).

**Forward pass (M3):**
- `Atlas(cfg)(state, dt)` runs with random-but-plausible input; output field shapes equal input field shapes for all 7 agents.
- With `IdentityExpert` everywhere and decoder weights zeroed, output state equals input state **exactly** (the increment path is a true no-op) — this verifies there is no accidental transformation hidden in pooling/unpooling.
- Backward pass produces finite gradients for every parameter; no parameter receives exactly zero gradient across a batch (an all-zero gradient means a module is disconnected from the loss — a common and silent wiring bug).
- One `AdamW` step completes and changes the loss.
- **Agent isolation test:** perturbing agent `f`'s input changes agent `a`'s output only if a path $f\to\dots\to a$ exists in the declared graph. With the declared edge list, $f$ reaches $a$ only via $e$ (i.e. $f\!-\!e\!-\!b\!-\!a$), so a **single** message-passing layer must leave `a` unchanged while three layers must not. This is the sharpest available test that the graph is doing what it claims.

  > ⚠️ **The second half of this is wrong, and the build proved it** ([[atlas-0.1-implementation-log]], 2026-08-09). Hop counts here are read off the *agent* graph, but the materialized graph carries only interface edges — a signal entering an agent on one interface can leave on another **only if some token is a boundary token of both**, and measured, $e$'s $e\!-\!f$ and $e\!-\!b$ boundary sets are disjoint. Combined with §3.6's own ordering (all MP layers first, experts once afterwards), nothing crosses an agent's interior during message passing. Measured first-reach hops from $f$: $e{=}1$, $g{=}1$, $d{=}2$, $c{=}3$, $b{=}6$, $a$ **unreached within 10**.
  >
  > **The test as built** asserts the stronger, verifiable claim: for $L\in\{1,2,3\}$ the set of agents whose output changes must *equal* the set reachable within $L$ hops by BFS over the materialized graph — catching undeclared leakage **and** a declared edge that silently does nothing. Whether Phase 2 should interleave experts with MP layers (which would make the original expectation true) is an open design decision, not a scaffold detail.

- **Bottleneck exclusion.** Isolation is a property of the **token-level graph**, not of the full forward pass: pooling to $\zeta$ and broadcasting back is a deliberate all-agent path (gravity and the rigid-body state must reach every token, per [[global-fields-and-topology-atlas-0.1]]). The isolation test runs with the bottleneck off; a companion test asserts that with it on, one layer *does* couple $f$ to $a$.
- Memory and step-time recorded at batch size 8 as a baseline for later regression comparison. *As built (CPU, Intel Core Ultra, torch 2.7.1+cpu, batch 8): forward 510 ms/step, forward+backward 1396 ms/step, process RSS 461 / 1774 MiB; 8.35 M parameters (experts absent).*

---

# 5. Pitfalls

- **Recomputing static geometry per forward pass.** `PatchLayout` and `Graph` depend only on config. Memoize.
- **Per-type `.any()` masking inside the layer loop.** Precompute type slices once per graph.
- **Absolute-position encoding assumed unnecessary.** It is necessary here; the geometry is not translation-invariant.
- **Predicting absolute state instead of an increment.** Breaks the identity default and destabilizes rollout.
- **Sharing a tokenizer across `c` and the gas agents.** Different field semantics (stress vs. momentum) and a different patch shape; keep them separate.
- **Testing only that shapes match.** Shapes matching is necessary and nearly worthless on its own — the agent-isolation test above is what actually verifies the architecture.
- **Zero-initializing the no-op heads.** The tokenizer's FiLM, the unpool and the decoder output all want to start as identity, and zeroing their last linear does that — while making the gradient of every parameter *feeding* that layer exactly zero on step 0, which is indistinguishable from a module not wired to the loss. Use near-zero ($\mathcal N(0,10^{-3})$) so the M3 gradient test can still do its job.
- **Closed-set domain predicates on a slanted wall.** A cell centre can land exactly on $h_{\text{in}}(z)$ (e.g. $z{=}0.31875$, $\lvert y\rvert{=}0.08875$), belonging to both the gas agent and the shell. Gas-cavity masks must be strictly interior; the wall belongs to the shell.

---

# 6. As-built deviations (2026-08-09)

Full root-cause records in [[atlas-0.1-implementation-log]]. Summary:

| # | Deviation | Why |
|---|---|---|
| 1 | Isolation test rewritten as a graph-BFS equality (§4) | The agent-hop premise is false at token level |
| 2 | $d\!-\!g$ curve narrowed to $0.6<\lvert y\rvert\le1.5$ | The plan's wider span is not on `g`'s boundary — it faces `f`. Exposes an **undeclared $d\!-\!f$ interface** (also unlisted: $a\!-\!c$, and the shell's aft face against `f`). Adding an edge is a design decision, so none was added |
| 3 | One extra active-mask channel per agent (§3.1) | `b`/`e` are bounding-box grids with a wall inside them |
| 4 | `n_mp_layers: 4` added, distinct from `n_layers: 6` (§3.7) | MP depth was unspecified; 4 = declared-graph diameter |
| 5 | Package is `src/atlas/`, not top-level `atlas/` | The repo uses a `src/` layout; still parallel to `src/noether11/`, still no shared imports |
| 6 | **Agents `b` and `e` are body-fitted, not bounding boxes with a masked wall** (revised by Phase 0) | A masked bounding box gives the Phase-0 finite-volume solver a *staircase* nozzle, which cannot meet an area–Mach oracle at any resolution. Token counts unchanged (140, 180); the active-mask channel of deviation 3 is now all-ones for these two, and open item 6 (ghost tokens) is closed |
| 7 | **`geometry/domains.py` derives the model's cells from `solvers/grid.build_blocks`** | Two independent derivations of the same grid drifted 1.7 mm on an 8 mm shell at the throat kink. One derivation, one answer — "no interpolation between solver and model" is now true by construction |

---

## See Also

- [[00-atlas-0.1-implementation-plan]] — global conventions
- [[graph-tokenizer-atlas-0.1]], [[edge-generation-atlas-0.1]], [[unet-hierarchy-atlas-0.1]] — design authority
- [[impl-atlas-0.1-phase2-experts]] — replaces the identity experts
- [[impl-atlas-0.1-phase3-integration]] — adds conservation and multi-rate stepping
