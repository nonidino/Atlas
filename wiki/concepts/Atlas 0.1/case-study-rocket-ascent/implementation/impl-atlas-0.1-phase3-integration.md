# Phase 3 — Integration: Routing, Multi-Rate Stepping, Conservation

**Type:** Implementation spec — agent task (folder: Atlas 0.1 / Atlas 0.1 implementation)
**Phase of:** [[00-atlas-0.1-implementation-plan]]. **Depends on:** [[impl-atlas-0.1-phase1-scaffold]], [[impl-atlas-0.1-phase2-experts]] (**all three gates passed**). **Unblocks:** [[impl-atlas-0.1-phase4-validation]].
**Design pages:** [[conservation-as-constraint-atlas-0.1]] (authoritative), [[global-fields-and-topology-atlas-0.1]], [[unet-hierarchy-atlas-0.1]], [[expert-library-atlas-0.1]].
**Milestone:** M5 — coupled forward pass with active conservation constraint.

---

# 1. Intuition

Phase 2 produced three specialists that each work alone. This phase makes them work *together*, and everything hard about Atlas lives here. Three problems, each with its own mechanism:

**Problem 1 — who computes what.** Most agents map to exactly one expert, so dispatch is a lookup table. But some *edges* are contested: $c\!-\!d$ carries `stress` and `heat` (which belong to the thermostruct expert) alongside `fluid` (which belongs to external_flow). Both experts have a legitimate claim to that interface, and their contributions must be blended. That blend is the one place a learned gate does real work in 0.1.

**Problem 2 — different clocks.** Combustion resolves at 1 ms; the structural shell responds over 50 ms; the trajectory integrates over the whole ascent. Forcing everything onto the fastest clock would waste ~98% of the compute on agents whose state barely changed. Subcycling solves this, but introduces a question with a real answer: *how often should two agents exchange information across their shared edge?* The answer used here is that **the exchange cadence is set by the coupling's own timescale** — concretely, the slower of the two agents' timesteps. That rule is not a heuristic pulled from nowhere; it falls out of the observation that a coupling cannot carry information faster than its slowest participant can respond.

**Problem 3 — the seams.** Two experts meeting at an interface will not spontaneously agree about how much mass crosses it. Each is separately trained and separately wrong by a little. Left alone, those disagreements act as sources and sinks: mass appears at the nozzle exit, energy vanishes into the wake, and a long rollout drifts into nonsense. The conservation constraint ([[conservation-as-constraint-atlas-0.1]]) is what stops this. Note the philosophy: Atlas does not *discover* that mass is conserved across $e\!-\!f$ — the edge was **declared** conservative, so the model is *told*, and the constraint enforces it. This is the deliberate inverse of the parallel Noether track's approach ([[discovered-conservation-1.1]]) and is the reason this architecture is not named after Noether.

---

# 2. Theory

## 2.1 Routing

**Agent-level dispatch is a static lookup**, determined by `AgentSpec.expert`:

$$a,b,e\to\texttt{reacting\_flow};\quad d,f,g\to\texttt{external\_flow};\quad c\to\texttt{thermostruct};\quad \text{bottleneck}\to\texttt{rigid\_body}$$

There is nothing to learn: the graph is declared, so the assignment is declared. Anything more elaborate would be machinery without a job.

**Edge-channel blending is learned.** At a *contested* edge — one whose declared types span more than one expert's domain — each claiming expert produces a message and an MLP gate blends them per token:

$$m_i=\sum_{k\in\text{claimants}} g_k(h_i,\mathbf z_{\text{cond}})\cdot m_i^{(k)},\qquad \mathbf g=\mathrm{softmax}\!\left(\mathrm{MLP}\!\left[h_i\,\|\,\mathbf z_{\text{cond}}\right]\right)$$

In Atlas 0.1 exactly one edge is contested: $c\!-\!d$ (`stress`,`heat` → thermostruct; `fluid` → external_flow). A softmax blend is used rather than a hard selection because the physical crossover there is smooth — boundary-layer pressure and wall stress vary continuously with flight condition, and this is the smooth branch of [[regime-moe-architecture]]'s finding 2. RL/discrete gating, which suits threshold-like decisions, is deferred ([[unet-hierarchy-atlas-0.1]]).

## 2.2 Multi-rate stepping

Let $\Delta t_{\text{macro}}=5\times10^{-2}$ s. Per macro-step:

| Group | $\Delta t_{\text{model}}$ | Substeps |
|---|---|---|
| `a`, `b`, `e` | $10^{-3}$ | 50 |
| `d`, `f`, `g` | $5\times10^{-3}$ | 10 |
| `c` | $5\times10^{-2}$ | 1 |
| rigid body | $5\times10^{-2}$ | 1 |

**Exchange cadence rule.** For an edge between agents $p$ and $q$:

$$\Delta t_{\text{exch}}(p,q)=\max\!\left(\Delta t_{\text{model}}^{(p)},\ \Delta t_{\text{model}}^{(q)}\right)$$

Applied to the declared edge list, this gives:

| Edge | Cadence | Interpretation |
|---|---|---|
| $a\!-\!b$, $e\!-\!b$ | $10^{-3}$ s | fast gas–gas, exchanged every substep |
| $e\!-\!f$, $d\!-\!g$, $g\!-\!f$ | $5\times10^{-3}$ s | gas–gas across the rate boundary |
| $b\!-\!c$, $c\!-\!d$ | $5\times10^{-2}$ s (= macro) | gas–solid, thermally/structurally slow |

The rule reproduces exactly the physically correct behaviour without being hand-specified per edge: the structure genuinely does not respond on a 1 ms timescale, so exchanging with it that often would be wasted computation, and the chamber genuinely does respond to the reaction zone that fast.

Between exchanges, a fast agent holds its neighbour's interface data constant (**zeroth-order hold**). Linear interpolation from the previous two exchanges is a documented upgrade; zeroth-order is correct for 0.1 and is the more conservative choice because interpolation can extrapolate a trend past a discontinuity.

**Coupling scheme:** Jacobi (all agents advance from the same synchronized state, then exchange), not Gauss–Seidel. Jacobi is order-independent, which makes results reproducible and lets agent groups run concurrently; Gauss–Seidel's slightly better stability is not worth introducing an ordering dependence into the model's semantics.

## 2.3 The conservation constraint

At a conservation-typed edge with interface normal $\mathbf n$ and normal velocity $u_n=\mathbf u\!\cdot\!\mathbf n$, the flux of the conserved vector is

$$\boldsymbol\Phi=\begin{bmatrix}\rho u_n\\ \rho u_n u+p\,n_z\\ \rho u_n v+p\,n_y\\ (E+p)u_n\end{bmatrix}$$

Both sides predict a trace; define the residual over the matched interface token pairs:

$$\mathbf r_{ij}=\boldsymbol\Phi^{\text{src}}_{i}-\boldsymbol\Phi^{\text{dst}}_{j}$$

**Upwinding the constraint.** Which side should be treated as authoritative is not arbitrary — it is set by the local characteristic direction, exactly as it would be in the classical solver:

- **Supersonic normal flow ($|u_n|>c$):** information cannot travel upstream. The **upstream** side is authoritative; the downstream side is corrected to match. This is the normal situation at $e\!-\!f$ (nozzle exit, supersonic by design).
- **Subsonic normal flow:** both sides carry information. Use the symmetric average $\bar{\boldsymbol\Phi}=\tfrac12(\boldsymbol\Phi^{\text{src}}+\boldsymbol\Phi^{\text{dst}})$ as the target. This is the normal situation at $d\!-\!g$ at low altitude.

Making the constraint's own directionality follow the physics is what stops it from injecting acausal upstream influence at a supersonic exit — a subtle but real failure mode of a naively symmetric projection.

**Hard on mass, soft on momentum and energy.** The projection modifies only the receiving side's normal velocity so that the mass flux matches exactly:

$$u_n^{\text{dst}}\ \leftarrow\ u_n^{\text{dst}}\cdot\frac{\left(\rho u_n\right)^{\text{target}}}{\left(\rho u_n\right)^{\text{dst}}}$$

while momentum and energy mismatch enter the loss as penalties:

$$\mathcal L_{\text{cons}}=\lambda_m\underbrace{\left\|r^{(\rho u_n)}\right\|^2}_{\text{also hard-projected}}+\lambda_p\left\|r^{(\text{mom})}\right\|^2+\lambda_E\left\|r^{(E)}\right\|^2$$

**Why this split rather than hard-projecting everything:** mass-flux violation is the one that compounds fastest — a small persistent mass source changes density, which changes pressure through the equation of state, which changes every other field, and the error grows secularly rather than saturating. Momentum and energy mismatches are more forgiving over the horizons Atlas 0.1 targets. Hard-projecting all four components simultaneously is also an over-determined correction on a single scalar degree of freedom per token pair and would require choosing which field absorbs the residual — a choice with no clean physical answer. Starting with the minimal hard constraint and measuring is the right order; escalate only if Phase 4 shows momentum or energy drift dominating.

**Interface integration, not evaluation.** Fluxes must be *integrated* over the interface using the per-pair length $\ell_{ij}$ stored in the edge attributes ([[impl-atlas-0.1-phase1-scaffold]] §2.2):

$$\dot m_\Gamma=\sum_{(i,j)}\left(\rho u_n\right)_{ij}\,\ell_{ij}$$

A pointwise-only match would leave the *total* flux unconstrained whenever the two sides' token resolutions differ — which they do at every gas–solid interface.

## 2.4 Global fields and the bottleneck

Per [[global-fields-and-topology-atlas-0.1]], gravity is **not an edge**. It enters the rigid-body expert directly:

$$\mathbf g(h)=-g_0\left(\frac{R_E}{R_E+h}\right)^{2}\hat{\mathbf y}$$

The bottleneck vector assembles the loads that the field agents produced, integrated over their interfaces:

$$\mathbf F_{\text{thrust}}=\sum_{(i,j)\in e\text{-}f}\left[\rho u_n\mathbf u+(p-p_\infty)\mathbf n\right]_{ij}\ell_{ij},\qquad
\mathbf F_{\text{aero}}=\sum_{(i,j)\in c\text{-}d}\left[-p\,\mathbf n+\boldsymbol\tau\!\cdot\!\mathbf n\right]_{ij}\ell_{ij}$$

then rigid-body integration runs, and the updated vehicle state (altitude, velocity, acceleration, angle of attack) is broadcast back to every token on unpool. This closes the only loop in Atlas that is genuinely global: local flow produces loads → loads move the vehicle → vehicle motion changes altitude and freestream → freestream changes local flow.

---

# 3. Implementation

## 3.1 `model/router.py`

```python
class Router(nn.Module):
    AGENT_EXPERT = {'a':'reacting_flow','b':'reacting_flow','e':'reacting_flow',
                    'd':'external_flow','f':'external_flow','g':'external_flow',
                    'c':'thermostruct'}
    CONTESTED = {('c','d'): ('thermostruct', 'external_flow')}

    def blend(self, h, cond, msgs: dict[str, Tensor]) -> Tensor:
        """softmax MLP gate over claimant experts' messages"""
```

## 3.2 `model/conservation.py`

```python
def interface_flux(state, edge, layout) -> Tensor:      # [N_pairs, 4]
def upwind_target(phi_src, phi_dst, mach_n) -> Tensor:  # supersonic -> upstream; else mean
def project_mass_flux(state_dst, target, edge) -> AtlasState   # hard, in-place on u_n
def conservation_loss(phi_src, phi_dst, weights) -> Tensor     # soft on mom/energy
```

The projection must be **differentiable** and must not produce a division by zero when $\rho u_n\to0$ (stagnation at low altitude on $d\!-\!g$). Guard with a floor: skip the projection where $|\rho u_n| < \varepsilon_{\dot m}$ and fall back to the soft penalty alone. Log how often this fires — a high rate means the interface is in a regime the hard constraint cannot serve, which is information, not just an edge case.

## 3.3 `model/atlas.py` — the multi-rate stepper

```python
def macro_step(self, state: AtlasState) -> AtlasState:
    iface = self.exchange(state, cadence='all')        # sync point

    for k in range(50):                                # a, b, e  @ 1e-3
        state = self.step_group(state, ['a','b','e'], dt=1e-3, iface=iface)
        iface = self.exchange(state, cadence=1e-3)     # a-b, e-b
        if k % 5 == 4:                                 # 5e-3 boundary
            state = self.step_group(state, ['d','f','g'], dt=5e-3, iface=iface)
            iface = self.exchange(state, cadence=5e-3) # e-f, d-g, g-f
            state = self.apply_conservation(state, ['e-f','d-g'])

    state = self.step_group(state, ['c'], dt=5e-2, iface=iface)   # b-c, c-d
    loads = self.integrate_loads(state)
    zeta  = self.hier.pool(state.tokens, self.layout)[1]
    zeta, state.rigid = self.experts['rigid_body'](zeta, state.rigid, loads, dt=5e-2)
    state.tokens = self.hier.unpool(state.tokens, zeta)
    state.cond   = self.recompute_conditioning(state)   # new altitude -> new p_inf, Ma, Re
    return state
```

**`recompute_conditioning` is load-bearing and easy to forget.** After the vehicle moves, the freestream conditions change, which changes every gas agent's Mach and Reynolds numbers, which changes the FiLM conditioning. Omitting it produces a model that flies to 20 km while still believing it is at sea level — and the symptom (slowly growing plume error) does not obviously point at its cause.

## 3.4 Joint fine-tuning (`train/train_joint.py`)

After wiring, a short joint fine-tune at low learning rate ($5\times10^{-5}$, 10% of the Phase 2 step budget) over full macro-steps, with the composed loss:

$$\mathcal L_{\text{joint}}=\sum_a w_a\left\|\hat x_a-x_a\right\|^2+\lambda_{\text{cons}}\mathcal L_{\text{cons}}+\lambda_{\text{traj}}\left\|\hat{\mathbf s}-\mathbf s\right\|^2$$

The trajectory term is what makes the loads physically meaningful rather than merely locally accurate: an expert can have excellent local pressure fields whose *integral* is wrong, and only the trajectory term penalizes that.

Freeze the donor backbones during joint fine-tuning; train only adapters, gates, and the conservation-adjacent output heads. Full joint fine-tuning of everything at once risks the composed system fixing an interface disagreement by degrading an expert that was independently validated — undoing the M4 guarantee.

---

# 4. Acceptance tests (M5)

- **Cadence rule correctness:** unit test that `Δt_exch` computed from the agent table equals the table in §2.2 for all 7 edges.
- **Mass-flux closure:** after projection, $\left|\dot m_\Gamma^{\text{src}}-\dot m_\Gamma^{\text{dst}}\right|/\dot m_\Gamma^{\text{src}} < 10^{-6}$ at $e\!-\!f$ and $d\!-\!g$.
- **Upwind switching:** with a synthetic supersonic $e\!-\!f$ state the target equals the upstream flux exactly; with a subsonic state it equals the mean. Both asserted.
- **Global mass budget over 100 macro-steps:** $\left|\Delta m_{\text{system}}+\int\dot m_{\text{exit}}dt\right|/m_0 < 10^{-3}$ — the same budget the classical corpus satisfies ([[impl-atlas-0.1-phase0-scope-and-data]] M2).
- **Conditioning propagation:** after 100 macro-steps of powered ascent, each gas agent's stored $p_\infty$ matches `atmosphere(h(t))` to < 0.1%. (Catches the omitted-`recompute_conditioning` bug directly.)
- **Gate sanity:** the $c\!-\!d$ blend weights are not saturated at 0/1 across the test set — a saturated gate means the blend is doing nothing and one expert is being silently ignored.
- **Determinism:** two runs from the same seed produce bit-identical trajectories.
- **No-regression:** each expert's solo M4 metric, re-measured after joint fine-tuning, has not degraded by more than 10% relative.

---

# 5. Pitfalls

- **Forgetting `recompute_conditioning`.** Silent, slow, and hard to trace.
- **Symmetric conservation targets at a supersonic interface.** Injects acausal upstream influence.
- **Pointwise flux matching without $\ell_{ij}$ weighting.** Leaves total flux unconstrained across resolution-mismatched interfaces.
- **Division by zero in the mass projection** at stagnation. Guard and log.
- **Gauss–Seidel coupling.** Introduces an ordering dependence into the model's semantics; use Jacobi.
- **Unfreezing everything during joint fine-tune.** Silently undoes M4.
- **Assuming the conservation constraint fixes bad experts.** It enforces agreement at the seam, not correctness in the interior. Two experts can agree perfectly on a wrong flux.

---

## See Also

- [[conservation-as-constraint-atlas-0.1]] — why imposed, not discovered
- [[global-fields-and-topology-atlas-0.1]] — gravity as a non-edge; the staging case (out of scope for 0.1)
- [[unet-hierarchy-atlas-0.1]] — pooling/unpooling and the gating argument
- [[impl-atlas-0.1-phase4-validation]] — the experiment this phase makes runnable
