# Phase 5 — Expansion to a Second Scenario

**Type:** Implementation spec — agent task (folder: Atlas 0.1 / Atlas 0.1 implementation)
**Phase of:** [[00-atlas-0.1-implementation-plan]]. **Depends on:** [[impl-atlas-0.1-phase4-validation]] (**M7 passed, and ablations A1/A3/A4 favouring the full architecture**). **Unblocks:** Atlas 0.2.
**Design pages:** [[pfm-purpose-and-direction]], [[edge-generation-atlas-0.1]], [[expert-library-atlas-0.1]], [[unet-hierarchy-atlas-0.1]], [[incremental-transfer-roadmap]].

---

# 1. Intuition

Phase 4 tested whether Atlas is **wired correctly**. Phase 5 tests whether it is **a foundation model**, and those are different claims. A single-scenario system that works is a well-engineered surrogate; the foundation-model claim requires that adding a second, structurally different scenario is *cheaper than building it from scratch* and does not break the first.

That gives Phase 5 three concrete jobs, in order of importance:

1. **Reuse.** Does `external_flow`, trained on rocket plume and atmosphere, transfer to turbine wake aerodynamics with only light fine-tuning? If yes, the expert library is accumulating capability. If no — if every scenario needs its own experts trained from scratch — then Atlas is a framework for organizing bespoke models, not a foundation model, and that should be said plainly.
2. **Non-regression.** Adding scenario 2 must not degrade scenario 1. This is the catastrophic-forgetting question, and the MoE structure's main claimed advantage ([[mixture-of-experts]]: "modular update — new domains can be added without disrupting existing expert knowledge") is exactly this. Phase 5 is where that claim gets its first test.
3. **Unlocking the deferred mechanisms.** RL edge instantiation ([[edge-generation-atlas-0.1]] Mechanism B) was deferred with a stated trigger: *not before scenario 2 exists*, because its justification — learning regime-pairing from past experience — requires more than one scenario's worth of experience. Phase 5 is that trigger. Same for RL expert gating and the deeper U-Net hierarchy.

The scenario chosen is the **offshore wind farm** from [[pfm-purpose-and-direction]], and it is chosen deliberately: it maximally reuses `external_flow` (wake aerodynamics is external compressible-to-incompressible flow) and `thermostruct` (tower and nacelle), while introducing exactly one genuinely new governing family (electromagnetics) and one genuinely new *structural* challenge — a **long-range, direction-dependent edge** that is not physical adjacency. That last property is what the rocket scenario could not test at all.

---

# 2. Theory

## 2.1 The wind-farm agent graph

Per-turbine agents, plus shared environment agents:

| ID | Agent | Governing family | Expert |
|---|---|---|---|
| `r` | rotor / blades | fluid–structure interaction | `external_flow` + `thermostruct` (contested) |
| `n` | nacelle / generator | electromagnetics + thermal | **new: `electromagnetic`** |
| `t` | tower | structural dynamics | `thermostruct` (extended) |
| `fd` | foundation | wave–structure interaction | **new: `wave_structure`** |
| `w` | near wake | external flow | `external_flow` |
| `amb` | ambient / far wake | external flow | `external_flow` |
| `sea` | sea surface | free-surface wave | `wave_structure` |

Edges (per turbine): $r\!-\!w$ (fluid, stress), $r\!-\!n$ (stress, torque), $n\!-\!t$ (stress, heat), $t\!-\!fd$ (stress), $fd\!-\!sea$ (fluid, stress), $w\!-\!amb$ (conservation).

**And the new kind of edge:** $w_i\!-\!r_j$ — turbine $i$'s wake to turbine $j$'s rotor, for turbines $j$ downwind of $i$. This edge:
- is **not adjacency** — the turbines are hundreds of metres apart;
- has **direction-dependent existence** — it exists only when the wind blows from $i$ toward $j$, and the wind direction changes hour to hour;
- **coexists with a purely local edge on the same agent** — the same tower's fatigue accumulation ($t$ internal) has no cross-turbine coupling at all.

This is precisely the case [[pfm-purpose-and-direction]] identified as breaking "proximity = adjacency," and it is the strongest argument for RL edge instantiation: whether the $w_i\!-\!r_j$ edge should exist is a *discrete, state-dependent* decision, which is what reward-driven search handles better than a differentiable relaxation.

## 2.2 The timescale problem gets worse

The rocket spanned $10^{-3}$ s to $60$ s — four orders of magnitude. The wind farm spans seconds (wake shedding) to **decades** (fatigue, corrosion) — nine orders. Multi-rate subcycling as implemented in [[impl-atlas-0.1-phase3-integration]] does not extend to $10^9$ substeps; it needs a genuinely different mechanism for the slow variables:

**Slow-variable accumulation, not integration.** Fatigue damage is not stepped; it is *accumulated* from a statistical summary of the fast dynamics. Miner's rule:

$$D=\sum_i \frac{n_i}{N_i(\sigma_i)},\qquad \text{failure at } D\to1$$

where $n_i$ counts stress cycles at amplitude $\sigma_i$ (obtained from rainflow counting of the fast structural response) and $N_i$ comes from the S–N curve. The correct architecture is a **fast/slow split**: the fast experts run over a representative window, their output is summarized into cycle statistics, and the slow variable takes one large step from that summary. This is a new mechanism, not a parameter change, and it should be scoped as such.

## 2.3 What "reuse" concretely means

Three levels, in increasing order of what they would demonstrate:

| Level | Test | Claim supported |
|---|---|---|
| **R1** | `external_flow` fine-tuned (LoRA) on wake data beats the same architecture trained from scratch on wake data | Pretrained representations transfer |
| **R2** | `external_flow` **frozen**, adapters only, reaches acceptable wake accuracy | The representation is genuinely general, not just a good initialization |
| **R3** | `external_flow` zero-shot (no wake training at all) produces physically plausible wake structure | In-context generality — the actual foundation-model bar ([[transfer-learning-fine-tuning]] Level 4) |

Expect R1. R2 would be a strong result. R3 would be surprising at this scale and should be treated with suspicion if observed — check for data leakage first.

---

# 3. Implementation

## 3.1 Order of work

1. **Regression suite first.** Before any wind-farm code, freeze the Phase 4 rocket results as a regression baseline (`eval/regression/rocket_m7.json`) with tolerances. Every subsequent change runs against it. Without this, forgetting is discovered late and attributed to the wrong cause.
2. **Data generation for scenario 2** — same discipline as [[impl-atlas-0.1-phase0-scope-and-data]]: classical solvers validated against analytic oracles first (actuator-disc/blade-element theory for the rotor, linear wave theory for the foundation, a lumped-parameter generator model for the nacelle).
3. **Reuse experiments R1/R2/R3** on `external_flow` alone, before touching the graph. This isolates the transfer question from the integration question.
4. **New experts** (`electromagnetic`, `wave_structure`) built to the same `Expert` interface from [[impl-atlas-0.1-phase2-experts]] §3.1 and solo-gated the same way.
5. **Graph extension**, including the long-range $w_i\!-\!r_j$ edge, still declared (not RL) at first — establish a working baseline before introducing a learned edge mechanism.
6. **RL edge instantiation** (Mechanism B), now that two scenarios exist. See §3.3.
7. **Hierarchy depth**: only if agent count demands it. A 3-turbine farm has ~21 agents; the intermediate pooling levels from [[unet-hierarchy-atlas-0.1]] become justifiable somewhere above that, and the moon-base scenario is where they clearly are.

## 3.2 Non-regression protocol

After every stage above, re-run the Phase 4 rocket evaluation:

- Per-agent $\varepsilon$ at the 60 s horizon must not degrade by more than **10% relative** to the frozen baseline.
- All four physical-milestone checks must still pass.
- Conservation residuals unchanged in order of magnitude.

If regression occurs, the mitigation order is: (a) freeze the shared experts and give scenario 2 its own adapters; (b) if that is insufficient, give scenario 2 its own expert instance entirely and record that as **evidence against** the modular-update claim in [[mixture-of-experts]]. Do not fix a regression by loosening the tolerance.

## 3.3 RL edge instantiation — scope

Only now does this have anything to learn from. Minimal viable form:

- **Action space:** for each candidate agent pair not already declared-adjacent, a binary "instantiate edge" decision plus a type assignment.
- **State:** per-agent conditioning summaries (dimensionless numbers), relative geometry, and the current global condition (wind direction, flight altitude).
- **Reward:** downstream rollout accuracy minus conservation residual, minus a sparsity penalty on edge count (otherwise the policy learns to connect everything, which A4 in [[impl-atlas-0.1-phase4-validation]] already measures as a baseline).
- **Constraint from [[edge-generation-atlas-0.1]]:** the proposed edge set must be *stable enough episode-to-episode* for any pooling hierarchy above it to be well-defined. Train the edge policy in a **separate, earlier phase** and freeze it before training hierarchical pooling on top. An edge set that changes every episode gives the supernode clustering nothing consistent to learn.

**Success criterion:** the RL policy rediscovers the hand-declared rocket edge list (including the *absence* of $b\!-\!g$ — see [[edge-generation-atlas-0.1]]) and independently discovers the direction-dependent $w_i\!-\!r_j$ wake edge. Rediscovering a known-correct graph is the honest test; discovering a novel edge is the payoff.

## 3.4 Versioning

Scenario 2 integration marks **Atlas 0.2**. Create `concepts/Atlas 0.2/` mirroring the 0.1 folder structure, with 0.1 frozen as the reference implementation — the same pattern the parallel track used for Noether 1.0 → 1.1. Page filenames carry the version suffix (`*-atlas-0.2.md`) so both versions' pages coexist without link collisions, per this vault's bare-link convention.

---

# 4. Acceptance criteria

- Regression suite green after every stage.
- R1 demonstrated (minimum bar); R2 attempted and reported either way.
- Both new experts pass solo gates before integration.
- Long-range $w_i\!-\!r_j$ edge produces measurable downstream-turbine wake deficit matching the classical solver within 15%.
- Fast/slow split implemented for fatigue; damage accumulation over a simulated year within 20% of a direct (expensive) reference computation on a short window extrapolated.
- RL edge policy rediscovers the rocket edge list at ≥ 90% precision/recall, or the attempt is documented as failed with the failure mode identified.

---

# 5. Pitfalls

- **Building scenario 2 before freezing the scenario 1 regression baseline.** Forgetting then looks like a wind-farm bug.
- **Testing transfer and integration simultaneously.** Isolate R1/R2/R3 on the expert alone first.
- **Extending multi-rate subcycling to decades.** It does not extend; fatigue needs the fast/slow accumulation split, which is a different mechanism.
- **Introducing RL edges and deeper hierarchy in the same change.** The ordering dependency in §3.3 is real — an unstable edge set makes hierarchical pooling untrainable, and debugging both at once is intractable.
- **Loosening regression tolerances to accommodate scenario 2.** The regression *is* the modular-update claim; weakening the test discards the evidence.
- **Treating R1 success as the foundation-model claim.** R1 is "pretraining helps," which is a much weaker statement than "the model generalizes."

---

## See Also

- [[pfm-purpose-and-direction]] — the five worked examples; the wind farm's long-range-edge property
- [[edge-generation-atlas-0.1]] — Mechanism B and its stability constraint
- [[unet-hierarchy-atlas-0.1]] — when deeper pooling becomes justified
- [[expert-library-atlas-0.1]] — the `Expert` interface new experts must implement
- [[mixture-of-experts]] — the modular-update claim this phase tests
- [[incremental-transfer-roadmap]] — the transfer-level hierarchy R1/R2/R3 map onto
