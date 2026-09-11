# The Decomposition Axis, Per Region — W171, and the carrier that had to be declared

**Type:** Concept page — **schema change**, with its control and one correction (folder: `Atlas 0.1/common/`)
**Status:** built 2026-09-10, Tier 44. `atlas/graph.py`, `atlas/compiler.py`, `atlas/scheme.py`, `scripts/w171_region_axis.py`, `out/w171b/w171b.json`, `tests/test_tier44_region_axis.py`. Worklist rows **W171** (closed, part 1), **W188**, **W189**.
**Related:** [[gap-worklist]] · [[case-study-ladder-to-f1]] · [[f1-pathmap-and-end-goal]] · [[atlas-implementation]] · [[master-error-bound]] · [[case-study-cooling-loop-atlas-0.1]] · [[case-study-powertrain-atlas-0.1]] · [[case-study-wing-fsi-atlas-0.1]] · [[end-to-end-architecture-spec]]

---

# 0. The result, in one paragraph

**Rung 9's blocker is removed, and the way it was removed is not the way it was scoped.** `CaseGraph.decomposition` was one field for the whole graph, and a vehicle is a fluid tiling and two circuits at once. Tier 40 established that the carrier is the **connection** and not the agent, and proposed *deriving* the axis from `Connection.geometrically_coincident` — measured on four graphs, with no exception. **Censused over all eleven constructible case graphs, per seam, that derivation does not separate the axis**: the pair $(\text{same family},\ \text{coincident})$ appears under *both* axes, because coincidence is a property of the two sides' **discretizations** and the axis is a property of their **domains**. So the axis is now **declared** on the connection — `Connection.cut_axis`, defaulting to `None` = *inherit the graph's* — which leaves every graph written before W171 with exactly one axis, equal to the one it declared, and `D_decomposition` unchanged in its artifact. On top of that, seven rule sites read the region their subject is in; the three that used to return early on the wrong axis **having emitted nothing** now scope to the overlapping regions and say so when there are none. Measured on a union graph — a coolant circuit and a fluid tiling in one compile — all three emitted nothing before and all three speak after.

**Part 2 is named and is not done**: `partition_of_unity`, `overlap` and `overlap_cells` are still one object per graph, and that boundary is now a measurement rather than a claim (§6). **W172 was not started.**

---

# 1. What the row said, and what was left of it

[[gap-worklist]] **W171**, opened at Tier 39:

> `decomposition` is ONE field for the whole graph, and rung 9's graph does not have one axis. Measured: `front_wing` is `OVERLAPPING` with a partition of unity, and `cooling_loop` and `powertrain` are both `NON_OVERLAPPING` with none. **Five rules branch on the axis and return early on the wrong one**, so a union graph would run the tiling's rules over the circuits or the circuits' over the tiling — and the failure is silent in the second direction, because a rule that returns early emits nothing.

Tier 40 scoped it and corrected it twice before anything was built:

- **The count is wrong in both numbers.** **Seven** functions read the axis, not five, and **three** return early emitting nothing — `_halo_rule`, `_r12_conservative_assembly`, `_w49_sigma_branch`. The other four emit on both branches, so on the wrong axis they say the wrong thing **loudly**, which is the cheaper failure and not the one the row is about.
- **The row's own `[AI Inference]` — that the natural carrier is the agent — is FALSE.** A cut is a **relation among several agents**, not a property of one: two agents of one family cut two different ways is one fact about one region and is unrepresentable on either agent; and an agent that is *not* cut has no axis at all.

Both of those stand and neither is re-litigated here. What did not survive is Tier 40's third finding.

---

# 2. The correction: the carrier is the connection, and the field was not already there

## 2.1 What was proposed

> **The carrier that works is the CONNECTION, and the field is already declared.** `Connection.geometrically_coincident` separates the axis **exactly**, on four graphs and with no exception: `front_wing` $7$ non-coincident and $2$ coincident; `wing_fsi` $7$ and $1$; `cooling_loop` $0$ and $5$; `powertrain` $0$ and $5$. Two overlapping graphs and two non-overlapping ones, which is this vault's own two-unrelated-graphs bar, met twice.

Every number in that paragraph is correct. The conclusion drawn from it is not.

## 2.2 The census, over eleven graphs instead of four

Per seam, over every constructible case graph, tabulating $(\text{same family},\ \text{coincident}) \mapsto \{\text{declared axes seen}\}$:

| same family | coincident | declared axes seen | |
|---|---|---|---|
| True | False | `{overlapping}` | separates |
| **True** | **True** | **`{overlapping, non-overlapping}`** | **ambiguous** |
| **False** | **True** | **`{overlapping, non-overlapping}`** | **ambiguous** |

**Two of the three populated cells are ambiguous, so no ordering of the clauses rescues it.** The graphs that break it are the two the four-graph sample did not contain:

- **`window_ns`** — four `reference.WindowNS` windows, `OVERLAPPING`, $21$ overlap cells, a partition of unity, and **all four seams declare `geometrically_coincident=True`**.
- **`wind_farm_real`** — six windows and two disks, `OVERLAPPING`, and **all fifteen seams coincident**.

Both are right to. `wind_farm`'s own source says so where it declares them: *"The tiles coincide, which satisfies E2's coincidence half and does NOT satisfy the amended connection rule."*

## 2.3 Why, in one sentence

**Coincidence is a property of the two sides' discretizations; the axis is a property of their domains.** E2's coincidence half asks whether the two interface meshes match at $\Gamma$. The decomposition axis asks whether the two subdomains *share cells*. A tiling with an overlap can perfectly well present matching rings on its artificial faces — that is exactly what `window_ns` does — and a substructuring cut can be non-conforming. The two questions are independent, and a sample of four happened not to contain a case that separates them.

**Nor does anything else derived separate it.** `partition_of_unity` and `overlap_cells` were the obvious fallbacks and they fail on a third graph: **`rocket` is `OVERLAPPING` with no partition of unity and no `overlap_cells`**.

## 2.4 So it is declared

```
Connection.cut_axis: Decomposition | None = None
```

`None` — the default, and what every graph written before 2026-09-10 means — **inherits the graph's own `decomposition`**. Naming an axis says *this seam is a cut of this kind*, which is what a union graph needs. Like `geometrically_coincident`, nothing checks it; a region whose own seams disagree is reported by `CaseGraph.region_axis_conflicts` rather than resolved, because one region cut two ways is a declaration defect and the compiler should say so rather than pick.

> **This is the second time W171's proposed carrier has been falsified by widening the sample.** Tier 40 falsified the row's *agent* inference by checking it; Tier 44 falsifies Tier 40's *derivability* finding the same way. The pattern is worth naming: both were measured, both were right on what they were measured on, and both generalised one population too far. **W188.**

---

# 3. The derivation, and the control

Four methods on `CaseGraph`, all derived:

$$\texttt{seam\_axis}(c) = \begin{cases} \texttt{None} & \text{families differ: } \Gamma \text{ is physical, not a cut} \\ c.\texttt{cut\_axis} & \text{declared} \\ \texttt{graph.decomposition} & \text{inherited} \end{cases}$$

A **region** is a `governing_family` with more than one agent in the graph — exactly `_decomposition_cuts`' cut set (W114), grouped instead of flattened. `region_axes()` gives one axis per region, from that region's own intra-family seams. `agent_axis(id)` is the axis of the region an agent is in, `None` when it is the sole agent of its family. `decomposition_axes()` is the set.

## 3.1 The control comes first, and it is the whole safety argument

| case | declared | regions derived | one axis? | reproduces |
|---|---|---|---|---|
| `brake_thermal` | non-overlapping | — (no cut region) | ✔ | ✔ |
| `cooling_loop` | non-overlapping | `incompressible-thermal-transport-1d: non-overlapping` | ✔ | ✔ |
| `powertrain` | non-overlapping | `lumped-dc-circuit: non-overlapping` | ✔ | ✔ |
| `rocket` | overlapping | 2 regions, both overlapping | ✔ | ✔ |
| `thermal_seam` | non-overlapping | — | ✔ | ✔ |
| `thermal_strain` | overlapping | — | ✔ | ✔ |
| `wind_farm` | overlapping | `incompressible-navier-stokes-2d: overlapping` | ✔ | ✔ |

**Seven graphs, zero disagreements**, and `D_decomposition` comes out of the artifact unchanged with the new `D_decomposition_axes` a one-element list containing it. A change that moved an axis on a graph nobody was asking about would not be a schema extension; it would be a behaviour change, and this is what says it is not.

---

# 4. The union: two axes in one compile

The smallest honest version of rung 9's shape — `cooling_loop`'s four coolant legs (a `NON_OVERLAPPING` region) beside a two-window fluid tiling declaring `cut_axis=OVERLAPPING`:

| | value |
|---|---|
| agents / seams | $7$ / $6$ |
| what the **one field** can say | `non-overlapping` |
| what the **regions** say | `{thermal-transport: non-overlapping, navier-stokes: overlapping}` |
| `D_decomposition` | `non-overlapping` — the scheme's own axis |
| `D_decomposition_axes` | `['non-overlapping', 'overlapping']` |

**The joining seams do not exist.** The union is disjoint, which is enough to put two axes in front of the rules and is *not* enough for W172 — that row needs the seams that would join the subsystems, and §7 says so.

**The scheme's own axis stays one value, deliberately.** The interface problem is one problem however many regions a graph has, and `Scheme.decomposition` is the scheme's axis. What was missing is the **set**, and an artifact reporting one axis for a two-axis graph is reporting a graph that does not exist. Both are emitted; `L5/W171/axes` names them in the decision record.

---

# 5. The seven rule sites

## 5.1 The three that were silent

Each opened with `if <the graph's axis> is not OVERLAPPING: return`. Each now asks `ctx.regions_on(OVERLAPPING)`, scopes its subject to those regions' agents, and **says so** when there are none.

| rule | before, on the union | after |
|---|---|---|
| `L2/R10/halo` | **nothing** | `admit-uncertified` on the tiling's agents, by name |
| `L6/R12` | **nothing** | `admit` (§6) |
| `L6/W49` | **nothing** | `admit-uncertified` (§6) |

*"A rule that quietly stops looking is the same failure in the other direction, and it is the one that is not self-announcing"* — `_halo_rule`'s own docstring, about the agents it excludes. The change applies that sentence to the axis.

## 5.2 The four that were loud

- **`_cut_policy`** — `L2/C2` is the overlapping cut criterion and `L2/C3` the substructuring one, and **a union graph is on both**. It ran once and emitted one of them; it now runs once per axis present, each pass naming its regions. On the union both `L2/C2` and `L2/C3/W57` appear.
- **the cross-point block** — a cross-point among a tiling's agents is handled by the partition of unity; one among a circuit's is refused (W162's own finding: *a tiling's triangle and a circuit's triangle are the same object in the adjacency and different objects in space*). Each vertex is now judged by the axis of the region its agents are in, and **a vertex spanning two regions is treated as non-overlapping** — the conservative direction, because it can be refused and inspected rather than silently admitted.
- **`_decide_rung`** and **`_l5_l7_scheme`** stay graph-global, on purpose. R2's lift to the non-overlapping view is a property of the **interface problem**, which is one problem however many regions there are — so `ctx.region_axis` respects the lift and returns non-overlapping everywhere under it, which is what keeps every lifted graph behaving exactly as before.

## 5.3 One helper, so seven sites cannot drift

`_Context.region_axis`, `regions_on` and `axis_summary`. This is W136's move one object along: `_decomposition_cuts` was factored out so that two rules resting on one premise could not come to disagree about it, and the axis now has the same treatment for seven.

---

# 6. Where part 1 stops, measured rather than claimed

Tier 41 priced part 1 and said `partition_of_unity`, `overlap` and `overlap_cells` *"stay graph-global and would each need the same treatment before a union graph could carry a tiling **and** two circuits"*. That boundary is now a measurement:

| union variant | `R10/halo` | `R12` | `W49` |
|---|---|---|---|
| **no partition of unity** | speaks | **silent** | **silent** |
| **with a partition of unity** | speaks | speaks | speaks |

`_r12_conservative_assembly` and `_w49_sigma_branch` are both gated on `graph.partition_of_unity` **before** they reach any axis. So with a partition present, part 1's change is **sufficient for all three rules**; what remains is that the partition is one object per graph, so a partition covering only the tiling still has to be declared as the whole graph's. **That is part 2, and it is exactly where Tier 41 drew it.** **W189.**

---

# 7. What this does NOT do

- **W172 was not started.** The union here is **disjoint** — the seams that would *join* the subsystems do not exist, so nothing here re-runs the $O(K)$ count across them. Tier 39's zero-per-pair count remains what it was: strong evidence about the **algebra** and no evidence about the **integration cost**, and it must not be quoted without that caveat.
- **It does not build rung 9's graph.** An $18$-agent $5$-family union with three joining seams is what rung 9 needs; this is a $7$-agent two-region fixture built to put two axes in front of the rules.
- **It does not check the declaration.** `cut_axis` is a label, exactly as `geometrically_coincident` is. `region_axis_conflicts` reports a region whose seams disagree and resolves nothing.
- **It does not make `partition_of_unity` per-region** (§6), and it does not touch `overlap` or `overlap_cells`.
- **It does not change any verdict on any existing graph** (§3.1), and the suite is what says so rather than this sentence.

---

## See Also

- [[gap-worklist]] — W171 (closed, part 1), W188 (a measured carrier falsified twice by widening the sample), W189 (part 2), W172 (not started)
- [[case-study-ladder-to-f1]] — §18, and the critical path rung 9 sits on
- [[f1-pathmap-and-end-goal]] — §3's rung 9, *"the rung that decides everything"*
- [[atlas-implementation]] — the nine layers this changes seven sites of
- [[case-study-cooling-loop-atlas-0.1]] · [[case-study-powertrain-atlas-0.1]] — the two circuits the union's non-overlapping region comes from
- [[case-study-wing-fsi-atlas-0.1]] — W114, whose same-family predicate the region grouping is
- [[master-error-bound]] — §4 and §4.1, the two $\sigma$ branches `_w49_sigma_branch` chooses between
- [[end-to-end-architecture-spec]] — §2, the decomposition layer this is the schema of
