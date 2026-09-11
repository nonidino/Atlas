# The Overlapping Mechanism, Per Region — W189, and the partition that could not be derived

**Type:** Concept page — **schema change**, with its control, a census and a union on a real tiling (folder: `Atlas 0.1/common/`)
**Status:** built 2026-09-10, Tier 45. `atlas/graph.py`, `atlas/compiler.py`, `atlas/verdict.py`, `atlas/assembly.py`, `atlas/scheme.py`, `atlas/emit.py`, `atlas/solve.py`, `scripts/w189_artifact_control.py`, `scripts/w189_region_assembly.py`, `out/w189/`, `tests/test_tier45_region_assembly.py`. Worklist rows **W189** (closed), **W190**, **W191**, **W192**, **W193** (opened), **W172** (annotated, not started).
**Related:** [[per-region-decomposition-axis]] · [[gap-worklist]] · [[case-study-ladder-to-f1]] · [[f1-pathmap-and-end-goal]] · [[atlas-implementation]] · [[master-error-bound]] · [[end-to-end-architecture-spec]] · [[case-study-cooling-loop-atlas-0.1]] · [[case-study-powertrain-atlas-0.1]] · [[tier0-measurements]]

---

# 0. The result, in one paragraph

**Rung 9's schema is complete, and the object it needed is one partition per OVERLAPPING region — not one per region.** `CaseGraph.partition_of_unity`, `overlap` and `overlap_cells` each accept a map keyed by region, and a single object still means the graph's own. The shape was chosen on a census of **all forty constructible graphs, taken before any code changed**: every one of the 22 that carries a partition has exactly one overlapping region, and the only graph with two overlapping regions (`rocket`) carries none — so the map is sparse, and a circuit files nothing. The key had to be **declared**, because the census rules out deriving it in either direction: in 12 of the 22 the family region also holds agents the partition does not blend, and in 2 the partition's subdomain names are not agent ids at all. **The control is the claim**: forty artifacts byte-identical before and after the change, against a repeat floor of forty out of forty under two hash seeds. On a union carrying `window_ns`'s real four-window tiling beside **both** circuits — fourteen agents, three regions — every assembly and halo decision names the fluid region and none names a circuit, the substructuring criterion names each circuit, and `window_ns` declared both ways reaches the same verdict under the same rules. The union also measured two things W172 will have to price: **joining graphs moved `powertrain`'s rotor into the fluid region**, and a named cross-point is still judged by the one axis field a graph has left.

---

# 1. What the row said, and what Tier 44 left

[[gap-worklist]] **W189**, opened at Tier 44 as W171's part 2:

> Three objects are still one per graph, and the boundary is now measured rather than claimed. `_r12_conservative_assembly` and `_w49_sigma_branch` are gated on `graph.partition_of_unity` **before** they reach any axis … what remains is that a partition covering only the tiling still has to be declared as the **whole graph's**.

Tier 44's measurement, from [[per-region-decomposition-axis]] §6:

| union variant | `L2/R10/halo` | `L6/R12` | `L6/W49` |
|---|---|---|---|
| no partition of unity | speaks | **silent** | **silent** |
| with a partition of unity | speaks | speaks | speaks |

The row carried one question it said had not been asked: **is one partition per region even the right object?** A vehicle has few overlapping regions and many non-overlapping ones. That question is §2, and it was answered by counting before anything was built — [[gap-worklist]] W188's standing instruction, applied: *price a proposed derivation against every constructible graph before building on it.*

**Intuitively.** A partition of unity is a recipe for blending overlapping local answers into one field: near the middle of a window its own answer counts, near the edge its neighbour's does, and the weights add to one everywhere. A recipe belongs to *the thing being blended* — the airflow the windows tile — and not to the vehicle. A coolant loop is cut into legs that meet at planes and share no cells, so there is nothing for it to blend at all. The question was only ever which things get a recipe, and how the compiler knows which recipe is whose.

---

# 2. The census, before the shape

## 2.1 What was counted

`scripts/w189_artifact_control.py census`, over every graph the package can build — forty variants from eighteen of the twenty case modules (`seam_placement` searches over decompositions and `wind_farm_design` optimises over rollouts; neither builds a graph). The two Poseidon-backed graphs were loaded from the local cache with the hub offline.

| | count |
|---|---|
| graph variants built | **40** |
| carrying a partition of unity | **22** |
| of those, with exactly one overlapping region | **22** |
| graphs with two overlapping regions | 1 — `rocket`, which carries **no** partition |
| partition subdomain names are agent ids | 20 of 22 |
| subdomains are exactly one region's agents | 8 of 22 |
| subdomains a strict subset of the region | **12 of 22** |
| subdomains name no agent at all | **2 of 22** |

The strict subsets are all the same kind of agent: **a lumped closure of the fluid's own family** — `SUSP` in `front_wing` and `ground_effect` (a two-line spring), the rotor disks `R1`–`R3` in `wake_array`, `reuse_probe` and the ladder's six-window rung, `R1`–`R5` at twelve windows. W114's proxy puts each in the fluid region, because it declares the fluid's governing family — correctly, since a closure *within* a continuum problem is not a different continuum problem — and no partition blends it. The two with no agent names are both `channel_ns`: its partition names `window_ns`'s windows, `W00`…`W11`, while its own agents are `C00`…`C11`.

## 2.2 Three conclusions, one per row

1. **Cardinality: one per overlapping region.** No graph needs two partitions, and no non-overlapping region needs one. The map is sparse, which is the vehicle's shape: one aero tiling, several circuits.
2. **The region does not determine the partition.** In 12 of 22 the region holds agents the partition does not blend. A rule that read "the partition covers its region" would be false on more than half the package.
3. **The partition does not determine the region.** In 2 of 22 the subdomain names are not agents, so the region cannot be read off the partition either.

So the key is **declared**, and what *can* be read off the names is checked rather than assumed (§5). **This is W188's instruction working as intended**: both derivations failed on the census before either was built, where Tier 40's failed two tiers after.

## 2.3 Why one graph-level object passed the checks, and what it could not say

A graph-level partition covering several disjoint regions is a **direct sum**. With each region's subdomains restricting to its own index set $\mathcal I_r$,

$$\sum_i R_i^{\mathsf T}\chi_i R_i \;=\; \bigoplus_r \Big(\sum_{i\in r} R_i^{\mathsf T}\chi_i R_i\Big),$$

so

$$\Big\lVert \sum_i R_i^{\mathsf T}\chi_i R_i - I \Big\rVert_\infty = \max_r \Big\lVert \sum_{i\in r} R_i^{\mathsf T}\chi_i R_i - I_r \Big\rVert_\infty, \qquad \min_{i,j}\chi_{ij} = \min_r\,\min_{i\in r,\,j}\chi_{ij}.$$

**The identity (`L6/E6`) and convexity (`R11`) decompose exactly**: a direct-sum partition passes precisely when every region's does. So one object per graph always gave the right *verdict* for those two, and could never *name the region that failed*. Two quantities do **not** decompose:

- **W49's contaminated weight.** $\Pi = \max_j \sum_i \chi_{ij}\,\mathbf 1[\,j\in\mathcal C_i\,]$ over the sum is $\max_r \Pi_r$ — the worst region's weight, quoted in $\sigma \le C_\mu\,\Pi\,\lVert\delta\lambda\rVert$ as every region's.
- **L2/C2's pairwise-form check** counts subdomains, $n_{\text{sub}}=\sum_r n_r$, and decertifies the pairwise neighbour-disagreement form above two — so two regions of two windows each read as a four-subdomain partition.

And **R12** rests on the commutator identity $C\big(\sum_i \chi_i u_i\big)=\sum_i [C,\chi_i]\,u_i$, a sum over **one** partition's subdomains, so its constraint and the agents that can break it belong to that partition's region. The graph-scoped rule approximated this by reading the first constrained family anywhere in the graph and filtering by family (W161). Per region it is structural.

---

# 3. The control, captured before the change

## 3.1 What byte-identical means here

For each of the forty graphs: build, `compile_scheme`, and the sha256 of `RunArtifact.to_json()` — the whole emitted record, every decision's message and evidence, the scheme, the certificate, the stamp and the hole ledger. Not the verdict and not the rule set: a change that rewrote every message would pass those.

**The repeat floor comes first**, because `verdict._jsonable` writes a *set* as a list in iteration order and string hashing is randomised per process — so an artifact could in principle differ between two runs of the same code. Two captures of the unchanged tree under `PYTHONHASHSEED=1` and `2`: **forty of forty identical**, and no artifact contains a memory address. A before/after difference is therefore a change and not noise.

## 3.2 The result

| comparison | identical | differ |
|---|---|---|
| before, seed 1 vs before, seed 2 (the floor) | **40** | 0 |
| before vs after the change | **40** | **0** |

`tests/test_tier45_region_assembly.py` asserts both from the committed manifests and **recompiles eleven of the forty live on every run**, chosen to cross every path the change touches: no assembly (`rocket`, `wind_farm`), a co-located pair, a multirate pair, both circuits and the triangle control, both Tier 44 unions — one of which declares its tiling's partition as the graph's, exactly the declaration W189 replaces — a bare grid partition with the halo rule and L2/C2 live (`window_ns`), and a projected assembly that reaches `admit` (`front_wing`).

## 3.3 The one existing test the change broke

**Byte-identical artifacts did not mean an untouched suite.** Tier 40's `test_W173_no_epsilon_branch_was_written_into_the_halo_rule` reads the SOURCE of `_halo_rule` — it pins that no tolerance branch was written into the halo rule and that the rule still names `probe.support_reach` — and W189 moved the rule's body into `_halo_rule_for` so a per-region pass could run it. The positive assertion failed, which is the test working. **The negative assertion had meanwhile become vacuous**: *no tolerance token in `_halo_rule`* was now true of a husk, and no run would ever have shown it. It is re-pointed at both functions with both assertions verbatim, plus two anchors — the subject still holds `required_halo()`, and `_halo_rule` still delegates — so the next move fails here rather than passing emptily. The full suite otherwise passed: $1187$ tests, $1186$ in the run and the re-pointed file $24$ of $24$ after.

---

# 4. The schema, and the one mechanism that scopes every rule

## 4.1 The fields

```
partition_of_unity : object | {region: object} | None
overlap            : float  | {region: float}  | None
overlap_cells      : int    | {region: int}    | None
```

A region is a `governing_family` with more than one agent — exactly `CaseGraph.regions()`' keys, as Tier 44 defined it. `per_region(name)`, `declares_per_region()`, `partition_for`, `overlap_for`, `overlap_cells_for` and `assembly_projection_for` read the map. **A single object is not read as any region's**: `partition_for(region)` returns `None` for it, because Tier 44's union is the case that shows why — its one partition covered only the tiling. And `assembly_projection`, the one-answer question, **raises** on a per-region graph rather than returning `None`, which would read as *no projection declared* on a graph whose regions may each declare one.

## 4.2 `DecisionRecord.rescope` — the same rule, scoped

Every rule W189 touches runs its **unchanged body** once per region and then calls `rescope(start, region)` on the decisions it just emitted: each sentence is kept and prefixed `region '…': `, a graph-scoped subject — `<graph>` or `<assembly>` — becomes `<region:…>` or `<assembly:…>`, and the evidence gains `region`. Agent and seam subjects stay, because they already name something inside the region.

**One mechanism, so a region-scoped decision cannot drift from the graph-scoped decision it is.** W136 factored `_decomposition_cuts` out so two rules resting on one premise could not come to disagree about it; this is the same move applied to scope. It is also what makes the equivalence control (§6.2) a structural fact rather than a hope.

## 4.3 What each rule does

| rule | graph-scoped — unchanged, byte for byte | per region |
|---|---|---|
| `L6/E6`, `R11`, `L6/C1` | one partition, subject `<assembly>` | each overlapping region's own partition, subject `<assembly:…>`; an overlapping region with nothing filed under it is decertified **by name** |
| `L6/R12` | constraint from the first constrained family in the graph | the **region's** family; enforcing agents intersected with the region; the region's projection; a family with no pointwise constraint is **stated** where the graph-scoped path returns in silence |
| `L6/W49` | $\Pi$ of the one partition | $\Pi_r$ of the region's partition; **one graph-level $C_\mu$ with two or more regions carrying partitions decertifies** (W190) |
| `L2/R10/halo` | one `overlap_cells` against every overlapping region's cut agents | each region's own count against its own agents; cut agents of **other** regions named separately, not called sole agents |
| `L2/C2`, `L2/C3` | once per axis present | once per region on that axis; **one graph-level `cut_defect_bound` with two or more regions on a branch decertifies** (W190) |
| `E6` in the stamp | stamped once | stamped once, **fails over unchecked over holds** — because `EnvelopeStamp.set` lets a later `unchecked` overwrite `holds`, region-by-region stamping would make E6 depend on the order regions were visited |
| artifact | as before | `assembly.regions`, `scheme.overlap_by_region`, `harness.regions` — **emitted only when present**, which is how forty artifacts stayed byte-identical |

---

# 5. The declaration check — `L2/W189/regions`

A key is a declaration like any other, and what it names is decidable from the graph alone, before anything is probed. `CaseGraph.region_declaration_report` reports; the compiler decides.

| what the key or the partition names | verdict | why |
|---|---|---|
| no agent's family | **refuse** | nothing reads the value, and the region it was meant for reads as undeclared |
| a family with one agent | **refuse** | a sole agent owns its whole region (W114): no artificial face, nothing to weight or outrun |
| a region whose seams are non-overlapping | **refuse** | a substructuring cut shares no cells; one region cut both ways is `region_axis_conflicts`' defect through another field |
| partition subdomains that are agents of **another** family | **refuse** | the blend would weight another region's state into this one, and neither the identity nor convexity can see it — both are properties of the weights alone |
| partition subdomains that name **no** agent | **decertify** | whether the partition covers the region cannot be decided |
| region agents the partition does not blend | disclosed in the admission | 12 of the 22 partitions in the package; not a defect |

**Read against the census**, had every existing partition been filed per region: it refuses none of the 22, discloses unblended agents on 12, and decertifies 2 — both `channel_ns`, whose names were never matched to anything because nothing read them (**W191**).

**Each outcome fired beside the case where it is quiet**, on the union of §6:

| variant of the union | `L2/W189/regions` |
|---|---|
| clean | one `admit`, listing what was filed where |
| partition filed under `heat-conduction-2d` (the block's own family) | `refuse`, `<region:heat-conduction-2d>` |
| `overlap_cells` filed under the coolant circuit's region | `refuse`, `<region:incompressible-thermal-transport-1d>` |
| a zero-weight subdomain named `PASS` added to the tiling's partition | `refuse` — identity and convexity both still pass |
| the tiling's subdomains renamed `X00`…`X11` | `admit-uncertified` |

---

# 6. The union, on a real tiling

## 6.1 The graph and what every rule says

`window_ns` in split-step mode at the Tier 0 state — four `reference.WindowNS` windows, its own `GridPartitionOfUnity`, a 21-cell halo — beside `cooling_loop` and `powertrain`, with the partition, the overlap and the overlap-cell count filed under the fluid region and nowhere else. **Fourteen agents, fourteen seams, three regions**, one sole agent (`BLOCK`). No seam joins the three subsystems; that is W172.

| rule | per region | the same union, graph-scoped |
|---|---|---|
| `L2/W189/regions` | `admit`, fluid | — |
| `L2/R10/halo` | `admit`, fluid: overlap $21$ cells covers the $20$-cell domain of dependence | same sentence, `<graph>` |
| `L2/C2/W58` | disclosure, fluid: $152$ cells contaminated for more than one window, up to $3$ at once | same, `<graph>` |
| `L2/C2` | `admit-uncertified`, fluid: no `cut_defect_bound` declared | same, `<graph>` |
| `L2/C3/W57` | `admit-uncertified` **twice**: the coolant circuit, the DC circuit | once, `<graph>` |
| `L6/L6/C1` | `admit`, fluid: $\min\chi = 0.001946$, identity residual $1.1\times10^{-16}$, $\lVert A\rVert = 1$ | same, `<assembly>` |
| `L6/R12` | `admit`, fluid: every window exposes its elliptic part | same, `<assembly>` |
| `L6/W49` | `admit-uncertified`, fluid: $\Pi = 0.09541$, $C_\mu$ not declared | same, `<assembly>` |
| decisions / verdict | $206$ / `refuse` | $204$ / `refuse` |

**Every assembly and halo decision names the fluid region and no decision names a circuit's**; the substructuring criterion names each circuit. The graph-scoped union reaches the same verdict under the same rules and **names no region anywhere** — which is W189's defect, measured on the graph it is about. The two extra decisions are the declaration check's admission and the second `L2/C3`.

**The union's one refusal is not the assembly's.** It is `L7/R9`: `powertrain`'s agents step at $0.2$ and the tiling's at $0.05$, and pointwise flux matching across different clocks is refused. The same union with only the coolant circuit, whose clock matches the tiling's, compiles `admit-uncertified` with no refusal and the same region naming. **The union declares no measured constants**, deliberately: the tiling's were measured on the tiling alone, and a constant measured on one graph is not the union's (W56).

## 6.2 The equivalence control

`window_ns` alone, declared both ways: **the same verdict, `admit-uncertified`, and the identical multiset of (layer, rule, verdict)** apart from the declaration check's own admission. The per-region form is the same rule scoped, so the union's differences are the union's and not the form's.

## 6.3 Two regions — the control a one-region union cannot give

A union with one overlapping region cannot distinguish *per region* from *the one overlapping region*. `rocket` has two, `reacting-compressible-flow` and `external-compressible-flow`, and no partition. Given one synthetic strip partition each — every window with an interior, so $\lVert A\rVert = 1$:

| variant | what the record says |
|---|---|
| both filed | `L2/R10/halo`, `L2/C2`, `L6/L6/C1`, `L6/R12`, `L6/W49` each speak **twice**, once naming each region; E6 holds |
| one filed | the other region is decertified at `L6/E6` **by name**; E6 is unchecked; $\lVert A\rVert$ is listed unmeasured |
| both filed, with one graph-level $C_\mu$ and cut-defect bound | `L6/W49/W190` and `L2/C2/W190` fire once per region — the constants are not quoted for either |

`R12` states twice that a compressible family carries no pointwise linear constraint — the sentence the graph-scoped path does not emit.

---

# 7. What the union cost to declare — handed to W172, not answered here

## 7.1 The graph-global fields that remain

Every field that is still one value per graph, as the three parts declare it and as the union had to:

| field | the parts | the union |
|---|---|---|
| `decomposition` | overlapping, non-overlapping, non-overlapping | overlapping, with every seam declaring `cut_axis` |
| `partition_of_unity`, `overlap`, `overlap_cells` | the tiling's; none; none | **filed under the fluid region** |
| `global_fields` | the tiling's pressure, `applies_to=()` meaning every agent | scoped to the four windows — unscoped, `L3/global-field` refuses it |
| `cross_points` | `("centre",)`; `()`; `()` | `("centre",)` — a name with no region (§7.3) |
| `loop_gains` | none; the coolant cycle; the circuit cycle | both, matched by agent set |
| `macro_dt` | $0.05$, $0.05$, $0.2$ | $0.05$ — and `L7/R9` refuses the clocks |
| `measured` | three different records | none |

This is not W172's count — no seam joins anything here — but it is the list of decisions a union makes before its first joining seam exists, and three of them (`decomposition`, `macro_dt`, `measured`) are the parts disagreeing.

## 7.2 Joining graphs changed an agent's region — W192

`powertrain`'s `ROTOR` declares the fluid's governing family, like every actuator disk here. In `powertrain` it is the **sole** agent of that family, uncut, in no region. In the union the tiling's four windows share its family, so **the same agent, unchanged, is cut and in the fluid region** beside `W00`…`W11`. A region is a property of the graph (W114's proxy), and putting two graphs together re-derived it for an agent neither graph touched. The declaration check discloses `ROTOR` as unblended, and **no verdict moves on this graph**, because the rotor has no stencil and no elliptic part.

**[AI Inference]:** this is the first measured instance of a union re-deriving a premise across its parts, and it is the shape the brief for W172 warns about — *a re-derivation … across the union*. An agent declaring `elliptic_subsolve=embedded` pulled into a region this way would move `L2/R10` from its `sole-family` admission to a refusal, with nothing about that agent changed. Whether any subsystem rung 9 needs is such an agent is unmeasured.

## 7.3 A named cross-point follows the leftover axis field — W193

`cross_points` names vertices by label. A vertex named `"centre"` contains no agents, so `_cp_axis` cannot find its region and falls back to the one axis field the graph still has. **The same union with `decomposition=NON_OVERLAPPING`** — every seam's own `cut_axis` unchanged — **is refused at `L2/I2/G1`** for the tiling's own cross-point. The verdict follows a field that W171 was supposed to have made a fallback.

---

# 8. What was found and not fixed

- **W190 — `MeasuredConstants` is one record per graph.** Its provenance is a probe state, a scheme and a depth, and no region. Guarded where it would be quoted for two or more regions ($C_\mu$ at W49, `cut_defect_bound` at L2/C2 and L2/C3); `L`, $\tau$ and $\sigma$ are untouched and are still graph-level.
- **W191 — `channel_ns`'s partition names `window_ns`'s windows.** Latent: no rule reads subdomain names on a graph-scoped partition. Renaming them would change `channel_ns`'s artifact, which this tier's control forbids; the check that would catch it exists now for the per-region form.
- **W192 — a union re-derives region membership** (§7.2).
- **W193 — `cross_points` has no region** (§7.3).
- **On a graph-scoped union the halo rule's W136 clause misdescribes other regions' cut agents as sole agents.** Tier 44's own union reads *"not against BLOCK, COLD, HOT, PASS, RAD, each of which is the sole agent of its governing_family"* — four of the five are coolant legs sharing a family. The per-region path names them separately; the graph-scoped path is left byte-identical, because the control is the claim and the repair is the per-region form.
- **`L6/C1`'s admission says $\lVert A\rVert = 1$ "for any convex partition of unity"**, where `norm_A`'s own docstring conditions it on some cell having a single owner. A first synthetic fixture with no interior read $0.5774$ beside that sentence; the fixture was given interiors rather than the legacy sentence changed.

---

# 9. What this tier did NOT do, named

- **W172 was not started.** The union is **disjoint** — no seam joins the tiling to either circuit or the circuits to each other — so the $O(K)$ count was not re-run across joining seams, and the caveat stands exactly as written: the zero-per-pair count is strong evidence about the **algebra** and no evidence about the **integration cost**. §7 is input to W172, not its answer.
- **Rung 9's graph was not built.** Fourteen agents and three regions with no joining seams; rung 9 is $\sim 18$ agents across five families with three joining seams.
- **`MeasuredConstants` was not made per region** (W190), **the region was not unpinned from the family** (W192), **cross-points were not given regions** (W193), and **`channel_ns`'s names were not fixed** (W191).
- **`solve.py`'s per-region assembly branch is exercised by no run.** It exists so a per-region graph does not crash the runtime path; nothing here rolls a union forward.
- **W182, W183–W188, W175, W178** untouched. Nothing downloaded — Poseidon-T loaded from the local cache with the hub offline — no machine rented, NeuberNet not loaded.

---

## See Also

- [[per-region-decomposition-axis]] — Tier 44, W171 part 1: the axis per region, and §6's boundary this page closes
- [[gap-worklist]] — W189 (closed), W190–W193 (opened), W172 (not started), W188 (the census instruction this applied)
- [[case-study-ladder-to-f1]] — §19, and the critical path rung 9 sits on
- [[f1-pathmap-and-end-goal]] — §3.3's rung 9 row
- [[master-error-bound]] — §4.1, the overlapping branch $\sigma \le C_\mu\,\Pi\,\lVert\delta\lambda\rVert$ whose $\Pi$ is now per region
- [[end-to-end-architecture-spec]] — §8, the assembly layer this is the schema of
- [[atlas-implementation]] — the nine layers
- [[case-study-cooling-loop-atlas-0.1]] · [[case-study-powertrain-atlas-0.1]] — the two circuits in the union
- [[tier0-measurements]] — the `window_ns` tiling the union carries
