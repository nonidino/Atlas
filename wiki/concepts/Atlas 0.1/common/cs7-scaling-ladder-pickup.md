# CS-7 `scaling_ladder.py` — self-contained pickup

**Type:** Core Concept — Build Brief / pickup prompt (folder: `Atlas 0.1/common/`)
**Status:** opened 2026-08-30. Written to be **the only page a fresh session needs to read before starting**, in the sense [[gap-worklist]] W24 says a pickup prompt should be and no page in this vault yet is. Everything below is either quoted from a measured artifact or marked as a choice to be made.
**Related:** [[case-study-ladder-to-f1]] · [[case-study-wake-array-atlas-0.1]] · [[gap-worklist]] · [[tier0-measurements]] · [[f1-pathmap-and-end-goal]] · [[master-error-bound]] · [[atlas-implementation]]

---

# 1. The mission, in one paragraph

Build the **seventh real case study**, `atlas/cases/scaling_ladder.py`, and its driver `scripts/w100_scaling_ladder.py`. It measures the one quantity this program's own falsification criteria call decisive and which has never been measured at any size: **how composition error grows with interface count**. Grow a tiling of $128$-cell windows from $N = 1$ to $N = 24$, holding every other variable fixed, and report the composed defect, the L2/C2 cut-defect bound, $\Pi$, $C_\mu$, $\tau$, $\sigma$, $\gamma$ and wall time per macro-step at each $N$ — in **two columns**, a classical expert and a frozen checkpoint. Close **W58**, **W54** and **W81** along the way, because each is a defect in the instrument this run depends on. Then write the results up as [[tier0-measurements]] §19 and a Tier 19 on [[gap-worklist]].

**Why this and not something else:** [[f1-pathmap-and-end-goal]] §3.2 names rung 9 as *the rung that decides everything* and §5's **F1** as the criterion that fails if composition error grows super-linearly in interface count. §5 schedules the sweep as rung 9's **early warning** and calls it *a cheap look at the expensive question, and a bad result there is a reason to stop and rethink, not to press on*. It has never been run. [[case-study-ladder-to-f1]] schedules every other case study on its result.

---

# 2. Orientation — read in this order, then stop reading

1. `atlas/CASE-STUDY-GUIDE.md` — what a case study is, the five things you declare, and the eight first-attempt mistakes. **Non-negotiable.** A case study is exactly one file in `atlas/cases/`; if you find yourself editing `compiler.py` or `probe.py` for this case, stop, because that is hand-written coupling and the whole architecture exists so none is needed.
2. `atlas/cases/wake_array.py` — the sixth real case study and the direct parent of this one. Its tiling, partition of unity, `transport_and_project`, port segmentation and absolute-trace convention are all reused.
3. `scripts/w93_wake_array.py` — the driver to model this one on, especially `march()`.
4. [[case-study-wake-array-atlas-0.1]] — the readable summary, and §3's honesty list.
5. [[gap-worklist]] Tier 18 — the five rows the parent case study opened. **W98 and W99 are traps you will hit.**
6. [[master-error-bound]] §4 and §4.1 — the substructuring branch and the overlapping branch. This run measures the overlapping one.

Skim only if needed: [[tier0-measurements]] §10 (L2/C2, the cut-defect identity), §18 (the parent's full record).

---

# 3. Environment — six things that have each cost a session before

1. **Two packages named `atlas` collide.** The vault's `atlas/` and the build repo's `src/atlas/` share a top-level name, so the build repo cannot go on `sys.path`. `window_ns.load_reference()` registers it under the private name `atlas_windfarm_reference`; import build-repo modules with `importlib.import_module("atlas_windfarm_reference.<mod>")` and never any other way.
2. **`KMP_DUPLICATE_LIB_OK` must be set before any import** when torch and numpy linear algebra are in the same process, which they are the moment Poseidon-T loads. `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` at the top of the driver, above `import numpy`. Verify it rather than trusting it.
3. **Never author LaTeX inside a Python or shell string literal.** Escapes corrupt the page silently — `\tau` becomes a tab, `\rho` becomes a carriage return — and the standing verification cannot always see it. Write wiki pages with the file-writing tool directly, then run `python scripts/vault_scan.py` and require **0 problems**.
4. **Redirect long runs to a file.** A background command piped to `tail` hides all progress until it exits. `> out/w100/run.log 2>&1` and read the file.
5. **Warn before any multi-hour run, and persist state on every stage** — the parent driver's `persist()` pattern, called after each stage, so a crash at $N=24$ does not discard $N=1..12$. Cache the marched state to `.npz` and support `--reuse-state`.
6. **Verify the date from the environment before stamping anything.** The log is append-only and Tier 14 went in a day fast.

Run the existing suite before you start and after you finish: `python -m pytest -q` (401 tests passing as of 2026-08-30).

---

# 4. What to build

## 4.1 `atlas/cases/scaling_ladder.py`

A parameterized tiling of $N_{\text{col}} \times N_{\text{row}}$ windows of `EXPERT_RES` $= 128$ cells. **It is a new file rather than a flag on `wake_array` because three things do not survive parameterization**: the rotor layout, the per-$N$ controls, and the referent (which is a monolith for the classical expert at small $N$ and a pair for the checkpoint always).

Reuse from `wake_array` — import it, do not copy it — `ArrayTiling` (generalized), `transport_and_project`, `fourier_basis`, `modes_for`, `port_segments`, `FluidWindow`, `RotorDisk`. Hold **fixed across the whole sweep**:

| held fixed | value | why |
|---|---|---|
| $\mathrm{d}x$ | $S_{\text{LEN}}/128 = 1/32\,D$ | the checkpoint's resolution is not a dial |
| $\Delta t_{\text{macro}}$ | $0.2$ | exactly one native expert lead |
| overlap | $16$ cells | changing it changes $\Pi$, which is a measured output |
| ramp | $8$ cells | W53 keeps ramp 8 so every prior constant stays comparable |
| $\nu_{\text{ref}}$ | $3.92\times10^{-3}$ | the referent's own viscosity, W0 4.2's grid-scale fit |
| disk model, induction $a$ | `ActuatorDisk`, $1/3$ | zero fitted parameters, unchanged |

**The ladder.** Five sizes, chosen so interface count spans a decade and the aspect ratio does not drift:

| $N$ | tiling | rotors | interfaces (approx.) |
|---|---|---|---|
| 1 | $1\times1$ | 0 | 0 — **the control, and it must return exactly zero defect** |
| 2 | $2\times1$ | 1 | 1 |
| 6 | $3\times2$ | 3 | 13 — **the parent case study, and its numbers must reproduce** |
| 12 | $4\times3$ | 6 | ~29 |
| 24 | $6\times4$ | 12 | ~62 |

Two of the five rows are controls and both matter. **$N=1$ is the control that would have redirected the parent's search immediately if it had been run** — [[tier0-measurements]] §8's own lesson, and it is one line. **$N=6$ is the reproduction control**: the same geometry as `wake_array`, so its array loss must come back at $14.62\%$ (Poseidon) and $25.27\%$ (reference), and if it does not, the ladder is measuring the port and not the physics.

Rotor placement: keep the parent's $3.5\,D$ spacing and its L, and tile it — every rotor at a whole number of window strides, every row carrying at least one clean-inflow control.

## 4.2 `scripts/w100_scaling_ladder.py`

Stages, in order, persisting after each:

1. **March**, per $N$, per expert, four configurations (turbine-free, solo, first-off, all) as the parent does. The turbine-free run is the noise floor and must be reported at every $N$.
2. **Composed defect against a referent.** For `WindowNS` the referent is the **monolith of the same class** at $N \times 128$ cells minus overlaps, which exists at any resolution. For Poseidon-T there is **no monolith at any resolution ever**, so the referent is a `WindowNS` pair, per `lambda_ref` — this is **W95** and it is why the classical column is load-bearing rather than a courtesy.
3. **The bound.** $\lVert\sum_i \chi_i \lvert D_i\rvert\rVert$ (L2/C2), $\Pi$, $C_\mu$ implied. Tightness is $0.2\%$ at $N=4$ and **unknown above it**; report the ratio at every $N$.
4. **The three-way split**, $\tau$, $\sigma$, $\gamma$, per seam, depth-tagged and harness-tagged (**W54**).
5. **Wall time** per macro-step per agent — Claim B's other half, which is $O(1)$ integration work per added agent and is as falsifiable as the error.
6. **Compile** at each $N$: verdict, refusals, decertifications. Expect `admit-uncertified` and expect the decertification count to **grow with $N$**; report how.

Artifact `out/w100/w100.json`. Cache state to `out/w100/state_N<k>.npz`.

## 4.3 The gate

**Composition error grows no faster than linearly in interface count, in both columns, over the full $24\times$ range.** Report the fitted exponent with an error bar, not an adjective.

Sub-linear is Claim B confirmed. Linear is survivable and should be stated as such. **Super-linear is the pathmap's stated reason to stop and rethink**, and if that is the result, write it up plainly and do not soften it — a clean negative here saves the entire Phase C of [[case-study-ladder-to-f1]].

---

# 5. The three W-rows to close, and why each is on this run

These are not side quests. Each is a defect in an instrument this measurement depends on.

## W58 — which `cut_defect_bound` was measured

`cut_defect_bound` can mean either $\lVert\sum_i\chi_i\lvert D_i\rvert\rVert$ (needs a monolithic reference, exactly as $\tau$ does) or the reference-free neighbour disagreement. **They are provably equal only when the agents' contaminated sets are pairwise disjoint**, and on the parent's tiling they are not — they agreed to $1.00007$ anyway, which is luck and not a result. At $N=24$ the contaminated sets overlap far more.

**Done when:** the record carries a provenance field naming which of the two was measured, and the compile reports whether the disjointness condition that makes them equal holds — which is **decidable from the `contaminated` geometry already declared** for W49's $\Pi$.

## W54 — an emitted defect carries its depth but not its harness

Three composition-layer defects have now worn an agent's label: the decomposed pressure solve ($99.8\%$ of the first defect), the partition of unity with full weight at the artificial edge, and the exchange cadence (two orders in $\tau$). **W98 was the fourth** and the depth tag caught none of them.

**Done when:** an emitted defect carries the harness parameters it is a function of — overlap, $\chi$'s shape, exchange cadence, elliptic placement — and a test asserts that changing a harness parameter changes the attribution. This run is the natural home for it because it varies exactly one harness parameter ($N$) by design, so anything else moving is a bug the schema will now show.

## W81 — $\beta_{\min}$ is undefined anywhere in this framework

It is *the whole discriminating power* of the plug-in guarantee. `certify_substitution` takes it from the caller with no default and the one sibling default is $10^{-12}$, at which **every seam in the vault is blind**.

**Done when:** either $\beta_{\min}$ is derived from quantities the compile already has — the honest candidate is $\varepsilon_{\text{tol}} = \min(\tau,\sigma)$ over the terms that carry a scale, which is already the interface tolerance — or the framework states that it is a user-supplied risk parameter and the certificate reports the **threshold above which the substitution becomes visible** rather than a verdict. Either is a result; silently defaulting is not.

---

# 6. Traps, all of them measured

1. **W98's class.** Transport and pressure are **global** operations. `step_many`'s `galilean` and `project` flags do both **per window and periodically**. At $N=1$ this is invisible; at $N=24$ the domain is larger and the wrap is further away, so the defect will look like it *improves* with $N$ — which would be a false confirmation of Claim B. Use `wake_array.transport_and_project` on the assembled domain, once, and add the $N=1$ zero-defect control that catches it.
2. **W99.** `FrozenFluidExpert.step` and `.step_many` **silently drop `force`** when `galilean=False` — the block is nested inside `if gal:` — and the same branch pins each window's output to zero mean, deleting the momentum deficit a disk just deposited. The parent's `march` applies the force after assembly and preserves each window's incoming mean; do the same and do not assume the build repo has been fixed.
3. **W93.** The checkpoint's support reach is the whole window, so `required_halo()` returns `None` and the halo rule decertifies. That is correct behaviour, not something to work around, and the decertification count per $N$ is one of the outputs.
4. **The freestream band.** Without holding the inlet and both laterals at $(U_\infty, 0)$ the disks drain the box — measured, every turbine's inflow falls monotonically to $0.06$ by step $45$. At larger $N$ there are more disks and this gets worse, not better.
5. **The disk's smearing thickness is derived, not chosen:** $\Delta_d = \langle U_d\rangle\,\Delta t$. At `disk.py`'s default $0.1\,D$ the impulse is roughly $2\times$ what momentum theory allows.
6. **Build the viewer.** [[case-study-wake-array-atlas-0.1]]'s single most transferable finding is that **a rendering of the state is a measurement instrument**: W98 was smooth, bounded, physically shaped, in the right units, passed every control, and was found by watching an animation beside the referent's. Reuse `scripts/w93_frames.py` and `scripts/w93_build_viewer.py`, and **look at $N=24$ before believing any number from it**.

---

# 7. When the run is done

Following this vault's own operating convention, the measurement is not finished until it is filed:

1. **[[tier0-measurements]] §19** — the full record: derivations, the failed hypotheses, every number with its provenance.
2. **[[gap-worklist]] Tier 19** — rows closed (W58, W54, W81), rows moved, rows opened. Do not delete rows; update the Status column.
3. **[[case-study-ladder-to-f1]]** — §7's F1 and F3 rows get their first measured values, and §4's Phase B/C schedule is either confirmed or stopped.
4. A short readable companion page if the result warrants one, on the model of [[case-study-wake-array-atlas-0.1]].
5. **`wiki/index.md`** — entries under the Atlas 0.1 section; **`wiki/log.md`** — append `## [YYYY-MM-DD] note | ...` with the date verified from the environment.
6. `python scripts/vault_scan.py` → **0 problems**. `python -m pytest -q` → all pass, with new regression tests for the case study and for each W-row closed.

**Report the negative if it is negative.** This vault's most valuable pages are the ones where the instrument said something the author did not want — §5's healthy diagnostics on a scheme solving the wrong equation, §18's coupling out-erroring the model it was measuring. A super-linear result here is worth more than a confirmatory one, because it is the difference between building Phase C and not.

---

## See Also

- [[case-study-ladder-to-f1]] — the plan this is step one of, and why CS-7 comes before everything
- [[case-study-wake-array-atlas-0.1]] — the parent case study, its numbers, and its six findings
- [[gap-worklist]] — W54, W58, W81 in full, with the reasoning behind each definition of done
- [[tier0-measurements]] — §10 for L2/C2, §18 for the parent's record
- [[master-error-bound]] — §4.1's overlapping branch, which is the one this run measures
- [[f1-pathmap-and-end-goal]] — F1 and F3, the two criteria this run reports against
