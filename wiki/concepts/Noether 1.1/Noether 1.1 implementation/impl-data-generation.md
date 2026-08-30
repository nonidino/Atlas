# Portion 2 — Training-data generation

**Type:** Implementation spec — agent task (folder: Noether 1.1 / Noether 1.1 implementation)
**Portion of:** [[00-implementation-plan]] (Phase I). **Depends on:** [[impl-repo-scaffold]]. **Unblocks:** [[impl-tokenizer-descriptors]], [[impl-training-harness]], everything downstream.
**Design pages:** [[training-scheme-1.1]] (regime curriculum + data diversity), [[conditioning-and-constants-1.1]] (constants per system), [[navier-stokes-equations]].

---

## Objective
Produce the datasets for the regime-curriculum ladder, one solver + dataset module per rung, emitting a **uniform on-disk schema** the tokenizer can consume for any field set. Reuse 1.0's `ns_solver.py`, `rbc_solver.py`, `burgers_solver.py`.

## The ladder (build rungs 1–3 now; 4–6 stubbed with clear TODOs)
| Rung | System | Fields | Constants | Solver |
|---|---|---|---|---|
| 1 | 1D Burgers | u | ν | reuse `burgers_solver.py` |
| 2 | 2D incompressible NS | velocity | Re | reuse `ns_solver.py` |
| 3 | 2D Rayleigh–Bénard | velocity, temperature | Ra, Pr | reuse `rbc_solver.py` |
| 4 | Compressible flow | velocity, density | Ma, Re | **new** (or Well loader) |
| 5 | Settling particles in fluid | velocity + particle set | Re, Stokes | **new** (coupled) |
| 6 | MHD | velocity, B | Re, Rm | **new** |

## Deliverables
- `noether11/data/<system>_solver.py` + `<system>_datasets.py` per rung (1–3 ported, 4–6 scaffolded with a raised `NotImplementedError` and a spec comment).
- **Uniform sample schema** (a `.npz`/`zarr` per trajectory): `fields: dict[name -> array[T,H,W,C]]`, `field_meta: dict[name -> {rank, parity, units}]` (feeds the structured block of $\theta_f$, [[field-descriptors-1.1]]), `constants: dict[name -> value]`, `dt`, `geometry/BC spec`, `grid`.
- **Diversity generators** ([[training-scheme-1.1]] "data diversity"): sweep characteristic numbers (a *range* of Re/Ra/Ma per rung, not a few points) and include the **off-attractor, energy-diverse** initial conditions the conservation-discovery search needs (per [[noether-1.0-rbc]] §5.1).
- `scripts/generate.py --system <name> --n_traj … --sweep …` writing to `data/<system>/`.
- A tiny "smoke" split (a handful of short trajectories) per system for CI.

## Interface contract
- `noether11.data.load(system, split) -> Dataset` yielding samples in the uniform schema above.
- The schema is **the** contract between data and model; the tokenizer reads only it, never a system-specific field name. Adding a system = adding a solver + registering its `field_meta`, never touching the model.

## Build steps
1. Port the three 1.0 solvers; wrap their output in the uniform schema (add `field_meta`, `constants`).
2. Implement the parameter sweeps + off-attractor IC generation.
3. Write `scripts/generate.py` and the `load()` registry.
4. Emit smoke splits; add `tests/test_data_schema.py` asserting every system's samples validate against the schema.
5. Scaffold rungs 4–6 solvers with specs (compressible: a shock-capturing scheme; particles: reuse NS + a Lagrangian particle integrator, the coupling handled later by [[impl-backbone]]'s typed edges; MHD: add induction equation).

## Acceptance tests
- `tests/test_data_schema.py` green for rungs 1–3; every sample has consistent `fields`/`field_meta`/`constants`.
- A generated NS trajectory renders correctly in the GUI ([[impl-gui-2d]], milestone M0).
- Re/Ra/Ma sweeps cover the ranges [[noether-1.1-medium]] §4's generality checks require.

## Pitfalls
- **`field_meta` rank/parity must be correct** — a mislabeled parity is a real bug that corrupts the covariant machinery ([[field-descriptors-1.1]] "descriptor grounding"). Velocity: rank 1 even; B: rank 1 odd (pseudovector); temperature/density: rank 0.
- Nondimensionalize at **load**, not in the solver, so raw physical units survive on disk ([[conditioning-and-constants-1.1]] "ruler").
- Ensure at least one field (velocity) appears in ≥2 systems early, so $\theta_f$ learns *field* identity not dataset ID ([[training-scheme-1.1]] open item).

## See Also
- [[impl-tokenizer-descriptors]] (consumes the schema) · [[training-scheme-1.1]] (curriculum order) · [[conditioning-and-constants-1.1]] (constants)
