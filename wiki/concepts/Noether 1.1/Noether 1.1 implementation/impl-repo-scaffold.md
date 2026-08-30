# Portion 0 — Repo scaffold & config system

**Type:** Implementation spec — agent task (folder: Noether 1.1 / Noether 1.1 implementation)
**Portion of:** [[00-implementation-plan]] (Phase I). **Depends on:** — . **Unblocks:** everything.
**Design pages:** [[00-noether-1.1-overview]] (design invariants), [[training-scheme-1.1]] (module inventory).

---

## Objective
Stand up the `noether-1.1` branch's directory layout, dependency/config system, and CI/test skeleton, reusing Noether 1.0's working code as the starting point. After this portion, an agent can `pip install -e .`, run an empty test suite green, and import an (empty) config for medium/large.

## Deliverables
- Branch `noether-1.1` reorganized to [[00-implementation-plan]] §1's layout: `src/noether11/{core,conserve,generate,condition,data,train,eval,configs}`, `gui/`, `scripts/`, `tests/`, `notebooks/`.
- **Port, don't rewrite,** from 1.0 (`src/noether/`): `data/*solver*.py`, `data/*datasets*.py`, `graph/`, `model/siren.py`, `model/projection.py`, `model/operators2d.py`, `model/field_norm.py` → their `noether11/` homes. Keep git history where practical (`git mv`).
- `configs/` — dataclass configs `MediumConfig`, `LargeConfig`, `SmokeConfig` with every symbol from [[noether-1.1-medium]] / [[noether-1.1-large]] §1 (d, L, H, T, N_q, d_hi, d_theta, d_diff, L_diff, edge K/K′, ε schedule, LR, batch…). One dataclass, three instances.
- `pyproject.toml` updated (package name `noether11`, deps: torch, numpy, scipy, einops, a mesh/KNN lib e.g. `torch_cluster` or a self-contained KNN); `.gitignore`; `README.md` describing the 1.1 layout and pointing to [[00-implementation-plan]].
- CI: a `pytest` skeleton and a `tests/test_config.py` asserting the three configs instantiate and their param-count estimator matches [[noether-1.1-medium]]/§2 within 5%.

## Interface contract
- `from noether11.configs import MediumConfig, LargeConfig, SmokeConfig`
- `cfg.param_estimate() -> dict[str,int]` implementing the §2 budget formula `[(9+3T)L + 5F + 7]d² + diffusion`.
- All core modules take a `cfg` object; **no magic numbers in module code** (design invariant: equation-agnostic core).

## Build steps
1. `git checkout noether-1.1`. Create the `src/noether11/` package tree with `__init__.py`s.
2. `git mv` the reusable 1.0 modules into place; fix imports; delete 1.0-specific glue that assumed a single field or fixed graph.
3. Write the config dataclass + three instances + `param_estimate()`.
4. Wire `pyproject.toml`; `pip install -e .`; make imports resolve.
5. Add `tests/test_config.py` and a trivial `tests/test_import.py`; get CI green.

## Acceptance tests
- `pytest` green; `pip install -e .` clean.
- `MediumConfig().param_estimate()['total']` ≈ 44M (±5%); `LargeConfig()` ≈ 327M (±5%).
- No module under `noether11/core/` imports anything from `data/` (enforced by a test that greps imports) — the equation-agnostic boundary.

## Pitfalls
- Do **not** carry over 1.0's `config.py`/`config_rbc.py` split — 1.1 has one config with a variable field set. 
- Keep the 2D operators (`operators2d.py`: div, curl, grad via fixed finite differences) — the tokenizer's redundant-derived-features depend on them ([[graph-tokenizer-1.1]]).

## See Also
- [[00-implementation-plan]] §1 (layout) · [[noether-1.1-medium]] §1–2 (config fields) · [[impl-data-generation]] (consumes the ported solvers)
