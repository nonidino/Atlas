# Noether 1.1 — Implementation Plan

**Type:** Implementation master plan (folder: Noether 1.1 / Noether 1.1 implementation)
**Status:** Active build plan. Translates the Noether 1.1 *design* ([[00-noether-1.1-overview]] and its portion pages) into an ordered, agent-executable build, from a 2D GUI through training-data generation to the model itself. Every portion below has a **standalone spec file** in this folder, written so an agent can execute it independently; this page is the index and the dependency graph tying them together.
**Design source of truth:** the Noether 1.1 concept pages ([[00-noether-1.1-overview]], [[backbone-1.1]], [[graph-tokenizer-1.1]], [[field-token-streams-1.1]], [[field-descriptors-1.1]], [[edge-generation-1.1]], [[decoder-1.1]], [[discovered-conservation-1.1]], [[post-decoder-diffusion-1.1]], [[conditioning-and-constants-1.1]], [[training-scheme-1.1]]). **This plan never overrides the design — if a spec file and a concept page disagree, the concept page wins and the spec file is a bug.**
**Sizing:** [[noether-1.1-medium]], [[noether-1.1-large]].
**Progress log:** [[implementation-log]] (append-only; update after every work session). **Resume prompt:** [[phase1-resume-prompt]] (self-contained pickup for Phase I in a fresh session).

---

## 0. Guiding principles

1. **Build in the training-scheme order, not the forward-pass order.** [[training-scheme-1.1]] fixes *what trains when* (P0→P5) precisely because several modules are unstable if built/trained early (learned edges, discovered conservation, diffusion). The build follows that order so each portion lands on a validated foundation.
2. **One system before many.** The regime curriculum ([[training-scheme-1.1]] §"physics-regime curriculum") climbs one generality axis at a time. The code is built and validated on **rung 1–2 (Burgers → 2D incompressible NS)** first; multi-field / coupled / compressible support is added as later rungs, never speculatively.
3. **Reuse Noether 1.0's working code.** The 1.1 branch starts from the complete 1.0 codebase (solvers, GUI, SIREN, projection, graph utilities) — see §1. 1.1 is a *reorganization + extension*, not a rewrite; the NS/RBC solvers and the 2D GUI are directly reusable.
4. **Every portion ships with an acceptance test.** No portion is "done" until its spec file's acceptance criteria pass. Design numbers are estimates until a test says otherwise (the vault's standing caveat).
5. **Nothing in `src/noether11/core/` may assume a specific equation or field set** — design invariant 1 ([[00-noether-1.1-overview]]). Equation-specific code lives only in `data/` (solvers) and in *conditioning*, never in the backbone/tokenizer/decoder.

---

## 1. Repository & branch strategy

**Code home:** `github.com/nonidino/physics-foundation-model` (the vault is design-only). Two organizing branches:

- **`noether-1.0`** — a frozen snapshot of the complete 1.0 build (Burgers + RBC + NS Phase C). **Reference only; never edited.** Anything an agent needs from 1.0 is read here or cherry-picked, not committed here.
- **`noether-1.1`** — the active development branch, reorganized (§ [[impl-repo-scaffold]]). All 1.1 work happens here.

Target 1.1 layout (established by [[impl-repo-scaffold]]):
```
src/noether11/
  core/        # equation-agnostic: tokenizer, streams, descriptors, edges, backbone, decoder
  conserve/    # discovered conservation: discovery search, readouts, projection
  generate/    # post-decoder diffusion
  condition/   # constants → ruler/dial/arrow
  data/        # solvers + datasets, one module per regime (reused/ported from 1.0)
  train/       # phased curriculum harness, losses
  eval/        # metrics, rollout, attractor statistics
  configs/     # medium / large / smoke configs (dataclasses)
gui/           # 2D front-end (ported/extended from 1.0)
scripts/       # entrypoints (generate, train, eval)
tests/         # per-portion acceptance tests
notebooks/     # Colab runners
```

---

## 2. The three macro-phases

The user-facing arc is **GUI → data → model**, but with the build dependency that the GUI and data generation are *prerequisites the model needs*, and the model is built module-by-module in training-scheme order.

### Phase I — Foundations (GUI + data + scaffold)
The model can't be trained or watched without these, so they come first.

| #   | Portion                                              | Spec file                | Depends on |
| --- | ---------------------------------------------------- | ------------------------ | ---------- |
| 0   | Repo scaffold & config system                        | [[impl-repo-scaffold]]   | —          |
| 1   | 2D GUI (interactive fields, model-vs-truth, rollout) | [[impl-gui-2d]]          | 0          |
| 2   | Training-data generation (regime curriculum solvers) | [[impl-data-generation]] | 0          |

### Phase II — Deterministic core (P0–P2 of [[training-scheme-1.1]])
Built and trained in the exact order the training scheme mandates.

| #   | Portion                                                       | Spec file                      | Depends on | Train phase                       |
| --- | ------------------------------------------------------------- | ------------------------------ | ---------- | --------------------------------- |
| 3   | Tokenizer + field streams + descriptors + conditioned queries | [[impl-tokenizer-descriptors]] | 0,2        | P0                                |
| 4   | Edge generation (KNN families + learned scorer)               | [[impl-edge-generation]]       | 3          | P2 (scorer off until motion loss) |
| 5   | Backbone (typed attention, gates, F1 FFN)                     | [[impl-backbone]]              | 3,4        | P1                                |
| 6   | Decoder (coordinate-implicit SIREN increment heads)           | [[impl-decoder]]               | 3,5        | P0/P1                             |
| 7   | Training harness (phased curriculum, losses)                  | [[impl-training-harness]]      | 3–6        | drives P0–P2                      |

### Phase III — Discovery, generation, evaluation (P3–P4)
Turned on only once the deterministic core is stable.

| # | Portion | Spec file | Depends on | Train phase |
|---|---|---|---|---|
| 8 | Discovered conservation (SFA search, whitening, projection) | [[impl-discovered-conservation]] | 5,6,7 | P3 |
| 9 | Post-decoder diffusion (residual gap-fill) | [[impl-diffusion]] | 6,7 | P4 |
| 10 | Evaluation & benchmarks (accuracy, spectra, Nu, conservation drift) | [[impl-eval-benchmarks]] | 6–9 | all phases |

---

## 3. Dependency graph (build order at a glance)

```
0 scaffold ─┬─ 1 GUI ─────────────────────────────────────────────► (visualize any stage)
            └─ 2 data ─┬─ 3 tokenizer/descriptors ─┬─ 5 backbone ─┬─ 6 decoder ─┬─ 7 harness ─┬─ 8 conservation
                       │                            └─ 4 edges ────┘             │            ├─ 9 diffusion
                       └────────────────────────────────────────────────────────┘            └─ 10 eval
```
Critical path: **0 → 2 → 3 → 5 → 6 → 7 → 8**. The GUI (1), edges' learned scorer (4, gated to P2), diffusion (9), and eval (10) hang off the critical path and can be built in parallel by separate agents once their dependency is met.

---

## 4. Milestones (each is a GUI-watchable, test-gated checkpoint)

| M      | Deliverable                                                                               | Gate                                                                           |
| ------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **M0** | Scaffold + configs + CI green; GUI renders a solver trajectory                            | tests pass, GUI shows ground-truth NS rollout                                  |
| **M1** | P0 tokenizer autoencoder passes the reconstruction floor (G0)                             | encode→decode error < data discretization error on held-out turbulent frames   |
| **M2** | P1 single-step deterministic model on 2D NS                                               | single-step rel-$L^2$ + spectral loss converged; GUI model-vs-truth at $t{+}1$ |
| **M3** | P2 multi-step rollout stable; learned edges earning their place                           | push-forward $1{\to}8$ stable; learned graph sparse, non-collapsed             |
| **M4** | P3 divergence discovered on NS; $\sigma_{\nabla\cdot u}(z_{\text{cond}})$ ramps with Mach | held-out drift gate passes; compressible run relaxes it with no code change    |
| **M5** | P4 diffusion restores plume-scale statistics on RBC                                       | rollout-mean Nu + spectra match; GUI shows sharp vs. blurred toggle            |
| **M6** | medium config trained end-to-end across rungs 1–3                                         | [[noether-1.1-medium]] acceptance table                                        |
| **M7** | large config; rungs 4–6 (compressible, particles, MHD)                                    | [[noether-1.1-large]] acceptance table                                         |

---

## 5. How to use this folder (for executing agents)

1. Read this plan and [[implementation-log]] to see what's done and what's next.
2. Pick the next unblocked portion (§2 tables); open its spec file.
3. The spec file is self-contained: objective, the design pages it implements, the interface contract, step-by-step build, and acceptance tests. Follow it; if it under-specifies something, the linked concept page is authoritative.
4. Work on the `noether-1.1` branch only. Add tests under `tests/`. 
5. On completion, append a dated entry to [[implementation-log]] (what shipped, tests passing, what's unblocked next), and check the portion's acceptance box in §2.

---

## See Also
- [[training-scheme-1.1]] — the P0–P5 order this plan's build sequence obeys
- [[00-noether-1.1-overview]] — the architecture this implements
- [[noether-1.0]] — the sized 1.0 model whose code the 1.1 branch starts from
- [[smoke-test-bringup]] — 1.0's bring-up ladder, the template for M0–M3
- [[implementation-log]] — progress tracker
