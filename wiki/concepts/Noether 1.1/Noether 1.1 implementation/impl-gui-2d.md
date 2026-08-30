# Portion 1 — 2D GUI

**Type:** Implementation spec — agent task (folder: Noether 1.1 / Noether 1.1 implementation)
**Portion of:** [[00-implementation-plan]] (Phase I). **Depends on:** [[impl-repo-scaffold]]. **Unblocks:** visual validation of every later milestone (M0–M5).
**Design pages:** [[pfm-interface-design]] (the 5-input interface), [[00-noether-1.1-overview]].

---

## Objective
A browser-based 2D front-end to (a) render any solver-generated trajectory, (b) run a trained 1.1 model in rollout and show **model-vs-truth side by side**, and (c) toggle fields, overlays (vorticity, divergence, plumes), and the diffusion on/off. Extend Noether 1.0's existing `gui/server_rbc.py` + `gui/index_rbc.html` (already does 2D RBC fields/combinations/model-vs-truth) rather than starting fresh.

## Deliverables
- `gui/server.py` (FastAPI, ported/generalized from 1.0's `server_rbc.py`): endpoints to list datasets/checkpoints, stream a ground-truth trajectory, and stream a model rollout given a checkpoint + initial context.
- `gui/index.html` (canvas/WebGL 2D field renderer, generalized from `index_rbc.html`): field selector (velocity magnitude, components, vorticity, temperature, divergence, |B|…), a **model | truth | error** three-panel view, a rollout scrubber/play, a colorbar, and toggles: *diffusion on/off* ([[post-decoder-diffusion-1.1]] "optional at inference"), *conservation-projection on/off*, *learned-edges overlay* (draw the learned long-range edges from [[edge-generation-1.1]]).
- `.claude/launch.json` entry so the preview harness can start it.

## Interface contract
- `GET /datasets` → list; `GET /checkpoints` → list.
- `GET /trajectory?dataset=…&traj=…` → frames (field arrays) as compact binary/JSON.
- `POST /rollout` body `{checkpoint, dataset, traj, n_steps, diffusion:bool, project:bool}` → predicted frames + per-step metrics (rel-$L^2$, divergence, conserved-quantity drift).
- Model is loaded via `noether11` public API (`load_checkpoint`, `rollout`) — the GUI must not reimplement inference.

## Build steps
1. Port `server_rbc.py` → `server.py`; replace RBC-specific field assumptions with the model's declared field set (read from the checkpoint's config / field descriptors $\theta_f$).
2. Generalize the HTML renderer to N fields with a dropdown; add the three-panel model|truth|error layout.
3. Wire the three toggles to `/rollout` params.
4. Add the learned-edges overlay (fetch edge list from a debug endpoint; draw as lines).
5. Register in `.claude/launch.json`; verify with the preview tools.

## Acceptance tests
- With a solver-only dataset (no model yet) the GUI renders a ground-truth NS trajectory and scrubs smoothly (this is milestone **M0**).
- Once a P1 checkpoint exists, model|truth|error renders and the error panel is quantitatively consistent with the eval script's rel-$L^2$ ([[impl-eval-benchmarks]]).
- Diffusion toggle visibly sharpens plume-scale texture on RBC (milestone **M5**).

## Pitfalls
- Keep the renderer field-agnostic — it must not hardcode "velocity+temperature"; read the field set from the model. This is the GUI-level image of design invariant 1.
- Large trajectories: stream/decimate frames; don't ship the whole rollout in one payload.

## See Also
- [[impl-eval-benchmarks]] (shares the metric definitions the error panel shows) · [[post-decoder-diffusion-1.1]] (the diffusion toggle) · [[edge-generation-1.1]] (edge overlay)
