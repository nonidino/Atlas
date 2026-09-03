# The Poseidon-T checkpoint in this bundle

`vendor/hf-cache/hub/models--camlab-ethz--Poseidon-T/` is a copy of the
**Poseidon-T** checkpoint published by the Computational and Applied Mathematics
Laboratory at ETH Zürich on the Hugging Face hub as
[`camlab-ethz/Poseidon-T`](https://huggingface.co/camlab-ethz/Poseidon-T).

- **20.8 M parameters**, a scalable Operator Transformer (scOT) trained on 2-D
  compressible and incompressible flow, fixed at $128 \times 128$.
- **Licence: CC-BY-NC-4.0** — Creative Commons Attribution-NonCommercial 4.0.
  **Research and other non-commercial use only.** No commercial or production
  use of these weights, or of anything derived from them.
- **Attribution:** Herde, Leoni, Molinaro, Mishra et al., *Poseidon: Efficient
  Foundation Models for PDEs* (2024), camlab-ethz. Paper:
  [arXiv:2405.19101](https://arxiv.org/abs/2405.19101).

Nothing in this project trains, fine-tunes or modifies these weights. They are
loaded frozen, with `requires_grad_(False)` on every parameter, and the
demonstration takes a gradient **through** them with respect to the design
variables only — no parameter gradient is accumulated.

The copy is included so that the demo runs on a machine with no network access
to the hub. It is byte-identical to the published snapshot
`ec976ed5d25883ec9db4e486ebbeeefa9e08303b`, laid out in the hub's own cache
format so that `from_pretrained("camlab-ethz/Poseidon-T")` resolves it offline.

## Why the model *code* is not bundled

`scOT`, the model class this checkpoint is loaded through, lives at
[github.com/camlab-ethz/poseidon](https://github.com/camlab-ethz/poseidon) and
that repository **publishes no licence file**, so it is not ours to
redistribute. The launchers install it at a pinned commit from the upstream
source archive instead. That is the one thing this bundle fetches from the
network at install time; the weights above are already here.

## What this means for the demo

The composed column's headline — faster than the undivided classical solver —
rests on this checkpoint. It is therefore a **research** result, and the
project's own survey of donors
(`expert-donor-survey`) records the consequence plainly: any production use
needs a permissively licensed expert instead (Walrus, MIT; DPOT, Apache-2.0;
GPhyT, MIT), which is opened as **W120**. The demo names the expert and its
licence on screen for the same reason.
