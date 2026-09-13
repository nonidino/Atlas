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
loaded frozen and used only as the **learned** expert a fluid window can be
switched to.

The copy is included so that the demo runs on a machine with no network access
to the hub. It is byte-identical to the published snapshot named in
`SOURCE_COMMITS`, laid out in the hub's own cache format so that
`from_pretrained("camlab-ethz/Poseidon-T")` resolves it offline.

## Why the model *code* is not bundled

`scOT`, the model class this checkpoint is loaded through, lives at
[github.com/camlab-ethz/poseidon](https://github.com/camlab-ethz/poseidon) and
that repository **publishes no licence file**, so it is not ours to
redistribute. The launchers install it at a pinned commit from the upstream
source archive instead. That is the one thing this bundle fetches from outside
pip's own package index at install time; the weights above are already here.
**If that fetch fails, the classical column still runs**, and the page greys
the learned switch out and says why.

## What this means for the demo

RaceLab's point is the switch: any fluid window can be flipped between the
classical solver and this checkpoint, and the speed and accuracy consequences
are shown. The page states what that switch is worth here rather than implying
more — a learned window is asked for a time step a thirty-second of the one the
checkpoint was trained to take, and a learned column leaves the fluid expert's
declared envelope within a handful of macro-steps. None of that is a claim about
production use, and the licence would not permit one: a production version would
need a permissively licensed expert instead, which the project records as
**W120**. The demo names the expert and its licence on screen, in the header, at
all times.
