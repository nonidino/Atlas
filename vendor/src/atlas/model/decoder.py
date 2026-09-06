"""The tokenizer run backwards, predicting a field INCREMENT.

    x_hat(t + dt) = x(t) + dt * D_theta(Z)

Increment prediction is the single most load-bearing choice for rollout
stability (phase-1 spec 3.6). It makes the identity map the zero-output default,
so an untrained model degrades to "nothing changes" rather than to noise, and it
keeps target magnitudes O(1) after nondimensionalization.

Weights are shared per tokenizer group, same as the encoder: `d`, `f`, `g` decode
the same five fields on the same patch shape.

Dropped patches (agent `g`'s plume hole) decode to a zero increment: those cells
are agent `f`'s, and writing anything there would be writing over another agent.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .utils import near_zero_


class AgentDecoder(nn.Module):
    """[B, P, d] -> [B, C_raw, Nz, Ny]."""

    def __init__(self, c_raw: int, patch: tuple[int, int], d_model: int, out_scale: float):
        super().__init__()
        self.c_raw, self.patch = c_raw, patch
        pz, py = patch
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, c_raw * pz * py)
        near_zero_(self.out, out_scale)

    def forward(self, tokens: Tensor, keep_index: Tensor, n_patch: tuple[int, int],
                grid: tuple[int, int]) -> Tensor:
        B = tokens.shape[0]
        npz, npy = n_patch
        nz, ny = grid
        pz, py = self.patch
        flat = self.out(self.norm(tokens))                       # [B, P, C*pz*py]
        full = flat.new_zeros(B, npz * npy, self.c_raw * pz * py)
        full = full.index_copy(1, keep_index, flat)
        return (full.view(B, npz, npy, self.c_raw, pz, py)
                    .permute(0, 3, 1, 4, 2, 5)
                    .reshape(B, self.c_raw, nz, ny))

    @torch.no_grad()
    def zero_output_(self) -> None:
        """Make the increment path an exact no-op -- used by the M3 test that the
        scaffold hides no transformation in pooling/unpooling."""
        self.out.weight.zero_()
        self.out.bias.zero_()


__all__ = ["AgentDecoder"]
