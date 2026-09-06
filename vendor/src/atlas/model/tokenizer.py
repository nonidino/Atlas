"""Per-agent patch tokenizer: fields -> tokens.

    derived features -> patch unfold -> linear -> Fourier position -> FiLM(cond)

Two things about this file are load-bearing.

**The tokenizer is per-agent and completely independent across agents.** No
cross-agent mixing happens here; the first place two agents can influence each
other is the edge layer. That is what makes agents separable computations and
what will later let one expert be swapped without touching another (phase-1 spec
section 1).

**Absolute position, not relative.** The geometry is not translation invariant --
the throat is *at* z = 0.40 -- so the Fourier encoding is of the token's absolute
body-frame centroid. In a homogeneous turbulence box that would be a mistake; here
the opposite is the mistake.

Weights are shared per TOKENIZER GROUP, not per agent: `d`, `f` and `g` are the
same kind of thing (same fields, same governing family), so they share one
instance, which is both a parameter saving and a mild inductive bias. `c` never
shares -- different field semantics (stress vs momentum) and a different patch
shape (phase-1 spec 3.4 and its pitfall list).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .utils import near_zero_


def fourier_encode(r: Tensor, n_bands: int) -> Tensor:
    """gamma(r) = [sin(2^k pi r), cos(2^k pi r)]_{k=0..K-1}; r [..., 2] in [-1,1]."""
    k = 2.0 ** torch.arange(n_bands, device=r.device, dtype=r.dtype) * math.pi
    a = r[..., None, :] * k[:, None]                     # [..., K, 2]
    a = a.flatten(-2)                                    # [..., K*2]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class AgentTokenizer(nn.Module):
    """One instance per tokenizer group.

    forward(x, keep_index, centroids_norm, cond)
        x            [B, C_total, Nz, Ny]  raw ‖ derived ‖ mask (see features.py)
        keep_index   [P]                   kept patches, flat index into the patch grid
        centroids_norm [P, 2]              token positions in [-1, 1]^2
        cond         [B, n_cond]
        ->           [B, P, d_model]
    """

    def __init__(self, c_total: int, patch: tuple[int, int], d_model: int,
                 n_fourier: int, n_cond: int):
        super().__init__()
        self.patch = patch
        self.c_total = c_total
        pz, py = patch
        self.proj = nn.Linear(c_total * pz * py, d_model)
        self.pos = nn.Linear(4 * n_fourier, d_model, bias=False)
        self.n_fourier = n_fourier
        self.film = nn.Sequential(
            nn.Linear(n_cond, d_model), nn.SiLU(), nn.Linear(d_model, 2 * d_model)
        )
        # near-zero FiLM head: an untrained tokenizer is (almost) the
        # unconditioned one, gamma ~ 1 and beta ~ 0, so conditioning earns its
        # effect. Near-zero rather than zero -- see utils.near_zero_.
        near_zero_(self.film[-1])

    def unfold(self, x: Tensor, keep_index: Tensor) -> Tensor:
        """[B, C, Nz, Ny] -> [B, P, C*pz*py], row-major over the patch grid.

        A reshape, not nn.Unfold: every grid divides evenly by its patch, so the
        patches are a partition and the general im2col machinery buys nothing."""
        B, C, nz, ny = x.shape
        pz, py = self.patch
        npz, npy = nz // pz, ny // py
        x = x.view(B, C, npz, pz, npy, py).permute(0, 2, 4, 1, 3, 5).reshape(B, npz * npy, -1)
        return x.index_select(1, keep_index)

    def forward(self, x: Tensor, keep_index: Tensor, centroids_norm: Tensor,
                cond: Tensor) -> Tensor:
        z = self.proj(self.unfold(x, keep_index))                    # [B, P, d]
        z = z + self.pos(fourier_encode(centroids_norm, self.n_fourier))[None]
        gamma, beta = self.film(cond).chunk(2, dim=-1)               # [B, d] each
        return (1.0 + gamma[:, None, :]) * z + beta[:, None, :]


__all__ = ["AgentTokenizer", "fourier_encode"]
