"""Typed message passing over the declared multigraph (phase-1 spec 2.3).

    m_i = sum_tau sum_{j in N_tau(i)} alpha^tau_ij W_V^tau [h_j ‖ a_ij]

One projection triple per edge TYPE, messages summed over types. Same mechanism
as Noether 1.1's backbone with one simplification: the type is declared, so there
is no learned gate deciding which type is active -- an edge either has type tau or
it does not.

Two properties this layer must keep:

* **A token with no incoming edge is only touched by the per-token FFN.** No
  global mixing hides here; that is what makes the agent-isolation test
  meaningful, and it is why whole-vehicle information has to travel through the
  bottleneck instead (`hierarchy.py`).
* **Per-type edge slices are precomputed on the graph**, never recomputed or
  masked per layer.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import AtlasConfig
from .edges import Graph


def segment_softmax(logit: Tensor, index: Tensor, n: int) -> Tensor:
    """Softmax over the edges sharing a destination. logit [B, E, H] -> same shape."""
    B, E, H = logit.shape
    idx = index.view(1, E, 1).expand(B, E, H)
    mx = torch.full((B, n, H), -1e30, device=logit.device, dtype=logit.dtype)
    mx = mx.scatter_reduce(1, idx, logit, reduce="amax", include_self=True).detach()
    ex = torch.exp(logit - mx.gather(1, idx))
    den = torch.zeros((B, n, H), device=logit.device, dtype=logit.dtype).scatter_add(1, idx, ex)
    return ex / den.gather(1, idx).clamp_min(1e-20)


class TypedMPLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, d_edge: int, n_types: int, ffn_mult: int):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.q = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(n_types))
        self.k = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(n_types))
        self.v = nn.ModuleList(nn.Linear(d + d_edge, d, bias=False) for _ in range(n_types))
        self.o = nn.Linear(d, d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d), nn.SiLU(), nn.Linear(ffn_mult * d, d)
        )

    def forward(self, h: Tensor, g: Graph) -> Tensor:
        B, P, d = h.shape
        x = self.norm1(h)
        m = torch.zeros(B, P, d, device=h.device, dtype=h.dtype)

        for t, sel in g.type_slices.items():
            src = g.edge_index[0].index_select(0, sel)
            dst = g.edge_index[1].index_select(0, sel)
            attr = g.edge_attr.index_select(0, sel).unsqueeze(0).expand(B, -1, -1)

            q = self.q[t](x).index_select(1, dst).view(B, -1, self.h, self.dh)
            k = self.k[t](x).index_select(1, src).view(B, -1, self.h, self.dh)
            v = self.v[t](torch.cat([x.index_select(1, src), attr], dim=-1))
            v = v.view(B, -1, self.h, self.dh)

            a = segment_softmax((q * k).sum(-1) / math.sqrt(self.dh), dst, P)
            msg = (a.unsqueeze(-1) * v).reshape(B, -1, d)
            m = m.index_add(1, dst, msg)

        h = h + self.o(m)
        return h + self.ffn(self.norm2(h))


class TypedMessagePassing(nn.Module):
    """`n_mp_layers` stacked typed layers.

    Depth is the graph diameter by default (4): information from any agent can
    reach any other exactly once. Depth is also exactly the reach -- with L
    layers an agent influences only agents within L hops, which is what the
    isolation test exercises."""

    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        m = cfg.model
        self.layers = nn.ModuleList(
            TypedMPLayer(m.d_model, m.n_heads, m.d_edge, len(cfg.edge_types), m.d_ffn_mult)
            for _ in range(m.n_mp_layers)
        )

    def forward(self, h: Tensor, g: Graph) -> Tensor:
        for layer in self.layers:
            h = layer(h, g)
        return h


__all__ = ["TypedMessagePassing", "TypedMPLayer", "segment_softmax"]
