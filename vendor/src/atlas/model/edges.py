"""Declared/geometric edge instantiation -- Mechanism A, and only Mechanism A.

There is no KNN over the domain, no learned scorer and no Gumbel sampling in
this file. That machinery belongs to Noether 1.1's *discovery* setting and to
Atlas's deferred Mechanism B (edge-generation-atlas-0.1); here the graph is
declared, so materializing it is deterministic arc-length matching at shared
interface geometry. Determinism is the point: it makes every later failure
attributable to something other than the graph.

Matching (phase-1 spec 2.2):

  1. boundary tokens = centroid within one patch diagonal of the curve (done in
     `geometry/layout.py`, which also records arc length and interface normal);
  2. connect each boundary token to its K=2 nearest partners in arc length,
     matched from BOTH sides and unioned -- K=2 both ways is what keeps a
     resolution mismatch across an interface from leaving dangling tokens;
  3. store both directions with the normal flipped, because heat flowing from
     chamber to wall is not the same operation as the wall constraining the gas.

Matching is per BRANCH. An upper and a lower wall are two branches of one
interface, and arc length restarts on each; matching that ignored the branch
would bond an upper-wall token to a lower-wall one at the same arc length.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..config import AtlasConfig
from ..geometry.layout import PatchLayout

# edge_attr column layout, d_edge = 16
ATTR_LAYOUT = (
    "ds", "s_src", "s_dst", "n_z", "n_y", "ell",
    "dz", "dy", "dr", "prox_src", "prox_dst",
    "type_0", "type_1", "type_2", "type_3", "type_4",
)


@dataclass(frozen=True, eq=False)
class Graph:
    edge_index: Tensor                  # [2, E] global token indices, message src -> dst
    edge_type: Tensor                   # [E] index into cfg.edge_types
    edge_attr: Tensor                   # [E, d_edge]
    type_slices: dict[int, Tensor]      # type id -> positions in [0, E)
    n_pairs: dict[int, int]             # declared-edge index -> matched token pairs
    fingerprint: str

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(self, device) -> "Graph":
        return Graph(
            self.edge_index.to(device), self.edge_type.to(device), self.edge_attr.to(device),
            {k: v.to(device) for k, v in self.type_slices.items()}, self.n_pairs, self.fingerprint,
        )


def _voronoi_width(s: np.ndarray, total: float) -> np.ndarray:
    """Arc-length extent each boundary token owns, so a flux can later be
    INTEGRATED over the interface rather than merely evaluated at a point."""
    if s.size == 1:
        return np.array([total])
    order = np.argsort(s, kind="stable")
    ss = s[order]
    mid = 0.5 * (ss[1:] + ss[:-1])
    lo = np.concatenate([[ss[0]], mid])
    hi = np.concatenate([mid, [ss[-1]]])
    w = np.empty_like(s)
    w[order] = np.maximum(hi - lo, 1e-9)
    return w


def _knn_1d(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Pairs (i, j) linking each entry of `a` to its k nearest in `b`, by |a-b|."""
    k = min(k, b.size)
    d = np.abs(a[:, None] - b[None, :])
    j = np.argsort(d, axis=1, kind="stable")[:, :k]
    i = np.repeat(np.arange(a.size), k)
    return np.stack([i, j.ravel()], axis=1)


def _build(layout: PatchLayout, cfg: AtlasConfig) -> Graph:
    cen = layout.centroids.numpy()
    n_types = len(cfg.edge_types)
    src_l, dst_l, typ_l, attr_l = [], [], [], []
    n_pairs: dict[int, int] = {}

    for ei, e in enumerate(layout.edges):
        b = layout.boundary[ei]
        L = max(b.length, 1e-9)
        s_s, s_d = b.src_s.numpy(), b.dst_s.numpy()
        br_s, br_d = b.src_branch.numpy(), b.dst_branch.numpy()
        tok_s, tok_d = b.src_tokens.numpy(), b.dst_tokens.numpy()

        # per-token interface width, computed within (branch, side)
        w_s, w_d = np.zeros_like(s_s), np.zeros_like(s_d)
        for br, s, out in ((br_s, s_s, w_s), (br_d, s_d, w_d)):
            for bi in np.unique(br):
                m = np.flatnonzero(br == bi)
                out[m] = _voronoi_width(s[m], L)

        pairs = set()
        for bi in np.union1d(np.unique(br_s), np.unique(br_d)):
            si = np.flatnonzero(br_s == bi)
            di = np.flatnonzero(br_d == bi)
            if si.size == 0 or di.size == 0:
                continue                      # a branch only one side reaches
            for p in _knn_1d(s_s[si], s_d[di], cfg.K_match):
                pairs.add((int(si[p[0]]), int(di[p[1]])))
            for p in _knn_1d(s_d[di], s_s[si], cfg.K_match):
                pairs.add((int(si[p[1]]), int(di[p[0]])))

        if not pairs:
            raise AssertionError(f"edge {e.src}-{e.dst}: no token pairs matched")
        n_pairs[ei] = len(pairs)
        pi = np.array(sorted(pairs), dtype=np.int64)          # sorted -> deterministic
        i, j = pi[:, 0], pi[:, 1]

        ti, tj = tok_s[i], tok_d[j]
        dr = cen[tj] - cen[ti]                                # [n, 2]
        dr_n = np.linalg.norm(dr, axis=1)
        # orient the interface normal from i's side toward j's side; the reverse
        # direction below gets it flipped, which is the whole point of storing
        # both directions rather than one symmetric edge.
        nrm = b.src_nrm.numpy()[i]
        nrm = nrm * np.sign((nrm * dr).sum(1, keepdims=True) + 1e-12)
        ell = np.minimum(w_s[i], w_d[j]) / L
        ds = (s_d[j] - s_s[i]) / L
        prox_i, prox_j = b.src_dist.numpy()[i], b.dst_dist.numpy()[j]

        base = np.stack([
            ds, s_s[i] / L, s_d[j] / L, nrm[:, 0], nrm[:, 1], ell,
            dr[:, 0] / L, dr[:, 1] / L, dr_n / L, prox_i, prox_j,
        ], axis=1)
        rev = np.stack([
            -ds, s_d[j] / L, s_s[i] / L, -nrm[:, 0], -nrm[:, 1], ell,
            -dr[:, 0] / L, -dr[:, 1] / L, dr_n / L, prox_j, prox_i,
        ], axis=1)

        for tid in e.type_ids:
            onehot = np.zeros((base.shape[0], n_types))
            onehot[:, tid] = 1.0
            for a_from, a_to, attrs in ((ti, tj, base), (tj, ti, rev)):
                src_l.append(a_from); dst_l.append(a_to)
                typ_l.append(np.full(a_from.size, tid, dtype=np.int64))
                attr_l.append(np.concatenate([attrs, onehot], axis=1))

    edge_index = torch.from_numpy(np.stack([np.concatenate(src_l), np.concatenate(dst_l)]))
    edge_type = torch.from_numpy(np.concatenate(typ_l))
    edge_attr = torch.from_numpy(np.concatenate(attr_l, axis=0)).float()
    if edge_attr.shape[1] != cfg.model.d_edge:
        raise AssertionError(
            f"edge_attr width {edge_attr.shape[1]} != d_edge {cfg.model.d_edge}; "
            f"ATTR_LAYOUT has {len(ATTR_LAYOUT)} columns"
        )

    # Precomputed ONCE per graph. Doing this per layer instead (a `.any()` mask
    # check per type per layer) is a host-device sync on GPU that costs more than
    # the computation it guards -- measured in the parallel Noether track,
    # implementation-log 2026-07-19.
    type_slices = {
        t: torch.from_numpy(np.flatnonzero(edge_type.numpy() == t).astype(np.int64))
        for t in range(n_types)
    }
    type_slices = {t: v for t, v in type_slices.items() if v.numel() > 0}

    return Graph(edge_index, edge_type, edge_attr, type_slices, n_pairs, layout.fingerprint)


_GRAPHS: dict[str, Graph] = {}


def build_graph(layout: PatchLayout, cfg: AtlasConfig) -> Graph:
    """Deterministic, no learned component, memoized on the layout fingerprint."""
    key = layout.fingerprint
    if key not in _GRAPHS:
        _GRAPHS[key] = _build(layout, cfg)
    return _GRAPHS[key]


def clear_graph_cache() -> None:
    _GRAPHS.clear()


__all__ = ["Graph", "build_graph", "clear_graph_cache", "ATTR_LAYOUT"]
