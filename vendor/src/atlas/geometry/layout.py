"""PatchLayout -- the static token geometry, built once and memoized.

A layout depends only on the config: which cells belong to which token, where
each token sits in the body frame, how much area it carries, and which tokens sit
on which declared interface. None of that is a function of field values or of
model weights, so it is built once per config fingerprint and reused (phase-1
spec 3.3 and its pitfall list; recomputing static geometry every step was a
measured cost in the parallel Noether track, implementation-log 2026-07-19).

Token ordering is fixed and global: agents in config order, and within an agent
row-major over the patch grid (`i_pz * n_py + i_py`), skipping dropped patches.
Every downstream index -- edge_index, agent_slice, the decoder's scatter -- is
built against that ordering, so it must not change silently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..config import AtlasConfig, load_config
from .domains import AgentSpec, EdgeSpec, build_agents, build_edges


@dataclass(frozen=True, eq=False)
class EdgeBoundary:
    """Boundary tokens of one declared edge, kept per side and per branch.

    `branch` matters: an interface that appears twice by symmetry (upper and
    lower wall) is two branches, and arc length restarts on each. Matching that
    ignored the branch would happily bond an upper-wall token to a lower-wall one
    at the same arc length."""
    src_tokens: Tensor      # [n_src] global token indices
    src_branch: Tensor      # [n_src]
    src_s: Tensor           # [n_src] arc length along its branch
    src_dist: Tensor        # [n_src] distance from centroid to the curve
    src_nrm: Tensor         # [n_src, 2] unit interface normal at the projection
    dst_tokens: Tensor
    dst_branch: Tensor
    dst_s: Tensor
    dst_dist: Tensor
    dst_nrm: Tensor
    length: float           # total curve length, for normalizing arc lengths


@dataclass(frozen=True, eq=False)
class PatchLayout:
    """Static token geometry for one config.

    centroids   [P, 2]  body-frame (z, y) of each token
    cell_area   [P]     physical area carried by the token (ACTIVE cells only, so
                        a wall-exterior token reports 0 -- any later flux
                        integration must weight by this, not by token count)
    agent_id    [P]     index into `agents`
    active_frac [P]     fraction of the token's cells inside the domain
    patch_diag  [P]     token bounding diagonal; the boundary-token radius
    """
    centroids: Tensor
    centroids_norm: Tensor            # centroids mapped to [-1, 1]^2 over the whole
                                      # modelled box. The Fourier band k=5 is 32*pi,
                                      # which on raw metres (z spans 8 m) aliases
                                      # between neighbouring tokens; normalize first.
    cell_area: Tensor
    agent_id: Tensor
    active_frac: Tensor
    patch_diag: Tensor
    agents: tuple[AgentSpec, ...]
    edges: tuple[EdgeSpec, ...]
    agent_slice: dict[str, slice]
    keep_index: dict[str, Tensor]     # kept tokens -> flat index in the FULL patch grid
    n_patch: dict[str, tuple[int, int]]
    boundary: dict[int, EdgeBoundary]
    fingerprint: str

    # -- phase-1 spec 3.3 names, derived from `boundary` -----------------------
    @property
    def boundary_of(self) -> dict[int, Tensor]:
        return {i: torch.cat([b.src_tokens, b.dst_tokens]) for i, b in self.boundary.items()}

    @property
    def arc_s(self) -> dict[int, Tensor]:
        return {i: torch.cat([b.src_s, b.dst_s]) for i, b in self.boundary.items()}

    @property
    def n_tokens(self) -> int:
        return int(self.centroids.shape[0])

    def agent(self, name: str) -> AgentSpec:
        for a in self.agents:
            if a.name == name:
                return a
        raise KeyError(name)

    def tokens_of(self, name: str) -> slice:
        return self.agent_slice[name]


def _patch_reduce(a: AgentSpec):
    """Per-token geometry for one agent, before dropped patches are removed.

    Returns (centroid [Q,2], area [Q], active_frac [Q], diag [Q], drop [Q]) with
    Q = n_pz * n_py in row-major order."""
    g, (pz, py) = a.grid, a.patch
    npz, npy = a.n_patch
    dz = float(g.z_c[1] - g.z_c[0])

    def blocks(x):                       # [n_z, n_y] -> [npz, npy, pz*py]
        return x.reshape(npz, pz, npy, py).transpose(0, 2, 1, 3).reshape(npz, npy, pz * py)

    z_full = np.broadcast_to(g.z_c[:, None], g.y_c.shape)
    zb, yb = blocks(z_full), blocks(g.y_c)
    ab, act, hole = blocks(g.area), blocks(g.active), blocks(g.in_hole)

    w = ab * act                                            # active-area weight
    tot = w.sum(-1)
    fallback = ab.sum(-1)
    wz = np.where(tot > 0, (zb * w).sum(-1) / np.maximum(tot, 1e-30),
                  (zb * ab).sum(-1) / fallback)
    wy = np.where(tot > 0, (yb * w).sum(-1) / np.maximum(tot, 1e-30),
                  (yb * ab).sum(-1) / fallback)

    # bounding diagonal: pz cells in z, and the tallest column of dy in y
    dy = (ab / dz).reshape(npz, npy, pz, py).sum(-1).max(-1)   # [npz, npy]
    diag = np.hypot(pz * dz, dy)

    return (
        np.stack([wz.ravel(), wy.ravel()], axis=1),
        tot.ravel(),
        act.mean(-1).ravel(),
        diag.ravel(),
        hole.all(-1).ravel(),                                # drop: entirely inside a hole
    )


def _build(cfg: AtlasConfig) -> PatchLayout:
    agents = build_agents(cfg)
    edges = build_edges(cfg)

    cen, area, frac, diag, aid, keep_index, slices, npatch = [], [], [], [], [], {}, {}, {}
    p0 = 0
    for k, a in enumerate(agents):
        c, ar, fr, dg, drop = _patch_reduce(a)
        keep = np.flatnonzero(~drop)
        cen.append(c[keep]); area.append(ar[keep]); frac.append(fr[keep]); diag.append(dg[keep])
        aid.append(np.full(keep.size, k, dtype=np.int64))
        keep_index[a.name] = torch.from_numpy(keep.astype(np.int64))
        slices[a.name] = slice(p0, p0 + keep.size)
        npatch[a.name] = a.n_patch
        p0 += keep.size

    centroids = np.concatenate(cen, axis=0)
    geo = cfg.geometry
    z_lo, z_hi = geo.atmos_front_z[0], geo.plume_z[1]
    norm = np.stack([
        2.0 * (centroids[:, 0] - z_lo) / (z_hi - z_lo) - 1.0,
        centroids[:, 1] / geo.farfield_halfwidth,
    ], axis=1)
    layout_args = dict(
        centroids=torch.from_numpy(centroids).float(),
        centroids_norm=torch.from_numpy(norm).float(),
        cell_area=torch.from_numpy(np.concatenate(area)).float(),
        agent_id=torch.from_numpy(np.concatenate(aid)),
        active_frac=torch.from_numpy(np.concatenate(frac)).float(),
        patch_diag=torch.from_numpy(np.concatenate(diag)).float(),
    )
    diag_all = np.concatenate(diag)

    # -- boundary tokens per declared edge ---------------------------------
    boundary: dict[int, EdgeBoundary] = {}
    for ei, e in enumerate(edges):
        side = {}
        for role, name in (("src", e.src), ("dst", e.dst)):
            sl = slices[name]
            pts = centroids[sl]
            br, s, d, nrm = e.curve.project(pts)
            tol = cfg.tol_patch_diag * diag_all[sl]
            hit = np.flatnonzero(d < tol)
            if hit.size == 0:
                raise AssertionError(
                    f"edge {e.src}-{e.dst}: agent {name} has no token within "
                    f"{cfg.tol_patch_diag} patch diagonals of the interface"
                )
            side[role] = (
                torch.from_numpy((hit + sl.start).astype(np.int64)),
                torch.from_numpy(br[hit].astype(np.int64)),
                torch.from_numpy(s[hit]).float(),
                torch.from_numpy(d[hit] / tol[hit]).float(),   # normalized by the token's own tol
                torch.from_numpy(np.ascontiguousarray(nrm[hit])).float(),
            )
        boundary[ei] = EdgeBoundary(*side["src"], *side["dst"], length=e.curve.length)

    return PatchLayout(
        agents=agents, edges=edges, agent_slice=slices, keep_index=keep_index,
        n_patch=npatch, boundary=boundary, fingerprint=cfg.fingerprint(), **layout_args,
    )


_LAYOUTS: dict[str, PatchLayout] = {}


def build_layout(cfg: AtlasConfig | None = None) -> PatchLayout:
    """Memoized on the config fingerprint, not on object identity -- two separate
    `load_config()` calls must hit the same layout."""
    cfg = cfg if cfg is not None else load_config()
    key = cfg.fingerprint()
    if key not in _LAYOUTS:
        _LAYOUTS[key] = _build(cfg)
    return _LAYOUTS[key]


def clear_layout_cache() -> None:
    _LAYOUTS.clear()


__all__ = ["PatchLayout", "EdgeBoundary", "build_layout", "clear_layout_cache"]
