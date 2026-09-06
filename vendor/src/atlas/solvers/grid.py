"""Structured curvilinear finite-volume grid.

One `Block` per agent (two for the two-panel agents `c` and `d`). It carries cell
volumes, face normals and face areas, so the *same* solver serves the straight
external domains and the contracting nozzle without special-casing
(impl-atlas-0.1-phase0-scope-and-data §3.1).

The grid is built from the SAME node positions the Phase-1 tokenizer patches, so
the corpus lands on exactly the cells the model reads. There is no interpolation
step between solver and model, by construction — an interpolation is precisely
what interface fluxes do not survive.

Index convention: `i` runs along z (0 .. n_z-1), `j` along y. Face arrays are
sized [n_z+1, n_y] for i-faces and [n_z, n_y+1] for j-faces. `n` is the unit
normal pointing in the +i / +j direction; `area` is the face length (per unit
depth, since this is a planar slab).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import AgentCfg, AtlasConfig, GeometryCfg
from ..geometry.contours import nozzle_inner, shell_outer


@dataclass(frozen=True, eq=False)
class Block:
    """A structured curvilinear FV block.

    nodes    [n_z+1, n_y+1, 2]  corner coordinates (z, y)
    blanked  [n_z, n_y]         cells that belong to a DIFFERENT agent (agent `g`'s
                                plume hole). Their faces are treated as boundaries;
                                nothing is integrated inside them.
    """
    name: str
    nodes: np.ndarray
    blanked: np.ndarray

    # -- metrics (derived once) --------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self.nodes.shape[0] - 1, self.nodes.shape[1] - 1

    def _metric(self):
        cache = getattr(self, "_m", None)
        if cache is not None:
            return cache
        p = self.nodes
        # corners of every cell, counter-clockwise
        c00, c10, c11, c01 = p[:-1, :-1], p[1:, :-1], p[1:, 1:], p[:-1, 1:]
        # shoelace area of the quad (positive for CCW ordering)
        vol = 0.5 * np.abs(
            (c00[..., 0] * c10[..., 1] - c10[..., 0] * c00[..., 1])
            + (c10[..., 0] * c11[..., 1] - c11[..., 0] * c10[..., 1])
            + (c11[..., 0] * c01[..., 1] - c01[..., 0] * c11[..., 1])
            + (c01[..., 0] * c00[..., 1] - c00[..., 0] * c01[..., 1])
        )
        centroid = 0.25 * (c00 + c10 + c11 + c01)

        # i-faces: the constant-i edges, spanning j -> j+1
        ei = p[:, 1:, :] - p[:, :-1, :]                      # [n_z+1, n_y, 2]
        ai = np.linalg.norm(ei, axis=-1)
        ni = np.stack([ei[..., 1], -ei[..., 0]], axis=-1) / np.maximum(ai, 1e-300)[..., None]
        # j-faces: the constant-j edges, spanning i -> i+1
        ej = p[1:, :, :] - p[:-1, :, :]                      # [n_z, n_y+1, 2]
        aj = np.linalg.norm(ej, axis=-1)
        nj = np.stack([-ej[..., 1], ej[..., 0]], axis=-1) / np.maximum(aj, 1e-300)[..., None]

        m = dict(vol=vol, centroid=centroid, n_i=ni, a_i=ai, n_j=nj, a_j=aj)
        object.__setattr__(self, "_m", m)
        return m

    vol = property(lambda self: self._metric()["vol"])
    centroid = property(lambda self: self._metric()["centroid"])
    n_i = property(lambda self: self._metric()["n_i"])
    a_i = property(lambda self: self._metric()["a_i"])
    n_j = property(lambda self: self._metric()["n_j"])
    a_j = property(lambda self: self._metric()["a_j"])

    def closure_residual(self) -> float:
        """max |sum_faces n*A| per cell. Zero to machine precision for any closed
        control volume; the first M1 test, because every conservation property
        downstream rests on it."""
        s = (self.n_i[1:] * self.a_i[1:, :, None] - self.n_i[:-1] * self.a_i[:-1, :, None]
             + self.n_j[:, 1:] * self.a_j[:, 1:, None] - self.n_j[:, :-1] * self.a_j[:, :-1, None])
        return float(np.abs(s).max())

    def min_spacing(self) -> np.ndarray:
        """Per-cell min(dz, dy), for the CFL bound.

        An i-face is at constant i and spans j -> j+1, so its length is the cell's
        TRANSVERSE extent; a j-face's length is the cell's AXIAL extent. The
        effective spacing in each direction is then volume / the opposite extent,
        which is exact for a parallelogram and right to O(slope^2) otherwise."""
        y_extent = 0.5 * (self.a_i[:-1] + self.a_i[1:])
        z_extent = 0.5 * (self.a_j[:, :-1] + self.a_j[:, 1:])
        dz = self.vol / np.maximum(y_extent, 1e-300)
        dy = self.vol / np.maximum(z_extent, 1e-300)
        return np.minimum(dz, dy)


# --------------------------------------------------------------------------
# agent -> block(s), from the same profiles the model geometry uses
# --------------------------------------------------------------------------


def _nodes_uniform(a: AgentCfg, z_n, geo):
    yh = a.y_halfwidth
    y = np.linspace(-yh, yh, a.grid[1] + 1)
    return np.stack([np.broadcast_to(z_n[:, None], (z_n.size, y.size)),
                     np.broadcast_to(y[None, :], (z_n.size, y.size))], axis=-1)


def _nodes_body_fitted(a: AgentCfg, z_n, geo):
    h = nozzle_inner(z_n, geo)[:, None]
    frac = np.linspace(-1.0, 1.0, a.grid[1] + 1)[None, :]
    y = h * frac
    return np.stack([np.broadcast_to(z_n[:, None], y.shape), y], axis=-1)


def _panel_nodes(z_n, inner, outer, n, upper: bool):
    """Nodes of one panel between two z-dependent surfaces, `n` cells thick."""
    t = np.linspace(0.0, 1.0, n + 1)[None, :]
    r = inner[:, None] + (outer - inner)[:, None] * t
    y = r if upper else -r[:, ::-1]
    return np.stack([np.broadcast_to(z_n[:, None], y.shape), y], axis=-1)


def _panel_nodes_stretched(z_n, inner, outer, n, ratio, upper: bool):
    k = np.arange(n + 1)
    geom = (ratio ** k - 1.0) / (ratio ** n - 1.0)
    r = inner[:, None] + (outer - inner)[:, None] * geom[None, :]
    y = r if upper else -r[:, ::-1]
    return np.stack([np.broadcast_to(z_n[:, None], y.shape), y], axis=-1)


def build_blocks(a: AgentCfg, cfg: AtlasConfig) -> tuple[Block, ...]:
    """Blocks for one agent. Two for `c` and `d` (the y index runs over two
    disjoint sheets); one for everything else."""
    geo: GeometryCfg = cfg.geometry
    n_z, n_y = a.grid
    z_n = np.linspace(a.z[0], a.z[1], n_z + 1)
    half = n_y // 2

    if a.y_map == "uniform":
        nodes = _nodes_uniform(a, z_n, geo)
        blank = np.zeros((n_z, n_y), dtype=bool)
        if a.hole == "plume":
            # the plume is another agent's domain; blank whole cells, so the
            # g-f interface the solver sees is the cell-face-aligned version of
            # the declared curve |y| = plume_halfwidth (0.59375 vs 0.600 here).
            yc = 0.5 * (nodes[:-1, :-1, 1] + nodes[:-1, 1:, 1])
            blank = np.abs(yc) < geo.plume_halfwidth
        return (Block(a.id, nodes, blank),)

    if a.y_map == "body_fitted":
        nodes = _nodes_body_fitted(a, z_n, geo)
        return (Block(a.id, nodes, np.zeros((n_z, n_y), dtype=bool)),)

    inner = nozzle_inner(z_n, geo) if a.y_map == "shell" else shell_outer(z_n, geo)
    if a.y_map == "shell":
        outer = inner + geo.shell_thickness
        mk = lambda up: _panel_nodes(z_n, inner, outer, half, up)      # noqa: E731
    else:
        outer = np.full_like(inner, geo.farfield_halfwidth)
        mk = lambda up: _panel_nodes_stretched(                         # noqa: E731
            z_n, inner, outer, half, geo.atmos_stretch_ratio, up)
    z = np.zeros((n_z, half), dtype=bool)
    return (Block(f"{a.id}_lo", mk(False), z), Block(f"{a.id}_hi", mk(True), z))


def block_map(cfg: AtlasConfig) -> dict[str, tuple[Block, ...]]:
    return {a.id: build_blocks(a, cfg) for a in cfg.agents}


__all__ = ["Block", "build_blocks", "block_map"]
