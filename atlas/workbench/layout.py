"""Windows generated from the geometry: they follow the domain's shape.

The owner's question, 2026-09-29, the day drawn domains and windows were built:
"it doesn't make sense for the windows themselves to be rectangles, does it? Why
doesn't their shape itself ADAPT to the geometry of the curve smoothly?"  Until
then a window was whatever was drawn, and the one generator made a grid of
rectangles: nothing derived a window from the domain.

**The domain's own coordinates.**  A decomposition that follows a curved domain
cuts it across its length.  Its length is found, not declared:

1. The domain's **Fiedler vector** -- the first non-constant eigenvector of the
   graph Laplacian on its cells, no flux through its edges -- runs from one
   extreme of the domain to the other.  The domain's edge (a drawn edge, or one
   of the grid's) where it is lowest is the START, and where it is highest the END.
2. The **along** coordinate is harmonic in the domain: 0 on the start, 1 on the
   end, no flux through every other edge -- `fv.assemble` with a unit
   conductivity, solved once.  Its level curves run from wall to wall and meet
   each wall at a right angle (no flux through it), and they are smooth.  On a
   ring's sector it is the angle itself, so the cuts are exact radii.
3. The **across** coordinate is harmonic too: 0 on the edges on one side between
   the start and the end, 1 on the other side's, no flux through the start and
   the end.  On a ring's sector it is a function of the radius alone, so its
   cuts are exact arcs.

The pieces are the cells between consecutive cuts at equal cell counts:
rectangles in the domain's own coordinates, which bend with it.  **By
material** is the other cut a geometry suggests: one piece per material, so a
Dirichlet-Neumann interface is the material interface itself.

**From pieces to windows.**  For styles A and B a window overlaps its
neighbours: each piece grows into the domain -- a dilation that stays inside
it, so it never jumps a gap -- by the least number of cells for which every cell
has some window at full weight (`geometry.analyse_masks`' rule), starting from
the ramp.  For C, D and the split the pieces meet along faces and do not grow.

**Deterministic, and stored.**  The eigensolver starts from a fixed vector, the
coordinates are rounded before they are ranked, ties go by cell index, and the
windows are stored in the case as cells (``shape = "cells"``), so a record
holds the windows it marched on rather than a recipe for them.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import fv
from . import geometry as geo


class LayoutError(ValueError):
    """The windows cannot follow this geometry, and why."""


# ---------------------------------------------------------------------------
# what the windows are generated from
# ---------------------------------------------------------------------------


def fingerprint(spec) -> str:
    """What generated windows depend on: the domain, the style, the ramp and the
    layout's settings (and the regions, for a cut by material)."""
    lay = spec.layout
    payload = {"domain": spec.domain.model_dump(), "style": spec.coupling.style,
               "ramp": spec.coupling.ramp_cells, "cut": lay.cut, "along": lay.along,
               "across": lay.across}
    if lay.cut == "materials":
        payload["regions"] = [r.model_dump() for r in spec.regions]
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def refresh(spec) -> bool:
    """Generate the windows again when what they follow has changed.  True when
    they were generated."""
    if spec.layout is None:
        return False
    key = fingerprint(spec)
    if key == spec.layout.source and spec.windows:
        return False
    windows, grown = generate(spec)
    spec.windows = windows
    spec.layout.source = key
    spec.layout.grown = grown
    return True


# ---------------------------------------------------------------------------
# the domain's coordinates
# ---------------------------------------------------------------------------


def _laplacian(mask: np.ndarray):
    """(the cells' flat indices, the graph Laplacian over their shared faces)."""
    ny, nx = mask.shape
    idx = np.flatnonzero(mask.ravel())
    pos = np.full(mask.size, -1, dtype=np.int64)
    pos[idx] = np.arange(idx.size)
    jj, ii = np.nonzero(mask[:, :-1] & mask[:, 1:])
    a, b = pos[jj * nx + ii], pos[jj * nx + ii + 1]
    jj, ii = np.nonzero(mask[:-1, :] & mask[1:, :])
    a = np.concatenate([a, pos[jj * nx + ii]])
    b = np.concatenate([b, pos[(jj + 1) * nx + ii]])
    n = idx.size
    adj = sp.csr_matrix((np.ones(2 * a.size), (np.concatenate([a, b]), np.concatenate([b, a]))),
                        shape=(n, n))
    deg = np.asarray(adj.sum(axis=1)).ravel()
    return idx, (sp.diags(deg) - adj).tocsc()


def fiedler(mask: np.ndarray) -> np.ndarray:
    """The Fiedler vector of the cells in ``mask`` on the grid (NaN elsewhere),
    signed to rise with x.  Where the first two non-constant modes are equal (a
    disc, a square), the combination that varies most along x."""
    ny, nx = mask.shape
    idx, lap = _laplacian(mask)
    n = idx.size
    out = np.full(mask.shape, np.nan)
    x = (idx % nx) + 0.5
    y = (idx // nx) + 0.5
    if n < 4:
        out.flat[idx] = x
        return out
    v0 = 1.0 + (x - x.mean()) / nx + 0.37 * (y - y.mean()) / ny      # a fixed start
    k = min(3, n - 1)
    vals, vecs = spla.eigsh(lap, k=k, sigma=-1e-3, which="LM", v0=v0, tol=1e-12)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    f = vecs[:, 1]
    if k >= 3 and abs(vals[2] - vals[1]) <= 1e-6 * max(abs(vals[2]), 1e-300):
        pair = vecs[:, 1:3]
        c = pair.T @ (x - x.mean())
        f = pair @ (c / max(np.linalg.norm(c), 1e-300))
    if float(np.dot(f, x - x.mean()) + 1e-3 * np.dot(f, y - y.mean())) < 0.0:
        f = -f
    out.flat[idx] = f
    return out


def _labels(domain, act):
    """Each boundary face's edge: per grid edge the label of each cell's face on it
    ('' where the cell is not in the domain), and the faces to the void inside the
    grid (`geometry.void_faces`, or None for a domain with no drawn shape)."""
    drawn = domain.outline is not None or bool(domain.holes)
    if domain.outline is not None:
        grid = geo.grid_edge_labels(domain)
    else:
        border = {"left": act[:, 0], "right": act[:, -1], "bottom": act[0, :],
                  "top": act[-1, :]}
        grid = {e: np.where(on, e, "").astype(object) for e, on in border.items()}
    return grid, (geo.void_faces(domain) if drawn else None)


def _ring(domain) -> list[str]:
    """The outer boundary's edges in order round it."""
    if domain.outline is not None:
        return [f"outline:{k}" for k in range(len(domain.outline.points))]
    return ["bottom", "right", "top", "left"]


def _edge_cells(domain, act, grid, vf) -> dict[str, np.ndarray]:
    """name -> the flat indices of the cells with a face on that edge."""
    nx, ny = domain.nx, domain.ny
    out: dict[str, list] = {}
    cells = {"left": np.arange(ny) * nx, "right": np.arange(ny) * nx + nx - 1,
             "bottom": np.arange(nx), "top": (ny - 1) * nx + np.arange(nx)}
    for e, lab in grid.items():
        for c, name in zip(cells[e].tolist(), lab.tolist()):
            if name:
                out.setdefault(name, []).append(c)
    if vf is not None:
        for c, name in zip(vf.cell.tolist(), vf.edge.tolist()):
            out.setdefault(name, []).append(c)
    return {k: np.unique(np.asarray(v, dtype=np.int64)) for k, v in out.items()}


def harmonic(domain, fixed: dict[str, float]) -> np.ndarray:
    """The harmonic function on the domain's cells equal to ``fixed[name]`` on the
    faces of each named edge, with no flux through every other edge (NaN off the
    domain): `fv.assemble` with a unit conductivity."""
    act = geo.domain_mask(domain)
    nx, ny = domain.nx, domain.ny
    grid, vf = _labels(domain, act)
    bc = {}
    for e in fv.EDGES:
        lab = grid[e].tolist()
        bc[e] = (np.array([fv.FIXED if name in fixed else fv.NO_FLUX for name in lab],
                          dtype=np.int8),
                 np.array([fixed.get(name, 0.0) for name in lab], dtype=float))
    if vf is None:
        f = fv.Field(nx, ny, 1.0, np.ones((ny, nx)), bc=bc)
    else:
        names = vf.edge.tolist()
        kinds = np.array([fv.FIXED if n in fixed else fv.NO_FLUX for n in names], dtype=np.int8)
        vals = np.array([fixed.get(n, 0.0) for n in names], dtype=float)
        f = fv.Field(nx, ny, 1.0, np.where(act, 1.0, 0.0), bc=bc, active=act,
                     void=(vf.cell, vf.outside, vf.direction, kinds, vals, vf.edge))
    s = fv.assemble(f)
    out = np.full((ny, nx), np.nan)
    out.flat[s.idx] = spla.spsolve(s.A.tocsc(), s.b)
    return out


def ends(domain) -> tuple[str, str]:
    """(start, end): the outer boundary's edges where the Fiedler vector is lowest
    and highest on average."""
    act = geo.domain_mask(domain)
    fd = fiedler(act)
    grid, vf = _labels(domain, act)
    cells = _edge_cells(domain, act, grid, vf)
    ring = [e for e in _ring(domain) if e in cells]
    if len(ring) < 2:
        raise LayoutError("the domain has fewer than two edges to run between")
    mean = {e: float(np.mean(fd.ravel()[cells[e]])) for e in ring}
    start = min(ring, key=lambda e: (mean[e], ring.index(e)))
    end = max(ring, key=lambda e: (mean[e], -ring.index(e)))
    if start == end:                                                  # pragma: no cover
        raise LayoutError("the domain has no two ends to run between")
    return start, end


def coordinates(domain, across: bool = False):
    """(along, across or None, start, end): the domain's own coordinates."""
    act = geo.domain_mask(domain)
    from scipy.ndimage import label
    _lab, n_parts = label(act)
    if n_parts != 1:
        raise LayoutError(f"the domain is in {n_parts} separate pieces; windows can follow "
                          f"one connected domain (or cut by material)")
    start, end = ends(domain)
    along = harmonic(domain, {start: 0.0, end: 1.0})
    if not across:
        return along, None, start, end
    if domain.holes:
        raise LayoutError("cutting across needs a domain without holes: a hole leaves "
                          "no single pair of sides")
    ring = _ring(domain)
    s, t = ring.index(start), ring.index(end)
    n = len(ring)
    one = [ring[(s + k) % n] for k in range(1, (t - s) % n)]
    two = [ring[(t + k) % n] for k in range(1, (s - t) % n)]
    if not one or not two:
        raise LayoutError(f"cutting across needs an edge on each side between the domain's "
                          f"two ends ({start}, {end}); one side has none")
    fixed = {e: 0.0 for e in one}
    fixed.update({e: 1.0 for e in two})
    return along, harmonic(domain, fixed), start, end


# ---------------------------------------------------------------------------
# pieces, and windows
# ---------------------------------------------------------------------------


def _cut(cells: np.ndarray, values: np.ndarray, parts: int) -> list[np.ndarray]:
    """``parts`` runs of ``cells`` in the order of ``values`` (rounded; ties by
    index), of equal counts."""
    if parts > cells.size:
        raise LayoutError(f"{parts} pieces from {cells.size} cells")
    order = np.lexsort((cells, np.round(values, 12)))
    bounds = np.round(np.linspace(0, cells.size, parts + 1)).astype(int)
    return [cells[order[bounds[k]:bounds[k + 1]]] for k in range(parts)]


def pieces(spec) -> list[tuple[str, np.ndarray]]:
    """The layout's pieces: disjoint cell sets that together are the domain."""
    d = spec.domain
    act = geo.domain_mask(d)
    lay = spec.layout
    style = spec.coupling.style
    shape = act.shape

    def mask_of(cells):
        m = np.zeros(shape, dtype=bool)
        m.flat[cells] = True
        return m
    if style in ("D", "split"):
        return [("whole", act.copy())]
    if lay.cut == "materials":
        owner = geo.region_owner(spec.regions, d.nx, d.ny)
        mats: list[str] = []
        for r in spec.regions:
            if r.material not in mats:
                mats.append(r.material)
        out = []
        for m in mats:
            ks = [k for k, r in enumerate(spec.regions) if r.material == m]
            cells = act & np.isin(owner, ks)
            if cells.any():
                out.append((m, cells))
        if len(out) < 2:
            raise LayoutError("cutting by material needs at least two materials in the "
                              "domain")
    else:
        along, across, _s, _e = coordinates(d, across=lay.across > 1)
        cells = np.flatnonzero(act.ravel())
        cols = _cut(cells, along.ravel()[cells], lay.along)
        wide = lay.along > 10 or lay.across > 10
        out = []
        for c, col in enumerate(cols):
            rows = _cut(col, across.ravel()[col], lay.across) if across is not None else [col]
            for r, piece in enumerate(rows):
                out.append((f"F{c:02d}{r:02d}" if wide else f"F{c}{r}", mask_of(piece)))
    if style == "C" and len(out) != 2:
        raise LayoutError(f"style C couples exactly two pieces, and this layout makes "
                          f"{len(out)}: cut along into 2, or by material with two materials")
    return out


_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_SQUARE = np.ones((3, 3), dtype=bool)


def grow(parts: list[tuple[str, np.ndarray]], act: np.ndarray,
         ramp: int) -> tuple[list[tuple[str, np.ndarray]], int]:
    """Each piece grown inside the domain by the least reach (in cells, from the
    ramp up) for which every cell has some window at full weight.  The growth
    alternates a plus and a square step, so its front is nearly round."""
    from scipy.ndimage import binary_dilation
    reach = max(int(ramp), 1)
    grown = [p.copy() for _n, p in parts]
    for s in range(reach):
        grown = [binary_dilation(g, structure=_CROSS if s % 2 == 0 else _SQUARE) & act
                 for g in grown]
    limit = 4 * reach + 16
    while True:
        wins = list(zip([n for n, _p in parts], grown))
        an = geo.analyse_masks(act, wins, ramp)
        if not an.ramp_only.any() and not an.uncovered.any():
            return wins, reach
        if reach >= limit:
            raise LayoutError(f"no reach up to {limit} cells gives every cell a window at "
                              f"full weight; use fewer windows or a smaller ramp")
        grown = [binary_dilation(g, structure=_CROSS if reach % 2 == 0 else _SQUARE) & act
                 for g in grown]
        reach += 1


def generate(spec):
    """The layout's windows, as `spec.Window`s of cells, and the reach they grew by."""
    from .spec import Window
    act = geo.domain_mask(spec.domain)
    if not act.any():
        raise LayoutError("the domain holds no cell")
    parts = pieces(spec)
    if spec.coupling.style in ("A", "B"):
        wins, reach = grow(parts, act, spec.coupling.ramp_cells)
    else:
        wins, reach = parts, 0
    return [Window(id=name, shape="cells", runs=geo.mask_to_runs(m & act))
            for name, m in wins], reach


__all__ = ["LayoutError", "fingerprint", "refresh", "fiedler", "harmonic", "ends",
           "coordinates", "pieces", "grow", "generate"]
