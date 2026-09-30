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


#: the families whose windows can follow a solved flow, and the boundary kinds the
#: flow runs between (a river from its inlets to its outlets)
FLOWING = {"transport-2d": ("river-inlet", "river-outlet")}

#: a cell whose along-coordinate changes by less than this share of the domain's mean
#: change is one the coordinate does not reach: the end of a branch that neither of
#: the two ends is in (and, for a river's flow, a branch with no water in it)
DEAD = 1e-3


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
        # the cooled block cuts by physics: which materials flow
        payload["materials"] = spec.materials
        payload["family"] = spec.physics.family
    elif spec.physics.family in FLOWING:
        # a branched river's windows follow its flow, which runs from its inlets to
        # its outlets: moving an outlet moves them
        payload["family"] = spec.physics.family
        payload["ends"] = sorted((b.edge, b.kind) for b in spec.boundaries
                                 if b.kind in FLOWING[spec.physics.family])
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def refresh(spec) -> bool:
    """Generate the windows again when what they follow has changed.  True when
    they were generated."""
    if spec.layout is None:
        return False
    key = fingerprint(spec)
    if key == spec.layout.source and spec.windows:
        return False
    windows, grown, how = generate_with_how(spec)
    spec.windows = windows
    spec.layout.source = key
    spec.layout.grown = grown
    spec.layout.how = how
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


def topological_holes(act: np.ndarray) -> int:
    """How many holes the domain's cells enclose: pieces of the void (8-connected, the
    right partner of cells joined by faces) that reach neither the grid's border nor,
    through it, the outline's outside.  A drawn hole that crosses the outline is a
    notch, not a hole (2026-09-30: holes may cross the outline)."""
    from scipy.ndimage import label
    lab, n = label(~np.asarray(act, dtype=bool), structure=np.ones((3, 3), dtype=bool))
    if not n:
        return 0
    border = set(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]).tolist())
    return sum(1 for k in range(1, n + 1) if k not in border)


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
    if topological_holes(act):
        raise LayoutError("cutting across needs a domain without holes: a hole leaves "
                          "no single pair of sides")
    ring = _ring(domain)
    s, t = ring.index(start), ring.index(end)
    n = len(ring)
    # a side's edges that a hole took whole bound nothing, so they fix nothing
    grid, vf = _labels(domain, act)
    live = _edge_cells(domain, act, grid, vf)
    one = [ring[(s + k) % n] for k in range(1, (t - s) % n) if ring[(s + k) % n] in live]
    two = [ring[(t + k) % n] for k in range(1, (s - t) % n) if ring[(t + k) % n] in live]
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


def cell_speed(values: np.ndarray, act: np.ndarray) -> np.ndarray:
    """How fast a field on the domain's cells changes, per cell: the length of its
    gradient from its faces' differences (each face's difference shared by its two
    cells, faces to the void counting nothing), 0 off the domain."""
    act = np.asarray(act, dtype=bool)
    v = np.where(act, np.nan_to_num(np.asarray(values, dtype=float)), 0.0)
    dxf = np.where(act[:, :-1] & act[:, 1:], v[:, 1:] - v[:, :-1], 0.0)
    dyf = np.where(act[:-1, :] & act[1:, :], v[1:, :] - v[:-1, :], 0.0)
    gx, gy = np.zeros(act.shape), np.zeros(act.shape)
    gx[:, :-1] += 0.5 * dxf
    gx[:, 1:] += 0.5 * dxf
    gy[:-1, :] += 0.5 * dyf
    gy[1:, :] += 0.5 * dyf
    return np.where(act, np.hypot(gx, gy), 0.0)


def dead_regions(speed: np.ndarray, act: np.ndarray, level: float = DEAD,
                 min_cells: int = 16) -> list[np.ndarray]:
    """The connected pieces of the domain where ``speed`` is under ``level`` of its
    mean over the domain, each of at least ``min_cells`` cells: where a harmonic
    coordinate does not reach.

    A branch that neither end is in is one: into it the coordinate decays like
    ``exp(-pi x / w)`` for a width ``w``, so past about ``2.2 w`` it changes by less
    than a thousandth of its mean.  A convex corner is not: the change there falls
    only linearly, and a cell or two at most is under the level (a rectangle, a ring,
    an L and every drawn example have none, and a test says so)."""
    from scipy.ndimage import label
    act = np.asarray(act, dtype=bool)
    if not act.any():
        return []
    mean = float(np.mean(speed[act]))
    if mean <= 0.0:
        return []
    lab, n = label(act & (speed < level * mean))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    return [lab == k for k in range(1, n + 1) if sizes[k] >= min_cells]


def geodesic(act: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Each cell's distance, in faces crossed, from the nearest of the ``sources``
    (a mask), through the domain's cells only (inf off the domain or unreached)."""
    from scipy.sparse.csgraph import dijkstra
    idx, lap = _laplacian(act)
    adj = sp.diags(lap.diagonal()) - lap
    src = np.flatnonzero(np.asarray(sources, dtype=bool).ravel()[idx])
    out = np.full(act.shape, np.inf)
    if src.size:
        dist = dijkstra(sp.csr_matrix(adj), directed=False, indices=src, unweighted=True,
                        min_only=True)
        out.flat[idx] = dist
    return out


def bisect(act: np.ndarray, parts: int) -> list[np.ndarray]:
    """``parts`` pieces of the domain by recursive spectral bisection (Pothen, Simon &
    Liou, 1990): the largest piece is split at the median of its own Fiedler vector
    until there are enough.  It needs no notion of two ends, so it serves a shape that
    branches.  Deterministic: the eigensolver's fixed start, ties by cell index; the
    half nearer the grid's left stays first."""
    out = [np.asarray(act, dtype=bool).copy()]
    while len(out) < parts:
        k = max(range(len(out)), key=lambda i: (int(out[i].sum()), -i))
        m = out[k]
        cells = np.flatnonzero(m.ravel())
        if cells.size < 2:
            raise LayoutError(f"{parts} pieces from {int(act.sum())} cells")
        f = fiedler(m).ravel()[cells]
        halves = []
        for part in _cut(cells, f, 2):
            h = np.zeros(act.shape, dtype=bool)
            h.flat[part] = True
            halves.append(h)
        out[k:k + 1] = halves
    return out


def _flow_along(spec, act: np.ndarray) -> np.ndarray | None:
    """``1 - phi`` for a family that carries a flow, ``phi`` its solved potential (1 on
    its inlets, 0 on its outlets): its level curves cross every branch from bank to
    bank, and it falls along every streamline.  None when the family carries no flow,
    the flow cannot be solved yet, or it leaves a branch with no water in it."""
    if spec.physics.family not in FLOWING:
        return None
    from .families.plume import river_flow
    from .flow import FlowError
    try:
        fl = river_flow(spec)
    except (FlowError, ValueError, KeyError, IndexError):
        return None
    phi = np.where(act, fl.phi, np.nan)
    if not np.all(np.isfinite(phi[act])) or dead_regions(cell_speed(phi, act), act):
        return None
    return 1.0 - phi


def pieces(spec) -> list[tuple[str, np.ndarray]]:
    """The layout's pieces: disjoint cell sets that together are the domain."""
    return pieces_and_how(spec)[0]


def pieces_and_how(spec) -> tuple[list[tuple[str, np.ndarray]], str]:
    """The layout's pieces, and how they were cut, in words for the page.

    Cut along: between the domain's two ends, as since case file 0.5, unless the
    coordinate between them leaves a dead region (`dead_regions`): then the shape
    branches.  **A branched river is cut along its own flow** (`_flow_along`), each
    band between two cuts a window per connected piece of it, so a band across both
    branches of a fork is two windows.  **Any other branched shape is cut by
    recursive spectral bisection** (`bisect`).  The owner's forked river, 2026-09-30:
    a Y's two ends are its branch tips, and the old cut split its stem lengthwise."""
    d = spec.domain
    act = geo.domain_mask(d)
    lay = spec.layout
    style = spec.coupling.style
    shape = act.shape
    how = ""

    def mask_of(cells):
        m = np.zeros(shape, dtype=bool)
        m.flat[cells] = True
        return m
    if style in ("D", "split"):
        return [("whole", act.copy())], "one piece: the whole domain"
    if lay.cut == "materials" and spec.physics.family == "conjugate-heat-2d":
        # the cooled block's pieces are its two physics, whatever the materials: the
        # solid (copper and a chip, say) and the coolant that flows beside it
        from .families.cooling import coolant_mask
        wet = coolant_mask(spec)
        out = [(name, m) for name, m in (("block", act & ~wet), ("channel", wet)) if m.any()]
        if len(out) < 2:
            raise LayoutError("cutting by physics needs a coolant and a solid in the domain")
        how = "one piece per physics: the block, and the channel the coolant flows in"
    elif lay.cut == "materials":
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
        how = "one piece per material: the interfaces are the materials' own"
    else:
        along, across, s_end, e_end = coordinates(d, across=lay.across > 1)
        cells = np.flatnonzero(act.ravel())
        wide = lay.along > 10 or lay.across > 10
        out = []
        dead = (dead_regions(cell_speed(along, act), act) if across is None and lay.along > 1
                else [])
        flow = _flow_along(spec, act) if dead else None
        if dead and flow is not None:
            # a branched river: bands along its own flow, each connected piece a window
            from scipy.ndimage import label
            for c, col in enumerate(_cut(cells, flow.ravel()[cells], lay.along)):
                lab, n = label(mask_of(col))
                for r in range(n):
                    out.append((f"F{c:02d}{r:02d}" if wide else f"F{c}{r}", lab == r + 1))
            how = ("along the river's own flow, from its inlet to its outlets: the shape "
                   "branches, so a band across two branches is two windows")
        elif dead:
            out = [(f"F{c:02d}00" if wide else f"F{c}0", m)
                   for c, m in enumerate(bisect(act, lay.along))]
            how = ("by recursive spectral bisection: the shape branches, and the length "
                   f"between its two ends ({s_end}, {e_end}) does not reach every branch")
        else:
            for c, col in enumerate(_cut(cells, along.ravel()[cells], lay.along)):
                rows = (_cut(col, across.ravel()[col], lay.across) if across is not None
                        else [col])
                for r, piece in enumerate(rows):
                    out.append((f"F{c:02d}{r:02d}" if wide else f"F{c}{r}", mask_of(piece)))
            how = f"along the shape's length, between its two ends ({s_end}, {e_end})"
    if style == "C" and len(out) != 2:
        raise LayoutError(f"style C couples exactly two pieces, and this layout makes "
                          f"{len(out)}: cut along into 2, or by material with two materials")
    return out, how


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
    windows, reach, _how = generate_with_how(spec)
    return windows, reach


def generate_with_how(spec):
    """`generate`, and how the pieces were cut (`pieces_and_how`)."""
    from .spec import Window
    act = geo.domain_mask(spec.domain)
    if not act.any():
        raise LayoutError("the domain holds no cell")
    parts, how = pieces_and_how(spec)
    if spec.coupling.style in ("A", "B"):
        wins, reach = grow(parts, act, spec.coupling.ramp_cells)
    else:
        wins, reach = parts, 0
    return ([Window(id=name, shape="cells", runs=geo.mask_to_runs(m & act))
             for name, m in wins], reach, how)


__all__ = ["LayoutError", "FLOWING", "DEAD", "fingerprint", "refresh", "fiedler", "harmonic",
           "ends", "topological_holes", "coordinates", "cell_speed", "dead_regions",
           "geodesic", "bisect", "pieces", "pieces_and_how", "grow", "generate",
           "generate_with_how"]
