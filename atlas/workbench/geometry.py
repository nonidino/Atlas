"""The geometry rules of a case, as plain functions of cell grids.

Everything the geometry section draws as a warning, and everything `spec.check`
refuses, is computed here, so a test can check a rule without a browser and the
canvas cannot say something the check does not.

Units are cells throughout.  A box is ``(x0, y0, nx, ny)``: the cells
``x0 .. x0+nx-1`` by ``y0 .. y0+ny-1``.

**The one rule that is Atlas's own, not generic geometry: every covered cell must
have some window at full weight.**  A window's partition-of-unity weight ramps
from 0 at each of its artificial faces (a face not on the domain boundary) to 1
over ``ramp`` cells, exactly as `wake_array.ArrayTiling.weights` builds it.  On
every measured tiling (a 16-cell overlap against an 8-cell ramp) each cell has
one window at weight 1.  Where that fails, the blend there is made only of
ramps: windows that touch without overlapping, or overlap by less than twice the
ramp, or a window thinner than two ramps between two artificial faces.  No
measurement covers such a blend, so the check refuses it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

from . import shapes

Box = tuple[int, int, int, int]

#: The snapping steps offered in the editor, in cells.  8 is the default (the
#: owner's decision 1 was left open; the plan's recommendation is taken).
SNAP_STEPS = (1, 8, 16)
DEFAULT_SNAP = 8


# ---------------------------------------------------------------------------
# snapping and boxes
# ---------------------------------------------------------------------------


def snap_value(v: float, step: int, limit: int) -> int:
    """The nearest multiple of ``step`` in ``[0, limit]``, with ``limit`` itself
    also a target (a domain need not be a multiple of the step)."""
    step = max(int(step), 1)
    cands = [min(max(int(round(v / step)) * step, 0), limit), limit, 0]
    return int(min(cands, key=lambda c: (abs(c - v), c)))


def snap_box(x0: float, y0: float, x1: float, y1: float, step: int,
             nx: int, ny: int) -> Box | None:
    """A drawn or moved box, snapped to the grid and kept inside the domain.

    The size is snapped first and then kept while the box is pushed back inside,
    so dragging a window against the edge stops it rather than shrinking it.
    Returns None when the box collapses below one cell.
    """
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    step = max(int(step), 1)
    w = max(step, int(round((x1 - x0) / step)) * step)
    h = max(step, int(round((y1 - y0) / step)) * step)
    w, h = min(w, nx), min(h, ny)
    if w < 1 or h < 1:
        return None
    bx = snap_value(x0, step, nx)
    by = snap_value(y0, step, ny)
    bx = min(max(bx, 0), nx - w)
    by = min(max(by, 0), ny - h)
    return int(bx), int(by), int(w), int(h)


def corners(box: Box) -> list[tuple[float, float]]:
    """The four corners, in the order the resize handles use: SW, SE, NE, NW."""
    x0, y0, w, h = box
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


def resize_from_corner(box: Box, corner: int, x: float, y: float, step: int,
                       nx: int, ny: int) -> Box | None:
    """The box after its ``corner`` (0..3, SW SE NE NW) is dragged to ``(x, y)``;
    the opposite corner stays where it was."""
    cs = corners(box)
    ox, oy = cs[(corner + 2) % 4]
    sx = snap_value(x, step, nx)
    sy = snap_value(y, step, ny)
    x0, x1 = sorted((ox, sx))
    y0, y1 = sorted((oy, sy))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def intersect(a: Box, b: Box) -> Box | None:
    x0, x1 = max(a[0], b[0]), min(a[0] + a[2], b[0] + b[2])
    y0, y1 = max(a[1], b[1]), min(a[1] + a[3], b[1] + b[3])
    if x1 > x0 and y1 > y0:
        return x0, y0, x1 - x0, y1 - y0
    return None


def clip_box(b: Box, nx: int, ny: int) -> Box | None:
    return intersect(b, (0, 0, nx, ny))


# ---------------------------------------------------------------------------
# tilings
# ---------------------------------------------------------------------------


def tile(nx: int, ny: int, cols: int, rows: int, overlap: int) -> list[tuple[str, Box]]:
    """A regular tiling that covers the domain exactly.

    Every window is the same size, the smallest that gives at least ``overlap``
    cells between neighbours, and the windows are spread so that the first starts
    at 0 and the last ends at the domain edge.  Names follow the tiling's own
    convention (`wake_array.ArrayTiling`), ``F<col><row>``, with two digits each past
    nine columns or rows.
    """
    def one_axis(n: int, k: int) -> tuple[int, list[int]]:
        if k < 1:
            raise ValueError("at least one window per axis")
        if k == 1:
            return n, [0]
        size = -(-(n + (k - 1) * overlap) // k)            # ceil
        if size > n:
            raise ValueError(f"{k} windows overlapping by {overlap} cells do not fit in "
                             f"{n} cells")
        stride = (n - size) / (k - 1)
        if stride < 1:
            raise ValueError(f"{k} windows overlapping by {overlap} cells in {n} cells would "
                             f"lie on top of each other")
        return size, [int(round(i * stride)) for i in range(k)]

    wx, xs = one_axis(nx, cols)
    wy, ys = one_axis(ny, rows)
    wide = cols > 10 or rows > 10
    out = []
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            name = f"F{c:02d}{r:02d}" if wide else f"F{c}{r}"
            out.append((name, (x0, y0, wx, wy)))
    return out


# ---------------------------------------------------------------------------
# the window analysis
# ---------------------------------------------------------------------------


@dataclass
class WindowAnalysis:
    count: np.ndarray                 # (ny, nx) windows covering each cell
    full: np.ndarray                  # (ny, nx) some window at weight 1
    uncovered: np.ndarray             # (ny, nx) bool
    ramp_only: np.ndarray             # (ny, nx) covered, but no window at weight 1
    ramp_only_in: dict[str, int]      # window -> ramp-only cells inside it
    thin_pairs: list[tuple[str, str, int]] = field(default_factory=list)
    nested: list[tuple[str, str]] = field(default_factory=list)       # (inner, outer)
    cross_points: list[tuple[tuple[str, ...], Box]] = field(default_factory=list)
    overlaps: list[tuple[str, str, Box]] = field(default_factory=list)
    #: cells in two or more windows (`analyse_masks` only; rectangles draw boxes)
    overlap_mask: np.ndarray | None = None


def _axis_full(n: int, lo_face: bool, hi_face: bool, ramp: int) -> np.ndarray:
    """Cells along one axis of a window at weight 1: the runner's own ramp
    (`tiling.axis_weight`, which is `ArrayTiling.weights`'), so the rule the
    canvas draws and the weights a run blends with cannot come apart."""
    from .tiling import axis_weight
    return axis_weight(n, lo_face, hi_face, ramp) >= 1.0


def analyse_windows(nx: int, ny: int, windows: Sequence[tuple[str, Box]],
                    ramp: int) -> WindowAnalysis:
    """Coverage, full-weight cells, thin seams, nesting and cross-points.

    Windows reaching past the domain are clipped here (the structural check
    reports them); a face on the domain boundary is not artificial.
    """
    count = np.zeros((ny, nx), dtype=np.int32)
    full = np.zeros((ny, nx), dtype=bool)
    clipped: list[tuple[str, Box]] = []
    for wid, b in windows:
        c = clip_box(b, nx, ny)
        if c is None:
            continue
        x0, y0, w, h = c
        clipped.append((wid, c))
        fx = _axis_full(w, x0 > 0, x0 + w < nx, ramp)
        fy = _axis_full(h, y0 > 0, y0 + h < ny, ramp)
        count[y0:y0 + h, x0:x0 + w] += 1
        full[y0:y0 + h, x0:x0 + w] |= fy[:, None] & fx[None, :]
    uncovered = count == 0
    ramp_only = (count > 0) & ~full
    ramp_only_in = {}
    for wid, (x0, y0, w, h) in clipped:
        k = int(ramp_only[y0:y0 + h, x0:x0 + w].sum())
        if k:
            ramp_only_in[wid] = k

    thin, nested, overlaps = [], [], []
    need = 2 * max(int(ramp), 1)
    for i, (ia, a) in enumerate(clipped):
        for ib, b in clipped[i + 1:]:
            ov = intersect(a, b)
            if ov is not None:
                overlaps.append((ia, ib, ov))
            if ov == a:
                nested.append((ia, ib))
                continue
            if ov == b:
                nested.append((ib, ia))
                continue
            # the seam's thickness: across the shared strip, 0 when they only touch
            ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
            oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
            if ox >= 0 and oy >= 0 and max(ox, oy) > 0:
                thick = min(ox, oy)
                # a corner contact of two diagonal neighbours is not a seam
                if thick < need and not (ox <= need and oy <= need and min(ox, oy) == 0):
                    # ...and a thin seam that other windows cover at full weight is
                    # harmless: name it only where the rule actually fails near it
                    r = max(int(ramp), 1)
                    zx0 = max(max(a[0], b[0]) - r, 0)
                    zx1 = min(min(a[0] + a[2], b[0] + b[2]) + r, nx)
                    zy0 = max(max(a[1], b[1]) - r, 0)
                    zy1 = min(min(a[1] + a[3], b[1] + b[3]) + r, ny)
                    if ramp_only[zy0:zy1, zx0:zx1].any():
                        thin.append((ia, ib, thick))

    cps: list[tuple[tuple[str, ...], Box]] = []
    ys, xs = np.nonzero(count >= 3)
    if len(ys):
        member = np.zeros((len(ys), len(clipped)), dtype=bool)
        for k, (_wid, (x0, y0, w, h)) in enumerate(clipped):
            member[:, k] = (xs >= x0) & (xs < x0 + w) & (ys >= y0) & (ys < y0 + h)
        keys, inv = np.unique(member, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        for g in range(len(keys)):
            sel = inv == g
            gx, gy = xs[sel], ys[sel]
            names = tuple(clipped[k][0] for k in np.nonzero(keys[g])[0])
            cps.append((names, (int(gx.min()), int(gy.min()),
                                int(gx.max() - gx.min() + 1), int(gy.max() - gy.min() + 1))))
        cps.sort(key=lambda t: (t[1][1], t[1][0]))
    return WindowAnalysis(count, full, uncovered, ramp_only, ramp_only_in, thin, nested,
                          cps, overlaps)


def mask_to_boxes(mask: np.ndarray, limit: int = 4000) -> list[Box]:
    """Cover a boolean cell mask with rectangles, to draw it cheaply.

    Runs along each row, merged with identical runs in the rows above.  Exact:
    the boxes cover the mask and nothing else.  Stops adding at ``limit`` boxes
    (a pathological mask is still drawn, just not all of it).
    """
    ny, _nx = mask.shape
    out: list[Box] = []
    open_runs: dict[tuple[int, int], int] = {}        # (x0, x1) -> start row
    for y in range(ny + 1):
        runs: set[tuple[int, int]] = set()
        if y < ny:
            row = mask[y].astype(np.int8)
            if row.any():
                d = np.diff(np.concatenate(([0], row, [0])))
                starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
                runs = set(zip(starts.tolist(), ends.tolist()))
        for key in list(open_runs):
            if key not in runs:
                y0 = open_runs.pop(key)
                out.append((key[0], y0, key[1] - key[0], y - y0))
                if len(out) >= limit:
                    return out
        for key in runs:
            open_runs.setdefault(key, y)
    return out


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------


Ring = tuple[tuple[float, float], ...]


@lru_cache(maxsize=64)
def _polygon_mask(rings: tuple[Ring, ...], nx: int, ny: int) -> np.ndarray:
    """Cells whose centres lie inside a set of rings by the even-odd rule, so a
    ring inside the outer one is a hole.  Cached, and returned read-only."""
    out = np.zeros((ny, nx), dtype=bool)
    rings = tuple(r for r in rings if len(r) >= 3)
    if not rings:
        out.flags.writeable = False
        return out
    allp = np.concatenate([np.asarray(r, dtype=float) for r in rings])
    x0 = max(int(np.floor(allp[:, 0].min())), 0)
    x1 = min(int(np.ceil(allp[:, 0].max())), nx)
    y0 = max(int(np.floor(allp[:, 1].min())), 0)
    y1 = min(int(np.ceil(allp[:, 1].max())), ny)
    if x1 <= x0 or y1 <= y0:
        out.flags.writeable = False
        return out
    X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    inside = np.zeros(X.shape, dtype=bool)
    for ring in rings:
        pts = np.asarray(ring, dtype=float)
        xa, ya = pts[:, 0], pts[:, 1]
        xb, yb = np.roll(xa, -1), np.roll(ya, -1)
        for i in range(len(pts)):
            if ya[i] == yb[i]:
                continue
            crosses = (ya[i] > Y) != (yb[i] > Y)
            xc = xa[i] + (Y - ya[i]) * (xb[i] - xa[i]) / (yb[i] - ya[i])
            inside ^= crosses & (X < xc)
    out[y0:y1, x0:x1] = inside
    out.flags.writeable = False            # cached: shared between callers
    return out


def _ring(points) -> Ring:
    return tuple((float(p[0]), float(p[1])) for p in points)


def region_mask(region, nx: int, ny: int) -> np.ndarray:
    """The cells a region is drawn over: its rectangle, the cell centres inside its
    polygon and outside its holes, or inside its drawn shape.  Read-only."""
    if getattr(region, "shape", "rect") == "curve":
        return shape_mask(region.outline, (), nx, ny)
    if getattr(region, "points", None):
        rings = (_ring(region.points),) + tuple(_ring(h) for h in (region.holes or []))
        return _polygon_mask(rings, nx, ny)
    m = np.zeros((ny, nx), dtype=bool)
    c = clip_box((region.x0, region.y0, region.nx, region.ny), nx, ny)
    if c is not None:
        x0, y0, w, h = c
        m[y0:y0 + h, x0:x0 + w] = True
    m.flags.writeable = False
    return m


def region_owner(regions: Sequence, nx: int, ny: int) -> np.ndarray:
    """(ny, nx) index of the region each cell belongs to, -1 for none.  Regions
    stack in list order: a later region takes the cells it covers."""
    owner = np.full((ny, nx), -1, dtype=np.int32)
    for k, r in enumerate(regions):
        owner[region_mask(r, nx, ny)] = k
    return owner


def polygon_area(points: Iterable[Sequence[float]]) -> float:
    p = np.asarray(list(points), dtype=float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ---------------------------------------------------------------------------
# boundaries
# ---------------------------------------------------------------------------

EDGES = ("left", "right", "bottom", "top")

#: a drawn edge's name: ``outline:<k>`` (edge k of the domain's outline) or
#: ``hole<h>:<k>`` (edge k of hole h)
DRAWN_EDGE = re.compile(r"^(outline|hole(\d+)):(\d+)$")


def edge_length(edge: str, nx: int, ny: int) -> int:
    return ny if edge in ("left", "right") else nx


def edge_segment_xy(edge: str, start: float, stop: float, nx: int,
                    ny: int) -> tuple[float, float, float, float]:
    """A boundary segment's end points in cells, for drawing."""
    if edge == "left":
        return 0.0, start, 0.0, stop
    if edge == "right":
        return float(nx), start, float(nx), stop
    if edge == "bottom":
        return start, 0.0, stop, 0.0
    return start, float(ny), stop, float(ny)


# ---------------------------------------------------------------------------
# drawn shapes (case file 0.4): the domain's outline and holes, curved windows
# and regions -- all resolved to the cells whose centres they contain
# ---------------------------------------------------------------------------


def outline_ring(o, step: float = shapes.STEP) -> np.ndarray:
    """A drawn shape (anything with ``points``, ``edges``, ``bulge``) as a polygon."""
    return shapes.sample(o.points, o.edges, o.bulge, step)


def shape_mask(outline, holes, nx: int, ny: int) -> np.ndarray:
    """Cells whose centres lie inside ``outline`` and outside every one of ``holes``.

    Each ring is rasterized on its own and the holes subtracted, rather than all
    rings together by the even-odd rule, so a hole that pokes out of the outline
    removes cells instead of adding them.  Read-only."""
    out = np.array(_polygon_mask((_ring(outline_ring(outline)),), nx, ny))
    for h in holes or ():
        out &= ~_polygon_mask((_ring(outline_ring(h)),), nx, ny)
    out.flags.writeable = False
    return out


def domain_mask(domain) -> np.ndarray:
    """The domain's cells: the whole grid, or inside its drawn outline, less its
    holes.  Read-only."""
    nx, ny = domain.nx, domain.ny
    outline = getattr(domain, "outline", None)
    holes = getattr(domain, "holes", None) or []
    if outline is None and not holes:
        out = np.ones((ny, nx), dtype=bool)
        out.flags.writeable = False
        return out
    if outline is not None:
        return shape_mask(outline, holes, nx, ny)
    out = np.ones((ny, nx), dtype=bool)
    for h in holes:
        out &= ~_polygon_mask((_ring(outline_ring(h)),), nx, ny)
    out.flags.writeable = False
    return out


def mask_to_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """A cell set as runs ``(row, first, stop)`` along its rows, row by row."""
    out: list[tuple[int, int, int]] = []
    for j in range(mask.shape[0]):
        row = mask[j].astype(np.int8)
        if not row.any():
            continue
        d = np.diff(np.concatenate(([0], row, [0])))
        for a, b in zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()):
            out.append((j, int(a), int(b)))
    return out


def runs_to_mask(runs, nx: int, ny: int) -> np.ndarray:
    """`mask_to_runs`' inverse, clipped to the grid."""
    m = np.zeros((ny, nx), dtype=bool)
    for j, a, b in runs or ():
        if 0 <= j < ny:
            m[j, max(a, 0):min(b, nx)] = True
    return m


def window_mask(w, nx: int, ny: int) -> np.ndarray:
    """The cells a window is drawn over (before the domain's own mask).  Read-only."""
    if getattr(w, "shape", "rect") == "curve":
        return shape_mask(w.outline, w.holes, nx, ny)
    if getattr(w, "shape", "rect") == "cells":
        m = runs_to_mask(w.runs, nx, ny)
        m.flags.writeable = False
        return m
    m = np.zeros((ny, nx), dtype=bool)
    c = clip_box((w.x0, w.y0, w.nx, w.ny), nx, ny)
    if c is not None:
        x0, y0, ww, hh = c
        m[y0:y0 + hh, x0:x0 + ww] = True
    m.flags.writeable = False
    return m


def is_plain(spec) -> bool:
    """A rectangle of a domain with no holes, cut into rectangles: the geometry the
    workbench had before 0.4, which keeps its own arithmetic (`analyse_windows`,
    `tiling.RectangleTiling`) so every earlier case runs to the bit as it did."""
    d = spec.domain
    return (getattr(d, "outline", None) is None and not getattr(d, "holes", None)
            and all(getattr(w, "shape", "rect") == "rect" for w in spec.windows))


def window_masks(spec) -> list[tuple[str, np.ndarray]]:
    """Each window's cells inside the domain, in the case's order."""
    d = spec.domain
    act = domain_mask(d)
    return [(w.id, window_mask(w, d.nx, d.ny) & act) for w in spec.windows]


def _neighbour(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """``out[j, i] = mask[j + dy, i + dx]``, False past the grid."""
    out = np.zeros_like(mask)
    ny, nx = mask.shape
    out[max(0, -dy):ny - max(0, dy), max(0, -dx):nx - max(0, dx)] = \
        mask[max(0, dy):ny - max(0, -dy), max(0, dx):nx - max(0, -dx)]
    return out


def touching(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cells of ``a`` that share a face with a cell of ``b``."""
    near = (_neighbour(b, 1, 0) | _neighbour(b, -1, 0) | _neighbour(b, 0, 1)
            | _neighbour(b, 0, -1))
    return a & near


def faces_between(a: np.ndarray, b: np.ndarray) -> int:
    """How many faces a cell of ``a`` shares with a cell of ``b``."""
    return int(sum(int((a & _neighbour(b, dx, dy)).sum())
                   for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))))


def ramp_weight(mask: np.ndarray, active: np.ndarray, ramp: int) -> np.ndarray:
    """A drawn window's raw partition-of-unity weight on the whole grid.

    Zero outside the window; inside it, ``(clip((d - 1/2) / ramp, 0, 1))**2`` where
    ``d`` is the distance from the cell's centre to the centre of the nearest
    cell of the domain OUTSIDE the window.  So the weight rises from its
    artificial faces -- the faces to cells of the domain it does not hold -- over
    ``ramp`` cells, exactly as a rectangle's does across one face (``d - 1/2`` is
    the distance to the face).  A face on the domain's own boundary is not
    artificial and has no ramp.  The rectangles' own rule, ``min(wy, wx)**2``,
    is kept for rectangles (`tiling.RectangleTiling`); this one differs from it
    only near a window's corners, where the distance is Euclidean.
    """
    m = mask & active
    ext = active & ~mask
    if not ext.any():
        return m.astype(float)
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(~ext)
    w = np.clip((d - 0.5) / max(int(ramp), 1), 0.0, 1.0) ** 2
    return np.where(m, w, 0.0)


def _bbox(mask: np.ndarray) -> Box:
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1))


def analyse_masks(active: np.ndarray, windows: Sequence[tuple[str, np.ndarray]],
                  ramp: int) -> WindowAnalysis:
    """`analyse_windows` for windows of any shape on a domain of any shape.

    The same rule -- every cell of the domain has some window at full weight --
    with the weights of `ramp_weight`.  Overlaps and cross-points are reported
    with their bounding boxes, as before, and ``overlap_mask`` holds the cells
    in two or more windows, which is what the canvas draws for a curved case.
    """
    active = np.asarray(active, dtype=bool)
    masks = [(wid, np.asarray(m, dtype=bool) & active) for wid, m in windows]
    count = np.zeros(active.shape, dtype=np.int32)
    full = np.zeros(active.shape, dtype=bool)
    for _wid, m in masks:
        count += m
        full |= ramp_weight(m, active, ramp) >= 1.0
    uncovered = active & (count == 0)
    ramp_only = active & (count > 0) & ~full
    ramp_only_in = {}
    for wid, m in masks:
        k = int((ramp_only & m).sum())
        if k:
            ramp_only_in[wid] = k
    thin, nested, overlaps = [], [], []
    need = 2 * max(int(ramp), 1)
    r = max(int(ramp), 1)
    from scipy.ndimage import binary_dilation, distance_transform_edt
    for i, (ia, a) in enumerate(masks):
        for ib, b in masks[i + 1:]:
            ov = a & b
            if ov.any():
                overlaps.append((ia, ib, _bbox(ov)))
            if a.any() and not (a & ~b).any():
                nested.append((ia, ib))
                continue
            if b.any() and not (b & ~a).any():
                nested.append((ib, ia))
                continue
            contact = ov | touching(a, b)
            if not contact.any():
                continue
            zone = binary_dilation(contact, iterations=r)
            if (zone & ramp_only).any():
                thick = int(2 * distance_transform_edt(ov).max()) if ov.any() else 0
                if thick < need:
                    thin.append((ia, ib, thick))
    cps: list[tuple[tuple[str, ...], Box]] = []
    ys, xs = np.nonzero(count >= 3)
    if len(ys):
        member = np.column_stack([m[ys, xs] for _wid, m in masks])
        keys, inv = np.unique(member, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        for g in range(len(keys)):
            sel = inv == g
            gx, gy = xs[sel], ys[sel]
            names = tuple(masks[k][0] for k in np.nonzero(keys[g])[0])
            cps.append((names, (int(gx.min()), int(gy.min()),
                                int(gx.max() - gx.min() + 1), int(gy.max() - gy.min() + 1))))
        cps.sort(key=lambda t: (t[1][1], t[1][0]))
    return WindowAnalysis(count, full, uncovered, ramp_only, ramp_only_in, thin, nested, cps,
                          overlaps, overlap_mask=count >= 2)


def analyse_case(spec) -> WindowAnalysis:
    """The window analysis for a case, on the arithmetic its geometry calls for."""
    d = spec.domain
    ramp = spec.coupling.ramp_cells
    if is_plain(spec):
        return analyse_windows(d.nx, d.ny, [(w.id, (w.x0, w.y0, w.nx, w.ny))
                                            for w in spec.windows], ramp)
    return analyse_masks(domain_mask(d), window_masks(spec), ramp)


# -- the boundary of a drawn domain ----------------------------------------------

#: the four face directions: (di, dj) from a cell to its neighbour
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def drawn_edges(domain, step: float = 1.0) -> list[tuple[str, np.ndarray]]:
    """Every drawn edge of the domain, named, as a polyline with both end points."""
    out: list[tuple[str, np.ndarray]] = []
    o = getattr(domain, "outline", None)
    if o is not None:
        for k, poly in enumerate(shapes.edges_sampled(o.points, o.edges, o.bulge, step)):
            out.append((f"outline:{k}", poly))
    for h, hole in enumerate(getattr(domain, "holes", None) or []):
        for k, poly in enumerate(shapes.edges_sampled(hole.points, hole.edges, hole.bulge,
                                                      step)):
            out.append((f"hole{h}:{k}", poly))
    return out


def nearest_edge(pts: np.ndarray, edges: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """For each point, the name of the drawn edge nearest it."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if not len(pts) or not edges:
        return np.array([""] * len(pts), dtype=object)
    names, a_all, b_all = [], [], []
    for name, poly in edges:
        names += [name] * (len(poly) - 1)
        a_all.append(poly[:-1])
        b_all.append(poly[1:])
    a = np.concatenate(a_all)
    ab = np.concatenate(b_all) - a
    L2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    names = np.array(names, dtype=object)
    best = np.empty(len(pts), dtype=object)
    for s in range(0, len(pts), 512):
        p = pts[s:s + 512]
        ap = p[:, None, :] - a[None, :, :]
        t = np.clip(np.einsum("pij,ij->pi", ap, ab) / L2[None, :], 0.0, 1.0)
        near = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        dist = np.hypot(p[:, None, 0] - near[:, :, 0], p[:, None, 1] - near[:, :, 1])
        best[s:s + 512] = names[np.argmin(dist, axis=1)]
    return best


@dataclass
class VoidFaces:
    """The faces between a domain cell and a cell of the grid outside the domain.

    ``cell`` and ``outside`` are flat indices ``j nx + i``, ``direction`` indexes
    `DIRECTIONS` (from the cell to the outside one), and ``edge`` names the drawn
    edge each face belongs to (the nearest one)."""

    cell: np.ndarray
    outside: np.ndarray
    direction: np.ndarray
    edge: np.ndarray

    @property
    def n(self) -> int:
        return int(self.cell.size)


def void_faces(domain) -> VoidFaces:
    """Every face between the domain and the void inside the grid, labelled."""
    act = domain_mask(domain)
    ny, nx = act.shape
    cells, outs, dirs = [], [], []
    for k, (di, dj) in enumerate(DIRECTIONS):
        nb_in_grid = np.zeros_like(act)
        nb_in_grid[max(0, -dj):ny - max(0, dj), max(0, -di):nx - max(0, di)] = True
        sel = act & nb_in_grid & ~_neighbour(act, di, dj)
        jj, ii = np.nonzero(sel)
        cells.append(jj * nx + ii)
        outs.append((jj + dj) * nx + (ii + di))
        dirs.append(np.full(jj.size, k, dtype=np.int8))
    cell = np.concatenate(cells).astype(np.int64)
    out = np.concatenate(outs).astype(np.int64)
    direc = np.concatenate(dirs)
    order = np.lexsort((direc, cell))
    cell, out, direc = cell[order], out[order], direc[order]
    ci, cj = cell % nx, cell // nx
    d = np.asarray(DIRECTIONS)[direc]
    mid = np.column_stack([ci + 0.5 + 0.5 * d[:, 0], cj + 0.5 + 0.5 * d[:, 1]])
    return VoidFaces(cell, out, direc, nearest_edge(mid, drawn_edges(domain)))


def grid_edge_labels(domain) -> dict[str, np.ndarray]:
    """For a domain with a drawn outline, the drawn edge each grid-edge face of a
    domain cell belongs to ('' where the cell is not in the domain), per grid
    edge -- left and right bottom to top, bottom and top left to right."""
    act = domain_mask(domain)
    ny, nx = act.shape
    edges = drawn_edges(domain)
    out = {}
    spec_ = {"left": (np.zeros(ny), np.arange(ny) + 0.5, act[:, 0]),
             "right": (np.full(ny, float(nx)), np.arange(ny) + 0.5, act[:, -1]),
             "bottom": (np.arange(nx) + 0.5, np.zeros(nx), act[0, :]),
             "top": (np.arange(nx) + 0.5, np.full(nx, float(ny)), act[-1, :])}
    for e, (x, y, on) in spec_.items():
        lab = np.array([""] * len(x), dtype=object)
        if on.any():
            lab[on] = nearest_edge(np.column_stack([x[on], y[on]]), edges)
        out[e] = lab
    return out


def drawn_edge_names(domain) -> list[str]:
    return [name for name, _poly in drawn_edges(domain)]


# -- every face of the domain's boundary, for the families that are not fv.py's -----


@dataclass
class BoundaryFaces:
    """Every face on a domain's boundary, labelled.

    The grid's own edge faces of domain cells come first -- left, right, bottom,
    top, each in order along it -- then a drawn domain's faces to the void inside
    the grid (`void_faces`).  ``cell`` is the domain cell's flat index,
    ``direction`` indexes `DIRECTIONS` (from the cell outward), ``edge`` names the
    edge the face belongs to: a grid edge (``left`` ...) on a domain with no drawn
    outline, else the drawn edge nearest it; ``along`` is a grid-edge face's place
    along its edge in cells (-1 for a face to the void)."""

    cell: np.ndarray
    direction: np.ndarray
    edge: np.ndarray
    along: np.ndarray

    @property
    def n(self) -> int:
        return int(self.cell.size)

    def nodes(self, nx: int) -> tuple[np.ndarray, np.ndarray]:
        """The two corner nodes of each face on the node grid ``(ny + 1) x (nx + 1)``,
        node ``(i, j)`` being ``j (nx + 1) + i``: a finite-element family's view."""
        i, j = self.cell % nx, self.cell // nx
        d = self.direction
        # right (+x): (i+1, j), (i+1, j+1); left: (i, j), (i, j+1);
        # top (+y): (i, j+1), (i+1, j+1); bottom: (i, j), (i+1, j)
        ia = np.where(d == 0, i + 1, i)
        ja = np.where(d == 2, j + 1, j)
        ib = np.where((d == 0) | (d == 2) | (d == 3), i + 1, i)
        jb = np.where((d == 0) | (d == 1) | (d == 2), j + 1, j)
        return ja * (nx + 1) + ia, jb * (nx + 1) + ib


#: DIRECTIONS index of each grid edge's outward direction
_EDGE_DIRECTION = {"left": 1, "right": 0, "bottom": 3, "top": 2}


def boundary_faces(domain) -> BoundaryFaces:
    """Every face of the domain's boundary, labelled (`BoundaryFaces`)."""
    act = domain_mask(domain)
    ny, nx = act.shape
    labels = grid_edge_labels(domain) if getattr(domain, "outline", None) is not None \
        else None
    cells, dirs, names, along = [], [], [], []
    for e in EDGES:
        n = ny if e in ("left", "right") else nx
        k = np.arange(n)
        c = {"left": k * nx, "right": k * nx + nx - 1, "bottom": k,
             "top": (ny - 1) * nx + k}[e]
        on = act.ravel()[c]
        cells.append(c[on])
        dirs.append(np.full(int(on.sum()), _EDGE_DIRECTION[e], dtype=np.int8))
        names.append(labels[e][on] if labels is not None
                     else np.full(int(on.sum()), e, dtype=object))
        along.append(k[on])
    drawn = getattr(domain, "outline", None) is not None or bool(getattr(domain, "holes", None))
    if drawn:
        vf = void_faces(domain)
        cells.append(vf.cell)
        dirs.append(vf.direction.astype(np.int8))
        names.append(vf.edge.astype(object))
        along.append(np.full(vf.n, -1, dtype=np.int64))
    return BoundaryFaces(np.concatenate(cells).astype(np.int64), np.concatenate(dirs),
                         np.concatenate(names).astype(object),
                         np.concatenate(along).astype(np.int64))


def face_conditions(boundaries, bf: BoundaryFaces, nx: int, ny: int) -> np.ndarray:
    """For each face of ``bf``, the index of the boundary (in ``boundaries``) that
    holds on it, or -1: a drawn edge's by its name, a grid edge's by the segment
    ``start <= along < stop`` it falls in."""
    out = np.full(bf.n, -1, dtype=np.int64)
    by_name = {}
    for k, b in enumerate(boundaries):
        if b.edge not in EDGES:
            by_name.setdefault(b.edge, k)
    for k, b in enumerate(boundaries):
        if b.edge in EDGES:
            stop = edge_length(b.edge, nx, ny) if b.stop is None else b.stop
            sel = (bf.edge == b.edge) & (bf.along >= b.start) & (bf.along < stop) & (out < 0)
            out[sel] = k
    for name, k in by_name.items():
        out[(bf.edge == name) & (out < 0)] = k
    return out


def edge_lengths(domain) -> dict[str, float]:
    """Each drawn edge's true length, in cells (the curve itself, not its staircase)."""
    return {name: float(np.sum(np.hypot(*np.diff(poly, axis=0).T)))
            for name, poly in drawn_edges(domain, step=0.25)}


def staircase_scale(domain, bf: BoundaryFaces) -> dict[str, float]:
    """For each drawn edge, its true length over the length of the cell faces the
    solver sees on it: a flux or a traction given per unit length of the drawn edge
    is applied on each face times this, so the edge carries its whole load and no
    more (a staircase along a diagonal is up to sqrt(2) longer than the diagonal)."""
    true = edge_lengths(domain)
    out = {}
    for name, length in true.items():
        n = int(np.sum(bf.edge == name))
        out[name] = length / n if n else 1.0
    return out


def along_grid_edge(domain, name: str, tol: float = 0.5) -> str | None:
    """The grid edge a drawn edge lies along (every point of it within ``tol`` cells
    of that edge), or None: how a family whose solver fixes its grid edges' conditions
    gives a drawn edge the condition of the grid edge it runs along."""
    nx, ny = domain.nx, domain.ny
    for e_name, poly in drawn_edges(domain, step=1.0):
        if e_name != name:
            continue
        x, y = poly[:, 0], poly[:, 1]
        for e, v in (("left", np.abs(x)), ("right", np.abs(x - nx)), ("bottom", np.abs(y)),
                     ("top", np.abs(y - ny))):
            if float(np.max(v)) <= tol:
                return e
        return None
    return None


def node_mask(cells: np.ndarray) -> np.ndarray:
    """The nodes of a set of cells -- their corners, on the ``(ny + 1) x (nx + 1)``
    node grid a finite-element family's unknowns live on."""
    cells = np.asarray(cells, dtype=bool)
    ny, nx = cells.shape
    out = np.zeros((ny + 1, nx + 1), dtype=bool)
    out[:-1, :-1] |= cells
    out[:-1, 1:] |= cells
    out[1:, :-1] |= cells
    out[1:, 1:] |= cells
    return out


def node_masks(spec) -> list[tuple[str, np.ndarray]]:
    """Each window's nodes: the corners of its cells inside the domain."""
    return [(wid, node_mask(m)) for wid, m in window_masks(spec)]


# -- smooth outlines of cell sets (for drawing generated windows) -----------------

# marching squares: for each of the 16 corner patterns of a square of four cell
# centres (bit 1 lower-left, 2 lower-right, 4 upper-right, 8 upper-left, set where
# the field is at or above the level), the pairs of square sides the contour joins
# (0 bottom, 1 right, 2 top, 3 left).  5 and 10 are saddles, settled by the centre.
_MS = {1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),), 4: ((1, 2),), 6: ((0, 2),),
       7: ((3, 2),), 8: ((2, 3),), 9: ((0, 2),), 11: ((1, 2),), 12: ((1, 3),),
       13: ((0, 1),), 14: ((3, 0),)}


def contour_lines(field: np.ndarray, level: float) -> list[np.ndarray]:
    """Polylines where a field sampled at the cell centres crosses ``level``, in the
    grid's cell coordinates (a centre is at ``(i + 1/2, j + 1/2)``): marching
    squares with linear interpolation, the segments chained end to end."""
    f = np.asarray(field, dtype=float) - level
    ny, nx = f.shape
    if ny < 2 or nx < 2:
        return []
    bl, br, tr, tl = f[:-1, :-1], f[:-1, 1:], f[1:, 1:], f[1:, :-1]
    code = ((bl >= 0) * 1 + (br >= 0) * 2 + (tr >= 0) * 4 + (tl >= 0) * 8).astype(np.int8)
    jj, ii = np.nonzero((code != 0) & (code != 15))

    def side(j, i, s):
        """The crossing on side ``s`` of the square with lower-left centre (i, j)."""
        a, b, pa, pb = {0: (bl, br, (0, 0), (1, 0)), 1: (br, tr, (1, 0), (1, 1)),
                        2: (tl, tr, (0, 1), (1, 1)), 3: (bl, tl, (0, 0), (0, 1))}[s]
        va, vb = a[j, i], b[j, i]
        t = va / (va - vb) if va != vb else 0.5
        return (i + 0.5 + pa[0] + t * (pb[0] - pa[0]), j + 0.5 + pa[1] + t * (pb[1] - pa[1]))
    segs = []
    for j, i in zip(jj.tolist(), ii.tolist()):
        c = int(code[j, i])
        if c in (5, 10):
            centre = 0.25 * (bl[j, i] + br[j, i] + tr[j, i] + tl[j, i])
            pairs = (((3, 2), (0, 1)) if (c == 5) == (centre >= 0) else ((3, 0), (1, 2)))
        else:
            pairs = _MS[c]
        for s0, s1 in pairs:
            segs.append((side(j, i, s0), side(j, i, s1)))
    # chain the segments: each crossing is shared by the two squares either side of it
    key = lambda p: (round(p[0], 9), round(p[1], 9))  # noqa: E731
    ends: dict[tuple, list[int]] = {}
    for n, (p, q) in enumerate(segs):
        ends.setdefault(key(p), []).append(n)
        ends.setdefault(key(q), []).append(n)
    used = [False] * len(segs)
    lines = []
    for n in range(len(segs)):
        if used[n]:
            continue
        used[n] = True
        line = [segs[n][0], segs[n][1]]
        for forward in (True, False):
            while True:
                tip = line[-1] if forward else line[0]
                nxt = [m for m in ends.get(key(tip), ()) if not used[m]]
                if not nxt:
                    break
                m = nxt[0]
                used[m] = True
                p, q = segs[m]
                new = q if key(p) == key(tip) else p
                if forward:
                    line.append(new)
                else:
                    line.insert(0, new)
        lines.append(np.asarray(line))
    return lines


def inner_boundary(mask: np.ndarray, active: np.ndarray, sigma: float = 0.9) -> list[np.ndarray]:
    """A cell set's boundary INSIDE the domain, as smooth polylines: the 1/2 contour
    of its indicator smoothed by a Gaussian of ``sigma`` cells, with the stretches
    along the domain's own edges left out (the domain's outline draws those).  For a
    generated window these are its cuts: smooth curves across the domain."""
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(np.asarray(mask, dtype=float), sigma, mode="constant")
    inside = gaussian_filter(np.asarray(active, dtype=float), 1.2, mode="constant") > 0.97
    ny, nx = f.shape
    out = []
    for line in contour_lines(f, 0.5):
        i = np.clip(np.floor(line[:, 0]).astype(int), 0, nx - 1)
        j = np.clip(np.floor(line[:, 1]).astype(int), 0, ny - 1)
        keep = inside[j, i]
        # split where the line runs along the domain's edge
        start = None
        for n in range(len(line) + 1):
            if n < len(line) and keep[n]:
                start = n if start is None else start
            elif start is not None:
                if n - start >= 2:
                    out.append(line[start:n])
                start = None
    return out


__all__ = ["Box", "SNAP_STEPS", "DEFAULT_SNAP", "snap_value", "snap_box", "corners",
           "resize_from_corner", "intersect", "clip_box", "tile", "WindowAnalysis",
           "analyse_windows", "mask_to_boxes", "region_mask", "region_owner",
           "polygon_area", "EDGES", "DRAWN_EDGE",
           "edge_length", "edge_segment_xy", "outline_ring", "shape_mask", "domain_mask",
           "window_mask", "is_plain", "window_masks", "touching", "faces_between",
           "ramp_weight", "analyse_masks", "analyse_case", "DIRECTIONS", "drawn_edges",
           "nearest_edge", "VoidFaces", "void_faces", "grid_edge_labels",
           "drawn_edge_names", "mask_to_runs", "runs_to_mask", "contour_lines",
           "inner_boundary", "BoundaryFaces", "boundary_faces", "face_conditions",
           "edge_lengths", "staircase_scale", "along_grid_edge", "node_mask",
           "node_masks"]
