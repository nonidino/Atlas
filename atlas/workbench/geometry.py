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

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

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
    """The cells a region is drawn over: its rectangle, or the cell centres inside
    its polygon and outside its holes.  Read-only."""
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


__all__ = ["Box", "SNAP_STEPS", "DEFAULT_SNAP", "snap_value", "snap_box", "corners",
           "resize_from_corner", "intersect", "clip_box", "tile", "WindowAnalysis",
           "analyse_windows", "mask_to_boxes", "region_mask", "region_owner",
           "polygon_area", "EDGES",
           "edge_length", "edge_segment_xy"]
