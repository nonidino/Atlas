"""Where a learned expert can sit on the body-fitted column.

PoC 3, Tier 67.  [[poc3-racelab-car-windows]].

The requirements decided this and the decision has a price nobody had measured.
Section 4.1's decided row reads: *the background keeps rectangular windows, and
the learned expert runs only there, because Poseidon-T accepts nothing but a
uniform 128 x 128 grid.*  This module works out what "only there" amounts to on
the car's own composite, and the answer is a smaller place than it sounds.

The arithmetic that fixes everything
------------------------------------

The body-fitted background is 241 x 672 at ``h = 1/64`` -- the porous column's
own lattice, 240 x 672, plus a row -- so `racelab.WX`, `racelab.WY` and the
scaling `racelab_switch` derives from them carry over unchanged: **128 cells at
1/64 is 2.0 length units**, and there is no choice in it.  A learned window is
therefore a 2.0 x 2.0 square, on a domain 10.5 x 3.766, with the car punched
through the middle of it.

Three numbers, and the third is the one that matters
----------------------------------------------------

Measured on the car's composite (Tier 67):

1. **29.2%** of the 62,130 possible placements are free of holes -- 18,109 of
   them.  So there is no shortage of *places*.
2. A greedy non-overlapping tiling finds **four** windows, covering 44.6% of the
   background's live cells; the union of every hole-free placement reaches
   85.7%, which is the ceiling no tiling can pass.
3. **The body-fitted grids hold 61.0% of the composite's 377,267 unknowns**, and
   a uniform 128 x 128 window cannot accept a curvilinear grid at all.

So the learned expert's territory is bounded by (3), not by (1).  At best it
touches about a third of the problem; under a real tiling, about a sixth.  **And
none of it is near the car**: the boundary layers, the wheel clearances and the
cooling duct are exactly the 61%, and they are the physics the body-fitted
column was built to resolve.

That is not an argument against the decision -- a uniform-grid checkpoint cannot
be asked to do otherwise -- but it is the number the demo's speed-and-accuracy
story has to be told against, and it belongs beside the switch rather than in a
footnote.

What a window is here
---------------------

A `Window` is a block of the BACKGROUND's cells, addressed by its top-left index
``(j, i)`` and carrying its physical box.  `extract` returns the dense ``(w, w)``
array, and `scatter_into` puts one back, so a window is a lossless view of a
piece of the composite and not a copy that can drift from it.

**There are two admissibility tests and the obvious one is wrong.**  "No holes"
(`require="live"`) is what a checkpoint needs to have something to read
everywhere.  But a hole-free window can still contain `INTERP` cells -- the
background's fringe, 2,153 of them, whose values are interpolated FROM the
body-fitted grids rather than computed by the background's own equation.  A
learned expert handed those is handed numbers it does not own, and writing its
answer back over them would overwrite the coupling that makes the composite one
implicit solve (Tier 65).  `require="owned"` demands every cell be a `DISC` cell.
`REQUIRE` names both and `territory` reports both, because the gap between them
is the difference between a window a checkpoint can be *fed* and a window it can
be given *responsibility for*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from . import overset as OV

#: What Poseidon-T accepts, and the only size it accepts.  `racelab.WX`/`WY`.
POSEIDON_CELLS = 128

#: The orders `tile` will greedily take placements in.  A greedy tiling depends
#: on its order, so the order is DECLARED rather than left to whatever
#: ``np.nonzero`` happens to return -- and Tier 67 measures whether the count
#: depends on it.
ORDERS = ("row", "column", "downstream", "upstream")


@dataclass(frozen=True)
class Window:
    """A 128 x 128 block of the background, and where it sits in the world."""

    j: int
    i: int
    w: int
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x0, self.x1, self.y0, self.y1)

    def slices(self) -> tuple[slice, slice]:
        return (slice(self.j, self.j + self.w), slice(self.i, self.i + self.w))


def _background(ov):
    if ov.bg is None:
        raise ValueError("a learned window needs a background grid")
    return ov.bg


#: The two admissibility tests, and the difference between them is a finding.
#:
#: ``"live"``  -- every cell carries an unknown (no holes).  The obvious test,
#:               and the wrong one.
#: ``"owned"`` -- every cell is a `DISC` cell, one the BACKGROUND'S OWN equation
#:               computes.  A hole-free window can still contain `INTERP` cells,
#:               whose values are interpolated FROM a body-fitted grid; handing
#:               those to a learned expert hands it numbers it does not own, and
#:               writing its answer back over them would overwrite the very
#:               coupling that makes the composite one solve (Tier 65).
REQUIRE = ("live", "owned")


def live_mask(ov) -> np.ndarray:
    """True where the background carries an unknown; False in the car's holes."""
    bg = _background(ov)
    return ov.status[bg.name] != OV.HOLE


def owned_mask(ov) -> np.ndarray:
    """True where the background's own equation computes the cell.

    `INTERP` cells are excluded: they are the fringe that receives from the body
    grids, and they are unknowns the background holds but does not decide.
    """
    bg = _background(ov)
    return ov.status[bg.name] == OV.DISC


def _mask_for(ov, require: str) -> np.ndarray:
    if require == "live":
        return live_mask(ov)
    if require == "owned":
        return owned_mask(ov)
    raise ValueError(f"require must be one of {REQUIRE}, got {require!r}")


def hole_free(ov, w: int = POSEIDON_CELLS, require: str = "live") -> np.ndarray:
    """``(ny-w+1, nx-w+1)`` booleans: is the ``w x w`` block at this top-left admissible?

    ``require`` picks the test -- see `REQUIRE`.  Computed with a summed-area
    table rather than a loop, because the background has 62,130 candidate
    placements and each would otherwise cost a 128 x 128 reduction.
    """
    live = _mask_for(ov, require)
    ny, nx = live.shape
    if w < 1:
        raise ValueError("a window needs at least one cell")
    if ny < w or nx < w:
        return np.zeros((max(0, ny - w + 1), max(0, nx - w + 1)), dtype=bool)
    ii = np.zeros((ny + 1, nx + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum((~live).astype(np.int64), axis=0), axis=1)
    J, I = ny - w + 1, nx - w + 1
    holes = ii[w:, w:] - ii[:J, w:] - ii[w:, :I] + ii[:J, :I]
    return holes == 0


def reach(ov, w: int = POSEIDON_CELLS, require: str = "live") -> np.ndarray:
    """The union of every hole-free placement: the most a learned expert could EVER
    touch on this background, whatever tiling is chosen."""
    free = hole_free(ov, w, require)
    live = _mask_for(ov, require)
    out = np.zeros(live.shape, dtype=bool)
    if not free.any():
        return out
    # a cell is reached when some admissible window contains it: the same
    # summed-area trick, run the other way round
    J, I = free.shape
    acc = np.zeros((live.shape[0] + 1, live.shape[1] + 1), dtype=np.int64)
    jj, iijj = np.nonzero(free)
    np.add.at(acc, (jj, iijj), 1)
    np.add.at(acc, (jj + w, iijj), -1)
    np.add.at(acc, (jj, iijj + w), -1)
    np.add.at(acc, (jj + w, iijj + w), 1)
    cover = np.cumsum(np.cumsum(acc, axis=0), axis=1)[:live.shape[0], :live.shape[1]]
    out = cover > 0
    return out


def _order_key(order: str, jj: np.ndarray, ii: np.ndarray, nx: int) -> np.ndarray:
    if order == "row":
        return jj.astype(np.int64) * nx + ii
    if order == "column":
        return ii.astype(np.int64) * (jj.max() + 1) + jj
    if order == "downstream":
        return -(ii.astype(np.int64)) * (jj.max() + 1) - jj
    if order == "upstream":
        return ii.astype(np.int64) * (jj.max() + 1) + jj
    raise ValueError(f"order must be one of {ORDERS}, got {order!r}")


def tile(ov, w: int = POSEIDON_CELLS, order: str = "row",
         require: str = "live") -> list[Window]:
    """A greedy non-overlapping set of admissible windows, in a DECLARED order.

    Greedy, so not provably maximal -- `ORDERS` exists so the dependence on the
    order can be measured rather than assumed away.
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    bg = _background(ov)
    free = hole_free(ov, w, require)
    ny, nx = _mask_for(ov, require).shape
    out: list[Window] = []
    if not free.any():
        return out
    jj, ii = np.nonzero(free)
    taken = np.zeros((ny, nx), dtype=bool)
    for k in np.argsort(_order_key(order, jj, ii, nx), kind="stable"):
        j, i = int(jj[k]), int(ii[k])
        if taken[j:j + w, i:i + w].any():
            continue
        taken[j:j + w, i:i + w] = True
        out.append(Window(j=j, i=i, w=w,
                          x0=float(bg.X[j, i]), x1=float(bg.X[j, i] + w * bg.h),
                          y0=float(bg.Y[j, i]), y1=float(bg.Y[j, i] + w * bg.h)))
    return out


# ---------------------------------------------------------------------------
# reading and writing one window
# ---------------------------------------------------------------------------


def extract(ov, win: Window, vec: np.ndarray) -> np.ndarray:
    """The window's values as a dense ``(w, w)`` array.

    Raises if the window has a hole, rather than returning an array with a
    quietly wrong number in it: a checkpoint cannot be told about a hole.
    """
    bg = _background(ov)
    idx = ov.index[bg.name][win.slices()]
    if np.any(idx < 0):
        raise OV.OversetError(
            f"window at (j={win.j}, i={win.i}) covers {int((idx < 0).sum())} cell(s) "
            "with no unknown -- it is not admissible and must not be extracted")
    vec = np.asarray(vec, dtype=float).ravel()
    return vec[idx]


def scatter_into(ov, win: Window, vec: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Write a window's values back into a solution vector, returning a new vector."""
    bg = _background(ov)
    idx = ov.index[bg.name][win.slices()]
    if np.any(idx < 0):
        raise OV.OversetError("window is not admissible")
    block = np.asarray(block, dtype=float)
    if block.shape != (win.w, win.w):
        raise ValueError(f"expected a {win.w} x {win.w} block, got {block.shape}")
    out = np.array(vec, dtype=float).ravel().copy()
    out[idx] = block
    return out


# ---------------------------------------------------------------------------
# the territory, which is the tier's actual finding
# ---------------------------------------------------------------------------


def territory(ov, w: int = POSEIDON_CELLS, order: str = "row",
              require: str = "live") -> dict[str, Any]:
    """What a learned expert can and cannot reach on this composite.

    The denominators matter and are all reported: a fraction of the BACKGROUND's
    live cells flatters the learned expert, because the background is only 39% of
    the composite's unknowns.  The fraction that decides anything is of the
    COMPOSITE.
    """
    bg = _background(ov)
    live = _mask_for(ov, require)
    free = hole_free(ov, w, require)
    R = reach(ov, w, require)
    wins = tile(ov, w, order=order, require=require)
    covered = np.zeros(live.shape, dtype=bool)
    for win in wins:
        covered[win.slices()] = True
    body = sum(int((ov.status[c.name] != OV.HOLE).sum()) for c in ov.comps)
    n = int(ov.n_unknowns)
    bg_live = int(live.sum())
    return {
        "window_cells": int(w), "window_span": float(w * bg.h),
        "background_shape": list(live.shape), "h": float(bg.h),
        "background_live": int(live_mask(ov).sum()),
        "background_owned": int(owned_mask(ov).sum()),
        "background_interp": int((ov.status[bg.name] == OV.INTERP).sum()),
        "background_holes": int((ov.status[bg.name] == OV.HOLE).sum()),
        "placements": int(free.size), "placements_hole_free": int(free.sum()),
        "placements_hole_free_frac": float(free.mean()) if free.size else 0.0,
        "require": require, "order": order, "n_windows": len(wins),
        "windows": [w_.__dict__ for w_ in wins],
        "tiled_cells": int(covered.sum()),
        "tiled_frac_of_background_live": float((covered & live).sum() / max(1, bg_live)),
        "reach_cells": int((R & live).sum()),
        "reach_frac_of_background_live": float((R & live).sum() / max(1, bg_live)),
        # the denominators that decide anything
        "composite_unknowns": n,
        "body_grid_unknowns": body,
        "body_grid_frac": float(body / n),
        "tiled_frac_of_composite": float((covered & live).sum() / n),
        "reach_frac_of_composite": float((R & live).sum() / n),
        "note": ("the learned expert is bounded by the body grids, not by the hole "
                 "pattern: a uniform window cannot accept a curvilinear grid at all"),
    }
