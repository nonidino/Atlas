"""A case's windows as a partition of unity: any rectangles, one weight rule.

`wake_array.ArrayTiling` assumes one window size and one stride, so it can only
express CS-7's regular tilings.  A case drawn in the workbench is any set of
rectangles.  This builds the same weights for any boxes:

* each window's raw weight ramps from 0 at each ARTIFICIAL face (a face not on
  the domain boundary) to 1 over ``ramp`` cells, per axis, and the two axes
  combine as ``min(wy, wx)**2`` -- `ArrayTiling.weights`' formula, and the ramp
  `geometry._axis_full` checks the full-weight rule on;
* the raw weights are summed window by window, in list order, and every window
  is divided by that sum cellwise.

**On a regular tiling the result is `ArrayTiling.weights()` to the bit** (a test
pins all three measured tilings).  That is what lets a run of an example case
be W346's arithmetic rather than a re-derivation of it: the blend below adds
``chi_k * u_k`` into the global field in window order, exactly as
`ArrayTiling.assemble` does.  The sum is accumulated inside each window's own
box rather than over full-domain copies, which is the same sequence of
floating-point additions (a window adds exactly ``0.0`` outside its box in the
full-array form) at a fraction of the memory: 48 full copies of the 21-rotor
domain would be 240 MB.

**The partition of unity is certified, not assumed**: `partition_of_unity`
returns `assembly.GridPartitionOfUnity`, and `certify` reads its identity
residual ``max |sum_i R_i^T chi_i R_i - 1|`` and its smallest weight, the two
quantities the compiler's L6 checks (R11 refuses a signed partition).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..assembly import CONVEX_TOL, POU_TOL, GridPartitionOfUnity

Box = tuple[int, int, int, int]


def axis_weight(n: int, lo_face: bool, hi_face: bool, ramp: int) -> np.ndarray:
    """One axis of a window's raw weight: `ArrayTiling.weights`' ramp, cell for cell."""
    r = max(int(ramp), 1)
    idx = np.arange(n) + 0.5
    w = np.ones(n)
    if lo_face:
        w = np.minimum(w, np.clip(idx / r, 0.0, 1.0))
    if hi_face:
        w = np.minimum(w, np.clip((n - idx) / r, 0.0, 1.0))
    return w


@dataclass(frozen=True)
class Certificate:
    """What `certify` read off the partition of unity."""

    identity_residual: float
    chi_min: float
    tolerance: float = POU_TOL

    @property
    def holds(self) -> bool:
        return self.identity_residual <= self.tolerance and self.chi_min >= -CONVEX_TOL

    def as_dict(self) -> dict:
        return {"identity_residual": self.identity_residual, "chi_min": self.chi_min,
                "tolerance": self.tolerance, "holds": self.holds}


class RectangleTiling:
    """Windows given as boxes ``(x0, y0, nx, ny)`` in cells, on an ``nx x ny`` domain.

    Windows must lie inside the domain (the case check refuses one that does
    not).  ``names`` and ``boxes`` keep the case's order, and that order is the
    order of every sum below.
    """

    def __init__(self, nx: int, ny: int, windows: Sequence[tuple[str, Box]], ramp: int):
        self.nx, self.ny, self.ramp = int(nx), int(ny), int(ramp)
        self.names = [str(n) for n, _ in windows]
        self.boxes: list[Box] = [tuple(int(v) for v in b) for _, b in windows]
        for name, (x0, y0, w, h) in zip(self.names, self.boxes):
            if x0 < 0 or y0 < 0 or w < 1 or h < 1 or x0 + w > self.nx or y0 + h > self.ny:
                raise ValueError(f"window {name} ({x0}, {y0}, {w} x {h}) is not inside "
                                 f"the {self.nx} x {self.ny} domain")
        raw: list[np.ndarray] = []
        tot = np.zeros((self.ny, self.nx))
        for x0, y0, w, h in self.boxes:
            wx = axis_weight(w, x0 > 0, x0 + w < self.nx, self.ramp)
            wy = axis_weight(h, y0 > 0, y0 + h < self.ny, self.ramp)
            r = np.minimum(wy[:, None], wx[None, :]) ** 2
            raw.append(r)
            tot[y0:y0 + h, x0:x0 + w] += r
        tot = np.where(tot <= 0.0, 1.0, tot)
        #: chi_k on window k's own box, (h, w)
        self.chi: list[np.ndarray] = [r / tot[y0:y0 + h, x0:x0 + w]
                                      for r, (x0, y0, w, h) in zip(raw, self.boxes)]

    # -- cutting and blending ---------------------------------------------

    @property
    def n_windows(self) -> int:
        return len(self.boxes)

    def shapes(self) -> dict[tuple[int, int], list[int]]:
        """(ny, nx) of a window -> the indices of the windows of that shape, in order.

        A batch solver steps windows of one shape as one array, so a tiling is
        stepped one shape group at a time.
        """
        out: dict[tuple[int, int], list[int]] = {}
        for k, (_x0, _y0, w, h) in enumerate(self.boxes):
            out.setdefault((h, w), []).append(k)
        return out

    def cut_one(self, f: np.ndarray, k: int) -> np.ndarray:
        x0, y0, w, h = self.boxes[k]
        return f[y0:y0 + h, x0:x0 + w]

    def cut(self, f: np.ndarray, idx: Sequence[int]) -> np.ndarray:
        """The windows ``idx`` (all one shape) as a stacked ``[B, h, w]`` copy."""
        return np.stack([self.cut_one(f, k) for k in idx])

    def assemble(self, locals_: Sequence[np.ndarray]) -> np.ndarray:
        """``sum_k R_k^T chi_k u_k``: the blend, in window order."""
        out = np.zeros((self.ny, self.nx))
        for k, (x0, y0, w, h) in enumerate(self.boxes):
            out[y0:y0 + h, x0:x0 + w] += self.chi[k] * locals_[k]
        return out

    # -- the same blend, its larger part on threads (demo item 1.4) -------------

    def _owned(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per window, (its cells no other window covers and its weight is exactly 1
        on, the rest of its box), as masks of its box; computed once.  On an owned
        cell `assemble`'s sum is ``0 + 1 u_k``, which is ``u_k``."""
        if getattr(self, "_owned_masks", None) is None:
            count = np.zeros((self.ny, self.nx), dtype=np.int32)
            for x0, y0, w, h in self.boxes:
                count[y0:y0 + h, x0:x0 + w] += 1
            out = []
            for k, (x0, y0, w, h) in enumerate(self.boxes):
                own = (count[y0:y0 + h, x0:x0 + w] == 1) & (self.chi[k] == 1.0)
                out.append((own, ~own))
            self._owned_masks = out
        return self._owned_masks

    def write_own(self, out: np.ndarray, k: int, local: np.ndarray) -> None:
        """Window ``k``'s owned cells into ``out`` (ny, nx) as they are.  Different
        windows' owned cells are disjoint, so this is safe from any thread."""
        x0, y0, w, h = self.boxes[k]
        np.copyto(out[y0:y0 + h, x0:x0 + w], local, where=self._owned()[k][0])

    def add_shared(self, out: np.ndarray, locals_: Sequence[np.ndarray]) -> None:
        """The cells more than one window covers, blended in window order: with
        `write_own` on a zeroed ``out``, `assemble`'s value on every cell."""
        for k, (x0, y0, w, h) in enumerate(self.boxes):
            shared = self._owned()[k][1]
            sub = out[y0:y0 + h, x0:x0 + w]
            sub[shared] += self.chi[k][shared] * locals_[k][shared]

    # -- the certificate ---------------------------------------------------

    def partition_of_unity(self) -> GridPartitionOfUnity:
        idx, wts = {}, {}
        for name, (x0, y0, w, h), chi in zip(self.names, self.boxes, self.chi):
            rows, cols = np.meshgrid(np.arange(y0, y0 + h), np.arange(x0, x0 + w),
                                     indexing="ij")
            idx[name] = (rows * self.nx + cols).reshape(-1)
            wts[name] = chi.reshape(-1)
        return GridPartitionOfUnity(
            self.nx * self.ny, idx, wts, ramp_cells=self.ramp,
            profile=f"min(wy, wx)^2 over a {self.ramp}-cell linear ramp, normalized "
                    f"(atlas/workbench/tiling.py, ArrayTiling's rule on any rectangles)")

    def certify(self) -> Certificate:
        pou = self.partition_of_unity()
        return Certificate(identity_residual=pou.identity_residual(), chi_min=pou.chi_min())


class MaskTiling:
    """Windows of any shape on a domain of any shape (case file 0.4).

    The same partition of unity as `RectangleTiling` -- each window's raw weight
    ramps from 0 at its artificial faces to 1 over ``ramp`` cells and is squared,
    the raw weights are summed window by window in list order, and every window
    is divided by that sum -- with the distance to the window's artificial
    boundary in place of the per-axis distance (`geometry.ramp_weight`).  A face
    on the domain's own boundary, drawn or the grid's, is not artificial.

    Each window's weights are held on its own cells: ``idx[k]`` are the flat
    indices of window k's cells in the domain, ascending -- the order
    `fv.assemble` numbers a set's cells in -- and ``chi[k]`` the weights there.
    The certificate is taken over the domain's cells only; a cell of the grid
    outside the domain is in no window and needs no weight.
    """

    def __init__(self, active: np.ndarray, windows: Sequence[tuple[str, np.ndarray]],
                 ramp: int):
        from .geometry import ramp_weight
        self.active = np.asarray(active, dtype=bool)
        self.ny, self.nx = self.active.shape
        self.ramp = int(ramp)
        self.names = [str(n) for n, _ in windows]
        masks = [np.asarray(m, dtype=bool) & self.active for _, m in windows]
        self.idx: list[np.ndarray] = [np.flatnonzero(m.ravel()) for m in masks]
        raw = [ramp_weight(m, self.active, self.ramp) for m in masks]
        tot = np.zeros((self.ny, self.nx))
        for r in raw:
            tot += r
        tot = np.where(tot <= 0.0, 1.0, tot)
        #: chi_k on window k's own cells, aligned with idx[k]
        self.chi: list[np.ndarray] = [(r / tot).ravel()[i] for r, i in zip(raw, self.idx)]

    @property
    def n_windows(self) -> int:
        return len(self.idx)

    def assemble(self, locals_: Sequence[np.ndarray]) -> np.ndarray:
        """``sum_k R_k^T chi_k u_k`` on the grid, in window order (0 off the domain)."""
        out = np.zeros(self.nx * self.ny)
        for i, c, u in zip(self.idx, self.chi, locals_):
            out[i] += c * u
        return out.reshape(self.ny, self.nx)

    def partition_of_unity(self) -> GridPartitionOfUnity:
        cells = np.flatnonzero(self.active.ravel())
        pos = np.full(self.nx * self.ny, -1, dtype=np.int64)
        pos[cells] = np.arange(cells.size)
        return GridPartitionOfUnity(
            int(cells.size), {n: pos[i] for n, i in zip(self.names, self.idx)},
            {n: c for n, c in zip(self.names, self.chi)}, ramp_cells=self.ramp,
            profile=f"clip((d - 1/2) / {self.ramp}, 0, 1)^2 with d the distance to the "
                    f"window's artificial boundary, normalized over the domain's cells "
                    f"(atlas/workbench/tiling.py MaskTiling, windows of any shape)")

    def certify(self) -> Certificate:
        pou = self.partition_of_unity()
        return Certificate(identity_residual=pou.identity_residual(), chi_min=pou.chi_min())


__all__ = ["Box", "axis_weight", "Certificate", "RectangleTiling", "MaskTiling"]
