"""The body-fitted composite on one raster: the demo's field overlay.

PoC 3, Tier 66.  [[poc3-racelab-car-render]].

The porous column's fluid is fourteen rectangular windows on one lattice, so a
picture of it is an array slice.  The body-fitted column is twelve overlapping
grids -- a Cartesian background and eleven curvilinear ones wrapped round the
car's parts -- and there is no array that holds the answer.  A screen needs one
rectangle of pixels, so something has to turn the composite into it, and the
whole question is whether that something can be trusted to draw the solve
rather than a second, independently wrong, opinion of it.

Why the background alone will not do
------------------------------------

The background covers the whole domain and is already rectangular, so the cheap
thing is to draw it and stop.  That picture is blank exactly where the physics
is.  Measured on the car's own composite at 320 x 180 (Tier 66): of the 54,815
pixels that show fluid, the body-fitted grids own **5,137**, and those are the
boundary layers, the wheel clearances and the cooling duct -- the places the
body-fitted column was built FOR.  The background has holes cut through it
there, so drawing it alone would not merely be coarse, it would show nothing.

The rule, which is the solver's own
-----------------------------------

`car_union.probe_matrix_masked` is the operator the devices' strips already use
to read the composite: for each point, the first body grid in ``ov.comps`` order
that holds it with a usable stencil, else the background.  This module asks that
same operator over a raster.  **It is deliberately not a second implementation**
-- if the picture's interpolation and the solve's interpolation were written
twice they could come to disagree, and a picture that disagrees with the solve
is the artifact this project exists not to produce.

What a raster has that a strip does not is points inside the bodies.  A device's
strip must reach every one of its points, so `probe_matrix` raises; a raster
covers the car as well as the air, and the pixels on the car are not an error,
they are the car.  Those become the **mask**: `probe_matrix_masked` reports them
and this module never reads them.

The check that makes the picture trustworthy
--------------------------------------------

The donor weights reproduce linear fields by construction, so a linear field
laid on every node of every grid must come back through the probe EXACTLY.  It
does, to 3.6e-15 at every raster tried (Tier 66) -- which is the statement that
the twelve grids have been stitched in the right order with the right weights,
because almost any error in the stitching would show up as a kink.

That control needs its own control, or it proves nothing: a renderer that
ignored the grids and evaluated an analytic plane would pass it too.  So a
QUADRATIC field is rendered beside it and must NOT be exact.  It is not: 4.5e-6,
and -- the part that identifies what is being measured -- that error does not
move when the raster is refined (5.4e-6 at 160 x 90, 5.5e-6 at 640 x 360),
because it is set by the GRID spacing and not by the pixel spacing.

The cost, and where it falls
----------------------------

Building the operator is the expensive half and it is paid once: 15.9 s at
160 x 90, 63.9 s at 320 x 180, 261.6 s at 640 x 360, all on mains on the user's
machine (which throttles; see the record's `machine`).  Applying it is a sparse
matrix-vector product: 0.0003 s at 320 x 180, against a march step of 0.62 to
0.81 s (Tier 64).  So a frame costs about 0.04% of a step and the demo's frame
rate is set by the solver, which is the right way round.

**The operator depends on the geometry and not on the state**, so it survives
every step and can be cached across runs of the same car; `RASTER_CACHE_NOTE`
says what would have to be in the key.
"""

from __future__ import annotations

import io
import time
from typing import Any, Sequence

import numpy as np

from . import car_union as CU

#: The demo's default raster.  320 x 180 is 16:9, costs 63.9 s to build and
#: 0.0003 s to apply, and resolves the duct (which is 29 cells across the band,
#: about 9 pixels here).  640 x 360 costs 261.6 s to build for a picture the
#: page scales down anyway.
NX, NY = 320, 180

#: What a cache key for the built operator carries: the operator is a function of
#: the geometry and the raster ALONE.
RASTER_CACHE_NOTE = (
    "racelab.geometry_fingerprint(), the grid settings car_solids.GRID, the "
    "hole margin, the stencil width, and (nx, ny, extent) -- but NOT the state, "
    "the time, or anything the march changes"
)

#: Bump when anything about how the operator is BUILT changes -- the donor
#: search, the grid order, the mask rule.  A key describes the inputs; this
#: describes the construction, and a cache that forgets it would serve an
#: operator built by code that no longer exists.
RASTER_CACHE_VERSION = 1


class CompositeRaster:
    """One rectangle of pixels over an overset composite, built once.

    ``ov`` is a `overset_multi.MultiOverset`; ``extent`` is ``(x0, x1, y0, y1)``
    and defaults to the background's own.  After construction, `sample` turns
    any solution vector into an ``(ny, nx)`` array with ``nan`` on the car.
    """

    def __init__(self, ov, nx: int = NX, ny: int = NY,
                 extent: tuple[float, float, float, float] | None = None) -> None:
        if nx < 2 or ny < 2:
            raise ValueError("a raster needs at least 2 x 2 pixels")
        self.ov = ov
        self.nx = int(nx)
        self.ny = int(ny)
        bg = ov.bg
        if bg is None:
            raise ValueError("a composite raster needs a background grid")
        if extent is None:
            extent = (float(bg.X.min()), float(bg.X.max()),
                      float(bg.Y.min()), float(bg.Y.max()))
        self.extent = tuple(float(v) for v in extent)
        x0, x1, y0, y1 = self.extent
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"extent must be increasing, got {self.extent}")
        self.xs = np.linspace(x0, x1, self.nx)
        self.ys = np.linspace(y0, y1, self.ny)
        PX, PY = np.meshgrid(self.xs, self.ys)
        self.px = PX.ravel()
        self.py = PY.ravel()
        t0 = time.perf_counter()
        self.M, missing, source = CU.probe_matrix_masked(ov, self.px, self.py)
        self.build_s = time.perf_counter() - t0
        #: True where no grid holds the pixel -- the car's interior.
        self.mask = missing.reshape(self.ny, self.nx)
        #: Which grid drew each pixel, indexing `grid_names`; -1 where masked.
        self.source = source.reshape(self.ny, self.nx)
        self.grid_names = [c.name for c in ov.comps] + [ov.bg.name]

    # -- reading the composite ---------------------------------------------

    def sample(self, vec: np.ndarray) -> np.ndarray:
        """``vec`` (one value per unknown) as an ``(ny, nx)`` array, nan on the car."""
        vec = np.asarray(vec, dtype=float).ravel()
        if vec.size != self.ov.n_unknowns:
            raise ValueError(f"expected {self.ov.n_unknowns} unknowns, got {vec.size}")
        a = (self.M @ vec).reshape(self.ny, self.nx)
        return np.where(self.mask, np.nan, a)

    def fields(self, U: np.ndarray, V: np.ndarray, P: np.ndarray | None = None
               ) -> dict[str, np.ndarray]:
        """The overlay's fields at once; `speed` is what the page draws by default."""
        u = self.sample(U)
        v = self.sample(V)
        out = {"u": u, "v": v, "speed": np.hypot(u, v)}
        if P is not None:
            out["p"] = self.sample(P)
        return out

    # -- what the record keeps ---------------------------------------------

    def report(self) -> dict[str, Any]:
        n = self.nx * self.ny
        masked = int(self.mask.sum())
        per_grid = {}
        for i, name in enumerate(self.grid_names):
            c = int((self.source == i).sum())
            if c:
                per_grid[name] = c
        return {"nx": self.nx, "ny": self.ny, "points": n, "extent": list(self.extent),
                "masked": masked, "masked_frac": masked / n,
                "drawn": n - masked, "nnz": int(self.M.nnz),
                "build_s": self.build_s, "per_grid": per_grid,
                "from_background": per_grid.get(self.ov.bg.name, 0),
                "from_body_grids": (n - masked) - per_grid.get(self.ov.bg.name, 0)}

    def apply_seconds(self, vec: np.ndarray, repeats: int = 20) -> float:
        """Measure the per-frame cost rather than quoting one (wall clock rots)."""
        vec = np.asarray(vec, dtype=float).ravel()
        t0 = time.perf_counter()
        for _ in range(int(repeats)):
            _ = self.M @ vec
        return (time.perf_counter() - t0) / float(repeats)


# ---------------------------------------------------------------------------
# the controls
# ---------------------------------------------------------------------------


def gather(ov, f) -> np.ndarray:
    """The global vector holding ``f(x, y)`` at every live node of every grid.

    The inverse of `overset.Overset.scatter`, and the way a manufactured field is
    put ON the composite so the renderer can be asked to take it off again.
    """
    vec = np.zeros(ov.n_unknowns)
    for g in ov.grids:
        idx = ov.index[g.name]
        live = idx >= 0
        X = g.X if hasattr(g, "X") else g.x
        Y = g.Y if hasattr(g, "Y") else g.y
        vec[idx[live]] = f(X[live], Y[live])
    return vec


def linear_control(raster: "CompositeRaster", a: float = 0.37, b: float = -0.81,
                   c: float = 2.5) -> dict[str, float]:
    """A plane through the composite and back out through the raster.

    Exact by construction -- the donor weights reproduce linear fields -- so any
    departure beyond rounding means the grids are being stitched wrongly.  The
    QUADRATIC beside it must not be exact, or the test is passing for a reason
    that has nothing to do with the grids.
    """
    ov = raster.ov
    ok = ~raster.mask
    if not ok.any():
        # a raster that draws nothing cannot pass a control, and must not be
        # allowed to pass it vacuously by reducing over an empty selection
        return {"linear_max_err": float("inf"), "quadratic_max_err": float("inf"),
                "a": a, "b": b, "c": c, "drawn": 0,
                "note": "every pixel is masked; there is nothing to check"}
    got = raster.sample(gather(ov, lambda X, Y: a * X + b * Y + c))
    want = (a * raster.px + b * raster.py + c).reshape(raster.ny, raster.nx)
    lin = float(np.max(np.abs(got[ok] - want[ok])))
    gq = raster.sample(gather(ov, lambda X, Y: X * X))
    wq = (raster.px ** 2).reshape(raster.ny, raster.nx)
    quad = float(np.max(np.abs(gq[ok] - wq[ok])))
    return {"linear_max_err": lin, "quadratic_max_err": quad,
            "a": a, "b": b, "c": c}


def mask_against_solids(raster: "CompositeRaster", solids: Sequence[Any]) -> dict[str, int]:
    """Is the masked set the car, exactly?

    The mask is defined by the DONOR SEARCH ("no grid holds this point"); the
    car is defined by the SOLIDS' OUTLINES, which the grids were generated from
    but which the search never consults.  They agreeing is therefore a real
    check and not a tautology: it says the hole cutting, the grid generation and
    the donor search all describe the same car.
    """
    from matplotlib.path import Path

    pts = np.column_stack([raster.px, raster.py])
    inside = np.zeros(pts.shape[0], dtype=bool)
    for s in solids:
        inside |= Path(np.asarray(s.outline, dtype=float)).contains_points(pts)
    inside = inside.reshape(raster.ny, raster.nx)
    m = raster.mask
    return {"masked": int(m.sum()), "inside_solid": int(inside.sum()),
            "both": int((m & inside).sum()),
            "masked_not_inside": int((m & ~inside).sum()),
            "inside_not_masked": int((inside & ~m).sum())}


# ---------------------------------------------------------------------------
# the picture
# ---------------------------------------------------------------------------

#: The masked pixels' colour.  A car drawn in a colour the field cannot produce
#: is a car the eye reads as a car; a car drawn in the field's own low colour is
#: a car the eye reads as slow air, which is the one misreading that matters
#: here (it is where the wake is).
#:
#: **That claim has to be measured, not asserted (Tier 66).**  The first choice
#: was (38, 38, 42), a near-black grey, and asserting only that the colour map
#: never produces it EXACTLY passed while the intent failed: the map's lowest
#: colour is (0, 0, 89), so the car and the slow air were both dark and the eye
#: read them as one thing.  `car_colour_margin` measures the real quantity --
#: the distance from this colour to the nearest colour the map can make -- and
#: it is 79.1 here against 56.8 for that first choice, 6.2 for a slate grey and
#: 3.6 for a mid grey, which would have been invisible as an object.
CAR_RGB = (92, 34, 44)

#: The smallest margin `car_colour_margin` may return before the car stops
#: reading as an object.  Chosen above the 56.8 that was measurably too close.
CAR_RGB_MIN_MARGIN = 60.0


def _lut(name: str = "speed") -> np.ndarray:
    """A 256-entry colour table.  Perceptually ordered light-to-dark is avoided:
    the page is dark, so low values must be dark and high values bright."""
    t = np.linspace(0.0, 1.0, 256)
    if name == "signed":
        r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    else:
        r = np.clip(1.6 * t - 0.2, 0, 1)
        g = np.clip(1.4 * t ** 1.2, 0, 1)
        b = np.clip(0.35 + 0.9 * t ** 2, 0, 1)
    return (np.stack([r, g, b], axis=1) * 255.0).astype(np.uint8)


def car_colour_margin(car_rgb: tuple[int, int, int] = CAR_RGB,
                      lut: str = "speed") -> float:
    """How far the car's colour is from the nearest colour the field can make.

    The quantity the overlay actually needs.  "The map never produces this
    colour exactly" is satisfied by any colour at all that is not one of 256
    entries, and says nothing about whether a viewer can tell the car from the
    air; this says how much room there is between them.
    """
    table = _lut(lut).astype(float)
    return float(np.min(np.linalg.norm(table - np.asarray(car_rgb, dtype=float), axis=1)))


def field_png(a: np.ndarray, lo: float, hi: float, *, lut: str = "speed",
              car_rgb: tuple[int, int, int] = CAR_RGB) -> bytes:
    """Colour-map an overlay field and encode it, with the car drawn as the car.

    Row 0 of the array is y = 0 and an image's row 0 is the top, so the array is
    flipped here, exactly as `demo_frontwing.engine.field_png` does it.
    """
    from PIL import Image

    a = np.asarray(a, dtype=float)
    if hi <= lo:
        raise ValueError(f"need hi > lo, got lo={lo} hi={hi}")
    bad = ~np.isfinite(a)
    t = np.clip((np.where(bad, lo, a) - lo) / (hi - lo), 0.0, 1.0)
    rgb = _lut(lut)[(t * 255.0).astype(np.uint8)]
    rgb[bad] = np.asarray(car_rgb, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb[::-1], mode="RGB").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# the cache (W287), and why it validates instead of trusting its key
# ---------------------------------------------------------------------------


def raster_cache_key(ov, nx: int, ny: int, extent, geometry: dict | None = None) -> str:
    """A hash of everything the operator depends on -- `RASTER_CACHE_NOTE`.

    Hashed over DEFINITIONS: the car's fingerprint (itself a hash of the built
    car rather than of segments computed from it, W250), the grid settings, the
    integers.  No derived float array goes in, because `cos` and `atan2` differ
    in the last bit between this machine and glibc and a key that moved with
    them would miss on the very cache it had just written.
    """
    import hashlib
    import json as _json

    from . import car_solids as CS
    from . import racelab as RL

    payload = {
        "version": RASTER_CACHE_VERSION,
        "geometry": RL.geometry_fingerprint(geometry=geometry),
        "grid": {k: CS.GRID[k] for k in sorted(CS.GRID)},
        "hole_margin": float(getattr(ov, "hole_margin", float("nan"))),
        "width": int(getattr(ov, "width", -1)),
        "n_unknowns": int(ov.n_unknowns),
        "grids": [g.name for g in ov.grids],
        "raster": [int(nx), int(ny), [float(v) for v in extent]],
    }
    blob = _json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def save_raster(raster: "CompositeRaster", path: str, key: str) -> str:
    """Write a built operator beside its key.

    `np.savez` appends ``.npz`` to whatever name it is given, so the temporary
    file is named WITH the suffix and `os.replace` is given that same name --
    otherwise the replace raises `FileNotFoundError` and a retry wrapper that
    only catches `PermissionError` does not see it.
    """
    import os

    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp.npz"
    M = raster.M.tocsr()
    np.savez(tmp, key=np.array(key), data=M.data, indices=M.indices,
             indptr=M.indptr, shape=np.asarray(M.shape, dtype=np.int64),
             mask=raster.mask, source=raster.source,
             nx=np.array(raster.nx), ny=np.array(raster.ny),
             extent=np.asarray(raster.extent, dtype=float),
             grid_names=np.array(raster.grid_names, dtype=object), allow_pickle=True)
    os.replace(tmp, path)
    return path


def load_raster(ov, path: str, key: str, *, validate: bool = True
                ) -> tuple["CompositeRaster | None", str]:
    """Rebuild a `CompositeRaster` from disk, or say why not.

    Returns ``(raster, why)``; ``why`` is ``"hit"``, or the reason it was
    rejected.

    **It validates rather than trusting the key.**  A key describes the inputs
    someone remembered to hash; a cache that is wrong because of an input nobody
    thought of is exactly the silent-wrongness this package refuses.  So a loaded
    operator is put through the tier's own positive control -- a linear field
    laid on the LIVE composite must come back through it at rounding -- and its
    mask is checked against the live composite's hole count.  Both are
    milliseconds against a 63 s rebuild, so validating always is free.
    """
    import os

    if not os.path.isfile(path):
        return None, "no cache file"
    try:
        z = np.load(path, allow_pickle=True)
        if str(z["key"]) != key:
            return None, "key differs (the geometry or the raster changed)"
        import scipy.sparse as sp

        M = sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                          shape=tuple(int(v) for v in z["shape"]))
        if M.shape[1] != ov.n_unknowns:
            return None, f"operator has {M.shape[1]} columns, composite has {ov.n_unknowns}"
        r = CompositeRaster.__new__(CompositeRaster)
        r.ov = ov
        r.nx = int(z["nx"])
        r.ny = int(z["ny"])
        r.extent = tuple(float(v) for v in z["extent"])
        x0, x1, y0, y1 = r.extent
        r.xs = np.linspace(x0, x1, r.nx)
        r.ys = np.linspace(y0, y1, r.ny)
        PX, PY = np.meshgrid(r.xs, r.ys)
        r.px, r.py = PX.ravel(), PY.ravel()
        r.M = M
        r.mask = np.asarray(z["mask"], dtype=bool)
        r.source = np.asarray(z["source"], dtype=np.int64)
        r.grid_names = [str(s) for s in z["grid_names"]]
        r.build_s = 0.0
    except Exception as exc:                                     # pragma: no cover
        return None, f"unreadable: {type(exc).__name__}: {exc}"
    if validate:
        # (0) a degenerate mask first, because a raster that draws nothing would
        # otherwise reach the control with an empty selection
        drawn = int((~r.mask).sum())
        if drawn == 0:
            return None, "every pixel is masked"
        if int(r.mask.sum()) == 0:
            return None, "nothing is masked; this composite has no bodies"
        # (a) the operator itself, against the LIVE composite.  `gather` lays a
        # plane on the live nodes, so if the geometry moved, M's columns point
        # at different nodes and the plane comes back bent.  This also catches a
        # mask that is too SMALL: an unmasked pixel whose row is empty returns
        # 0, which is not the plane.
        c = linear_control(r)
        if not (c["linear_max_err"] < 1e-10):
            return None, f"failed the linear control at {c['linear_max_err']:.3e}"
        # (b) a mask that is too LARGE hides car and (a) cannot see it, because
        # it never reads a masked pixel.  So a sample of masked pixels is put
        # back through the donor search: every one must still be unreachable.
        # About 1 ms a point, so 256 of them against a 63 s rebuild.
        idx = np.flatnonzero(r.mask.ravel())
        take = idx[:: max(1, idx.size // 256)][:256]
        _M, missing, _s = CU.probe_matrix_masked(ov, r.px[take], r.py[take])
        if not bool(np.all(missing)):
            n = int((~missing).sum())
            return None, f"{n} of {take.size} sampled masked pixels are reachable now"
    return r, "hit"


def cached_raster(ov, nx: int = NX, ny: int = NY, extent=None, *, path: str | None = None,
                  geometry: dict | None = None, validate: bool = True
                  ) -> tuple["CompositeRaster", dict]:
    """The operator, from disk if it is there and sound, else built and written.

    Closes **W287**: the operator is a function of the geometry and the raster
    alone, costs 63 s to build at 320 x 180 and 0.0002 s to apply, so a demo that
    rebuilt it at every start would pay a minute for nothing.
    """
    import os
    import time as _time

    bg = ov.bg
    if extent is None:
        extent = (float(bg.X.min()), float(bg.X.max()),
                  float(bg.Y.min()), float(bg.Y.max()))
    key = raster_cache_key(ov, nx, ny, extent, geometry=geometry)
    if path is None:
        path = os.path.join("out", "cache", f"raster_{nx}x{ny}_{key}.npz")
    t0 = _time.perf_counter()
    r, why = load_raster(ov, path, key, validate=validate)
    if r is not None:
        return r, {"hit": True, "why": why, "path": path, "key": key,
                   "load_s": _time.perf_counter() - t0, "build_s": None}
    t0 = _time.perf_counter()
    r = CompositeRaster(ov, nx, ny, extent)
    build_s = _time.perf_counter() - t0
    try:
        save_raster(r, path, key)
        wrote = True
    except Exception as exc:                                     # pragma: no cover
        wrote = f"{type(exc).__name__}: {exc}"
    return r, {"hit": False, "why": why, "path": path, "key": key,
               "load_s": None, "build_s": build_s, "wrote": wrote}
