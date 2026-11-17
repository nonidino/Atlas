"""Parameters, formulas and generators, resolved into the numbers each feature holds.

Before the tree is evaluated (`geometry_eval`), its parameters are given their values
(a formula after the parameters it reads; `formula.resolve_parameters`), every field a
feature binds to a formula is set from it (``"w": "W"``, ``"cx": "W - 0.15"``), and
every scatter places its inclusions from its seed.  The numbers are written into the
features, so a saved file holds both the drivers and what they drove: change W and
every shape bound to it moves; open the file elsewhere and the shapes are where they
were, before anything is evaluated.

**A scatter is reproducible from its numbers**: its count, its radius range, its gap,
its box and its seed go through NumPy's PCG64 generator in a fixed order, so the same
seed gives the same inclusions, byte for byte in the case file (the plan's P2
done-when).  Inclusions that cannot be placed after a bounded number of tries are left
out, and the feature's notes say how many were placed.
"""

from __future__ import annotations

import math

import numpy as np

from . import formula
from .document import Feature, Geometry

#: fields a binding may drive, by kind
BINDABLE = {
    "rect": ("x", "y", "w", "h", "r"),
    "ellipse": ("cx", "cy", "rx", "ry"),
    "channel": ("w",),
    "scatter": ("x", "y", "w", "h"),
}
SCATTER_DEFAULTS = {"count": 8, "rmin": 0.02, "rmax": 0.05, "gap": 0.01, "seed": 1,
                    "shape": "circles", "aspect": 8.0}
#: tries per inclusion before the scatter gives up on it
TRIES = 400


class ResolveError(ValueError):
    """A parameter, binding or generator that cannot be resolved; the message says why."""


def scatter_items(f: Feature) -> list[tuple[float, ...]]:
    """The inclusions a scatter's numbers place in its box, a gap apart: circles as
    (cx, cy, r); channels as (cx, cy, r, angle), a strip of length 2r and width
    2r / aspect.  ``shape`` is circles, channels or both (each inclusion drawn as one
    or the other, half and half).  A channel keeps its gap from the circle around it.
    The draws come from one PCG64 stream in a fixed order, so a seed is a layout."""
    p = {**SCATTER_DEFAULTS, **f.params}
    try:
        count = int(p["count"])
        seed = int(p["seed"])
        rmin, rmax, gap = float(p["rmin"]), float(p["rmax"]), float(p["gap"])
        aspect = float(p.get("aspect", 8.0))
    except (TypeError, ValueError):
        raise ResolveError(f"{f.label}: count, seed, rmin, rmax and gap are numbers") from None
    shape = str(p.get("shape", "circles"))
    if shape not in ("circles", "channels", "both"):
        raise ResolveError(f"{f.label}: a scatter's shape is circles, channels or both")
    if not (0 < rmin <= rmax) or gap < 0 or not 0 <= count <= 500 or aspect < 1:
        raise ResolveError(f"{f.label}: 0 < rmin <= rmax, gap >= 0, aspect >= 1, at most "
                           f"500 inclusions")
    rng = np.random.default_rng(seed)
    placed: list[tuple[float, ...]] = []
    for _ in range(count):
        for _t in range(TRIES):
            strip = shape == "channels" or (shape == "both" and rng.random() < 0.5)
            r = float(rng.uniform(rmin, rmax))
            if 2 * r >= min(f.w, f.h):
                continue
            x = float(rng.uniform(f.x + r, f.x + f.w - r))
            y = float(rng.uniform(f.y + r, f.y + f.h - r))
            a = float(rng.uniform(0.0, math.pi)) if strip else None
            if all(math.hypot(x - q[0], y - q[1]) >= r + q[2] + gap for q in placed):
                item = (round(x, 12), round(y, 12), round(r, 12))
                placed.append(item + ((round(a, 12),) if strip else ()))
                break
    return placed


def resolve(geom: Geometry) -> tuple[Geometry, list[str]]:
    """A copy of the tree with every driven number set; and what was noted."""
    notes: list[str] = []
    g = geom.model_copy(deep=True)
    try:
        values = formula.resolve_parameters(g.parameters)
    except formula.FormulaError as exc:
        raise ResolveError(str(exc)) from None
    for p in g.parameters:
        p.value = values[p.name]
    for f in g.features:
        allowed = BINDABLE.get(f.kind, ())
        for field_, expr in f.bind.items():
            if field_ not in allowed:
                raise ResolveError(f"{f.label}: {field_!r} cannot be driven by a formula "
                                   f"(a {f.kind} can drive {', '.join(allowed) or 'nothing'})")
            try:
                tree = formula.parse(expr)
                unknown = formula.names(tree) - set(values) - set(formula.CONSTANTS)
                if unknown:
                    raise formula.FormulaError(f"no parameter called {sorted(unknown)[0]!r}")
                v = formula.value(expr, values)
            except formula.FormulaError as exc:
                raise ResolveError(f"{f.label}, {field_} = {expr}: {exc}") from None
            setattr(f, field_, v)
        if f.kind == "scatter":
            f.items = scatter_items(f)
            want = int({**SCATTER_DEFAULTS, **f.params}["count"])
            if len(f.items) < want:
                notes.append(f"{f.label} placed {len(f.items)} of {want} inclusions; the box "
                             f"is too full for the gap")
    return g, notes


__all__ = ["BINDABLE", "ResolveError", "resolve", "scatter_items"]
