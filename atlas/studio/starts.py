"""The start screen's shapes (plan §3.2): each a small feature tree from a few sizes.

Every starting shape is built from the width and height the start screen asks for,
in the physics' length unit, at the origin (the grid's cells start there):

* **plate**: a rectangle;
* **plate with hole**: a rectangle and a round cut, its diameter and position set
  from the plate;
* **L-bracket**: an L with a filleted inner corner (an arc edge);
* **bend**: a quarter ring, the shape the bend-3 benchmark is;
* **S-channel**: an S-shaped channel of constant width, the centre line's buffer
  with flat ends, so each end is one straight edge (an inlet, an outlet);
* **annulus**: a ring, an outer circle and an inner cut;
* **high contrast**: a plate with a conducting strip and conducting inclusions
  placed by a seeded random draw, so the same seed gives the same layout;
* **blank**: nothing yet.

The materials are the family's own library's: the first for the body, the most
conductive one for the high-contrast inclusions.
"""

from __future__ import annotations

import math

import numpy as np

from atlas.workbench import registry

from .document import Feature

#: what the start screen shows: key, label, hover text (the mockup's)
CATALOG = [
    {"key": "plate", "label": "Plate", "tip": "A rectangle with every size a named parameter"},
    {"key": "plate-hole", "label": "Plate with hole",
     "tip": "Plate with a round hole; diameter and position are parameters"},
    {"key": "l-bracket", "label": "L-bracket", "tip": "L-shaped bracket with an inner fillet"},
    {"key": "bend", "label": "Bend", "tip": "Quarter-ring bend; the bend-3 benchmark is one"},
    {"key": "s-channel", "label": "S-channel", "tip": "S-shaped channel of constant width"},
    {"key": "annulus", "label": "Annulus", "tip": "Ring: outer and inner radius"},
    {"key": "high-contrast", "label": "High contrast",
     "tip": "Plate with random high-contrast inclusions and channels from a seed: the Stage 0b "
            "family"},
    {"key": "blank", "label": "Blank", "tip": "An empty canvas"},
]

#: the chart operator's three held-out problems, opened as the workbench's examples
#: of the same name (an editable copy; the benchmark itself is never changed)
BENCHMARKS = [
    {"key": "bend-3", "label": "bend-3", "meta": "3 pieces",
     "tip": "Held out of training; read-only shape, editable copy"},
    {"key": "s-channel", "label": "s-channel", "meta": "curved",
     "tip": "Held out of training; read-only shape, editable copy"},
    {"key": "insert-round", "label": "insert-round", "meta": "2 materials",
     "tip": "Held out of training; read-only shape, editable copy"},
]


def _num(params: dict, key: str, default: float) -> float:
    v = params.get(key, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{key} is not a number") from None
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{key} must be a positive number")
    return v


def _best_conductor(family: str) -> str | None:
    lib = registry.family(family).material_library()
    keyed = [(props.get("k", props.get("sigma", 0.0)), m) for m, props in lib.items()]
    return max(keyed)[1] if keyed else None


def build(shape: str, params: dict, material: str | None, family: str) -> list[Feature]:
    W = _num(params, "W", 0.6)
    H = _num(params, "H", 0.3)
    if params.get("unit") == "mm":
        W, H = W / 1000.0, H / 1000.0
    body = dict(material=material)
    if shape == "blank":
        return []
    if shape == "plate":
        return [Feature(id="f1", kind="rect", role="add", name="Plate", x=0, y=0, w=W, h=H,
                        **body)]
    if shape == "plate-hole":
        d = 0.35 * H
        return [Feature(id="f1", kind="rect", role="add", name="Plate", x=0, y=0, w=W, h=H,
                        **body),
                Feature(id="f2", kind="ellipse", role="cut", name="Hole", cx=0.65 * W,
                        cy=0.5 * H, rx=d / 2, ry=d / 2)]
    if shape == "l-bracket":
        t = 0.35 * min(W, H)
        r = 0.4 * t
        b = math.tan(math.pi / 8)
        pts = [(0, 0), (W, 0), (W, t), (t + r, t), (t, t + r), (t, H), (0, H)]
        edges = ["line", "line", "line", "arc", "line", "line", "line"]
        bulge = [0, 0, 0, -b, 0, 0, 0]
        return [Feature(id="f1", kind="path", role="add", name="Bracket", points=pts,
                        edges=edges, bulge=bulge, **body)]
    if shape == "bend":
        ro = min(W, H)
        ri = 0.5 * ro
        b = math.tan(math.pi / 8)
        pts = [(ri, 0), (ro, 0), (0, ro), (0, ri)]
        return [Feature(id="f1", kind="path", role="add", name="Bend", points=pts,
                        edges=["line", "arc", "line", "arc"], bulge=[0, b, 0, -b], **body)]
    if shape == "s-channel":
        # the centre line: level, an S of smoothstep form, level again; each bank
        # its points offset by half the width along the normal, joined by the
        # engine's splines, and each end one straight edge (an inlet, an outlet)
        w = 0.3 * H
        t = np.array([0.0, 0.2, 0.32, 0.44, 0.56, 0.68, 0.8, 1.0])
        s = np.clip((t - 0.2) / 0.6, 0.0, 1.0)
        xs = W * t
        ys = w / 2 + (H - w) * (s * s * (3 - 2 * s))
        dy = (H - w) * 6 * s * (1 - s) / 0.6 / W           # the slope dy/dx
        nx_, ny_ = -dy / np.hypot(1, dy), 1 / np.hypot(1, dy)
        lower = [(float(x - w / 2 * a), float(y - w / 2 * b)) for x, y, a, b in zip(xs, ys, nx_, ny_)]
        upper = [(float(x + w / 2 * a), float(y + w / 2 * b)) for x, y, a, b in zip(xs, ys, nx_, ny_)]
        lower[0], lower[-1] = (0.0, lower[0][1]), (W, lower[-1][1])
        upper[0], upper[-1] = (0.0, upper[0][1]), (W, upper[-1][1])
        pts = lower + upper[::-1]
        m = len(lower)
        # the level runs at each end are straight; the S between them is spline
        bank = ["line"] + ["spline"] * (m - 3) + ["line"]
        edges = bank + ["line"] + bank + ["line"]
        return [Feature(id="f1", kind="path", role="add", name="Channel", points=pts,
                        edges=edges, **body)]
    if shape == "annulus":
        R = 0.5 * min(W, H)
        return [Feature(id="f1", kind="ellipse", role="add", name="Ring", cx=R, cy=R, rx=R,
                        ry=R, **body),
                Feature(id="f2", kind="ellipse", role="cut", name="Bore", cx=R, cy=R,
                        rx=0.45 * R, ry=0.45 * R)]
    if shape == "high-contrast":
        cond = _best_conductor(family)
        seed = int(params.get("seed", 1))
        count = int(params.get("count", 8))
        rng = np.random.default_rng(seed)
        feats = [Feature(id="f1", kind="rect", role="add", name="Plate", x=0, y=0, w=W, h=H,
                         **body)]
        w = max(0.04 * H, 1e-9)
        feats.append(Feature(id="f2", kind="rect", role="region", name="Strip", material=cond,
                             x=0, y=0.55 * H, w=W, h=w))
        placed: list[tuple[float, float, float]] = []
        tries = 0
        while len(placed) < count and tries < 2000:
            tries += 1
            r = H * rng.uniform(0.03, 0.06)
            x, y = rng.uniform(r * 1.5, W - r * 1.5), rng.uniform(r * 1.5, H - r * 1.5)
            gap = 0.02 * H
            if abs(y - (0.55 * H + w / 2)) < r + w / 2 + gap:
                continue
            if all(math.hypot(x - a, y - b) > r + c + gap for a, b, c in placed):
                placed.append((x, y, r))
        for k, (x, y, r) in enumerate(placed):
            feats.append(Feature(id=f"f{3 + k}", kind="ellipse", role="region",
                                 name=f"Inclusion {k + 1}", material=cond, cx=x, cy=y, rx=r,
                                 ry=r))
        return feats
    raise ValueError(f"no starting shape {shape!r}")


__all__ = ["BENCHMARKS", "CATALOG", "build"]
