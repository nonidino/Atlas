"""The registration after section 11.17 (`gates_squares`): square-first layouts from a
registered generator, the new problems on which rule R2's hybrid is judged.

Accepted by the owner on 2026-11-08 ("Accept as drafted",
[[chart-operator-kernel-exact-forms]] section 11.18).  Each layout is a plate of ``W x H``
squares of side `S_SIDE` with one feature.  `draw` gives its specification from the seed
alone (numpy's PCG64); `build` gives the problem as a `benchmarks.Bench`, with `benchmarks`'
own helpers.  `benchmarks` and `benchmarks_workbench` are pinned, and stay unchanged.

**The draw** (`draw`).  ``W`` from `W_RANGE`, ``H`` from `H_RANGE`, the plate and the
feature in two different materials of `MATERIALS`.  The feature is `FEATURES`'
``(seed - SEED_BASE) mod 5``-th, so that any five consecutive seeds hold every feature; its
place and size are drawn on the grid.
- ``inclusion``: a block of 1-3 x 1-3 squares, at least one column from each end;
- ``spreader``: 2-3 x 1-2 squares against the hot end;
- ``layer``: the top 1 to ``H - 1`` rows, across the whole plate;
- ``hole``: an insulated round hole, and ``round-inclusion``: a disc of the feature's
  material.  Either sits in a box of 2 x 2 squares at least one column from each end, its
  radius a quarter of the box's side.

**The round features** are `benchmarks.insert_round`'s construction at two pieces to a side:
rays at -45, 0 and 45 degrees from the middle of each side.  They meet the box at its corners
and at the middle of each side, so the squares around the box keep their sides.  The plate
ring between the circle and the box is charted by Winslow (8 pieces).  A round inclusion
adds the disc ring (Winslow, 8 pieces) and the disc's core: 2 x 2 squares of a quarter of the
box's side.

**The data**, as `benchmarks_workbench`'s: the hot end at T_HOT + 20 cos(pi s) and the cold
end at T_COLD + 10 cos(2 pi s), ``s`` from 0 to 1 along each end; every other side insulated,
the hole's included.  A steady problem, and one backward-Euler step from 300 K whose time
step is `benchmarks_workbench.rule_dt`'s for the layout (the 1-2-5 value nearest to where
the median piece's Fourier number is 1).

The registered layouts are the first `N_LAYOUTS` seeds from `SEED_BASE` whose classical
floors meet E4 in both frames (``scripts/chartop_register_squares.py``).
"""

from __future__ import annotations

import math

import numpy as np

from . import chart as ch
from .benchmarks import Bench, _alpha, _arc, _finish, _material
from .benchmarks_workbench import _ends, _rect_chart, rule_dt

S_SIDE = 0.04
W_RANGE = (6, 9)
H_RANGE = (3, 5)
FEATURES = ("inclusion", "spreader", "layer", "hole", "round-inclusion")
MATERIALS = ("steel", "copper", "aluminium")
SEED_BASE = 7_000_000
N_LAYOUTS = 8
PROBLEMS = ("steady", "step")
ROUND = ("hole", "round-inclusion")
#: insert-round's ratio: the box's half-width is twice the circle's radius
BOX_OVER_RADIUS = 2.0
RING_SPLIT = 2


def draw(seed: int) -> dict:
    """The layout of ``seed``: plain values only, so that it can be recorded as it is."""
    rng = np.random.default_rng(seed)
    feature = FEATURES[(seed - SEED_BASE) % len(FEATURES)]
    W = int(rng.integers(W_RANGE[0], W_RANGE[1] + 1))
    H = int(rng.integers(H_RANGE[0], H_RANGE[1] + 1))
    plate, feat = (str(m) for m in rng.choice(MATERIALS, size=2, replace=False))
    if feature == "inclusion":
        w = int(rng.integers(1, min(3, W - 2) + 1))
        h = int(rng.integers(1, min(3, H - 1) + 1))
        box = (int(rng.integers(1, W - 1 - w + 1)), int(rng.integers(0, H - h + 1)), w, h)
    elif feature == "spreader":
        w = int(rng.integers(2, 4))
        h = int(rng.integers(1, min(2, H - 1) + 1))
        box = (0, int(rng.integers(0, H - h + 1)), w, h)
    elif feature == "layer":
        k = int(rng.integers(1, H))
        box = (0, H - k, W, k)
    else:
        box = (int(rng.integers(1, W - 3 + 1)), int(rng.integers(0, H - 2 + 1)), 2, 2)
    return {"seed": int(seed), "W": W, "H": H, "plate": plate, "feature": feature,
            "feature_material": None if feature == "hole" else feat, "box": list(box)}


def _ring_pieces(spec: dict, n: int, plate: dict, feat: dict | None) -> list:
    s = S_SIDE
    x0, y0 = spec["box"][0], spec["box"][1]
    c = np.array([(x0 + 1) * s, (y0 + 1) * s])
    b = s                                    # the box's half-width
    R = b / BOX_OVER_RADIUS
    a = R / 2.0                              # the disc core's half-width, as insert_round
    S = RING_SPLIT
    deg = math.pi / 180.0
    tq = np.tan((-45.0 + 90.0 * np.arange(S + 1) / S) * deg)

    def ray_point(phi_deg: float, k: int, dist: float) -> np.ndarray:
        rr = dist / math.cos((phi_deg - 90.0 * k) * deg)
        return c + rr * np.array([math.cos(phi_deg * deg), math.sin(phi_deg * deg)])

    def on_circle(phi_deg: float) -> np.ndarray:
        return c + R * np.array([math.cos(phi_deg * deg), math.sin(phi_deg * deg)])
    out = []
    if spec["feature"] == "round-inclusion":
        for q in range(S):
            for p in range(S):
                out.append((_rect_chart(float(c[0] + a * tq[p]), float(c[0] + a * tq[p + 1]),
                                        float(c[1] + a * tq[q]), float(c[1] + a * tq[q + 1]), n),
                            feat, f"disc core {p},{q}"))
    rings = [("plate ring", plate)] + ([("disc ring", feat)] if spec["feature"] == "round-inclusion" else [])
    for ring, mat in rings:
        for k in range(4):
            for l in range(S):
                lo = 90.0 * k - 45.0 + 90.0 * l / S
                hi = 90.0 * k - 45.0 + 90.0 * (l + 1) / S
                if ring == "disc ring":
                    pin_lo, pin_hi = ray_point(lo, k, a), ray_point(hi, k, a)
                    sides = [ch.line(pin_lo, on_circle(lo)), _arc(c, R, lo * deg, hi * deg),
                             ch.line(on_circle(hi), pin_hi), ch.line(pin_hi, pin_lo)]
                else:
                    pb_lo, pb_hi = ray_point(lo, k, b), ray_point(hi, k, b)
                    sides = [ch.line(on_circle(lo), pb_lo), ch.line(pb_lo, pb_hi),
                             ch.line(pb_hi, on_circle(hi)), _arc(c, R, hi * deg, lo * deg)]
                Xw, info = ch.winslow(sides, n)
                out.append((ch.Chart(Xw, "winslow-arc-length", info), mat, f"{ring} {k},{l}"))
    return out


def pieces(spec: dict, n: int) -> list:
    """[(chart, material, label)]: the grid's squares, row by row, then the round feature's
    pieces, if it has one."""
    s = S_SIDE
    W, H = spec["W"], spec["H"]
    x0, y0, w, h = spec["box"]
    plate = _material(spec["plate"])
    feat = _material(spec["feature_material"]) if spec["feature_material"] else None
    rnd = spec["feature"] in ROUND
    out = []
    for j in range(H):
        for i in range(W):
            inside = x0 <= i < x0 + w and y0 <= j < y0 + h
            if rnd and inside:
                continue
            role, mat = (spec["feature"], feat) if inside else ("plate", plate)
            out.append((_rect_chart(i * s, (i + 1) * s, j * s, (j + 1) * s, n), mat, f"{role} {i},{j}"))
    if rnd:
        out += _ring_pieces(spec, n, plate, feat)
    return out


def dt_of(spec: dict, n: int = 64) -> float:
    """The step's time step: `rule_dt` over the layout's pieces."""
    return rule_dt([_alpha(mat) / chart.element_fields()["area"] for chart, mat, _ in pieces(spec, n)])


def build(spec: dict, n: int = 64, problem: str = "steady") -> Bench:
    if problem not in PROBLEMS:
        raise ValueError(f"problem {problem!r}: one of {PROBLEMS}")
    ps = pieces(spec, n)
    dt = rule_dt([_alpha(mat) / chart.element_fields()["area"] for chart, mat, _ in ps])
    probs = [_finish(chart, mat, problem, dt, label) for chart, mat, label in ps]
    bc = _ends(spec["W"] * S_SIDE, spec["H"] * S_SIDE)
    return Bench(f"squares-{spec['seed']}", problem, probs, bc,
                 {"spec": dict(spec), "dt": dt, "source": "benchmarks_squares"})
