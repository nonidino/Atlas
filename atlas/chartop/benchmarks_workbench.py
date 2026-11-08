"""The registration after section 11.14 (`gates_hybrid`): six workbench scenarios as layouts
of four-sided pieces with conforming seams, the new problems on which the fixed rule's
hybrid is judged.

Accepted by the owner on 2026-11-07 ("Accept as drafted",
[[chart-operator-kernel-exact-forms]] section 11.15).  `benchmarks` is hash-pinned, so the
new problems live here.  They reuse its helpers unchanged (`Bench`, `_finish`, `_material`,
`_held_out_bc`, `_arc`, `datum_shift`), and so the same piece problems, data and frames.
They are measured with the classical solver only (`scripts/chartop_register_hybrid.py`).
No learned expert, and not U's form, reads them before the one evaluation.

**The set**: every workbench scenario whose heat problem is plain conduction and that is
not already a benchmark, and one more:
- ``wall-2``: 0.4 m x 0.1 m, 0.24 m of steel then 0.16 m of copper;
- ``heated-strip``, the heat half of the bimetal strip: 0.2 m x 20 mm, 10 mm of steel under
  10 mm of copper;
- ``plate-insert``: a steel plate 0.4 m x 0.24 m with a copper insert 0.12 m x 0.08 m whose
  lower-left corner is at (0.14, 0.08) m;
- ``fast-heat``: the same plate with a copper spreader 0.12 m x 0.08 m against the hot edge,
  in the middle third;
- ``bimetal-arc``, the heat half of the curved bimetal strip: a quarter ring of radii 60 and
  100 mm, steel inside the 80 mm arc and copper outside it;
- ``plate-hole``, an elasticity scenario posed as conduction: a steel plate 0.4 m x 0.2 m
  with an insulated hole of radius 30 mm in its middle.  It is added for a Winslow ring
  away from the ends, which post hoc only insert-round had.
The dimensions are written here, not read from the workbench, so that this file alone
defines the problems; a test checks them against the workbench's scenarios.

**The data**, as `benchmarks`' held-out variants: the hot end at T_HOT + 20 cos(pi s) and the
cold end at T_COLD + 10 cos(2 pi s), ``s`` running from 0 to 1 along each end, both with zero
slope where the end meets an insulated wall.  Every other side is insulated, the hole's
included.  A scenario held at one end only gets its cold end at the far side, as every
benchmark has.  Each scenario gives a steady problem and one backward-Euler step of
``DT[name]`` seconds from 300 K.

**The layout**: pieces as near square as conforming seams allow.  The lines where the
material or the outline changes cut the domain into bands; each band is cut into equal
pieces, their number its aspect rounded (half to even, as Python rounds), as decision 2
cuts a channel.  A band's aspect is its length over the mean width of the rows it crosses.
Rectangles are charted by affine maps, the quarter ring's pieces by the log-polar map, and
the ring around the hole by Winslow: insert-round's plate ring, scaled, the box's
half-width twice the hole's radius and three pieces to a side.

**The step's time step** (`dt_by_rule`): the value in 1, 2, 5 x 10^k nearest, in log, to the
time at which the median piece's Fourier number ``alpha dt / area`` is 1.  `DT` holds the
values the rule gave, so that they are fixed in this file, and a test checks that the
rule still gives them.
"""

from __future__ import annotations

import math

import numpy as np

from . import chart as ch
from .benchmarks import Bench, _alpha, _arc, _finish, _held_out_bc, _material, datum_shift

NAMES = ("wall-2", "heated-strip", "plate-insert", "fast-heat", "bimetal-arc", "plate-hole")
PROBLEMS = ("steady", "step")
#: the step's time step, seconds: `dt_by_rule`'s values, fixed here
DT = {"wall-2": 100.0, "heated-strip": 2.0, "plate-insert": 500.0, "fast-heat": 500.0,
      "bimetal-arc": 10.0, "plate-hole": 100.0}

# the scenarios' dimensions, metres (checked against the workbench by a test)
WALL = {"length": 0.4, "height": 0.1, "steel": 0.24}
STRIP = {"length": 0.2, "layer": 0.01}
PLATE = {"width": 0.4, "height": 0.24}
INSERT = {"x0": 0.14, "y0": 0.08, "width": 0.12, "height": 0.08}
SPREADER = {"x0": 0.0, "y0": 0.08, "width": 0.12, "height": 0.08}
ARC = {"r_in": 0.06, "r_mid": 0.08, "r_out": 0.10}
HOLE = {"width": 0.4, "height": 0.2, "radius": 0.03, "box_over_radius": 2.0, "split": 3}

__all__ = ["NAMES", "PROBLEMS", "DT", "build", "dt_by_rule", "rule_dt", "datum_shift"]


def _rect_chart(x0: float, x1: float, y0: float, y1: float, n: int) -> ch.Chart:
    def rect(xi, eta, x0=x0, x1=x1, y0=y0, y1=y1):
        return x0 + (x1 - x0) * xi, y0 + (y1 - y0) * eta
    return ch.Chart(ch.from_map(rect, n), "affine")


def _band_cuts(lines, width: float) -> list:
    """The cut coordinates: each band between consecutive ``lines`` cut into
    ``round(length / width)`` equal pieces (at least one).  The lines themselves are
    kept exactly, so that two bands meet at the same coordinate."""
    cuts = [float(lines[0])]
    for a, b in zip(lines[:-1], lines[1:]):
        k = max(1, round((b - a) / width))
        cuts += [a + (b - a) * j / k for j in range(1, k)] + [float(b)]
    return cuts


def _blocks(xlines, ylines, n: int, role_of) -> list:
    """Rectangles on the grid cut from ``xlines`` (each column band cut by the mean row
    height) and ``ylines`` (the rows, not cut), row by row: [(chart, material, label)].
    ``role_of(x, y)`` names the material at a piece's centre: (role, material)."""
    xs = _band_cuts(xlines, float(np.mean(np.diff(ylines))))
    out = []
    for r in range(len(ylines) - 1):
        for q in range(len(xs) - 1):
            x0, x1, y0, y1 = xs[q], xs[q + 1], float(ylines[r]), float(ylines[r + 1])
            role, mat = role_of(0.5 * (x0 + x1), 0.5 * (y0 + y1))
            out.append((_rect_chart(x0, x1, y0, y1, n), mat, f"{role} {q},{r}"))
    return out


def _ends(width: float, height: float):
    """The held-out data on a domain whose hot end is x = 0 and cold end x = width."""
    tol = 1e-6 * width
    return _held_out_bc(lambda P: bool(np.all(np.abs(P[:, 0]) < tol)), lambda P: P[:, 1] / height,
                        lambda P: bool(np.all(np.abs(P[:, 0] - width) < tol)),
                        lambda P: P[:, 1] / height)


# ---------------------------------------------------------------------------
# the scenarios: geometry and materials, without the problem
# ---------------------------------------------------------------------------


def _wall(n: int):
    st, cu = _material("steel"), _material("copper")
    L, H, xs = WALL["length"], WALL["height"], WALL["steel"]
    pieces = _blocks([0.0, xs, L], [0.0, H], n,
                     lambda x, y: ("steel", st) if x < xs else ("copper", cu))
    return pieces, _ends(L, H), {"materials": {"steel": st, "copper": cu}}


def _strip(n: int):
    st, cu = _material("steel"), _material("copper")
    L, t = STRIP["length"], STRIP["layer"]
    pieces = _blocks([0.0, L], [0.0, t, 2.0 * t], n,
                     lambda x, y: ("steel", st) if y < t else ("copper", cu))
    return pieces, _ends(L, 2.0 * t), {"materials": {"steel": st, "copper": cu}}


def _plate_with(n: int, box: dict, role: str):
    st, cu = _material("steel"), _material("copper")
    W, H = PLATE["width"], PLATE["height"]
    x0, y0 = box["x0"], box["y0"]
    x1, y1 = x0 + box["width"], y0 + box["height"]
    xlines = sorted({0.0, x0, x1, W})
    ylines = sorted({0.0, y0, y1, H})

    def role_of(x, y):
        return (role, cu) if (x0 < x < x1 and y0 < y < y1) else ("plate", st)
    pieces = _blocks(xlines, ylines, n, role_of)
    return pieces, _ends(W, H), {"materials": {"plate": st, role: cu}}


def _arc_layers(n: int):
    st, cu = _material("steel"), _material("copper")
    r0, rm, r1 = ARC["r_in"], ARC["r_mid"], ARC["r_out"]
    layers = (("steel", st, math.log(r0), math.log(rm)), ("copper", cu, math.log(rm), math.log(r1)))
    n_th = round((math.pi / 2) / float(np.mean([b - a for _, _, a, b in layers])))
    dth = (math.pi / 2) / n_th
    pieces = []
    for b in range(n_th):
        for role, mat, rho0, rho1 in layers:
            dr = rho1 - rho0

            def F(xi, eta, rho0=rho0, dr=dr, b=b):
                rr = np.exp(rho0 + xi * dr)
                th = (b + eta) * dth
                return rr * np.cos(th), rr * np.sin(th)
            pieces.append((ch.Chart(ch.from_map(F, n), "conformal-log-polar", {"modulus": dr / dth}),
                           mat, f"{role} {b}"))
    tol = 1e-6 * r1
    bc = _held_out_bc(lambda P: bool(np.all(np.abs(P[:, 1]) < tol)), lambda P: (P[:, 0] - r0) / (r1 - r0),
                      lambda P: bool(np.all(np.abs(P[:, 0]) < tol)), lambda P: (P[:, 1] - r0) / (r1 - r0))
    return pieces, bc, {"materials": {"steel": st, "copper": cu}, "pieces": (2, n_th)}


def _hole(n: int):
    """insert-round's plate ring around the hole (Winslow; rays at -45 + 90 l / split
    degrees from each side's middle, as `benchmarks.insert_round` cuts it), the strips above
    and below its box cut where the rays meet the box, and the plate's two sides cut into
    rows where the box's corners and the rays meet them."""
    st = _material("steel")
    W, H, R = HOLE["width"], HOLE["height"], HOLE["radius"]
    c = np.array([0.5 * W, 0.5 * H])
    b = HOLE["box_over_radius"] * R
    S = HOLE["split"]
    deg = math.pi / 180.0
    tq = np.tan((-45.0 + 90.0 * np.arange(S + 1) / S) * deg)

    def ray_point(phi_deg: float, k: int, dist: float) -> np.ndarray:
        rr = dist / math.cos((phi_deg - 90.0 * k) * deg)
        return c + rr * np.array([math.cos(phi_deg * deg), math.sin(phi_deg * deg)])

    def on_circle(phi_deg: float) -> np.ndarray:
        return c + R * np.array([math.cos(phi_deg * deg), math.sin(phi_deg * deg)])
    pieces = []
    for k in range(4):
        for l in range(S):
            lo = 90.0 * k - 45.0 + 90.0 * l / S
            hi = 90.0 * k - 45.0 + 90.0 * (l + 1) / S
            pb_lo, pb_hi = ray_point(lo, k, b), ray_point(hi, k, b)
            sides = [ch.line(on_circle(lo), pb_lo), ch.line(pb_lo, pb_hi),
                     ch.line(pb_hi, on_circle(hi)), _arc(c, R, hi * deg, lo * deg)]
            Xw, info = ch.winslow(sides, n)
            pieces.append((ch.Chart(Xw, "winslow-arc-length", info), st, f"plate ring {k},{l}"))
    xs_box = c[0] + b * tq
    for label, y0, y1 in (("strip below", 0.0, c[1] - b), ("strip above", c[1] + b, H)):
        for p in range(S):
            pieces.append((_rect_chart(float(xs_box[p]), float(xs_box[p + 1]), y0, y1, n), st,
                           f"{label} {p}"))
    ys = [0.0] + [float(v) for v in c[1] + b * tq] + [H]
    for label, x0, x1 in (("left side", 0.0, float(c[0] - b)), ("right side", float(c[0] + b), W)):
        for chart, mat, lab in _blocks([x0, x1], ys, n, lambda x, y: (label, st)):
            pieces.append((chart, mat, lab))
    return pieces, _ends(W, H), {"materials": {"plate": st}}


_SCENARIOS = {"wall-2": _wall, "heated-strip": _strip,
              "plate-insert": lambda n: _plate_with(n, INSERT, "insert"),
              "fast-heat": lambda n: _plate_with(n, SPREADER, "spreader"),
              "bimetal-arc": _arc_layers, "plate-hole": _hole}


def build(name: str, n: int = 64, problem: str = "steady") -> Bench:
    """The scenario ``name`` as a `benchmarks.Bench`: ``problem`` "steady" or "step"."""
    if problem not in PROBLEMS:
        raise ValueError(f"problem {problem!r}: one of {PROBLEMS}")
    pieces, bc, meta = _SCENARIOS[name](n)
    probs = [_finish(chart, mat, problem, DT[name], label) for chart, mat, label in pieces]
    return Bench(name, problem, probs, bc, dict(meta, scenario=name, dt=DT[name],
                                               source="benchmarks_workbench"))


def rule_dt(rates) -> float:
    """The 1-2-5 value nearest, in log, to 1 / median(rates), ``rates`` being each piece's
    ``alpha / area``: the time step at which the median piece's Fourier number is 1."""
    target = 1.0 / float(np.median(np.asarray(rates, float)))
    k = math.floor(math.log10(target))
    cands = [m * 10.0 ** e for e in (k - 1, k, k + 1) for m in (1.0, 2.0, 5.0)]
    return min(cands, key=lambda v: abs(math.log10(v) - math.log10(target)))


def dt_by_rule(name: str, n: int = 64) -> float:
    pieces, _, _ = _SCENARIOS[name](n)
    return rule_dt([_alpha(mat) / chart.element_fields()["area"] for chart, mat, _ in pieces])
