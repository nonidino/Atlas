"""The coolant plan's development layouts (Phase 2): cold plates drawn as the studio
draws them, for the decisive test ([[chart-operator-stage-0b-and-1-plan]] section 6).

Three plates, each an aluminium plate with channels drawn by the Channel tool (a centre
line and a width, `document.Feature` kind ``channel``) and a heated device drawn as a
rectangle, the way a person draws them in the studio; each is a studio case (file 0.7)
that opens there, and the Pieces screen's channel cut (`channel_cut`) cuts it:

- ``dev-block``: a block beside a channel; the channel along the plate's upper part, the
  device under the plate's lower part;
- ``dev-parallel``: two straight channels across the plate, the device below them;
- ``dev-winding``: one channel winding across the plate, an S with level run-ins at the
  edges (the plan's serpentine; a serpentine that doubles back is not cut, `channel_cut`).

The coolant is water, entering at 300 K through a coolant-inlet stretch of the left edge
(Danckwerts' form, `cot`) and leaving through a coolant-outlet stretch of the right; every
other edge is insulated, so the device's heat leaves with the coolant.  Each layout comes
at two speeds, ``slow`` (the largest piece Peclet number near 20, n = 32) and ``fast``
(near 180 at n = 128, or 110 at n = 64 for the larger plate), the speeds calibrated once
and fixed here (`SPEEDS`), so a layout is the same bytes every time it is built.

These are development layouts: Phase 2 and the dataset's checks run on them.  The
evaluation layouts are drawn and hashed at registration and never run before the one
evaluation; the code-built layouts of `transport_cases` are neither.
"""

from __future__ import annotations

import math

import numpy as np

from atlas.workbench.spec import Boundary, example_case

from . import document as D
from . import geometry_eval as GE

NAMES = ("dev-block", "dev-parallel", "dev-winding")
TIERS = ("slow", "fast")
T_IN = 300.0
DX = 0.00025                   # the grid's cells, 0.25 mm: boundaries and regions in cells
W = 0.016                      # every plate is 16 mm wide
WIDTH = 0.002                  # every channel is 2 mm wide
Q_DEVICE = 1.0e7               # W/m^3 under the device's footprint
#: the cut each layout is judged on: columns and, per tier, the grid per piece
CUT = {"dev-block": {"cols": 8, "n": {"slow": 32, "fast": 128}},
       "dev-parallel": {"cols": 8, "n": {"slow": 32, "fast": 64}},
       "dev-winding": {"cols": 8, "n": {"slow": 32, "fast": 128}}}
#: the mean coolant speeds, m/s, calibrated on 2026-11-19 (`calibrate`) so the largest
#: piece Peclet number is near 20 (slow) and 180 at n = 128 or 110 at n = 64 (fast)
SPEEDS = {"dev-block": {"slow": 0.000958, "fast": 0.00861},
          "dev-parallel": {"slow": 0.000958, "fast": 0.00526},
          "dev-winding": {"slow": 0.000909, "fast": 0.00817}}
TARGET_PE = {("slow", 32): 20.0, ("fast", 128): 180.0, ("fast", 64): 110.0}


def _channel(fid: str, y_mid_fn, run: float = 0.001, samples: int = 57) -> D.Feature:
    """A channel across the plate with level run-ins of ``run`` at both edges."""
    xm = np.linspace(run, W - run, samples)
    ym = np.array([y_mid_fn((x - run) / (W - 2 * run)) for x in xm])
    xs = np.concatenate([[0.0], xm, [W]])
    ys = np.concatenate([[ym[0]], ym, [ym[-1]]])
    pts = [(round(float(x), 12), round(float(y), 12)) for x, y in zip(xs, ys)]
    return D.Feature(id=fid, kind="channel", role="region", name=f"Channel {fid}", material="water",
                     w=WIDTH, points=pts, edges=["line"] * (len(pts) - 1))


def geometry(name: str) -> tuple[D.Geometry, float, list]:
    """The drawing, the plate's height, and each channel's centre height at its ends."""
    if name == "dev-block":
        H = 0.008
        chans = [("f2", lambda s: 0.006)]
        device = (0.004, 0.0, 0.008, 0.002)
    elif name == "dev-parallel":
        H = 0.012
        chans = [("f2", lambda s: 0.005), ("f3", lambda s: 0.009)]
        device = (0.004, 0.0, 0.008, 0.002)
    elif name == "dev-winding":
        H = 0.012
        chans = [("f2", lambda s: 0.0065 + 0.002 * math.sin(2 * math.pi * s) * math.sin(math.pi * s))]
        device = (0.004, 0.0, 0.008, 0.0015)
    else:
        raise ValueError(f"{name!r}: one of {NAMES}")
    feats = [D.Feature(id="f1", kind="rect", role="add", name="Plate", x=0, y=0, w=W, h=H,
                       material="aluminium")]
    ends = []
    for fid, fn in chans:
        feats.append(_channel(fid, fn))
        ends.append(fn(0.0))
    x, y, w, h = device
    feats.append(D.Feature(id="fd", kind="rect", role="region", name="Device", x=x, y=y, w=w, h=h,
                           material="device"))
    return D.Geometry(features=feats), H, ends


def build(name: str, tier: str = "fast", speed: float | None = None):
    """(spec, geometry) of a development layout; ``speed`` overrides the fixed one."""
    if tier not in TIERS:
        raise ValueError(f"{tier!r}: one of {TIERS}")
    geom, H, ends = geometry(name)
    spec = example_case("cooled-block")
    spec.domain.dx = DX
    al = dict(spec.materials.get("aluminium") or {"k": 205.0, "rho": 2700.0, "cp": 900.0,
                                                  "flows": 0.0, "heat": 0.0})
    spec.materials["aluminium"] = dict(al, heat=0.0)
    spec.materials["device"] = dict(al, heat=Q_DEVICE)
    spec.materials["water"] = dict(spec.materials["water"])
    spec.regions = []
    applied, res = GE.apply(spec, geom, dx=DX)
    bad = [n_ for n_ in res.notes if n_.get("severity") == "error"]
    if bad:
        raise ValueError(f"{name}: the drawing does not evaluate: {bad[0]['message']}")
    ny = applied.domain.ny
    bounds = []
    for side, kind in (("left", "coolant-inlet"), ("right", "coolant-outlet")):
        stops = []
        for yc in sorted(ends):
            a, b = round((yc - WIDTH / 2) / DX), round((yc + WIDTH / 2) / DX)
            stops.append((a, b))
        cur = 0
        for k, (a, b) in enumerate(stops):
            if a > cur:
                bounds.append(Boundary(id=f"{side}-wall{k}", edge=side, kind="insulated", start=cur, stop=a))
            bounds.append(Boundary(id=f"{side}-{kind}-{k}", edge=side, kind=kind, start=a, stop=b,
                                   value=T_IN if kind == "coolant-inlet" else None))
            cur = b
        if cur < ny:
            bounds.append(Boundary(id=f"{side}-wall-end", edge=side, kind="insulated", start=cur, stop=ny))
    bounds += [Boundary(id="base", edge="bottom", kind="insulated"),
               Boundary(id="lid", edge="top", kind="insulated")]
    applied.boundaries = bounds
    u = speed if speed is not None else SPEEDS[name][tier]
    if u is None:
        raise ValueError(f"{name} {tier}: no fixed speed yet; calibrate first")
    applied.physics.params["u_coolant"] = float(u)
    applied.run.mode = "steady"
    applied.windows = []
    applied.layout = None
    return applied, geom


def cut(spec, name: str, tier: str, cols_factor: int = 1, rows: int = 1):
    """The layout's registered cut (``cols_factor`` and ``rows`` for the finer cut, at the
    grid divided by the same factor, so the mesh is unchanged)."""
    from .pieces import cut_channels
    c = CUT[name]
    n = c["n"][tier] // cols_factor
    return cut_channels(spec, n, cols=c["cols"] * cols_factor, rows=rows)


def calibrate(name: str, tier: str) -> float:
    """The speed whose largest piece Peclet number on the registered cut is the tier's
    target (Peclet numbers are linear in the speed), rounded to three figures."""
    from . import cot
    n = CUT[name]["n"][tier]
    spec, _ = build(name, tier, speed=1e-3)
    pe = max(p.peclet() for p in cot.problems(cut(spec, name, tier), spec))
    u = 1e-3 * TARGET_PE[(tier, n)] / pe
    return float(f"{u:.3g}")


__all__ = ["CUT", "NAMES", "SPEEDS", "TIERS", "build", "calibrate", "cut", "geometry"]
