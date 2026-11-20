"""The coolant plan's evaluation layouts: cold plates drawn as the studio draws them, kept
apart from the development layouts ([[chart-operator-stage-0b-and-1-plan]] section 6).

**They are for the one evaluation only.**  They are drawn and hashed in Phase 3, before
any network is trained, and only their classical floors are measured
(`scripts/cot_phase3_eval_layouts.py`): the classical 16-mode expert against the
full-domain solve on the same mesh.  No learned arm, no TL0 and no other coarse arm runs on
them before the one evaluation, so nothing chosen during the plan was chosen on them.

They are drawn with the development layouts' tools (`cot_layouts.build`: an aluminium
plate 16 mm wide, 2 mm water channels drawn with the Channel tool, heated devices of
1e7 W/m^3 drawn as rectangles, water entering at 300 K through Danckwerts' inlet on the
left and leaving on the right, every other edge insulated) but differ from them in what
a network would have to carry:

- ``eval-two-devices``: one straight channel between two devices, one under the plate's
  lower left and one over its upper right, so heat enters the coolant from both walls;
- ``eval-three-channels``: three parallel channels over one device, a larger plate with
  more pieces than any development layout;
- ``eval-arch``: one channel rising in an arch away from a device under its middle, so
  the flow turns twice and leaves the hot zone (the development layouts' winding channel
  is an S).

Each comes at the development tiers' two speeds, calibrated once (`cot_layouts.calibrate`)
and fixed here, so a layout is the same bytes every time it is built.
"""

from __future__ import annotations

import math

from . import cot_layouts as CL
from . import document as D

NAMES = ("eval-two-devices", "eval-three-channels", "eval-arch")
TIERS = CL.TIERS
#: the cut each layout is judged on: columns and, per tier, the grid per piece
CUT = {"eval-two-devices": {"cols": 8, "n": {"slow": 32, "fast": 128}},
       "eval-three-channels": {"cols": 8, "n": {"slow": 32, "fast": 64}},
       "eval-arch": {"cols": 8, "n": {"slow": 32, "fast": 128}}}
#: the mean coolant speeds, m/s, calibrated on 2026-11-19 (`calibrate`) so the largest
#: piece Peclet number is near 20 (slow) and 180 at n = 128 or 110 at n = 64 (fast)
SPEEDS = {"eval-two-devices": {"slow": 0.000958, "fast": 0.00861},
          "eval-three-channels": {"slow": 0.000958, "fast": 0.00526},
          "eval-arch": {"slow": 0.000931, "fast": 0.00837}}


def geometry(name: str) -> tuple[D.Geometry, float, list]:
    """The drawing, the plate's height, and each channel's centre height at its ends."""
    if name == "eval-two-devices":
        H = 0.010
        chans = [("f2", lambda s: 0.005)]
        devices = [(0.002, 0.0, 0.004, 0.002), (0.009, 0.008, 0.004, 0.002)]
    elif name == "eval-three-channels":
        H = 0.014
        chans = [("f2", lambda s: 0.0035), ("f3", lambda s: 0.0070), ("f4", lambda s: 0.0105)]
        devices = [(0.005, 0.0, 0.006, 0.0015)]
    elif name == "eval-arch":
        H = 0.012
        chans = [("f2", lambda s: 0.005 + 0.003 * math.sin(math.pi * s) ** 2)]
        devices = [(0.006, 0.0, 0.004, 0.002)]
    else:
        raise ValueError(f"{name!r}: one of {NAMES}")
    feats = [D.Feature(id="f1", kind="rect", role="add", name="Plate", x=0, y=0, w=CL.W, h=H,
                       material="aluminium")]
    ends = []
    for fid, fn in chans:
        feats.append(CL._channel(fid, fn))
        ends.append(fn(0.0))
    for k, (x, y, w, h) in enumerate(devices):
        feats.append(D.Feature(id=f"fd{k}" if k else "fd", kind="rect", role="region",
                               name="Device" if not k else f"Device {k + 1}", x=x, y=y, w=w, h=h,
                               material="device"))
    return D.Geometry(features=feats), H, ends


def build(name: str, tier: str = "fast", speed: float | None = None):
    """(spec, geometry) of an evaluation layout; ``speed`` overrides the fixed one."""
    if name not in NAMES:
        raise ValueError(f"{name!r}: one of {NAMES}")
    return CL.build(name, tier, speed, geometry_of=geometry, speeds=SPEEDS)


def cut(spec, name: str, tier: str):
    """The layout's registered cut."""
    return CL.cut(spec, name, tier, cuts=CUT)


def calibrate(name: str, tier: str) -> float:
    return CL.calibrate(name, tier, geometry_of=geometry, cuts=CUT)


__all__ = ["CUT", "NAMES", "SPEEDS", "TIERS", "build", "calibrate", "cut", "geometry"]
