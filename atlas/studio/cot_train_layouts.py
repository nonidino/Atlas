"""The coolant plan's training layouts: random cold plates whose coupled classical solves
supply the port traces that weight the training loss ([[chart-operator-stage-0b-and-1-plan]]
sections 5 and 6: "training layouts, generated (random channel networks on a plate)").

They are drawn with the development layouts' tools (`cot_layouts.build`: an aluminium plate
16 mm wide, 2 mm water channels drawn with the Channel tool, devices of 1e7 W/m^3, water in
at 300 K on the left through Danckwerts' inlet, out on the right), from a seed:
- the plate's height, 8 to 14 mm;
- a device under the plate (always) and one over it (two in five), 3 to 6 mm wide;
- one to three channels between them, each straight, an arch or an S, with at least 1 mm of
  aluminium between channels, between a channel and a device, and at the plate's edges;
- a speed tier: slow (Pe 20 at n = 32) or fast (110 at n = 64, 180 at n = 128), the speed
  calibrated on the layout's own cut (`cot_layouts.calibrate`).
A drawing the channel cut refuses is drawn again from the seed's next stream.  None of them
is a development or an evaluation layout, and nothing measured on them chooses a bar.
"""

from __future__ import annotations

import math

import numpy as np

from . import cot_layouts as CL
from . import document as D

TIERS = (("slow", 32), ("fast", 64), ("fast", 128))
SEED_BASE = 20_000_000


def _snap(v: float) -> float:
    """To the layouts' 0.25 mm grid: regions and channel ends sit on cell edges."""
    return round(round(v / CL.DX) * CL.DX, 9)


def _draw(rng):
    H = float(rng.choice([0.008, 0.010, 0.012, 0.014]))
    hd = _snap(float(rng.uniform(0.0015, 0.002)))
    top = bool(rng.uniform() < 0.4)
    ht = _snap(float(rng.uniform(0.0015, 0.002))) if top else 0.0
    lo, hi = hd + 0.001, H - (ht + 0.001 if top else 0.001)
    devices = []
    for y, h in [(0.0, hd)] + ([(H - ht, ht)] if top else []):
        w = _snap(float(rng.uniform(0.003, 0.006)))
        x = _snap(float(rng.uniform(0.001, CL.W - 0.001 - w)))
        devices.append((x, _snap(y), w, h))
    nch = int(rng.integers(1, 4))
    chans = []
    while nch >= 1:
        amps = [float(rng.uniform(0.0005, 0.0015)) for _ in range(nch)]
        need = sum(2 * a for a in amps) + nch * 0.002 + (nch - 1) * 0.001
        if need <= hi - lo:
            break
        nch -= 1
    if nch < 1:
        return None
    slack = (hi - lo) - need
    y = lo + float(rng.uniform(0.0, slack))
    for k in range(nch):
        a = amps[k]
        c = _snap(y + a + 0.001)
        shape = str(rng.choice(["straight", "arch", "s"], p=[0.5, 0.25, 0.25]))
        sign = 1.0 if rng.uniform() < 0.5 else -1.0
        if shape == "straight":
            fn = (lambda s, c=c: c)
        elif shape == "arch":
            base = c - a if sign > 0 else c + a
            fn = (lambda s, b=base, A=sign * 2 * a: b + A * math.sin(math.pi * s) ** 2)
        else:
            fn = (lambda s, c=c, A=sign * a / 0.77: c + A * math.sin(2 * math.pi * s) * math.sin(math.pi * s))
        chans.append((f"f{k + 2}", fn, shape))
        y = c + a + 0.001 + 0.001
    return H, devices, chans


def geometry(seed: int):
    """(drawing, plate height, channel ends, what was drawn) of a seed's layout, before the
    cut is tried."""
    rng = np.random.default_rng([seed, 31])
    while True:
        d = _draw(rng)
        if d is not None:
            break
    H, devices, chans = d
    feats = [D.Feature(id="f1", kind="rect", role="add", name="Plate", x=0, y=0, w=CL.W, h=H, material="aluminium")]
    ends = []
    for fid, fn, _shape in chans:
        feats.append(CL._channel(fid, fn))
        ends.append(fn(0.0))
    for k, (x, y, w, h) in enumerate(devices):
        feats.append(D.Feature(id=f"fd{k}" if k else "fd", kind="rect", role="region",
                               name="Device" if not k else f"Device {k + 1}", x=x, y=y, w=w, h=h, material="device"))
    meta = {"height": H, "devices": devices, "channels": [s for _f, _fn, s in chans]}
    return D.Geometry(features=feats), H, ends, meta


def build(i: int, tries: int = 20):
    """(name, spec, geometry, cut, meta) of training layout i: its seed SEED_BASE + i, redrawn
    from the next stream while the channel cut refuses it."""
    from .pieces import CutError
    tier, n = TIERS[i % len(TIERS)]
    for t in range(tries):
        seed = SEED_BASE + 1000 * i + t
        geom, H, ends, meta = geometry(seed)
        name = f"train-{i:03d}"
        cuts = {name: {"cols": 8, "n": {tier: n}}}
        g = lambda _name, _r=(geom, H, ends): _r                     # noqa: E731
        try:
            u = CL.calibrate(name, tier, geometry_of=g, cuts=cuts)
            spec, geo = CL.build(name, tier, speed=u, geometry_of=g)
            cut = CL.cut(spec, name, tier, cuts=cuts)
        except (CutError, ValueError):
            continue
        meta.update(seed=seed, tier=tier, n=n, u_coolant=u, redrawn=t)
        return name, spec, geo, cut, meta
    raise RuntimeError(f"training layout {i}: no drawing the cut accepts in {tries} tries")


__all__ = ["SEED_BASE", "TIERS", "build", "geometry"]
