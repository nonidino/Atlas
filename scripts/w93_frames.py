"""Field snapshots for the wake-array viewer.

Re-marches the array with both experts, saving the streamwise velocity every
``--snap-every`` macro-steps at half resolution, and writes them quantized to
uint8 with the scale stated beside them.  Output `out/w93/frames.json` is the
data the shareable viewer embeds; nothing here measures anything the Tier 18
record quotes, and the marches are the same ones `w93_wake_array.py` runs.

    python scripts/w93_frames.py --steps 60
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.cases import wake_array as wa                             # noqa: E402
from w93_wake_array import OUT, disk_rows, march, power_of, settled  # noqa: E402

#: The colour range the viewer maps onto.  Fixed rather than per-frame, so the
#: two experts' animations are directly comparable and a frame's brightness means
#: the same thing throughout.
U_LO, U_HI = 0.20, 1.50


#: every frame's (min, max), so the ramp is reported against the WHOLE run
#: and not merely against the frames that already fell outside it.
SEEN: list = []


def encode(field: np.ndarray) -> str:
    SEEN.append((float(np.min(field)), float(np.max(field))))
    q = np.clip((field - U_LO) / (U_HI - U_LO), 0.0, 1.0)
    return base64.b64encode((q * 255.0 + 0.5).astype(np.uint8).tobytes()).decode()


def run(kind, steps, snap_every):
    wa.load_reference()                      # registers the private package
    snaps: list = []
    _u, _v, hist, _p = march({"R1", "R2", "R3"}, steps, kind,
                             snapshots=snaps, snap_every=snap_every)
    frames = [encode(u) for u, _ in snaps]
    ny, nx = snaps[0][0].shape
    series = {r.rotor_id: [h[r.rotor_id] for h in hist] for r in wa.ROTORS}
    return {
        "expert": kind, "nx": nx, "ny": ny, "snap_every": snap_every,
        "n_steps": steps, "frames": frames,
        "Ud": series,
        "P": {k: [power_of(x) for x in v] for k, v in series.items()},
        "settled": {r.rotor_id: settled(hist, r.rotor_id)[0] for r in wa.ROTORS},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--snap-every", type=int, default=2)
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)

    out = {
        "u_range": [U_LO, U_HI],
        "geometry": {
            "S_LEN": wa.S_LEN, "dx": wa.DX, "dt": wa.MACRO_DT,
            "NX": wa.NX, "NY": wa.NY, "N": wa.N,
            "halo": wa.HALO, "stride": wa.STRIDE, "ramp": wa.RAMP,
            "rotor_D": wa.ROTOR_D, "rotor_cells": wa.ROTOR_CELLS,
            "windows": [{"name": n, "ox": o[0], "oy": o[1]}
                        for n, o in zip(wa.DEFAULT_TILING.names,
                                        wa.DEFAULT_TILING.offsets)],
            "rotors": [{"id": r.rotor_id, "x": r.x_plane, "y": r.y_centre,
                        "up": r.upstream_window, "down": r.downstream_window}
                       for r in wa.ROTORS],
            "seams": [{"id": c.seam_id, "a": list(c.a), "b": list(c.b)}
                      for c in wa.connections()],
        },
        "runs": {},
    }
    for kind in ("poseidon", "reference"):
        print(f"marching {kind} ...", flush=True)
        out["runs"][kind] = run(kind, args.steps, args.snap_every)
        print(f"  {len(out['runs'][kind]['frames'])} frames at "
              f"{out['runs'][kind]['nx']}x{out['runs'][kind]['ny']}", flush=True)

    path = os.path.join(OUT, "frames.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")
    lo = min(a for a, _ in SEEN)
    hi = max(b for _, b in SEEN)
    n_clip = sum(1 for a, b in SEEN if a < U_LO or b > U_HI)
    print(f"  u over every frame of both experts: [{lo:.4f}, {hi:.4f}]  "
          f"ramp [{U_LO}, {U_HI}]  uses {(hi - lo) / (U_HI - U_LO):.1%} of it")
    if n_clip:
        print(f"  !! {n_clip} frame(s) CLIPPED -- widen the ramp and move the "
              f"neutral stop, or the picture understates the wake")
    return 0


if __name__ == "__main__":
    sys.exit(main())
