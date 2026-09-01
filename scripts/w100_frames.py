"""Field snapshots for the scaling-ladder viewer -- and for LOOKING at N=24.

`case-study-wake-array-atlas-0.1`'s single most transferable finding is that **a
rendering of the state is a measurement instrument**: W98 was smooth, bounded,
physically shaped, in the right units, passed every control, and was found by
watching an animation beside the referent's.  A ladder whose whole output is an
exponent is exactly the kind of result that can be wrong in that way, so this
runs before any number from it is believed.

Two outputs:

  * ``out/w100/frames.json`` -- quantized uint8 snapshots for the shareable
    viewer, three runs per rung (Poseidon, WindowNS composed, WindowNS monolith),
    downsampled so the whole file stays a couple of megabytes.
  * ``out/w100/look_<rung>.png`` -- a full-resolution three-panel still of the
    final state plus the two difference fields, which is the one to actually look
    at.  The differences are the point: a composed field that looks right beside
    a monolith that looks right can still differ somewhere structured, and the
    difference panel is where W98's class shows up as a wake with no cause.

    python scripts/w100_frames.py --rungs 2,6,24
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atlas.cases import scaling_ladder as sl                         # noqa: E402
from atlas.cases import wake_array as wa                             # noqa: E402
from w100_scaling_ladder import (                                    # noqa: E402
    OUT,
    _forcing,
    march,
    march_monolith,
)

#: The colour range the viewer maps onto.  FIXED rather than per-frame and
#: rather than per-rung, so a frame's brightness means the same thing at every
#: size -- which is the whole point when the question is whether the picture
#: changes with N.
U_LO, U_HI = 0.05, 1.60

#: Every frame's (min, max), so the ramp is reported against the WHOLE run.
SEEN: list = []


def encode(field: np.ndarray) -> str:
    SEEN.append((float(np.min(field)), float(np.max(field))))
    q = np.clip((field - U_LO) / (U_HI - U_LO), 0.0, 1.0)
    return base64.b64encode((q * 255.0 + 0.5).astype(np.uint8).tobytes()).decode()


def stride_for(r, target=180):
    """Downsample so every rung lands near the same pixel size."""
    return max(1, int(round(max(r.tiling.nx, r.tiling.ny) / target)))


def run_rung(r, steps, snap_every, stride):
    rotors = {x.rotor_id for x in r.tiling.rotors}
    out = {"rung": r.label, "n_col": r.n_col, "n_row": r.n_row,
           "n_windows": r.n_windows, "n_overlaps": r.n_overlaps,
           "n_seams": r.n_seams, "stride": stride,
           "nx": r.tiling.nx, "ny": r.tiling.ny,
           "domain_D": [r.tiling.nx * wa.DX, r.tiling.ny * wa.DX],
           "windows": [{"name": n, "ox": o[0], "oy": o[1]}
                       for n, o in zip(r.tiling.names, r.tiling.offsets)],
           "rotors": [{"id": x.rotor_id, "x": x.x_plane, "y": x.y_centre}
                      for x in r.tiling.rotors],
           "runs": {}}
    finals = {}
    for label, fn in (("poseidon", lambda s: march(r, rotors, steps, "poseidon",
                                                   snapshots=s,
                                                   snap_every=snap_every)),
                      ("reference", lambda s: march(r, rotors, steps, "reference",
                                                    snapshots=s,
                                                    snap_every=snap_every))):
        snaps: list = []
        print(f"  {r.label} {label} ...", flush=True)
        u, v, _h, _p = fn(snaps)
        finals[label] = (u, v)
        out["runs"][label] = {
            "frames": [encode(su[::stride, ::stride]) for su, _sv in snaps],
            "fw": snaps[0][0][::stride, ::stride].shape[1],
            "fh": snaps[0][0][::stride, ::stride].shape[0],
        }
    # the monolith, snapshotted the same way -- the referent to look BESIDE
    print(f"  {r.label} monolith ...", flush=True)
    mono_snaps = []
    u = np.ones(r.shape)
    v = np.zeros(r.shape)
    from atlas.cases.scaling_ladder import reference_monolith
    mono = reference_monolith(r.tiling.nx, r.tiling.ny, wa.NU_REF)
    for step in range(steps):
        if step % snap_every == 0:
            mono_snaps.append(u.copy())
        fx, _rec = _forcing(r, u, rotors)
        uu, vv = mono.step_batch(u[None], v[None], wa.MACRO_DT, bc0=None,
                                 force=(fx[None], np.zeros_like(fx)[None]))
        u, v = uu[0], vv[0]
        u[:, :8] = wa.U_INF
        v[:, :8] = 0.0
        u[:8, :] = wa.U_INF
        v[:8, :] = 0.0
        u[-8:, :] = wa.U_INF
        v[-8:, :] = 0.0
    finals["monolith"] = (u, v)
    out["runs"]["monolith"] = {
        "frames": [encode(f[::stride, ::stride]) for f in mono_snaps],
        "fw": mono_snaps[0][::stride, ::stride].shape[1],
        "fh": mono_snaps[0][::stride, ::stride].shape[0],
    }
    return out, finals


def still(r, finals, path):
    """A full-resolution still: three fields and two difference fields.

    The differences are the instrument.  Both composed fields and the monolith
    look like a wind farm; whether the composed one is WRONG somewhere with a
    structure -- a deficit upstream of a turbine, a discontinuity on a window
    edge, a wake that stops at a seam -- is only visible in the difference, and
    that is the shape W98 had.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [0, r.tiling.nx * wa.DX, 0, r.tiling.ny * wa.DX]
    mono = finals["monolith"][0]
    panels = [
        ("WindowNS monolith (the referent E)", mono, dict(vmin=0.2, vmax=1.5,
                                                          cmap="viridis")),
        ("WindowNS composed", finals["reference"][0], dict(vmin=0.2, vmax=1.5,
                                                           cmap="viridis")),
        ("Poseidon-T composed", finals["poseidon"][0], dict(vmin=0.2, vmax=1.5,
                                                            cmap="viridis")),
        ("composed - monolith  (WindowNS)", finals["reference"][0] - mono,
         dict(vmin=-0.25, vmax=0.25, cmap="RdBu_r")),
        ("composed - monolith  (Poseidon-T)", finals["poseidon"][0] - mono,
         dict(vmin=-0.25, vmax=0.25, cmap="RdBu_r")),
    ]
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(11, 2.6 * len(panels)), constrained_layout=True)
    for ax, (title, field, kw) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(field, origin="lower", extent=ext, aspect="equal", **kw)
        for ox, oy in r.tiling.offsets:
            ax.add_patch(plt.Rectangle((ox * wa.DX, oy * wa.DX),
                                       wa.N * wa.DX, wa.N * wa.DX, fill=False,
                                       ec="w", lw=0.4, alpha=0.35))
        for rot in r.tiling.rotors:
            ax.plot([rot.x_plane, rot.x_plane],
                    [rot.y_centre - 0.5, rot.y_centre + 0.5], "k-", lw=2.2)
            ax.plot([rot.x_plane, rot.x_plane],
                    [rot.y_centre - 0.5, rot.y_centre + 0.5], "w-", lw=1.0)
        ax.set_title(f"{title}   [{r.label}, {r.n_col}x{r.n_row}, "
                     f"{r.n_overlaps} overlaps]", fontsize=10, loc="left")
        ax.set_xlabel("x / D", fontsize=8)
        ax.set_ylabel("y / D", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--snap-every", type=int, default=4)
    ap.add_argument("--rungs", default="2,6,24")
    ap.add_argument("--stills-only", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)

    want = {int(x) for x in args.rungs.split(",")}
    rungs = [r for r in sl.ladder() if r.n_windows in want]

    out = {"u_range": [U_LO, U_HI], "snap_every": args.snap_every,
           "n_steps": args.steps, "dt": wa.MACRO_DT, "dx": wa.DX,
           "window_cells": wa.N, "halo": wa.HALO, "stride_cells": wa.STRIDE,
           "rungs": []}
    for r in rungs:
        row, finals = run_rung(r, args.steps, args.snap_every, stride_for(r))
        out["rungs"].append(row)
        still(r, finals, os.path.join(OUT, f"look_{r.label}.png"))
        if not args.stills_only:
            path = os.path.join(OUT, "frames.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(out, fh)
            print(f"  frames -> {path} "
                  f"({os.path.getsize(path) / 1e6:.2f} MB)", flush=True)

    lo = min(a for a, _ in SEEN)
    hi = max(b for _, b in SEEN)
    n_clip = sum(1 for a, b in SEEN if a < U_LO or b > U_HI)
    print(f"u over every frame of every run: [{lo:.4f}, {hi:.4f}]  "
          f"ramp [{U_LO}, {U_HI}]  uses {(hi - lo) / (U_HI - U_LO):.1%} of it")
    if n_clip:
        print(f"  !! {n_clip} frame(s) CLIPPED -- widen the ramp, or the picture "
              f"understates the wake")
    return 0


if __name__ == "__main__":
    sys.exit(main())
