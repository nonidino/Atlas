r"""Turn an episode into the compact record a page can draw.

The .npz carries nondimensionalised cell fields per agent plus the rigid state,
the loads and all seven interface records. A browser needs geometry too, and the
episode does not store it -- so the coarsened blocks are rebuilt here from
`grid.build_blocks` and `generate._coarsen_block`, which is the same construction
the run used, not a reimplementation of it.

**What is and is not dimensional.** `snapshot` nondimensionalises every gas field
through the run's own per-agent scales, so `T` here is not kelvin. The rigid
state and the loads ARE dimensional. The page says so rather than letting a
colour bar imply otherwise.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from atlas.cases import rocket_experts as RE

GAS = ("a", "b", "e", "d", "f", "g")
GAS_FIELDS = ("rho", "u", "v", "p", "T")
SHELL_FIELDS = ("T", "ux", "uy", "s_zz", "s_yy", "s_zy")
IFACE = ("mass_flux", "mom_z_flux", "mom_y_flux", "energy_flux",
         "heat_flux", "p_trace", "T_trace", "seg_length")
RIGID = ("x", "y", "theta", "vx", "vy", "omega", "m")
LOADS = ("Fz_thrust", "Fy_thrust", "Fz_aero", "Fy_aero")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="out/w321")
    ap.add_argument("--out", default="out/w321/viz.json")
    ap.add_argument("--field", default="T")
    a = ap.parse_args(argv)

    with open(os.path.join(a.run, "episode.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    z = np.load(os.path.join(a.run, "episode.npz"))

    RE.load_solvers()
    gen = importlib.import_module("atlas_build_solvers.data.generate")
    grid = importlib.import_module("atlas_build_solvers.solvers.grid")
    cfg = importlib.import_module("atlas_build_solvers.config").load_config()

    fi = GAS_FIELDS.index(a.field)
    agents = {}
    for aid in GAS:
        blocks = [gen._coarsen_block(b, meta["coarsen"])
                  for b in grid.build_blocks(cfg.agent(aid), cfg)]
        blk = blocks[-1] if len(blocks) > 1 else blocks[0]
        nodes = np.asarray(blk.nodes, dtype=float)          # (nz+1, ny+1, 2)
        zc = 0.25 * (nodes[:-1, :-1, 0] + nodes[1:, :-1, 0]
                     + nodes[1:, 1:, 0] + nodes[:-1, 1:, 0])
        yc = 0.25 * (nodes[:-1, :-1, 1] + nodes[1:, :-1, 1]
                     + nodes[1:, 1:, 1] + nodes[:-1, 1:, 1])
        f = z["field_%s" % aid][:, fi]                      # (n_macro, nz, ny)
        # the episode concatenates multi-block agents along axis 1; keep the
        # slice this block owns so the grid and the field line up
        nz, ny = zc.shape
        f = f[:, -nz:, :ny] if f.shape[1] >= nz else f
        agents[aid] = dict(
            nz=int(nz), ny=int(ny),
            z=[round(float(v), 5) for v in zc.ravel()],
            y=[round(float(v), 5) for v in yc.ravel()],
            blanked=[bool(v) for v in np.asarray(blk.blanked).ravel()],
            frames=[[round(float(v), 4) for v in fr.ravel()] for fr in f],
        )

    rigid, loads, t = z["rigid"], z["loads"], z["t"]
    iface = {}
    for k in z.files:
        if not k.startswith("iface_"):
            continue
        arr = z[k]                                          # (n_macro, 8, ncell)
        iface[k[6:]] = {ch: [round(float(v), 6) for v in arr[:, i, :].mean(axis=1)]
                        for i, ch in enumerate(IFACE) if ch != "seg_length"}

    out = dict(
        meta=meta, field=a.field,
        note=("gas fields are NONDIMENSIONAL -- `snapshot` scales every one "
              "through the run's own per-agent scales. The rigid state and the "
              "loads are dimensional."),
        t=[round(float(v), 6) for v in t],
        agents=agents,
        rigid={k: [round(float(v), 6) for v in rigid[:, i]]
               for i, k in enumerate(RIGID)},
        loads={k: [round(float(v), 4) for v in loads[:, i]]
               for i, k in enumerate(LOADS)},
        iface=iface,
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    kb = os.path.getsize(a.out) / 1024.0
    print("wrote %s (%.0f KB): %d frames, %d agents, %d seams"
          % (a.out, kb, len(t), len(agents), len(iface)))
    for aid, d in agents.items():
        fr = np.array(d["frames"])
        print("  %-2s %2dx%-2d  %s in [%.4g, %.4g]"
              % (aid, d["nz"], d["ny"], a.field, fr.min(), fr.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
