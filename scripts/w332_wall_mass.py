r"""W332 -- how much mass an isothermal wall passed, before and after.

Two measurements, each taken both ways with the current build repo. "Before"
hands the inviscid flux the isothermal ghost again, which is what
`Compressible2D.residual` did until W332; "after" is the residual as it stands.

1. **One wall face.** 2500 K gas at 5 MPa beside a 288 K wall, moving along it
   at 800 m/s and into it at 50 m/s. The isothermal ghost sets
   $T_g = \max(2T_w - T_i,\ 20\ \mathrm{K})$ at the cell's pressure, so it is
   $T_i/T_g$ times denser than the cell: 125 here. Reported: the HLLC mass flux
   through the wall face, over the cell's own normal mass flux
   $\rho\,|v|\,a$.
2. **Gate C6 on the coupler's engine blocks.** `a`, `b` and `e` wired by
   `CoupledEpisode._wire_engine` (their own isothermal walls on j), closed with
   adiabatic walls on i, with the gas pushed at both walls at 50 m/s. An
   impermeable box conserves mass, so the net mass rate over
   $\rho\,|v|\,L_{\text{walls}}$ should be round-off.

    python scripts/w332_wall_mass.py      -> out/w332_wall_mass.json
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from atlas.cases import rocket_experts as RE

OUT = os.path.join("out", "w332_wall_mass.json")


def main():
    RE.load_solvers()
    gen = importlib.import_module("atlas_build_solvers.data.generate")
    sw = importlib.import_module("atlas_build_solvers.data.sweep")
    cfgm = importlib.import_module("atlas_build_solvers.config")
    c2d = importlib.import_module("atlas_build_solvers.solvers.compressible2d")
    grid = importlib.import_module("atlas_build_solvers.solvers.grid")
    thermo = importlib.import_module("atlas_build_solvers.solvers.thermo")
    rie = importlib.import_module("atlas_build_solvers.solvers.riemann")
    NG, BC = c2d.NG, c2d.BC

    # 1. one wall face
    nz, ny = 6, 4
    z, y = np.linspace(0, 0.06, nz + 1), np.linspace(0, 0.03, ny + 1)
    Z, Y = np.meshgrid(z, y, indexing="ij")
    blk = grid.Block("w", np.stack([Z, Y], -1), np.zeros((nz, ny), bool))
    gas = c2d.GasConfig(gamma=1.22, R=361.0)
    s = c2d.Compressible2D(blk, gas, bcs={"jmin": BC("wall_noslip", {"T_wall": np.full(nz, 288.0)}),
                                          "jmax": BC("wall_noslip"), "imin": BC("extrapolate"),
                                          "imax": BC("extrapolate")})
    W = np.zeros((nz, ny, 4))
    W[..., 0] = 5.0e6 / (gas.R * 2500.0)
    W[..., 1] = 800.0
    W[..., 2] = -50.0
    W[..., 3] = 5.0e6
    U = thermo.prim_to_cons(W, gas.gamma)

    def jmin(Ue):
        L, R = s._faces(Ue[NG:nz + NG, :], axis=1)
        return (rie.FLUXES["hllc"](L, R, blk.n_j, gas.gamma) * blk.a_j[..., None])[:, 0, 0]

    scale = W[0, 0, 0] * 50.0 * blk.a_j[0, 0]
    Ue = s.ghost(U)
    face = dict(T_gas=2500.0, T_wall=288.0, p=5.0e6, v_into_wall=50.0,
                ghost_over_cell_density=float(Ue[NG, NG - 1, 0] / U[0, 0, 0]),
                before=float(np.abs(jmin(Ue)).max() / scale),
                after=float(np.abs(jmin(s.ghost(U, thermal=False))).max() / scale))

    # 2. gate C6, both ways
    ep = gen.CoupledEpisode(gen.EpisodeSpec(point=sw.corner_cases()[0], n_macro=2,
                                            dt_macro=5e-3, coarsen=4), cfgm.load_config())
    ep._wire_engine()
    blocks = {}
    for agent in gen.ENGINE_AGENTS:
        so = ep.sol[agent][0]
        b = so.block
        saved = dict(so.bcs)
        try:
            so.bcs = {"jmin": saved["jmin"], "jmax": saved["jmax"],
                      "imin": BC("wall_noslip"), "imax": BC("wall_noslip")}
            Wp = thermo.cons_to_prim(ep.U[agent][0], so.cfg.gamma).copy()
            Wp[..., 2] = 50.0 * np.sign(b.centroid[..., 1])      # at both walls
            Up = thermo.prim_to_cons(Wp, so.cfg.gamma)
            sc = float(Wp[..., 0].max()) * 50.0 * float(b.a_j[:, 0].sum() + b.a_j[:, -1].sum())
            Ug = so.ghost(Up)
            old = so._inviscid_residual(Ug) + so._viscous_residual(Ug)
            new = so.residual(Up)
            blocks[agent] = dict(before=float(abs((old[..., 0] * b.vol).sum()) / sc),
                                 after=float(abs((new[..., 0] * b.vol).sum()) / sc))
        finally:
            so.bcs = saved

    out = dict(face=face, c6=blocks, build_repo=RE.build_repo_identity())
    print("one wall face: ghost/cell density %.1f; mass flux / (rho v a): before %.1f, after %.1e"
          % (face["ghost_over_cell_density"], face["before"], face["after"]))
    for k, v in blocks.items():
        print("C6 %s: before %.3e, after %.3e" % (k, v["before"], v["after"]))
    for _ in range(40):
        try:
            with open(OUT, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1)
            break
        except PermissionError:
            time.sleep(0.25)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
