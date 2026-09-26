"""W332's 'before' numbers, computed with the current code by handing the
inviscid flux the thermal ghost again (what residual() did before W332)."""
import sys
sys.path.insert(0, r"C:\Users\Nauni\physics-foundation-model\src")
import numpy as np  # noqa: E402
from atlas.config import load_config  # noqa: E402
from atlas.data import generate as gen  # noqa: E402
from atlas.data import sweep as sw  # noqa: E402
from atlas.solvers import thermo  # noqa: E402
from atlas.solvers.compressible2d import BC, NG, Block, Compressible2D, GasConfig  # noqa: E402
from atlas.solvers.riemann import FLUXES  # noqa: E402

# (1) the unit-test block: 2500 K gas at 5 MPa, 288 K wall, 50 m/s into it
nz, ny = 6, 4
z, y = np.linspace(0, 0.06, nz + 1), np.linspace(0, 0.03, ny + 1)
Z, Y = np.meshgrid(z, y, indexing="ij")
blk = Block("w", np.stack([Z, Y], -1), np.zeros((nz, ny), bool))
gas = GasConfig(gamma=1.22, R=361.0)
s = Compressible2D(blk, gas, bcs={"jmin": BC("wall_noslip", {"T_wall": np.full(nz, 288.0)}),
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
    return (FLUXES["hllc"](L, R, blk.n_j, gas.gamma) * blk.a_j[..., None])[:, 0, 0]


scale = W[0, 0, 0] * 50.0 * blk.a_j[0, 0]
Ue = s.ghost(U)
print("ghost/cell density at the wall: %.1f" % (Ue[NG, NG - 1, 0] / U[0, 0, 0]))
print("wall-face mass flux / (rho v a): thermal ghost %.1f, mirror %.1e"
      % (np.abs(jmin(Ue)).max() / scale, np.abs(jmin(s.ghost(U, thermal=False))).max() / scale))

# (2) gate C6 on the coupler's engine blocks, both ways
cfg = load_config()
ep = gen.CoupledEpisode(gen.EpisodeSpec(point=sw.corner_cases()[0], n_macro=2, dt_macro=5e-3,
                                        coarsen=4), cfg)
ep._wire_engine()
for agent in gen.ENGINE_AGENTS:
    so = ep.sol[agent][0]
    b = so.block
    saved = dict(so.bcs)
    so.bcs = {"jmin": saved["jmin"], "jmax": saved["jmax"],
              "imin": BC("wall_noslip"), "imax": BC("wall_noslip")}
    Wp = thermo.cons_to_prim(ep.U[agent][0], so.cfg.gamma).copy()
    Wp[..., 2] = 50.0 * np.sign(b.centroid[..., 1])
    Up = thermo.prim_to_cons(Wp, so.cfg.gamma)
    sc = float(Wp[..., 0].max()) * 50.0 * float(b.a_j[:, 0].sum() + b.a_j[:, -1].sum())
    Ue = so.ghost(Up)
    old = so._inviscid_residual(Ue) + so._viscous_residual(Ue)
    new = so.residual(Up)
    print("C6 %s: before W332 %.3e, after %.3e" % (agent, abs((old[..., 0] * b.vol).sum()) / sc,
                                                 abs((new[..., 0] * b.vol).sum()) / sc))
    so.bcs = saved
