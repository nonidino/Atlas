"""Is the nozzle's HLLC mode driven by its WALL treatment?

Inviscid engine, e on HLLC. Only e's two walls change:
  iso     -- wall_noslip + T_wall (the coupler's wall; the ghost is 2Tw-Ti, clamped at 20 K)
  adiab   -- wall_noslip, no T_wall (reflection at equal density)
  slip    -- wall_slip
"""
import sys
import time
sys.path.insert(0, r"C:\Users\Nauni\physics-foundation-model\src")
import numpy as np  # noqa: E402
from atlas.config import load_config  # noqa: E402
from atlas.data import generate as gen  # noqa: E402
from atlas.data import sweep as sw  # noqa: E402
from atlas.solvers.compressible2d import BC  # noqa: E402

variant = sys.argv[1]
n_macro = int(sys.argv[2]) if len(sys.argv) > 2 else 4
spec = gen.EpisodeSpec(point=sw.corner_cases()[0], n_macro=n_macro, dt_macro=5.0e-3,
                       coarsen=4, riemann="hllc", inviscid=True)
ep = gen.CoupledEpisode(spec, load_config())
ep._wall_T = lambda agent, k, side: np.full(ep.blocks[agent][k].shape[0], 288.15)


def asym(a):
    W = gen.thermo.cons_to_prim(ep.U[a][0], ep.sol[a][0].cfg.gamma)
    q = W[..., 3]
    return float(np.abs(q - q[:, ::-1]).max() / np.abs(q).max())


print("e walls %s, inviscid, hllc" % variant, flush=True)
t0 = time.perf_counter()
for m in range(n_macro):
    ep._wire_engine()
    if variant == "adiab":
        for s in ("jmin", "jmax"):
            ep.sol["e"][0].bcs[s] = BC("wall_noslip", {})
    elif variant == "slip":
        for s in ("jmin", "jmax"):
            ep.sol["e"][0].bcs[s] = BC("wall_slip", {})
    for a in gen.ENGINE_AGENTS:
        line = []
        for c in range(5):
            ep.U[a][0], n = ep.sol[a][0].advance(ep.U[a][0], ep.dt_macro / 5)
            line.append(asym(a))
        if a == "e":
            print("  macro %d  e  %s" % (m + 1, " ".join("%.1e" % x for x in line)), flush=True)
print("wall %.0f s" % (time.perf_counter() - t0), flush=True)
