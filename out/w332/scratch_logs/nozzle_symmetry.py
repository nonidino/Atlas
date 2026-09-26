"""Does the engine's gas flow break mirror symmetry by itself?

The coupled episode's nozzle (agent e) goes from round-off asymmetry to ~1e-2
within one 5 ms macro step, in the pre-fix run as well as the fixed one. Here
the engine (a, b, e) is marched ALONE, with the SAME wiring the coupler uses
but both walls pinned to one scalar temperature, so nothing asymmetric enters
from outside. Variants:
  hllc      -- the production flux
  rusanov   -- the dissipative flux (a carbuncle would vanish here)
  inviscid  -- no viscous terms, no no-slip layer
Output: per-ms asymmetry max|p - p_mirror|/max|p| for a, b, e.
"""
import os
import sys
import time

sys.path.insert(0, r"C:\Users\Nauni\physics-foundation-model\src")
import numpy as np  # noqa: E402

from atlas.config import load_config  # noqa: E402
from atlas.data import generate as gen  # noqa: E402
from atlas.data import sweep as sw  # noqa: E402

variant = sys.argv[1]
n_macro = int(sys.argv[2]) if len(sys.argv) > 2 else 4
chunks = 5
cfg = load_config()
p = sw.corner_cases()[0]
spec = gen.EpisodeSpec(point=p, n_macro=n_macro, dt_macro=5.0e-3, coarsen=4,
                       riemann="rusanov" if variant == "rusanov" else "hllc",
                       inviscid=(variant == "inviscid"))
ep = gen.CoupledEpisode(spec, cfg)
T_fixed = 288.15
ep._wall_T = lambda agent, k, side: np.full(ep.blocks[agent][k].shape[0], T_fixed)


def asym(a):
    U = ep.U[a][0]
    W = gen.thermo.cons_to_prim(U, ep.sol[a][0].cfg.gamma)
    q = W[..., 3]
    return float(np.abs(q - q[:, ::-1]).max() / np.abs(q).max())


print("variant %s, blocks a%s b%s e%s" % (variant, ep.blocks["a"][0].shape,
      ep.blocks["b"][0].shape, ep.blocks["e"][0].shape), flush=True)
t0 = time.perf_counter()
for m in range(n_macro):
    ep._wire_engine()
    dt = ep.dt_macro
    for a in gen.ENGINE_AGENTS:
        line = []
        for c in range(chunks):
            ep.U[a][0], n = ep.sol[a][0].advance(ep.U[a][0], dt / chunks)
            line.append(asym(a))
        print("  macro %d  %s  %s" % (m + 1, a, " ".join("%.1e" % x for x in line)),
              flush=True)
print("wall %.0f s" % (time.perf_counter() - t0), flush=True)
