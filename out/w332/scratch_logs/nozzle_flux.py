"""Is the nozzle's symmetry-breaking mode the HLLC flux's?

Inviscid engine (a, b, e) marched alone with symmetric walls, as in
nozzle_symmetry.py, but ONLY agent e's flux is swapped:
  hllc     -- production
  hlle     -- two-wave HLL (Einfeldt speeds); carbuncle-free by construction
  rusanov  -- local Lax-Friedrichs
Saves e's primitive state after every macro step to nozzle_flux_<v>.npz.
"""
import dataclasses
import sys
import time

sys.path.insert(0, r"C:\Users\Nauni\physics-foundation-model\src")
import numpy as np  # noqa: E402

from atlas.config import load_config  # noqa: E402
from atlas.data import generate as gen  # noqa: E402
from atlas.data import sweep as sw  # noqa: E402
from atlas.solvers import riemann  # noqa: E402
from atlas.solvers.thermo import sound_speed  # noqa: E402


def hlle(UL, UR, n, gamma):
    L, R = riemann._rotate(UL, n), riemann._rotate(UR, n)
    uL, uR = L[..., 1] / L[..., 0], R[..., 1] / R[..., 0]
    cL, cR = sound_speed(L, gamma), sound_speed(R, gamma)
    SL = np.minimum(np.minimum(uL - cL, uR - cR), 0.0)
    SR = np.maximum(np.maximum(uL + cL, uR + cR), 0.0)
    FL, FR = riemann._flux_1d(L, gamma), riemann._flux_1d(R, gamma)
    F = (SR[..., None] * FL - SL[..., None] * FR
         + (SL * SR)[..., None] * (R - L)) / (SR - SL)[..., None]
    return riemann._unrotate_flux(F, n)


riemann.FLUXES["hlle"] = hlle

variant = sys.argv[1]
n_macro = int(sys.argv[2]) if len(sys.argv) > 2 else 5
chunks = 5
cfg = load_config()
p = sw.corner_cases()[0]
spec = gen.EpisodeSpec(point=p, n_macro=n_macro, dt_macro=5.0e-3, coarsen=4,
                       riemann="hllc", inviscid=True)
ep = gen.CoupledEpisode(spec, cfg)
ep._wall_T = lambda agent, k, side: np.full(ep.blocks[agent][k].shape[0], 288.15)
se = ep.sol["e"][0]
se.cfg = dataclasses.replace(se.cfg, riemann=variant)


def asym(a):
    W = gen.thermo.cons_to_prim(ep.U[a][0], ep.sol[a][0].cfg.gamma)
    q = W[..., 3]
    return float(np.abs(q - q[:, ::-1]).max() / np.abs(q).max())


print("e flux %s (a, b stay hllc), inviscid" % variant, flush=True)
t0 = time.perf_counter()
states = []
for m in range(n_macro):
    ep._wire_engine()
    dt = ep.dt_macro
    for a in gen.ENGINE_AGENTS:
        line = []
        for c in range(chunks):
            ep.U[a][0], n = ep.sol[a][0].advance(ep.U[a][0], dt / chunks)
            line.append(asym(a))
        if a == "e":
            print("  macro %d  e  %s" % (m + 1, " ".join("%.1e" % x for x in line)), flush=True)
    states.append(gen.thermo.cons_to_prim(ep.U["e"][0], se.cfg.gamma))
np.savez("nozzle_flux_%s.npz" % variant, W=np.array(states), gamma=se.cfg.gamma,
         R=se.cfg.R, zc=ep.blocks["e"][0].centroid[..., 0], yc=ep.blocks["e"][0].centroid[..., 1])
print("wall %.0f s" % (time.perf_counter() - t0), flush=True)
