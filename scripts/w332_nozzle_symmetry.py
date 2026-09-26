r"""W332 -- does the engine's gas flow keep its mirror symmetry?

The first fixed rocket march (out/w321_seams_only) failed its two symmetry
gates. The nozzle's pressure field was asymmetric at 1e-2, and so was the
pre-fix run's (out/w321_prefix, 4.6e-3 by its third step). This script is the
experiment that found why. It marches the ENGINE ALONE (agents a, b, e), with
the coupler's own wiring but both walls pinned to one temperature, so nothing
asymmetric enters from outside. It prints the mirror asymmetry

    max |p - p_mirror| / max |p|

of each block, five times per 5 ms macro step. Round-off is ~1e-15.

Switches:
  --flux {hllc,hlle,rusanov}  the nozzle's flux only (a and b stay on hllc).
                               HLLE is two-wave, with Einfeldt speeds, defined
                               here.
  --wall {iso,adiab,slip}     the nozzle's walls: the coupler's isothermal
                               no-slip wall, an adiabatic no-slip wall, or slip.
  --viscous                   Navier-Stokes rather than Euler.
  --old-ghost                 hand the inviscid flux the isothermal ghost
                               again -- what `Compressible2D.residual` did
                               before W332 -- to reproduce the failure.

    python scripts/w332_nozzle_symmetry.py --wall iso --old-ghost
"""
from __future__ import annotations

import argparse
import dataclasses
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flux", default="hllc", choices=("hllc", "hlle", "rusanov"))
    ap.add_argument("--wall", default="iso", choices=("iso", "adiab", "slip"))
    ap.add_argument("--viscous", action="store_true")
    ap.add_argument("--old-ghost", action="store_true")
    ap.add_argument("--macros", type=int, default=4)
    ap.add_argument("--out", default="out/w332")
    a = ap.parse_args(argv)

    RE.load_solvers()
    gen = importlib.import_module("atlas_build_solvers.data.generate")
    sw = importlib.import_module("atlas_build_solvers.data.sweep")
    cfgm = importlib.import_module("atlas_build_solvers.config")
    rie = importlib.import_module("atlas_build_solvers.solvers.riemann")
    c2d = importlib.import_module("atlas_build_solvers.solvers.compressible2d")
    thermo = importlib.import_module("atlas_build_solvers.solvers.thermo")

    def hlle(UL, UR, n, gamma):
        L, R = rie._rotate(UL, n), rie._rotate(UR, n)
        uL, uR = L[..., 1] / L[..., 0], R[..., 1] / R[..., 0]
        cL, cR = thermo.sound_speed(L, gamma), thermo.sound_speed(R, gamma)
        SL = np.minimum(np.minimum(uL - cL, uR - cR), 0.0)
        SR = np.maximum(np.maximum(uL + cL, uR + cR), 0.0)
        FL, FR = rie._flux_1d(L, gamma), rie._flux_1d(R, gamma)
        F = (SR[..., None] * FL - SL[..., None] * FR
             + (SL * SR)[..., None] * (R - L)) / (SR - SL)[..., None]
        return rie._unrotate_flux(F, n)

    rie.FLUXES["hlle"] = hlle

    if a.old_ghost:
        def residual(self, U):
            Ue = self.ghost(U)                       # the isothermal ghost, for both
            r = self._inviscid_residual(Ue)
            if not self.cfg.inviscid:
                r = r + self._viscous_residual(Ue)
            if self._hole is not None:
                j0, j1 = self._hole
                r[:, j0:j1 + 1] = 0.0
            return r
        c2d.Compressible2D.residual = residual

    spec = gen.EpisodeSpec(point=sw.corner_cases()[0], n_macro=a.macros + 1, dt_macro=5.0e-3,
                           coarsen=4, riemann="hllc", inviscid=not a.viscous)
    ep = gen.CoupledEpisode(spec, cfgm.load_config())
    ep._wall_T = lambda agent, k, side: np.full(ep.blocks[agent][k].shape[0], 288.15)
    se = ep.sol["e"][0]
    se.cfg = dataclasses.replace(se.cfg, riemann=a.flux)

    def asym(agent):
        W = thermo.cons_to_prim(ep.U[agent][0], ep.sol[agent][0].cfg.gamma)
        p = W[..., 3]
        return float(np.abs(p - p[:, ::-1]).max() / np.abs(p).max())

    name = "%s_%s_%s%s" % (a.flux, a.wall, "viscous" if a.viscous else "inviscid",
                           "_oldghost" if a.old_ghost else "")
    print("W332 engine symmetry: nozzle flux %s, nozzle walls %s, %s%s" % (
        a.flux, a.wall, "viscous" if a.viscous else "inviscid",
        ", OLD ghost in the inviscid flux" if a.old_ghost else ""), flush=True)
    rows, t0 = [], time.perf_counter()
    for m in range(a.macros):
        ep._wire_engine()
        if a.wall != "iso":
            for side in ("jmin", "jmax"):
                se.bcs[side] = c2d.BC("wall_noslip" if a.wall == "adiab" else "wall_slip", {})
        for agent in gen.ENGINE_AGENTS:
            vals = []
            for _ in range(5):
                ep.U[agent][0], _n = ep.sol[agent][0].advance(ep.U[agent][0], ep.dt_macro / 5)
                vals.append(asym(agent))
            rows.append(dict(macro=m + 1, agent=agent, asym=vals))
            print("  %2d ms  %s  %s" % (5 * (m + 1), agent, " ".join("%.1e" % v for v in vals)),
                  flush=True)
    wall = time.perf_counter() - t0
    os.makedirs(a.out, exist_ok=True)
    out = dict(variant=name, flux=a.flux, wall=a.wall, viscous=a.viscous, old_ghost=a.old_ghost,
               rows=rows, wall_seconds=wall, dt_macro=5.0e-3,
               nozzle_max=max(max(r["asym"]) for r in rows if r["agent"] == "e"))
    path = os.path.join(a.out, name + ".json")
    for _ in range(40):
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1)
            break
        except PermissionError:
            time.sleep(0.25)
    print("nozzle max %.2e; wrote %s (%.0f s)" % (out["nozzle_max"], path, wall), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
