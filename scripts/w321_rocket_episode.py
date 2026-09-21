r"""W321 -- a COUPLED rocket episode, coarsened, through the build repo's own runner.

Every tier since 76 has probed this graph one seam at a time. This one marches
it: `generate.CoupledEpisode.step_macro` advances the six gas blocks, then the
shell, then computes the loads and integrates the rigid body -- the whole
vehicle, with the trajectory agent Tier 81 built now doing the job it was built
for, inside the build repo's own coupler rather than beside it.

**Nothing here is a solver.** The episode, the coupling order and the loads are
`generate.py`'s; this script chooses an operating point from `sweep.py`'s own
corner cases, states its two declarations, runs, and records.

**The two declarations, and both are reported rather than buried.**

``coarsen``    divides every grid dimension. Measured on this machine: one macro
               step at ``dt_macro = 5e-3`` costs 49.2 s at coarsen 4 and 17.1 s
               at coarsen 8, so the run is priced before it is spent.
``dt_macro``   5e-3 s against the config's own 5e-2. That is a 10x REDUCTION,
               and by Tier 80 it moves the coupling defect the right way: the
               chamber's flow-through is about 6e-3 s, so 5e-3 is 0.83 of one --
               below the knee Tier 84 measured at 1.67, where sigma is still
               first order in the interval and the CS-11 bound still holds. At
               the config's 5e-2 (8.3 flow-throughs) it does not.

    python scripts/w321_rocket_episode.py --coarsen 4 --n-macro 30 --out out/w321
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from atlas.cases import rocket_experts as RE

STATE = ("x", "y", "theta", "vx", "vy", "omega", "m")
LOADS = ("F_thrust_x", "F_thrust_y", "F_aero_x", "F_aero_y")


def _mods():
    RE.load_solvers()
    return (importlib.import_module("atlas_build_solvers.data.generate"),
            importlib.import_module("atlas_build_solvers.data.sweep"),
            importlib.import_module("atlas_build_solvers.config"),
            importlib.import_module("atlas_build_solvers.solvers.atmosphere"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coarsen", type=int, default=4)
    ap.add_argument("--n-macro", type=int, default=30)
    ap.add_argument("--dt-macro", type=float, default=5.0e-3)
    ap.add_argument("--case", type=int, default=0,
                    help="index into sweep.corner_cases()")
    ap.add_argument("--out", default="out/w321")
    a = ap.parse_args(argv)

    gen, sw, cfgmod, atm = _mods()
    cfg = cfgmod.load_config()
    pts = sw.corner_cases()
    p = pts[a.case]

    print("=" * 84)
    print("W321 -- a coupled rocket episode")
    print("=" * 84)
    print("  operating point : corner case %d, %r" % (a.case, p.label))
    print("                    p_c = %.4g Pa, T_c = %.1f K, h0 = %.0f m, "
          "M_inf = %.2f, alpha = %.1f deg" % (p.p_c, p.T_c, p.h0, p.M_inf, p.alpha))
    print("  declarations    : coarsen = %d, dt_macro = %g s "
          "(config's own is %g)" % (a.coarsen, a.dt_macro, cfg.dt_macro))
    print("                    %g s is %.2f chamber flow-throughs, and Tier 84 "
          "put the" % (a.dt_macro, a.dt_macro / 6.0e-3))
    print("                    first-order/second-order knee at 1.67 -- so this "
          "runs BELOW it,")
    print("                    where sigma is first order and the CS-11 bound "
          "still holds.")
    print("  steps           : %d macro steps = %.4g s of flight"
          % (a.n_macro, a.n_macro * a.dt_macro))

    spec = gen.EpisodeSpec(point=p, n_macro=a.n_macro, dt_macro=a.dt_macro,
                           coarsen=a.coarsen)
    ep = gen.CoupledEpisode(spec, cfg)
    shapes = {k: tuple(ep.blocks[k][0].shape) for k in gen.GAS_AGENTS}
    print("  blocks          : %s" % shapes)
    print()
    print("marching ...", flush=True)
    t0 = time.perf_counter()
    r = ep.run()
    wall = time.perf_counter() - t0

    rigid, loads = r["rigid"], r["loads"]
    t = np.arange(rigid.shape[0]) * a.dt_macro

    print()
    print("=== the vehicle ===")
    print("  %-8s %-12s %-12s %-12s %-12s" % ("t [s]", "y [m]", "vy [m/s]",
                                              "m [kg]", "F_thrust_y [N]"))
    idx = sorted(set([0, 1, rigid.shape[0] // 2, rigid.shape[0] - 1]))
    for i in idx:
        print("  %-8.4f %-12.3f %-12.4f %-12.2f %-12.4g"
              % (t[i], rigid[i, 1], rigid[i, 4], rigid[i, 6], loads[i, 1]))

    #: **The acceleration audit.** A trajectory that does not reconcile with its
    #: own loads is a trajectory nobody should plot. This checks the integrator
    #: against the forces it was handed, and reports the residual rather than
    #: asserting agreement -- `rhs` is (fx/m, fy/m - g), so the vertical
    #: acceleration is fully determined by loads, mass and altitude.
    print()
    print("=== the acceleration audit ===")
    #: **The indexing is not obvious and getting it wrong looks like a defect.**
    #: `run()` seeds the lists with the PRE-run state and then, inside the loop,
    #: appends after `step_macro()` -- which computes the loads and then
    #: integrates. So ``loads[k]`` is the load that PRODUCED ``rigid[k]``, and
    #: ``loads[0]`` is a pre-run zero that drove nothing. Pairing ``loads[i]``
    #: with the step out of ``rigid[i]`` reports a clean one-step lag as a
    #: disagreement; measured[i] then equals predicted[i+1] to five digits,
    #: which is the signature of an off-by-one rather than of a broken
    #: integrator, and it is how this one was found.
    rows = []
    for i in range(min(6, rigid.shape[0] - 1)):
        dv = (rigid[i + 1, 4] - rigid[i, 4]) / a.dt_macro
        m = float(rigid[i, 6])
        g = float(atm.gravity(float(rigid[i, 1])))
        want = (loads[i + 1, 1] + loads[i + 1, 3]) / m - g
        rows.append(dict(step=i, measured=float(dv), predicted=float(want),
                         ratio=float(dv / want) if want else None,
                         g=g, m=m))
        print("  step %d: dvy/dt measured %-12.4f predicted %-12.4f ratio %-8.5f"
              % (i, dv, want, dv / want if want else float("nan")))
    ok = all(abs(x["ratio"] - 1.0) < 0.02 for x in rows if x["ratio"])
    print("  reconciles to 2%%: %s   (loads[k] is the load that PRODUCED rigid[k])"
          % ok)
    if not ok:
        print("  ** the integrator and the loads DISAGREE -- recorded, not hidden **")
    print()
    print("  and the sign: theta = %.4f rad, F_thrust_y = %.4g N"
          % (rigid[0, 2], loads[1, 1]))
    print("  `generate.py` sets F_thrust = (-thrust cos th, -thrust sin th) with")
    print("  the comment \"body -z is 'up'\", so a nose-up vehicle at th = pi/2")
    print("  gets a NEGATIVE vertical thrust. Reported as measured.")

    os.makedirs(a.out, exist_ok=True)
    npz = os.path.join(a.out, "episode.npz")
    np.savez_compressed(
        npz, rigid=rigid, loads=loads, t=t,
        **{"field_%s" % k: v for k, v in r["fields"].items()},
        **{"iface_%s" % k: v for k, v in r["iface"].items()})
    summary = dict(
        case=a.case, label=p.label, p_c=p.p_c, T_c=p.T_c, h0=p.h0,
        M_inf=p.M_inf, alpha=p.alpha,
        coarsen=a.coarsen, n_macro=a.n_macro, dt_macro=a.dt_macro,
        config_dt_macro=float(cfg.dt_macro),
        flow_throughs=a.dt_macro / 6.0e-3,
        blocks={k: list(v) for k, v in shapes.items()},
        field_shapes={k: list(v.shape) for k, v in r["fields"].items()},
        iface_shapes={k: list(v.shape) for k, v in r["iface"].items()},
        substeps=int(r["substeps"]), wall_seconds=wall,
        s_per_macro_step=wall / max(1, a.n_macro),
        rigid_first=[float(x) for x in rigid[0]],
        rigid_last=[float(x) for x in rigid[-1]],
        loads_last=[float(x) for x in loads[-1]],
        acceleration_audit=rows, audit_ok=bool(ok),
        state_fields=list(STATE), load_fields=list(LOADS))
    with open(os.path.join(a.out, "episode.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print()
    print("=== cost ===")
    print("  %d macro steps, %d CFL sub-steps, %.0f s wall (%.1f s a step)"
          % (a.n_macro, r["substeps"], wall, wall / max(1, a.n_macro)))
    print("  written to %s and %s"
          % (npz, os.path.join(a.out, "episode.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
