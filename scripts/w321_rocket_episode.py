r"""W321 -- a COUPLED rocket episode, coarsened, through the build repo's own runner.

Every tier since 76 has probed this graph one seam at a time. This one marches
it: `generate.CoupledEpisode.step_macro` advances the six gas blocks, then the
shell, then computes the loads and integrates the rigid body -- the whole
vehicle, with the trajectory agent Tier 81 built now doing the job it was built
for, inside the build repo's own coupler rather than beside it.

**Nothing here is a solver.** The episode, the coupling order and the loads are
`generate.py`'s; this script chooses an operating point from `sweep.py`'s own
corner cases, states its two declarations, marches, and records.

**The two declarations, and both are reported rather than buried.**

``coarsen``    divides every grid dimension (the blocks are REBUILT at that
               resolution since Tier 86, not subsampled).
``dt_macro``   5e-3 s against the config's own 5e-2. That is a 10x REDUCTION,
               and a conservative one: the chamber's flow-through is about
               6e-3 s, so 5e-3 is 0.83 of one. sigma is first order in the
               interval out to the config's 5e-2 and the CS-11 bound holds
               there too, 2.28x loose (W334, Tier 86), so the reduction cuts
               the coupling defect tenfold rather than rescuing a bound. The
               knee Tier 84 measured at 1.67 flow-throughs, and Tier 80's
               violation at 5e-2, were the old isothermal wall's.

**Provenance.** The build repo's commit, branch, and -- when its working tree is
dirty -- a hash of `git diff HEAD` are written into ``episode.json``, so a number
from this run names the code that produced it even before that code is committed.

**Persistence.** The march is stepped here (the same calls `CoupledEpisode.run`
makes) so the whole trajectory so far is written every ``--save-every`` steps: a
march that dies at step 29 keeps 28 steps. While it runs, the process asks
Windows not to sleep (``SetThreadExecutionState``; released on exit).

    python scripts/w321_rocket_episode.py --coarsen 4 --n-macro 30 --out out/w321
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from atlas.cases import rocket_experts as RE
from atlas.cases.thermal_seam import build_repo

STATE = ("x", "y", "theta", "vx", "vy", "omega", "m")
LOADS = ("F_thrust_x", "F_thrust_y", "F_aero_x", "F_aero_y")


def _mods():
    RE.load_solvers()
    return (importlib.import_module("atlas_build_solvers.data.generate"),
            importlib.import_module("atlas_build_solvers.data.sweep"),
            importlib.import_module("atlas_build_solvers.config"),
            importlib.import_module("atlas_build_solvers.solvers.atmosphere"))


def provenance() -> dict:
    """The build repo's identity, including an uncommitted working tree."""
    def git(*a):
        try:
            return subprocess.run(["git", "-C", build_repo(), *a], capture_output=True,
                                  text=True, encoding="utf-8", timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            return ""
    commit = git("rev-parse", "--short", "HEAD").strip()
    ident = os.environ.get("ATLAS_BUILD_REPO_IDENTITY", "")
    if not commit and ident:
        # a rented machine gets the build repo without its history; the
        # identity was read where the history is (scripts/box/make_payload.py)
        c, _, h = ident.partition("+dirty:")
        return dict(commit=c, branch="", dirty=bool(h), diff_sha256=h, untracked=[],
                    identity=ident)
    diff = git("diff", "HEAD")
    untracked = git("ls-files", "--others", "--exclude-standard").split()
    return dict(commit=commit,
                branch=git("rev-parse", "--abbrev-ref", "HEAD").strip(),
                dirty=bool(diff.strip() or untracked),
                diff_sha256=hashlib.sha256(diff.encode("utf-8")).hexdigest()[:16] if diff else "",
                untracked=untracked)


class _Awake:
    """Ask Windows to keep the system awake while the march runs. A request, not
    a setting: it lasts as long as this process and changes no power plan."""
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

    def __enter__(self):
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(
                self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED)
        except (AttributeError, OSError):
            pass
        return self

    def __exit__(self, *exc):
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
        except (AttributeError, OSError):
            pass
        return False


def _save(path, **arrays):
    """np.savez appends .npz to a tmp name without it, so the tmp name ends in
    .npz and the replace is retried while OneDrive holds the file."""
    tmp = path[:-4] + ".partial.npz"
    np.savez_compressed(tmp, **arrays)
    for k in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coarsen", type=int, default=4)
    ap.add_argument("--n-macro", type=int, default=30)
    ap.add_argument("--dt-macro", type=float, default=5.0e-3)
    ap.add_argument("--case", type=int, default=0,
                    help="index into sweep.corner_cases()")
    ap.add_argument("--save-every", type=int, default=3)
    ap.add_argument("--out", default="out/w321")
    a = ap.parse_args(argv)

    gen, sw, cfgmod, atm = _mods()
    cfg = cfgmod.load_config()
    pts = sw.corner_cases()
    p = pts[a.case]
    prov = provenance()

    print("=" * 84)
    print("W321 -- a coupled rocket episode")
    print("=" * 84)
    print("  build repo      : %s on %s%s" % (prov["commit"], prov["branch"],
          ("  (DIRTY, diff %s, %d untracked)" % (prov["diff_sha256"], len(prov["untracked"])))
          if prov["dirty"] else ""))
    print("  operating point : corner case %d, %r" % (a.case, p.label))
    print("                    p_c = %.4g Pa, T_c = %.1f K, h0 = %.0f m, "
          "M_inf = %.2f, alpha = %.1f deg" % (p.p_c, p.T_c, p.h0, p.M_inf, p.alpha))
    print("  declarations    : coarsen = %d, dt_macro = %g s "
          "(config's own is %g)" % (a.coarsen, a.dt_macro, cfg.dt_macro))
    print("                    %g s is %.2f chamber flow-throughs; the bound holds "
          "to the config's own step (W334)" % (a.dt_macro, a.dt_macro / 6.0e-3))
    # n_macro counts FRAMES, the pre-run state included (generate.run's
    # convention), so the march is n_macro - 1 steps
    print("  steps           : %d macro steps (%d frames) = %.4g s of flight"
          % (a.n_macro - 1, a.n_macro, (a.n_macro - 1) * a.dt_macro))

    spec = gen.EpisodeSpec(point=p, n_macro=a.n_macro, dt_macro=a.dt_macro,
                           coarsen=a.coarsen)
    ep = gen.CoupledEpisode(spec, cfg)
    shapes = {k: tuple(ep.blocks[k][0].shape) for k in gen.GAS_AGENTS}
    print("  blocks          : %s" % shapes)
    print()
    print("marching ...", flush=True)

    os.makedirs(a.out, exist_ok=True)
    npz = os.path.join(a.out, "episode.npz")
    fields, iface, rigid, loads, body = {}, {}, [], [], []

    def record():
        f, _c, i = ep.snapshot()
        for k, v in f.items():
            fields.setdefault(k, []).append(v)
        for k, v in i.items():
            iface.setdefault(k, []).append(v)
        rigid.append(ep.rigid.copy())
        loads.append(ep.loads.copy())
        body.append(getattr(ep, "loads_body", np.zeros(4)).copy())

    def persist():
        _save(npz, rigid=np.stack(rigid), loads=np.stack(loads), loads_body=np.stack(body),
              t=np.arange(len(rigid)) * a.dt_macro,
              **{"field_%s" % k: np.stack(v) for k, v in fields.items()},
              **{"iface_%s" % k: np.stack(v) for k, v in iface.items()})

    t0 = time.perf_counter()
    with _Awake():
        record()
        for s in range(a.n_macro - 1):
            ts = time.perf_counter()
            ep.step_macro()
            record()
            print("  step %2d/%d  %.1f s  vy %.4f m/s  F_thrust_y %.4g N  F_aero_y %.4g N"
                  % (s + 1, a.n_macro - 1, time.perf_counter() - ts, ep.rigid[4],
                     ep.loads[1], ep.loads[3]), flush=True)
            if (s + 1) % a.save_every == 0:
                persist()
        persist()
    wall = time.perf_counter() - t0

    rigid, loads, body = np.stack(rigid), np.stack(loads), np.stack(body)
    t = np.arange(rigid.shape[0]) * a.dt_macro

    print()
    print("=== the vehicle ===")
    print("  %-8s %-12s %-12s %-12s %-12s %-12s" % ("t [s]", "y [m]", "vy [m/s]",
                                                     "m [kg]", "F_thrust_y [N]", "F_aero_y [N]"))
    idx = sorted(set([0, 1, rigid.shape[0] // 2, rigid.shape[0] - 1]))
    for i in idx:
        print("  %-8.4f %-12.3f %-12.4f %-12.2f %-12.4g %-12.4g"
              % (t[i], rigid[i, 1], rigid[i, 4], rigid[i, 6], loads[i, 1], loads[i, 3]))

    #: **The acceleration audit.** A trajectory that does not reconcile with its
    #: own loads is a trajectory nobody should plot. This checks the integrator
    #: against the forces it was handed, and reports the residual rather than
    #: asserting agreement -- `rhs` is (fx/m, fy/m - g), so the vertical
    #: acceleration is fully determined by loads, mass and altitude.
    print()
    print("=== the acceleration audit ===")
    #: **The indexing is not obvious and getting it wrong looks like a defect.**
    #: The lists are seeded with the PRE-run state and then appended after each
    #: `step_macro()` -- which computes the loads and then integrates. So
    #: ``loads[k]`` is the load that PRODUCED ``rigid[k]``, and ``loads[0]`` is a
    #: pre-run zero that drove nothing.
    rows = []
    for i in range(min(6, rigid.shape[0] - 1)):
        dv = (rigid[i + 1, 4] - rigid[i, 4]) / a.dt_macro
        m = float(rigid[i, 6])
        g = float(atm.gravity(float(rigid[i, 1])))
        want = (loads[i + 1, 1] + loads[i + 1, 3]) / m - g
        rows.append(dict(step=i, measured=float(dv), predicted=float(want),
                         ratio=float(dv / want) if want else None, g=g, m=m))
        print("  step %d: dvy/dt measured %-12.4f predicted %-12.4f ratio %-8.5f"
              % (i, dv, want, dv / want if want else float("nan")))
    ok = all(abs(x["ratio"] - 1.0) < 0.02 for x in rows if x["ratio"])
    print("  reconciles to 2%%: %s   (loads[k] is the load that PRODUCED rigid[k])" % ok)

    #: **The loads' directions**, the W322 check made permanent: at a nose-up
    #: attitude thrust must point up, the drag (+z_body, towards the tail) down,
    #: and a symmetric body at zero incidence carries no side force.
    print()
    print("=== the loads' directions ===")
    th = float(rigid[-1, 2])
    print("  theta %.4f rad; last step: F_thrust = (%.4g, %.4g) N, F_aero = (%.4g, %.4g) N"
          % (th, *loads[-1]))
    print("  body frame: thrust axial %.4g N (towards the nose is negative), aero axial "
          "(drag) %.4g N, aero normal %.4g N" % (body[-1, 0], body[-1, 2], body[-1, 3]))
    directions = dict(thrust_up=bool(loads[-1, 1] > 0.0), drag_to_tail=bool(body[-1, 2] > 0.0),
                      drag_down=bool(loads[-1, 3] < 0.0),
                      side_over_drag=float(abs(body[-1, 3]) / max(abs(body[-1, 2]), 1e-300)))
    print("  thrust up %(thrust_up)s, drag to the tail %(drag_to_tail)s, drag down "
          "%(drag_down)s, |side|/drag %(side_over_drag).2e" % directions)

    summary = dict(
        case=a.case, label=p.label, p_c=p.p_c, T_c=p.T_c, h0=p.h0,
        M_inf=p.M_inf, alpha=p.alpha,
        coarsen=a.coarsen, n_macro=a.n_macro, dt_macro=a.dt_macro,
        config_dt_macro=float(cfg.dt_macro),
        flow_throughs=a.dt_macro / 6.0e-3,
        build_repo=prov,
        blocks={k: list(v) for k, v in shapes.items()},
        field_shapes={k: [len(v)] + list(v[0].shape) for k, v in fields.items()},
        iface_shapes={k: [len(v)] + list(v[0].shape) for k, v in iface.items()},
        substeps=int(ep.substeps), wall_seconds=wall,
        s_per_macro_step=wall / max(1, a.n_macro - 1),
        rigid_first=[float(x) for x in rigid[0]],
        rigid_last=[float(x) for x in rigid[-1]],
        loads_last=[float(x) for x in loads[-1]],
        loads_body_last=[float(x) for x in body[-1]],
        acceleration_audit=rows, audit_ok=bool(ok), directions=directions,
        state_fields=list(STATE), load_fields=list(LOADS))
    with open(os.path.join(a.out, "episode.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print()
    print("=== cost ===")
    print("  %d macro steps, %d CFL sub-steps, %.0f s wall (%.1f s a step)"
          % (a.n_macro - 1, ep.substeps, wall, wall / max(1, a.n_macro - 1)))
    print("  written to %s and %s" % (npz, os.path.join(a.out, "episode.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
