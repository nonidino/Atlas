r"""W336 -- where the engine's thermal seam loses 4.7%.

The audit found the shell receiving about 4.7% less heat than the gas loses at
the engine's walls, steadily, from the first macro step. This splits the gas's
wall energy flux into what the solver's viscous operator builds it from, and
bins it onto the shell's face segments, for one macro step after two to let the
start settle:

* the one-sided conduction W301 installs, k(T_i) (T_i - T_w) / dn, face by face;
* everything else the viscous energy flux carries at a no-slip wall face -- the
  face average of u.tau between the cell and its mirrored ghost;
* the inviscid energy flux, which the mirror ghost makes zero;

against the shell's Robin heat on each segment, h L (T_gas - mean T at t+dt).

    python scripts/w336_thermal_seam_diag.py [--coarsen 4] [--warm 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import w336_episode_residuals as W  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coarsen", type=int, default=4)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--dt-macro", type=float, default=5.0e-3)
    ap.add_argument("--json", default=os.path.join(W.OUT, "thermal_seam_diag.json"))
    a = ap.parse_args(argv)
    M = W._mods()
    gen = M["gen"]
    ep, p = W._episode(M, 0, a.coarsen, a.dt_macro, a.warm + 2)
    for _ in range(a.warm):
        ep.step_macro()

    # capture the conduction W301 installs, separately from the face's total:
    # the one-sided term recomputed exactly as `_isothermal_wall_conduction`
    # computes it, and the face's final viscous energy flux beside it
    cls = M["C2"].Compressible2D
    NG = M["C2"].NG
    o_iwc = cls._isothermal_wall_conduction
    cap = {}
    active = {"w": None}

    def iwc(s, Fi, Fj, T, Qz, Qy, k):
        o_iwc(s, Fi, Fj, T, Qz, Qy, k)
        if active["w"] is None:
            return
        w = active["w"]
        blk = s.block
        nz, ny = blk.shape
        c = cap.setdefault(id(s), {sd: dict(total=np.zeros(nz), cond=np.zeros(nz))
                                   for sd in ("jmin", "jmax")})
        for side in ("jmin", "jmax"):
            bc = s.bcs.get(side)
            if bc is None or bc.kind != "wall_noslip" or "T_wall" not in bc.params:
                continue
            jc, f = (NG, 0) if side == "jmin" else (ny + NG - 1, -1)
            a_ = blk.a_j[:, f]
            vol = blk.vol[:, 0 if f == 0 else -1]
            T_i = T[NG:nz + NG, jc]
            T_w = np.asarray(bc.params["T_wall"], dtype=float)
            T_w = np.broadcast_to(T_w, T_i.shape) if T_w.ndim == 0 else T_w.reshape(T_i.shape)
            dn = 0.5 * vol / np.maximum(a_, 1e-30)
            grad = (T_i - T_w) / dn if f == 0 else (T_w - T_i) / dn
            wall = k[NG:nz + NG, jc] * grad * a_
            face = Fj[:, 0 if f == 0 else ny, 3]
            sign = -1.0 if side == "jmin" else 1.0         # viscous inflow convention
            c[side]["total"] += w * sign * face
            c[side]["cond"] += w * sign * wall
    cls._isothermal_wall_conduction = iwc

    o_step = cls.step

    def step(s, U, dt):
        prev = active["w"]
        active["w"] = 0.5 * dt
        try:
            return o_step(s, U, dt)
        finally:
            active["w"] = prev
    cls.step = step

    led = W.Ledger(M)
    led.install()
    led.reset()
    t0 = time.perf_counter()
    ep.step_macro()
    wall_s = time.perf_counter() - t0

    # the gas's side, per wall cell, with each cell's z extent
    out = dict(coarsen=a.coarsen, warm=a.warm, t=float(ep.t), wall_s=wall_s, sides={})
    for side in ("jmin", "jmax"):
        z_edges, parts = [], {"total_visc": [], "conduction": [], "inviscid": []}
        for ag in gen.ENGINE_AGENTS:
            s = ep.sol[ag][0]
            acc = led.acc_for(s)
            c = cap[id(s)][side]
            z = ep._wall_edges(ag, 0, side)
            z_edges.append(z if not z_edges else z[1:])
            parts["total_visc"].append(-acc["visc"][side][:, 3])
            parts["conduction"].append(-c["cond"])
            assert np.allclose(-c["total"], -acc["visc"][side][:, 3], rtol=1e-12, atol=1e-12)
            parts["inviscid"].append(-acc["inv"][side][:, 3])
        zE = np.concatenate(z_edges)
        gas = {k: np.concatenate(v) for k, v in parts.items()}
        # the shell's segments on this side, over the engine
        k = ep._panel(side)
        rec = led.shell[id(ep.shell[k])][-1]
        zs, _ = ep._shell_face(k, "inner")
        eng = rec["z_in"] > float(ep.geo.z_injector)
        # bin the gas's per-cell heat onto the shell's segments by z overlap
        lo = np.maximum(zE[None, :-1], zs[:-1, None])
        hi = np.minimum(zE[None, 1:], zs[1:, None])
        wgt = np.clip(hi - lo, 0.0, None) / np.maximum(np.diff(zE)[None, :], 1e-300)
        binned = {kk: wgt @ v for kk, v in gas.items()}
        shell_q = rec["q_in"]
        seg = []
        for i in np.flatnonzero(eng):
            seg.append(dict(z0=float(zs[i]), z1=float(zs[i + 1]), gas_total=float(binned["total_visc"][i]),
                            gas_conduction=float(binned["conduction"][i]),
                            shell=float(shell_q[i])))
        tot = {kk: float(v.sum()) for kk, v in gas.items()}
        # the face carries the one-sided conduction plus whatever else the
        # viscous energy flux averages onto a no-slip face: that remainder is work
        out["sides"][side] = dict(
            panel=k, gas_total_visc=tot["total_visc"], gas_inviscid=tot["inviscid"],
            gas_conduction=tot["conduction"], gas_work=tot["total_visc"] - tot["conduction"],
            shell_engine=float(shell_q[eng].sum()), segments=seg)
        print("%s: gas loses %.3f J/m (conduction %.3f, work %.3f, inviscid %.2e); shell gains %.3f"
              " -- shell/conduction %.5f, shell/total %.5f"
              % (side, tot["total_visc"], tot["conduction"], tot["total_visc"] - tot["conduction"],
                 tot["inviscid"], float(shell_q[eng].sum()),
                 float(shell_q[eng].sum()) / tot["conduction"], float(shell_q[eng].sum()) / tot["total_visc"]))
    W._persist(a.json, out)
    print("wrote", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
