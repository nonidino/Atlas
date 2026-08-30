"""L6, part two: does the candidate condition separate schemes that WORK?

`l6_assembly_condition.py` measures the certificate's quantities on one sub-step.
This measures what they are supposed to predict: the composed macro-step's defect
against the monolith, under the same six partitions of unity, using the split-step
scheme `tier0_window_ns.py` reports.

The question the certificate has to answer is not "is this blend pretty" but
"does refusing it prevent a wrong answer".  A condition that refuses a scheme
which measures fine, or admits one that measures badly, is not a condition.

Run:  python scripts/l6_macro_consequence.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.cases import window_ns as W                             # noqa: E402
from l6_assembly_condition import (                                # noqa: E402
    PARTITIONS, assemble_with, chi_stats, defects, overlap_mask,
)
from tier0_window_ns import (                                      # noqa: E402
    _pin_domain, make_solvers, mono_step, rel_l2,
)


def composed_step_with(ws, tiling, solvers, u, v, dt, fx, ring,
                       substeps=W.SUBSTEPS, exact_ring=None):
    """`tier0_window_ns.composed_step` for scheme='split-step', with chi injected.

    Identical to the driver's version except that the assembly uses the supplied
    partition of unity rather than the tiling's own.
    """
    mono, _embedded, exposed = solvers
    fxs, zero = tiling.cut(fx), np.zeros((4, tiling.n, tiling.n))
    U, V = u.copy(), v.copy()
    hs = dt / substeps
    for m in range(substeps):
        us, vs = tiling.cut(U), tiling.cut(V)
        if exact_ring is not None:
            s = (m + 1) / substeps
            ru = (1 - s) * u + s * exact_ring[0]
            rv = (1 - s) * v + s * exact_ring[1]
            b0 = (tiling.cut(ru), tiling.cut(rv))
        else:
            b0 = (us, vs)
        u1, v1 = exposed.step_batch(us, vs, hs, bc0=b0, bc1=None, force=(fxs, zero))
        U = assemble_with(ws, tiling, u1)
        V = assemble_with(ws, tiling, v1)
        pu, pv = mono._project(U[None], V[None])
        U, V = _pin_domain(pu[0], pv[0], ring)
    return U, V


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default=os.path.join("out", "tier0_verify", "s0_state.npz"))
    ap.add_argument("--out", default=os.path.join("out", "l6"))
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.state)
    u, v, fx = d["u"], d["v"], d["fx"]
    tiling = W.DEFAULT_TILING
    solvers = make_solvers(tiling)
    mono = solvers[0]
    ring = (u.copy(), v.copy())
    dt = W.MACRO_DT

    u_ref, v_ref = mono_step(mono, u, v, dt, fx)
    mask = overlap_mask(tiling)

    print(f"tiling n={tiling.n} halo={tiling.halo} steps={a.steps}\n")
    print(f"{'partition':26s} {'chi_min':>9s} {'norm_A':>7s} {'hull_esc':>10s} "
          f"{'Theta':>9s} | {'tau (1 step)':>12s} {'total':>11s} {'sigma':>11s} "
          f"{'N=' + str(a.steps):>11s}")

    rows = {}
    for name, make in PARTITIONS.items():
        ws = make(tiling)
        st = chi_stats(ws, tiling)

        u_c, v_c = composed_step_with(ws, tiling, solvers, u, v, dt, fx, ring)
        u_t, v_t = composed_step_with(ws, tiling, solvers, u, v, dt, fx, ring,
                                      exact_ring=(u_ref, v_ref))
        total = rel_l2(u_c, u_ref, v_c, v_ref)
        tau = rel_l2(u_t, u_ref, v_t, v_ref)
        sigma = rel_l2(u_c, u_t, v_c, v_t)

        # a short rollout, both maps from the same state
        Uc, Vc, Um, Vm = u.copy(), v.copy(), u.copy(), v.copy()
        traj = []
        for _ in range(a.steps):
            Uc, Vc = composed_step_with(ws, tiling, solvers, Uc, Vc, dt, fx, ring)
            Um, Vm = mono_step(mono, Um, Vm, dt, fx)
            traj.append(rel_l2(Uc, Um, Vc, Vm))

        # certificate quantities from the same step's locals
        _mono, _emb, exposed = solvers
        us, vs = tiling.cut(u), tiling.cut(v)
        fxs, zero = tiling.cut(fx), np.zeros((4, tiling.n, tiling.n))
        cls = W._no_projection_class()
        monox = cls(nu=W.NU, length=W.MONO_L, n=W.MONO_N, cfl=0.4,
                    transmission="dirichlet")
        hs = dt / W.SUBSTEPS
        l1, _l2 = exposed.step_batch(us, vs, hs, bc0=(us, vs), bc1=None,
                                     force=(fxs, zero))
        rr, _rv = monox.step_batch(u[None], v[None], hs, bc0=None, bc1=None,
                                   force=(fx[None], np.zeros_like(fx)[None]))
        cert = defects(ws, tiling, l1, rr[0], mask)

        rows[name] = {"chi": st, "tau": tau, "total": total, "sigma": sigma,
                      "rollout": traj, "certificate": cert}
        print(f"{name:26s} {st['chi_min']:+9.4f} {cert['norm_A_assembly']:7.4f} "
              f"{cert['hull_escape_max']:10.3e} {cert['Theta']:9.3e} | "
              f"{tau:12.4e} {total:11.4e} {sigma:11.4e} {traj[-1]:11.4e}")

    with open(os.path.join(a.out, "l6_macro.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=float)
    print(f"\nwrote {os.path.join(a.out, 'l6_macro.json')}")


if __name__ == "__main__":
    main()
