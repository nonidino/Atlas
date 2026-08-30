"""Every Tier 0 number, at four configurations instead of one.

`tier0-measurements` section 7 lists "one configuration, one state, one expert"
as the standing caveat on every constant in it, and section 8 did not remove it:
the remediation moved the numbers but not the scope.  This varies the two knobs
that can be varied without a second expert -- the Reynolds number and the
macro-step -- and re-measures the quantities each claim rests on.

For each (nu, dt):

  base    develop to t = 5 on the monolith, record the steady residual
  W2      the probed seam operator: beta, kappa, mu, null dim, asymmetry
  W3      the three-way split for both schemes, and the improvement factor
  L6/C1   chi_min and the convexity verdict
  W49     Pi, ||d_lambda||, and the implied C_mu against the measured sigma
  W1      the fitted L for the monolith and for the split-step composition

The question is not whether the numbers move -- they will -- but whether the
CLAIMS survive: L6/C1 holds, R10's ordering holds, L matches the monolith, the
W49 factorization keeps one constant.

Run:  python scripts/tier0_sweep.py [--with-L]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atlas import ProbeBudget                                      # noqa: E402
from atlas.assembly import sigma_halo_bound                        # noqa: E402
from atlas.cases import window_ns as W                             # noqa: E402
from atlas.probe import assemble_seam                              # noqa: E402
from l6_assembly_condition import pou_ramp                         # noqa: E402
from tier0_window_ns import (                                      # noqa: E402
    _fit_L, body_force, composed_step, make_solvers, mono_step,
    probe_all_seams, rel_l2, streamfunction_blob,
)
from w49_sigma_halo import paired_step                             # noqa: E402

CONFIGS = [
    ("baseline", 1.0 / 255.0, 0.05),
    ("Re x2", 1.0 / 510.0, 0.05),
    ("Re /2", 1.0 / 128.0, 0.05),
    ("dt /2", 1.0 / 255.0, 0.025),
    ("dt x2", 1.0 / 255.0, 0.10),
]


def develop(nu, dt, t_dev=5.0):
    mono, _e, _x = make_solvers(W.DEFAULT_TILING, nu)
    fx = body_force()
    u = np.full((W.MONO_N, W.MONO_N), W.U_INF)
    v = np.zeros_like(u)
    n_steps = int(round(t_dev / dt))
    res = 0.0
    for _ in range(n_steps):
        up, vp = u, v
        u, v = mono_step(mono, u, v, dt, fx)
        res = rel_l2(u, up, v, vp) / dt
    return u, v, fx, res


def run(name, nu, dt, with_L, out_dir):
    t0 = time.perf_counter()
    tiling = W.DEFAULT_TILING
    u, v, fx = None, None, None
    u, v, fx, steady = develop(nu, dt)
    solvers = make_solvers(tiling, nu)
    mono = solvers[0]
    ring = (u.copy(), v.copy())
    row = {"nu": nu, "dt": dt, "Re": 1.0 / nu, "steady_residual": steady}

    # --- W2: the probed operator, split-step agent ------------------------
    graph, _experts = W.build(u, v, mode="split-step", tiling=tiling, dt=dt, nu=nu)
    ops = probe_all_seams(graph, ProbeBudget(), f"{name}: nu={nu:.4g} dt={dt}")
    op, _tr = ops["sx0"]
    row["W2"] = {"beta": op.beta, "kappa": op.kappa, "null_dim": op.null_dim,
                 "mu": op.passivity_lambda_min, "pi": op.passivity_defect,
                 "asymmetry": float(np.linalg.norm(op.S - op.S.T)
                                    / max(np.linalg.norm(op.S), 1e-300))}

    # --- W3: the three-way split, both schemes ----------------------------
    u_ref, v_ref = mono_step(mono, u, v, dt, fx)
    row["W3"] = {}
    for scheme in ("as-built", "split-step"):
        u_c, v_c = composed_step(scheme, tiling, solvers, u, v, dt, fx, ring)
        u_t, v_t = composed_step(scheme, tiling, solvers, u, v, dt, fx, ring,
                                 exact_ring=(u_ref, v_ref))
        row["W3"][scheme] = {
            "total": rel_l2(u_c, u_ref, v_c, v_ref),
            "tau": rel_l2(u_t, u_ref, v_t, v_ref),
            "sigma": rel_l2(u_c, u_t, v_c, v_t),
        }
    row["W3"]["improvement"] = (row["W3"]["as-built"]["total"]
                                / max(row["W3"]["split-step"]["total"], 1e-300))

    # --- L6/C1 and W49 ----------------------------------------------------
    pou = tiling.partition_of_unity()
    pi = pou.contaminated_weight()
    m = paired_step(pou_ramp(tiling, tiling.ramp), tiling, solvers, u, v, dt, fx, ring)
    bound = sigma_halo_bound(pou, m["dlambda_rel"])
    row["L6_C1"] = {"chi_min": pou.chi_min(), "convex": pou.chi_min() >= -1e-12,
                    "identity_residual": pou.identity_residual(),
                    "norm_A": pou.norm_A()}
    row["W49"] = {"Pi": pi, "dlambda": m["dlambda_rel"], "sigma": m["sigma"],
                  "bound": bound, "holds": bound >= m["sigma"],
                  "C_mu_implied": m["sigma"] / (pi * m["dlambda_rel"])}

    # --- W1: the fitted L -------------------------------------------------
    if with_L:
        du, dv = streamfunction_blob(1e-3)
        row["W1"] = {}
        for scheme in ("monolith", "split-step"):
            errs = []
            A = (u.copy(), v.copy())
            B = (u + du, v + dv)
            for _ in range(40):
                if scheme == "monolith":
                    A = mono_step(mono, A[0], A[1], dt, fx)
                    B = mono_step(mono, B[0], B[1], dt, fx)
                else:
                    A = composed_step("split-step", tiling, solvers, A[0], A[1],
                                      dt, fx, ring)
                    B = composed_step("split-step", tiling, solvers, B[0], B[1],
                                      dt, fx, ring)
                errs.append(rel_l2(B[0], A[0], B[1], A[1]))
            f = _fit_L(errs, dt)
            row["W1"][scheme] = {"L": f["L"], "stderr": f["L_stderr"],
                                 "eta": f["eta"]}
        d = abs(row["W1"]["split-step"]["L"] - row["W1"]["monolith"]["L"])
        row["W1"]["gap"] = d
        row["W1"]["gap_in_stderr"] = d / max(row["W1"]["monolith"]["stderr"], 1e-300)

    row["seconds"] = time.perf_counter() - t0
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join("out", "sweep"))
    ap.add_argument("--with-L", action="store_true", default=True)
    ap.add_argument("--no-L", dest="with_L", action="store_false")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    rows = {}
    for name, nu, dt in CONFIGS:
        print(f"\n===== {name}: nu = {nu:.5g} (Re = {1/nu:.0f}), dt = {dt} =====",
              flush=True)
        r = run(name, nu, dt, a.with_L, a.out)
        rows[name] = r
        w2, w3 = r["W2"], r["W3"]
        print(f"  W2   beta={w2['beta']:.4e} kappa={w2['kappa']:.3f} "
              f"null={w2['null_dim']} mu={w2['mu']:+.4e} asym={w2['asymmetry']:.4f}")
        print(f"  W3   as-built total={w3['as-built']['total']:.4e}  "
              f"split-step total={w3['split-step']['total']:.4e}  "
              f"({w3['improvement']:.1f}x)")
        print(f"  L6C1 chi_min={r['L6_C1']['chi_min']:.4e} "
              f"convex={r['L6_C1']['convex']} normA={r['L6_C1']['norm_A']:.4f}")
        print(f"  W49  Pi={r['W49']['Pi']:.4e} sigma={r['W49']['sigma']:.4e} "
              f"bound={r['W49']['bound']:.4e} holds={r['W49']['holds']} "
              f"C_mu={r['W49']['C_mu_implied']:.4f}")
        if a.with_L:
            w1 = r["W1"]
            print(f"  W1   monolith L={w1['monolith']['L']:.6f} "
                  f"split-step L={w1['split-step']['L']:.6f}  "
                  f"gap={w1['gap']:.2e} ({w1['gap_in_stderr']:.2f} stderr)")
        print(f"  ({r['seconds']:.0f} s)")

    with open(os.path.join(a.out, "tier0_sweep.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=float)
    print(f"\nwrote {os.path.join(a.out, 'tier0_sweep.json')}")

    print("\n--- do the CLAIMS survive? ---")
    print(f"L6/C1 convex everywhere:        "
          f"{all(r['L6_C1']['convex'] for r in rows.values())}")
    print(f"W49 bound holds everywhere:     "
          f"{all(r['W49']['holds'] for r in rows.values())}")
    cm = [r["W49"]["C_mu_implied"] for r in rows.values()]
    print(f"C_mu implied range:             [{min(cm):.4f}, {max(cm):.4f}] "
          f"spread {max(cm)/min(cm):.2f}x")
    print(f"split-step beats as-built:      "
          f"{all(r['W3']['improvement'] > 1 for r in rows.values())} "
          f"(factors {[round(r['W3']['improvement']) for r in rows.values()]})")
    print(f"null dim 0 everywhere:          "
          f"{all(r['W2']['null_dim'] == 0 for r in rows.values())}")
    print(f"mu > 0 everywhere (passive):    "
          f"{all(r['W2']['mu'] > 0 for r in rows.values())}")
    if a.with_L:
        print(f"L matches monolith within 3 se: "
              f"{all(r['W1']['gap_in_stderr'] < 3 for r in rows.values())} "
              f"(worst {max(r['W1']['gap_in_stderr'] for r in rows.values()):.2f})")


if __name__ == "__main__":
    main()
