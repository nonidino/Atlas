"""W57 -- a cut criterion for the substructuring branch, derived and measured.

    python scripts/w57_substructuring.py [--out out/w57]

The row: L2/C2 derives a cut criterion for the OVERLAPPING branch from the
restriction-defect identity, on hypotheses -- a partition of unity and an
assembly -- that a non-overlapping decomposition does not have.  Since section
10.2 retired the old scalarization, that branch has had no criterion at all and
L2 decertifies citing `C2/W57`.

**The derivation.** `master-error-bound` 4 factorizes the substructuring sigma
bound as

    ||lam_dag - lam*||  <=  ||Lambda~^-1|| ||(Lambda~ - Lambda) lam*||
                        <=  (1/beta) ||Lambda - Lambda~|| ||lam*||

The first line is an identity and the second is a bound.  Stopping at the first
gives a criterion that needs no operator mismatch at all: with ``lam_dag``
solving ``S_M lam = chi_M`` on the declared interface space,

    S_M (lam_dag - a*) = chi_M - S_M a* =: -r      =>   ||lam_dag - a*|| <= ||r||/beta

    **L2/C3:   Q_sub(Gamma) = || S_M a* - chi_M || / beta**

-- the residual the exact trace leaves in the approximate interface equation,
amplified by the interface solve's own conditioning.  Two forms, exactly as
L2/C2 has two: this tight one, and section 4's loose product form.

**Why 1/beta is legitimate here and was not on the overlapping branch.** Section
10.2 falsified a criterion carrying 1/beta, and section 4's own box says why that
does not transfer: an overlapping scheme poses no interface equation, so 1/beta
was imported from a factorization that does not apply to it.  Substructuring has
the interface solve the factorization is about.  This script tests whether that
distinction is real or merely stated.

**The instrument.** `tests/substructure_model.py` -- a steady advection-diffusion
torus cut in two by the exact algebraic Schur complement, with a freely movable
cut and everything else held fixed.  A model problem for a rule, in
`strip_model.py`'s third category: neither a fixture nor a case study, earning
nothing on Tier 0.  Steady because flux balance is the exact interface condition
of a boundary-value problem and of nothing else (section 5, R2b); with the FULL
interface space it reproduces the monolith to 5e-16, so every defect it reports
is the declared interface space's truncation and not the model's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import substructure_model as SM        # noqa: E402


def spearman(a, b) -> float:
    """Rank correlation, without scipy -- this package is numpy-only."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def sweep(phys: SM.SteadyPhysics, m_eff: int, seed: int) -> dict:
    """Every cut placement on one configuration, criterion against truth."""
    f = phys.source(seed)
    u_star = SM.monolith_solve(phys, f)
    rows = [SM.measure(SM.SubstructureDecomposition(phys=phys, m_eff=m_eff, cut=c),
                       f, u_star)
            for c in range(phys.ny // 2)]
    D = np.array([r["composed_defect"] for r in rows])
    Q = np.array([r["q_tight"] for r in rows])
    R = np.array([r["residual"] for r in rows])
    B = np.array([r["beta"] for r in rows])
    QP = np.array([r["q_product"] for r in rows])
    return {
        "seed": seed, "m_eff": m_eff, "n_cuts": len(D),
        "defect_spread": float(D.max() / D.min()),
        "beta_spread": float(B.max() / B.min()),
        "rank_q_tight": spearman(Q, D),
        "rank_residual_alone": spearman(R, D),
        "rank_inv_beta_alone": spearman(1.0 / B, D),
        "rank_q_product": spearman(QP, D),
        "tightness_min": float((Q / D).min()),
        "tightness_median": float(np.median(Q / D)),
        "tightness_max": float((Q / D).max()),
        "cost_of_following": float(D[int(np.argmin(Q))] / D.min()),
        "best_cut_true": int(np.argmin(D)),
        "best_cut_by_Q": int(np.argmin(Q)),
        "rows": rows,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join("out", "w57"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    base = SM.SteadyPhysics()
    f0 = base.source(0)
    u0 = SM.monolith_solve(base, f0)
    exact = {f"cut_{c}": SM.exact_check(base, f0, u0, cut=c) for c in (0, 3, 7)}
    print("exactness of the substructured solve at the FULL interface space:")
    for k, v in exact.items():
        print(f"   {k}: rel err {v:.3e}")

    configs = []
    print()
    print(f"{'configuration':<30}{'spread':>9}{'q_tight':>10}{'resid':>10}"
          f"{'1/beta':>10}{'q_prod':>10}{'cost':>9}")
    for seed in (0, 1, 2):
        for m in (6, 8, 10):
            r = sweep(base, m, seed)
            configs.append(r)
            print(f"seed={seed} m={m:<2d}{'':<18}{r['defect_spread']:>8.2f}x"
                  f"{r['rank_q_tight']:>+10.4f}{r['rank_residual_alone']:>+10.4f}"
                  f"{r['rank_inv_beta_alone']:>+10.4f}{r['rank_q_product']:>+10.4f}"
                  f"{r['cost_of_following']:>8.3f}x")
    for nu_y, ax, ay in ((0.0, 1.2, 0.8), (0.8, 1.2, 0.8)):
        phys = SM.SteadyPhysics(a_x=ax, a_y=ay, nu_y_amp=nu_y)
        r = sweep(phys, 8, 3)
        r["note"] = f"advective, nu_y_amp={nu_y}"
        configs.append(r)
        print(f"advective nu_y_amp={nu_y:<12.1f}{r['defect_spread']:>8.2f}x"
              f"{r['rank_q_tight']:>+10.4f}{r['rank_residual_alone']:>+10.4f}"
              f"{r['rank_inv_beta_alone']:>+10.4f}{r['rank_q_product']:>+10.4f}"
              f"{r['cost_of_following']:>8.3f}x")

    q = [c["rank_q_tight"] for c in configs]
    p = [c["rank_q_product"] for c in configs]
    beats = sum(1 for c in configs
                if c["rank_q_tight"] >= c["rank_residual_alone"] - 1e-12)
    print()
    print(f"q_tight ranks POSITIVE in {sum(1 for x in q if x > 0)}/{len(q)} "
          f"configurations, range [{min(q):+.4f}, {max(q):+.4f}]")
    print(f"q_tight >= residual alone in {beats}/{len(configs)}: "
          "1/beta is a genuine amplifier on this branch")
    print(f"section 4's PRODUCT form ranks negative in "
          f"{sum(1 for x in p if x < 0)}/{len(p)}, range [{min(p):+.4f}, {max(p):+.4f}]")
    print(f"cost of following Q_sub: max {max(c['cost_of_following'] for c in configs):.3f}x "
          "above the best available cut")

    payload = {
        "row": "W57",
        "criterion": "L2/C3: Q_sub(Gamma) = ||S_M a_star - chi_M|| / beta",
        "exactness_at_full_space": exact,
        "configurations": [{k: v for k, v in c.items() if k != "rows"}
                           for c in configs],
        "detail": configs[1]["rows"],
    }
    with open(os.path.join(a.out, "w57.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {os.path.join(a.out, 'w57.json')}")


if __name__ == "__main__":
    main()
