"""T3's finding, run through the workbench's own Schwarz driver.

The formal-proofs plan stated T3 as "every fixed point of restricted additive
Schwarz built from exact local solves, with a partition of unity, solves
A u = f".  That is false without a further hypothesis, and this script shows
it on the driver the workbench runs (`atlas.workbench.styles.schwarz`), which
is only imported here, never changed.

**Example 1: three unknowns, an indefinite matrix** (the one checked in Lean,
`AtlasProofs/SchwarzCounterexample.lean`).  Two windows {0, 1} and {1, 2}; the
shared unknown belongs entirely to the first window:

        [1 1 0]
    A = [1 0 1],   b = 0,   u = (1, -1, -1).
        [0 1 1]

A is invertible, so the only solution of A u = 0 is u = 0.  Both windows'
matrices are invertible, so both window solves are exact.  The driver, started
at u, returns u after one sweep with `converged=True` and an update of exactly
zero.  The residual b - A u is (0, 0, 2).

Three controls, each expected to behave the OTHER way:

  * the same A, started from the solution 0: stays at 0 (consistency);
  * a symmetric positive definite A with the same windows and weights: started
    from the same u, the driver leaves u and goes to the solution;
  * the same indefinite A with NO overlap (windows {0, 1} and {2}): block
    Jacobi's preconditioner is invertible, so the sweep moves u.

**Example 2: six unknowns, a SYMMETRIC POSITIVE DEFINITE matrix.**  Positive
definiteness does not rescue the statement once there are three windows.  The
matrix below has rational entries and its six leading principal minors are
positive; the windows are {0,1,2}, {1,2,3,4}, {3,4,5}; each unknown is owned by
exactly one window (0/1 weights).  In EXACT rational arithmetic (sympy) the
blended sum of the window solves is singular, and the vector u computed here is
a fixed point of the sweep with b = 0 although A u is not 0.  The ownership is
unusual (the first window keeps unknown 2 and gives away unknown 1), and the
sweep does not converge on this problem (its error propagation has an
eigenvalue of modulus 1.148), so the example refutes the STATEMENT, not the
practice: a sweep that converges from every start has no such fixed point
(`AtlasProofs/SchwarzConvergence.lean`).

Writes out/lean/t3_counterexample.json.

    python scripts/lean_t3_counterexample.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import sympy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from atlas.workbench.fv import LocalSystem  # noqa: E402
from atlas.workbench.styles import Factor, schwarz  # noqa: E402

OUT = os.path.join(REPO, "out", "lean")


def window(A: np.ndarray, b: np.ndarray, idx: list[int]) -> LocalSystem:
    """The window on `idx`: its own block of A, and the coupling C to the outside."""
    n = A.shape[0]
    idx = np.asarray(idx, dtype=np.int64)
    inside = np.zeros(n, dtype=bool)
    inside[idx] = True
    C = A[idx].copy()
    C[:, inside] = 0.0
    return LocalSystem(idx=idx, A=sp.csr_matrix(A[np.ix_(idx, idx)]), b=b[idx].copy(),
                       C=sp.csr_matrix(C), face_rows=np.zeros(0, dtype=np.int64),
                       face_outside=np.zeros(0, dtype=np.int64), face_g=np.zeros(0),
                       face_mode="neighbour")


def precond(A: np.ndarray, windows: list[list[int]], chi: list[np.ndarray]) -> np.ndarray:
    """M = sum_i R_i^T chi_i A_i^{-1} R_i, assembled densely."""
    n = A.shape[0]
    M = np.zeros((n, n))
    for idx, w in zip(windows, chi):
        ix = np.ix_(idx, idx)
        M[ix] += np.diag(w) @ np.linalg.inv(A[ix])
    return M


def run(name: str, A: np.ndarray, windows, chi, u0: np.ndarray, max_it: int = 500,
        tol: float = 1e-12, scale: float = 1.0) -> dict:
    n = A.shape[0]
    b = np.zeros(n)
    pou = np.zeros(n)
    for idx, w in zip(windows, chi):
        pou[np.asarray(idx)] += w
    systems = [window(A, b, idx) for idx in windows]
    factors = [Factor(s.A) for s in systems]
    it = schwarz(systems, factors, chi, u0, tol=tol, max_it=max_it, scale=scale)
    M = precond(A, windows, chi)
    T = np.eye(n) - M @ A
    return {
        "name": name,
        "A": A.tolist(),
        "windows": [list(map(int, w)) for w in windows],
        "weights": [w.tolist() for w in chi],
        "partition_of_unity_defect": float(np.max(np.abs(pou - 1.0))),
        "det_A": float(np.linalg.det(A)),
        "det_windows": [float(np.linalg.det(A[np.ix_(w, w)])) for w in windows],
        "start": u0.tolist(),
        "returned": it.u.tolist(),
        "converged": bool(it.converged),
        "sweeps": int(it.iterations),
        "last_update": float(it.history[-1]),
        "residual_of_returned": (b - A @ it.u).tolist(),
        "error_of_returned": float(np.max(np.abs(it.u))),     # the solution is 0
        "det_preconditioner": float(np.linalg.det(M)),
        "eig_modulus_of_error_propagation": sorted(float(abs(z)) for z in np.linalg.eigvals(T)),
    }


def spd_example() -> dict:
    """Example 2, in exact rational arithmetic, then through the driver."""
    R = sympy.Rational
    p, t, s, q = R(1, 2), R(5, 12), R(3, 4), R(1, 4)
    A = sympy.Matrix([
        [1, q, 0, 0, 0, 0],
        [q, 1, p, 0, 0, 0],
        [0, p, 1, t, s, 0],
        [0, 0, t, 1, s, 0],
        [0, 0, s, s, 1, q],
        [0, 0, 0, 0, q, 1],
    ])
    n = 6
    windows = [[0, 1, 2], [1, 2, 3, 4], [3, 4, 5]]
    owner = {0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 2}
    minors = [A[:k, :k].det() for k in range(1, n + 1)]
    M = sympy.zeros(n, n)
    window_dets = []
    for i, idx in enumerate(windows):
        Ai = A.extract(idx, idx)
        window_dets.append(Ai.det())
        Bi = Ai.inv()
        for a, ga in enumerate(idx):
            if owner[ga] != i:
                continue
            for c, gc in enumerate(idx):
                M[ga, gc] += Bi[a, c]
    owned_once = [sum(1 for i, idx in enumerate(windows) if j in idx and owner[j] == i)
                  for j in range(n)]
    ker = M.nullspace()
    r = ker[0] * sympy.ilcm(*[sympy.fraction(x)[1] for x in ker[0]])
    u = A.LUsolve(-r)                       # b = 0, so b - A u = r
    T = sympy.eye(n) - M * A
    fixed = (T * u - u) == sympy.zeros(n, 1)

    exact = {
        "A": [[str(x) for x in A.row(i)] for i in range(n)],
        "symmetric": bool(A == A.T),
        "leading_principal_minors": [str(m) for m in minors],
        "positive_definite": bool(A == A.T and all(m > 0 for m in minors)),
        "windows": windows,
        "owner_of_each_unknown": [owner[j] for j in range(n)],
        "each_unknown_owned_once": owned_once,
        "window_determinants": [str(d) for d in window_dets],
        "det_preconditioner": str(M.det()),
        "kernel_dimension": len(ker),
        "residual_sent_to_zero": [str(x) for x in r],
        "u": [str(x) for x in u],
        "A_u": [str(x) for x in (A * u)],
        "sweep_leaves_u_fixed": bool(fixed),
    }

    # the same example through the workbench's driver, in floating point
    Af = np.array(A.tolist(), dtype=float)
    uf = np.array([float(x) for x in u])
    chi = [np.array([1.0 if owner[j] == i else 0.0 for j in idx]) for i, idx in enumerate(windows)]
    scale = float(np.max(np.abs(uf)))
    driver = run("example 2: positive definite A, three windows", Af, windows, chi, uf,
                 tol=1e-9, scale=scale)
    return {"exact": exact, "driver": driver, "scale": scale}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    u = np.array([1.0, -1.0, -1.0])
    A_bad = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    A_spd = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
    overlap = [[0, 1], [1, 2]]
    chi = [np.array([1.0, 1.0]), np.array([0.0, 1.0])]
    runs = [
        run("example 1: indefinite A, overlapping windows", A_bad, overlap, chi, u),
        run("control 1: the same A, started at the solution", A_bad, overlap, chi, np.zeros(3)),
        run("control 2: positive definite A, the same windows and weights", A_spd, overlap,
            chi, u),
        # ONE sweep: the question is only whether the sweep moves u.  (Without
        # overlap this A's block Jacobi does not converge either: its error
        # propagation has eigenvalues of modulus one.)
        run("control 3: the same indefinite A, no overlap, one sweep", A_bad, [[0, 1], [2]],
            [np.array([1.0, 1.0]), np.array([1.0])], u, max_it=1),
    ]
    spd = spd_example()
    c, e, d = runs[0], spd["exact"], spd["driver"]
    verdict = {
        "example_1_spurious_fixed_point": bool(
            c["converged"] and c["last_update"] == 0.0 and c["error_of_returned"] == 1.0
            and max(abs(x) for x in c["residual_of_returned"]) == 2.0),
        "control_1_stays_at_solution": bool(runs[1]["converged"]
                                            and runs[1]["error_of_returned"] == 0.0),
        "control_2_reaches_solution": bool(runs[2]["converged"]
                                           and runs[2]["error_of_returned"] < 1e-9),
        "control_3_not_a_fixed_point": bool(runs[3]["last_update"] > 0.0
                                            and abs(runs[3]["det_preconditioner"]) > 0.5),
        "example_2_positive_definite_exact": bool(
            e["positive_definite"] and e["det_preconditioner"] == "0"
            and e["kernel_dimension"] == 1 and e["sweep_leaves_u_fixed"]
            and e["each_unknown_owned_once"] == [1] * 6
            and any(x != "0" for x in e["A_u"])),
        "example_2_driver_reports_converged": bool(
            d["converged"] and d["sweeps"] == 1
            and d["error_of_returned"] > 0.99 * spd["scale"]),
    }

    with open(os.path.join(OUT, "t3_counterexample.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump({"runs": runs, "positive_definite_example": spd, "verdict": verdict}, fh,
                  indent=1)
        fh.write("\n")
    for r in runs + [d]:
        print(r["name"])
        print("   returned %s" % np.round(r["returned"], 6).tolist())
        print("   converged %s in %d sweep(s), last update %.3g"
              % (r["converged"], r["sweeps"], r["last_update"]))
        print("   residual b - A u = %s" % np.round(r["residual_of_returned"], 6).tolist())
        print("   det M = %.3g   |eig(I - M A)| = %s"
              % (r["det_preconditioner"],
                 ["%.3f" % z for z in r["eig_modulus_of_error_propagation"]]))
    print()
    print("example 2, exact: minors %s" % e["leading_principal_minors"])
    print("   det M = %s, kernel dimension %d, r = %s"
          % (e["det_preconditioner"], e["kernel_dimension"], e["residual_sent_to_zero"]))
    print("   u = %s" % e["u"])
    print("   A u = %s" % e["A_u"])
    print()
    for k, v in verdict.items():
        print("%-38s %s" % (k, v))
    return 0 if all(v is True for v in verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
