"""The two-dimensional dual certificate ([[chart-operator-family-catalog]] section 5).

On a simply connected 2-D piece a divergence-free flux is the rotated gradient of a
stream function, ``tau = W grad psi``, and its complementary energy with coefficient
``A`` is the Dirichlet energy of ``psi`` with ``A / det A`` (Prager and Synge, *Quart.
Appl. Math.* 5, 1947).  For the pulled-back ``A = kappa_hat J G^-1`` that is
``kappa_hat^-1 J G^-1``: **the same expert with log kappa_hat negated** (v2's addition
(b)).  For any trace ``g`` and any stream function ``psi`` with trace in the dual ports,

    E(g) >= 2 |<psi, g>| - psi^T M* psi,   <psi, g> = loop integral of g d(psi),

so the dual's modes give ``Lambda_low = Y^T G*^-1 Y`` with ``G* = Psi^T M* Psi`` and
``Y = Q*^T T Q`` (``T`` the exact loop pairing of piecewise-linear traces, from
`scripts/arch_family_checks.py`).  With bilinear elements on both sides (conforming,
decision D7) and any network outputs,

    Lambda~_low <= Lambda_low <= E <= Lambda <= Lambda~,

the outer two by T25 (ii) applied to each expert (and inversion reversing the Loewner
order).  Steady pieces only: a reaction term breaks ``div tau = 0``.  The dual ports
are the primal basis with the constant direction removed (a constant stream function
carries no flux and has no energy).
"""

from __future__ import annotations

import numpy as np

from . import fe
from .piece import PieceProblem


def loop_pairing(nb: int) -> np.ndarray:
    """T with psi^T T g = sum over boundary segments of (psi_{i+1} - psi_i)(g_i + g_{i+1}) / 2,
    the exact loop integral of g d(psi) for piecewise-linear traces."""
    T = np.zeros((nb, nb))
    for i in range(nb):
        j = (i + 1) % nb
        T[j, i] += 0.5
        T[j, j] += 0.5
        T[i, i] -= 0.5
        T[i, j] -= 0.5
    return T


def constant_free(pb) -> np.ndarray:
    """P (4m x 4m-1): an orthonormal basis of the coefficient vectors orthogonal to the
    constant's (the sum of the corner hats, or of the sides' zeroth cosines)."""
    e = np.zeros(pb.size)
    for k, c in enumerate(pb.columns):
        if c.what == "vertex" or (pb.kind == "cosine" and c.j == 0):
            e[k] = 1.0
    e /= np.linalg.norm(e)
    U, _s, _vt = np.linalg.svd(np.eye(pb.size) - np.outer(e, e))
    return U[:, :pb.size - 1]


def dual_problem(p: PieceProblem) -> PieceProblem:
    if not p.steady:
        raise ValueError("the dual bracket holds for steady pieces only")
    return PieceProblem(p.chart, log_kappa_hat=-p.log_kappa_hat,
                        log_kappa_hat_e=-p.log_kappa_hat_e, fo=None,
                        g=np.zeros_like(p.g), kappa_ref=1.0 / p.kappa_ref, label=p.label + " (dual)")


def _mineig(A: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(0.5 * (A + A.T))[0])


def bracket(p: PieceProblem, pb, H_learned: np.ndarray, H_dual_learned: np.ndarray,
            traces: list | None = None) -> dict:
    """The chain on one steady piece, dimensionless: the smallest eigenvalue of each
    difference relative to ||Lambda||, and for each coefficient vector in ``traces``
    the energy, the network's error c^T (Lambda~ - Lambda) c, the computable gap
    c^T (Lambda~ - Lambda~_low) c and the discrete bracket's width c^T (Lambda - Lambda_low) c."""
    n = p.n
    Q = pb.Q
    M = p.matrix()
    ex = fe.Exact(M, n)
    H = ex.modes(Q)
    Lam = ex.port_matrix(Q, H)
    Lt = fe.gram(M, fe.full_field(n, H_learned, Q))
    d = dual_problem(p)
    Ms = d.matrix()
    P = constant_free(pb)
    Qs = Q @ P
    exs = fe.Exact(Ms, n)
    Hs = exs.modes(Qs)
    Gs = fe.gram(Ms, fe.full_field(n, Hs, Qs))
    Gst = fe.gram(Ms, fe.full_field(n, H_dual_learned @ P, Qs))
    Y = Qs.T @ loop_pairing(4 * n) @ Q
    Llow = Y.T @ np.linalg.solve(Gs, Y)
    Ltlow = Y.T @ np.linalg.solve(Gst, Y)
    nrm = float(np.max(np.abs(np.linalg.eigvalsh(Lam))))
    out = {"min_eig_rel": {"Lt-L": _mineig(Lt - Lam) / nrm, "L-Llow": _mineig(Lam - Llow) / nrm,
                           "Llow-Ltlow": _mineig(Llow - Ltlow) / nrm},
           "data": []}
    for c in traces or []:
        c = np.asarray(c, float)
        e = float(c @ Lam @ c)
        out["data"].append({"energy": e, "network_error": float(c @ (Lt - Lam) @ c),
                            "computable_gap": float(c @ (Lt - Ltlow) @ c),
                            "discrete_bracket": float(c @ (Lam - Llow) @ c)})
    return out
