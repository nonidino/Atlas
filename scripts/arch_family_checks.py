"""arch_family_checks -- four checks behind the chart-operator family catalog (2026-10-10).

Classical solves stand in for networks, as in `arch_superelement_gram.py`. The discretisation
is bilinear (Q1) finite elements on the reference square with an element-wise constant
coefficient tensor A = kappa J G^{-1}, built from a Beltrami field mu and a conductivity field
kappa exactly as in the paper's pull-back. Q1 stands in for the paper's face-hybridised finite
volumes because a two-point flux cannot carry the off-diagonal entries of a full tensor; every
element integral below is computed exactly (2 x 2 Gauss is exact for these integrands).

S2. **The fast reference solve as a preconditioner.** Conjugate gradients on the pulled-back
    operator, preconditioned by the plain-square Laplacian applied with a fast sine transform.
    Derived: b/a <= (kappa_max/kappa_min) K_max^2 with K = (1+|mu|)/(1-|mu|), so the
    iteration count is at most ceil(sqrt(b/a)/2 * ln(2/tol)) on every grid. Measured: the
    count at n = 32..256 against that bound, the actual generalised eigenvalue range at
    n = 32, and the wall time against a sparse direct solve (a ratio, this container).
S7. **Decay of port modes into the square.** Reference constraint modes of a trace of
    wavenumber k on one side (steady Laplacian; implicit steps at Fo = 1e-2 and 1e-3; a
    constant anisotropic tensor diag(1/K, K) with K = 3). Measured: the depth beyond which a
    mode stays below tol times its trace, against ln(1/tol)/rate; the fraction of the square
    a strip-cropped mode head would cover for 16 modes per side; the energy lost by cropping.
S9. **Precision of the mode fields.** Exact modes of a distorted piece, the plain-square
    principal part, and the correction rounded to bfloat16, float16 or float32 (principal
    part and traces kept in float64), with the Gram matrix formed in float64. Against: the
    whole field rounded, and the Gram formed in float32.
D.  **The two-dimensional dual bracket.** In 2-D a divergence-free flux is the rotated
    gradient of a stream function psi, and the complementary energy with coefficient A is
    the Dirichlet energy of psi with coefficient A/det A (for the pull-back, kappa -> 1/kappa
    with the same mu). So the same constraint-mode machinery, run on A/det A, gives a
    *lower* bound on the exact port matrix, while the learned primal Gram is an upper bound
    (the paper's Theorem 5). Measured: the Loewner chain
        L~_low <= L_low(m dual ports) <= L_low(all nodes) <= L_low(fine) <= L_fine <= L <= L~
    for three dual port sets (63 loop-Fourier; corner hats plus side sines, 63, 67, 127), the
    largest relative gap over all port data and the gap for smooth port data, and, with both
    experts perturbed, the computable gap L~ - L~_low against the true error. The continuum
    energy is estimated by the midpoint of the fine bracket, whose width is reported.

Writes ``out/arch/family_checks.txt`` and ``.json``. Pure numpy/scipy; under a minute.

    set PYTHONIOENCODING=utf-8
    python scripts/arch_family_checks.py
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import scipy.fft as sfft
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out", "arch")
SEED = 11


# ---------------------------------------------------------------------------
# Q1 finite elements on the unit square
# ---------------------------------------------------------------------------


def _q1_reference():
    """Element matrices on the unit cell, nodes (0,0), (1,0), (1,1), (0,1). The stiffness parts
    are scale-free in 2-D; the mass part is multiplied by h^2 at assembly."""
    gp = (np.array([-1.0, 1.0]) / math.sqrt(3.0) + 1.0) / 2.0
    kxx = np.zeros((4, 4)); kyy = np.zeros((4, 4)); kxy = np.zeros((4, 4)); ms = np.zeros((4, 4))
    for x in gp:
        for y in gp:
            phi = np.array([(1 - x) * (1 - y), x * (1 - y), x * y, (1 - x) * y])
            dx = np.array([-(1 - y), 1 - y, y, -y])
            dy = np.array([-(1 - x), -x, x, 1 - x])
            kxx += 0.25 * np.outer(dx, dx)
            kyy += 0.25 * np.outer(dy, dy)
            kxy += 0.25 * (np.outer(dx, dy) + np.outer(dy, dx))
            ms += 0.25 * np.outer(phi, phi)
    return kxx, kyy, kxy, ms


Q1 = _q1_reference()


def assemble(n, a11, a12, a22, mass=None):
    """Global Q1 matrix on an n x n element grid; a11, a12, a22 (and mass, a reaction
    coefficient) are element arrays indexed [j, i]. Node (i, j) has index j (n+1) + i."""
    kxx, kyy, kxy, ms = Q1
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    n1 = n + 1
    nodes = np.stack([j * n1 + i, j * n1 + i + 1, (j + 1) * n1 + i + 1, (j + 1) * n1 + i],
                     axis=-1).reshape(-1, 4)
    ke = (a11.reshape(-1)[:, None, None] * kxx + a22.reshape(-1)[:, None, None] * kyy
          + a12.reshape(-1)[:, None, None] * kxy)
    if mass is not None:
        ke = ke + (np.broadcast_to(mass, (n, n)).reshape(-1)[:, None, None] / n ** 2) * ms
    rows = np.repeat(nodes, 4, axis=1).ravel()
    cols = np.tile(nodes, (1, 4)).ravel()
    return sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(n1 * n1, n1 * n1))


def boundary_loop(n):
    """The 4n boundary nodes, counter-clockwise from (0, 0); arc length k/n, perimeter 4."""
    n1 = n + 1
    bottom = [i for i in range(n)]
    right = [j * n1 + n for j in range(n)]
    top = [n * n1 + i for i in range(n, 0, -1)]
    left = [j * n1 for j in range(n, 0, -1)]
    return np.array(bottom + right + top + left)


def split(n):
    bnd = boundary_loop(n)
    mask = np.ones((n + 1) ** 2, bool)
    mask[bnd] = False
    return bnd, np.flatnonzero(mask)


def loop_fourier(n, m=64):
    """m orthonormal Fourier functions of arc length around the boundary loop (continuous at
    the corners, unlike per-side cosines): 1, cos, sin, ..., up to frequency m/2."""
    s = np.arange(4 * n) / n
    cols = [np.ones(4 * n)]
    f = 1
    while len(cols) < m:
        cols.append(np.cos(2 * math.pi * f * s / 4.0))
        if len(cols) < m:
            cols.append(np.sin(2 * math.pi * f * s / 4.0))
        f += 1
    Q, _ = np.linalg.qr(np.column_stack(cols))
    return Q * np.sign(np.diag(_))[None, :]


def smooth(n, rng, kmax=5):
    s = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(s, s)
    f = np.zeros((n, n))
    for a in range(kmax + 1):
        for b in range(kmax + 1):
            w = math.exp(-(a * a + b * b) / 6.0)
            f += (w * rng.standard_normal() * np.cos(a * math.pi * X + rng.uniform(0, 2 * math.pi))
                  * np.cos(b * math.pi * Y + rng.uniform(0, 2 * math.pi)))
    return f


def pulled_back(n, k_mu, contrast, rng):
    """Element tensor A = kappa J G^{-1} for a smooth Beltrami field with sup |mu| = k_mu and a
    smooth conductivity with max/min = contrast."""
    re, im = smooth(n, rng), smooth(n, rng)
    mu = (re + 1j * im) * (k_mu / np.abs(re + 1j * im).max())
    lk = smooth(n, rng)
    kap = np.exp((lk - lk.min()) / (lk.max() - lk.min()) * math.log(contrast))
    den = 1.0 - np.abs(mu) ** 2
    return (kap * np.abs(1 - mu) ** 2 / den, kap * (-2.0 * mu.imag) / den,
            kap * np.abs(1 + mu) ** 2 / den, kap, mu)


def refine(a, r):
    return np.kron(a, np.ones((r, r)))


def bubble_perturbation(n, k, rng, interior):
    """k smooth random interior fields that vanish on the boundary (nodal, Q1)."""
    s = np.arange(n + 1) / n
    X, Y = np.meshgrid(s, s)
    beta = 16 * X * (1 - X) * Y * (1 - Y)
    out = np.empty((interior.size, k))
    for c in range(k):
        f = smooth(n + 1, rng)
        out[:, c] = (beta * f).ravel()[interior]
    return out


def modes(A, bnd, inner, Q):
    """Exact constraint modes H = -A_II^{-1} A_IB Q and the LU of A_II."""
    lu = spla.splu(sp.csc_matrix(A[inner][:, inner]))
    H = -lu.solve(np.asarray(A[inner][:, bnd] @ Q))
    return H, lu


def gram(A, bnd, inner, H, Q):
    """Lambda = V^T A V, V = (H on the interior, Q on the boundary)."""
    V = np.zeros((A.shape[0], Q.shape[1]))
    V[inner] = H
    V[bnd] = Q
    L = V.T @ (A @ V)
    return 0.5 * (L + L.T)


def scaled_errors(A, inner, H, E, eps):
    """Scale each perturbation column to energy eps times the median mode energy."""
    AII = A[inner][:, inner]
    eh = np.sqrt(np.maximum(np.einsum("ik,ik->k", H, AII @ H), 0.0))
    ref = float(np.median(eh[eh > 0]))
    ee = np.sqrt(np.einsum("ik,ik->k", E, AII @ E))
    return E * (eps * ref / ee)[None, :]


def specnorm(M):
    return float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (M + M.T)))))


def mineig(M):
    return float(np.linalg.eigvalsh(0.5 * (M + M.T))[0])


# ---------------------------------------------------------------------------
# S2: preconditioned CG with the fast plain-square solve
# ---------------------------------------------------------------------------


def dst_solver(n):
    """Inverse of the plain-square Q1 Laplacian (K1 x M1 + M1 x K1) on interior nodes."""
    h = 1.0 / n
    c = np.cos(np.arange(1, n) * math.pi / n)
    k1 = (2 - 2 * c) / h
    m1 = h * (4 + 2 * c) / 6
    lam = np.outer(m1, k1) + np.outer(k1, m1)       # [q (y), p (x)]

    def solve(r):
        R = sfft.dstn(r.reshape(n - 1, n - 1), type=1, norm="ortho")
        return sfft.dstn(R / lam, type=1, norm="ortho").ravel()
    return solve


def pcg(A, b, prec, x_true, tols, maxit=5000):
    x = np.zeros_like(b); r = b.copy(); z = prec(r); p = z.copy(); rz = r @ z
    e0 = math.sqrt(x_true @ (A @ x_true))
    hits = {}
    for it in range(1, maxit + 1):
        Ap = A @ p
        alpha = rz / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        e = x_true - x
        en = math.sqrt(max(e @ (A @ e), 0.0)) / e0
        for t in tols:
            if t not in hits and en <= t:
                hits[t] = it
        if len(hits) == len(tols):
            break
        z = prec(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return hits


def pcg_plain(A, b, prec, tol, maxit=5000):
    """The same iteration with a relative preconditioned-residual stop, for timing."""
    x = np.zeros_like(b); r = b.copy(); z = prec(r); p = z.copy(); rz = r @ z
    rz0 = rz
    for it in range(1, maxit + 1):
        Ap = A @ p
        alpha = rz / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        z = prec(r)
        rz_new = r @ z
        if math.sqrt(abs(rz_new / rz0)) <= tol:
            return x, it
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, maxit


def check_S2(k_mu=0.5, contrast=10.0) -> dict:
    K = (1 + k_mu) / (1 - k_mu)
    ba_bound = contrast * K * K
    tols = (1e-6, 1e-10)
    out = {"k_mu": k_mu, "contrast": contrast, "K_max": K, "b_over_a_bound": ba_bound,
           "iteration_bound": {str(t): math.ceil(0.5 * math.sqrt(ba_bound) * math.log(2 / t))
                               for t in tols},
           "grids": []}
    for n in (32, 64, 128, 256):
        rng = np.random.default_rng(SEED)
        a11, a12, a22, kap, mu = pulled_back(n, k_mu, contrast, rng)
        A = assemble(n, a11, a12, a22).tocsr()
        bnd, inner = split(n)
        g = loop_fourier(n, 16) @ rng.standard_normal(16)
        AII = sp.csc_matrix(A[inner][:, inner])
        b = -A[inner][:, bnd] @ g
        t0 = time.perf_counter()
        x_true = spla.splu(AII).solve(b)
        t_direct = time.perf_counter() - t0
        prec = dst_solver(n)
        hits = pcg(AII, b, prec, x_true, tols)
        t0 = time.perf_counter()
        _x, it6 = pcg_plain(AII, b, prec, 1e-6)
        t_pcg = time.perf_counter() - t0
        row = {"n": n, "iterations": {str(t): hits.get(t) for t in tols},
               "pcg_over_direct_time": t_pcg / t_direct}
        if n == 32:
            L = assemble(n, np.ones((n, n)), np.zeros((n, n)), np.ones((n, n)))
            ev = sla.eigh(A[inner][:, inner].toarray(), L[inner][:, inner].toarray(),
                          eigvals_only=True)
            row["b_over_a_measured"] = float(ev[-1] / ev[0])
        out["grids"].append(row)
    return out


# ---------------------------------------------------------------------------
# S7: decay of port modes, strip-cropped heads
# ---------------------------------------------------------------------------


def check_S7(n=64, m=16) -> dict:
    tols = (1e-3, 1e-6)
    bnd, inner = split(n)
    s = np.arange(n + 1) / n
    ones = np.ones((n, n)); zeros = np.zeros((n, n))
    K = 3.0
    cases = [("steady", ones, zeros, ones, None, 0.0, 1.0),
             ("Fo=1e-2", ones, zeros, ones, 1e2, 1e2, 1.0),
             ("Fo=1e-3", ones, zeros, ones, 1e3, 1e3, 1.0),
             ("steady, K=3", ones / K, zeros, ones * K, None, 0.0, 1.0 / K)]
    out = {"n": n, "modes_per_side": m, "cases": []}
    for name, a11, a12, a22, mass, screen, aniso in cases:
        A = assemble(n, a11, a12, a22, mass).tocsr()
        lu = spla.splu(sp.csc_matrix(A[inner][:, inner]))
        AIB = A[inner][:, bnd]
        depth = {t: [] for t in tols}
        pred = {t: [] for t in tols}
        lost = {t: [] for t in tols}
        for k in range(m):
            trace = np.zeros((n + 1) ** 2)
            if k == 0:
                trace[1:n] = 1.0                           # constant on the open bottom side
            else:
                trace[0:n + 1] = np.sin(k * math.pi * s)
            u = trace.copy()
            u[inner] = -lu.solve(AIB @ trace[bnd])
            U = np.abs(u.reshape(n + 1, n + 1))
            rowmax = U.max(axis=1)                         # by y index j
            tail = np.maximum.accumulate(rowmax[::-1])[::-1]
            rate = math.sqrt((k * math.pi * aniso) ** 2 + screen)
            energy = u @ (A @ u)
            for t in tols:
                below = np.flatnonzero(tail <= t)
                d = s[below[0]] if below.size else 1.0
                depth[t].append(float(d))
                pred[t].append(min(1.0, math.log(1 / t) / rate) if rate > 0 else 1.0)
                cut = u.copy().reshape(n + 1, n + 1)
                cut[s > d + 1e-12, :] = 0.0
                cut = cut.ravel()
                e = u - cut
                lost[t].append(float((e @ (A @ e)) / energy))
        row = {"case": name}
        for t in tols:
            frac = float(np.mean(np.minimum(1.0, depth[t])))
            row[str(t)] = {"depth": depth[t], "predicted": pred[t],
                           "head_area_fraction": frac, "head_speedup": 1.0 / frac,
                           "energy_lost_max": float(max(lost[t])),
                           "energy_lost_over_tol2_max": float(max(lost[t]) / t ** 2)}
        out["cases"].append(row)
    return out


# ---------------------------------------------------------------------------
# S9: precision of the mode fields
# ---------------------------------------------------------------------------


def to_bf16(x):
    f = np.ascontiguousarray(x, dtype=np.float32)
    u = f.view(np.uint32)
    r = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32).astype(np.float64)


def check_S9(n=64, k_mu=0.3, contrast=4.0) -> dict:
    rng = np.random.default_rng(SEED + 1)
    a11, a12, a22, _kap, _mu = pulled_back(n, k_mu, contrast, rng)
    A = assemble(n, a11, a12, a22).tocsr()
    L0 = assemble(n, np.ones((n, n)), np.zeros((n, n)), np.ones((n, n))).tocsr()
    bnd, inner = split(n)
    Q = loop_fourier(n, 64)
    H, _ = modes(A, bnd, inner, Q)
    H0, _ = modes(L0, bnd, inner, Q)
    N = H - H0
    Lam = gram(A, bnd, inner, H, Q)
    nrm = specnorm(Lam)
    AII = A[inner][:, inner]
    en = lambda X: float(np.sqrt(np.sum(np.einsum("ik,ik->k", X, AII @ X))))
    rounders = {"bfloat16": to_bf16,
                "float16": lambda x: np.asarray(x, np.float16).astype(np.float64),
                "float32": lambda x: np.asarray(x, np.float32).astype(np.float64)}
    out = {"n": n, "k_mu": k_mu, "contrast": contrast, "modes": 64,
           "correction_over_mode_energy": en(N) / en(H), "rows": []}
    for name, rnd in rounders.items():
        Hc = H0 + rnd(N)
        Hf = rnd(H)
        Lc = gram(A, bnd, inner, Hc, Q)
        Lf = gram(A, bnd, inner, Hf, Q)
        Ec = Hc - H
        ident = specnorm(Lc - Lam - Ec.T @ (AII @ Ec)) / nrm
        out["rows"].append({"precision": name,
                            "gram_err_correction_rounded": specnorm(Lc - Lam) / nrm,
                            "gram_err_field_rounded": specnorm(Lf - Lam) / nrm,
                            "identity_residual": ident,
                            "min_eig_gram_minus_exact": mineig(Lc - Lam) / nrm})
    V = np.zeros((A.shape[0], 64), np.float32)
    V[inner] = H.astype(np.float32)
    V[bnd] = Q.astype(np.float32)
    L32 = (V.T @ (A.astype(np.float32) @ V)).astype(np.float64)
    out["gram_formed_in_float32"] = specnorm(0.5 * (L32 + L32.T) - Lam) / nrm
    return out


# ---------------------------------------------------------------------------
# D: the two-dimensional dual bracket
# ---------------------------------------------------------------------------


def loop_pairing(nb):
    """T with psi^T T g = sum over boundary segments of (psi_{i+1} - psi_i)(g_i + g_{i+1})/2,
    the exact value of the loop integral of g d(psi)/ds for piecewise-linear traces."""
    T = np.zeros((nb, nb))
    for i in range(nb):
        j = (i + 1) % nb
        T[j, i] += 0.5; T[j, j] += 0.5; T[i, i] -= 0.5; T[i, j] -= 0.5
    return T


def loop_prolong(n, r):
    """Linear interpolation of coarse boundary-loop values to the r-times finer loop."""
    nb, nf = 4 * n, 4 * n * r
    P = np.zeros((nf, nb))
    for k in range(nf):
        i, t = divmod(k, r)
        P[k, i] += 1 - t / r
        P[k, (i + 1) % nb] += t / r
    return P


def dual_full(Astar, bnd, Y):
    """max over all Q1 psi of 2 psi_B^T y - psi^T A* psi, for the columns y of Y (each
    orthogonal to constants): solve the Neumann problem with one node pinned."""
    N = Astar.shape[0]
    keep = np.arange(1, N)
    rhs = np.zeros((N, Y.shape[1]))
    rhs[bnd] = Y
    lu = spla.splu(sp.csc_matrix(Astar[keep][:, keep]))
    psi = np.zeros((N, Y.shape[1]))
    psi[keep] = lu.solve(rhs[keep])
    L = Y.T @ psi[bnd]
    return 0.5 * (L + L.T)


def corner_sine_ports(n, per_side):
    """Continuous traces with kinks allowed at the corners: per side sin(j pi t), j = 1..per_side,
    plus the four corner hats with the constant removed (3 + 4 per_side orthonormal columns)."""
    nb = 4 * n
    hats = np.zeros((nb, 4))
    sines = []
    for side in range(4):
        idx = side * n + np.arange(n)
        t = np.arange(n) / n
        hats[idx, side] += 1 - t
        hats[idx, (side + 1) % 4] += t
        for j in range(1, per_side + 1):
            v = np.zeros(nb)
            v[idx] = np.sin(j * math.pi * t)
            sines.append(v)
    one = np.ones((nb, 1)) / math.sqrt(nb)
    hats = hats - one @ (one.T @ hats)
    B = np.column_stack([hats] + sines)
    U, sv, _ = np.linalg.svd(B, full_matrices=False)
    return U[:, sv > 1e-10 * sv[0]]


def rel_gap(hi, lo, ref):
    """max over non-constant port data c of c^T (hi - lo) c / c^T ref c."""
    r = slice(1, None)
    return float(sla.eigh(0.5 * (hi - lo)[r, r] + 0.5 * (hi - lo)[r, r].T, ref[r, r],
                          eigvals_only=True)[-1])


def check_D(n=64, r=2, k_mu=0.4, contrast=10.0) -> dict:
    rng = np.random.default_rng(SEED + 2)
    a11, a12, a22, _kap, _mu = pulled_back(n, k_mu, contrast, rng)
    det = a11 * a22 - a12 ** 2
    A = assemble(n, a11, a12, a22).tocsr()
    As = assemble(n, a11 / det, a12 / det, a22 / det).tocsr()
    Af = assemble(n * r, refine(a11, r), refine(a12, r), refine(a22, r)).tocsr()
    Asf = assemble(n * r, refine(a11 / det, r), refine(a12 / det, r), refine(a22 / det, r)).tocsr()
    bnd, inner = split(n)
    bndf, innerf = split(n * r)
    Q = loop_fourier(n, 64)
    T = loop_pairing(4 * n)
    P = loop_prolong(n, r)
    Tf = loop_pairing(4 * n * r)

    H, _ = modes(A, bnd, inner, Q)
    Lam = gram(A, bnd, inner, H, Q)                          # exact discrete port matrix
    Hf, _ = modes(Af, bndf, innerf, P @ Q)
    Lam_fine = gram(Af, bndf, innerf, Hf, P @ Q)             # same traces, finer interior
    low_fine = dual_full(Asf, bndf, Tf @ P @ Q)              # all fine psi
    low_full = dual_full(As, bnd, T @ Q)                     # all coarse psi
    E_true = 0.5 * (Lam_fine + low_fine)                     # midpoint of the fine bracket
    nrm = specnorm(Lam)
    out = {"n": n, "fine_factor": r, "k_mu": k_mu, "contrast": contrast, "primal_ports": 64,
           "chain_min_eigs_rel": {
               "L_low_fine - L_low_full": mineig(low_fine - low_full) / nrm,
               "L_fine - L_low_fine": mineig(Lam_fine - low_fine) / nrm,
               "L - L_fine": mineig(Lam - Lam_fine) / nrm},
           "rel_energy_gaps": {
               "fine bracket (L_fine - L_low_fine)": rel_gap(Lam_fine, low_fine, E_true),
               "coarse discretisation (L - L_fine)": rel_gap(Lam, Lam_fine, E_true),
               "all-node dual (L - L_low_full)": rel_gap(Lam, low_full, E_true)},
           "dual_port_sets": [], "perturbed": []}
    sets = {"loop Fourier, 63": loop_fourier(n, 64)[:, 1:],
            "corner hats + side sines, 63": corner_sine_ports(n, 15),
            "corner hats + side sines, 67": corner_sine_ports(n, 16),
            "corner hats + side sines, 127": corner_sine_ports(n, 31)}
    keep = {}
    for name, Qs in sets.items():
        Hs, _ = modes(As, bnd, inner, Qs)
        Gs = gram(As, bnd, inner, Hs, Qs)
        Yc = Qs.T @ T @ Q
        low = Yc.T @ np.linalg.solve(Gs, Yc)
        keep[name] = (Qs, Hs, Yc, low)
        out["dual_port_sets"].append({
            "ports": name,
            "min_eig (L_low_full - L_low)": mineig(low_full - low) / nrm,
            "rel_gap (L - L_low)": rel_gap(Lam, low, E_true)})
    Qs, Hs, Yc, low_net = keep["corner hats + side sines, 127"]
    freq = np.array([0] + [1 + (k - 1) // 2 for k in range(1, 64)])
    c = np.random.default_rng(SEED + 3).standard_normal(64) / (1.0 + freq) ** 2
    c[0] = 0.0
    ce = float(c @ E_true @ c)
    rel_c = lambda hi, lo: float(c @ (hi - lo) @ c) / ce
    out["smooth_data"] = {"spectrum": "c_f ~ N(0,1)/(1+f)^2 by loop frequency f",
                          "fine bracket (L_fine - L_low_fine)": rel_c(Lam_fine, low_fine),
                          "coarse discretisation (L - L_fine)": rel_c(Lam, Lam_fine),
                          "all-node dual (L - L_low_full)": rel_c(Lam, low_full)}
    for name, (_q, _h, _y, low) in keep.items():
        out["smooth_data"][f"dual {name} (L - L_low)"] = rel_c(Lam, low)
    for eps in (0.01, 0.1, 0.3):
        Ep = scaled_errors(A, inner, H, bubble_perturbation(n, 64, rng, inner), eps)
        Ed = scaled_errors(As, inner, Hs, bubble_perturbation(n, Qs.shape[1], rng, inner), eps)
        Lt = gram(A, bnd, inner, H + Ep, Q)
        Gt = gram(As, bnd, inner, Hs + Ed, Qs)
        low_t = Yc.T @ np.linalg.solve(Gt, Yc)
        out["perturbed"].append({
            "eps": eps,
            "min_eig L~ - L": mineig(Lt - Lam) / nrm,
            "min_eig L_low - L~_low": mineig(low_net - low_t) / nrm,
            "rel true error (L~ - E)": rel_gap(Lt, E_true, E_true),
            "rel network part (L~ - L)": rel_gap(Lt, Lam, E_true),
            "rel computable gap (L~ - L~_low)": rel_gap(Lt, low_t, E_true),
            "smooth data: true error": rel_c(Lt, E_true),
            "smooth data: network part": rel_c(Lt, Lam),
            "smooth data: computable gap": rel_c(Lt, low_t)})
    return out


# ---------------------------------------------------------------------------


def text(res) -> str:
    L = ["arch_family_checks -- checks behind the chart-operator family catalog",
         "=" * 70, ""]
    s2 = res["S2"]
    L.append(f"S2  PCG with the fast plain-square solve; sup|mu| = {s2['k_mu']}, contrast "
             f"{s2['contrast']:g}, K_max = {s2['K_max']:.3g}")
    L.append(f"    derived b/a <= {s2['b_over_a_bound']:.4g}; iteration bound "
             f"{s2['iteration_bound']['1e-06']} (tol 1e-6), {s2['iteration_bound']['1e-10']} (tol 1e-10)")
    for g in s2["grids"]:
        extra = f"; measured b/a {g['b_over_a_measured']:.4g}" if "b_over_a_measured" in g else ""
        L.append(f"    n = {g['n']:4d}: iterations {g['iterations']['1e-06']} (1e-6), "
                 f"{g['iterations']['1e-10']} (1e-10); PCG time / direct time "
                 f"{g['pcg_over_direct_time']:.3g}{extra}")
    L.append("")
    s7 = res["S7"]
    L.append(f"S7  decay of port modes, n = {s7['n']}, {s7['modes_per_side']} modes per side")
    for c in s7["cases"]:
        for t in ("0.001", "1e-06"):
            d = c[t]
            L.append(f"    {c['case']:12s} tol {t:>6s}: head area {d['head_area_fraction']:.3f} "
                     f"(x{d['head_speedup']:.2f}); energy lost <= {d['energy_lost_max']:.2e} "
                     f"= {d['energy_lost_over_tol2_max']:.2f} tol^2")
            L.append("        depth   " + " ".join(f"{v:.3f}" for v in d["depth"]))
            L.append("        predict " + " ".join(f"{v:.3f}" for v in d["predicted"]))
    L.append("")
    s9 = res["S9"]
    L.append(f"S9  precision, n = {s9['n']}, sup|mu| = {s9['k_mu']}, contrast {s9['contrast']:g}, "
             f"64 modes; correction / mode energy = {s9['correction_over_mode_energy']:.3f}")
    for rw in s9["rows"]:
        L.append(f"    {rw['precision']:9s}: Gram error, correction rounded "
                 f"{rw['gram_err_correction_rounded']:.2e}; whole field rounded "
                 f"{rw['gram_err_field_rounded']:.2e}; identity residual "
                 f"{rw['identity_residual']:.1e}; min eig of the error {rw['min_eig_gram_minus_exact']:.1e}")
    L.append(f"    Gram formed in float32 from exact modes: {s9['gram_formed_in_float32']:.2e}")
    L.append("")
    d = res["D"]
    L.append(f"D   dual bracket, n = {d['n']} (fine x{d['fine_factor']}), sup|mu| = {d['k_mu']}, "
             f"contrast {d['contrast']:g}, {d['primal_ports']} primal ports (loop Fourier)")
    for k, v in d["chain_min_eigs_rel"].items():
        L.append(f"    min eig ({k}) / |L| = {v:.2e}")
    for k, v in d["rel_energy_gaps"].items():
        L.append(f"    largest relative energy gap, {k} = {v:.3e}")
    for q in d["dual_port_sets"]:
        L.append(f"    dual ports {q['ports']:30s}: min eig(L_low_full - L_low) "
                 f"{q['min_eig (L_low_full - L_low)']:.1e}; relative gap L - L_low {q['rel_gap (L - L_low)']:.3e}")
    L.append(f"    smooth port data ({d['smooth_data']['spectrum']}), relative energy gaps:")
    for k, v in d["smooth_data"].items():
        if k != "spectrum":
            L.append(f"        {k} = {v:.3e}")
    L.append("    perturbed experts (certificate from the 127-port dual):")
    for p in d["perturbed"]:
        L.append(f"      smooth data, eps {p['eps']:<5g}: true error {p['smooth data: true error']:.3e}; "
                 f"network part {p['smooth data: network part']:.3e}; computable gap "
                 f"{p['smooth data: computable gap']:.3e}")
    for p in d["perturbed"]:
        L.append(f"    eps {p['eps']:<5g}: min eig(L~ - L) {p['min_eig L~ - L']:.1e}; "
                 f"min eig(L_low - L~_low) {p['min_eig L_low - L~_low']:.1e}; relative: true error "
                 f"{p['rel true error (L~ - E)']:.3e}, network part {p['rel network part (L~ - L)']:.3e}, "
                 f"computable gap {p['rel computable gap (L~ - L~_low)']:.3e}")
    L.append("")
    return "\n".join(L)


def main() -> None:
    t0 = time.perf_counter()
    res = {"S2": check_S2(), "S7": check_S7(), "S9": check_S9(), "D": check_D()}
    res["elapsed_s"] = time.perf_counter() - t0
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "family_checks.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    txt = text(res)
    with open(os.path.join(OUT, "family_checks.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print(f"elapsed {res['elapsed_s']:.1f} s")


if __name__ == "__main__":
    main()
