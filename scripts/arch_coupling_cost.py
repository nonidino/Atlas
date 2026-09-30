"""arch_coupling_cost -- how many local solves, and how much time, each coupling strategy needs.

For the architecture conversation (the coupling contract, decision (h), and the cost model):
traditional Schwarz against the wave-variable family and the superelement (static
condensation) idea, **with exact classical local solves standing in for the experts**.
That is the best case for any learned expert: it measures the number of local solves --
which becomes the number of expert calls -- and the wall time of everything around the
expert.  The network's own forward cost is not measured here.

**The problem** is the one the charts produce in reference space: a grid of Px x Py
square pieces, each n x n cells, cell-centred finite volumes (unit spacing, 5 points),
a smooth log-normal coefficient kappa (face conductance: harmonic mean), zero Dirichlet
data on the outer boundary, a smooth source.  Steady, and implicit time steps with
a capacity 1/Fo_h on the diagonal, where Fo_h = Fo_piece n^2 and Fo_piece = kappa dt / L^2
is the step's Fourier number on one piece of side L.

**Strategies** (iterations are counted to a true relative residual ||b - A x|| / ||b||
below 1e-8; one iteration = one round of local solves, one per piece):

- ``RAS-Rich``  restricted additive Schwarz, Dirichlet, overlap 2 cells, Richardson: the
  traditional scheme of SNI, NEST and L-DDM;
- ``RAS-GMRES`` the same preconditioner inside right-preconditioned GMRES;
- ``OSM-Rich``  the non-overlapping optimized Schwarz method (Lions' Robin exchange): the
  wave-variable exchange of version 0.  Each piece solves with Robin data g on its
  interface faces (hybridised: face value lambda, outward flux q, g = p lambda - q) and
  hands its neighbour p lambda + q.  No cross-point unknowns arise in this face form.
  The Robin coefficient is the zeroth-order optimum p* = kappa sqrt(k_lo k_hi),
  k_lo = sqrt((pi/n)^2 + s), k_hi = sqrt(pi^2 + s), s = 1/Fo_h, times a multiplier tuned
  once on the smallest case;
- ``OSM-AA``    the same fixed point with Anderson acceleration (depth 5);
- ``OSM-GMRES`` GMRES on the same (affine) interface fixed point;
- ``ORAS-GMRES`` optimized RAS with zero overlap as a GMRES preconditioner (as a plain
  Richardson iteration it diverges: ORAS equals OSM only with algebraic overlap >= 1);
- ``2L-GMRES``  two levels: that preconditioner plus an additive coarse correction on the
  span of {1, x, y, xy} per piece (4 functions per piece), inside GMRES;
- ``Schur``     exact direct substructuring, hybridised on the interface faces: each
  piece's port matrix S_i = D_i - B_i^T A_i^-1 B_i (its Dirichlet-to-Neumann map on its
  interface faces) is formed by one local solve per interface face (probing), assembled,
  factorised once and solved: no iteration;
- ``Port-m``    the superelement: the same with each interface edge's faces reduced to
  its first m cosine modes (S_red = Q^T S Q), so a piece needs 4m local solves to form its
  port matrix -- the matrix a learned superelement would output in one forward pass.
  Its error against the exact solution is reported, since truncating the ports is an
  approximation.

A monolithic sparse direct solve (SuperLU on the whole grid) is timed as the classical
reference.  Wall times are this cloud container's (single SuperLU thread; numpy BLAS as
installed), so only **ratios** are meaningful.  Headless; not a timing of the workbench.
Writes ``out/arch/coupling_cost.json`` and ``out/arch/coupling_cost.txt``.

    set PYTHONIOENCODING=utf-8
    python scripts/arch_coupling_cost.py
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out", "arch")

import numpy as np                                                     # noqa: E402
import scipy.sparse as sp                                              # noqa: E402
import scipy.sparse.linalg as spla                                     # noqa: E402

TOL = 1e-8
MAX_IT = 4000
RNG = np.random.default_rng(20260930)

#: (pieces x, pieces y, cells per piece side, Fo_piece or None for steady)
CASES = [
    (4, 1, 32, None), (8, 1, 32, None), (16, 1, 32, None),
    (2, 2, 32, None), (4, 4, 32, None), (8, 8, 32, None),
    (4, 4, 64, None),
    (4, 4, 32, 1.0), (4, 4, 32, 0.1), (4, 4, 32, 0.01),
    (8, 8, 32, 1.0), (8, 8, 32, 0.1), (8, 8, 32, 0.01),
]
PORT_MODES = (4, 8, 16)


# ---------------------------------------------------------------------------
# the problem
# ---------------------------------------------------------------------------


def smooth_field(nx: int, ny: int, scale: float, rng) -> np.ndarray:
    """A smooth random field: white noise filtered by a Gaussian in Fourier space."""
    w = rng.standard_normal((ny, nx))
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    filt = np.exp(-0.5 * (kx ** 2 + ky ** 2) * scale ** 2)
    f = np.real(np.fft.ifft2(np.fft.fft2(w) * filt))
    return (f - f.mean()) / (f.std() + 1e-300)


class Problem:
    def __init__(self, px: int, py: int, n: int, fo_piece):
        self.px, self.py, self.n = px, py, n
        self.NX, self.NY = px * n, py * n
        NX, NY = self.NX, self.NY
        self.N = NX * NY
        self.kappa = np.exp(0.5 * smooth_field(NX, NY, 0.35 * n, RNG))
        self.fo_piece = fo_piece
        self.cap = 0.0 if fo_piece is None else 1.0 / (fo_piece * n * n)
        k = self.kappa
        idx = np.arange(self.N).reshape(NY, NX)
        rows, cols, vals = [], [], []
        diag = np.full(self.N, self.cap)
        # x faces
        ch = 2.0 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])
        a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        rows += [a, b]; cols += [b, a]; vals += [-ch.ravel(), -ch.ravel()]
        np.add.at(diag, a, ch.ravel()); np.add.at(diag, b, ch.ravel())
        cv = 2.0 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :])
        a, b = idx[:-1, :].ravel(), idx[1:, :].ravel()
        rows += [a, b]; cols += [b, a]; vals += [-cv.ravel(), -cv.ravel()]
        np.add.at(diag, a, cv.ravel()); np.add.at(diag, b, cv.ravel())
        # outer boundary faces: Dirichlet 0 at half a cell, conductance 2 kappa
        for sl in (idx[:, 0], idx[:, -1], idx[0, :], idx[-1, :]):
            np.add.at(diag, sl.ravel(), 2.0 * k.ravel()[sl.ravel()])
        rows.append(np.arange(self.N)); cols.append(np.arange(self.N)); vals.append(diag)
        self.A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                       np.concatenate(cols))),
                               shape=(self.N, self.N))
        self.b = (1.0 + 0.5 * smooth_field(NX, NY, 0.25 * n, RNG)).ravel()
        self.idx = idx
        pid = (np.arange(NY)[:, None] // n) * px + (np.arange(NX)[None, :] // n)
        self.pid = pid.ravel()
        self.P = px * py
        self.cells = [np.flatnonzero(self.pid == p) for p in range(self.P)]

    def p_star(self) -> float:
        s = self.cap
        k_lo = math.sqrt((math.pi / self.n) ** 2 + s)
        k_hi = math.sqrt(math.pi ** 2 + s)
        return float(np.exp(np.mean(np.log(self.kappa)))) * math.sqrt(k_lo * k_hi)


# ---------------------------------------------------------------------------
# one-level preconditioners
# ---------------------------------------------------------------------------


def dilate(P: Problem, cells: np.ndarray, delta: int) -> np.ndarray:
    m = np.zeros((P.NY, P.NX), dtype=bool)
    m.flat[cells] = True
    for _ in range(delta):
        g = m.copy()
        g[1:, :] |= m[:-1, :]; g[:-1, :] |= m[1:, :]
        g[:, 1:] |= m[:, :-1]; g[:, :-1] |= m[:, 1:]
        m = g
    return np.flatnonzero(m.ravel())


class OneLevel:
    """Restricted additive Schwarz; ``robin`` = None for Dirichlet (RAS), else the Robin
    coefficient p of ORAS: each face from the extended subdomain to a cell outside it
    contributes 2 kappa_a p / (2 kappa_a + p) to the diagonal instead of its conductance."""

    def __init__(self, P: Problem, overlap: int, robin):
        self.P = P
        self.locals = []
        A = P.A.tocsr()
        kap = P.kappa.ravel()
        t0 = time.perf_counter()
        for own in P.cells:
            ext = dilate(P, own, overlap)
            Aloc = A[ext][:, ext].tolil()
            if robin is not None:
                inside = np.zeros(P.N, dtype=bool)
                inside[ext] = True
                sub = A[ext]
                d = np.zeros(ext.size)
                for r in range(ext.size):
                    lo, hi = sub.indptr[r], sub.indptr[r + 1]
                    cs, vs = sub.indices[lo:hi], sub.data[lo:hi]
                    out = (~inside[cs]) & (cs != ext[r])
                    if out.any():
                        c_f = -vs[out]                                  # conductances
                        ka = kap[ext[r]]
                        d[r] = np.sum(2.0 * ka * robin / (2.0 * ka + robin) - c_f)
                Aloc = Aloc + sp.diags(d)
            lu = spla.splu(sp.csc_matrix(Aloc))
            keep = np.isin(ext, own)
            self.locals.append((ext, lu, keep, own))
        self.setup_s = time.perf_counter() - t0

    def apply(self, r: np.ndarray) -> np.ndarray:
        z = np.zeros_like(r)
        for ext, lu, keep, own in self.locals:
            z[own] = lu.solve(r[ext])[keep]
        return z


class TwoLevel:
    """ORAS plus an additive coarse correction on {1, x, y, xy} per piece."""

    def __init__(self, P: Problem, one: OneLevel):
        self.one = one
        t0 = time.perf_counter()
        cols = []
        jj, ii = np.divmod(np.arange(P.N), P.NX)
        for own in P.cells:
            x = (ii[own] % P.n + 0.5) / P.n - 0.5
            y = (jj[own] % P.n + 0.5) / P.n - 0.5
            for fvec in (np.ones(own.size), x, y, x * y):
                v = np.zeros(P.N)
                v[own] = fvec
                cols.append(v)
        self.Z = sp.csc_matrix(np.column_stack(cols))
        A0 = (self.Z.T @ P.A @ self.Z).tocsc()
        self.lu0 = spla.splu(A0)
        self.setup_s = time.perf_counter() - t0

    def apply(self, r: np.ndarray) -> np.ndarray:
        return self.one.apply(r) + self.Z @ self.lu0.solve(self.Z.T @ r)


class OSM:
    """Non-overlapping optimized Schwarz (Lions' Robin exchange) on the interface faces.

    Data g live on each side of each interface face: g[:, 0] for the piece holding the
    face's first cell, g[:, 1] for the other.  A piece solves with the Robin condition
    p lambda - q = g on its faces (q the outward flux 2 kappa_a (u_a - lambda)), which
    adds 2 kappa_a p / (p + 2 kappa_a) to the diagonal and 2 kappa_a g / (p + 2 kappa_a)
    to the right-hand side, and returns p lambda + q, which is its neighbour's next g."""

    def __init__(self, P: Problem, p: float):
        self.P, self.p = P, p
        F, _eid, _along, _n = interface(P)
        self.F = F
        kap = P.kappa.ravel()
        A = P.A.tocsr()
        self.pa, self.pb = P.pid[F[:, 0]], P.pid[F[:, 1]]
        c_f = -np.asarray(A[F[:, 0], F[:, 1]]).ravel()
        self.pieces = []
        t0 = time.perf_counter()
        for q_, own in enumerate(P.cells):
            pos = np.full(P.N, -1)
            pos[own] = np.arange(own.size)
            f0 = np.flatnonzero(self.pa == q_)
            f1 = np.flatnonzero(self.pb == q_)
            c0, c1 = F[f0, 0], F[f1, 1]
            k0, k1 = kap[c0], kap[c1]
            d = np.zeros(own.size)
            np.add.at(d, pos[c0], 2.0 * k0 * p / (p + 2.0 * k0) - c_f[f0])
            np.add.at(d, pos[c1], 2.0 * k1 * p / (p + 2.0 * k1) - c_f[f1])
            lu = spla.splu(sp.csc_matrix(A[own][:, own] + sp.diags(d)))
            self.pieces.append((own, pos, lu, f0, f1, c0, c1, k0, k1))
        self.nF = len(F)
        self.setup_s = time.perf_counter() - t0

    def sweep(self, g: np.ndarray, with_b: bool = True):
        P, p = self.P, self.p
        x = np.zeros(P.N)
        h = np.zeros_like(g)
        for own, pos, lu, f0, f1, c0, c1, k0, k1 in self.pieces:
            rhs = P.b[own].copy() if with_b else np.zeros(own.size)
            np.add.at(rhs, pos[c0], 2.0 * k0 * g[f0, 0] / (p + 2.0 * k0))
            np.add.at(rhs, pos[c1], 2.0 * k1 * g[f1, 1] / (p + 2.0 * k1))
            u = lu.solve(rhs)
            x[own] = u
            for fs, cs, ks, side in ((f0, c0, k0, 0), (f1, c1, k1, 1)):
                ua = u[pos[cs]]
                lam = (g[fs, side] + 2.0 * ks * ua) / (p + 2.0 * ks)
                h[fs, side] = p * lam + 2.0 * ks * (ua - lam)
        gn = np.empty_like(g)
        gn[:, 0], gn[:, 1] = h[:, 1], h[:, 0]
        return gn, x


def osm_fixed_point(P: Problem, osm: OSM, anderson: int = 0):
    g = np.zeros((osm.nF, 2))
    nb = np.linalg.norm(P.b)
    hist = []
    Gs, Fs = [], []
    it = 0
    res = 1.0
    t0 = time.perf_counter()
    while it < MAX_IT:
        gn, x = osm.sweep(g)                             # one round of local solves
        it += 1
        res = np.linalg.norm(P.b - P.A @ x) / nb         # residual of the current iterate
        hist.append(res)
        if res < TOL:
            break
        f = (gn - g).ravel()
        if anderson:
            Gs.append(g.ravel().copy()); Fs.append(f.copy())
            if len(Gs) > anderson + 1:
                Gs.pop(0); Fs.pop(0)
            if len(Fs) >= 2:
                dF = np.column_stack([Fs[k + 1] - Fs[k] for k in range(len(Fs) - 1)])
                dG = np.column_stack([Gs[k + 1] - Gs[k] for k in range(len(Gs) - 1)])
                gam, *_ = np.linalg.lstsq(dF, f, rcond=None)
                g = (g.ravel() + f - (dG + dF) @ gam).reshape(g.shape)
                continue
        g = gn
    elapsed = time.perf_counter() - t0
    return finish(it, res, hist, elapsed, None, P)


def osm_gmres(P: Problem, osm: OSM, restart: int = 200):
    """GMRES on (I - K) g = c, the affine interface fixed point g = K g + c; stopped on
    the interface residual at 1e-10, then the true residual is reported."""
    t0 = time.perf_counter()
    c, _ = osm.sweep(np.zeros((osm.nF, 2)), with_b=True)
    c = c.ravel()
    shape = (osm.nF, 2)
    count = [1]                                          # the round for c

    def matvec(v):
        count[0] += 1
        kv, _ = osm.sweep(v.reshape(shape), with_b=False)
        return v - kv.ravel()
    g, _it, hist = gmres_generic(matvec, c, None, 1e-10, restart)
    _, x = osm.sweep(g.reshape(shape))
    it = count[0] + 1                                    # every round of local solves
    elapsed = time.perf_counter() - t0
    res = np.linalg.norm(P.b - P.A @ x) / np.linalg.norm(P.b)
    out = finish(it, res, hist, elapsed, None, P)
    out["converged"] = bool(res < TOL)
    return out


def gmres_generic(matvec, rhs, precond, tol, restart):
    """Right-preconditioned GMRES(restart) for matvec(x) = rhs; returns x, the number of
    matvecs, and the relative residual history."""
    nb = np.linalg.norm(rhs)
    x = np.zeros_like(rhs)
    hist = [1.0]
    it = 0
    res = 1.0
    apply_m = precond if precond is not None else (lambda v: v)
    while it < MAX_IT and res >= tol:
        r = rhs - matvec(x) if it else rhs.copy()
        beta = np.linalg.norm(r)
        res = beta / nb
        if res < tol:
            break
        V = [r / beta]
        Zs = []
        H = np.zeros((restart + 1, restart))
        cs, sn = np.zeros(restart), np.zeros(restart)
        gv = np.zeros(restart + 1)
        gv[0] = beta
        k_done = 0
        for k in range(restart):
            z = apply_m(V[k])
            Zs.append(z)
            w = matvec(z)
            it += 1
            for j in range(k + 1):
                H[j, k] = np.dot(w, V[j])
                w = w - H[j, k] * V[j]
            H[k + 1, k] = np.linalg.norm(w)
            V.append(w / H[k + 1, k] if H[k + 1, k] > 0 else w)
            for j in range(k):
                t = cs[j] * H[j, k] + sn[j] * H[j + 1, k]
                H[j + 1, k] = -sn[j] * H[j, k] + cs[j] * H[j + 1, k]
                H[j, k] = t
            den = math.hypot(H[k, k], H[k + 1, k])
            cs[k], sn[k] = H[k, k] / den, H[k + 1, k] / den
            H[k, k] = den
            H[k + 1, k] = 0.0
            gv[k + 1] = -sn[k] * gv[k]
            gv[k] = cs[k] * gv[k]
            k_done = k + 1
            res = abs(gv[k + 1]) / nb
            hist.append(res)
            if res < tol or it >= MAX_IT:
                break
        y = np.linalg.solve(np.triu(H[:k_done, :k_done]), gv[:k_done])
        x = x + np.column_stack(Zs) @ y
        res = np.linalg.norm(rhs - matvec(x)) / nb
    return x, it, hist


# ---------------------------------------------------------------------------
# iterations
# ---------------------------------------------------------------------------


def richardson(P: Problem, M, anderson: int = 0):
    x = np.zeros(P.N)
    nb = np.linalg.norm(P.b)
    hist_r = []
    t0 = time.perf_counter()
    Xs, Fs = [], []
    it = 0
    res = 1.0
    while it < MAX_IT:
        r = P.b - P.A @ x
        res = np.linalg.norm(r) / nb
        hist_r.append(res)
        if res < TOL:
            break
        f = M.apply(r)                                   # G(x) - x
        it += 1
        if anderson:
            Xs.append(x.copy()); Fs.append(f.copy())
            if len(Xs) > anderson + 1:
                Xs.pop(0); Fs.pop(0)
            if len(Fs) >= 2:
                dF = np.column_stack([Fs[k + 1] - Fs[k] for k in range(len(Fs) - 1)])
                dX = np.column_stack([Xs[k + 1] - Xs[k] for k in range(len(Xs) - 1)])
                gam, *_ = np.linalg.lstsq(dF, f, rcond=None)
                x = x + f - (dX + dF) @ gam
                continue
        x = x + f
    elapsed = time.perf_counter() - t0
    return finish(it, res, hist_r, elapsed, x, P)


def gmres_right(P: Problem, M, restart: int = 200):
    """Right-preconditioned GMRES: the residual it monitors is the true one."""
    nb = np.linalg.norm(P.b)
    x = np.zeros(P.N)
    hist_r = [1.0]
    it = 0
    t0 = time.perf_counter()
    res = 1.0
    while it < MAX_IT and res >= TOL:
        r = P.b - P.A @ x
        beta = np.linalg.norm(r)
        res = beta / nb
        if res < TOL:
            break
        V = [r / beta]
        Zs = []
        H = np.zeros((restart + 1, restart))
        cs, sn = np.zeros(restart), np.zeros(restart)
        g = np.zeros(restart + 1)
        g[0] = beta
        k_done = 0
        for k in range(restart):
            z = M.apply(V[k])
            it += 1
            Zs.append(z)
            w = P.A @ z
            for j in range(k + 1):
                H[j, k] = np.dot(w, V[j])
                w = w - H[j, k] * V[j]
            H[k + 1, k] = np.linalg.norm(w)
            V.append(w / H[k + 1, k] if H[k + 1, k] > 0 else w)
            for j in range(k):
                t = cs[j] * H[j, k] + sn[j] * H[j + 1, k]
                H[j + 1, k] = -sn[j] * H[j, k] + cs[j] * H[j + 1, k]
                H[j, k] = t
            den = math.hypot(H[k, k], H[k + 1, k])
            cs[k], sn[k] = H[k, k] / den, H[k + 1, k] / den
            H[k, k] = den
            H[k + 1, k] = 0.0
            g[k + 1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]
            k_done = k + 1
            res = abs(g[k + 1]) / nb
            hist_r.append(res)
            if res < TOL or it >= MAX_IT:
                break
        y = np.linalg.solve(np.triu(H[:k_done, :k_done]), g[:k_done])
        x = x + np.column_stack(Zs) @ y
        res = np.linalg.norm(P.b - P.A @ x) / nb
    elapsed = time.perf_counter() - t0
    return finish(it, res, hist_r, elapsed, x, P)


def finish(it, res, hist, elapsed, x, P):
    conv = res < TOL
    est = None
    if not conv and len(hist) > 50:
        # extrapolate the last hundred iterations' mean rate to the tolerance
        h = np.array(hist[-min(200, len(hist)):])
        rate = (h[-1] / h[0]) ** (1.0 / (len(h) - 1))
        if 0 < rate < 1:
            est = it + math.log(TOL / hist[-1]) / math.log(rate)
    return {"iterations": it, "converged": bool(conv), "final_residual": float(res),
            "iterations_extrapolated": est, "time_s": elapsed,
            "time_per_iteration_s": elapsed / max(it, 1)}


# ---------------------------------------------------------------------------
# substructuring: exact and port-reduced
# ---------------------------------------------------------------------------


def interface(P: Problem):
    """Interface faces (between cells of different pieces), each once: (cell a, cell b,
    edge id, position along the edge); an edge is the set of faces between one pair of
    pieces."""
    idx, pid = P.idx, P.pid.reshape(P.NY, P.NX)
    faces = []
    for (a2, b2) in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        pa, pb = pid.ravel()[a2.ravel()], pid.ravel()[b2.ravel()]
        m = pa != pb
        faces.append(np.column_stack([a2.ravel()[m], b2.ravel()[m]]))
    F = np.vstack(faces)
    pa, pb = P.pid[F[:, 0]], P.pid[F[:, 1]]
    key = np.minimum(pa, pb) * P.P + np.maximum(pa, pb)
    edges, eid = np.unique(key, return_inverse=True)
    # position along the edge: the coordinate that varies along it
    jj_a, ii_a = np.divmod(F[:, 0], P.NX)
    jj_b, ii_b = np.divmod(F[:, 1], P.NX)
    along = np.where(jj_a == jj_b, jj_a % P.n, ii_a % P.n)
    order = np.lexsort((along, eid))
    return F[order], eid[order], along[order], len(edges)


class Substructure:
    def __init__(self, P: Problem):
        self.P = P
        t0 = time.perf_counter()
        F, eid, along, n_edges = interface(P)
        self.F, self.eid, self.along, self.n_edges = F, eid, along, n_edges
        kap = P.kappa.ravel()
        A = P.A.tocsr()
        self.pieces = []
        nF = len(F)
        for p, own in enumerate(P.cells):
            sub = A[own][:, own].tolil()
            # faces of this piece: which side is inside
            ins_a = P.pid[F[:, 0]] == p
            ins_b = P.pid[F[:, 1]] == p
            fidx = np.flatnonzero(ins_a | ins_b)
            cell_in = np.where(ins_a[fidx], F[fidx, 0], F[fidx, 1])
            cell_out = np.where(ins_a[fidx], F[fidx, 1], F[fidx, 0])
            pos = np.full(P.N, -1)
            pos[own] = np.arange(own.size)
            # remove the coupling to the neighbour cell, add the half-cell link to the face
            c_f = -np.asarray(A[cell_in, cell_out]).ravel()
            ka = kap[cell_in]
            d = np.zeros(own.size)
            np.add.at(d, pos[cell_in], 2.0 * ka - c_f)
            sub = sub + sp.diags(d)
            lu = spla.splu(sp.csc_matrix(sub))
            B = sp.csc_matrix((2.0 * ka, (pos[cell_in], np.arange(fidx.size))),
                              shape=(own.size, fidx.size))
            self.pieces.append((own, lu, fidx, B, 2.0 * ka))
        self.nF = nF
        self.setup_s = time.perf_counter() - t0

    def solve(self, Q=None):
        """Assemble the (port-reduced if Q is given) interface system, solve, reconstruct.
        Q: dense matrix nF x q (block diagonal over edges).  Returns x, timings, solves."""
        P = self.P
        t0 = time.perf_counter()
        nq = self.nF if Q is None else Q.shape[1]
        rows, cols, vals = [], [], []
        rhs = np.zeros(nq)
        local_solves = 0
        for own, lu, fidx, B, D in self.pieces:
            if Q is None:
                Bq = B.toarray()
                cols_g = fidx
                Dq = np.diag(D)
                qmap = None
            else:
                Qi = Q[fidx]                                  # faces of piece x q
                keepq = np.flatnonzero(np.any(Qi != 0.0, axis=0))
                Qi = Qi[:, keepq]
                Bq = B @ Qi
                cols_g = keepq
                Dq = Qi.T @ (D[:, None] * Qi)
                qmap = Qi
            X = lu.solve(np.asarray(Bq))                      # one local solve per column
            local_solves += Bq.shape[1]
            BT = B.T.toarray() if qmap is None else (qmap.T @ B.T.toarray())
            S = Dq - BT @ X
            r, c = np.meshgrid(cols_g, cols_g, indexing="ij")
            rows.append(r.ravel()); cols.append(c.ravel()); vals.append(S.ravel())
            y = lu.solve(P.b[own])
            local_solves += 1
            rhs[cols_g] += BT @ y
        t_form = time.perf_counter() - t0
        t1 = time.perf_counter()
        S = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(nq, nq))
        lam = spla.splu(S).solve(rhs)
        t_solve = time.perf_counter() - t1
        t2 = time.perf_counter()
        x = np.zeros(P.N)
        for own, lu, fidx, B, D in self.pieces:
            lam_i = lam[fidx] if Q is None else (Q[fidx] @ lam)
            x[own] = lu.solve(P.b[own] + B @ lam_i)
            local_solves += 1
        t_rec = time.perf_counter() - t2
        return x, {"form_port_matrices_s": t_form, "interface_factor_solve_s": t_solve,
                   "reconstruct_s": t_rec, "interface_unknowns": int(nq),
                   "local_solves_total": int(local_solves),
                   "local_solves_per_piece": local_solves / P.P}


def port_basis(sub: Substructure, m: int) -> np.ndarray:
    """Block-diagonal orthonormal cosine basis: the first m modes of each edge."""
    nF = sub.nF
    cols = []
    for e in range(sub.n_edges):
        fe = np.flatnonzero(sub.eid == e)
        L = fe.size
        s = sub.along[fe]
        for k in range(min(m, L)):
            v = np.zeros(nF)
            v[fe] = np.cos(math.pi * k * (s + 0.5) / L)
            v[fe] /= np.linalg.norm(v[fe])
            cols.append(v)
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------


def run_case(px, py, n, fo, p_mult) -> dict:
    P = Problem(px, py, n, fo)
    out = {"pieces": [px, py], "n": n, "cells": P.N, "Fo_piece": fo,
           "p_star": P.p_star(), "p_multiplier": p_mult}
    t0 = time.perf_counter()
    lu = spla.splu(sp.csc_matrix(P.A))
    t_fact = time.perf_counter() - t0
    t0 = time.perf_counter()
    x_ref = lu.solve(P.b)
    t_sol = time.perf_counter() - t0
    out["monolithic"] = {"factor_s": t_fact, "solve_s": t_sol, "total_s": t_fact + t_sol}
    nrm = np.linalg.norm(x_ref)

    ras = OneLevel(P, 2, None)
    oras = OneLevel(P, 0, p_mult * P.p_star())
    osm = OSM(P, p_mult * P.p_star())
    out["setup"] = {"RAS_factor_s": ras.setup_s, "ORAS_factor_s": oras.setup_s,
                    "OSM_factor_s": osm.setup_s}
    runs = {
        "RAS-Rich": lambda: richardson(P, ras),
        "RAS-GMRES": lambda: gmres_right(P, ras),
        "OSM-Rich": lambda: osm_fixed_point(P, osm),
        "OSM-AA": lambda: osm_fixed_point(P, osm, anderson=5),
        "OSM-GMRES": lambda: osm_gmres(P, osm),
        "ORAS-GMRES": lambda: gmres_right(P, oras),
    }
    for name, f in runs.items():
        out[name] = f()
    two = TwoLevel(P, oras)
    out["setup"]["coarse_s"] = two.setup_s
    out["2L-GMRES"] = gmres_right(P, two)

    sub = Substructure(P)
    out["setup"]["substructure_factor_s"] = sub.setup_s
    x, info = sub.solve(None)
    info["error_rel"] = float(np.linalg.norm(x - x_ref) / nrm)
    out["Schur"] = info
    for m in PORT_MODES:
        if m > n:
            continue
        Q = port_basis(sub, m)
        x, info = sub.solve(Q)
        info["error_rel"] = float(np.linalg.norm(x - x_ref) / nrm)
        out[f"Port-{m}"] = info
    return out


def tune_p() -> float:
    """The Robin multiplier: tuned once, on the smallest steady square case."""
    best = None
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        P = Problem(2, 2, 32, None)
        r = osm_fixed_point(P, OSM(P, mult * P.p_star()))
        it = r["iterations"] if r["converged"] else (r["iterations_extrapolated"] or 1e9)
        if best is None or it < best[1]:
            best = (mult, it)
    return best[0]


def fmt_it(r):
    if r["converged"]:
        return f"{r['iterations']:>6d}"
    e = r["iterations_extrapolated"]
    return (f"~{int(round(e)):>5d}" if e else "  >cap")


def text(results, p_mult) -> str:
    L = []
    w = L.append
    w("arch_coupling_cost -- local solves and time per coupling strategy (exact local solves)")
    w("=" * 92)
    w(f"host: {platform.platform()}, {os.cpu_count()} cpus, python {platform.python_version()};"
      f" Robin multiplier tuned once: {p_mult}")
    w("iterations = rounds of local solves (one per piece) to ||b-Ax||/||b|| < 1e-8;"
      " '~' extrapolated past the cap")
    w("")
    w("A. ITERATIONS (rounds of local solves, one per piece per round); for Schur and Port-m,")
    w("   local solves per piece to form the port matrix and reconstruct (no iteration)")
    w(" pieces   n  Fo_piece | RAS-Rich RAS-GMRES | OSM-Rich  OSM-AA OSM-GMRES | ORAS-GMRES"
      " 2L-GMRES | Schur  Port-4 Port-8 Port-16")
    for r in results:
        fo = "steady" if r["Fo_piece"] is None else f"{r['Fo_piece']:g}"
        ports = " ".join(f"{r[k]['local_solves_per_piece']:6.0f}" if k in r else "     -"
                         for k in ("Port-4", "Port-8", "Port-16"))
        w(f" {r['pieces'][0]:>2}x{r['pieces'][1]:<2} {r['n']:>3} {fo:>7} |"
          f" {fmt_it(r['RAS-Rich'])}  {fmt_it(r['RAS-GMRES'])}   |"
          f"  {fmt_it(r['OSM-Rich'])}  {fmt_it(r['OSM-AA'])}   {fmt_it(r['OSM-GMRES'])} |"
          f"   {fmt_it(r['ORAS-GMRES'])}  {fmt_it(r['2L-GMRES'])} |"
          f" {r['Schur']['local_solves_per_piece']:5.0f} {ports}")
    w("")
    w("B. PORT-REDUCED SUPERELEMENT: error against the exact solution, interface size")
    w(" pieces   n  Fo_piece | Schur err  unknowns | Port-4 err  unk | Port-8 err  unk |"
      " Port-16 err  unk")
    for r in results:
        fo = "steady" if r["Fo_piece"] is None else f"{r['Fo_piece']:g}"
        s = r["Schur"]
        cells = []
        for k in ("Port-4", "Port-8", "Port-16"):
            if k in r:
                cells.append(f"{r[k]['error_rel']:9.2e} {r[k]['interface_unknowns']:>5}")
            else:
                cells.append("        -     -")
        w(f" {r['pieces'][0]:>2}x{r['pieces'][1]:<2} {r['n']:>3} {fo:>7} |"
          f" {s['error_rel']:9.2e} {s['interface_unknowns']:>6}  | " + " | ".join(cells))
    w("")
    w("C. WALL TIME, setup (factorisations) + solve, as a ratio to RAS-Rich's; the")
    w("   monolithic sparse LU (factor + solve) as the classical reference; '~' rows use an")
    w("   extrapolated iteration count at the measured time per iteration")
    w(" pieces   n  Fo_piece | RAS-Rich [s] | RAS-GMRES OSM-Rich OSM-AA OSM-GMRES ORAS-GMRES"
      " 2L-GMRES | Schur  Port-8 | monolithic")
    for r in results:
        fo = "steady" if r["Fo_piece"] is None else f"{r['Fo_piece']:g}"
        su = r["setup"]

        def tt(k, setup):
            v = r[k]
            t = v["time_s"]
            if not v["converged"] and v.get("iterations_extrapolated"):
                t = v["time_per_iteration_s"] * v["iterations_extrapolated"]
            return t + setup

        def ts(k):
            v = r[k]
            return (su["substructure_factor_s"] + v["form_port_matrices_s"]
                    + v["interface_factor_solve_s"] + v["reconstruct_s"])
        base = tt("RAS-Rich", su["RAS_factor_s"])
        p8 = f"{ts('Port-8') / base:7.4f}" if "Port-8" in r else "      -"
        w(f" {r['pieces'][0]:>2}x{r['pieces'][1]:<2} {r['n']:>3} {fo:>7} | {base:11.3f}  |"
          f"  {tt('RAS-GMRES', su['RAS_factor_s']) / base:7.4f}  "
          f"{tt('OSM-Rich', su['OSM_factor_s']) / base:7.4f} "
          f"{tt('OSM-AA', su['OSM_factor_s']) / base:7.4f}  "
          f"{tt('OSM-GMRES', su['OSM_factor_s']) / base:7.4f}   "
          f"{tt('ORAS-GMRES', su['ORAS_factor_s']) / base:7.4f}  "
          f"{tt('2L-GMRES', su['ORAS_factor_s'] + su['coarse_s']) / base:7.4f} |"
          f" {ts('Schur') / base:7.4f} {p8} | {r['monolithic']['total_s'] / base:7.4f}")
    return "\n".join(L) + "\n"


def main() -> None:
    p_mult = tune_p()
    results = []
    for c in CASES:
        t0 = time.perf_counter()
        r = run_case(*c, p_mult)
        r["case_wall_s"] = time.perf_counter() - t0
        results.append(r)
        sys.stdout.write(f"done {c} in {r['case_wall_s']:.1f} s\n")
        sys.stdout.flush()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "coupling_cost.json"), "w", encoding="utf-8") as fh:
        json.dump({"probe": "arch_coupling_cost", "tol": TOL, "max_it": MAX_IT,
                   "p_multiplier": p_mult, "host": platform.platform(),
                   "cpus": os.cpu_count(), "results": results}, fh, indent=1)
    t = text(results, p_mult)
    with open(os.path.join(OUT, "coupling_cost.txt"), "w", encoding="utf-8") as fh:
        fh.write(t)
    sys.stdout.write(t)


if __name__ == "__main__":
    main()
