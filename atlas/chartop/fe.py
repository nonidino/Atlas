"""Bilinear (Q1) finite elements on the unit square's reference grid: the piece's energy.

Decision D7 (the owner, 2026-10-18).  The document's piece solver was face-hybridised
two-point-flux finite volumes, and a two-point flux cannot carry the off-diagonal
entry -2 Im(mu) / (1 - |mu|^2) of the pulled-back tensor, which every shearing chart
(Winslow at a convex corner that is not a right angle) produces.  Bilinear elements
on the reference grid's vertices carry the full tensor exactly, element by element.
They are also exactly the isoparametric bilinear elements of the physical mesh the
chart draws, so Proposition 3's certificate (a positive bilinear Jacobian at every
cell corner) is the validity condition of that mesh; and they are conforming, which
is the setting of Theorem 2 and of the dual bracket.  Promoted from
`scripts/arch_family_checks.py` (`_q1_reference`, `assemble`, `modes`, `gram`), with
the reaction term, the boundary numbering and a matrix-free energy added.

**Grid.**  ``n`` elements per side, ``h = 1/n``.  Node ``(i, j)`` sits at
``(xi, eta) = (i h, j h)`` with index ``j (n + 1) + i``.  Element ``(j, i)`` has the
local nodes ``(i, j), (i+1, j), (i+1, j+1), (i, j+1)``.  Element arrays are indexed
``[j, i]`` (eta rows, xi columns), as images and torch tensors are.

**The energy.**  Per element, a constant tensor ``A = (a11, a12, a22)`` and a reaction
``sigma`` (the backward-Euler capacity ``J~ / Fo``, zero when steady).  For a nodal
field ``u``,

    E(u) = sum_e [ int_e grad u . A grad u  +  sigma_e int_e u^2 ],

integrated exactly for bilinear ``u``: the stiffness part is scale-free in 2-D and the
mass part carries ``h^2``.  ``M`` is its matrix, ``E(u) = u^T M u`` (no factor 1/2).

**Boundary and interior.**  The ``4n`` boundary nodes in loop order, counter-clockwise
from ``(0, 0)``: side 0 (``eta = 0``) with ``xi`` rising, side 1 (``xi = 1``) with
``eta`` rising, side 2 (``eta = 1``) with ``xi`` falling, side 3 (``xi = 0``) with
``eta`` falling; corner ``c`` is loop position ``c n``.  The ``(n - 1)^2`` interior nodes
in index order.  With ``I`` the interior and ``G`` the loop,

    M = [[A_I, -B], [-B^T, D]],  A_I = M[I, I],  B = -M[I, G],  D = M[G, G],

and the exact objects of the design are ``H = A_I^-1 B Q``, ``u_p = A_I^-1 b_I`` and
``Lambda = Q^T (D - B^T A_I^-1 B) Q``.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _q1_reference():
    """Element matrices on the unit cell, local nodes (0,0), (1,0), (1,1), (0,1), by
    2 x 2 Gauss quadrature (exact for these integrands): stiffness parts for
    u_xi^2, u_eta^2 and 2 u_xi u_eta, and the mass."""
    gp = (np.array([-1.0, 1.0]) / math.sqrt(3.0) + 1.0) / 2.0
    kxx = np.zeros((4, 4))
    kyy = np.zeros((4, 4))
    kxy = np.zeros((4, 4))
    ms = np.zeros((4, 4))
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


@functools.lru_cache(maxsize=None)
def grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(loop, interior): the 4n boundary node indices counter-clockwise from (0, 0),
    and the (n-1)^2 interior node indices in index order."""
    n1 = n + 1
    bottom = [i for i in range(n)]
    right = [j * n1 + n for j in range(n)]
    top = [n * n1 + i for i in range(n, 0, -1)]
    left = [j * n1 for j in range(n, 0, -1)]
    loop = np.array(bottom + right + top + left, dtype=np.int64)
    mask = np.ones(n1 * n1, bool)
    mask[loop] = False
    inner = np.flatnonzero(mask)
    loop.setflags(write=False)
    inner.setflags(write=False)
    return loop, inner


@functools.lru_cache(maxsize=None)
def element_nodes(n: int) -> np.ndarray:
    """(n^2, 4) global node indices of each element, elements in [j, i] order."""
    n1 = n + 1
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    nodes = np.stack([j * n1 + i, j * n1 + i + 1, (j + 1) * n1 + i + 1, (j + 1) * n1 + i],
                     axis=-1).reshape(-1, 4)
    nodes.setflags(write=False)
    return nodes


def _scatter(n: int, ke: np.ndarray) -> sp.csr_matrix:
    nodes = element_nodes(n)
    rows = np.repeat(nodes, 4, axis=1).ravel()
    cols = np.tile(nodes, (1, 4)).ravel()
    N = (n + 1) ** 2
    return sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(N, N))


def assemble(n: int, a11, a12, a22, sigma=None) -> sp.csr_matrix:
    """The energy matrix M on the (n+1)^2 nodes.  ``a11, a12, a22`` (and ``sigma``) are
    element arrays [j, i], or scalars."""
    kxx, kyy, kxy, ms = Q1
    shape = (n, n)
    a11 = np.broadcast_to(np.asarray(a11, float), shape).reshape(-1)
    a12 = np.broadcast_to(np.asarray(a12, float), shape).reshape(-1)
    a22 = np.broadcast_to(np.asarray(a22, float), shape).reshape(-1)
    ke = (a11[:, None, None] * kxx + a22[:, None, None] * kyy + a12[:, None, None] * kxy)
    if sigma is not None:
        s = np.broadcast_to(np.asarray(sigma, float), shape).reshape(-1)
        ke = ke + (s[:, None, None] / n ** 2) * ms
    return _scatter(n, ke)


def mass(n: int, w=1.0) -> sp.csr_matrix:
    """The mass matrix weighted per element by ``w``: the load of a nodal source g is
    ``mass(n, J~) @ g``, consistent with the reaction ``sigma = J~ / Fo``."""
    ms = Q1[3]
    wv = np.broadcast_to(np.asarray(w, float), (n, n)).reshape(-1)
    return _scatter(n, (wv[:, None, None] / n ** 2) * ms)


def energy(a11, a12, a22, sigma, U) -> np.ndarray:
    """Matrix-free ``u^T M u`` for nodal fields ``U`` of shape (..., n+1, n+1) [j, i]
    and element arrays of shape (n, n) (broadcast over the leading axes).

    With the element's corner values u00, u10, u11, u01 and the edge differences
    d0 = u10 - u00, d1 = u11 - u01 (along xi) and e0 = u01 - u00, e1 = u11 - u10
    (along eta), the exact integrals over the unit element are

        int u_xi^2 = (d0^2 + d0 d1 + d1^2) / 3,   int u_eta^2 = (e0^2 + e0 e1 + e1^2) / 3,
        int 2 u_xi u_eta = (d0 + d1)(e0 + e1) / 2,
        int u^2 = (sum u^2 + sum_edges u u) / 9 + (u00 u11 + u10 u01) / 18,

    the last times h^2.  The same formula, in torch, is the training loss's energy."""
    U = np.asarray(U, float)
    n = U.shape[-1] - 1
    u00 = U[..., :-1, :-1]
    u10 = U[..., :-1, 1:]
    u11 = U[..., 1:, 1:]
    u01 = U[..., 1:, :-1]
    d0, d1 = u10 - u00, u11 - u01
    e0, e1 = u01 - u00, u11 - u10
    dens = (a11 * (d0 * d0 + d0 * d1 + d1 * d1) / 3.0
            + a22 * (e0 * e0 + e0 * e1 + e1 * e1) / 3.0
            + a12 * (d0 + d1) * (e0 + e1) / 2.0)
    if sigma is not None:
        sq = u00 * u00 + u10 * u10 + u11 * u11 + u01 * u01
        ed = u00 * u10 + u10 * u11 + u11 * u01 + u01 * u00
        dg = u00 * u11 + u10 * u01
        dens = dens + (np.asarray(sigma, float) / n ** 2) * (sq / 9.0 + ed / 9.0 + dg / 18.0)
    return dens.sum(axis=(-2, -1))


def full_field(n: int, interior_values, loop_values) -> np.ndarray:
    """Nodal vectors from interior values ((n-1)^2, k) and loop values (4n, k)."""
    loop, inner = grid(n)
    iv = np.asarray(interior_values, float)
    lv = np.asarray(loop_values, float)
    k = () if iv.ndim == 1 else (iv.shape[1],)
    out = np.zeros(((n + 1) ** 2,) + k)
    out[inner] = iv
    out[loop] = lv
    return out


def gram(M: sp.spmatrix, V: np.ndarray) -> np.ndarray:
    """V^T M V for full nodal columns V, symmetrised (the host's port matrix)."""
    L = V.T @ (M @ V)
    return 0.5 * (L + L.T)


class Exact:
    """The classical piece solve: one sparse LU of A_I, then triangular solves.

    ``modes(Q)`` gives H = A_I^-1 B Q (one pair of triangular solves per column),
    ``particular(b_I)`` u_p = A_I^-1 b_I, ``port_matrix(Q, H)`` Lambda.  This is the
    classical superelement of gates E3 and E5 and the label generator."""

    def __init__(self, M: sp.spmatrix, n: int):
        self.n = n
        loop, inner = grid(n)
        M = M.tocsr()
        self.M = M
        self.A_I = M[inner][:, inner].tocsc()
        self.M_IG = M[inner][:, loop].tocsr()          # = -B
        self.D = M[loop][:, loop].toarray()
        self.lu = spla.splu(self.A_I)

    def modes(self, Q: np.ndarray) -> np.ndarray:
        return -self.lu.solve(np.asarray(self.M_IG @ Q))

    def particular(self, b_I: np.ndarray) -> np.ndarray:
        return self.lu.solve(np.asarray(b_I, float))

    def port_matrix(self, Q: np.ndarray, H: np.ndarray | None = None) -> np.ndarray:
        """Lambda = Q^T D Q + Q^T M_GI H, which is Q^T (D - B^T A_I^-1 B) Q."""
        if H is None:
            H = self.modes(Q)
        L = Q.T @ (self.D @ Q) + Q.T @ (self.M_IG.T @ H)
        return 0.5 * (L + L.T)

    def energy_I(self, X: np.ndarray) -> np.ndarray:
        """||x||^2_{A_I} per column of interior fields X."""
        X = np.asarray(X, float)
        AX = self.A_I @ X
        return np.einsum("i...,i...->...", X, AX)
