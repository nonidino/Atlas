"""TL0 for CO-T: the two-level, coarse-exact expert with no network, and the classical
coarse bases it is priced against (the coolant plan's Phase 2,
[[chart-operator-stage-0b-and-1-plan]] section 5; the rework's `coarse.py` for CO-D,
which is pinned and only called here).

**The coarse correction of a non-symmetric piece.**  With K_II the piece's interior
block (advection makes it non-symmetric), F = -K_IG Q the trial modes' data,
F* = -K_GI^T Q the adjoint modes' data, b_I the load, and P a full-rank map from
(n_c - 1)^2 coarse unknowns to the fine interior,

    C(X)   = X + P A_c^-1  P^T (F  - K_II X),      A_c = P^T K_II P,
    C*(Y)  = Y + P A_c^-T  P^T (F* - K_II^T Y),
    C_p(u) = u + P A_c^-1  P^T (b_I - K_II u).

What carries over from Proposition A (rework section 3.2), for any input:
1. Galerkin orthogonality: P^T (F - K_II C(X)) = 0, and likewise for C* and C_p (CX);
3. idempotent and blind to coarse shifts: C(C(X)) = C(X), C(X + P v) = C(X);
4. exact at the top: at n_c = n, P = I and C(X) = H for every X (the positive control);
5. the traces untouched: P maps into the interior.
What does not: item 2, "never worse".  With no energy, C(X) - H = (I - P A_c^-1 P^T K_II) E
is an oblique projection of the error, which can be larger than E in any norm.  That is
the G0' price, and why the plan's safety switch reads a residual (section 4).

**Lemma B carries over for a steady piece**: K's interior rows annihilate a constant
(advection and diffusion both do), so if the inputs reproduce a uniform temperature
the outputs do.  On a step the identity needs the level field, as CO-D's did.

**The adjoint is the reversed piece's.**  K_II^T is the interior block of the same piece
with its flow reversed (`transport`), so C* on a piece is C on the reversed piece.

**TL0** is C applied to the plain square's modes at the piece's reaction
(`principal.modes`, the design's H^(0): diffusion only, no chart, no flow), C* to the
same, and C_p to the plain square's solve of the load.  So the coarse grid carries the
piece's true operator (chart, flow, material, step) and the fine grid the plain
square's.  Untrained, nothing is learned; with a network, it would add only what C keeps.

**The classical coarse bases** (the plan's priced alternatives):
- ``"bilinear"``: `coarse.prolongation`, the coarse hats;
- ``"msfem"``: multiscale basis functions (Hou and Wu, J. Comput. Phys. 134, 1997): on
  every coarse cell, the hats' bilinear traces extended into the cell by the piece's own
  operator, so the basis bends with the flow; conforming;
- ``"msfem-os"``: with oversampling (Hou, Wu and Cai, Math. Comp. 68, 1999): the four
  bilinear functions extended by the piece's operator over the cell enlarged by ``os``
  fine cells each way, restricted to the cell and recombined to take the corner values
  1 and 0; nonconforming across cells, so a node shared by two cells takes their mean.
Every one is full rank, so items 1, 3, 4 and 5 hold for each.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import coarse as co
from . import fe
from . import principal as pr
from . import transport as tr

BASES = ("bilinear", "msfem", "msfem-os")


# ---------------------------------------------------------------------------
# coarse bases
# ---------------------------------------------------------------------------


def _interior_index(n: int) -> np.ndarray:
    """Map from the full node index to the interior index (-1 on the loop)."""
    loop, inner = fe.grid(n)
    idx = -np.ones((n + 1) ** 2, int)
    idx[inner] = np.arange(len(inner))
    return idx


def _local_extension(K: sp.csr_matrix, n: int, i0: int, i1: int, j0: int, j1: int,
                     data: np.ndarray) -> np.ndarray:
    """Solve K's rows inside the node box [i0, i1] x [j0, j1] (inclusive) with Dirichlet
    data on the box's boundary nodes.  ``data``: (n+1, n+1, k) nodal values used on the
    box boundary.  Returns the box's nodal values (j1-j0+1, i1-i0+1, k)."""
    n1 = n + 1
    jj, ii = np.meshgrid(np.arange(j0, j1 + 1), np.arange(i0, i1 + 1), indexing="ij")
    box = (jj * n1 + ii).ravel()
    inner = ((jj > j0) & (jj < j1) & (ii > i0) & (ii < i1)).ravel()
    I, G = box[inner], box[~inner]
    k = data.shape[-1]
    flat = data.reshape(-1, k)
    out = np.zeros((len(box), k))
    out[~inner] = flat[G]
    if len(I):
        KII = K[I][:, I].tocsc()
        KIG = K[I][:, G]
        out[inner] = spla.splu(KII).solve(-np.asarray(KIG @ flat[G]))
    return out.reshape(j1 - j0 + 1, i1 - i0 + 1, k)


def _bilinear_cell_functions(n: int, n_c: int, J: int, I: int) -> np.ndarray:
    """The four bilinear coordinate functions of coarse cell (J, I) (local corners
    (0,0), (1,0), (1,1), (0,1)), evaluated at every fine node: (n+1, n+1, 4)."""
    t = np.arange(n + 1) / n
    XI, ETA = np.meshgrid(t, t)
    h = 1.0 / n_c
    x = (XI - I * h) / h
    y = (ETA - J * h) / h
    return np.stack([(1 - x) * (1 - y), x * (1 - y), x * y, (1 - x) * y], axis=-1)


def stencil(K: sp.spmatrix, m: int) -> dict:
    """The nine-point stencil of a matrix on an (m+1) x (m+1) structured node grid
    (row-major, j outer): {(dj, di): array of length (m+1)^2 with entry g holding
    K[g, g + dj (m+1) + di]}, zero where the neighbour is off the grid.  Bilinear
    elements couple a node to its eight neighbours only, so this is the whole matrix."""
    K = K.tocsr()
    m1 = m + 1
    N = m1 * m1
    out = {}
    jj, ii = np.divmod(np.arange(N), m1)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            off = dj * m1 + di
            d = K.diagonal(off)
            full = np.zeros(N)
            if off >= 0:
                full[:N - off] = d
            else:
                full[-off:] = d
            ok = (jj + dj >= 0) & (jj + dj <= m) & (ii + di >= 0) & (ii + di <= m)
            full[~ok] = 0.0
            out[(dj, di)] = full
    return out


def local_solves(st: dict, m: int, boxes: list, datas: list) -> list:
    """Dirichlet solves of the stencil's rows inside node boxes, batched by shape.

    ``boxes``: (j0, j1, i0, i1) inclusive node ranges on the (m+1)^2 grid; ``datas[q]``:
    box q's (by+1, bx+1, k) nodal values, whose boundary part is the Dirichlet data.
    Returns, per box, its (by+1, bx+1, k) nodal solution (interior solved, boundary
    the data)."""
    m1 = m + 1
    out = [None] * len(boxes)
    groups: dict = {}
    for q, (j0, j1, i0, i1) in enumerate(boxes):
        groups.setdefault((j1 - j0, i1 - i0), []).append(q)
    for (by, bx), qs in groups.items():
        la, lb = np.meshgrid(np.arange(by + 1), np.arange(bx + 1), indexing="ij")
        inner = ((la > 0) & (la < by) & (lb > 0) & (lb < bx)).ravel()
        li = np.flatnonzero(inner)
        lo = np.flatnonzero(~inner)
        pos_i = -np.ones((by + 1) * (bx + 1), int)
        pos_i[li] = np.arange(len(li))
        pos_b = -np.ones((by + 1) * (bx + 1), int)
        pos_b[lo] = np.arange(len(lo))
        nb = len(qs)
        Aii = np.zeros((nb, len(li), len(li)))
        Aib = np.zeros((nb, len(li), len(lo)))
        org = np.array([(boxes[q][0], boxes[q][2]) for q in qs])           # (j0, i0)
        ra, rb = la.ravel()[li], lb.ravel()[li]
        g = (org[:, 0][:, None] + ra[None, :]) * m1 + (org[:, 1][:, None] + rb[None, :])
        for (dj, di), arr in st.items():
            ta, tb = ra + dj, rb + di
            ok = (ta >= 0) & (ta <= by) & (tb >= 0) & (tb <= bx)
            tl = np.where(ok, ta * (bx + 1) + tb, 0)
            vals = arr[g]                                                    # (nb, len(li))
            to_i = ok & inner[tl]
            to_b = ok & ~inner[tl]
            rows = np.arange(len(li))
            if np.any(to_i):
                Aii[:, rows[to_i], pos_i[tl[to_i]]] += vals[:, to_i]
            if np.any(to_b):
                Aib[:, rows[to_b], pos_b[tl[to_b]]] += vals[:, to_b]
        D = np.stack([datas[q] for q in qs])                                # (nb, by+1, bx+1, k)
        k = D.shape[-1]
        flat = D.reshape(nb, -1, k)
        sol = flat.copy()
        if len(li):
            sol[:, li] = np.linalg.solve(Aii, -np.einsum("cib,cbk->cik", Aib, flat[:, lo]))
        for t, q in enumerate(qs):
            out[q] = sol[t].reshape(by + 1, bx + 1, k)
    return out


def msfem_prolongation(K: sp.spmatrix, n: int, n_c: int, oversample: int = 0) -> sp.csr_matrix:
    """The multiscale coarse basis of a piece, as a map from the (n_c-1)^2 interior
    coarse nodes to the (n-1)^2 interior fine nodes (`fe.grid` order), the coarse
    nodes in `coarse.prolongation`'s order.  ``oversample = 0``: conforming MsFEM
    (each cell's bilinear traces extended by K); ``> 0``: oversampled by that many fine
    cells on each side of the cell, clipped to the square.  The cells' local problems
    are solved together, batched by shape (`local_solves`)."""
    co.check_sizes(n, n_c)
    r = n // n_c
    n1 = n + 1
    st = stencil(K, n)
    t = np.arange(n + 1) / n
    hc = 1.0 / n_c
    cells, boxes = [], []
    for J in range(n_c):
        for I in range(n_c):
            i0, i1, j0, j1 = I * r, (I + 1) * r, J * r, (J + 1) * r
            if oversample > 0:
                a0, a1 = max(0, i0 - oversample), min(n, i1 + oversample)
                b0, b1 = max(0, j0 - oversample), min(n, j1 + oversample)
            else:
                a0, a1, b0, b1 = i0, i1, j0, j1
            cells.append((J, I, i0, i1, j0, j1, a0, a1, b0, b1))
            boxes.append((b0, b1, a0, a1))

    datas = []
    for (J, I, i0, i1, j0, j1, a0, a1, b0, b1) in cells:
        x = (t[a0:a1 + 1] - I * hc) / hc
        y = (t[b0:b1 + 1] - J * hc) / hc
        X, Y = np.meshgrid(x, y)
        datas.append(np.stack([(1 - X) * (1 - Y), X * (1 - Y), X * Y, (1 - X) * Y], axis=-1))
    sols = local_solves(st, n, boxes, datas)
    acc = np.zeros((n1, n1, (n_c + 1) ** 2))
    cnt = np.zeros((n1, n1, 1))
    for (J, I, i0, i1, j0, j1, a0, a1, b0, b1), big in zip(cells, sols):
        sub = big[j0 - b0:j1 - b0 + 1, i0 - a0:i1 - a0 + 1]
        if oversample > 0:
            Cv = np.array([sub[0, 0], sub[0, -1], sub[-1, -1], sub[-1, 0]])
            sub = np.einsum("jik,kl->jil", sub, np.linalg.inv(Cv))
        for c, (JJ, II) in enumerate([(J, I), (J, I + 1), (J + 1, I + 1), (J + 1, I)]):
            acc[j0:j1 + 1, i0:i1 + 1, JJ * (n_c + 1) + II] += sub[..., c]
        cnt[j0:j1 + 1, i0:i1 + 1] += 1.0
    acc /= cnt
    cols = [JJ * (n_c + 1) + II for JJ in range(1, n_c) for II in range(1, n_c)]
    inner = fe.grid(n)[1]
    Pm = acc.reshape(n1 * n1, -1)[inner][:, cols]
    Pm[np.abs(Pm) < 1e-15] = 0.0
    return sp.csr_matrix(Pm)


def prolongation(K: sp.spmatrix, n: int, n_c: int, basis: str = "bilinear",
                 oversample: int | None = None) -> sp.csr_matrix:
    if basis == "bilinear":
        return co.prolongation(n, n_c)
    if basis == "msfem":
        return msfem_prolongation(K, n, n_c, 0)
    if basis == "msfem-os":
        os_ = max(1, (n // n_c) // 2) if oversample is None else int(oversample)
        return msfem_prolongation(K, n, n_c, os_)
    raise ValueError(f"basis {basis!r}: one of {BASES}")


# ---------------------------------------------------------------------------
# the coarse correction
# ---------------------------------------------------------------------------


class TransportCoarse:
    """The coarse correction of one transport piece (trial, adjoint, particular)."""

    def __init__(self, K: sp.spmatrix, n: int, n_c: int, basis: str = "bilinear",
                 oversample: int | None = None):
        co.check_sizes(n, n_c)
        self.n, self.n_c, self.basis = n, n_c, basis
        loop, inner = fe.grid(n)
        K = K.tocsr()
        self.K_II = K[inner][:, inner].tocsr()
        self.K_IG = K[inner][:, loop].tocsr()
        self.K_GI = K[loop][:, inner].tocsr()
        self.P = prolongation(K, n, n_c, basis, oversample).tocsr()
        Ac = (self.P.T @ self.K_II @ self.P)
        # the coarse operator is local (bilinear or cell-supported columns): a sparse LU
        self._sparse = spla.splu(sp.csc_matrix(Ac))
        self._dense = None

    def _solve(self, r, trans: bool = False):
        r = np.asarray(r, float)
        if self._dense is not None:
            return sla.lu_solve(self._dense, r, trans=1 if trans else 0)
        return self._sparse.solve(r, trans="T" if trans else "N")

    def correct(self, X, F):
        X = np.asarray(X, float)
        r = np.asarray(F, float) - self.K_II @ X
        return X + self.P @ self._solve(self.P.T @ r)

    def correct_adjoint(self, Y, Fs):
        Y = np.asarray(Y, float)
        r = np.asarray(Fs, float) - self.K_II.T @ Y
        return Y + self.P @ self._solve(self.P.T @ r, trans=True)

    def trial_data(self, Q):
        return -np.asarray(self.K_IG @ Q)

    def adjoint_data(self, Q):
        return -np.asarray(self.K_GI.T @ Q)

    def cx(self, X, F, adjoint: bool = False) -> float:
        """CX: ||P^T r|| / ||P^T F|| for the output's residual r."""
        KX = (self.K_II.T @ X) if adjoint else (self.K_II @ X)
        num = np.linalg.norm(self.P.T @ (np.asarray(F) - KX))
        den = np.linalg.norm(self.P.T @ np.asarray(F))
        return float(num / den) if den > 0 else float(num)


class TL0TransportExpert:
    """TL0 for CO-T: the plain square's modes, then the coarse correction with the piece's
    true operator.  ``principal``: ``"laplace"`` (the plain square's, at the piece's
    reaction) or ``"zero"``."""

    def __init__(self, n_c: int = 16, basis: str = "bilinear", principal: str = "laplace",
                 oversample: int | None = None):
        if principal not in ("laplace", "zero"):
            raise ValueError(f"principal part {principal!r}: 'laplace' or 'zero'")
        if basis not in BASES:
            raise ValueError(f"basis {basis!r}: one of {BASES}")
        self.n_c, self.basis, self.principal, self.oversample = n_c, basis, principal, oversample
        self.name = f"TL0-cot({basis}, n_c={n_c})"
        self.envelope = None
        self.calls = 0

    def response(self, p: tr.TransportPiece, Q: np.ndarray) -> tr.TransportResponse:
        n = p.n
        tc = TransportCoarse(p.matrix(), n, min(self.n_c, n), self.basis, self.oversample)
        b_I = p.load()[fe.grid(n)[1]]
        if self.principal == "laplace":
            X0 = pr.modes(n, Q, p.s0)
            u0 = pr.solve(n, b_I, p.s0)
        else:
            X0 = np.zeros(((n - 1) ** 2, Q.shape[1]))
            u0 = np.zeros((n - 1) ** 2)
        H = tc.correct(X0, tc.trial_data(Q))
        G = tc.correct_adjoint(X0, tc.adjoint_data(Q))
        up = tc.correct(u0, b_I)
        return tr.TransportResponse(H, G, up)

    def respond(self, problems, pb_of) -> list:
        out = []
        for p in problems:
            out.append(self.response(p, pb_of(p.n).Q))
            self.calls += 1
        return out


__all__ = ["BASES", "TL0TransportExpert", "TransportCoarse", "msfem_prolongation", "prolongation"]
