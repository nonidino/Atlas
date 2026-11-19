"""The safety switch's residual for CO-T (the coolant plan, section 4, Phase 2).

**What the switch does.**  A learned piece answer (trial modes, adjoint modes, a
particular field, each after the coarse correction) is kept only when its residual on
the piece's own operator is no larger than TL0's; otherwise TL0's answer is used.  The
residual needs no reference and is checked whatever the weights.

**Which norm.**  For trial modes X with data F, the residual is R = F - K_II X.  A plain
l2 norm of R is dominated by the finest modes (a residual's l2 norm weights a mode by
its eigenvalue squared), which would repeat P6 inside the switch.  The natural norm is
the dual one,

    eta(X)^2 = sum over columns of R^T D_II^-1 R,

with D_II the symmetric part of K_II: the piece's diffusion and reaction (the advection
matrix's symmetric part lives on the loop, `transport`).  It weights the residual as
the energy weights an error: for a symmetric operator it is exactly the energy norm of
the error, and here it bounds the error in the D-norm by inf-sup arguments, with a
constant that shrinks as the Peclet number grows (the plan's second limit: the switch
guarantees the residual, not the error).

**Three ways to compute it** (`dual_residual`):
- ``"exact"``: one sparse LU of D_II and a solve per column, as costly as a classical
  piece solve (a second factorisation of the same size);
- ``"schwarz"``: a two-level additive Schwarz estimate,
  B = P (P^T D P)^-1 P^T + sum over cells of R_c^T D_c^-1 R_c, with the coarse cells
  overlapping by ``overlap`` fine cells.  B is spectrally equivalent to D^-1, with
  constants independent of h (Dryja and Widlund, 1987-94).  After the coarse correction
  P^T R = 0 exactly (`transport_coarse`, Proposition A item 1), so the coarse term is
  zero and only the local solves cost anything;
- ``"jacobi"``: R^T diag(D)^-1 R, the cheapest, and the one that leans on the fine modes
  (kept to show why it is not used).

**[AI Inference]:** the Schwarz estimate's cost is that of the multiscale basis's local
solves, a fraction of a classical solve; Phase 2 measures it (`scripts/cot_phase2_*`).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import coarse as co
from . import fe

MODES = ("exact", "schwarz", "jacobi")


def interior_diffusion(p) -> sp.csr_matrix:
    """D_II: the piece's diffusion and reaction on its interior nodes."""
    inner = fe.grid(p.n)[1]
    return p.diffusion_matrix().tocsr()[inner][:, inner].tocsr()


class Schwarz:
    """The local part of the two-level additive Schwarz estimate of D^-1 on one piece:
    the coarse cells of an n_c grid, each grown by ``overlap`` fine cells, each block of
    D inverted once.  The blocks are built from D's nine-point stencil and inverted
    together, batched by shape (`transport_coarse.stencil`)."""

    def __init__(self, D_II: sp.spmatrix, n: int, n_c: int, overlap: int = 1):
        from .transport_coarse import stencil
        co.check_sizes(n, n_c)
        r = n // n_c
        m1 = n - 1                                 # interior nodes per side
        st = stencil(D_II, n - 2)
        shapes: dict = {}
        for J in range(n_c):
            for I in range(n_c):
                i0, i1 = max(1, I * r - overlap), min(n - 1, (I + 1) * r + overlap)
                j0, j1 = max(1, J * r - overlap), min(n - 1, (J + 1) * r + overlap)
                if i1 < i0 or j1 < j0:
                    continue
                shapes.setdefault((j1 - j0, i1 - i0), []).append((j0 - 1, i0 - 1))
        self.groups = []
        for (by, bx), origins in shapes.items():
            la, lb = np.meshgrid(np.arange(by + 1), np.arange(bx + 1), indexing="ij")
            la, lb = la.ravel(), lb.ravel()
            org = np.array(origins)
            idx = (org[:, 0][:, None] + la[None, :]) * m1 + (org[:, 1][:, None] + lb[None, :])
            nb, bs = idx.shape
            Dk = np.zeros((nb, bs, bs))
            rows = np.arange(bs)
            for (dj, di), arr in st.items():
                ta, tb = la + dj, lb + di
                ok = (ta >= 0) & (ta <= by) & (tb >= 0) & (tb <= bx)
                tl = ta * (bx + 1) + tb
                if np.any(ok):
                    Dk[:, rows[ok], tl[ok]] += arr[idx[:, ok]]
            self.groups.append((idx, np.linalg.inv(Dk)))
        self.size = m1 * m1

    def apply(self, R: np.ndarray) -> np.ndarray:
        """B_local R."""
        R = np.asarray(R, float)
        R2 = R.reshape(R.shape[0], -1)
        out = np.zeros_like(R2)
        for idx, inv in self.groups:
            Y = np.einsum("cij,cjk->cik", inv, R2[idx])
            np.add.at(out, idx.ravel(), Y.reshape(-1, R2.shape[1]))
        return out.reshape(R.shape)


def dual_residual(R: np.ndarray, D_II: sp.spmatrix, mode: str = "schwarz", *, n: int | None = None,
                  n_c: int = 16, overlap: int = 1, P=None, cache: dict | None = None) -> float:
    """eta = sqrt(sum over columns of R^T B R), B per ``mode`` (see the module).  ``P``:
    the coarse basis, for the Schwarz estimate's coarse term (zero after the coarse
    correction; computed anyway when given).  ``cache``: reuse factorisations."""
    R = np.asarray(R, float)
    R2 = R.reshape(R.shape[0], -1)
    cache = {} if cache is None else cache
    if mode == "exact":
        lu = cache.get("exact") or spla.splu(sp.csc_matrix(D_II))
        cache["exact"] = lu
        return float(np.sqrt(max(np.einsum("ij,ij->", R2, lu.solve(R2)), 0.0)))
    if mode == "jacobi":
        d = np.asarray(D_II.diagonal(), float)
        return float(np.sqrt(np.einsum("ij,ij->", R2, R2 / d[:, None])))
    if mode == "schwarz":
        if n is None:
            raise ValueError("the Schwarz estimate needs the grid size n")
        S = cache.get(("schwarz", n_c, overlap)) or Schwarz(D_II, n, n_c, overlap)
        cache[("schwarz", n_c, overlap)] = S
        val = float(np.einsum("ij,ij->", R2, S.apply(R2)))
        if P is not None:
            Pt = (P.T @ R2)
            Ac = (P.T @ D_II @ P).toarray()
            val += float(np.einsum("ij,ij->", Pt, np.linalg.solve(Ac, Pt)))
        return float(np.sqrt(max(val, 0.0)))
    raise ValueError(f"mode {mode!r}: one of {MODES}")


def piece_eta(p, resp, Q: np.ndarray, mode: str = "schwarz", n_c: int = 16, overlap: int = 1,
              cache: dict | None = None) -> dict:
    """The switch's residuals for one piece's answer: trial, adjoint and particular, and
    their sum, the number the switch compares."""
    from . import transport as tr
    n = p.n
    loop, inner = fe.grid(n)
    K = p.matrix().tocsr()
    K_II = K[inner][:, inner]
    F = -np.asarray(K[inner][:, loop] @ Q)
    Fs = -np.asarray(K[loop][:, inner].T @ Q)
    b_I = p.load()[inner]
    D = interior_diffusion(p)
    cache = {} if cache is None else cache
    kw = dict(n=n, n_c=min(n_c, n), overlap=overlap, cache=cache)
    t = dual_residual(F - K_II @ resp.H, D, mode, **kw)
    a = 0.0 if resp.G is None else dual_residual(Fs - K_II.T @ resp.G, D, mode, **kw)
    u = dual_residual((b_I - K_II @ resp.up)[:, None], D, mode, **kw)
    return {"trial": t, "adjoint": a, "particular": u, "total": float(np.sqrt(t * t + a * a + u * u))}


class SwitchedExpert:
    """A learned CO-T expert behind the switch: per piece, its answer if its total dual
    residual is at most TL0's (times ``1 + slack``), TL0's otherwise.  Records the
    decisions (``kept``)."""

    def __init__(self, learned, tl0, mode: str = "schwarz", n_c: int = 16, overlap: int = 1,
                 slack: float = 0.0):
        self.learned, self.tl0 = learned, tl0
        self.mode, self.n_c, self.overlap, self.slack = mode, n_c, overlap, slack
        self.name = f"switched({getattr(learned, 'name', 'learned')})"
        self.envelope = None
        self.calls = 0
        self.kept: list = []

    def respond(self, problems, pb_of) -> list:
        a = self.learned.respond(problems, pb_of)
        b = self.tl0.respond(problems, pb_of)
        out, kept = [], []
        for p, ra, rb in zip(problems, a, b):
            Q = pb_of(p.n).Q
            cache: dict = {}
            ea = piece_eta(p, ra, Q, self.mode, self.n_c, self.overlap, cache)["total"]
            eb = piece_eta(p, rb, Q, self.mode, self.n_c, self.overlap, cache)["total"]
            keep = ea <= eb * (1.0 + self.slack)
            kept.append({"learned": ea, "tl0": eb, "kept": bool(keep)})
            out.append(ra if keep else rb)
            self.calls += 1
        self.kept = kept
        return out


__all__ = ["MODES", "Schwarz", "SwitchedExpert", "dual_residual", "interior_diffusion", "piece_eta"]
