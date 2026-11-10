"""Branch 2c's enlarged coarse level in torch, for training: candidate B's corner patches.

[[chart-operator-stage-0-rework]] sections 15-17; the owner adopted B with blocks of n/8 for
Phase 4's arms on 2026-11-10.  `coarse.CoarseSpace` with ``extra=`` (`coarse.CornerTreatment`)
is the numpy reference.  This is the same correction, batched, float64 and differentiable, for
a batch whose examples each have their own non-right corners (``flags`` [B, 4], corner 0 at
(0, 0), then counter-clockwise, as `chart.corner_angles`).

**The space.**  ``P_e = [P | S_b]``: example ``b``'s ``S_b`` holds the fine hats at the
interior nodes of its flagged corners' blocks of ``n/frac``, less the coarse nodes
(`coarse.corner_patch_columns`, in the same order: `patch_nodes`).  The examples' extra
unknowns are padded to the batch's largest count with decoupled unknowns: a one on the
diagonal, nothing else, a zero right-hand side.  Their solution is exactly zero, so one
batched Cholesky serves the whole batch.  A batch with no flagged corner is
`coarse.TorchCoarseSpace` itself, bit for bit.

**The matrix**, ``A_e = [[A_c, (S^T A P)^T], [S^T A P, S^T A S]]``, by probing, as
`coarse.coarse_matrix_torch` assembles ``A_c``.  ``A P`` comes from the nine colourings of the
coarse hats: a coarse hat's image under ``A`` lies within ``r = n/n_c`` fine cells of its
node, and hats of one colour are ``3r`` apart.  ``A S`` comes from the nine colourings of the
fine hats: an image lies within one cell, and hats of one colour are 3 apart.  That is
eighteen `coarse.apply_M` calls besides ``A_c``'s nine.

**The correction.**  Restriction gathers the residual at the extra nodes.  Prolongation adds
the extra values back with an out-of-place scatter onto distinct nodes; the padding goes to a
spare slot that is dropped.  The result therefore does not depend on the order of additions
on CUDA (`gpu-determinism`: every real node receives one addition).  ``A_e`` depends only on
the data, so the gradient flows through the residual alone, as in `coarse.TorchCoarseSpace`.
"""

from __future__ import annotations

import numpy as np

from . import coarse as co

#: the owner's choice for Phase 4 (2026-11-10): B with blocks of n/8
FRAC = 8


def corner_flags(X, tol_deg: float = co.RIGHT_DEG) -> np.ndarray:
    """[4] bool: which of the chart's corners are non-right (`coarse.non_right_corners`)."""
    nr = co.non_right_corners(X, tol_deg)
    return np.array([c in nr for c in range(4)], dtype=bool)


def patch_nodes(n: int, n_c: int, flags, frac: int = FRAC) -> list:
    """Per example, the flat interior indices (``(j-1)(n-1) + (i-1)``) of its extra hats, in
    `coarse.corner_patch_columns`' column order; empty at ``n_c = n`` or with no flag."""
    out = []
    for f in np.asarray(flags, bool).reshape(-1, 4):
        corners = tuple(c for c in range(4) if f[c])
        S = co.corner_patch_columns(n, n_c, corners, frac).tocsc()
        if np.any(np.diff(S.indptr) != 1):
            raise ValueError("a patch column holds other than one hat")
        out.append(np.asarray(S.indices, np.int64).copy())
    return out


def enlarged_matrix_torch(a11, a12, a22, sigma, n_c: int, nodes: list):
    """``(A_e [B, N, N], idx [B, s])``: the enlarged operator (float64, symmetrised) and the
    extra nodes, padded with ``(n-1)^2`` (the spare slot) to the batch's largest count ``s``;
    ``N = (n_c - 1)^2 + s``.  With no extra node anywhere, ``A_e`` is
    `coarse.coarse_matrix_torch`'s ``A_c`` and ``idx`` is empty."""
    torch = co._torch()
    B, n = a11.shape[0], a11.shape[-1]
    if len(nodes) != B:
        raise ValueError(f"{len(nodes)} node lists for a batch of {B}")
    dev = a11.device
    m = n_c - 1
    r = n // n_c
    pad = (n - 1) ** 2
    Ac = co.coarse_matrix_torch(a11, a12, a22, sigma, n_c)
    s = max((len(x) for x in nodes), default=0)
    if s == 0:
        return Ac, torch.zeros(B, 0, dtype=torch.long, device=dev)
    idx = torch.full((B, s), pad, dtype=torch.long, device=dev)
    for b, x in enumerate(nodes):
        if len(x):
            if len(set(np.asarray(x).tolist())) != len(x) or int(np.max(x)) >= pad or int(np.min(x)) < 0:
                raise ValueError(f"example {b}: the extra nodes must be distinct interior nodes")
            idx[b, :len(x)] = torch.as_tensor(np.asarray(x, np.int64), device=dev)
    valid = idx < pad
    jl = torch.div(idx, n - 1, rounding_mode="floor") + 1                    # fine node (i, j), 1..n-1
    il = idx % (n - 1) + 1

    def probe(fields):                      # A on [9, n+1, n+1] fields, interior, with a spare slot
        AF = co.apply_M(a11, a12, a22, sigma, fields[None].expand(B, -1, -1, -1))[..., 1:-1, 1:-1]
        AF = AF.reshape(B, 9, pad)
        return torch.cat([AF, torch.zeros(B, 9, 1, dtype=AF.dtype, device=dev)], -1)

    # A P: the coarse hats' nine colourings, read at the extra nodes
    p = co.prolongation_torch(n, n_c, dev)
    V = torch.zeros(9, m, m, dtype=torch.float64, device=dev)
    for c in range(9):
        V[c, c // 3::3, c % 3::3] = 1.0
    F = torch.zeros(9, n + 1, n + 1, dtype=torch.float64, device=dev)
    F[:, 1:-1, 1:-1] = co.prolong(V, p)
    at_nodes = probe(F).gather(-1, idx[:, None, :].expand(B, 9, s)).transpose(1, 2)      # [B, s, 9]
    Jz, Iz = torch.meshgrid(torch.arange(m, device=dev), torch.arange(m, device=dev), indexing="ij")
    Jz, Iz = Jz.reshape(-1), Iz.reshape(-1)
    cls = ((Jz % 3) * 3 + (Iz % 3))[None, None, :].expand(B, s, m * m)
    near = ((jl[:, :, None] - (Jz + 1)[None, None, :] * r).abs() <= r) & \
           ((il[:, :, None] - (Iz + 1)[None, None, :] * r).abs() <= r) & valid[:, :, None]
    C = torch.where(near, at_nodes.gather(-1, cls), torch.zeros((), dtype=torch.float64, device=dev))

    # A S: the fine hats' nine colourings, read at the extra nodes
    G = torch.zeros(9, n + 1, n + 1, dtype=torch.float64, device=dev)
    for c in range(9):
        G[c, 1 + c // 3:n:3, 1 + c % 3:n:3] = 1.0
    fine = probe(G).gather(-1, idx[:, None, :].expand(B, 9, s)).transpose(1, 2)          # [B, s(k), 9]
    cl = ((jl - 1) % 3) * 3 + ((il - 1) % 3)                                             # [B, s]
    D = fine.gather(-1, cl[:, None, :].expand(B, s, s))                                  # D[b, k, l]: A[k, l]
    adj = ((jl[:, :, None] - jl[:, None, :]).abs() <= 1) & ((il[:, :, None] - il[:, None, :]).abs() <= 1)
    D = torch.where(adj & valid[:, :, None] & valid[:, None, :], D,
                    torch.zeros((), dtype=torch.float64, device=dev))
    D = 0.5 * (D + D.transpose(1, 2)) + torch.diag_embed((~valid).to(torch.float64))

    Ae = torch.zeros(B, m * m + s, m * m + s, dtype=torch.float64, device=dev)
    Ae[:, :m * m, :m * m] = Ac
    Ae[:, m * m:, :m * m] = C
    Ae[:, :m * m, m * m:] = C.transpose(1, 2)
    Ae[:, m * m:, m * m:] = D
    return Ae, idx


class TorchEnlargedSpace:
    """A batch's enlarged coarse level: the Cholesky factor of ``A_e`` and ``C_e`` as a
    differentiable map, as `coarse.TorchCoarseSpace` (which it is, bit for bit, when no
    example has an extra node)."""

    def __init__(self, Ae, n: int, n_c: int, idx):
        torch = co._torch()
        co.check_sizes(n, n_c)
        if Ae.dtype != torch.float64:
            raise ValueError("the coarse operator is float64")
        self.n, self.n_c, self.idx = n, n_c, idx
        self.s = int(idx.shape[1])
        self.base = co.TorchCoarseSpace(Ae, n, n_c) if self.s == 0 else None
        if self.base is None:
            self.p = co.prolongation_torch(n, n_c, Ae.device)
            self.L = torch.linalg.cholesky(Ae)

    @classmethod
    def of_arrays(cls, a11, a12, a22, sigma, n_c: int, nodes: list) -> "TorchEnlargedSpace":
        Ae, idx = enlarged_matrix_torch(a11, a12, a22, sigma, n_c, nodes)
        return cls(Ae, a11.shape[-1], n_c, idx)

    def correction(self, r):
        """``P_e A_e^-1 P_e^T r`` for fine residuals ``r`` [B, K, n-1, n-1] (or [B, n-1, n-1])."""
        if self.base is not None:
            return self.base.correction(r)
        torch = co._torch()
        squeeze = r.dim() == 3
        if squeeze:
            r = r[:, None]
        B, K = r.shape[0], r.shape[1]
        m, s, pad = self.n_c - 1, self.s, (self.n - 1) ** 2
        rc = co.restrict(r, self.p).reshape(B, K, m * m)
        spare = torch.zeros(B, K, 1, dtype=r.dtype, device=r.device)
        ix = self.idx[:, None, :].expand(B, K, s)
        rs = torch.cat([r.reshape(B, K, pad), spare], -1).gather(-1, ix)
        v = torch.cholesky_solve(torch.cat([rc, rs], -1).transpose(1, 2), self.L).transpose(1, 2)
        out = co.prolong(v[..., :m * m].reshape(B, K, m, m), self.p).reshape(B, K, pad)
        out = torch.cat([out, spare], -1).scatter_add(-1, ix, v[..., m * m:])[..., :pad]
        out = out.reshape(B, K, self.n - 1, self.n - 1)
        return out[:, 0] if squeeze else out

    def correct(self, X, r):
        """``C_e(X) = X + P_e A_e^-1 P_e^T r``."""
        return X + self.correction(r)
