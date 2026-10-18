"""Trace bases on the reference square's boundary loop (decision D3).

A port basis is a matrix ``Q`` of shape (4n, 4m): column ``k`` is the trace of port
mode ``k`` on the ``4n`` loop nodes of `fe.grid` (counter-clockwise from (0, 0), side
``s`` covering loop positions ``s n .. s n + n``, its end corner being the next side's
first node).  ``m`` modes per side, 16 by default (the Expert Card's
``port_modes_per_side``).

**The default (D3): corner hats plus side sines.**  Four hats, one per corner,
linear along the two sides that meet there and one at their corner; then, per side,
``sin(j pi t)`` for ``j = 1 .. m-1`` with ``t`` the side's own reference parameter.
That is ``4 + 4(m - 1) = 4m`` columns.  Every trace is continuous round the loop, so a
corner value is one number shared by every piece that meets there: the coefficients
are **vertex values** and **side amplitudes**, which is how the host assembles
cross-points.  The constant (a uniform temperature) is the sum of the hats, so total
flux per piece is exact.

**Comparison C6: per-side cosines.**  ``cos(j pi t)``, ``j = 0 .. m-1``, on each side's
interior nodes; a corner node, which two sides share in bilinear elements, takes the
average of its two sides' values.  This is the nodal form of version 1's cosine
ports.  Two pieces that meet at a corner each average their own two sides, so the
coupled field can differ between pieces at that node: the arm is nonconforming at
corners, the bilinear-element face of the document's Remark 4.

Reversal: a side shared by two pieces is traversed in opposite directions (both
loops are counter-clockwise), so the neighbour's parameter is ``1 - t``.  Then
``sin(j pi (1 - t)) = (-1)^(j+1) sin(j pi t)`` and ``cos(j pi (1 - t)) = (-1)^j cos(j pi t)``;
hats are vertex values and only change place.  `reversal_sign` gives the sign.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

KINDS = ("sine-hat", "cosine")


@dataclass(frozen=True)
class Column:
    """What a column of Q is: ``("vertex", c, 0)`` for corner c's hat, or
    ``("side", s, j)`` for mode j of side s."""
    what: str
    index: int
    j: int


@dataclass(frozen=True)
class PortBasis:
    kind: str
    n: int
    m: int
    Q: np.ndarray
    columns: tuple

    @property
    def size(self) -> int:
        return self.Q.shape[1]

    def reversal_sign(self, k: int) -> float:
        c = self.columns[k]
        if c.what == "vertex":
            return 1.0
        if self.kind == "sine-hat":
            return -1.0 if c.j % 2 == 0 else 1.0       # (-1)^(j+1)
        return -1.0 if c.j % 2 == 1 else 1.0           # (-1)^j

    def side_columns(self, s: int) -> list[int]:
        return [k for k, c in enumerate(self.columns) if c.what == "side" and c.index == s]

    def vertex_column(self, c: int) -> int | None:
        for k, col in enumerate(self.columns):
            if col.what == "vertex" and col.index == c:
                return k
        return None


def side_positions(n: int, s: int) -> np.ndarray:
    """Loop positions of side s with both corners: n + 1 positions, t = 0 .. 1."""
    return (s * n + np.arange(n + 1)) % (4 * n)


@functools.lru_cache(maxsize=None)
def basis(kind: str, n: int, m: int = 16) -> PortBasis:
    if kind not in KINDS:
        raise ValueError(f"port basis {kind!r}: one of {KINDS}")
    if m < 2 or m > n:
        raise ValueError(f"{m} modes per side on a grid of {n}: need 2 <= m <= n")
    t = np.arange(n) / n                     # each side's own nodes, its end corner excluded
    cols, meta = [], []
    if kind == "sine-hat":
        for c in range(4):
            v = np.zeros(4 * n)
            v[c * n:(c + 1) * n] = 1.0 - t                      # side c starts at corner c
            prev = (c - 1) % 4
            v[prev * n:(prev + 1) * n] += t                     # side c-1 ends at corner c
            cols.append(v)
            meta.append(Column("vertex", c, 0))
        for s in range(4):
            for j in range(1, m):
                v = np.zeros(4 * n)
                v[s * n:(s + 1) * n] = np.sin(j * math.pi * t)
                cols.append(v)
                meta.append(Column("side", s, j))
    else:
        for s in range(4):
            for j in range(m):
                v = np.zeros(4 * n)
                v[s * n + 1:(s + 1) * n] = np.cos(j * math.pi * t[1:])
                v[s * n] += 0.5                                  # cos(0) at its start corner
                v[((s + 1) % 4) * n] += 0.5 * (-1.0) ** j        # cos(j pi) at its end corner
                cols.append(v)
                meta.append(Column("side", s, j))
    Q = np.column_stack(cols)
    Q.setflags(write=False)
    return PortBasis(kind, n, m, Q, tuple(meta))


def side_mass(n: int) -> np.ndarray:
    """The 1-D bilinear mass matrix on a side's n + 1 nodes, parameter t in [0, 1]."""
    h = 1.0 / n
    Mm = np.zeros((n + 1, n + 1))
    for i in range(n):
        Mm[i:i + 2, i:i + 2] += h * np.array([[1.0 / 3, 1.0 / 6], [1.0 / 6, 1.0 / 3]])
    return Mm


def project_side(pb: PortBasis, s: int, g: np.ndarray) -> dict[int, float]:
    """Coefficients of nodal data ``g`` (n + 1 values on side s, t = 0 .. 1) in the
    columns that live on side s, by L2 projection in the side's parameter, the corner
    values fixed for the sine-hat basis.  Returns {column: coefficient}."""
    n = pb.n
    pos = side_positions(n, s)
    g = np.asarray(g, float)
    Mm = side_mass(n)
    out: dict[int, float] = {}
    if pb.kind == "sine-hat":
        c0, c1 = pb.vertex_column(s), pb.vertex_column((s + 1) % 4)
        out[c0], out[c1] = float(g[0]), float(g[-1])
        tt = np.linspace(0.0, 1.0, n + 1)
        r = g - (g[0] * (1 - tt) + g[-1] * tt)
        ks = pb.side_columns(s)
        S = pb.Q[pos][:, ks]
        S[-1] = 0.0                                      # the end corner belongs to the hats
        coef = np.linalg.solve(S.T @ Mm @ S, S.T @ Mm @ r)
        out.update({k: float(v) for k, v in zip(ks, coef)})
        return out
    ks = pb.side_columns(s)
    tt = np.linspace(0.0, 1.0, n + 1)
    S = np.column_stack([np.cos(pb.columns[k].j * math.pi * tt) for k in ks])
    coef = np.linalg.solve(S.T @ Mm @ S, S.T @ Mm @ g)
    out.update({k: float(v) for k, v in zip(ks, coef)})
    return out
