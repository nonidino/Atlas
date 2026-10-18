"""The plain square's exact discrete operator, inverted by fast sine transforms.

On the uniform grid with the identity tensor and a uniform reaction ``s0 = 1/Fo``,
the interior block of `fe.assemble` is a Kronecker sum,

    A_I^0 = K1 (x) M1 + M1 (x) K1 + s0 M1 (x) M1,
    K1 = n tridiag(-1, 2, -1),   M1 = (1 / 6n) tridiag(1, 4, 1)   (size n - 1),

and the type-I discrete sine transform diagonalises both 1-D matrices: on the
vector ``sin(p pi i / n)``, ``K1`` has eigenvalue ``k_p = 2n (1 - cos(p pi / n))`` and
``M1`` has ``m_p = (4 + 2 cos(p pi / n)) / 6n``.  So ``A_I^0`` is solved by two
transforms and a division by ``k_p m_q + m_p k_q + s0 m_p m_q`` (Lynch, Rice and Thomas,
*Numer. Math.* 6, 1964), exactly, in O(n^2 log n).

Two uses here: the **principal part** ``H0 = (A_I^0)^-1 B^0 Q``, the exact constraint
modes of the plain square at the piece's Fourier number (the design's
``H_k^(0)``), and the **fast-solve output form** of comparison C7, whose correction is
``(A_I^0)^-1`` applied to the network's field.  `scripts/arch_family_checks.py`
(`dst_solver`) measured the same transform as a conjugate-gradient preconditioner.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import scipy.fft as sfft

from . import fe


@functools.lru_cache(maxsize=None)
def _eig(n: int) -> tuple[np.ndarray, np.ndarray]:
    c = np.cos(np.arange(1, n) * math.pi / n)
    return 2.0 * n * (1.0 - c), (4.0 + 2.0 * c) / (6.0 * n)


def symbol(n: int, s0: float) -> np.ndarray:
    """The eigenvalues of A_I^0 on the sine basis, shape (n-1, n-1) [q (eta), p (xi)]."""
    k, m = _eig(n)
    return np.outer(m, k) + np.outer(k, m) + s0 * np.outer(m, m)


def solve(n: int, r: np.ndarray, s0: float = 0.0) -> np.ndarray:
    """(A_I^0)^-1 r for interior fields r of shape ((n-1)^2,) or ((n-1)^2, k)."""
    r = np.asarray(r, float)
    k = r.shape[1:] if r.ndim > 1 else ()
    R = r.reshape((n - 1, n - 1) + k)
    lam = symbol(n, s0).reshape((n - 1, n - 1) + (1,) * len(k))
    X = sfft.dstn(sfft.dstn(R, type=1, axes=(0, 1), norm="ortho") / lam, type=1,
                  axes=(0, 1), norm="ortho")
    return X.reshape(r.shape)


@functools.lru_cache(maxsize=64)
def _coupling(n: int) -> "object":
    """The stiffness part of M^0[I, G] (identity tensor).  The mass also couples the
    first interior ring to the loop, so `modes` adds s0 times `_mass_coupling`."""
    loop, inner = fe.grid(n)
    K = fe.assemble(n, 1.0, 0.0, 1.0)
    return K[inner][:, loop].tocsr()


@functools.lru_cache(maxsize=64)
def _mass_coupling(n: int) -> "object":
    loop, inner = fe.grid(n)
    Mm = fe.mass(n, 1.0)
    return Mm[inner][:, loop].tocsr()


def modes(n: int, Q: np.ndarray, s0: float = 0.0) -> np.ndarray:
    """H0 = (A_I^0)^-1 B^0 Q, with B^0 = -M^0[I, G] and M^0 the plain square's energy
    at reaction s0."""
    MIG = _coupling(n) + s0 * _mass_coupling(n)
    return -solve(n, np.asarray(MIG @ Q), s0)
