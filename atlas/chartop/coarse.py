"""The coarse level of the two-level, coarse-exact CO-D expert (Stage 0 reworked).

[[chart-operator-stage-0-rework]] section 3.2.  Nothing here is learned, and no pinned
module is changed: `forms`, `model`, `fe`, `data` and `chart` are only read.

**The coarse space.**  ``n_c`` divides ``n``, both powers of two.  ``P`` [(n-1)^2,
(n_c-1)^2] is bilinear interpolation from the coarse grid's interior nodes to the fine
grid's interior nodes, the coarse boundary held at zero: the 1-D hats

    p[i, J] = max(0, 1 - |i/n - J/n_c| n_c),     i = 1..n-1,  J = 1..n_c-1,

and ``P = kron(p, p)``, in `fe.grid`'s interior order (the row index ``j`` major, then
``i``), the coarse unknowns likewise (``J`` major, then ``I``).  Its range ``V_c`` is the
fine interior fields that are bilinear on the coarse cells and zero on the loop.  The
piece's **Galerkin coarse operator** is ``A_c = P^T A_I P``, assembled from the piece's
own operator, so it carries the piece's shear, stretch, material and reaction.

**The coarse correction.**  For interior fields ``X`` whose right-hand side is ``F``
(``B Q`` for the modes, ``b_1`` for the level field, ``b_I`` for the particular field),

    C(X) = X + P A_c^-1 P^T (F - A_I X).

It needs only the fine residual ``r = F - A_I X`` and one coarse solve.  The residual is
computed **from the full field**: ``r = rhs - (M u_full)_I`` with ``u_full`` the interior
values ``X`` and the loop values attached (the port basis ``Q`` for the modes, zero for
the fields), so that ``B Q - A_I X`` is formed without forming ``B``.  Writing
``E = X - H``, ``C(X) - H = Q_c E`` with ``Q_c = I - P A_c^-1 P^T A_I``, the
``A_I``-orthogonal projector onto the ``A_I``-orthogonal complement of ``V_c``.

**Proposition A** (the rework page; checked in ``tests/test_chartop_coarse.py``), for
any input:
1. Galerkin orthogonality: ``P^T (F - A_I C(X)) = 0``;
2. never worse, column by column: ``||H_k - C(X)_k||_A = min_{v in V_c} ||H_k - X_k - v||_A``;
3. idempotent, and blind to coarse shifts: ``C(C(X)) = C(X)``, ``C(X + P v) = C(X)``;
4. exact at the top: ``n_c = n`` gives ``P = I`` and ``C(X) = A_I^-1 F``;
5. traces untouched: ``P`` maps into the interior;
6. affine, so the gradient through it is ``Q_c^T`` of the upstream gradient.

**Lemma B and the level field's right-hand side.**  ``Q_c`` acts on every column through
the same operator, so if the inputs satisfy the kernel identity ``X c1 + s0 X_1 = 1``
and the exact fields do too, the outputs do.  The exact fields satisfy it in the
discrete system only if ``s0 b_1 = (M 1)_I``.  The generator stores ``sigma`` and ``J~``
in float32 separately (`data.arrays`), so ``s0 mass(J~) 1`` misses ``(M 1)_I`` by the
float32 rounding (``2e-10``-``3e-8`` of the identity, the corner controls' record).  The
rework therefore defines

    b_1 = (M 1)_I / s0          (s0 > 0),

the reaction's own row sums, ``mass(sigma / s0) 1``: `level_rhs`.  Steady pieces have no
level field here.

**CX**, the coarse exactness of an output, label-free: ``||P^T (F - A_I X)|| /
||P^T F||`` over all columns (Frobenius), `coarse_exactness`.  A field with no data
(``P^T F = 0``, a sourceless particular field) has no ratio; the absolute value is
reported instead.

**For the host**, `CoarseCorrected` wraps any expert: each piece's answers pass through
``C`` against the piece's own matrix, load and (on a step) ``b_1``.  Around U's untrained
form (S's form with zero heads) it is **TL0**, the fully classical two-level expert;
``n_c=None`` returns the inner answers bit for bit.

**The enlarged coarse space (branch 2c, the rework page's section 15).**  Proposition A's
proof uses only that ``P`` has full column rank.  So ``P_e = [P | S]``, with ``S``'s
columns interior fields (zero on the loop), gives a correction ``C_e`` with all of it:
Galerkin orthogonality on ``V_e = range P_e``, idempotence and blindness to shifts in
``V_e``, traces untouched.  It is also never worse than ``C``, column by column, since it
minimises the same error over a larger set.  Lemma B holds unchanged, since every column
still sees the same operator.  `CornerTreatment` builds ``S`` at a piece's **non-right
corners**, those more than 2 degrees from 90 (`chart.corner_angles`, as the section 12
regions read them):
- **A**, singular functions: per corner, ``K`` columns ``chi(r/R) rho^(k pi/omega)
  sin(k pi phi/omega)``, ``k = 1..K``, sampled at the interior nodes and scaled to a
  largest absolute value of one.  Take the corner's own frame: the corner at the origin,
  the piece in the positive quadrant.  There the corner element's tensor ``A``, frozen and
  scaled to determinant one, maps the quadrant by ``y = A^(-1/2) x`` onto a wedge of angle
  ``omega``, with ``cos omega = -a12 / sqrt(a11 a22)`` in that frame (`corner_wedge`).
  ``(rho, phi)`` are ``y``'s polar coordinates, ``phi`` measured from the image of the side
  along which ``eta`` is fixed.  ``r = |x|`` in the reference square, ``R = 1/2``, and
  ``chi`` is one inside ``R/2``, zero beyond ``R``, with a cubic smoothstep between.
- **B**, corner patches: per corner, the fine hats at the interior nodes within ``w = n /
  frac`` of the corner in both directions (``w^2`` nodes), less the coarse nodes among
  them.  This is a hierarchical complement, so ``P_e`` keeps full column rank.

``S`` is empty on a piece with no non-right corner and at ``n_c = n``, where ``V_c`` is
already the whole interior.  Every path is then the coarse grid's own, bit for bit.  The
enlarged space is numpy only: the torch path (training) has the coarse grid alone.

**Torch, batched, float64.**  `apply_M` is ``M U`` from the element formulas of
`model.energy` (half the energy's gradient), with deterministic slice additions, no
``scatter_add``.  `coarse_matrix_torch` assembles ``A_c`` by probing with nine
colourings of the coarse hats: ``A_c`` has a nine-point stencil, and hats three apart
in both directions never meet an element together.  `TorchCoarseSpace` holds the
batched Cholesky factor; its `correct` is differentiable, so ``C`` sits in the graph.
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import fe

#: the coarse sizes the owner decided (16 the default); 4 is added for the curve n_c*
COARSE_SIZES = (8, 16, 32)
DEFAULT_N_C = 16
CURVE_N_C = (4, 8, 16, 32)


def check_sizes(n: int, n_c: int) -> None:
    """``n_c`` must divide ``n``, both powers of two, and ``2 <= n_c <= n``."""
    def pow2(k):
        return k >= 1 and (k & (k - 1)) == 0
    if not (pow2(n) and pow2(n_c) and 2 <= n_c <= n):
        raise ValueError(f"the coarse grid needs n and n_c powers of two with 2 <= n_c <= n, "
                         f"not n = {n}, n_c = {n_c}")


# ---------------------------------------------------------------------------
# the prolongation
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def prolongation_1d(n: int, n_c: int) -> np.ndarray:
    """``p`` [n-1, n_c-1]: the 1-D coarse hats at the fine interior nodes,
    ``p[i-1, J-1] = max(0, 1 - |i/n - J/n_c| n_c)``.  With ``r = n / n_c`` this is
    ``max(0, (r - |i - J r|) / r)``, computed from integers, so every entry is exact."""
    check_sizes(n, n_c)
    r = n // n_c
    i = np.arange(1, n)[:, None]
    J = np.arange(1, n_c)[None, :]
    p = np.maximum(0, r - np.abs(i - J * r)).astype(np.float64) / r
    p.setflags(write=False)
    return p


@functools.lru_cache(maxsize=None)
def prolongation(n: int, n_c: int) -> sp.csr_matrix:
    """``P = kron(p, p)`` [(n-1)^2, (n_c-1)^2], sparse: row ``(j-1)(n-1) + (i-1)`` is the fine
    interior node ``(i, j)`` in `fe.grid`'s order, column ``(J-1)(n_c-1) + (I-1)`` the
    coarse node ``(I, J)``.  Cached: do not modify the returned matrix."""
    p = sp.csr_matrix(prolongation_1d(n, n_c))
    return sp.kron(p, p, format="csr")


# ---------------------------------------------------------------------------
# the fine residual (numpy)
# ---------------------------------------------------------------------------


def residual(M, n: int, X, loop_values=None, rhs=None) -> np.ndarray:
    """``r = rhs - (M u_full)_I`` for interior fields ``X`` ((n-1)^2,) or ((n-1)^2, k), with
    ``u_full`` ``X`` on the interior and ``loop_values`` (4n,) or (4n, k) on the loop (zero
    when ``None``), and ``rhs`` zero when ``None``.  For the modes (``loop_values = Q``,
    no ``rhs``) this is ``B Q - A_I X``; for a field (no loop values) ``rhs - A_I X``."""
    loop, inner = fe.grid(n)
    X = np.asarray(X, np.float64)
    if loop_values is None:
        loop_values = np.zeros((4 * n,) + X.shape[1:])
    U = fe.full_field(n, X, loop_values)
    r = -np.asarray(M @ U)[inner]
    if rhs is not None:
        r = r + np.asarray(rhs, np.float64)
    return r


def level_rhs(M, n: int, s0: float) -> np.ndarray:
    """``b_1 = (M 1)_I / s0`` ((n-1)^2,): the level field's right-hand side, the reaction's
    own row sums (``mass(sigma / s0) 1``), for which ``H c1 + s0 u_1 = 1`` holds exactly
    in the discrete system (Lemma B's condition).  Steady pieces have no level field:
    ``s0 <= 0`` is refused."""
    if not s0 > 0.0:
        raise ValueError(f"the level field needs s0 > 0 (a step), not {s0!r}: "
                         "steady pieces have no level field")
    _loop, inner = fe.grid(n)
    return np.asarray(M @ np.ones(M.shape[0]))[inner] / s0


# ---------------------------------------------------------------------------
# the coarse space of one piece (numpy)
# ---------------------------------------------------------------------------


class CoarseSpace:
    """One piece's coarse level: ``P``, ``A_c = P^T A_I P`` (symmetrised) and its Cholesky
    factor (`scipy.linalg.cho_factor`), all float64.  ``A_I`` is the piece's interior
    block ((n-1)^2 square, sparse or dense).

    At ``n_c = n`` (the positive control) the coarse space is the whole interior, ``P = I``
    and ``A_c = A_I``; it is then kept sparse and factored by a sparse LU, since a dense
    factor of ``(n-1)^2`` unknowns would not fit.

    Built with `of_matrix`, it also keeps the **restricted operator** ``G = P^T M[I, :]``
    (coarse rows), split into its interior and loop columns ``G_I = P^T A_I`` and
    ``G_G = P^T M[I, G]``, so that `apply` forms the coarse right-hand side
    ``P^T (rhs - (M u_full)_I) = P^T rhs - G_I X - G_G (loop values)`` without forming the
    fine residual or the full field, and ``A_c = G_I P``: the same numbers as `correct`
    with `residual`, to rounding, at about half the work.

    ``extra`` [(n-1)^2, m] (branch 2c, `CornerTreatment`) enlarges the space to ``P_e =
    [P | extra]``: ``P`` is then ``P_e``, ``A_c`` is ``P_e^T A_I P_e`` and every method
    works on ``V_e``; ``n_extra`` counts the added columns.  ``None`` or no columns is the
    coarse grid alone, bit for bit; columns at ``n_c = n`` are refused."""

    def __init__(self, A_I, n: int, n_c: int, _restricted=None, _A_c=None, extra=None):
        check_sizes(n, n_c)
        extra = _extra_columns(extra, n, n_c)
        self.n, self.n_c = n, n_c
        self.P = prolongation(n, n_c)
        self.n_extra = 0
        if extra is not None:
            self.P = sp.hstack([self.P, extra], format="csr")
            self.n_extra = extra.shape[1]
        self.top = n_c == n
        if A_I is not None:
            A = sp.csr_matrix(A_I, dtype=np.float64)
            if A.shape != ((n - 1) ** 2,) * 2:
                raise ValueError(f"A_I is {A.shape}, not the interior block of n = {n}")
        elif self.top or _A_c is None:
            raise ValueError("the coarse space needs A_I")
        if self.top:
            self.A_c = A.tocsc()
            self._lu = spla.splu(self.A_c)
            self.factor = None
        else:
            Ac = np.asarray((self.P.T @ (A @ self.P)).todense()) if _A_c is None else _A_c
            self.A_c = 0.5 * (Ac + Ac.T)
            self.factor = sla.cho_factor(self.A_c, lower=True)
        self._G_I, self._G_G = _restricted if _restricted is not None else (None, None)

    @classmethod
    def of_matrix(cls, M, n: int, n_c: int, extra=None) -> "CoarseSpace":
        """From the piece's whole energy matrix ``M`` (``(n+1)^2`` square), keeping the
        restricted operator for `apply`; ``A_c = G_I P`` (``P_e`` with ``extra``)."""
        check_sizes(n, n_c)
        extra = _extra_columns(extra, n, n_c)
        loop, inner = fe.grid(n)
        M_I = sp.csr_matrix(M, dtype=np.float64)[inner]
        P = prolongation(n, n_c)
        if extra is not None:
            P = sp.hstack([P, extra], format="csr")
        G = (P.T @ M_I).tocsc()
        G_I, G_G = G[:, inner].tocsr(), G[:, loop].tocsr()
        if n_c == n:
            return cls(M_I[:, inner], n, n_c, _restricted=(G_I, G_G))
        return cls(None, n, n_c, _restricted=(G_I, G_G), _A_c=(G_I @ P).toarray(), extra=extra)

    @property
    def size(self) -> int:
        return self.P.shape[1]

    def solve(self, rc) -> np.ndarray:
        """``A_c^-1 rc`` for coarse vectors ``rc`` ((n_c-1)^2,) or ((n_c-1)^2, k)."""
        rc = np.asarray(rc, np.float64)
        if self.top:
            return self._lu.solve(rc)
        return sla.cho_solve(self.factor, rc)

    def correction(self, r) -> np.ndarray:
        """``P A_c^-1 P^T r`` for fine residuals ``r``."""
        return np.asarray(self.P @ self.solve(self.P.T @ np.asarray(r, np.float64)))

    def correct(self, X, r) -> np.ndarray:
        """``C(X) = X + P A_c^-1 P^T r``, ``r`` the fine residual of ``X`` (`residual`)."""
        return np.asarray(X, np.float64) + self.correction(r)

    def coarse_residual(self, X, loop_values=None, rhs=None) -> np.ndarray:
        """``P^T (rhs - (M u_full)_I) = P^T rhs - G_I X - G_G (loop values)`` (needs
        `of_matrix`)."""
        if self._G_I is None:
            raise ValueError("the restricted residual needs the whole matrix: CoarseSpace.of_matrix")
        rc = -np.asarray(self._G_I @ np.asarray(X, np.float64))
        if loop_values is not None:
            rc -= np.asarray(self._G_G @ np.asarray(loop_values, np.float64))
        if rhs is not None:
            rc += np.asarray(self.P.T @ np.asarray(rhs, np.float64))
        return rc

    def apply(self, X, loop_values=None, rhs=None) -> np.ndarray:
        """``C(X)`` for interior fields ``X`` with ``loop_values`` on the loop (``Q`` for the
        modes) and right-hand side ``rhs`` (a field's load), by the restricted residual."""
        X = np.asarray(X, np.float64)
        out = np.asarray(self.P @ self.solve(self.coarse_residual(X, loop_values, rhs)))
        out += X
        return out


def correct_fields(M, n: int, n_c: int, H, up, level, s0: float, b, Q, extra=None):
    """One piece's two-level answer from any answer ``(H, up, level)``: ``C(H)`` with the
    loop values ``Q``, ``C_p(up)`` against the load ``b`` (interior), and on a step
    (``s0 > 0``) ``C_1(level)`` against ``b_1 = (M 1)_I / s0``; a steady piece's level field
    (or ``None``) is returned as it came.  The coarse space is built here from ``M``: this
    is the whole of the coarse level's work for one call on one piece.  ``extra`` enlarges
    it (`CoarseSpace`; branch 2c)."""
    cs = CoarseSpace.of_matrix(M, n, n_c, extra=extra)
    Hc = cs.apply(H, Q)
    upc = cs.apply(up, rhs=b)
    if level is not None and s0 > 0.0:
        level = cs.apply(level, rhs=level_rhs(M, n, s0))
    return Hc, upc, level


class CoarseCorrected:
    """Any expert's answers passed through the coarse correction, piece by piece, in numpy
    float64, for the host (`host.Layout.solve`).  Around U's untrained form this is TL0.

    Per piece, from its own matrix (`PieceProblem.matrix`) and at coarse size ``n_c``:
    - the modes: ``C(H)``, the loop values the port basis ``Q``;
    - the particular field: ``C_p(u_p)`` against the piece's own load ``b_I``
      (`PieceProblem.load`), so it is coarse-exact and, at ``n_c = n``, the exact
      particular field;
    - the level field, on a step: ``C_1(u_1)`` against ``b_1 = (M 1)_I / s0`` (`level_rhs`),
      so that ``H c1 + s0 u_1 = 1`` survives (Lemma B).  A steady piece's level field
      (the unit source's response, which only a march reads) is passed on as it is.

    ``n_c=None`` returns the inner expert's answers unchanged, bit for bit (the control).
    The uniform state's particular field is ``C_p(s0 u_1)`` against ``s0 mass(J~) 1``,
    which equals ``s0 C_1(u_1)`` to the rounding by which ``sigma`` differs from
    ``s0 J~``: one float64 rounding on a `PieceProblem` (``sigma = J~ / Fo``).  The coarse
    level's own time (building the space and the three corrections, from the assembled
    matrix) is summed in ``coarse_seconds``.  A march that keeps modes (``modes_from``) is
    not built here.

    ``corners`` (a `CornerTreatment`, branch 2c) enlarges each piece's coarse space at its
    non-right corners, the columns built from the piece's own chart and tensors inside the
    timed work; ``extra_columns`` sums them.  ``None``, the default, is the coarse grid
    alone, bit for bit."""

    name = "two-level"

    def __init__(self, inner, n_c: int | None, corners=None):
        self.inner, self.n_c, self.corners = inner, n_c, corners
        self.envelope = getattr(inner, "envelope", None)
        self.kind, self.m = getattr(inner, "kind", None), getattr(inner, "m", None)
        self.calls = 0
        self.coarse_seconds = 0.0
        self.extra_columns = 0

    def correct_piece(self, problem, response, Q):
        """One piece's corrected `host.Response` (`correct_fields` on the piece's own
        matrix and load)."""
        from .host import Response
        n = problem.n
        M = problem.matrix()
        _loop, inner = fe.grid(n)
        b = problem.load()[inner]
        t0 = time.perf_counter()
        extra = None if self.corners is None else self.corners.of_problem(problem, self.n_c)
        H, up, level = correct_fields(M, n, self.n_c, response.H, response.up, response.level,
                                      problem.s0, b, Q, extra=extra)
        self.coarse_seconds += time.perf_counter() - t0
        if extra is not None:
            self.extra_columns += extra.shape[1]
        return Response(H, up, level)

    def respond(self, problems, pb_of, modes_from=None):
        if modes_from is not None:
            raise NotImplementedError("a march that keeps the corrected modes is not built")
        resp = self.inner.respond(problems, pb_of)
        self.calls += len(problems)
        if self.n_c is None:
            return resp
        return [self.correct_piece(p, r, pb_of(p.n).Q) for p, r in zip(problems, resp)]


def coarse_exactness(P, r_out, r_data) -> dict:
    """CX of one output: ``||P^T r_out|| / ||P^T r_data||`` (Frobenius over the columns), with
    ``r_out = F - A_I X`` the output's fine residual and ``r_data = F`` the data's own
    (the residual of ``X = 0``).  With no data (``P^T F = 0``) there is no ratio: ``rel``
    is ``None`` and ``num`` is what to read."""
    num = float(np.linalg.norm(P.T @ np.asarray(r_out, np.float64)))
    den = float(np.linalg.norm(P.T @ np.asarray(r_data, np.float64)))
    return {"num": num, "den": den, "rel": (num / den) if den > 0.0 else None}


# ---------------------------------------------------------------------------
# the enlarged coarse space at non-right corners (branch 2c)
# ---------------------------------------------------------------------------

#: a corner within this many degrees of 90 is right (the rework page's sections 12 and 15)
RIGHT_DEG = 2.0
#: candidate A's cut-off radius, in the reference square's units
SINGULAR_R = 0.5
#: per corner (0 at (0, 0), then counter-clockwise, as `chart.corner_angles`): the corner in
#: the reference square, ``(xi, eta)``, and the signs that put the piece in the corner's
#: positive quadrant
CORNERS = (((0, 0), (1, 1)), ((1, 0), (-1, 1)), ((1, 1), (-1, -1)), ((0, 1), (1, -1)))


def _extra_columns(extra, n: int, n_c: int):
    """``extra`` as a float64 csr matrix [(n-1)^2, m], or ``None`` when absent or without
    columns.  Columns at ``n_c = n`` are refused: ``V_c`` is already the whole interior,
    and ``P_e`` would lose full rank."""
    if extra is None:
        return None
    S = sp.csr_matrix(extra, dtype=np.float64)
    if S.shape[0] != (n - 1) ** 2:
        raise ValueError(f"the added columns have {S.shape[0]} rows, not the (n-1)^2 = "
                         f"{(n - 1) ** 2} interior nodes of n = {n}")
    if S.shape[1] == 0:
        return None
    if n_c == n:
        raise ValueError("at n_c = n the coarse space is already the whole interior: "
                         "nothing can be added")
    return S


def non_right_corners(X, tol_deg: float = RIGHT_DEG) -> tuple:
    """The corners of the chart ``X`` [n+1, n+1, 2] whose angle (`chart.corner_angles`) is
    more than ``tol_deg`` from 90 degrees, in corner order."""
    from .chart import corner_angles
    ang = corner_angles(np.asarray(X, np.float64))
    return tuple(c for c in range(4) if abs(float(ang[c]) - 90.0) > tol_deg)


def corner_offsets(n: int, c: int):
    """``(a, b)`` for every interior node, in `fe.grid`'s order: its distance from corner
    ``c`` in fine cells along the side where ``eta`` is fixed (``a``) and the side where
    ``xi`` is fixed (``b``).  Both run over ``1..n-1``."""
    (xc, ec), (s1, s2) = CORNERS[c]
    jj, ii = np.meshgrid(np.arange(1, n), np.arange(1, n), indexing="ij")
    return (s1 * (ii - xc * n)).ravel(), (s2 * (jj - ec * n)).ravel()


def corner_tensors(a11, a12, a22) -> list:
    """Per corner, the corner element's ``(a11, a12, a22)`` from element arrays [n, n]
    (``[j, i]``), in the piece's own frame."""
    n = np.shape(a11)[-1]
    out = []
    for (xc, ec), _signs in CORNERS:
        j, i = ec * (n - 1), xc * (n - 1)
        out.append((float(a11[j, i]), float(a12[j, i]), float(a22[j, i])))
    return out


def corner_wedge(a11: float, a12: float, a22: float, c: int) -> tuple:
    """``(omega, T)`` for corner ``c`` under its frozen tensor: in the corner's own frame
    the tensor is ``A = [[a11, s a12], [s a12, a22]]`` with ``s = s1 s2``, and ``T =
    (A / sqrt(det A))^(-1/2)`` maps the positive quadrant onto a wedge of angle ``omega``,
    ``cos omega = -s a12 / sqrt(a11 a22)``."""
    _corner, (s1, s2) = CORNERS[c]
    b = s1 * s2 * a12
    A = np.array([[a11, b], [b, a22]], np.float64)
    det = float(np.linalg.det(A))
    if not (det > 0.0 and a11 > 0.0):
        raise ValueError(f"corner {c}: the frozen tensor is not positive definite")
    lam, V = np.linalg.eigh(A / math.sqrt(det))
    T = (V / np.sqrt(lam)) @ V.T
    omega = math.acos(max(-1.0, min(1.0, -b / math.sqrt(a11 * a22))))
    return omega, T


def smoothstep_cutoff(t):
    """One for ``t <= 1/2``, zero for ``t >= 1``, the cubic smoothstep ``s^2 (3 - 2 s)``,
    ``s = 2 (1 - t)``, between."""
    s = np.clip(2.0 * (1.0 - np.asarray(t, np.float64)), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def corner_singular_columns(n: int, n_c: int, corners, tensors, K: int,
                            R: float = SINGULAR_R) -> sp.csr_matrix:
    """Candidate A's ``S`` [(n-1)^2, K len(corners)], sparse.  Per corner ``c`` in
    ``corners``, with ``tensors[c]`` its frozen ``(a11, a12, a22)`` (`corner_tensors`):
    ``chi(r/R) rho^nu sin(nu phi)``, ``nu = k pi / omega``, for ``k = 1..K``, at the interior
    nodes.  ``x`` is the node in the corner's frame (`corner_offsets` over ``n``), ``y = T
    x`` (`corner_wedge`), ``(rho, phi)`` its polar coordinates with ``phi`` from ``T e_1``,
    ``r = |x|`` and ``chi`` `smoothstep_cutoff`.  Each column is scaled to a largest
    absolute value of one.  Empty at ``n_c = n`` and without corners."""
    check_sizes(n, n_c)
    if K < 1:
        raise ValueError(f"K must be at least 1, not {K}")
    if n_c == n or not corners:
        return sp.csr_matrix(((n - 1) ** 2, 0))
    cols = []
    for c in corners:
        omega, T = corner_wedge(*tensors[c], c)
        a, b = corner_offsets(n, c)
        x = np.stack([a / n, b / n])
        y = T @ x
        v1 = T[:, 0]
        phi = np.mod(np.arctan2(v1[0] * y[1] - v1[1] * y[0], v1 @ y), 2.0 * math.pi)
        rho = np.hypot(y[0], y[1])
        chi = smoothstep_cutoff(np.hypot(x[0], x[1]) / R)
        for k in range(1, K + 1):
            nu = k * math.pi / omega
            s = chi * rho ** nu * np.sin(nu * phi)
            top = float(np.abs(s).max())
            if not top > 0.0:
                raise ValueError(f"corner {c}, k = {k}: the singular function vanishes at every node")
            cols.append(s / top)
    return sp.csr_matrix(np.column_stack(cols))


def corner_patch_columns(n: int, n_c: int, corners, frac: int) -> sp.csr_matrix:
    """Candidate B's ``S``, sparse, one unit entry per column.  Per corner in ``corners``,
    the fine hats at the interior nodes whose offsets (`corner_offsets`) both lie in
    ``1..w``, ``w = n / frac``, less the coarse nodes among them, which ``P`` already
    holds; corner by corner, then ``b``, then ``a``.  The blocks must not meet (``2 w <
    n``).  At ``n_c = n`` every node is coarse and ``S`` is empty."""
    check_sizes(n, n_c)
    w = n // frac if frac >= 1 else 0
    if not (frac >= 1 and n % frac == 0 and w >= 1 and 2 * w < n):
        raise ValueError(f"corner blocks of n / {frac} do not fit n = {n}")
    r = n // n_c
    rows = []
    for c in corners:
        (xc, ec), _signs = CORNERS[c]
        for b in range(1, w + 1):
            j = b if ec == 0 else n - b
            for a in range(1, w + 1):
                i = a if xc == 0 else n - a
                if i % r == 0 and j % r == 0:
                    continue                                   # a coarse node
                rows.append((j - 1) * (n - 1) + (i - 1))
    m = len(rows)
    return sp.csr_matrix((np.ones(m), (np.asarray(rows, np.int64), np.arange(m))),
                         shape=((n - 1) ** 2, m))


@dataclass(frozen=True)
class CornerTreatment:
    """Branch 2c's enlargement of a piece's coarse space at its non-right corners
    (`non_right_corners`): ``kind`` ``"A"``, ``K`` singular functions per corner
    (`corner_singular_columns`), or ``"B"``, the corner blocks of ``n / frac``
    (`corner_patch_columns`)."""

    kind: str
    K: int = 3
    frac: int = 8
    R: float = SINGULAR_R
    tol_deg: float = RIGHT_DEG

    def __post_init__(self):
        if self.kind not in ("A", "B"):
            raise ValueError(f"a corner treatment is 'A' or 'B', not {self.kind!r}")

    @property
    def name(self) -> str:
        return f"A-K{self.K}" if self.kind == "A" else f"B-n/{self.frac}"

    def columns(self, n: int, n_c: int, X, a11=None, a12=None, a22=None) -> sp.csr_matrix:
        """``S`` for a piece of chart ``X`` [n+1, n+1, 2] and, for A, element tensors [n, n]."""
        corners = non_right_corners(X, self.tol_deg)
        if self.kind == "B":
            return corner_patch_columns(n, n_c, corners, self.frac)
        if not corners:
            return sp.csr_matrix(((n - 1) ** 2, 0))
        return corner_singular_columns(n, n_c, corners, corner_tensors(a11, a12, a22), self.K, self.R)

    def of_problem(self, problem, n_c: int) -> sp.csr_matrix:
        """``S`` for a `host.PieceProblem`, from its chart and its own tensors."""
        t = problem.tensors() if self.kind == "A" else {}
        return self.columns(problem.n, n_c, problem.chart.X, t.get("a11"), t.get("a12"), t.get("a22"))


# ---------------------------------------------------------------------------
# torch, batched, float64
# ---------------------------------------------------------------------------


def _torch():
    import torch
    return torch


def apply_M(a11, a12, a22, sigma, U):
    """``M U`` for nodal fields ``U`` [B, K, n+1, n+1] [j, i] and element arrays [B, n, n]
    (``sigma`` may be ``None``): half the gradient of `model.energy`'s element formulas.

    Per element, with ``d0 = u10 - u00``, ``d1 = u11 - u01``, ``e0 = u01 - u00``,
    ``e1 = u11 - u10``: the ``a11`` part sends ``(2 d0 + d1) / 6`` to ``u10`` (minus to
    ``u00``) and ``(d0 + 2 d1) / 6`` to ``u11`` (minus to ``u01``); the ``a22`` part likewise
    along ``eta``; the ``a12`` part, ``a12 (d0 + d1)(e0 + e1) / 2`` in the energy, sends
    ``a12 (e0 + e1) / 4`` along ``d0 + d1``'s signs and ``a12 (d0 + d1) / 4`` along
    ``e0 + e1``'s; the mass row is ``(1/9, 1/18, 1/36, 1/18)`` around each node, times
    ``sigma / n^2``.  Accumulated by slice additions in a fixed order (deterministic)."""
    torch = _torch()
    n = U.shape[-1] - 1
    a11, a12, a22 = a11[:, None], a12[:, None], a22[:, None]
    u00, u10 = U[..., :-1, :-1], U[..., :-1, 1:]
    u11, u01 = U[..., 1:, 1:], U[..., 1:, :-1]
    d0, d1 = u10 - u00, u11 - u01
    e0, e1 = u01 - u00, u11 - u10
    gx0 = a11 * (2.0 * d0 + d1) / 6.0
    gx1 = a11 * (d0 + 2.0 * d1) / 6.0
    gy0 = a22 * (2.0 * e0 + e1) / 6.0
    gy1 = a22 * (e0 + 2.0 * e1) / 6.0
    hs = a12 * (e0 + e1) / 4.0
    ht = a12 * (d0 + d1) / 4.0
    y00 = -gx0 - gy0 - hs - ht
    y10 = gx0 - gy1 + hs - ht
    y11 = gx1 + gy1 + hs + ht
    y01 = -gx1 + gy0 - hs + ht
    if sigma is not None:
        s = sigma[:, None] / n ** 2
        y00 = y00 + s * (u00 / 9.0 + (u10 + u01) / 18.0 + u11 / 36.0)
        y10 = y10 + s * (u10 / 9.0 + (u00 + u11) / 18.0 + u01 / 36.0)
        y11 = y11 + s * (u11 / 9.0 + (u10 + u01) / 18.0 + u00 / 36.0)
        y01 = y01 + s * (u01 / 9.0 + (u00 + u11) / 18.0 + u10 / 36.0)
    out = torch.zeros_like(U)
    out[..., :-1, :-1] += y00
    out[..., :-1, 1:] += y10
    out[..., 1:, 1:] += y11
    out[..., 1:, :-1] += y01
    return out


def residual_torch(a11, a12, a22, sigma, X, QB=None, rhs=None):
    """``r = rhs - (M u_full)_I`` [B, K, n-1, n-1] for interior fields ``X`` [B, K, n-1, n-1]:
    ``u_full`` has ``X`` inside and ``QB`` [K, n+1, n+1] (`model.PlainSquare.QB`, the port
    basis's boundary images) on the loop, zero when ``None``; ``rhs`` [B, K, n-1, n-1] or
    broadcastable, zero when ``None``."""
    torch = _torch()
    B, K, m1 = X.shape[0], X.shape[1], X.shape[-1]
    if QB is None:
        U = torch.zeros(B, K, m1 + 2, m1 + 2, dtype=X.dtype, device=X.device)
    else:
        U = QB[None].expand(B, -1, -1, -1).to(X.dtype).clone()
    U[..., 1:-1, 1:-1] = X
    r = -apply_M(a11, a12, a22, sigma, U)[..., 1:-1, 1:-1]
    return r if rhs is None else r + rhs


def level_rhs_torch(a11, a12, a22, sigma, s0):
    """(``b_1`` [B, n-1, n-1], ``transient`` [B] bool): ``b_1 = (M 1)_I / s0`` where
    ``s0 > 0``, and zero on steady examples, which have no level field (the caller leaves
    theirs as they are).  The stiffness part of ``M 1`` is exactly zero in `apply_M`'s
    arithmetic, so this is the mass part, ``mass(sigma / s0) 1``."""
    torch = _torch()
    B, n = a11.shape[0], a11.shape[-1]
    ones = torch.ones(B, 1, n + 1, n + 1, dtype=torch.float64, device=a11.device)
    MI = apply_M(a11, a12, a22, sigma, ones)[:, 0, 1:-1, 1:-1]
    s0 = s0.to(torch.float64)
    transient = s0 > 0.0
    safe = torch.where(transient, s0, torch.ones_like(s0))
    b1 = torch.where(transient[:, None, None], MI / safe[:, None, None], torch.zeros_like(MI))
    return b1, transient


@functools.lru_cache(maxsize=None)
def _p_torch(n: int, n_c: int, device: str):
    torch = _torch()
    return torch.tensor(prolongation_1d(n, n_c), dtype=torch.float64, device=torch.device(device))


def prolongation_torch(n: int, n_c: int, device="cpu"):
    """``p`` [n-1, n_c-1] (float64) on ``device``, cached."""
    return _p_torch(n, n_c, str(device))


def prolong(v, p):
    """``P v`` as images: [..., n_c-1, n_c-1] -> [..., n-1, n-1], ``p v p^T``."""
    return p @ v @ p.transpose(0, 1)


def restrict(r, p):
    """``P^T r`` as images: [..., n-1, n-1] -> [..., n_c-1, n_c-1], ``p^T r p``."""
    return p.transpose(0, 1) @ r @ p


def coarse_matrix_torch(a11, a12, a22, sigma, n_c: int):
    """``A_c = P^T A_I P`` [B, (n_c-1)^2, (n_c-1)^2] (float64, symmetrised) from a batch's
    element arrays, by probing: ``A_c`` couples coarse nodes at most one apart in each
    direction, so the hats of each of the nine classes ``(J mod 3, I mod 3)`` are applied
    together, nine `apply_M` calls in all, and each entry is read back from the one class
    that holds its column's node."""
    torch = _torch()
    B, n = a11.shape[0], a11.shape[-1]
    dev = a11.device
    p = prolongation_torch(n, n_c, dev)
    m = n_c - 1
    V = torch.zeros(9, m, m, dtype=torch.float64, device=dev)
    for c in range(9):
        V[c, c // 3::3, c % 3::3] = 1.0
    F = torch.zeros(9, n + 1, n + 1, dtype=torch.float64, device=dev)
    F[:, 1:-1, 1:-1] = prolong(V, p)
    MF = apply_M(a11, a12, a22, sigma, F[None].expand(B, -1, -1, -1))[..., 1:-1, 1:-1]
    R = restrict(MF, p)                                    # [B, 9, m, m]: row node (Ja, Ia)
    Ja, Ia = torch.meshgrid(torch.arange(m, device=dev), torch.arange(m, device=dev), indexing="ij")
    Ja, Ia = Ja.reshape(-1), Ia.reshape(-1)
    rows, cols, cls = [], [], []
    for dJ in (-1, 0, 1):
        for dI in (-1, 0, 1):
            Jb, Ib = Ja + dJ, Ia + dI
            ok = (Jb >= 0) & (Jb < m) & (Ib >= 0) & (Ib < m)
            rows.append((Ja * m + Ia)[ok])
            cols.append((Jb * m + Ib)[ok])
            cls.append(((Jb % 3) * 3 + (Ib % 3))[ok])
    rows, cols, cls = torch.cat(rows), torch.cat(cols), torch.cat(cls)
    Ac = torch.zeros(B, m * m, m * m, dtype=torch.float64, device=dev)
    Ac[:, rows, cols] = R[:, cls, rows // m, rows % m]
    return 0.5 * (Ac + Ac.transpose(1, 2))


class TorchCoarseSpace:
    """A batch's coarse level: the Cholesky factor of ``A_c`` [B, N_c, N_c] (float64,
    `torch.linalg.cholesky`), and ``C`` as a differentiable map of the fine fields.  ``A_c``
    comes from `coarse_matrix_torch`, or precomputed per piece (`CoarseSpace.A_c`)."""

    def __init__(self, A_c, n: int, n_c: int):
        torch = _torch()
        check_sizes(n, n_c)
        if A_c.dtype != torch.float64:
            raise ValueError("the coarse operator is float64")
        self.n, self.n_c = n, n_c
        self.p = prolongation_torch(n, n_c, A_c.device)
        self.L = torch.linalg.cholesky(A_c)

    def solve(self, rc):
        """``A_c^-1 rc`` for ``rc`` [B, N_c, K]."""
        torch = _torch()
        return torch.cholesky_solve(rc, self.L)

    def correction(self, r):
        """``P A_c^-1 P^T r`` for fine residuals ``r`` [B, K, n-1, n-1] (or [B, n-1, n-1])."""
        squeeze = r.dim() == 3
        if squeeze:
            r = r[:, None]
        B, K = r.shape[0], r.shape[1]
        m = self.n_c - 1
        rc = restrict(r, self.p).reshape(B, K, m * m).transpose(1, 2)
        v = self.solve(rc).transpose(1, 2).reshape(B, K, m, m)
        out = prolong(v, self.p)
        return out[:, 0] if squeeze else out

    def correct(self, X, r):
        """``C(X) = X + P A_c^-1 P^T r``."""
        return X + self.correction(r)
