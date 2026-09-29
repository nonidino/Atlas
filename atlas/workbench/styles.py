"""The showcase plan's coupling styles B, C and D, as drivers over a family's pieces.

**The solver-family interface, derived from two real families** (the plan's
rule: build it from the wind farm and conduction, not in advance).  Each slot,
and what fills it in each family:

=====================  ===========================================  ===========================================
slot                   the wind farm (`families/windfarm.py`)        conduction (`families/conduction.py`)
=====================  ===========================================  ===========================================
state                  the global velocity field ``(u, v)``          the global temperature vector
restriction            ``RectangleTiling.cut`` of the global state   ``fv.assemble(cells=window)`` and ``u[idx]``
a step given           ``WindowNS.step_batch`` with the ring taken   a local sparse solve: ``b - C u`` for a
boundary traces        from the global state (``bc0=None``)          window (Schwarz), a face value or a face
                                                                     flow for a Dirichlet-Neumann piece
the full-domain        ``RectangularNS`` on the undivided domain     ``fv.assemble`` on every cell
equivalent
assembly               the partition of unity, then the global       the partition of unity (overlapping) or
                       projection                                    the disjoint union (non-overlapping)
a balance              incompressibility in the enforcing operator   heat in through the boundary against heat
                                                                     stored and produced
=====================  ===========================================  ===========================================

The two families share no code at the solver level, and should not: the wind
farm's step is W346's to the bit.  What they share is the runner's side of the
interface (``initial``, ``step``, ``observe``, ``field``, ``compare``;
`runner.py`) and, for the families built on `fv.py`, these drivers.

**The drivers**, one per style, each over already-factorized local systems:

``schwarz``           style B: restricted additive Schwarz iterated to a
                      tolerance, every window solved from the same iterate and
                      blended by the partition of unity.  Additive, so the
                      windows may be solved in any order or on threads and the
                      iterate is the same to the bit.
``dirichlet_neumann`` style C: two non-overlapping pieces, the Dirichlet one
                      given the interface's face values, the Neumann one given
                      the flow the Dirichlet one sent across, the face values
                      relaxed (fixed or Aitken) until they agree.
``field_lumped``      style D: a field's boundary integral is a lumped part's
                      port variable; the field is given the port efforts, the
                      lumped network the port flows, relaxed the same way.

Every driver records its convergence history, which the page draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import scipy.sparse.linalg as spla

from .fv import LocalSystem


class Factor:
    """A sparse LU of a local matrix, built once per run.

    SuperLU with its defaults.  The same matrix always gives the same factors and
    the same solves, which is what the one-window control rests on."""

    def __init__(self, A):
        self.lu = spla.splu(A.tocsc())

    def solve(self, b: np.ndarray) -> np.ndarray:
        return self.lu.solve(np.asarray(b, dtype=float))


@dataclass
class Iteration:
    """What a driver returns: the answer and how it got there."""

    u: np.ndarray
    iterations: int
    converged: bool
    history: list[float] = field(default_factory=list)
    #: per iteration, the relaxation factor used (styles C and D)
    relaxation: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# style B
# ---------------------------------------------------------------------------


def schwarz(systems: Sequence[LocalSystem], factors: Sequence[Factor],
            chi: Sequence[np.ndarray], u0: np.ndarray, tol: float, max_it: int,
            scale: float | None, extra_b: Sequence[np.ndarray] | None = None,
            pool=None, keep: np.ndarray | None = None) -> Iteration:
    """``u <- sum_i R_i^T chi_i A_i^{-1} (b_i - C_i u)`` until the update is below
    ``tol * scale`` (max norm), or ``max_it`` sweeps.  ``scale=None`` measures the
    update against the new iterate's own largest entry, for a field with no
    scale known in advance (a structure's displacement).

    ``extra_b[i]`` is added to window i's right-hand side (a backward-Euler
    ``cap dx^2 / dt * u_old`` restricted to it).  With ``pool`` the windows are
    solved on threads; the blend is always summed in window order afterwards,
    so the iterate does not depend on which thread finished first.

    ``keep``: cells no window holds -- a drawn domain's cells outside it -- carried
    through unchanged, so they neither read as an update nor lose their value.
    """
    u = np.array(u0, dtype=float)
    hist: list[float] = []
    n = len(systems)

    def local(i):
        s = systems[i]
        r = s.b if extra_b is None else s.b + extra_b[i]
        if s.C is not None:
            r = r - s.C @ u
        return factors[i].solve(r)

    for it in range(1, max_it + 1):
        vs = list(pool.map(local, range(n))) if pool is not None else [local(i)
                                                                        for i in range(n)]
        new = np.zeros_like(u)
        for i in range(n):
            new[systems[i].idx] += chi[i] * vs[i]
        if keep is not None:
            new[keep] = u[keep]
        den = scale if scale is not None else max(float(np.max(np.abs(new))), 1e-300)
        upd = float(np.max(np.abs(new - u))) / den
        hist.append(upd)
        u = new
        if upd <= tol:
            return Iteration(u, it, True, hist)
    return Iteration(u, max_it, False, hist)


# ---------------------------------------------------------------------------
# style C
# ---------------------------------------------------------------------------


def pair_faces(sd: LocalSystem, sn: LocalSystem) -> np.ndarray:
    """For each of the Dirichlet piece's cut faces, the index of the same face in
    the Neumann piece's list.  A face is its two cells, so the pairing is exact."""
    key_n = {(int(sn.face_outside[g]), int(sn.idx[sn.face_rows[g]])): g
             for g in range(sn.face_rows.size)}
    out = np.empty(sd.face_rows.size, dtype=np.int64)
    for f in range(sd.face_rows.size):
        k = (int(sd.idx[sd.face_rows[f]]), int(sd.face_outside[f]))
        if k not in key_n:
            raise ValueError("the two pieces do not share this face; a Dirichlet-Neumann "
                             "pair must meet along faces, cell for cell")
        out[f] = key_n[k]
    if sorted(out.tolist()) != list(range(sn.face_rows.size)):
        raise ValueError("the Neumann piece has cut faces the Dirichlet piece does not; "
                         "style C couples exactly two pieces along one interface")
    return out


def _aitken(theta: float, r_old: np.ndarray, r_new: np.ndarray) -> float:
    """Aitken's dynamic relaxation (Irons and Tuck), with the previous factor."""
    d = r_new - r_old
    den = float(d @ d)
    if den == 0.0:
        return theta
    return -theta * float(r_old @ d) / den


def dirichlet_neumann(sd: LocalSystem, fd: Factor, sn: LocalSystem, fn: Factor,
                      lam0: np.ndarray, tol: float, max_it: int, scale: float,
                      theta0: float = 0.5, aitken: bool = True,
                      extra_bd: np.ndarray | None = None,
                      extra_bn: np.ndarray | None = None,
                      pairing: np.ndarray | None = None,
                      n_total: int | None = None) -> Iteration:
    """Dirichlet-Neumann across one non-overlapping interface.

    The Dirichlet piece is solved with the interface's face values ``lam``; the
    flow it sends through each face, ``g_D (u_D - lam)``, is imposed on the
    Neumann piece; the Neumann piece's own face value ``u_N + q / g_N`` is the
    new estimate, and ``lam`` moves toward it by the relaxation factor.  At
    convergence the face flow is ``(u_P - u_Q) / (1/g_D + 1/g_N)`` -- the
    harmonic-mean face of the undivided discretization -- so the converged pair
    IS the full-domain solution, to the tolerance.
    """
    pr = pair_faces(sd, sn) if pairing is None else pairing
    rows_n = sn.face_rows[pr]
    g_n = sn.face_g[pr]
    lam = np.array(lam0, dtype=float)
    theta = float(theta0)
    hist, thetas = [], []
    r_old = None
    ud = un = None
    for it in range(1, max_it + 1):
        rd = sd.b.copy() if extra_bd is None else sd.b + extra_bd
        np.add.at(rd, sd.face_rows, sd.face_g * lam)
        ud = fd.solve(rd)
        q = sd.face_g * (ud[sd.face_rows] - lam)            # flow D -> N, per face
        rn = sn.b.copy() if extra_bn is None else sn.b + extra_bn
        np.add.at(rn, rows_n, q)
        un = fn.solve(rn)
        lam_n = un[rows_n] + q / g_n
        r = lam_n - lam
        res = float(np.max(np.abs(r))) / scale
        hist.append(res)
        if res <= tol:
            thetas.append(0.0)
            return Iteration(_union(sd, ud, sn, un, n_total), it, True, hist, thetas,
                             {"lam": lam, "flow": q})
        if aitken and r_old is not None:
            theta = _aitken(theta, r_old, r)
        thetas.append(theta)
        lam = lam + theta * r
        r_old = r
    return Iteration(_union(sd, ud, sn, un, n_total), max_it, False, hist, thetas,
                     {"lam": lam, "flow": q})


def _union(sa: LocalSystem, ua: np.ndarray, sb: LocalSystem, ub: np.ndarray,
           n: int | None = None) -> np.ndarray:
    """The two pieces as one field; NaN where neither is (a drawn domain's void,
    when ``n`` is the whole grid's length)."""
    if n is None:
        n = int(max(sa.idx.max(), sb.idx.max())) + 1
    u = np.full(n, np.nan)
    u[sa.idx] = ua
    u[sb.idx] = ub
    return u


# ---------------------------------------------------------------------------
# style D
# ---------------------------------------------------------------------------


def field_lumped(field_solve: Callable[[np.ndarray], tuple[np.ndarray, object]],
                 lumped_solve: Callable[[np.ndarray], np.ndarray],
                 x0: np.ndarray, tol: float, max_it: int, scale: float,
                 theta0: float = 0.5, aitken: bool = True) -> Iteration:
    """A field joined to lumped parts through its ports.

    ``x`` are the port efforts (for a resistive plate, its electrodes'
    potentials).  ``field_solve(x)`` returns the port flows -- each a boundary
    integral of the field (the current through an electrode) -- and the field;
    ``lumped_solve(flows)`` returns the efforts the lumped network puts on the
    ports when those flows cross them.  The efforts are relaxed toward the
    network's until the two agree.
    """
    x = np.array(x0, dtype=float)
    theta = float(theta0)
    hist, thetas = [], []
    r_old = None
    flows, fld = None, None
    for it in range(1, max_it + 1):
        flows, fld = field_solve(x)
        x_new = lumped_solve(flows)
        r = x_new - x
        res = float(np.max(np.abs(r))) / scale
        hist.append(res)
        if res <= tol:
            thetas.append(0.0)
            return Iteration(np.asarray(fld), it, True, hist, thetas,
                             {"efforts": x, "flows": flows})
        if aitken and r_old is not None:
            theta = _aitken(theta, r_old, r)
        thetas.append(theta)
        x = x + theta * r
        r_old = r
    return Iteration(np.asarray(fld), max_it, False, hist, thetas,
                     {"efforts": x, "flows": flows})


__all__ = ["Factor", "Iteration", "schwarz", "pair_faces", "dirichlet_neumann",
           "field_lumped"]
