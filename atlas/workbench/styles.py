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


# ---------------------------------------------------------------------------
# style M
# ---------------------------------------------------------------------------


class TwoRate:
    """Style M: one explicit finite-volume update at two rates, refluxed.

    The update is ``u <- u + dt / c (b - A u)``, with ``A u = b`` the system
    `fv.assemble` builds on every cell and ``c`` a cell's capacity times its area.
    The **slow** cells take one step of the macro-step ``dt``.  The **fast** cells
    take ``m`` steps of ``dt / m``, reading their slow neighbours linearly
    interpolated in time between the macro-step's two ends.  Then every face between
    the two sets is **refluxed** (Berger & Colella, *J. Comput. Phys.* 82, 1989): the
    slow cell had taken one flux over ``dt``, from the values at the start, and is
    given instead the sum of the ``m`` fluxes the fast cell took.  What one side lost
    the other gained, so the balance closes to round-off.  What moves is the
    agreement with the full domain: the interface is interpolated in time.

    **The rows are the full matrix's own** (``A[rows]``).  With ``m = 1`` both sets
    read the same values at the same instant, every reflux is ``x - x = 0``, and the
    step is the full domain's (`full_steps`) to the bit: the N = 1 control.  A step
    costs one product with the slow rows and ``m`` with the fast rows, and touches
    only the slow cells next to the fast ones (the halo) and the faces between.

    ``faces`` are the interior faces ``(P, Q, G, F)`` in the system's numbering, as
    `fv.interior_faces` gives them (``F`` the advective flow from P to Q).
    ``bfaces`` are the boundary faces ``(cell, kind, value, g, inflow)`` in the same
    numbering, for the ledger: each step returns the heat (or mass) that entered
    through them over the macro-step, summed the way each cell took its steps.
    """

    def __init__(self, f, A, b: np.ndarray, cap: np.ndarray, fast: np.ndarray, m: int,
                 dt: float, faces, bfaces):
        from . import fv
        self._inflow = lambda u, bf: fv._face_inflow(f, u, *bf)   # noqa: E731
        A = A.tocsr()
        n = A.shape[0]
        self.n, self.m, self.dt = n, int(m), float(dt)
        if self.m < 1:
            raise ValueError("a piece takes at least one step per macro-step")
        is_fast = np.zeros(n, dtype=bool)
        is_fast[np.asarray(fast, dtype=np.int64)] = True
        self.fast, self.slow = np.flatnonzero(is_fast), np.flatnonzero(~is_fast)
        self.A_f, self.A_s = A[self.fast], A[self.slow]
        self.b_f, self.b_s = b[self.fast], b[self.slow]
        self.c_f = (self.dt / self.m) / cap[self.fast]
        self.c_s = self.dt / cap[self.slow]
        self.cap_s = cap[self.slow]
        pos = np.full(n, -1, dtype=np.int64)
        pos[self.slow] = np.arange(self.slow.size)
        # the halo: the slow cells a fast row reads
        cols = np.unique(self.A_f.indices)
        self.halo = cols[~is_fast[cols]]
        self.halo_pos = pos[self.halo]
        # the faces between the sets, seen from the slow side: what the slow cell
        # gains per unit of the fast cell's value, and loses per unit of its own
        P, Q, G, F = (np.asarray(x) for x in faces)
        a = ~is_fast[P] & is_fast[Q]
        c = is_fast[P] & ~is_fast[Q]
        self.fs = np.concatenate([Q[a], P[c]])
        self.ss = np.concatenate([P[a], Q[c]])
        self.in_s = np.concatenate([G[a] + np.maximum(-F[a], 0.0),
                                    G[c] + np.maximum(F[c], 0.0)])
        self.out_s = np.concatenate([G[a] + np.maximum(F[a], 0.0),
                                     G[c] + np.maximum(-F[c], 0.0)])
        self.ss_pos = pos[self.ss]
        cell = np.asarray(bfaces[0])
        on_fast = is_fast[cell]
        self.bf_fast = tuple(np.asarray(x)[on_fast] for x in bfaces)
        self.bf_slow = tuple(np.asarray(x)[~on_fast] for x in bfaces)
        # the slow step is taken on every row, a fast row's coefficient zero, which
        # spares gathering and scattering the slow cells (most of the domain)
        self.A, self.b = A, np.asarray(b, dtype=float)
        self.c_slow_all = np.zeros(n)
        self.c_slow_all[self.slow] = self.dt / cap[self.slow]
        #: the fast rows read only fast and halo cells, so only those are ever set
        self._w = np.zeros(n)
        self.ss_u, self.ss_inv = np.unique(self.ss, return_inverse=True)
        self.cap_ss_u = cap[self.ss_u]

    def step(self, u: np.ndarray) -> tuple[np.ndarray, float]:
        """One macro-step: the new state, and what entered through the boundary."""
        u = np.asarray(u, dtype=float)
        dts = self.dt / self.m
        # the slow cells' one step (a fast row is left as it was, and replaced below)
        out = u + self.c_slow_all * (self.b - self.A @ u)
        q_in = self.dt * float(np.sum(self._inflow(u, self.bf_slow)))
        w = self._w
        uf = u[self.fast]
        uh0 = u[self.halo]
        w[self.fast] = uf
        w[self.halo] = uh0
        dh = out[self.halo] - uh0
        acc = np.zeros(self.fs.size)
        for k in range(self.m):
            if k:
                w[self.halo] = uh0 + (k / self.m) * dh
                w[self.fast] = uf
            acc += self.in_s * w[self.fs] - self.out_s * w[self.ss]
            q_in += dts * float(np.sum(self._inflow(w, self.bf_fast)))
            uf = uf + self.c_f * (self.b_f - self.A_f @ w)
        coarse = self.in_s * u[self.fs] - self.out_s * u[self.ss]
        corr = dts * acc - self.dt * coarse
        out[self.ss_u] += np.bincount(self.ss_inv, weights=corr,
                                      minlength=self.ss_u.size) / self.cap_ss_u
        out[self.fast] = uf
        return out, q_in


def full_steps(f, A, b: np.ndarray, cap: np.ndarray, m: int, dt: float, u: np.ndarray,
               bfaces) -> tuple[np.ndarray, float]:
    """The full domain's macro-step for style M: ``m`` explicit steps of ``dt / m`` on
    every cell (the most restrictive cell's step, everywhere), and what entered
    through the boundary.  The arithmetic `TwoRate` reproduces at ``m = 1``."""
    from . import fv
    u = np.asarray(u, dtype=float)
    dts = dt / m
    c = dts / cap
    q_in = 0.0
    for _k in range(int(m)):
        q_in += dts * float(np.sum(fv._face_inflow(f, u, *bfaces)))
        u = u + c * (b - A @ u)
    return u, q_in


__all__ = ["Factor", "Iteration", "schwarz", "pair_faces", "dirichlet_neumann",
           "field_lumped", "TwoRate", "full_steps"]
