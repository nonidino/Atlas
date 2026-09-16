"""Incompressible Navier-Stokes on body-fitted overset grids.

PoC 3, Tier 61.  [[poc3-racelab-overset-flow]].

Tier 60 (`atlas.cases.overset`) built the grids, the overlap and the composite
pressure system and verified them on an elliptic problem.  This module marches
a flow on them: velocity and pressure on every grid, the grids coupled through
the same interpolation equations, the bodies as real no-slip walls (moving, if
their velocity is given), and forces on each body from the stresses at its wall.

The scheme
----------

Second-order backward differences (BDF2) in time, backward Euler on the first
step, with the ROTATIONAL incremental pressure-correction projection
(Timmermans, Minev and Van De Vosse 1996; analysed with open boundaries by
Guermond, Minev and Shen).  With ``a0 = 3/2`` and the extrapolated advecting
velocity ``u* = 2 u^k - u^(k-1)``:

  1. momentum, implicit in viscosity AND in advection (linearised about u*):
         (a0 u~ - 2 u^k + u^(k-1)/2) / dt + (u* . grad) u~ - nu lap u~
                                                    = -grad p^k + f(t^(k+1))
     u~ carries every velocity boundary condition;
  2. the pressure increment:
         lap phi = (a0 / dt) div u~,
     dphi/dn = 0 wherever the velocity is prescribed, phi = 0 at an outflow;
  3. the projection:   u^(k+1) = u~ - (dt / a0) grad phi;
  4. the pressure:     p^(k+1) = p^k + phi - chi nu div u~.

``chi`` below 1 is what the analysis asks for when part of the boundary is
open (it must stay under 2/d); Dirichlet-only problems may take 1.

**Measured first order in time, and why (W271).**  The analysis's second
order needs an EXACT discrete projection.  On a collocated grid the pressure
Laplacian (compact) and the divergence of the gradient the correction applies
(wide) differ, so the projection is approximate; the manufactured solution's
successive time-step differences shrink at order 1 with it.  Building the
pressure operator as the exact divergence-of-correction restores second order
on one grid and is unstable at an overlap (`PROJECTIONS`).  The route overset
codes take instead is a velocity-pressure formulation with a pressure Poisson
equation and no projection; it is not built here.

**Why implicit advection.**  A body grid clusters its rows toward the wall and
spaces its columns finely round a curved part, so an explicit advective step
would be limited far below the background's own step.  Treating advection
implicitly costs one linear solve per component per step, which the implicit
viscous term costs anyway, and it is the kind of step the live certified mode
the user chose will need.

**Why the pressure is solved once per step.**  Tier 60 priced one factored
solve of the composite system at the car's size at 0.10 s (W269); the matrix
never changes while the geometry does not, so it is factored once.

What the discretisation is
--------------------------

Collocated, second order: central differences through each grid's metrics for
first derivatives, the nine-point conservative Laplacian Tier 60 verified
(`Overset.laplacian_triplets`, the same code), and the width-3 interpolation
equations coupling every grid IMPLICITLY inside each linear solve.  The
background's outermost ring of cells carries the box's conditions directly at
the cell centres: a Dirichlet velocity with ``dphi/dn = 0``, or, on an outflow
face, a zero normal velocity gradient with ``phi = 0`` -- both gradients
one-sided second order.  That places the box's faces half a cell out, which is
exact for a manufactured solution evaluated at those centres and immaterial
for a far field.

Not here
--------

Several bodies whose grids overlap each other or the road (W266), the car,
turbulence, and anything learned.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .overset import (CartesianGrid, CurvilinearGrid, DISC, HOLE, INTERP, OUTER_BC,
                      Overset, OversetError, WALL, circle_outline, ogrid_annulus)

__all__ = [
    "OversetFlow", "FACES", "mms_fields", "MMS", "mms_run", "cylinder_flow",
    "shedding_statistics", "CYLINDER",
]

FACES = ("xlo", "xhi", "ylo", "yhi")

#: How the pressure increment's operator is built.  **Neither choice is what a
#: second-order overset solver needs, and the measurements say which way each
#: fails (Tier 61, W271):**
#:
#: ``'compact'`` (the default): Tier 60's nine-point conservative Laplacian.
#: STABLE, second order in space, and FIRST ORDER IN TIME.  The Laplacian and
#: the divergence-of-gradient the correction applies differ, so the projection
#: is approximate, and an incremental pressure correction needs an exact one.
#: On the background alone, successive time-step differences shrank at order
#: 0.98 where BDF2 momentum with the exact pressure read 1.96.  Repeating the
#: projection within the step (``passes``) shrank the constant by half at five
#: passes and left the order at 1.10.
#:
#: ``'exact'``: the discrete divergence of the correction the step actually
#: makes -- the central gradient at discretisation points, zero on walls and
#: edges, re-interpolated at interpolation points.  Second order in time on the
#: background alone (rms order 1.93), and UNSTABLE on a composite grid: the
#: error grows eightfold every five steps from the overlap, not as a
#: checkerboard, and the march blows up in about thirty steps.  Two grids each
#: held exactly divergence-free over the region they share over-constrain it.
PROJECTIONS = ("compact", "exact")

#: How the momentum solve is preconditioned; see `OversetFlow`.
PRECONDITIONERS = ("jacobi", "ilu")

VelocityFn = Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]]


def _live_xy(g) -> tuple[np.ndarray, np.ndarray]:
    return (g.X, g.Y) if isinstance(g, CartesianGrid) else (g.x, g.y)


class OversetFlow:
    """A march on an `Overset`: velocity ``(u, v)`` and pressure ``p`` everywhere
    that carries an equation, advanced by the rotational pressure correction.

    ``box`` maps each background face to ``'dirichlet'`` (``box_velocity`` at the
    edge cell centres) or ``'outflow'``.  ``wall_velocity(grid, t)`` gives each
    body's wall velocity, zero if omitted.  ``forcing(x, y, t)`` is a body force,
    used by the manufactured solution.  A box with no outflow face fixes the
    pressure increment's constant by a zero-mean constraint with a uniform
    compatibility source (`_build_pressure`), never by pinning a cell.
    """

    def __init__(self, ov: Overset, nu: float, dt: float, *,
                 box: dict[str, str] | None = None,
                 box_velocity: VelocityFn | None = None,
                 wall_velocity: Callable[[CurvilinearGrid, float],
                                         tuple[np.ndarray, np.ndarray]] | None = None,
                 forcing: VelocityFn | None = None,
                 chi: float = 0.5, rtol: float = 1e-10, maxiter: int = 400,
                 projection: str = "compact", passes: int = 1,
                 precond: str = "jacobi", ilu_every: int = 20, ilu_drop_tol: float = 1e-4,
                 ilu_fill: float = 8.0) -> None:
        if ov.bg is None:
            raise ValueError("the flow solver needs a background grid")
        if projection not in PROJECTIONS:
            raise ValueError(f"projection must be one of {PROJECTIONS}")
        if precond not in PRECONDITIONERS:
            raise ValueError(f"precond must be one of {PRECONDITIONERS}")
        self.projection = projection
        self.passes = int(passes)
        #: Tier 62: the car's momentum rows -- cells 27 times longer than wide and
        #: skewed 60 degrees, at a wall spacing of 0.0014 -- took Jacobi-
        #: preconditioned BiCGSTAB 100 to 300 iterations a component.  ``'ilu'``
        #: factors the momentum matrix incompletely every ``ilu_every`` steps and
        #: reuses it between, because the matrix changes only through the
        #: advecting velocity.  ``'jacobi'`` is Tier 61's, unchanged.
        self.precond = precond
        self.ilu_every = int(ilu_every)
        self.ilu_drop_tol = float(ilu_drop_tol)
        self.ilu_fill = float(ilu_fill)
        self._ilu = None
        self._ilu_k = -1
        self.ov = ov
        self.nu = float(nu)
        self.dt = float(dt)
        self.box = dict(box or {f: "dirichlet" for f in FACES})
        for f in FACES:
            if self.box.get(f) not in ("dirichlet", "outflow"):
                raise ValueError(f"box face {f} must be 'dirichlet' or 'outflow'")
        self.box_velocity = box_velocity or (lambda x, y, t: (np.ones_like(x), np.zeros_like(x)))
        self.wall_velocity = wall_velocity
        self.forcing = forcing
        self.chi = float(chi)
        self.rtol = float(rtol)
        self.maxiter = int(maxiter)
        self.t = 0.0
        self.k = 0
        t0 = time.perf_counter()
        self._classify_rows()
        self._build_operators()
        self._build_pressure()
        self.setup_s = time.perf_counter() - t0
        n = ov.n_unknowns
        self.U = np.zeros(n)
        self.V = np.zeros(n)
        self.P = np.zeros(n)
        self.Um1 = None
        self.Vm1 = None
        #: The momentum solve `step` uses.  ``None`` is `_momentum_solve`, the
        #: classical BiCGSTAB whose answer every other mode is scored against.
        #: `atlas.cases.certified_step` puts a defect-corrected solve here, which
        #: is what makes the certified mode the SAME step with a different solve
        #: rather than a second march free to drift from it.  Anything installed
        #: must take ``(M, bu, bv, us, vs, precond)`` and return
        #: ``(ut, vt, iterations)``.
        self.momentum_solver: Callable[..., tuple[np.ndarray, np.ndarray,
                                                  list[int]]] | None = None
        self.log: list[dict[str, Any]] = []

    # -- rows ---------------------------------------------------------------

    def _classify_rows(self) -> None:
        ov, bg = self.ov, self.ov.bg
        n = ov.n_unknowns
        self.X = np.empty(n)
        self.Y = np.empty(n)
        for g in ov.grids:
            idx = ov.index[g.name]
            live = idx >= 0
            gx, gy = _live_xy(g)
            self.X[idx[live]] = gx[live]
            self.Y[idx[live]] = gy[live]
        disc = np.zeros(n, dtype=bool)
        wall = np.zeros(n, dtype=bool)
        interp = np.zeros(n, dtype=bool)
        edge_d = np.zeros(n, dtype=bool)
        edge_out = np.zeros(n, dtype=bool)
        inner = np.full(n, -1, dtype=np.int64)
        # the background
        S = ov.status[bg.name]
        idx = ov.index[bg.name]
        ny, nx = bg.shape
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        face = {"xlo": ii == 0, "xhi": ii == nx - 1, "ylo": jj == 0, "yhi": jj == ny - 1}
        on_edge = face["xlo"] | face["xhi"] | face["ylo"] | face["yhi"]
        # Tier 62: along a ROAD a wheel's hole may reach the edge cells, which are
        # then holes or interpolation points (`overset_multi.MultiOverset` decides
        # which, and allows it on the road only).  Every edge cell that still
        # carries the box's condition needs its two cells inward.
        multi = hasattr(ov, "road")
        if not multi and np.any(on_edge & (S != DISC)):
            raise OversetError("a hole or an interpolation point reaches the box's outer "
                               "ring of cells, which carries the box's conditions")
        dirichlet_edge = np.zeros_like(on_edge)
        for f in FACES:
            if self.box[f] == "dirichlet":
                dirichlet_edge |= face[f]
        live_edge = on_edge & (S == DISC)
        out_edge = live_edge & ~dirichlet_edge
        dirichlet_edge = dirichlet_edge & live_edge
        # the cells one and two steps in from each edge cell (diagonally at a corner)
        ji = np.clip(jj, 1, ny - 2)
        ii_in = np.clip(ii, 1, nx - 2)
        inner_bg = idx[ji, ii_in]
        ji2 = np.where(jj == 0, 2, np.where(jj == ny - 1, ny - 3, jj))
        ii2 = np.where(ii == 0, 2, np.where(ii == nx - 1, nx - 3, ii))
        inner2_bg = idx[ji2, ii2]
        disc[idx[(S == DISC) & ~on_edge]] = True
        interp[idx[S == INTERP]] = True
        edge_d[idx[dirichlet_edge]] = True
        edge_out[idx[out_edge]] = True
        inner[idx[live_edge]] = inner_bg[live_edge]
        inner2 = np.full(n, -1, dtype=np.int64)
        inner2[idx[live_edge]] = inner2_bg[live_edge]
        if np.any(inner[idx[live_edge]] < 0) or np.any(inner2[idx[live_edge]] < 0):
            raise OversetError("an edge cell that carries the box's condition has a hole "
                               "one or two cells inward")
        self.inner2 = inner2
        # body grids
        self.outer_dirichlet = np.zeros(n, dtype=bool)
        for c in ov.comps:
            Sc = ov.status[c.name]
            ic = ov.index[c.name]
            disc[ic[Sc == DISC]] = True
            wall[ic[Sc == WALL]] = True
            interp[ic[Sc == INTERP]] = True
            self.outer_dirichlet[ic[Sc == OUTER_BC]] = True
        self.disc, self.wall, self.interp = disc, wall, interp
        self.edge_d, self.edge_out, self.inner = edge_d, edge_out, inner
        self.has_outflow = bool(edge_out.any())
        classes = disc.astype(int) + wall + interp + edge_d + edge_out + self.outer_dirichlet
        if not np.all(classes == 1):
            raise AssertionError("every unknown must belong to exactly one row class")

    # -- operators ----------------------------------------------------------

    def _build_operators(self) -> None:
        ov, bg = self.ov, self.ov.bg
        n = ov.n_unknowns
        DxR, DxC, DxV = [], [], []
        DyR, DyC, DyV = [], [], []
        LR, LC, LV = [], [], []
        # background interior discretisation points
        S = ov.status[bg.name]
        idx = ov.index[bg.name]
        ny, nx = bg.shape
        jj, ii = np.nonzero(S == DISC)
        keep = (jj > 0) & (jj < ny - 1) & (ii > 0) & (ii < nx - 1)
        jj, ii = jj[keep], ii[keep]
        rows = idx[jj, ii]
        h = bg.h
        for dj, di, sx, sy in ((0, 1, 1.0, 0.0), (0, -1, -1.0, 0.0), (1, 0, 0.0, 1.0),
                               (-1, 0, 0.0, -1.0)):
            nb = idx[jj + dj, ii + di]
            if np.any(nb < 0):                                   # pragma: no cover
                raise OversetError("a background stencil reaches a hole")
            if sx:
                DxR.append(rows); DxC.append(nb); DxV.append(np.full(rows.size, sx / (2 * h)))
            if sy:
                DyR.append(rows); DyC.append(nb); DyV.append(np.full(rows.size, sy / (2 * h)))
            LR.append(rows); LC.append(nb); LV.append(np.full(rows.size, 1.0 / h ** 2))
        LR.append(rows); LC.append(rows); LV.append(np.full(rows.size, -4.0 / h ** 2))
        # body grids
        for c in ov.comps:
            Sc = ov.status[c.name]
            ic = ov.index[c.name]
            nj, ni = c.shape
            cj, ci = np.nonzero(Sc == DISC)
            r = ic[cj, ci]
            J = c.J[cj, ci]
            ye, xe = c.y_eta[cj, ci], c.x_eta[cj, ci]
            yx, xx = c.y_xi[cj, ci], c.x_xi[cj, ci]
            ip, im = ic[cj, np.mod(ci + 1, ni)], ic[cj, np.mod(ci - 1, ni)]
            jp, jm = ic[cj + 1, ci], ic[cj - 1, ci]
            # f_x = (y_eta f_xi - y_xi f_eta) / J ;  f_y = (-x_eta f_xi + x_xi f_eta) / J
            DxR += [r, r, r, r]; DxC += [ip, im, jp, jm]
            DxV += [ye / (2 * J), -ye / (2 * J), -yx / (2 * J), yx / (2 * J)]
            DyR += [r, r, r, r]; DyC += [ip, im, jp, jm]
            DyV += [-xe / (2 * J), xe / (2 * J), xx / (2 * J), -xx / (2 * J)]
            R9, C9, V9, _jj, _ii = ov.laplacian_triplets(c)
            LR += R9; LC += C9; LV += V9

        def csr(R, C, V):
            A = sp.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                              shape=(n, n)).tocsr()
            A.sum_duplicates()
            return A

        self.Dx = csr(DxR, DxC, DxV)
        self.Dy = csr(DyR, DyC, DyV)
        self.Lap = csr(LR, LC, LV)
        # interpolation: row k reads sum_m w_km x_m
        WR, WC, WV = [], [], []
        for name, spec in ov.donors.items():
            ix = ov.index[name]
            for e in spec["entries"]:
                rows_i = ix[e["j"], e["i"]]
                donor = ov.index[e["donor"]].ravel()[e["flat"]]
                WR.append(np.repeat(rows_i, e["flat"].shape[1]))
                WC.append(donor.ravel())
                WV.append(e["weights"].ravel())
        self.W = csr(WR, WC, WV) if WR else sp.csr_matrix((n, n))
        eye = lambda mask: sp.diags(mask.astype(float), format="csr")   # noqa: E731
        self.P_disc = eye(self.disc)
        out_rows = np.nonzero(self.edge_out)[0]
        # an outflow edge cell: zero normal gradient AT its centre, second order,
        # u_edge = (4 u_in - u_in2) / 3
        S_out = sp.coo_matrix(
            (np.concatenate([np.full(out_rows.size, 4.0 / 3.0), np.full(out_rows.size, -1.0 / 3.0)]),
             (np.concatenate([out_rows, out_rows]),
              np.concatenate([self.inner[out_rows], self.inner2[out_rows]]))),
            shape=(n, n)).tocsr()
        #: every row that is not a PDE row, and does not change from step to step
        self.K_fixed = (eye(self.wall) + eye(self.edge_d) + eye(self.outer_dirichlet)
                        + eye(self.edge_out) - S_out + eye(self.interp) - self.W).tocsr()
        self.K_visc = (-self.nu * self.Lap).tocsr()

    def _build_pressure(self) -> None:
        ov = self.ov
        n = ov.n_unknowns
        R, C, V = [], [], []
        if self.projection == "exact":
            # the correction the step makes: grad at discretisation rows, zero on
            # walls and edges, then re-interpolated at interpolation rows
            Pd = sp.diags(self.disc.astype(float), format="csr")
            Gx = (Pd + self.W @ Pd) @ self.Dx
            Gy = (Pd + self.W @ Pd) @ self.Dy
            A_div = (self.Dx @ Gx + self.Dy @ Gy).tocoo()
        else:
            A_div = self.Lap.tocoo()
        R.append(A_div.row); C.append(A_div.col); V.append(A_div.data)
        for c in ov.comps:
            if c.outer == "dirichlet":
                raise ValueError("the flow solver takes body grids joined by interpolation")
            NR, NC, NV = ov.neumann_wall_triplets(c)
            R += NR; C += NC; V += NV
        # Dirichlet-velocity edges: dphi/dn = 0 AT the edge cell's centre, where the
        # velocity is imposed, second order: 3 phi_edge - 4 phi_in + phi_in2 = 0.
        # (A first version wrote phi_edge = phi_in, first order, and the background's
        # pressure converged at about first order with it.)
        ed = np.nonzero(self.edge_d)[0]
        R += [ed, ed, ed]; C += [ed, self.inner[ed], self.inner2[ed]]
        V += [np.full(ed.size, 3.0), np.full(ed.size, -4.0), np.ones(ed.size)]
        eo = np.nonzero(self.edge_out)[0]
        R.append(eo); C.append(eo); V.append(np.ones(eo.size))
        ir = np.nonzero(self.interp)[0]
        R.append(ir); C.append(ir); V.append(np.ones(ir.size))
        Wc = self.W.tocoo()
        R.append(Wc.row); C.append(Wc.col); V.append(-Wc.data)
        # **With no outflow face the increment is fixed only up to a constant, and
        # the interpolation equations make the discrete problem slightly
        # incompatible.**  Pinning one cell to zero made that mismatch a point
        # source, and the march blew up from the box's corner within ten steps
        # (Tier 61's first run).  Instead one extra unknown, lambda, is added to
        # every discretisation row -- the mismatch absorbed as a uniform source --
        # and one extra row asks for a zero mean over the discretisation points.
        m = n
        if not self.has_outflow:
            dr = np.nonzero(self.disc)[0]
            R.append(dr); C.append(np.full(dr.size, n)); V.append(np.ones(dr.size))
            R.append(np.full(dr.size, n)); C.append(dr); V.append(np.full(dr.size, 1.0 / dr.size))
            m = n + 1
        A = sp.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                          shape=(m, m)).tocsc()
        A.sum_duplicates()
        self.pressure_augmented = m == n + 1
        t0 = time.perf_counter()
        self.p_lu = spla.splu(A, permc_spec="COLAMD")
        self.p_factor_s = time.perf_counter() - t0
        self.A_p = A

    # -- state --------------------------------------------------------------

    def fields(self, vec: np.ndarray) -> dict[str, np.ndarray]:
        return self.ov.scatter(vec)

    def set_state(self, u, v, p=None, u_prev=None, v_prev=None, t: float = 0.0) -> None:
        """Vectors in the composite numbering; ``u_prev`` starts BDF2 at once."""
        self.U = np.asarray(u, dtype=float).copy()
        self.V = np.asarray(v, dtype=float).copy()
        self.P = np.zeros_like(self.U) if p is None else np.asarray(p, dtype=float).copy()
        self.Um1 = None if u_prev is None else np.asarray(u_prev, dtype=float).copy()
        self.Vm1 = None if v_prev is None else np.asarray(v_prev, dtype=float).copy()
        self.t = float(t)
        self.k = 0 if u_prev is None else 1

    def _wall_values(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        uw = np.zeros(self.ov.n_unknowns)
        vw = np.zeros(self.ov.n_unknowns)
        if self.wall_velocity is None:
            return uw, vw
        for c in self.ov.comps:
            a, b = self.wall_velocity(c, t)
            rows = self.ov.index[c.name][0, :]
            live = self.ov.status[c.name][0, :] == WALL
            if np.all(live):
                uw[rows] = a
                vw[rows] = b
            else:
                # a road patch's two end points are interpolation points
                uw[rows[live]] = np.broadcast_to(a, rows.shape)[live]
                vw[rows[live]] = np.broadcast_to(b, rows.shape)[live]
        return uw, vw

    # -- the step -----------------------------------------------------------

    def _assemble_momentum(self) -> tuple[Any, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, float, float]:
        """The momentum system this step must solve: ``(M, bu, bv, us, vs, a0, t1)``.

        One definition, used by `step` and by the certified step in
        `atlas.cases.certified_step`, which needs the SAME ``M`` and ``b`` the
        classical solve is given -- a replica of this assembly would be a second
        car free to drift from the first.

        ``M`` is the same matrix for both components; only the right-hand side
        differs.  ``us, vs`` are the extrapolated advecting velocity, which is
        also what `step` hands the solver as its starting guess.
        """
        dt = self.dt
        t1 = self.t + dt
        if self.k == 0 or self.Um1 is None:
            a0 = 1.0
            hist_u, hist_v = self.U / dt, self.V / dt
            us, vs = self.U, self.V
        else:
            a0 = 1.5
            hist_u = (2.0 * self.U - 0.5 * self.Um1) / dt
            hist_v = (2.0 * self.V - 0.5 * self.Vm1) / dt
            us, vs = 2.0 * self.U - self.Um1, 2.0 * self.V - self.Vm1
        M = (self.K_fixed + self.K_visc + (a0 / dt) * self.P_disc
             + sp.diags(us * self.disc) @ self.Dx + sp.diags(vs * self.disc) @ self.Dy).tocsr()
        gpx = self.Dx @ self.P
        gpy = self.Dy @ self.P
        uw, vw = self._wall_values(t1)
        bu = np.zeros_like(self.U)
        bv = np.zeros_like(self.V)
        d = self.disc
        bu[d] = hist_u[d] - gpx[d]
        bv[d] = hist_v[d] - gpy[d]
        if self.forcing is not None:
            fu, fv = self.forcing(self.X[d], self.Y[d], t1)
            bu[d] += fu
            bv[d] += fv
        w = self.wall
        bu[w], bv[w] = uw[w], vw[w]
        ed = self.edge_d | self.outer_dirichlet
        if np.any(ed):
            ub, vb = self.box_velocity(self.X[ed], self.Y[ed], t1)
            bu[ed], bv[ed] = ub, vb
        return M, bu, bv, us, vs, a0, t1

    def _momentum_precond(self, M) -> tuple[Any, float, str | None]:
        """The preconditioner for this step's momentum system, refreshing the
        incomplete factor when it is due: ``(precond, t_ilu, ilu_note)``."""
        diag = M.diagonal()
        if np.any(diag == 0):                                    # pragma: no cover
            raise AssertionError("a momentum row has a zero diagonal")
        t_ilu = 0.0
        ilu_note = None
        if self.precond == "ilu":
            if self._ilu is None or self.k - self._ilu_k >= self.ilu_every:
                t0 = time.perf_counter()
                self._ilu = None
                # SuperLU's incomplete factor can meet an exactly zero pivot after
                # dropping (it did on the car); tighter dropping first, then Jacobi
                # for this refresh, and the step's record says which ran
                for drop, fill in ((self.ilu_drop_tol, self.ilu_fill),
                                   (0.01 * self.ilu_drop_tol, 1.5 * self.ilu_fill)):
                    try:
                        self._ilu = spla.spilu(M.tocsc(), drop_tol=drop, fill_factor=fill)
                        ilu_note = f"ilu drop {drop:g} fill {fill:g}"
                        break
                    except RuntimeError as exc:
                        ilu_note = f"ilu failed ({exc}) at drop {drop:g}"
                self._ilu_k = self.k
                t_ilu = time.perf_counter() - t0
            if self._ilu is not None:
                ilu = self._ilu
                precond = spla.LinearOperator(M.shape, matvec=ilu.solve)
            else:
                precond = spla.LinearOperator(M.shape, matvec=lambda r: r / diag)
        else:
            precond = spla.LinearOperator(M.shape, matvec=lambda r: r / diag)
        return precond, t_ilu, ilu_note

    def _momentum_solve(self, M, bu, bv, us, vs, precond
                        ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Solve ``M x = b`` for both components: ``(ut, vt, iterations)``.

        This is the CLASSICAL answer the certified step is certified against --
        ``x* = M^{-1} b`` to `rtol`.  The certified step reaches the same ``x*``
        by a different route and is checked against this one.
        """
        its = []
        sol = []
        for b, x0 in ((bu, us), (bv, vs)):
            count = [0]
            bn = np.linalg.norm(b)
            # A starting guess that already solves the system -- a uniform stream
            # does -- makes BiCGSTAB's first inner product exactly zero, which it
            # reports as a breakdown; accept the guess instead.
            if np.linalg.norm(b - M @ x0) <= self.rtol * max(bn, 1e-300):
                its.append(0)
                sol.append(x0.copy())
                continue
            x, info = spla.bicgstab(M, b, x0=x0.copy(), rtol=self.rtol, atol=0.0,
                                    maxiter=self.maxiter, M=precond,
                                    callback=lambda _x: count.__setitem__(0, count[0] + 1))
            if info < 0:
                # a genuine breakdown, not a failure to converge: GMRES does not break down
                x, info = spla.gmres(M, b, x0=x0.copy(), rtol=self.rtol, atol=0.0, restart=60,
                                     maxiter=self.maxiter, M=precond,
                                     callback=lambda _r: count.__setitem__(0, count[0] + 1),
                                     callback_type="pr_norm")
            if info != 0:
                raise RuntimeError(f"the momentum solve did not converge (info {info}) at "
                                   f"step {self.k} -- a blow-up reads as this")
            its.append(count[0])
            sol.append(x)
        return sol[0], sol[1], its

    def step(self) -> dict[str, Any]:
        t_start = time.perf_counter()
        dt, nu = self.dt, self.nu
        M, bu, bv, us, vs, a0, t1 = self._assemble_momentum()
        precond, t_ilu, ilu_note = self._momentum_precond(M)
        t_asm = time.perf_counter() - t_start
        t0 = time.perf_counter()
        solve = self.momentum_solver or self._momentum_solve
        ut, vt, its = solve(M, bu, bv, us, vs, precond)
        t_mom = time.perf_counter() - t0
        d = self.disc
        div_t = self.Dx @ ut + self.Dy @ vt
        t0 = time.perf_counter()
        un, vn = ut.copy(), vt.copy()
        phi = np.zeros_like(ut)
        eo = self.edge_out
        ir = self.interp
        # ``passes`` > 1 repeats the projection on what the last pass left behind,
        # with the same factorisation, and the increments add: an approximate
        # projection applied again removes most of its own residual divergence.
        div_k = div_t
        for _pass in range(self.passes):
            bp = np.zeros(self.A_p.shape[0])
            bp[:ut.size][d] = (a0 / dt) * div_k[d]
            dphi = self.p_lu.solve(bp)[:ut.size]
            phi += dphi
            un[d] -= (dt / a0) * (self.Dx @ dphi)[d]
            vn[d] -= (dt / a0) * (self.Dy @ dphi)[d]
            un[eo] = (4.0 * un[self.inner[eo]] - un[self.inner2[eo]]) / 3.0
            vn[eo] = (4.0 * vn[self.inner[eo]] - vn[self.inner2[eo]]) / 3.0
            un[ir] = (self.W @ un)[ir]
            vn[ir] = (self.W @ vn)[ir]
            if _pass + 1 < self.passes:
                div_k = self.Dx @ un + self.Dy @ vn
        t_p = time.perf_counter() - t0
        pn = self.P + phi
        pn[d] -= self.chi * nu * div_t[d]
        for c in self.ov.comps:
            ic = self.ov.index[c.name]
            dw = c.divergence(ut[ic], vt[ic])[0]
            at_wall = self.ov.status[c.name][0, :] == WALL
            if np.all(at_wall):
                pn[ic[0, :]] -= self.chi * nu * dw
            else:
                pn[ic[0, at_wall]] -= self.chi * nu * dw[at_wall]
        pn[ir] = (self.W @ pn)[ir]
        self.Um1, self.Vm1 = self.U, self.V
        self.U, self.V, self.P = un, vn, pn
        self.t = t1
        self.k += 1
        div_n = self.Dx @ un + self.Dy @ vn
        rec = {"k": self.k, "t": t1, "iterations": its, "assemble_s": t_asm,
               "momentum_s": t_mom, "pressure_s": t_p, "ilu_s": t_ilu, "ilu_note": ilu_note,
               "step_s": time.perf_counter() - t_start,
               "max_div": float(np.abs(div_n[d]).max()) if d.any() else 0.0,
               "u_max": float(np.hypot(un, vn).max())}
        if not np.isfinite(rec["u_max"]):
            raise RuntimeError(f"the march blew up at step {self.k}")
        return rec

    # -- diagnostics ----------------------------------------------------------

    def forces(self, name: str) -> dict[str, float]:
        """Force on one body from the stress at its wall.

        ``F = -sum (sigma . n) ds`` with ``sigma = -p I + nu (grad u + grad u^T)``
        and ``n`` the wall normal pointing out of the fluid; the pressure and
        viscous parts are kept apart.
        """
        c = next(g for g in self.ov.comps if g.name == name)
        ic = self.ov.index[c.name]
        u, v, p = self.U[ic], self.V[ic], self.P[ic]
        ux, uy = c.gradient(u)
        vx, vy = c.gradient(v)
        nx_, ny_ = c.wall_normal()
        ds = np.hypot(c.x_xi[0], c.y_xi[0])
        pw = p[0]
        fpx = -np.sum(-pw * nx_ * ds)
        fpy = -np.sum(-pw * ny_ * ds)
        sxx, syy, sxy = 2 * self.nu * ux[0], 2 * self.nu * vy[0], self.nu * (uy[0] + vx[0])
        fvx = -np.sum((sxx * nx_ + sxy * ny_) * ds)
        fvy = -np.sum((sxy * nx_ + syy * ny_) * ds)
        return {"fx": float(fpx + fvx), "fy": float(fpy + fvy), "fx_pressure": float(fpx),
                "fy_pressure": float(fpy), "fx_viscous": float(fvx), "fy_viscous": float(fvy)}

    def ring_flux(self, name: str, j: int) -> float:
        """Net volume flux out through row ``j`` of a body grid: ``sum_i (-y_xi u + x_xi v)``."""
        c = next(g for g in self.ov.comps if g.name == name)
        ic = self.ov.index[c.name]
        return float(np.sum(-c.y_xi[j] * self.U[ic[j]] + c.x_xi[j] * self.V[ic[j]]))

    def box_flux(self, i0: int, i1: int, j0: int, j1: int) -> float:
        """Net volume flux out of the background rectangle of cells ``[i0, i1] x [j0, j1]``,
        through its faces, with face velocities averaged from the two cells beside each."""
        bg = self.ov.bg
        idx = self.ov.index[bg.name]
        h = bg.h

        def face(field, ka, kb):
            # the INDICES must be live; a velocity may be negative, an index may not
            if np.any(ka < 0) or np.any(kb < 0):
                raise ValueError("the rectangle's faces touch a hole")
            return 0.5 * (field[ka] + field[kb])

        js = np.arange(j0, j1 + 1)
        is_ = np.arange(i0, i1 + 1)
        right = face(self.U, idx[js, i1], idx[js, i1 + 1])
        left = face(self.U, idx[js, i0], idx[js, i0 - 1])
        top = face(self.V, idx[j1, is_], idx[j1 + 1, is_])
        bottom = face(self.V, idx[j0, is_], idx[j0 - 1, is_])
        return float(h * (right.sum() - left.sum() + top.sum() - bottom.sum()))

    def divergence_report(self) -> dict[str, float]:
        div = self.Dx @ self.U + self.Dy @ self.V
        out = {}
        for g in self.ov.grids:
            ix = self.ov.index[g.name]
            rows = ix[ix >= 0]
            rows = rows[self.disc[rows]]
            out[g.name] = float(np.abs(div[rows]).max()) if rows.size else 0.0
        return out


# ===========================================================================
# verification: a manufactured solution
# ===========================================================================

#: A divergence-free velocity from ``psi = A sin(a x + b0) sin(c y + d0) cos(omega t)``
#: and a pressure ``B cos(e x) sin(g y) sin(omega t)``; the body force makes them
#: an exact solution of the forced Navier-Stokes equations.
MMS = {"A": 0.5, "a": 1.1, "b0": 0.3, "c": 0.9, "d0": -0.2, "omega": 2.0,
       "B": 0.3, "e": 0.8, "g": 1.3}


def mms_fields(x, y, t, nu):
    """``(u, v, p, fu, fv)`` of the manufactured solution."""
    A, a, b0, c, d0, w, B, e, g = (MMS[k] for k in ("A", "a", "b0", "c", "d0", "omega",
                                                     "B", "e", "g"))
    Sa, Ca = np.sin(a * x + b0), np.cos(a * x + b0)
    Sc, Cc = np.sin(c * y + d0), np.cos(c * y + d0)
    T, Tp = math.cos(w * t), -w * math.sin(w * t)
    u = A * c * Sa * Cc * T
    v = -A * a * Ca * Sc * T
    ut = A * c * Sa * Cc * Tp
    vt = -A * a * Ca * Sc * Tp
    ux = A * c * a * Ca * Cc * T
    uy = -A * c * c * Sa * Sc * T
    vx = A * a * a * Sa * Sc * T
    vy = -A * a * c * Ca * Cc * T
    s = math.sin(w * t)
    p = B * np.cos(e * x) * np.sin(g * y) * s
    px = -B * e * np.sin(e * x) * np.sin(g * y) * s
    py = B * g * np.cos(e * x) * np.cos(g * y) * s
    k2 = a * a + c * c
    fu = ut + u * ux + v * uy + px + nu * k2 * u
    fv = vt + u * vx + v * vy + py + nu * k2 * v
    return u, v, p, fu, fv


def mms_run(level: int, dt: float, T: float, *, nu: float = 0.01, chi: float = 1.0,
            twist: float = 0.0, projection: str = "compact") -> dict[str, Any]:
    """March the manufactured solution to ``T`` on Tier 60's verification geometry
    at ``h = 1/(16 2^(level-1))`` and measure the error against it.

    Every box face and the wall carry the exact velocity (the wall MOVES), the
    march starts from the exact fields at ``t = 0`` and ``t = -dt`` so BDF2 runs
    from the first step, and the pressure is compared after removing its mean,
    because a Dirichlet-only box fixes it only up to a constant.
    """
    from .overset import MS_BODY, MS_BOX, MS_HOLE_MARGIN, MS_THICKNESS
    n16 = 2 ** (level - 1)
    h = 1.0 / (16 * n16)
    Lx, Ly = MS_BOX
    cx, cy, R = MS_BODY
    bg = CartesianGrid("bg", int(round(Lx / h)), int(round(Ly / h)), h)
    body = ogrid_annulus("body", cx, cy, R, R + MS_THICKNESS, 48 * n16, 6 * n16 + 1,
                         beta=1.0, twist=twist)
    ov = Overset(bg, [body], hole_margin=MS_HOLE_MARGIN, width=3)

    def box_velocity(x, y, t):
        u, v, _p, _fu, _fv = mms_fields(x, y, t, nu)
        return u, v

    def wall_velocity(g, t):
        u, v, _p, _fu, _fv = mms_fields(g.x[0], g.y[0], t, nu)
        return u, v

    def forcing(x, y, t):
        _u, _v, _p, fu, fv = mms_fields(x, y, t, nu)
        return fu, fv

    flow = OversetFlow(ov, nu, dt, box={f: "dirichlet" for f in FACES},
                       box_velocity=box_velocity, wall_velocity=wall_velocity,
                       forcing=forcing, chi=chi, projection=projection)
    u0, v0, p0, _a, _b = mms_fields(flow.X, flow.Y, 0.0, nu)
    um, vm, _pm, _c, _d = mms_fields(flow.X, flow.Y, -dt, nu)
    flow.set_state(u0, v0, p0, u_prev=um, v_prev=vm, t=0.0)
    steps = int(round(T / dt))
    t0 = time.perf_counter()
    its = []
    for _ in range(steps):
        rec = flow.step()
        its.append(rec["iterations"])
    wall_s = time.perf_counter() - t0
    ue, ve, pe, _e, _f = mms_fields(flow.X, flow.Y, flow.t, nu)
    live = np.ones(ov.n_unknowns, dtype=bool)
    dp = flow.P - pe
    dp = dp - dp.mean()
    out = {"level": level, "h": h, "dt": dt, "T": flow.t, "steps": steps, "nu": nu,
           "chi": chi, "projection": projection, "n_unknowns": ov.n_unknowns, "wall_s": wall_s,
           "setup_s": flow.setup_s, "pressure_factor_s": flow.p_factor_s,
           "momentum_iterations_max": int(np.max(its)) if its else 0,
           "divergence": flow.divergence_report(), "grids": {}}
    edges = flow.edge_d | flow.edge_out
    kinds = {"disc": flow.disc, "wall": flow.wall, "interp": flow.interp, "box edge": edges}
    for g in ov.grids:
        ix = ov.index[g.name]
        rows = ix[ix >= 0]
        eu = np.abs(flow.U[rows] - ue[rows])
        ev = np.abs(flow.V[rows] - ve[rows])
        ep = np.abs(dp[rows])
        ku = int(rows[np.argmax(np.maximum(eu, ev))])
        kp = int(rows[np.argmax(ep)])
        off = ~edges[rows]
        out["grids"][g.name] = {
            "u_max_err": float(max(eu.max(), ev.max())),
            "u_rms_err": float(np.sqrt(np.mean(eu ** 2 + ev ** 2))),
            "p_max_err": float(ep.max()), "p_rms_err": float(np.sqrt(np.mean(ep ** 2))),
            # the box's edge cells carry its conditions half a cell out; these
            # are the errors everywhere else
            "u_max_err_off_edges": float(max(eu[off].max(), ev[off].max())),
            "p_max_err_off_edges": float(ep[off].max()),
            "u_max_at": {"x": float(flow.X[ku]), "y": float(flow.Y[ku]),
                         "kind": next(k for k, m in kinds.items() if m[ku])},
            "p_max_at": {"x": float(flow.X[kp]), "y": float(flow.Y[kp]),
                         "kind": next(k for k, m in kinds.items() if m[kp])},
        }
    out["fields"] = {"u": flow.U, "v": flow.V, "p": flow.P - flow.P.mean()}
    return out


# ===========================================================================
# the benchmark: a circular cylinder at Re = 100
# ===========================================================================

#: The cylinder benchmark's geometry in tiling units, where ``nu = 0.004`` and the
#: free stream is 1, so a diameter of 0.4 is Re = 100: 30 diameters long
#: (10 upstream, 20 down) and 20 across, a blockage of 5%.
CYLINDER = {"box": (12.0, 8.0), "centre": (4.0, 4.0), "diameter": 0.4, "nu": 0.004,
            "thickness": 0.3, "beta": 2.5, "hole_margin": 0.08}


def cylinder_flow(h: float = 1.0 / 32, ni: int = 256, nj: int = 33, dt: float = 0.0125,
                  chi: float = 0.5, perturb: float = 0.05,
                  projection: str = "compact") -> OversetFlow:
    """The flow round a circular cylinder, set up and started impulsively.

    Uniform free stream on the inlet, top and bottom, an outflow on the right,
    the wall at rest.  A small asymmetric bump in ``v`` behind the cylinder breaks
    the symmetry so shedding starts in tens of time units rather than hundreds;
    it decays long before any statistic is read.
    """
    Lx, Ly = CYLINDER["box"]
    cx, cy = CYLINDER["centre"]
    r = 0.5 * CYLINDER["diameter"]
    bg = CartesianGrid("bg", int(round(Lx / h)), int(round(Ly / h)), h)
    body = ogrid_annulus("cylinder", cx, cy, r, r + CYLINDER["thickness"], ni, nj,
                         beta=CYLINDER["beta"])
    ov = Overset(bg, [body], hole_margin=CYLINDER["hole_margin"], width=3)
    flow = OversetFlow(ov, CYLINDER["nu"], dt,
                       box={"xlo": "dirichlet", "ylo": "dirichlet", "yhi": "dirichlet",
                            "xhi": "outflow"},
                       box_velocity=lambda x, y, t: (np.ones_like(x), np.zeros_like(x)),
                       chi=chi, projection=projection)
    u = np.ones(ov.n_unknowns)
    u[flow.wall] = 0.0
    v = perturb * np.exp(-((flow.X - cx - 1.0) ** 2 + (flow.Y - cy - 0.2) ** 2) / 0.1)
    v[flow.wall] = 0.0
    flow.set_state(u, v)
    return flow


def shedding_statistics(t: Sequence[float], cd: Sequence[float], cl: Sequence[float],
                        t_from: float, diameter: float, u_inf: float = 1.0) -> dict[str, Any]:
    """Strouhal number, mean and amplitude of drag and lift over ``t >= t_from``.

    The period is read from the upward zero crossings of the lift about its mean,
    interpolated linearly, over whole cycles only; a single draw is not an
    effect, so the spread across cycles is reported beside the mean.
    """
    t = np.asarray(t, dtype=float)
    cd = np.asarray(cd, dtype=float)
    cl = np.asarray(cl, dtype=float)
    m = t >= t_from
    t, cd, cl = t[m], cd[m], cl[m]
    if t.size < 10:
        return {"cycles": 0}
    s = cl - cl.mean()
    up = np.nonzero((s[:-1] < 0) & (s[1:] >= 0))[0]
    tz = t[up] - s[up] * (t[up + 1] - t[up]) / (s[up + 1] - s[up])
    out: dict[str, Any] = {"cycles": int(max(0, tz.size - 1)), "t_from": float(t[0]),
                           "t_to": float(t[-1])}
    if tz.size >= 3:
        periods = np.diff(tz)
        f = 1.0 / periods.mean()
        out.update(period_mean=float(periods.mean()), period_std=float(periods.std()),
                   strouhal=float(f * diameter / u_inf))
        a, b = tz[0], tz[-1]
        w = (t >= a) & (t <= b)
        out.update(cd_mean=float(cd[w].mean()), cd_amplitude=float(0.5 * (cd[w].max() - cd[w].min())),
                   cl_mean=float(cl[w].mean()), cl_amplitude=float(0.5 * (cl[w].max() - cl[w].min())))
    return out
