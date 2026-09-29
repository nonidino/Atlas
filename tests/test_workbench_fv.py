"""The finite-volume core (`fv.py`) and the style drivers (`styles.py`).

Pinned here, before any family is built on them:

* the discretization is EXACT for a layered wall whose interfaces lie on cell
  faces: the heat flow through it is ``dT / sum_i L_i/k_i`` to round-off;
* a steady solve balances its boundary flows to round-off, and a
  backward-Euler step balances storage against them;
* the upwind advection conserves what it carries;
* one window IS the full domain, to the bit, through the Schwarz driver;
* style B converges to the full-domain solution, and threads do not change a bit;
* style C converges to the full-domain solution; unrelaxed, it diverges exactly
  when the 1-D contraction factor ``rho = k_D L_N / (k_N L_D)`` exceeds one, and
  relaxation or Aitken rescues it;
* style D converges on a scalar port to the closed form of a series circuit.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from atlas.workbench import fv, styles                                   # noqa: E402
from atlas.workbench.tiling import RectangleTiling                       # noqa: E402

K_STEEL, K_CU = 45.0, 400.0


def wall(nx=64, ny=16, dx=0.01, split=40, t_hot=400.0, t_cold=300.0):
    """Steel on the left ``split`` columns, copper on the rest; hot left, cold
    right, insulated top and bottom."""
    k = np.full((ny, nx), K_STEEL)
    k[:, split:] = K_CU
    bc = {"left": (np.full(ny, fv.FIXED), np.full(ny, t_hot)),
          "right": (np.full(ny, fv.FIXED), np.full(ny, t_cold))}
    return fv.Field(nx, ny, dx, k, bc=bc)


def solve_full(f, diag_add=None, extra=None):
    s = fv.assemble(f, diag_add=diag_add)
    return styles.Factor(s.A).solve(s.b if extra is None else s.b + extra), s


def test_a_layered_wall_is_exact():
    f = wall()
    u, _s = solve_full(f)
    q = fv.boundary_inflow(f, u)
    L1, L2 = 40 * f.dx, 24 * f.dx
    closed = (400.0 - 300.0) * (f.ny * f.dx) / (L1 / K_STEEL + L2 / K_CU)
    assert abs(q["left"] - closed) / closed < 1e-12
    assert abs(q["left"] + q["right"]) / closed < 1e-12            # the balance
    assert q["top"] == 0.0 and q["bottom"] == 0.0


def test_a_backward_euler_step_balances_storage():
    f = wall()
    cap = np.full((f.ny, f.nx), 3.8e6)
    dt = 5.0
    m = cap * f.dx * f.dx / dt
    u0 = np.full(f.n, 300.0)
    u1, _s = solve_full(f, diag_add=m, extra=m.ravel() * u0)
    stored = float(np.sum(m.ravel() * (u1 - u0)))
    q = sum(fv.boundary_inflow(f, u1).values())
    assert abs(stored - q) / abs(q) < 1e-11


def test_upwind_advection_carries_what_enters():
    nx, ny, dx = 40, 10, 0.1
    f = fv.Field(nx, ny, dx, np.full((ny, nx), 1e-3),
                 fx=np.full((ny, nx + 1), 0.5), fy=np.zeros((ny + 1, nx)),
                 bc={"left": (np.full(ny, fv.INLET), np.full(ny, 2.0)),
                     "right": (np.full(ny, fv.OUTLET), np.zeros(ny))})
    u, _s = solve_full(f)
    q = fv.boundary_inflow(f, u)
    assert q["left"] == pytest.approx(0.5 * dx * ny * 2.0, rel=1e-13)
    assert abs(q["left"] + q["right"]) / q["left"] < 1e-12
    # and a closed edge that the flow crosses is refused, not leaked through
    bad = fv.Field(nx, ny, dx, np.full((ny, nx), 1e-3), fx=np.full((ny, nx + 1), 0.5))
    with pytest.raises(ValueError, match="crosses a boundary declared closed"):
        fv.assemble(bad)


def _windows(f, boxes, ramp=4):
    t = RectangleTiling(f.nx, f.ny, boxes, ramp)
    systems, chi = [], []
    for (x0, y0, w, h), c in zip(t.boxes, t.chi):
        jj, ii = np.meshgrid(np.arange(y0, y0 + h), np.arange(x0, x0 + w), indexing="ij")
        systems.append(fv.assemble(f, (jj * f.nx + ii).ravel()))
        chi.append(c.ravel())
    return systems, [styles.Factor(s.A) for s in systems], chi


def test_one_window_is_the_full_domain_to_the_bit():
    f = wall()
    u_full, _s = solve_full(f)
    systems, factors, chi = _windows(f, [("W", (0, 0, f.nx, f.ny))])
    assert systems[0].C is None and np.all(chi[0] == 1.0)
    it = styles.schwarz(systems, factors, chi, np.full(f.n, 300.0), 1e-12, 10, 400.0)
    assert np.array_equal(it.u, u_full)
    assert it.iterations == 2 and it.history[-1] == 0.0


def test_style_b_converges_to_the_full_domain_and_threads_change_no_bit():
    f = wall()
    u_full, _s = solve_full(f)
    boxes = [("A", (0, 0, 40, 16)), ("B", (24, 0, 40, 16))]
    systems, factors, chi = _windows(f, boxes)
    serial = styles.schwarz(systems, factors, chi, np.full(f.n, 300.0), 1e-11, 2000, 400.0)
    with ThreadPoolExecutor(2) as pool:
        threaded = styles.schwarz(systems, factors, chi, np.full(f.n, 300.0), 1e-11, 2000,
                                  400.0, pool=pool)
    assert serial.converged and np.array_equal(serial.u, threaded.u)
    assert np.max(np.abs(serial.u - u_full)) / 100.0 < 1e-8
    assert serial.history[0] > 1e-3 > serial.history[-1]


def _pieces(f, d_cols, n_cols):
    def cells(cols):
        jj, ii = np.meshgrid(np.arange(f.ny), np.asarray(cols), indexing="ij")
        return (jj * f.nx + ii).ravel()
    sd = fv.assemble(f, cells(d_cols), cut="dirichlet")
    sn = fv.assemble(f, cells(n_cols), cut="neumann")
    return sd, styles.Factor(sd.A), sn, styles.Factor(sn.A)


@pytest.mark.parametrize("aitken", [False, True])
def test_style_c_converges_to_the_full_domain(aitken):
    f = wall()
    u_full, _s = solve_full(f)
    # steel is the Dirichlet side: rho = k_D L_N / (k_N L_D) = 45*24/(400*40) = 0.0675
    sd, fd, sn, fn = _pieces(f, range(0, 40), range(40, 64))
    it = styles.dirichlet_neumann(sd, fd, sn, fn, np.full(sd.face_rows.size, 300.0), 1e-12,
                                  200, 400.0, theta0=1.0, aitken=aitken)
    assert it.converged
    assert np.max(np.abs(it.u - u_full)) / 100.0 < 1e-10
    assert it.iterations < 20


def test_style_c_needs_relaxation_when_the_dirichlet_side_conducts_more():
    """Copper as the Dirichlet side: rho = 400*40/(45*24) = 14.8 > 1, so the
    unrelaxed iteration diverges, and the optimal 1-D factor 1/(1 + rho)
    converges -- the relaxation is doing the work, and the page says which."""
    f = wall()
    u_full, _s = solve_full(f)
    sd, fd, sn, fn = _pieces(f, range(40, 64), range(0, 40))
    lam0 = np.full(sd.face_rows.size, 300.0)
    bare = styles.dirichlet_neumann(sd, fd, sn, fn, lam0, 1e-12, 12, 400.0, theta0=1.0,
                                    aitken=False)
    assert not bare.converged and bare.history[-1] > bare.history[0]
    rho = (K_CU * 40) / (K_STEEL * 24)
    good = styles.dirichlet_neumann(sd, fd, sn, fn, lam0, 1e-12, 50, 400.0,
                                    theta0=1.0 / (1.0 + rho), aitken=False)
    assert good.converged and np.max(np.abs(good.u - u_full)) / 100.0 < 1e-10
    auto = styles.dirichlet_neumann(sd, fd, sn, fn, lam0, 1e-12, 50, 400.0, theta0=0.1,
                                    aitken=True)
    assert auto.converged


def test_style_d_reaches_the_closed_form_of_a_series_circuit():
    """A scalar port: a conductance G driven through a resistance R by a source
    V.  The port effort converges to ``V / (1 + G R)``."""
    G, R, V = 3.0, 2.0, 12.0

    def field(x):
        return G * x, None                       # the flow through the port

    def lumped(flow):
        return V - R * flow                      # the effort the network puts back

    it = styles.field_lumped(field, lumped, np.array([0.0]), 1e-13, 100, V, theta0=0.5)
    assert it.converged
    assert it.extra["efforts"][0] == pytest.approx(V / (1.0 + G * R), rel=1e-12)
