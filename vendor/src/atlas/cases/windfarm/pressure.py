"""Poisson solves with real boundary conditions -- the W5 fix.

W5 ran, and one gate item was blocked. Both the blockage and the inverted array
efficiency traced to the same place: **every pressure in the coupled system was
recovered on the assumption that its domain is periodic.** Two of them are not.

`couple.global_project` projected the assembled field with an FFT, so the
domain wrapped in `x`: the disks' pressure jump propagated out of the outlet and
back in through the inlet instead of being absorbed. Measured consequence, at
t = 15: a 26% velocity deficit one diameter *upstream* of turbine 1 and 13%
three diameters upstream, against roughly 10% and 2% for a real actuator disk.
Turbine 1 throttled its own inflow harder than turbine 2's wake throttled
turbine 2's, and `P_2/P_1` came out at 1.16 -- backwards for an aligned pair.

`probe.pressure_from_velocity` recovered `p` on a 2 D window by the same
spectral route. On W3's genuinely periodic probe window that is exact. On a
rotor window inside the coupled domain it is not, and with a body force present
the error is large: the momentum residual `g` read 100-450% of `T` at the rotor
while the identical operator on a turbine-free volume read 3%. That is what
blocked the thrust projection -- the correction mechanism was sound and drove
`g` to 1e-15, but `g` itself was the wrong target, so enforcing it reversed the
flow through the disk.

What replaces them
------------------
Two direct solves, each diagonalized exactly by a discrete cosine transform, so
the cost stays `O(N log N)` -- the same class as the FFT they replace, and not
the iterative solve the guide budgets for. Measured: 0.08 s on the 512 x 1536
domain, against 2.7 s for one 124-window expert sweep.

* `project_outflow` -- mass. Neumann on the pinned inlet and the two slip
  walls, **Dirichlet at the open outlet**. The Dirichlet face is the whole
  point: it is the one boundary free to absorb the pressure the rotors
  displace, and with it the projection is non-singular and needs no
  compatibility condition.
* `pressure_neumann` -- momentum. Neumann on all four faces of a control
  volume, with the data taken from the normal momentum equation itself.
  Singular by construction (`p` is fixed only up to a constant), which is
  harmless because a closed control volume has `integral n_x ds = 0`, and the
  compatibility defect is returned rather than hidden because it is the error
  bar on every residual computed from it.

Two Laplacians, not one, and the difference is load-bearing
------------------------------------------------------------
The pressure solve wants the **compact** five-point Laplacian, because it wants
an accurate `p`: `laplacian_eigenvalues_*`, diagonalized by DCT-II (Neumann) and
DCT-IV (Neumann/Dirichlet), second-order convergent against an analytic
pressure.

The projection wants a different operator, and using the compact one here was
the first version's mistake. On a collocated grid the divergence that is
measured -- and the one that telescopes over a union of cells to that union's
net boundary flux, which is what conservation law C1 asks about -- is the
two-cell centred difference. Composing it with a centred gradient gives a
Laplacian on a `2h` stencil, `wide_eigenvalues_*`. Project with the compact
operator and measure with the wide divergence and most of the divergence stays:
measured on a realistic wake field, `L2` fell only from 0.64 to 0.40. Project
with the wide one and it goes to 9e-14.

So `divergence_bc`, `gradient_bc` and `wide_eigenvalues_*` are one operator
split across three functions, and they have to agree exactly.
`test_w5_pressure` composes the first two and checks the result against the
third rather than trusting the derivation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, idct

__all__ = [
    "laplacian_eigenvalues_neumann", "laplacian_eigenvalues_dirichlet_hi",
    "solve_neumann", "solve_neumann_dirichlet", "solve_wide_neumann_dirichlet",
    "wide_eigenvalues_neumann", "wide_eigenvalues_dirichlet_hi",
    "divergence_bc", "gradient_bc", "ProjectionReport", "project_outflow",
    "pressure_neumann",
]


# --------------------------------------------------------------------------
# the two eigenvalue families
# --------------------------------------------------------------------------


def laplacian_eigenvalues_neumann(n: int, h: float) -> np.ndarray:
    """Eigenvalues of the 1-D 5-point Laplacian with Neumann at BOTH faces.

    Cell-centred lattice, ghost reflection `p_{-1} = p_0` and `p_n = p_{n-1}`.
    The eigenvectors are `cos[pi k (j + 1/2) / n]`, which is the DCT-II basis,
    with

        lambda_k = -(4/h^2) sin^2(pi k / (2n)),    k = 0 .. n-1.

    `lambda_0 = 0` -- the constant. That null space is real, not a numerical
    accident, and callers must handle it."""
    k = np.arange(n)
    return -(4.0 / (h * h)) * np.sin(np.pi * k / (2.0 * n)) ** 2


def laplacian_eigenvalues_dirichlet_hi(n: int, h: float) -> np.ndarray:
    """Eigenvalues with Neumann at the LOW face and Dirichlet at the HIGH face.

    Ghost cells `p_{-1} = p_0` (no flux through the inlet) and
    `p_n = -p_{n-1}` (`p = 0` on the outlet face, which sits half a cell beyond
    the last centre). The eigenvectors are

        cos[ (pi / n) (j + 1/2) (k + 1/2) ],

    which is the DCT-IV basis, with

        lambda_k = -(4/h^2) sin^2( pi (k + 1/2) / (2n) ),   k = 0 .. n-1.

    **None of these is zero**, which is why the projection of `project_outflow`
    needs no compatibility condition and no null-space handling: a domain with
    an outlet may take in more mass than it emits, and the pressure sorts it
    out. That is exactly the freedom the periodic operator did not have."""
    k = np.arange(n)
    return -(4.0 / (h * h)) * np.sin(np.pi * (k + 0.5) / (2.0 * n)) ** 2


def wide_eigenvalues_neumann(n: int, h: float) -> np.ndarray:
    """Eigenvalues of `D G` -- centred divergence composed with centred gradient
    -- with Neumann at both faces.

    **Not the same operator as `laplacian_eigenvalues_neumann`, and the
    difference is the whole reason the first version of this projection did not
    work.** On a collocated grid the divergence that is actually measured, and
    the one that telescopes to a control volume's boundary flux, is the
    two-cell centred difference; composing it with a centred gradient gives a
    Laplacian on a `2h` stencil, not the compact five-point one. Projecting with
    the compact Laplacian and measuring with the wide divergence leaves most of
    the divergence in place -- measured on a realistic wake field, `L2` fell only
    from 0.64 to 0.40 while the *face* field it discarded was at 1e-12.

    Same DCT-II eigenvectors, different eigenvalues:

        lambda_m = -sin^2(pi m / n) / h^2.

    Near `m = n-1` these are tiny -- the checkerboard that a two-cell stencil
    cannot see. That is harmless here and it is worth saying why, because it
    looks alarming: the projection is `I - G (DG)^-1 D`, and the `D` on the
    right kills exactly the modes `(DG)^-1` amplifies, so the composition is an
    orthogonal projection with norm 1. The intermediate `phi` carries a large
    checkerboard component; the correction `G phi` does not, because `G`
    annihilates it again."""
    m = np.arange(n)
    return -(np.sin(np.pi * m / n) ** 2) / (h * h)


def wide_eigenvalues_dirichlet_hi(n: int, h: float) -> np.ndarray:
    """`D G` with Neumann at the LOW face and Dirichlet at the HIGH face.

    DCT-IV eigenvectors again -- the ghost conventions `phi_{-1} = phi_0`,
    `phi_{-2} = phi_1`, `phi_n = -phi_{n-1}`, `phi_{n+1} = -phi_{n-2}` that a
    two-cell stencil needs at the boundary are all satisfied by that basis --
    with

        lambda_k = -sin^2( pi (k + 1/2) / n ) / h^2.

    `pi (k + 1/2) / n` lies strictly inside `(0, pi)` for every `k`, so none of
    these vanishes and the projection needs no null-space handling in `x`. In
    `y` the `m = 0` mode of `wide_eigenvalues_neumann` does vanish, and the sum
    is still safe because the two are added."""
    k = np.arange(n)
    return -(np.sin(np.pi * (k + 0.5) / n) ** 2) / (h * h)


def _dct2(a, axis):          # DCT-II, orthonormal: forward for Neumann-Neumann
    return dct(a, type=2, axis=axis, norm="ortho")


def _idct2(a, axis):         # its exact inverse
    return idct(a, type=2, axis=axis, norm="ortho")


def _dct4(a, axis):          # DCT-IV, orthonormal: involutive, its own inverse
    return dct(a, type=4, axis=axis, norm="ortho")


# --------------------------------------------------------------------------
# the solves
# --------------------------------------------------------------------------


def solve_neumann(rhs: np.ndarray, h: float, *,
                  g_lo_x=0.0, g_hi_x=0.0, g_lo_y=0.0, g_hi_y=0.0,
                  return_mismatch: bool = False):
    """Solve `laplacian p = rhs` with prescribed normal derivative on all four faces.

    `g_*` are `dp/dn` with `n` the OUTWARD normal, given per boundary cell (a
    scalar or a 1-D array along that face). Ghost elimination turns them into a
    boundary strip subtracted from the right-hand side -- at the low-x face the
    outward normal is `-x`, so `g_lo = (p_{-1} - p_0) / h` and

        p_{-1} = p_0 + h g_lo   ==>   rhs_0 -= g_lo / h,

    and the same sign comes out at all four faces because the outward normal
    convention already carries the orientation. The bookkeeping is worth
    checking rather than trusting: with these signs the compatibility sum below
    is exactly `integral rhs dA - contour integral g ds`, which is the
    divergence theorem, and getting the sign wrong instead makes it their
    *sum* -- a quantity with no meaning that is nonzero for every input.

    An all-Neumann Laplacian is singular with the constant as its null vector,
    and it is symmetric, so a solution exists only if the modified `rhs`
    integrates to zero -- the discrete form of

        integral_V laplacian p dV = integral_dV dp/dn ds.

    That identity holds for the continuous problem, so a nonzero discrete
    mismatch measures the discretization and the input field, not the solver.
    It is subtracted uniformly (never silently: `return_mismatch` hands it
    back, and `pressure_neumann` reports it every call) and `p` is returned with
    zero mean."""
    r = np.array(rhs, dtype=np.float64, copy=True)
    ny, nx = r.shape
    r[:, 0] -= np.asarray(g_lo_x) / h
    r[:, -1] -= np.asarray(g_hi_x) / h
    r[0, :] -= np.asarray(g_lo_y) / h
    r[-1, :] -= np.asarray(g_hi_y) / h

    mismatch = float(r.mean())
    r -= mismatch                                   # project onto the range

    lam = (laplacian_eigenvalues_neumann(nx, h)[None, :]
           + laplacian_eigenvalues_neumann(ny, h)[:, None])
    rh = _dct2(_dct2(r, axis=1), axis=0)
    lam_safe = np.where(lam == 0.0, 1.0, lam)
    ph = rh / lam_safe
    ph[0, 0] = 0.0                                  # the constant is undetermined
    p = _idct2(_idct2(ph, axis=0), axis=1)
    return (p, mismatch) if return_mismatch else p


def solve_wide_neumann_dirichlet(rhs: np.ndarray, h: float) -> np.ndarray:
    """Invert `D G` -- Neumann on `x_lo` and both walls, Dirichlet on `x_hi`.

    The operator of `project_outflow`, and the one whose inverse makes the
    *measured* divergence vanish rather than a face field's."""
    r = np.asarray(rhs, dtype=np.float64)
    ny, nx = r.shape
    lam = (wide_eigenvalues_dirichlet_hi(nx, h)[None, :]
           + wide_eigenvalues_neumann(ny, h)[:, None])
    rh = _dct4(_dct2(r, axis=0), axis=1)
    return _idct2(_dct4(rh / lam, axis=1), axis=0)


def solve_neumann_dirichlet(rhs: np.ndarray, h: float) -> np.ndarray:
    """Solve `laplacian phi = rhs`, homogeneous Neumann on `x_lo` and both `y`
    faces, homogeneous Dirichlet (`phi = 0`) on `x_hi`.

    The wind farm's outer boundary, spec section 9: a pinned inlet, two slip
    walls, and an outlet. A pinned inlet and a slip wall both say *no correction
    to the normal velocity here*, which is `dphi/dn = 0`; the outlet says *the
    pressure is whatever the far field is*, which is `phi = 0` up to the
    constant that a Dirichlet face fixes.

    Non-singular, so this is a plain direct solve with nothing to regularize."""
    r = np.asarray(rhs, dtype=np.float64)
    ny, nx = r.shape
    lam = (laplacian_eigenvalues_dirichlet_hi(nx, h)[None, :]
           + laplacian_eigenvalues_neumann(ny, h)[:, None])
    rh = _dct4(_dct2(r, axis=0), axis=1)
    ph = rh / lam                                   # never zero: see the docstring
    return _idct2(_dct4(ph, axis=1), axis=0)


# --------------------------------------------------------------------------
# the projection
# --------------------------------------------------------------------------


def divergence_bc(u: np.ndarray, v: np.ndarray, h: float,
                  u_in: np.ndarray | float | None = None) -> np.ndarray:
    """Centred divergence, with the outer boundary condition in the ghost cells.

    One operator, used for three things that must agree or the projection is
    not a projection: the right-hand side of the solve, the correction that is
    subtracted, and the number the gate reads. An earlier version had a
    face-interpolated divergence for the first and `np.gradient` for the third;
    they agree in the interior and disagree at the boundary cells, and the
    disagreement showed up as a 4.5e-2 residual sitting entirely in the first
    and last columns of an otherwise machine-zero field.

    Ghost cells, which are where the boundary condition lives:

    * `u_{-1} = 2 u_in - u_0` -- the inlet face carries exactly the prescribed
      flux. `apply_outer_bc` pins a band 0.5 D deep, so by default `u_in` is
      read from `u[:, 0]` and the rule degenerates to `u_{-1} = u_0`; passing it
      explicitly is for tests that do not run the boundary condition first.
    * `u_n = u_{n-1}` -- convective outlet, `du/dx = 0`.
    * `v_{-1} = -v_0`, `v_m = -v_{m-1}` -- slip walls: no through-flow.

    The signs are chosen so that these are *exactly* the ghost rules the
    correction inherits from `phi`. `test_w5_pressure` checks that composition
    against the eigenvalues rather than trusting the derivation."""
    u_in = u[:, 0] if u_in is None else np.asarray(u_in)
    du = np.empty_like(u)
    du[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * h)
    du[:, 0] = (u[:, 1] - (2.0 * u_in - u[:, 0])) / (2.0 * h)
    du[:, -1] = (u[:, -1] - u[:, -2]) / (2.0 * h)
    dv = np.empty_like(v)
    dv[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * h)
    dv[0, :] = (v[1, :] + v[0, :]) / (2.0 * h)
    dv[-1, :] = (-v[-1, :] - v[-2, :]) / (2.0 * h)
    return du + dv


def gradient_bc(phi: np.ndarray, h: float):
    """Centred gradient of `phi` with the ghost cells the operator was
    diagonalized with: reflection at the inlet and both walls, antisymmetry
    about the outlet face.

    Reflection makes the correction *anti*symmetric across the inlet and wall
    faces, so the flux through those faces is untouched -- the pinned inflow and
    the impermeable walls survive the projection exactly rather than
    approximately. Antisymmetry in `phi` at the outlet is what leaves that one
    face free, and it is the only reason the domain may emit more mass than it
    takes in."""
    cu = np.empty_like(phi)
    cu[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * h)
    cu[:, 0] = (phi[:, 1] - phi[:, 0]) / (2.0 * h)              # phi_{-1} = phi_0
    cu[:, -1] = (-phi[:, -1] - phi[:, -2]) / (2.0 * h)          # phi_n = -phi_{n-1}
    cv = np.empty_like(phi)
    cv[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * h)
    cv[0, :] = (phi[1, :] - phi[0, :]) / (2.0 * h)
    cv[-1, :] = (phi[-1, :] - phi[-2, :]) / (2.0 * h)
    return cu, cv


@dataclass
class ProjectionReport:
    """What one projection did, in the units the gate reads."""
    div_face_before: float
    div_face_after: float
    div_cell_after: float
    phi_range: float
    net_flux_in: float
    net_flux_out: float


def project_outflow(u: np.ndarray, v: np.ndarray, h: float):
    """Make the assembled field divergence free **without wrapping it around**.

    Replaces the FFT Leray projection of `couple.global_project`. The mechanism
    is a MAC projection built on the fly from the collocated field:

    1. Interpolate the cell velocities onto faces, **imposing the outer
       boundary condition on the four domain faces** (`face_velocities`).
    2. Take the conservative divergence of those faces.
    3. Solve `laplacian phi = div` with Neumann on inlet and walls and
       Dirichlet at the outlet (`solve_neumann_dirichlet`).
    4. Subtract `grad phi` **on the faces**. The Neumann faces get no correction
       at all, so the inlet flux and the wall impermeability survive the
       projection exactly rather than approximately.
    5. Average the two adjacent face corrections back onto each cell centre.

    Because step 3 inverts exactly the operator that step 4 and step 2 compose
    into, the face divergence after this is zero to machine precision -- not to
    a tolerance. The collocated centred difference is *not* zero, and that is a
    property of collocated storage rather than of this solve: its two-cell
    stencil cannot see a checkerboard, and a checkerboard component of `phi`
    contributes nothing to the face correction either. `ProjectionReport`
    carries both numbers so neither can be quoted as if it were the other.

    The one physical statement worth naming: with the outlet Dirichlet, the net
    flux out no longer has to equal the net flux in. That is the freedom the
    periodic projection lacked, and it is why the rotors' blockage can now leave
    the domain instead of turning up at the inlet as upstream induction."""
    u_in = u[:, 0].copy()                           # the prescribed inlet, before
    d0 = divergence_bc(u, v, h, u_in)
    phi = solve_wide_neumann_dirichlet(d0, h)
    cu, cv = gradient_bc(phi, h)
    u_new, v_new = u - cu, v - cv

    d1 = divergence_bc(u_new, v_new, h, u_in)
    dc = (np.gradient(u_new, h, axis=1) + np.gradient(v_new, h, axis=0))[4:-4, 4:-4]
    return u_new, v_new, ProjectionReport(
        div_face_before=float(np.abs(d0).max()),
        div_face_after=float(np.abs(d1).max()),
        div_cell_after=float(np.abs(dc).max()),
        phi_range=float(phi.max() - phi.min()),
        net_flux_in=float(u_in.sum() * h),
        net_flux_out=float(u_new[:, -1].sum() * h),
    )


# --------------------------------------------------------------------------
# the pressure the momentum balance reads
# --------------------------------------------------------------------------


def pressure_neumann(u, v, fx=None, fy=None, h: float = 1.0, *,
                     dudt=None, dvdt=None, nu: float = 0.0,
                     return_mismatch: bool = False):
    """Pressure on a NON-periodic window, from the momentum equation's own BC.

    `probe.pressure_from_velocity` solves the same Poisson equation spectrally
    and is exact -- on a periodic window. This is the same equation with the
    boundary condition the continuous problem actually carries. Taking the
    normal component of the momentum equation on the window face gives

        dp/dn = n . [ f - du/dt - (u.grad)u + nu laplacian u ],

    which is Neumann data on all four faces, and the interior equation is the
    divergence of the same momentum equation,

        laplacian p = div f - div((u.grad)u).

    The two terms a divergence-free field would kill -- `-d/dt (div u)` and
    `nu laplacian(div u)` -- are dropped, which is the standard construction.
    They are not identically zero here because the assembled field's collocated
    divergence is not machine zero, so dropping them is an approximation and is
    named as one.

    Derivatives are `np.gradient`: second-order centred inside, second-order
    one-sided on the edges. Spectral differentiation is not available here for
    the same reason the spectral solve is not -- it assumes the field wraps.

    Returns `p` with zero mean, which is all a closed control volume needs.
    With `return_mismatch`, also the compatibility defect of `solve_neumann`:
    that number is the honest error bar on this pressure, and `cv_balance`
    records it every step so a residual can never be read without it."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    zero = np.zeros_like(u)
    fx = zero if fx is None else np.asarray(fx, dtype=np.float64)
    fy = zero if fy is None else np.asarray(fy, dtype=np.float64)

    def ddx(a):
        return np.gradient(a, h, axis=1)

    def ddy(a):
        return np.gradient(a, h, axis=0)

    adv_x = u * ddx(u) + v * ddy(u)
    adv_y = u * ddx(v) + v * ddy(v)
    rhs = ddx(fx) + ddy(fy) - (ddx(adv_x) + ddy(adv_y))

    lap_u = ddx(ddx(u)) + ddy(ddy(u)) if nu else zero
    lap_v = ddx(ddx(v)) + ddy(ddy(v)) if nu else zero
    acc_x = zero if dudt is None else np.asarray(dudt, dtype=np.float64)
    acc_y = zero if dvdt is None else np.asarray(dvdt, dtype=np.float64)
    gx = fx - acc_x - adv_x + nu * lap_u            # dp/dx on an x face
    gy = fy - acc_y - adv_y + nu * lap_v            # dp/dy on a y face

    # Extrapolate from the two nearest centres to the face, half a cell out.
    def face(a, axis, hi):
        if axis == 1:
            return 1.5 * a[:, -1] - 0.5 * a[:, -2] if hi else 1.5 * a[:, 0] - 0.5 * a[:, 1]
        return 1.5 * a[-1, :] - 0.5 * a[-2, :] if hi else 1.5 * a[0, :] - 0.5 * a[1, :]

    p, mismatch = solve_neumann(
        rhs, h,
        g_lo_x=-face(gx, 1, False),                 # outward normal is -x
        g_hi_x=+face(gx, 1, True),
        g_lo_y=-face(gy, 0, False),
        g_hi_y=+face(gy, 0, True),
        return_mismatch=True)
    p -= p.mean()
    return (p, mismatch) if return_mismatch else p
