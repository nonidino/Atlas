"""A classical pseudo-spectral Navier-Stokes solver on the probe window.

Not part of the model. This exists so that W3's `r_T` is a *result* rather than
a number: the same control-volume balance, the same Strang splitting and the
same body force, driven by an evolution that is known to solve the equations,
establishes what the metric reads when the physics is right. Without it, an
`r_T` of 8% cannot be separated into "the expert is wrong" and "the balance
operator is wrong", which is exactly the confusion the guide's rule about
building metrics before the system is meant to prevent.

It also gives W3 a miniature of the W9 monolithic baseline for free.

Method: vorticity-streamfunction on the periodic square, 2/3 dealiased, with an
exact integrating factor for the viscous term and Heun (RK2) for advection.
Sub-steps internally so the caller can ask for the same macro-step the frozen
expert takes. `step(u, v, dt)` matches `FrozenFluidExpert.step`, so the probe
does not know which one it is driving.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, idct

from .adapters import EXPERT_RES, Scaling
from .backend import get_backend


@dataclass
class SpectralNS:
    """Reference solver. `nu` is in case-study units (`D = U_inf = 1`)."""
    nu: float
    length: float = 2.0
    n: int = EXPERT_RES
    cfl: float = 0.4
    scaling: Scaling | None = None            # duck-types FrozenFluidExpert
    galilean: bool = False

    def __post_init__(self):
        n, L = self.n, self.length
        k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
        k_ny = k.copy()
        if n % 2 == 0:
            k_ny[n // 2] = 0.0
        self.kx = k_ny[None, :]
        self.ky = k_ny[:, None]
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.k2_inv = np.where(self.k2 == 0.0, 1.0, self.k2)
        kmax = np.abs(k).max()
        self.dealias = (np.abs(self.kx) <= 2.0 / 3.0 * kmax) & (np.abs(self.ky) <= 2.0 / 3.0 * kmax)
        self.n_calls = 0
        self.last_mean_drift = (0.0, 0.0)
        if self.scaling is None:
            self.scaling = Scaling(length=L, velocity=1.0)

    # -- transforms --------------------------------------------------------

    def vorticity(self, u, v):
        return np.real(np.fft.ifft2(1j * self.kx * np.fft.fft2(v)
                                    - 1j * self.ky * np.fft.fft2(u)))

    def velocity(self, wh, u_mean=0.0, v_mean=0.0):
        psih = wh / self.k2_inv
        psih[0, 0] = 0.0
        u = np.real(np.fft.ifft2(1j * self.ky * psih)) + u_mean
        v = np.real(np.fft.ifft2(-1j * self.kx * psih)) + v_mean
        return u, v

    def _rhs(self, wh, u_mean, v_mean):
        """-(u.grad)w, dealiased."""
        u, v = self.velocity(wh, u_mean, v_mean)
        wx = np.real(np.fft.ifft2(1j * self.kx * wh))
        wy = np.real(np.fft.ifft2(1j * self.ky * wh))
        adv = np.fft.fft2(u * wx + v * wy)
        return -adv * self.dealias

    # -- the interface the probe drives ------------------------------------

    def step(self, u, v, dt, project: bool = False, galilean=None, force=None):
        """Advance by `dt`, sub-stepping to respect the advective CFL.

        The mean velocity is carried analytically rather than through the
        vorticity, which is exact: a constant mean has zero vorticity and, on a
        periodic domain, is conserved by the unforced equations."""
        u_mean, v_mean = float(np.mean(u)), float(np.mean(v))
        wh = np.fft.fft2(self.vorticity(u, v))
        umax = float(np.max(np.hypot(u, v))) + 1e-12
        dx = self.length / self.n
        n_sub = max(1, int(np.ceil(dt / (self.cfl * dx / umax))))
        h = dt / n_sub
        for _ in range(n_sub):
            # advection in the frame moving with the mean, then translate: the
            # same exact Galilean step the frozen expert is given, so the two
            # are compared on equal terms
            e1 = np.exp(-self.nu * self.k2 * h)
            e2 = np.exp(-self.nu * self.k2 * h * 0.5)
            k1 = self._rhs(wh, 0.0, 0.0)
            wh_p = e1 * (wh + h * k1)
            k2_ = self._rhs(wh_p, 0.0, 0.0)
            wh = e1 * wh + h * 0.5 * (e1 * k1 + k2_)
            wh *= self.dealias
            shift = np.exp(-1j * (self.kx * u_mean * h + self.ky * v_mean * h))
            wh = wh * shift
            _ = e2
        self.n_calls += 1
        return self.velocity(wh, u_mean, v_mean)

    step_unchecked = step


__all__ = ["SpectralNS"]


# --------------------------------------------------------------------------
# the monolithic baseline -- the whole domain, undivided, no expert
# --------------------------------------------------------------------------


@dataclass
class ChannelNS:
    """2-D incompressible Navier-Stokes on the **wind-farm domain itself**.

    `SpectralNS` above is periodic on a 2 D square, which is what W3's probe
    needed and is the wrong instrument for any question about the outer
    boundary. This solves the same equations on the real domain with the real
    boundary conditions -- pinned inlet, slip walls, open outlet -- reusing the
    projection of `pressure.project_outflow`, which exists because W5 needed it
    for the coupled system.

    Two jobs, and it was built for the first:

    1. **Is the upstream induction the coupled system produces correct?**
       W5 measured a 17.3% centreline deficit one diameter ahead of turbine 1
       and called it excessive against 3-D actuator-disk theory. This answers
       it with the same disk, the same domain, the same boundary conditions and
       a real fluid operator, so the *only* difference from the coupled run is
       the frozen expert.
    2. It is most of [[impl-wind-farm-guide]] section 7.3's **W9 monolithic
       baseline**, which the W3 entry already predicted would come cheap once a
       reference solver existed.

    Method: explicit projection. Advect and diffuse with second-order centred
    differences, add the body force, impose the outer boundary condition, then
    project. Centred advection needs a cell Reynolds number of order 1 for a
    wiggle-free steady state, so `nu` is a parameter and the caller is expected
    to vary it -- upstream induction is a potential-flow effect and must come
    out insensitive to it. If it does not, the answer is not converged."""
    nu: float
    dx: float
    u_inf: float = 1.0
    cfl: float = 0.4
    #: Domain override for the N-turbine sweep (`windfarm_n_sweep.py`):
    #: `geometry.build_geometry_case(n).domain`. `None` (the default) keeps the
    #: original N=2 behaviour byte-for-bit, reading `geometry.DOMAIN` exactly as
    #: before.
    domain: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if self.domain is not None:
            x0, x1, y0, y1 = self.domain
        else:
            from .geometry import DOMAIN
            x0, x1, y0, y1 = DOMAIN
        self.nx = int(round((x1 - x0) / self.dx))
        self.ny = int(round((y1 - y0) / self.dx))
        self.x_c = x0 + (np.arange(self.nx) + 0.5) * self.dx
        self.y_c = y0 + (np.arange(self.ny) + 0.5) * self.dx
        self.u = np.full((self.ny, self.nx), float(self.u_inf))
        self.v = np.zeros((self.ny, self.nx))
        self.t = 0.0

    # -- operators ---------------------------------------------------------

    def _ddx(self, f):
        return np.gradient(f, self.dx, axis=1)

    def _ddy(self, f):
        return np.gradient(f, self.dx, axis=0)

    def _lap(self, f):
        return (np.roll(f, 1, 1) + np.roll(f, -1, 1)
                + np.roll(f, 1, 0) + np.roll(f, -1, 0) - 4.0 * f) / self.dx ** 2

    def _rhs(self, u, v, fx):
        adv_u = u * self._ddx(u) + v * self._ddy(u)
        adv_v = u * self._ddx(v) + v * self._ddy(v)
        return (-adv_u + self.nu * self._lap(u) + fx,
                -adv_v + self.nu * self._lap(v))

    def dt_max(self) -> float:
        umax = float(np.max(np.hypot(self.u, self.v))) + 1e-12
        return min(self.cfl * self.dx / umax, 0.25 * self.dx ** 2 / max(self.nu, 1e-12))

    def step(self, dt: float, fx) -> None:
        """One Heun step plus a projection. `fx` may be a callable of `(u, v)`
        so the disk can read its own inflow, which is the local-induction form
        spec section 4.2 requires and the reason turbine 2 responds to turbine
        1's wake rather than to the inlet."""
        from .couple import apply_outer_bc, GlobalField
        from .pressure import project_outflow

        f = fx(self.u, self.v) if callable(fx) else fx
        k1u, k1v = self._rhs(self.u, self.v, f)
        pu, pv = self.u + dt * k1u, self.v + dt * k1v
        k2u, k2v = self._rhs(pu, pv, f)
        self.u = self.u + 0.5 * dt * (k1u + k2u)
        self.v = self.v + 0.5 * dt * (k1v + k2v)

        g = GlobalField(self.dx, self.x_c, self.y_c, self.u, self.v)
        apply_outer_bc(g, self.t + dt, band=max(0.5, 8 * self.dx), u_inf=self.u_inf)
        self.u, self.v, _ = project_outflow(g.u, g.v, self.dx)
        self.t += dt

    def run(self, t_end: float, fx, on_step=None) -> None:
        while self.t < t_end - 1e-12:
            dt = min(self.dt_max(), t_end - self.t)
            self.step(dt, fx)
            if on_step is not None:
                on_step(self)

    def centreline(self, half: float = 0.5) -> np.ndarray:
        """Disk-averaged `u` down the centreline, the same average the rotor
        takes, so a deficit here is comparable with `disk_velocity`."""
        m = np.abs(self.y_c) <= half
        return self.u[m, :].mean(axis=0)


def make_forcing(solver: "ChannelNS", disks, thickness: float = 0.25):
    """`f_x(u, v)` for both rotors, each reading its own local inflow.

    Spec section 4.2's local-induction form: the thrust follows the velocity
    one cell upstream of the rotor face, which is what makes turbine 2 respond
    to turbine 1's wake instead of to the domain inlet. Identical in structure
    to `couple.rotor_force` + `couple.disk_velocity`, so the coupled system and
    this baseline are driven by the same rotor model."""
    from .geometry import ROTOR_HALFSPAN
    strips = []
    for x_face, d in disks:
        unit = d.body_force_field(solver.x_c, solver.y_c, solver.dx, solver.dx,
                                  1.0, x0=x_face, y0=-ROTOR_HALFSPAN)
        strips.append((x_face, d, unit / d.force_density(1.0) * -1.0))
    m = np.abs(solver.y_c) <= ROTOR_HALFSPAN

    def fx(u, v):
        out = np.zeros_like(u)
        for x_face, d, frac in strips:
            i = int(np.argmin(np.abs(solver.x_c - (x_face - solver.dx))))
            thrust = d(u_disk=float(np.mean(u[m, i]))).thrust
            out -= d.force_density(thrust) * frac
        return out

    return fx





# --------------------------------------------------------------------------
# the non-periodic window solve -- OP-5's precondition
# --------------------------------------------------------------------------


@dataclass
class WindowNS:
    """2-D incompressible Navier-Stokes on ONE expert window, **not periodic**.

    **Why this exists, and it is not a performance variant.** `SpectralNS` is
    FFT-based, so every window it solves wraps: what leaves the right edge
    re-enters at the left. A Schwarz fixed point is the global solution *only
    if each local solve is the exact restriction of the global operator to its
    subdomain*, and a periodic box is not the restriction of an open channel.
    Iterating an interface exchange around locally-wrong solves converges --
    to the fixed point of the wrong map, making the tiles agree with each other
    more precisely while leaving them all wrong the same way, and removing the
    interface residual that was the only visible symptom.

    That is why per-window periodicity is a *precondition* of iterating rather
    than a peer of it, and why this class had to exist before a Schwarz loop
    could mean anything. See OP-5.

    **The measurement that forced it.** The assembled sweep is a constant map in
    the iteration index: `gather` reads the whole 128x128 window out of the
    global field at `t^n` and the operator returns one field, so re-running it
    reproduces the previous result *bitwise*. There is no boundary channel to
    iterate on. This class supplies one -- the Dirichlet ring is separate from
    the interior initial condition, so the neighbours' current iterate can enter
    through it while the initial condition stays pinned at `t^n`.

    Method: explicit fractional step. Skew-symmetric centred advection, centred
    diffusion, body force in the right-hand side, Dirichlet ring re-imposed after
    every stage, then a projection whose pressure carries homogeneous Neumann
    data on all four faces -- which is the condition that leaves the prescribed
    normal velocity untouched.

    **Skew-symmetric advection, deliberately.** The cell Reynolds number here is
    `h U / nu = 0.0156 / (1/255) = 4`, twice the classical limit for wiggle-free
    centred convection, and the obvious remedies both damage the thing under
    test: upwinding adds numerical diffusion to a system whose known defect is
    over-dissipation, and raising `nu` changes the physics. The skew-symmetric
    form `1/2[u.grad f + div(uf)]` conserves kinetic energy for a
    divergence-free field, so it is stable at this cell Reynolds number without
    adding any dissipation at all. It costs one extra derivative pair.

    Measured against the alternatives rather than assumed, on Taylor-Green, the
    analytic wake-diffusion check, and eight macro-steps of a rough field:

        skew   cfl | TG rel err   wake retention   stable   124-window sweep
        True   0.4 |   9.42e-03      0.86491        yes         91 s
        True   0.8 |   1.44e-02      0.86485        yes          -
        False  0.4 |   9.47e-03      0.86491        yes         66 s
        False  0.6 |   1.19e-02      0.86488        yes         47 s
                                     ^ analytic: 0.86111

    The two forms are indistinguishable to five digits, which is the expected
    result rather than a surprising one: they differ by `f * div(u)`, and the
    projection above drives `div(u)` to machine zero every sub-step. So the skew
    form is kept for what it protects against -- the strong shear layers at the
    wake edges, where these smooth tests say nothing -- and not because it was
    measured to be more accurate here. `cfl` is left at 0.4 for the same reason:
    0.8 was stable on every test above and buys a factor of two, and none of
    those tests contains a wake edge.
    """
    nu: float
    length: float = 2.0
    n: int = EXPERT_RES
    cfl: float = 0.4
    skew: bool = True
    #: What the ring imposes. `'characteristic'` pins the velocity only where the
    #: flow **enters** the window and lets it leave under zero normal gradient;
    #: `'dirichlet'` pins the whole ring.
    #:
    #: Pure Dirichlet is what classical alternating Schwarz uses, and for an
    #: elliptic problem it is right. For an advection-dominated one it is
    #: over-constrained: prescribing velocity on an *outflow* boundary is
    #: ill-posed, and here it has a concrete consequence -- the wake deficit the
    #: rotor creates cannot leave the window, because the cells it would leave
    #: through are pinned to upstream data from `t^n`.
    #:
    #: The over-constraint is exactly what a converged Schwarz iteration would
    #: relieve, since at the fixed point the ring *is* the true solution. So the
    #: two are alternative repairs of the same defect: iterate until the pinned
    #: outflow happens to be right, or stop pinning the outflow. The second costs
    #: nothing per step, which is why it is the default. `'dirichlet'` is kept as
    #: the control the first is read against.
    transmission: str = "characteristic"

    #: Array backend for the interior kernels -- `'numpy'` (the default, and the
    #: reference path) or `'torch'`. Profiling put 95% of a macro-step in the
    #: batched stencils on a `[124, 128, 128]` array, and a CPU thread pool
    #: saturates at 2.8x because the wall is memory bandwidth, so `'torch'` with
    #: `device='cuda'` is the lever that actually moves this. See `backend.py`.
    #: **The physics is unchanged**: `tests/atlas/windfarm/test_backend.py` pins
    #: torch-CPU against numpy on a real `step_batch`.
    backend: str = "numpy"
    device: str = "cpu"

    def __post_init__(self):
        from .pressure import wide_eigenvalues_neumann
        if self.transmission not in ("characteristic", "dirichlet", "convective"):
            raise ValueError(self.transmission)
        self.h = self.length / self.n
        self.b = get_backend(self.backend, self.device)
        # Built in numpy and handed over once: these are constants of the
        # discretization, not state, so they are the one thing that should NOT
        # go through the backend's elementwise layer.
        ev = wide_eigenvalues_neumann(self.n, self.h)
        lam = ev[None, None, :] + ev[None, :, None]                # [1, n, n]
        self._lam = self.b.asarray(lam)
        self._lam_safe = self.b.asarray(np.where(lam == 0.0, 1.0, lam))
        self.n_calls = 0
        self.last_substeps = 0
        self.last_mismatch = 0.0
        self.last_div = 0.0

    # -- operators, all batched over the leading axis ----------------------

    def _ddx(self, f):
        """Centred difference under an **even** (Neumann) extension.

        `np.gradient` would do the interior identically and the two edge cells
        differently -- it drops to a one-sided first-order difference there,
        which does not match the DCT-II basis the Poisson solve is diagonalized
        in. The edge cells are re-pinned to the Dirichlet ring immediately after
        every use, so the difference never survives; matching the basis exactly
        is still worth the two lines, because the *divergence* fed to the solve
        is taken with this operator and an edge inconsistency there pollutes
        `phi` over the whole window rather than at its rim.

        Written with slices rather than `np.pad` because the pad allocates a
        fresh `[B, n, n+2]` array on every call and this is the innermost loop of
        the whole case study -- roughly forty of these per sub-step, fifty-odd
        sub-steps per macro-step, 124 windows at a time."""
        c = 1.0 / (2.0 * self.h)
        out = self.b.empty_like(f)
        out[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) * c
        out[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) * c        # even extension
        out[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) * c
        return out

    def _ddy(self, f):
        c = 1.0 / (2.0 * self.h)
        out = self.b.empty_like(f)
        out[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) * c
        out[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) * c
        out[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) * c
        return out

    def _lap(self, f):
        """5-point Laplacian with **edge** padding, never `np.roll`.

        `ChannelNS._lap` rolls, which wraps -- harmless there because the ring is
        overwritten, but this class exists precisely to remove wrapping, and a
        periodic stencil hidden inside a non-periodic solver is the kind of thing
        that reads as a small boundary error for a week."""
        c = 1.0 / self.h ** 2
        out = self.b.empty_like(f)
        out[:, 1:-1, 1:-1] = (f[:, 1:-1, 2:] + f[:, 1:-1, :-2]
                              + f[:, 2:, 1:-1] + f[:, :-2, 1:-1]
                              - 4.0 * f[:, 1:-1, 1:-1]) * c
        # The rim is re-pinned to the Dirichlet ring immediately after every use,
        # so its value is discarded; zero keeps the reflected ghost out of it.
        out[:, 0, :] = out[:, -1, :] = 0.0
        out[:, :, 0] = out[:, :, -1] = 0.0
        return out

    def _adv(self, u, v, f):
        a1 = u * self._ddx(f) + v * self._ddy(f)
        if not self.skew:
            return a1
        a2 = self._ddx(u * f) + self._ddy(v * f)
        return 0.5 * (a1 + a2)

    def _rhs(self, u, v, fx, fy):
        return (-self._adv(u, v, u) + self.nu * self._lap(u) + fx,
                -self._adv(u, v, v) + self.nu * self._lap(v) + fy)

    def _pin(self, a, ring, ring_u, ring_v, inflow_only: bool = False,
             open_faces=None):
        """Impose the ring. One cell wide, which is the whole boundary condition
        for a collocated 5-point stencil: the first interior cell differences
        against it and nothing reaches further.

        Under `transmission='characteristic'` the ring is imposed only on the
        **inflow** part of each face; where the flow leaves, the boundary cell
        takes its interior neighbour's value, which is the discrete zero normal
        gradient. Inflow is decided from the ring's own normal velocity, not from
        the interior, so the test is a property of the data being imposed rather
        than of the state being overwritten -- otherwise a cell can flip its own
        condition mid-step and the boundary chatters.

        `ring_u`/`ring_v` are always the *velocity* ring; `a` may be either
        component, which is why the normal velocity is passed separately."""
        if self.transmission == "dirichlet":
            a[:, 0, :] = ring[:, 0, :]
            a[:, -1, :] = ring[:, -1, :]
            a[:, :, 0] = ring[:, :, 0]
            a[:, :, -1] = ring[:, :, -1]
            return
        # `inflow_only` is what runs *after* the projection. The zero-gradient
        # outflow value exists to give the advection and diffusion stencils a
        # well-posed cell to difference against; it is not a constraint on the
        # answer, and re-imposing it after the projection un-does the projection
        # exactly where the flow leaves. Measured: doing so left an interior
        # divergence of 0.13 on Taylor-Green instead of 1e-15. So inflow is
        # prescribed hard, and outflow keeps whatever the projection produced.
        # Under 'convective' the outflow rim carries its own evolving value,
        # advanced once per sub-step by `_convect_outflow`. Copying the interior
        # neighbour over it here is precisely the O(1) per-sub-step overwrite
        # that condition exists to remove, so the outflow is never touched.
        if self.transmission == "convective":
            inflow_only = True
        xlo, xhi, ylo, yhi = self._faces(open_faces, a.shape[0])
        # low x: outward normal is -x, so flow enters where u > 0
        m = ring_u[:, :, 0] > 0.0
        o = self.b.where(m, ring[:, :, 0], a[:, :, 0] if inflow_only else a[:, :, 1])
        a[:, :, 0] = self.b.where(xlo, o, ring[:, :, 0])
        # high x: enters where u < 0
        m = ring_u[:, :, -1] < 0.0
        o = self.b.where(m, ring[:, :, -1], a[:, :, -1] if inflow_only else a[:, :, -2])
        a[:, :, -1] = self.b.where(xhi, o, ring[:, :, -1])
        # low y: enters where v > 0
        m = ring_v[:, 0, :] > 0.0
        o = self.b.where(m, ring[:, 0, :], a[:, 0, :] if inflow_only else a[:, 1, :])
        a[:, 0, :] = self.b.where(ylo, o, ring[:, 0, :])
        # high y: enters where v < 0
        m = ring_v[:, -1, :] < 0.0
        o = self.b.where(m, ring[:, -1, :], a[:, -1, :] if inflow_only else a[:, -2, :])
        a[:, -1, :] = self.b.where(yhi, o, ring[:, -1, :])

    def _faces(self, open_faces, B):
        """Per-face `[B, 1]` flags for *which faces are genuinely open*.

        The distinction the 2026-08-24 measurement forced. An open-boundary
        condition belongs on a face that is a real outlet of the **domain**. On
        an interior tile seam the neighbour's data is not merely available, it is
        the *right answer* -- classical Schwarz -- and extrapolating over it
        throws away good data and substitutes a condition that is inconsistent
        wherever streamwise structure crosses. Only 7 of the 124 tiles have a
        face on the true outlet; the other 117 were being given an open boundary
        on faces that are interior seams.

        `None` means every face is treated as open, which is the pre-2026-08-24
        behaviour and is what the unit tests on a *single* window intend."""
        if open_faces is None:
            t = self.b.bool_asarray(np.ones((B, 1), dtype=bool))
            return t, t, t, t
        f = self.b.bool_asarray(np.asarray(open_faces, dtype=bool).reshape(B, 4))
        return f[:, 0:1], f[:, 1:2], f[:, 2:3], f[:, 3:4]

    def _convect_outflow(self, a, hs, ring_u, ring_v, open_faces=None):
        """Advance the outflow rim by $\\partial_t a + U_c\\,\\partial_n a = 0$.

        The fix for the failure measured on 2026-08-24. `'characteristic'` sets
        the outflow rim equal to its interior neighbour -- a *full* O(1)
        replacement applied once per sub-step, with **no dependence on the
        sub-step size at all**. That is not a discretization of anything: refine
        the time step and the same O(1) overwrite happens more often, so the
        boundary error accumulates per *sub-step* rather than per unit time.
        Measured on a developed field with the rotor force switched off, one
        macro-step: `max|u|` of 1.17, 3.10, 11.87 at 48, 190 and 379 sub-steps,
        with the outlet velocity reversing from +0.92 to -3.74. Refining the
        time step made the answer worse, which is the signature of an
        inconsistent boundary condition rather than an unstable one.

        The convective (Orlanski) condition is the standard remedy and it fixes
        exactly that defect: the rim *relaxes* toward its neighbour at the rate
        the flow actually carries information out, so the change per sub-step is
        $O(h_s)$ and the accumulated change per unit time is mesh-independent.
        As $h_s \\to 0$ it converges; the zero-gradient condition is its
        $C \\to 1$ limit, applied unconditionally.

        `U_c` and the inflow/outflow decision both come from the **ring**, not
        from the interior, for the reason `_pin` documents: a cell that decides
        its own condition from the state being overwritten can flip mid-step and
        chatter. `C` is clipped to $[0, 1]$, which is the upwind stability limit
        and makes this reduce to the old behaviour only when the flow crosses a
        whole cell in one sub-step."""
        c = hs / self.h
        xlo, xhi, ylo, yhi = self._faces(open_faces, a.shape[0])

        def face(cur, nbr, speed, outflow):
            C = self.b.clip(speed * c, 0.0, 1.0)
            return self.b.where(outflow, cur - C * (cur - nbr), cur)

        # low x: outward normal -x, so the flow leaves where the ring's u < 0
        nrm = ring_u[:, :, 0]
        a[:, :, 0] = face(a[:, :, 0], a[:, :, 1], self.b.maximum(-nrm, 0.0), ~(nrm > 0.0) & xlo)
        # high x: leaves where u > 0
        nrm = ring_u[:, :, -1]
        a[:, :, -1] = face(a[:, :, -1], a[:, :, -2], self.b.maximum(nrm, 0.0), ~(nrm < 0.0) & xhi)
        # low y: leaves where v < 0
        nrm = ring_v[:, 0, :]
        a[:, 0, :] = face(a[:, 0, :], a[:, 1, :], self.b.maximum(-nrm, 0.0), ~(nrm > 0.0) & ylo)
        # high y: leaves where v > 0
        nrm = ring_v[:, -1, :]
        a[:, -1, :] = face(a[:, -1, :], a[:, -2, :], self.b.maximum(nrm, 0.0), ~(nrm < 0.0) & yhi)

    def _balance_flux_scaled(self, u, v, ring_u, ring_v, open_faces=None):
        """Compatibility by **rescaling** the outflow profile, not by shifting it.

        `_balance_flux` distributes the flux defect as a uniform additive `adj`
        over the outflow cells. Under `'characteristic'` that bias is harmless
        only because the next sub-step overwrites the whole rim with its interior
        neighbour, erasing it. Take the overwrite away -- which is the entire
        point of `'convective'` -- and the additive bias has nothing to erase it
        and accumulates: measured, `max|u|` of 1.23, 5.41, 28.2 at 48, 190, 379
        sub-steps, worse than the condition it was meant to repair.

        A uniform shift is also the wrong *direction* on its own terms, and for
        the reason [[open-problems-atlas-0.1]] OP-1 records about the min-norm
        thrust correction: a constant is the one profile an open boundary should
        never acquire, because it moves the whole face rather than the part of it
        carrying flow. Rescaling the outflow to pass exactly what enters keeps
        the profile's shape and its zeros, which is what an open boundary
        actually does. It falls back to the additive form only when there is no
        outflow to scale."""
        h = self.h
        xlo, xhi, ylo, yhi = self._faces(open_faces, u.shape[0])
        out_lo_x = ~(ring_u[:, :, 0] > 0.0) & xlo
        out_hi_x = ~(ring_u[:, :, -1] < 0.0) & xhi
        out_lo_y = ~(ring_v[:, 0, :] > 0.0) & ylo
        out_hi_y = ~(ring_v[:, -1, :] < 0.0) & yhi
        # outward-normal velocity on each face, counted only where flow leaves
        o_lo_x = self.b.where(out_lo_x, self.b.maximum(-u[:, :, 0], 0.0), 0.0)
        o_hi_x = self.b.where(out_hi_x, self.b.maximum(u[:, :, -1], 0.0), 0.0)
        o_lo_y = self.b.where(out_lo_y, self.b.maximum(-v[:, 0, :], 0.0), 0.0)
        o_hi_y = self.b.where(out_hi_y, self.b.maximum(v[:, -1, :], 0.0), 0.0)
        out = (o_lo_x.sum(1) + o_hi_x.sum(1) + o_lo_y.sum(1) + o_hi_y.sum(1)) * h
        # everything that is not counted as outflow above is inflow
        net = (-(u[:, :, 0]).sum(1) + (u[:, :, -1]).sum(1)
               - (v[:, 0, :]).sum(1) + (v[:, -1, :]).sum(1)) * h
        inflow = out - net                       # what must leave, by conservation
        ok = out > 1e-12 * max(self.length, 1.0)
        s = self.b.where(ok, inflow / self.b.where(out > 0, out, 1.0), 1.0)
        s = self.b.clip(s, 0.0, 10.0)[:, None]
        u[:, :, 0] = self.b.where(out_lo_x, u[:, :, 0] * s, u[:, :, 0])
        u[:, :, -1] = self.b.where(out_hi_x, u[:, :, -1] * s, u[:, :, -1])
        v[:, 0, :] = self.b.where(out_lo_y, v[:, 0, :] * s, v[:, 0, :])
        v[:, -1, :] = self.b.where(out_hi_y, v[:, -1, :] * s, v[:, -1, :])
        if not self.b.all_true(ok):
            return self._balance_flux(u, v, ring_u, ring_v, open_faces)
        return u, v

    def _balance_flux(self, u, v, ring_u, ring_v, open_faces=None):
        """Make the window's net boundary flux zero, on the outflow cells only.

        An all-Neumann pressure solve is singular, and a solution exists only if
        the right-hand side integrates to zero -- which by the divergence theorem
        is the statement that as much leaves the window as enters it. With the
        inflow ring prescribed and the outflow free, nothing enforces that, and
        `_poisson` would absorb the defect as a uniform constant. A uniform
        constant in the right-hand side is a *quadratic* in `phi`, whose gradient
        is nonzero over the whole window: measured, that added 4.8e-2 of interior
        divergence to a field which arrived with 2.4e-14.

        So the defect is put back where it belongs -- on the boundary the flow
        leaves through -- rather than spread over the interior. Physically this
        is the outflow adjusting to pass whatever the inflow delivers, which is
        what an open boundary does."""
        h = self.h
        xlo, xhi, ylo, yhi = self._faces(open_faces, u.shape[0])
        out_lo_x = ~(ring_u[:, :, 0] > 0.0) & xlo
        out_hi_x = ~(ring_u[:, :, -1] < 0.0) & xhi
        out_lo_y = ~(ring_v[:, 0, :] > 0.0) & ylo
        out_hi_y = ~(ring_v[:, -1, :] < 0.0) & yhi
        flux = (-(u[:, :, 0]).sum(1) + (u[:, :, -1]).sum(1)
                - (v[:, 0, :]).sum(1) + (v[:, -1, :]).sum(1)) * h
        n_out = (out_lo_x.sum(1) + out_hi_x.sum(1)
                 + out_lo_y.sum(1) + out_hi_y.sum(1))
        n_out = self.b.to_float(n_out)
        adj = self.b.where(n_out > 0, -flux / self.b.maximum(n_out, 1.0) / h, 0.0)
        u[:, :, 0] += self.b.where(out_lo_x, -adj[:, None], 0.0)      # outward normal -x
        u[:, :, -1] += self.b.where(out_hi_x, adj[:, None], 0.0)
        v[:, 0, :] += self.b.where(out_lo_y, -adj[:, None], 0.0)
        v[:, -1, :] += self.b.where(out_hi_y, adj[:, None], 0.0)
        return u, v

    def _project(self, u, v):
        """`I - G (DG)^-1 D`, with `D` and `G` the same centred operators.

        **Not a MAC projection, and the reason is dissipation.** The obvious
        collocated fix is to difference on faces -- interpolate cell values to
        faces, take the conservative divergence, correct the faces, average back
        to centres. It gives machine-zero face divergence, and the round trip
        `centre -> face -> centre` is the filter `[1, 2, 1]/4`, whose gain at a
        16-cell wavelength is 0.962 **per sub-step**. At 50 sub-steps to a
        macro-step that is `0.962^50 = 0.14`: an 86% loss of exactly the scales
        the wake lives at. A projection cannot be allowed to dissipate in a case
        study whose open defect is over-dissipation.

        So this uses the wide (`2h`) operator throughout, which is the choice
        `pressure.wide_eigenvalues_neumann` already documents for this codebase:
        the divergence that is *measured* on a collocated grid is the two-cell
        centred one, and inverting `DG` for that same `D` and `G` drives the
        measured divergence to machine zero with no interpolation anywhere. The
        checkerboard `phi` that the wide stencil cannot see is annihilated again
        by `G`, so the composition is an orthogonal projection of norm 1."""
        phi = self._poisson(self._ddx(u) + self._ddy(v))
        return u - self._ddx(phi), v - self._ddy(phi)

    def _poisson(self, rhs):
        """`D G phi = rhs`, homogeneous Neumann on all four faces.

        Singular with the constant as its null vector, so a solution exists only
        if `rhs` integrates to zero. The defect is the net boundary flux the
        Dirichlet data carries, which is small but not zero because the global
        field is divergence-free only to the projection's own tolerance. It is
        subtracted uniformly and **reported** on `last_mismatch` rather than
        absorbed silently -- the same contract `pressure.solve_neumann` keeps."""
        m = self.b.mean_keepdims(rhs, (1, 2))
        self.last_mismatch = self.b.absmax(m)
        r = rhs - m
        rh = self.b.dct2(r)
        ph = rh / self._lam_safe
        ph[:, 0, 0] = 0.0
        return self.b.idct2(ph)

    # -- the step ----------------------------------------------------------

    def step_batch(self, u, v, dt, bc0=None, bc1=None, force=None, open_faces=None):
        """Advance `[B, n, n]` windows by `dt` under a Dirichlet ring.

        `bc0` / `bc1` are `(u, v)` pairs whose **rings** supply the transmission
        data at the start and end of the macro-step; the ring is ramped linearly
        between them across the sub-steps. `bc0=None` takes the ring from the
        input itself, which is the one-pass case and makes the boundary data
        `t^n` data held constant -- exactly the explicit coupling the periodic
        path already had, minus the wrap. `bc1=None` holds `bc0`.

        The interior of `bc0`/`bc1` is ignored: only the ring is read. That is
        what separates the transmission condition from the initial condition and
        is the whole reason a Schwarz iteration on this operator is not a no-op.
        """
        u = self.b.array(u)
        v = self.b.array(v)
        if u.ndim != 3 or tuple(u.shape) != tuple(v.shape):
            raise ValueError(f"expected matching [B, n, n], got "
                             f"{tuple(u.shape)} {tuple(v.shape)}")
        r0u, r0v = ((u, v) if bc0 is None
                    else (self.b.asarray(bc0[0]), self.b.asarray(bc0[1])))
        r1u, r1v = ((r0u, r0v) if bc1 is None
                    else (self.b.asarray(bc1[0]), self.b.asarray(bc1[1])))

        umax = self.b.amax(self.b.hypot(u, v)) + 1e-12
        n_adv = int(np.ceil(dt / (self.cfl * self.h / umax)))
        n_vis = int(np.ceil(dt / (0.2 * self.h ** 2 / max(self.nu, 1e-12))))
        n_sub = max(1, n_adv, n_vis)
        self.last_substeps = n_sub
        hs = dt / n_sub
        # Through the backend, not straight in. A numpy `fx` added to a torch
        # `u` happens to work on CPU (numpy's interop protocol, with a
        # DeprecationWarning) and RAISES on CUDA -- so leaving it unconverted
        # would have passed every test on this machine and failed on the box,
        # which is the one failure mode this port exists to avoid.
        fx, fy = ((0.0, 0.0) if force is None
                  else (self.b.asarray(force[0]), self.b.asarray(force[1])))

        for m in range(n_sub):
            s0, s1 = m / n_sub, (m + 1) / n_sub
            ra_u, ra_v = (1 - s0) * r0u + s0 * r1u, (1 - s0) * r0v + s0 * r1v
            rb_u, rb_v = (1 - s1) * r0u + s1 * r1u, (1 - s1) * r0v + s1 * r1v
            if self.transmission == "convective":
                # advance the outflow rim in time before it is used as a stencil
                # value, so the boundary is integrated with the interior rather
                # than overwritten alongside it
                self._convect_outflow(u, hs, ra_u, ra_v, open_faces)
                self._convect_outflow(v, hs, ra_u, ra_v, open_faces)
            self._pin(u, ra_u, ra_u, ra_v, open_faces=open_faces)
            self._pin(v, ra_v, ra_u, ra_v, open_faces=open_faces)

            k1u, k1v = self._rhs(u, v, fx, fy)
            pu, pv = u + hs * k1u, v + hs * k1v
            self._pin(pu, rb_u, rb_u, rb_v, open_faces=open_faces)
            self._pin(pv, rb_v, rb_u, rb_v, open_faces=open_faces)
            k2u, k2v = self._rhs(pu, pv, fx, fy)
            u = u + 0.5 * hs * (k1u + k2u)
            v = v + 0.5 * hs * (k1v + k2v)

            # Hard boundary condition only -- the zero-gradient outflow value is
            # a *stencil* value for the stages above, and it must not reach the
            # projection. Feeding it in makes the window's net boundary flux
            # inconsistent, `_poisson` absorbs that defect as a uniform constant,
            # and a uniform constant in the right-hand side is a quadratic in
            # `phi` whose gradient is nonzero over the whole window: measured, the
            # projection *added* 4.8e-2 of interior divergence to a field that
            # arrived with 2.4e-14.
            self._pin(u, rb_u, rb_u, rb_v, inflow_only=True, open_faces=open_faces)
            self._pin(v, rb_v, rb_u, rb_v, inflow_only=True, open_faces=open_faces)
            if self.transmission == "convective":
                u, v = self._balance_flux_scaled(u, v, rb_u, rb_v, open_faces)
            elif self.transmission == "characteristic":
                u, v = self._balance_flux(u, v, rb_u, rb_v, open_faces)
            u, v = self._project(u, v)
            self._pin(u, rb_u, rb_u, rb_v, inflow_only=True, open_faces=open_faces)
            self._pin(v, rb_v, rb_u, rb_v, inflow_only=True, open_faces=open_faces)

        self.n_calls += int(u.shape[0])
        d = self._ddx(u) + self._ddy(v)
        self.last_div = self.b.absmax(d[:, 2:-2, 2:-2])
        # Back to numpy at the class boundary. Everything outside `WindowNS` --
        # the assembly, the ledger, the metrics, the tests -- is numpy and stays
        # numpy, so a GPU run changes one flag rather than the type of every
        # array in the case study. Three transfers per macro-step, ~20 ms.
        return self.b.to_numpy(u), self.b.to_numpy(v)


# --------------------------------------------------------------------------
# the same coupling, with the learned operator taken out
# --------------------------------------------------------------------------


@dataclass
class SolverExpert:
    """A drop-in replacement for `FrozenFluidExpert` backed by `SpectralNS`.

    **Why this exists.** W9 measured composition error at 0.273 against an
    expert error of 1.762 -- the frozen checkpoint contributes 6.5x more error
    than the decomposition does. That makes the coupling layer impossible to
    develop against the real expert: any change worth 10% of composition error
    is 1.5% of the total and is invisible under the checkpoint's own noise.

    So the expert is replaced by a solver that is exact to its own truncation,
    and everything else is held identical -- the same tiling, the same window
    size, the same per-window periodicity, the same Galilean framing, the same
    explicit forcing, the same assembly. Expert error goes to roughly zero and
    every number that remains belongs to the composition layer.

    This is the positive-control pattern the case study already uses three times
    over: R1 as the control for R2, the null control volumes for `r_T`, and the
    reference solver for W3's floor. It is not a new baseline, it is the same
    idea applied to the coupling itself.

    **What is deliberately NOT changed.** `SpectralNS` is periodic on its
    window, exactly as the checkpoint is. That is the point: if the per-window
    periodicity were also fixed here, a difference between this and the frozen
    run would confound "learned vs exact" with "periodic vs not", and neither
    could be attributed. The window's periodicity is part of the harness, not
    part of what is under test.
    """
    nu: float
    scaling: Scaling
    cfl: float = 0.4
    galilean: bool = True
    #: `'periodic'` is the original harness -- `SpectralNS`, wrapping, with the
    #: Galilean decomposition around it. `'dirichlet'` swaps in `WindowNS` and
    #: is OP-5's candidate (1): the local solve becomes a faithful restriction of
    #: the global problem, which is the precondition for a Schwarz iteration to
    #: converge to anything meaningful. Not available to the frozen checkpoint,
    #: and that asymmetry is a finding rather than a limitation of this class.
    window_bc: str = "periodic"
    #: Where the rotor body force enters. `'impulse'` adds `f*dt` after the step,
    #: which is the only thing a frozen one-shot operator can do and is therefore
    #: what the periodic path does. `'rhs'` integrates it inside the sub-stepping,
    #: which is what `ChannelNS` -- the target baseline -- does. Kept as a switch
    #: so that "non-periodic" and "properly-forced" can be varied one at a time
    #: instead of arriving together and being inseparable afterwards.
    force_mode: str = "impulse"
    #: Passed through to `WindowNS.transmission`. `'characteristic'` pins the
    #: ring only where flow enters; `'dirichlet'` pins all of it and is the
    #: control. Measured on a forced window: pure Dirichlet drives the wake back
    #: to freestream at the outflow ring (last interior cell 0.9864 against a
    #: 0.7970 minimum inside), so the deficit cannot leave the window at all.
    window_transmission: str = "characteristic"
    #: Array backend for the `WindowNS` path only -- `'numpy'` or `'torch'`,
    #: with `device` selecting CPU or CUDA. `SpectralNS` (the periodic path) is
    #: untouched: it is FFT-bound rather than stencil-bound, so it is not where
    #: the time goes, and it is the harness for the frozen-checkpoint runs whose
    #: published numbers must not move.
    backend: str = "numpy"
    device: str = "cpu"

    def __post_init__(self):
        if self.window_bc not in ("periodic", "dirichlet"):
            raise ValueError(self.window_bc)
        if self.force_mode not in ("impulse", "rhs"):
            raise ValueError(self.force_mode)
        if self.backend != "numpy" and self.window_bc != "dirichlet":
            raise ValueError(
                f"backend={self.backend!r} only affects the WindowNS path, which "
                f"needs window_bc='dirichlet' (got {self.window_bc!r}). Running "
                f"periodic windows on a GPU would silently do nothing.")
        self._solver = SpectralNS(nu=self.nu, length=self.scaling.length,
                                  n=EXPERT_RES, cfl=self.cfl)
        self._window = WindowNS(nu=self.nu, length=self.scaling.length,
                                n=EXPERT_RES, cfl=self.cfl,
                                transmission=self.window_transmission,
                                backend=self.backend, device=self.device)
        self.n_calls = 0
        self.last_mean_drift = (0.0, 0.0)

    def step_many(self, u, v, dt: float, project: bool = False,
                  galilean: bool | None = None, force=None, chunk: int = 32,
                  frame: tuple | None = None, bc=None, open_faces=None):
        """Mirror of `FrozenFluidExpert.step_many`, term for term.

        The sequence is copied rather than paraphrased -- subtract the frame,
        advance the fluctuation, re-zero its mean and record the drift,
        translate by the frame, add the frame back, then apply the explicit
        impulse. A harness that framed or forced differently from the operator
        it is standing in for would not be a control.

        `chunk` is accepted and ignored: batching is a property of the neural
        checkpoint's threading, not of a solver.

        `bc` is the transmission channel, and it is `None` for every caller that
        is not running a Schwarz iteration. When given it is a `(u, v)` pair
        whose **ring** is the neighbours' current estimate of the field at
        `t + dt`; the ring is ramped linearly from the input's own ring across
        the sub-steps. Only meaningful under `window_bc='dirichlet'` -- a
        periodic window has no boundary to impose it on, and passing one is
        refused rather than ignored."""
        from .adapters import spectral_shift
        gal = self.galilean if galilean is None else galilean
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        if u.ndim != 3 or u.shape[1:] != (EXPERT_RES, EXPERT_RES):
            raise ValueError(f"expected [B, {EXPERT_RES}, {EXPERT_RES}], got {u.shape}")
        B, L = u.shape[0], self.scaling.length

        if self.window_bc == "dirichlet":
            # No Galilean decomposition and no spectral shift. Both exist solely
            # to work around a periodic operator's inability to hold a mean --
            # the frozen checkpoint takes a uniform flow from 1 to 0.969 in one
            # call, and `SpectralNS` carries the mean analytically because a
            # periodic box cannot carry it any other way. A Dirichlet window has
            # no such problem: a uniform flow is an exact steady solution and the
            # inflow ring supplies it, so the mean is preserved by the boundary
            # condition itself. Worse than unnecessary, `spectral_shift` is a
            # *periodic* translation and would reintroduce the wrap this mode
            # exists to remove.
            f = None
            if force is not None and self.force_mode == "rhs":
                f = (np.asarray(force[0], dtype=np.float64),
                     np.asarray(force[1], dtype=np.float64))
            u1, v1 = self._window.step_batch(u, v, dt, bc0=None,
                                             bc1=None if bc is None else bc, force=f,
                                             open_faces=open_faces)
            if force is not None and self.force_mode == "impulse":
                u1 = u1 + np.asarray(force[0], dtype=np.float64) * dt
                v1 = v1 + np.asarray(force[1], dtype=np.float64) * dt
            self.n_calls += B
            self.last_mean_drift = (0.0, 0.0)
            if project:
                from .adapters import leray_project
                for b in range(B):
                    u1[b], v1[b] = leray_project(u1[b], v1[b], L)
            return u1, v1

        if open_faces is not None:
            raise ValueError(
                "open_faces= was supplied under window_bc='periodic'. A periodic "
                "window has no faces to declare open or closed, so the argument "
                "could only be silently discarded.")
        if bc is not None:
            raise ValueError(
                "bc= was supplied under window_bc='periodic'. A periodic window has "
                "no boundary to impose a transmission condition on, so the argument "
                "could only be silently discarded -- and a Schwarz loop whose "
                "transmission data is discarded still converges, reports a falling "
                "interface residual, and means nothing. Refused here rather than "
                "left to be discovered from a suspiciously clean sweep.")

        if not gal:
            ubar = vbar = np.zeros(B)
        elif frame is not None:
            ubar = np.full(B, float(frame[0]))
            vbar = np.full(B, float(frame[1]))
        else:
            ubar = u.mean(axis=(1, 2))
            vbar = v.mean(axis=(1, 2))
        uf = u - ubar[:, None, None]
        vf = v - vbar[:, None, None]

        u1 = np.empty_like(uf)
        v1 = np.empty_like(vf)
        for b in range(B):
            # the fluctuation has zero mean, so the solver's own Galilean
            # carriage is the identity and it simply integrates -- which is what
            # keeps this equivalent to the checkpoint path rather than merely
            # similar to it
            u1[b], v1[b] = self._solver.step(uf[b], vf[b], dt)
        self.n_calls += B

        if gal:
            drift_u = u1.mean(axis=(1, 2))
            drift_v = v1.mean(axis=(1, 2))
            self.last_mean_drift = (float(np.abs(drift_u).max()),
                                    float(np.abs(drift_v).max()))
            u1 -= drift_u[:, None, None]
            v1 -= drift_v[:, None, None]
            for b in range(B):
                u1[b] = spectral_shift(u1[b], ubar[b] * dt, vbar[b] * dt, L)
                v1[b] = spectral_shift(v1[b], ubar[b] * dt, vbar[b] * dt, L)
            u1 += ubar[:, None, None]
            v1 += vbar[:, None, None]
            if force is not None:
                u1 += np.asarray(force[0], dtype=np.float64) * dt
                v1 += np.asarray(force[1], dtype=np.float64) * dt

        if project:
            from .adapters import leray_project
            for b in range(B):
                u1[b], v1[b] = leray_project(u1[b], v1[b], L)
        return u1, v1

    def step(self, u, v, dt, project=False, galilean=None, force=None, bc=None):
        U, V = self.step_many(u[None], v[None], dt, project, galilean,
                              None if force is None else (force[0][None], force[1][None]),
                              bc=None if bc is None else (bc[0][None], bc[1][None]))
        return U[0], V[0]


__all__ = ["SpectralNS", "ChannelNS", "WindowNS", "make_forcing", "SolverExpert"]
