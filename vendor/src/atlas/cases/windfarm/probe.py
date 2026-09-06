"""Phase W3 -- the single-interface probe. The go/no-go for the case study.

`impl-wind-farm-guide` section 4. One turbine, two agents, no graph, no message
passing: hand the disk the fluid's disk-averaged inflow, hand the fluid expert
the resulting body force, and ask whether the momentum deficit the expert
produces across the strip equals the thrust the disk computed.

    r_T = | T_disk - |Phi_x(V_1)| | / T_disk  <  5%,  before any projection.

Five things have to be pinned down before that number means anything.

1. The expert has no forcing input
----------------------------------
Poseidon takes a velocity field and returns a velocity field. There is no body-
force channel -- forced Navier-Stokes appears in the Poseidon paper only as a
*downstream finetuning* task, and this case study does not finetune. So the
force enters by operator splitting, exactly as it would with any black-box
propagator: Strang-split as half an impulse, one expert step, half an impulse,
with a Leray projection after each impulse to absorb the pressure. Over the
whole periodic window the applied impulse is preserved exactly, because the
projection touches every Fourier mode except `k = 0` and the total force *is*
the `k = 0` mode. Over a sub-slab it is not preserved, and it should not be:
that redistribution is the pressure field, and it appears explicitly in the
control-volume balance below.

2. There is no pressure to read
-------------------------------
The incompressible datasets fill the density and pressure channels with the
constants 1 and 0. The checkpoint therefore predicts no pressure at all, and the
momentum balance needs `p` on the control-volume faces. Pressure is not an
independent field in incompressible flow, though -- it is a functional of the
velocity, recovered from `laplacian p = div f - div((u.grad)u)`, spectrally and
exactly on the periodic window (verified against the analytic Taylor-Green
pressure to 2e-15). That is a derivation, not a model.

3. The control volume is a full-height slab
-------------------------------------------
Taking the control volume across the whole (periodic) window height makes its
lateral faces coincide, so they cancel identically and the balance is between
two vertical faces plus the unsteady term. A volume hugging `|y| <= 0.5` instead
adds two lateral faces carrying most of the flux, whose discretization error
would land squarely on `r_T` and be indistinguishable from the expert being
wrong.

Both forms of the balance are reported:

    steady    Phi_x(V) = -T                       (the spec's form)
    unsteady  d/dt integral_V u_x dV + Phi_x = -T (true at every instant)

The steady form is only meaningful once the near field has settled, and it is
the one the gate is written against.

4. Teacher forcing is an exchange, not a frozen thrust
------------------------------------------------------
Each side is handed the other's value once per step with no fixed-point
iteration: the disk reads `<U_d>` off the fluid, the fluid takes `f` from the
disk, explicitly.

Freezing `T` at its freestream value instead is a different problem, and the
reference solver says so. With an inflow band it settles (`<U_d>` = 0.94 under
R1, 0.82 under R2), but with no band -- the wake recirculating through a
periodic window -- it runs away and reverses the flow entirely (`<U_d>` = -0.42
at t = 20): a fixed force plus a residence time that grows as the strip slows is
a positive feedback, and only fresh inflow breaks it.

The reason not to freeze `T` is the physics rather than the stability, though.
A frozen thrust is independent of the local inflow, which is exactly the failure
mode section 4.2 of the spec forbids for turbine 2 -- it deletes the coupling the
case study exists to test. `T = 1/2 C_T' <U_d>^2` is what makes a turbine read
the wake it stands in.

5. R1 and R2 are band widths
----------------------------
The guide's two interface variants have to be realized against an operator that
accepts no boundary conditions, so the neighbour's data is written into the
state itself, in a band along the upstream edge:

* **R2, flux-BC tokens** -- a band one *expert patch* wide (4 cells, 0.0625 D).
  The declared interface carries the port values and nothing else. This is the
  out-of-distribution path and the real test.
* **R1, Schwarz halo** -- a wide band (0.5 D by default) filled from the
  neighbour's interior, so the expert sees a locally complete stencil that looks
  like the interior of a full domain. The positive control.

`band_width` is worth sweeping rather than fixing at two values, because "how
much halo is required" is a more useful answer than "does 0.0625 D work", and
the guide says an R1-only pass changes Mechanism A of
`edge-generation-atlas-0.1`.

Array layout: **`[y, x]`** everywhere, matching what W0 experiment E2 measured
for the expert's own velocity channels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adapters import EXPERT_RES, Scaling, divergence, leray_project
from .disk import ActuatorDisk, disk_average

#: The probe window: 2 D square, so the 1 D rotor spans the middle half and
#: there is a 0.5 D bypass corridor on each side. One rotor diameter per window
#: (the W0 scaling) would put the rotor across the full height, leaving the flow
#: nowhere to go around -- a blockage problem, not a wake problem. 50% blockage
#: is still high against the spec's 8 D-tall domain, and it is a limitation of
#: the probe, not of the case study; `r_T` is an identity and holds at any
#: blockage, which is why the probe can afford it.
PROBE_SCALING = Scaling(length=2.0, velocity=4.0)

X_STRIP = 0.0            # the rotor's upstream face, at the window centre in x
PATCH_CELLS = 4          # scOT patch_size: the width of exactly one boundary token


@dataclass(frozen=True)
class Window:
    """A 128x128 periodic window in case-study coordinates."""
    x0: float
    y0: float
    length: float = PROBE_SCALING.length
    n: int = EXPERT_RES

    @property
    def dx(self) -> float:
        return self.length / self.n

    @property
    def x_c(self) -> np.ndarray:
        return self.x0 + (np.arange(self.n) + 0.5) * self.dx

    @property
    def y_c(self) -> np.ndarray:
        return self.y0 + (np.arange(self.n) + 0.5) * self.dx

    def index_of(self, x: float) -> int:
        return int(np.argmin(np.abs(self.x_c - x)))


#: Centred on the rotor: x in [-1, 1], y in [-1, 1] with the strip at x in [0, 0.1].
PROBE_WINDOW = Window(x0=-1.0, y0=-1.0)


# --------------------------------------------------------------------------
# pressure and the control-volume balance
# --------------------------------------------------------------------------


def _k(n: int, length: float):
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    if n % 2 == 0:
        k[n // 2] = 0.0
    return k


def pressure_from_velocity(u, v, fx=None, fy=None, length: float = 1.0) -> np.ndarray:
    """Solve `laplacian p = div f - div((u.grad)u)` spectrally.

    Returns `p` with zero mean. The constant is undetermined and irrelevant: a
    closed control volume has `integral n_x ds = 0`, so a uniform shift in `p`
    contributes exactly nothing to the balance."""
    n = u.shape[-1]
    kx, ky = _k(n, length)[None, :], _k(n, length)[:, None]
    k2 = kx * kx + ky * ky
    k2 = np.where(k2 == 0.0, 1.0, k2)

    def ddx(f):
        return np.real(np.fft.ifft2(1j * kx * np.fft.fft2(f)))

    def ddy(f):
        return np.real(np.fft.ifft2(1j * ky * np.fft.fft2(f)))

    nlx = u * ddx(u) + v * ddy(u)
    nly = u * ddx(v) + v * ddy(v)
    rhs = -(ddx(nlx) + ddy(nly))
    if fx is not None:
        rhs = rhs + ddx(fx) + (ddy(fy) if fy is not None else 0.0)
    rh = np.fft.fft2(rhs)
    rh[0, 0] = 0.0
    return np.real(np.fft.ifft2(-rh / k2))


def _ddx(f, length):
    n = f.shape[-1]
    kx = _k(n, length)[None, :]
    return np.real(np.fft.ifft2(1j * kx * np.fft.fft2(f)))


@dataclass
class SlabBalance:
    """Streamwise momentum budget of a full-height slab `x in [xl, xr]`."""
    flux_convective: float
    flux_pressure: float
    flux_viscous: float
    d_momentum_dt: float
    forcing: float

    @property
    def phi_x(self) -> float:
        """Net streamwise momentum outflux, the spec's `Phi_x(V)`."""
        return self.flux_convective + self.flux_pressure + self.flux_viscous

    @property
    def residual_steady(self) -> float:
        """`Phi_x - integral f`, which the spec's steady form says is zero."""
        return self.phi_x - self.forcing

    @property
    def residual_unsteady(self) -> float:
        """`dM/dt + Phi_x - integral f`, zero at every instant, settled or not."""
        return self.d_momentum_dt + self.phi_x - self.forcing


def slab_balance(u0, v0, u1, v1, window: Window, xl: float, xr: float, dt: float,
                 nu: float, fx: np.ndarray | None = None) -> SlabBalance:
    """Momentum budget between two states a macro-step apart.

    Fluxes are evaluated at the midpoint in time (the average of the two states)
    so that they are second-order consistent with the centred `dM/dt`. The
    pressure is recovered from the midpoint velocity."""
    L = window.length
    dxc = window.dx
    il, ir = window.index_of(xl), window.index_of(xr)
    um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    p = pressure_from_velocity(um, vm, fx, None, L)
    dudx = _ddx(um, L)

    # arrays are [y, x], so a slab face is a COLUMN and `dxc` doubles as the
    # quadrature weight dy along it
    conv = dxc * (float(np.sum(um[:, ir] ** 2)) - float(np.sum(um[:, il] ** 2)))
    pres = dxc * (float(np.sum(p[:, ir])) - float(np.sum(p[:, il])))
    visc = -nu * dxc * (float(np.sum(dudx[:, ir])) - float(np.sum(dudx[:, il])))

    cell = dxc * dxc
    sl = slice(il, ir)
    m0 = cell * float(np.sum(u0[:, sl]))
    m1 = cell * float(np.sum(u1[:, sl]))
    forcing = cell * float(np.sum(fx[:, sl])) if fx is not None else 0.0
    return SlabBalance(conv, pres, visc, (m1 - m0) / dt, forcing)


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------


@dataclass
class ProbeConfig:
    variant: str = "R2"                 # 'R1' | 'R2' | 'none'
    band_width: float | None = None     # in D; defaults per variant
    a: float = 1.0 / 3.0
    u_inf: float = 1.0
    freeze_thrust: bool = False         # the diagnostic of docstring point 4
    u_disk_frozen: float | None = None
    dt: float = 0.05
    n_steps: int = 40
    n_settle: int = 0                   # steps run before r_T is recorded
    cv_upstream: float = 0.25           # slab face, D upstream of the strip
    cv_downstream: float = 0.25         # D downstream of the strip's back face
    #: W0/E3 measured the checkpoint's dissipation at the cutoff scale as
    #: nu = 4.9e-4 in EXPERT units. The balance works in case-study units, where
    #: nu_ours = nu_expert * L * U_s -- converted in `run_probe` from the scaling
    #: actually in use, never hard-coded, because the two differ by a factor of 8
    #: at the probe scaling and a viscous term off by 8x would be charged to the
    #: expert.
    nu_expert: float = 4.9e-4
    strang: bool = True
    #: Add a uniform `+T/Area` body force so the *total* applied force has zero
    #: mean. This is not cosmetic. The pressure is recovered from a periodic
    #: spectral solve, and a periodic pressure cannot represent the mean
    #: streamwise gradient that a net body force requires -- so with the disk
    #: force alone the momentum balance is short by exactly `<f_x> * V_slab`.
    #: The reference solver showed this as a rock-steady 30.4% residual at a
    #: converged state with `dM/dt = 1e-12`: not a transient, not the expert, a
    #: missing term. Driving a periodic domain with a uniform force balancing the
    #: turbines' drag is the standard construction for periodic wind-farm LES
    #: (Calaf, Meyers & Meneveau 2010), and it makes the problem well posed
    #: rather than papering over it.
    counter_force: bool = True

    def resolved_band(self) -> float:
        if self.band_width is not None:
            return self.band_width
        if self.variant == "R1":
            return 0.5
        if self.variant == "R2":
            return PATCH_CELLS * PROBE_WINDOW.dx      # exactly one expert patch
        return 0.0


def _apply_impulse(u, v, fx, dt, length):
    """u <- Leray( u + dt f ). The projection is Chorin's: it absorbs the
    pressure that keeps the forced field divergence free. It is NOT the thrust
    projection of guide section 6.3 -- nothing here is corrected toward the
    disk's answer, which is the whole point of the measurement."""
    return leray_project(u + dt * fx, v, length)


def run_probe(expert, cfg: ProbeConfig, window: Window = PROBE_WINDOW,
              u_init=None, v_init=None, teacher=None) -> dict:
    """One probe run. Returns the measured `r_T` and every part of the budget.

    `expert` is anything with `.step(u, v, dt)` and `.scaling` -- the frozen
    checkpoint or the reference solver, so both are driven by identical code and
    a difference between them is a difference between them.

    `teacher` is the neighbouring agent's field as `(u, v)` on the same window,
    from which the upstream band is filled. Passing the reference solver's
    converged state is teacher forcing in the guide's sense; the default,
    uniform inflow, is the cruder version and carries the disk's own upstream
    induction as an error at the band."""
    n = window.n
    dxc = window.dx
    L = window.length
    s = expert.scaling
    nu = cfg.nu_expert * s.length * s.velocity          # -> case-study units

    disk = ActuatorDisk(a=cfg.a)
    x_probe = X_STRIP - dxc                             # upstream face, one cell clear

    u = np.full((n, n), cfg.u_inf) if u_init is None else np.array(u_init, dtype=float)
    v = np.zeros((n, n)) if v_init is None else np.array(v_init, dtype=float)

    band = cfg.resolved_band()
    n_band = int(round(band / dxc))
    if teacher is None:
        tu, tv = np.full((n, n), cfg.u_inf), np.zeros((n, n))
    else:
        tu = np.asarray(teacher[0], dtype=float)
        tv = np.asarray(teacher[1], dtype=float)

    def impose(u, v):
        """Write the neighbour's data into the upstream band -- columns, since
        arrays are [y, x]. The band is also what breaks periodicity: it erases
        the wake that would otherwise wrap around and re-enter as inflow."""
        if not n_band:
            return u, v
        u, v = u.copy(), v.copy()
        u[:, :n_band] = tu[:, :n_band]
        v[:, :n_band] = tv[:, :n_band]
        return u, v

    hist = []
    for i in range(cfg.n_steps):
        u, v = impose(u, v)
        if cfg.freeze_thrust:
            u_d = cfg.u_disk_frozen if cfg.u_disk_frozen is not None \
                else cfg.u_inf * (1.0 - cfg.a)
        else:
            u_d = disk_average(u, window.x_c, window.y_c, x_probe)
        state = disk(u_d)
        fx = disk.body_force_field(window.x_c, window.y_c, dxc, dxc, state.thrust, x0=X_STRIP)
        f_disk_total = float((fx * dxc * dxc).sum())
        if cfg.counter_force:
            fx = fx + state.thrust / (L * L)

        u0, v0 = u.copy(), v.copy()
        if cfg.strang:
            u, v = _apply_impulse(u, v, fx, 0.5 * cfg.dt, L)
            u, v = expert.step(u, v, cfg.dt)
            u, v = _apply_impulse(u, v, fx, 0.5 * cfg.dt, L)
        else:
            u, v = _apply_impulse(u, v, fx, cfg.dt, L)
            u, v = expert.step(u, v, cfg.dt)

        bal = slab_balance(u0, v0, u, v, window,
                           X_STRIP - cfg.cv_upstream,
                           X_STRIP + disk.thickness + cfg.cv_downstream,
                           cfg.dt, nu, fx)
        t_ = max(state.thrust, 1e-30)
        hist.append({
            "step": i + 1, "t": (i + 1) * cfg.dt,
            "u_disk": u_d, "thrust": state.thrust, "power": state.power,
            "phi_x": bal.phi_x, "d_momentum_dt": bal.d_momentum_dt,
            "flux_convective": bal.flux_convective,
            "flux_pressure": bal.flux_pressure,
            "flux_viscous": bal.flux_viscous,
            "forcing_in_slab": bal.forcing,
            "applied_impulse_error": abs(f_disk_total + state.thrust) / t_,
            "net_force_on_window": float((fx * dxc * dxc).sum()),
            "r_T_steady": abs(bal.residual_steady) / t_,
            "r_T_unsteady": abs(bal.residual_unsteady) / t_,
            "u_min": float(u.min()), "u_max": float(u.max()),
            "div_linf": float(np.max(np.abs(divergence(u, v, L)))),
            "finite": bool(np.all(np.isfinite(u)) and np.all(np.isfinite(v))),
        })
        if not hist[-1]["finite"]:
            break

    kept = [h for h in hist if h["step"] > cfg.n_settle] or hist
    tail = kept[max(0, len(kept) - 5):]
    return {
        "config": {**vars(cfg), "band_resolved": band, "n_band_cells": n_band},
        "window": {"x0": window.x0, "y0": window.y0, "length": L, "n": n, "dx": dxc},
        "nu_case_units": nu,
        "re_eff": 1.0 / nu,
        "history": hist,
        "r_T_steady_final": kept[-1]["r_T_steady"],
        "r_T_unsteady_final": kept[-1]["r_T_unsteady"],
        "r_T_steady_tail_mean": float(np.mean([h["r_T_steady"] for h in tail])),
        "r_T_unsteady_tail_mean": float(np.mean([h["r_T_unsteady"] for h in tail])),
        "u_disk_final": kept[-1]["u_disk"],
        "thrust_final": kept[-1]["thrust"],
        "u_disk_drift": abs(kept[-1]["u_disk"] - kept[max(0, len(kept) - 6)]["u_disk"]),
        "all_finite": all(h["finite"] for h in hist),
        "u": u, "v": v,
    }


__all__ = [
    "PROBE_SCALING", "PROBE_WINDOW", "X_STRIP", "PATCH_CELLS",
    "Window", "ProbeConfig", "SlabBalance",
    "pressure_from_velocity", "slab_balance", "run_probe",
]
