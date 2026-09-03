"""The actuator-disk expert. Zero parameters, closed form, exact.

Phase W2. This is the only component of the case study with an answer that can
be checked against theory rather than against another model, which is why the
guide builds it before anything is coupled: it is the ruler everything else is
measured with.

Everything is nondimensional -- `D = U_inf = rho = 1`, so the swept width `A`
is 1 and the strip thickness is `Delta_d = 0.1`.

Why the local-induction form
----------------------------
Classical actuator-disk theory references the freestream:

    U_d = U_inf (1 - a),   C_T = 4a(1-a),   C_P = 4a(1-a)^2,   T = 1/2 rho A U_inf^2 C_T

which is fine for an isolated turbine and **unusable for turbine 2**. `U_inf` is
not defined inside a wake, and using the domain inlet value would make turbine
2's thrust independent of the wake it stands in -- silently deleting the
coupling this entire case study exists to test. So the disk uses the
local-induction (Calaf / Meyers) form standard in actuator-disk CFD:

    C_T' = C_T / (1-a)^2 = 4a/(1-a),   T = 1/2 rho A C_T' <U_d>^2,   P = T <U_d>

where `<U_d>` is the disk-averaged streamwise velocity the *fluid* expert
reports on the rotor's upstream face. Every turbine now reads its own inflow.
At `a = 1/3`, `C_T' = 2`.

The Betz bound is not enforced, it is inherited
-----------------------------------------------
`C_P = 4a(1-a)^2` cannot exceed 16/27 for any `a` -- that is a property of the
formula. The disk never outputs `C_P` directly, only `a`. So a measured `C_P`
above 16/27 anywhere in the coupled system is not a violated constraint, it is
proof of a sign error or double-counted momentum somewhere else. Canary, not
cage (spec section 7.5).

Validity
--------
Momentum theory breaks down around `a ~ 0.4`, where the wake enters the
turbulent-wake state and an empirical `C_T` correction is required. The spec
stays at or below 0.35 and treats anything past 0.4 as out of scope: values in
`(0.35, 0.4]` are clamped with a recorded flag, and `a > 0.4` raises. Silently
extrapolating a formula past the physics it describes is how a plot ends up
looking reasonable and being wrong.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A_CLAMP = 0.35          # spec section 4.2: the operating limit
A_INVALID = 0.40        # past this, momentum theory does not describe the flow
BETZ = 16.0 / 27.0

#: Declared, not learned, and not measured: the disk has no rotor-speed model,
#: but the `ROT` port has to carry an (effort, flow) pair whose product is the
#: shaft power. A design tip-speed ratio fixes the split. Nothing downstream
#: depends on it -- the port is unconnected, so only the product `P = tau omega`
#: enters `P_ext` -- and it is exposed so that connecting a generator at rung 7
#: has something to attach to.
TIP_SPEED_RATIO = 7.5


def clamp_induction(a: float) -> tuple[float, bool]:
    """Return `(a_used, was_clamped)`; raise past the validity limit."""
    a = float(a)
    if not (a > 0.0):
        raise ValueError(f"axial induction must be positive, got {a}")
    if a > A_INVALID:
        raise ValueError(
            f"a = {a} exceeds {A_INVALID}: momentum theory breaks down in the turbulent-wake "
            "state and an empirical C_T correction would be required. Out of scope for this "
            "case study (spec section 4.2) -- not extrapolated."
        )
    return (A_CLAMP, True) if a > A_CLAMP else (a, False)


# -- closed-form coefficients ---------------------------------------------


def c_t(a: float) -> float:
    """Freestream-referenced thrust coefficient, `4a(1-a)`."""
    return 4.0 * a * (1.0 - a)


def c_t_prime(a: float) -> float:
    """Local-induction thrust coefficient, `4a/(1-a)` -- equals `c_t/(1-a)^2`."""
    return 4.0 * a / (1.0 - a)


def c_p(a: float) -> float:
    """Freestream-referenced power coefficient, `4a(1-a)^2`. Maximal at `a=1/3`,
    where it equals exactly 16/27."""
    return 4.0 * a * (1.0 - a) ** 2


def u_disk_from_freestream(a: float, u_inf: float = 1.0) -> float:
    return u_inf * (1.0 - a)


def u_wake_from_freestream(a: float, u_inf: float = 1.0) -> float:
    return u_inf * (1.0 - 2.0 * a)


# -- the expert ------------------------------------------------------------


@dataclass(frozen=True)
class DiskState:
    """What one evaluation of the disk expert produces."""
    a: float
    a_requested: float
    clamped: bool
    c_t_prime: float
    u_disk: float
    thrust: float
    power: float
    torque: float
    omega: float

    def c_p_measured(self, u_ref: float) -> float:
        """`P / (1/2 rho A u_ref^3)`. `u_ref` is the *undisturbed* inflow for
        this turbine -- for turbine 2 that is the local wake velocity, not the
        domain inlet, and the choice must be stated wherever the number is
        quoted (spec section 7.1, C4)."""
        return self.power / (0.5 * u_ref ** 3)


@dataclass(frozen=True)
class ActuatorDisk:
    """A rotor, as a permeable surface that removes streamwise momentum.

    Parameter count: zero. `a`, `area` and `thickness` are declared geometry and
    a control setting, not fitted quantities."""
    a: float = 1.0 / 3.0
    area: float = 1.0                # swept width, = D = 1
    thickness: float = 0.1           # Delta_d
    tip_speed_ratio: float = TIP_SPEED_RATIO
    rho: float = 1.0

    def __post_init__(self):
        clamp_induction(self.a)      # fail at construction, not mid-rollout

    def __call__(self, u_disk: float) -> DiskState:
        """Evaluate at a disk-averaged upstream velocity. Algebraic: no state,
        no timestep, no history (spec section 8.2)."""
        a, clamped = clamp_induction(self.a)
        ctp = c_t_prime(a)
        ud = float(u_disk)
        thrust = 0.5 * self.rho * self.area * ctp * ud ** 2
        power = thrust * ud
        radius = 0.5 * self.area
        omega = self.tip_speed_ratio * ud / radius
        torque = power / omega if omega != 0.0 else 0.0
        return DiskState(a=a, a_requested=self.a, clamped=clamped, c_t_prime=ctp,
                         u_disk=ud, thrust=thrust, power=power,
                         torque=torque, omega=omega)

    # -- the body force ----------------------------------------------------

    def force_density(self, thrust: float) -> float:
        """Magnitude of the streamwise momentum sink per unit area inside the
        strip. The force itself is `-force_density * xhat`."""
        return thrust / (self.area * self.thickness)

    def body_force_field(self, x_c: np.ndarray, y_c: np.ndarray, dx: float, dy: float,
                         thrust: float, x0: float, y0: float | None = None) -> np.ndarray:
        """`f_x` on a cell-centred lattice, as an *exact* discrete sink.

        The strip is `[x0, x0+thickness] x [y0, y0+area]` and rarely lines up
        with cell edges, so each cell gets the force density weighted by the
        fraction of its area inside the strip. Computed from exact rectangle
        overlap rather than by sampling, which makes

            sum(f_x * cell_area) == -thrust

        true to floating point on any lattice -- and gate W2 checks exactly
        that, because a discretization that loses 3% of the thrust would look
        like a 3% interface residual in W3 and be blamed on the fluid expert.

        Returned shaped `[ny, nx]`: **axis 0 is y and axis 1 is x**, the layout
        W0 experiment E2 measured for the expert's own velocity channels. The
        first version of this function returned `[nx, ny]` and the transposition
        went unnoticed through the whole of gate W2, because a disk centred in a
        square window looks much the same either way."""
        if y0 is None:
            y0 = -0.5 * self.area
        x1, y1 = x0 + self.thickness, y0 + self.area
        xl, xr = x_c[None, :] - dx / 2, x_c[None, :] + dx / 2
        yl, yr = y_c[:, None] - dy / 2, y_c[:, None] + dy / 2
        ox = np.clip(np.minimum(xr, x1) - np.maximum(xl, x0), 0.0, None)
        oy = np.clip(np.minimum(yr, y1) - np.maximum(yl, y0), 0.0, None)
        frac = (ox * oy) / (dx * dy)
        return -self.force_density(thrust) * frac


def disk_average(u: np.ndarray, x_c: np.ndarray, y_c: np.ndarray, x_face: float,
                 half_span: float = 0.5) -> float:
    """`<U_d>`: the streamwise velocity averaged over the rotor's upstream face.

    Reads the column of cells whose centres are nearest the face and averages
    over `|y| <= half_span`. Deliberately *upstream* of the strip: sampling
    inside it would read a velocity the disk's own body force has already
    slowed, which is the classic actuator-disk double-counting error and would
    make the fixed-point iteration of W5 converge to the wrong thrust."""
    i = int(np.argmin(np.abs(x_c - x_face)))
    m = np.abs(y_c) <= half_span + 1e-12
    if not m.any():
        raise ValueError(f"no lattice rows within |y| <= {half_span}")
    return float(np.mean(u[m, i]))          # [y, x]


__all__ = [
    "A_CLAMP", "A_INVALID", "BETZ", "TIP_SPEED_RATIO", "clamp_induction",
    "c_t", "c_t_prime", "c_p", "u_disk_from_freestream", "u_wake_from_freestream",
    "DiskState", "ActuatorDisk", "disk_average",
]
