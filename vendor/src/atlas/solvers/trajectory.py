"""3-DOF planar rigid body -- data generator AND runtime expert.

This module is written once and reused **verbatim** by Atlas's non-learned
`rigid_body` expert (impl-atlas-0.1-phase0-scope-and-data §3.3): it is not a
data-generation helper that gets replaced later. That is why `rhs` and
`rk4_step` are pure functions of state and applied loads, with no dependency on
any solver internals.

State (world frame): (x, y, theta, vx, vy, omega, m).

> On the integrator: a symplectic scheme is the natural default for a
> conservative system, and this system is not one -- mass is expelled, drag
> dissipates, thrust does work. RK4 is correct here; do not "upgrade" it to
> leapfrog.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atmosphere import gravity

STATE_FIELDS = ("x", "y", "theta", "vx", "vy", "omega", "m")


@dataclass(frozen=True)
class Loads:
    """Applied loads for one step, in the WORLD frame.

    F_thrust comes from the e-f interface integral, F_aero from the c-d surface
    integral of (-p n + tau.n). `mdot` is the propellant expenditure rate (a
    positive number; mass decreases)."""
    F_thrust: tuple[float, float] = (0.0, 0.0)
    F_aero: tuple[float, float] = (0.0, 0.0)
    torque: float = 0.0
    mdot: float = 0.0
    inertia: float = 1.0e4


def rhs(s: np.ndarray, loads: Loads) -> np.ndarray:
    """ds/dt for state `s` = (x, y, theta, vx, vy, omega, m)."""
    x, y, theta, vx, vy, omega, m = s
    m = max(m, 1e-6)
    g = gravity(y)
    fx = loads.F_thrust[0] + loads.F_aero[0]
    fy = loads.F_thrust[1] + loads.F_aero[1] - m * g
    return np.array([vx, vy, omega, fx / m, fy / m,
                     loads.torque / max(loads.inertia, 1e-9), -loads.mdot])


def rk4_step(s: np.ndarray, dt: float, loads: Loads) -> np.ndarray:
    """Classical RK4. Loads are held constant across the step -- they come from
    the coupled solvers at the macro-step cadence, which is exactly the loose
    (Gauss-Seidel) coupling the surrogate will imitate."""
    s = np.asarray(s, dtype=float)
    k1 = rhs(s, loads)
    k2 = rhs(s + 0.5 * dt * k1, loads)
    k3 = rhs(s + 0.5 * dt * k2, loads)
    k4 = rhs(s + dt * k3, loads)
    out = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    out[6] = max(out[6], 1e-6)
    return out


def initial_state(h0: float = 0.0, v0: float = 0.0, m0: float = 5.0e4,
                  theta0: float = np.pi / 2) -> np.ndarray:
    return np.array([0.0, h0, theta0, 0.0, v0, 0.0, m0])


def dynamic_pressure(state: np.ndarray, rho: float) -> float:
    """q = 0.5 rho V^2 -- the quantity max-Q is the maximum of."""
    return 0.5 * rho * (state[3] ** 2 + state[4] ** 2)


__all__ = ["Loads", "rhs", "rk4_step", "initial_state", "dynamic_pressure", "STATE_FIELDS"]
