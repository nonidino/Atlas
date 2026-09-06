"""US Standard Atmosphere 1976, 0-32 km, plus Sutherland viscosity.

Three layers (troposphere, tropopause, lower stratosphere), exactly as
impl-atlas-0.1-phase0-scope-and-data §2.4. Vectorized over altitude, because the
trajectory module evaluates it every RK4 stage.
"""
from __future__ import annotations

import numpy as np

G0 = 9.80665
R_AIR = 287.053
GAMMA_AIR = 1.4
T0, P0 = 288.15, 101325.0
L_TROPO = 0.0065          # K/m, temperature LAPSE (T decreases with h)
H_TROPO, H_TROPOPAUSE, H_TOP = 11000.0, 20000.0, 32000.0
T_TROPOPAUSE = T0 - L_TROPO * H_TROPO                     # 216.65 K
L_STRATO = -0.001         # K/m as a lapse: T INCREASES above 20 km

P_TROPOPAUSE = P0 * (T_TROPOPAUSE / T0) ** (G0 / (L_TROPO * R_AIR))
P_20KM = P_TROPOPAUSE * np.exp(-G0 * (H_TROPOPAUSE - H_TROPO) / (R_AIR * T_TROPOPAUSE))


R_E_STD = 6356766.0        # US1976 effective earth radius, for the H <-> Z map


def geopotential(z):
    """Geometric altitude Z -> geopotential altitude H = R Z/(R+Z).

    The layer formulas below are in H; the published tables and the trajectory
    module are both in Z. Skipping this conversion is a silent 2% pressure error
    at 30 km -- small enough to look like solver noise and large enough to move
    a drag calculation."""
    z = np.asarray(z, dtype=float)
    return R_E_STD * z / (R_E_STD + z)


def properties(h, geometric: bool = True):
    """(T [K], p [Pa], rho [kg/m^3], c [m/s]) at altitude h [m].

    `geometric=True` (the default, and what the trajectory state carries)
    converts to geopotential first. Clamped to [0, 32] km; the trajectory never
    leaves that band in 0.1, and a silent extrapolation into the mesosphere would
    be worse than a clamp."""
    h = np.asarray(h, dtype=float)
    if geometric:
        h = geopotential(h)
    h = np.clip(h, 0.0, H_TOP)

    T = np.where(h <= H_TROPO, T0 - L_TROPO * h,
        np.where(h <= H_TROPOPAUSE, T_TROPOPAUSE,
                 T_TROPOPAUSE - L_STRATO * (h - H_TROPOPAUSE)))
    p = np.where(
        h <= H_TROPO,
        P0 * np.maximum(T / T0, 1e-6) ** (G0 / (L_TROPO * R_AIR)),
        np.where(
            h <= H_TROPOPAUSE,
            P_TROPOPAUSE * np.exp(-G0 * (h - H_TROPO) / (R_AIR * T_TROPOPAUSE)),
            P_20KM * np.maximum(T / T_TROPOPAUSE, 1e-6) ** (G0 / (L_STRATO * R_AIR)),
        ),
    )
    rho = p / (R_AIR * T)
    return T, p, rho, np.sqrt(GAMMA_AIR * R_AIR * T)


def gravity(h) -> np.ndarray:
    """Inverse-square gravity magnitude at altitude h."""
    R_E = 6.371e6
    return G0 * (R_E / (R_E + np.asarray(h, dtype=float))) ** 2


def viscosity(T):
    from .thermo import sutherland
    return sutherland(T, 1.716e-5, 273.15, 110.4)


__all__ = ["properties", "gravity", "viscosity", "G0", "R_AIR", "GAMMA_AIR"]
