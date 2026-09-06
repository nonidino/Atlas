"""Closed-form solutions the solvers are graded against (M1).

The discipline this phase exists to enforce: **a solver is not trusted to teach
anything until it reproduces a closed-form answer.** A surrogate trained on a
subtly wrong solver learns the wrong physics perfectly, and Phase 4 would then be
measuring agreement with a bug.

Nothing here is ever called inside a time loop.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

# --------------------------------------------------------------------------
# compressible gas dynamics
# --------------------------------------------------------------------------


def isentropic_T_ratio(M, gamma):
    """T0/T."""
    return 1.0 + 0.5 * (gamma - 1.0) * np.asarray(M, dtype=float) ** 2


def isentropic_p_ratio(M, gamma):
    """p0/p."""
    return isentropic_T_ratio(M, gamma) ** (gamma / (gamma - 1.0))


def area_ratio(M, gamma):
    """A/A* for the quasi-1D isentropic nozzle (planar: 'area' = 2 * half-height)."""
    M = np.asarray(M, dtype=float)
    return (1.0 / M) * ((2.0 / (gamma + 1.0)) * isentropic_T_ratio(M, gamma)) ** (
        (gamma + 1.0) / (2.0 * (gamma - 1.0)))


def mach_from_area_ratio(AR, gamma, supersonic: bool = True):
    """Invert A/A*. Two roots; `supersonic` picks the diverging-section one."""
    AR = np.atleast_1d(np.asarray(AR, dtype=float))
    out = np.empty_like(AR)
    for i, ar in enumerate(AR):
        if ar <= 1.0 + 1e-12:
            out[i] = 1.0
            continue
        lo, hi = (1.0 + 1e-9, 50.0) if supersonic else (1e-6, 1.0 - 1e-9)
        out[i] = brentq(lambda m: area_ratio(m, gamma) - ar, lo, hi, xtol=1e-12)
    return out


def rayleigh_T0_ratio(M, gamma):
    """T0/T0* for constant-area flow with heat addition. Heat addition drives
    M -> 1 from either side; T0/T0* = 1 is the thermal-choking limit."""
    M2 = np.asarray(M, dtype=float) ** 2
    return ((gamma + 1.0) * M2 * (2.0 + (gamma - 1.0) * M2)) / (1.0 + gamma * M2) ** 2


def thermal_choking_margin(M_in: float, T0_in: float, T0_out: float, gamma: float) -> float:
    """(T0* - T0_out)/T0* for a Rayleigh-flow reaction zone.

    Negative means the requested heat release would drive the flow through M=1
    inside agent `a`: the solver would stall or produce nonsense. The generator
    must check this BEFORE running, not discover it 40 minutes in."""
    T0_star = T0_in / rayleigh_T0_ratio(M_in, gamma)
    return float((T0_star - T0_out) / T0_star)


def ideal_thrust(mdot: float, u_e: float, p_e: float, p_inf: float, A_e: float) -> float:
    """F = mdot u_e + (p_e - p_inf) A_e -- validates the e-f interface integral."""
    return mdot * u_e + (p_e - p_inf) * A_e


def adiabatic_flame_temperature(T_in: float, q_rxn: float, cp: float) -> float:
    """Constant-pressure complete burn of the progress variable."""
    return T_in + q_rxn / cp


# --------------------------------------------------------------------------
# viscous
# --------------------------------------------------------------------------


def blasius_thickness(z, Re_z):
    """delta ~ 5.0 z / sqrt(Re_z), the 99% laminar flat-plate thickness."""
    return 5.0 * np.asarray(z, dtype=float) / np.sqrt(np.maximum(Re_z, 1e-30))


# --------------------------------------------------------------------------
# conduction and elasticity
# --------------------------------------------------------------------------


def erf_slab(x, t, T0: float, Ts: float, alpha_th: float):
    """Semi-infinite slab, step surface temperature:
    T(x,t) = Ts + (T0 - Ts) erf( x / (2 sqrt(alpha t)) )."""
    from scipy.special import erf
    x = np.asarray(x, dtype=float)
    return Ts + (T0 - Ts) * erf(x / (2.0 * np.sqrt(max(alpha_th * t, 1e-30))))


def constrained_thermal_stress(E: float, alpha: float, dT: float, nu: float) -> float:
    """A fully constrained uniformly heated plate: sigma = -E alpha dT/(1-nu).
    Free expansion of the same plate must give exactly zero stress."""
    return -E * alpha * dT / (1.0 - nu)


# --------------------------------------------------------------------------
# trajectory
# --------------------------------------------------------------------------


def tsiolkovsky(v_e: float, m0: float, mf: float) -> float:
    """delta-v = v_e ln(m0/mf), with no gravity and no drag."""
    return v_e * np.log(m0 / mf)


__all__ = [
    "isentropic_T_ratio", "isentropic_p_ratio", "area_ratio", "mach_from_area_ratio",
    "rayleigh_T0_ratio", "thermal_choking_margin", "ideal_thrust",
    "adiabatic_flame_temperature", "blasius_thickness", "erf_slab",
    "constrained_thermal_stress", "tsiolkovsky",
]
