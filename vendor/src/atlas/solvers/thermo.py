"""Ideal-gas closure and conserved/primitive conversion.

Conserved state `U` is `[..., 4]` = (rho, rho*u, rho*v, E), optionally with a
5th slot `rho*Y` for the reaction progress variable of agent `a`. Primitive `W`
is (rho, u, v, p) (+ Y).

    E = p/(gamma-1) + rho(u^2+v^2)/2,   p = rho R T,   c = sqrt(gamma p / rho)

Pressure and density are floored rather than allowed to go negative. A negative
pressure in a compressible solver is not a small error to be tolerated -- the
sound speed becomes NaN and the whole field dies within a step -- so the floor
is a loud, countable event, not a silent clamp: `positivity_violations` records
how often it fired, and the corpus generator rejects an episode that trips it.
"""
from __future__ import annotations

import numpy as np

P_FLOOR = 1.0e-6
RHO_FLOOR = 1.0e-8

_violations = {"p": 0, "rho": 0}


def positivity_violations() -> dict:
    return dict(_violations)


def reset_positivity_counter() -> None:
    _violations["p"] = 0
    _violations["rho"] = 0


def pressure_from_cons(U: np.ndarray, gamma: float) -> np.ndarray:
    rho = np.maximum(U[..., 0], RHO_FLOOR)
    ke = 0.5 * (U[..., 1] ** 2 + U[..., 2] ** 2) / rho
    p = (gamma - 1.0) * (U[..., 3] - ke)
    bad = p < P_FLOOR
    if bad.any():
        _violations["p"] += int(bad.sum())
    return np.where(bad, P_FLOOR, p)


def sound_speed(U: np.ndarray, gamma: float) -> np.ndarray:
    return np.sqrt(gamma * pressure_from_cons(U, gamma) / np.maximum(U[..., 0], RHO_FLOOR))


def cons_to_prim(U: np.ndarray, gamma: float) -> np.ndarray:
    rho = np.maximum(U[..., 0], RHO_FLOOR)
    W = np.empty_like(U)
    W[..., 0] = rho
    W[..., 1] = U[..., 1] / rho
    W[..., 2] = U[..., 2] / rho
    W[..., 3] = pressure_from_cons(U, gamma)
    if U.shape[-1] > 4:
        W[..., 4:] = U[..., 4:] / rho[..., None]
    return W


def prim_to_cons(W: np.ndarray, gamma: float) -> np.ndarray:
    U = np.empty_like(W)
    rho = W[..., 0]
    U[..., 0] = rho
    U[..., 1] = rho * W[..., 1]
    U[..., 2] = rho * W[..., 2]
    U[..., 3] = W[..., 3] / (gamma - 1.0) + 0.5 * rho * (W[..., 1] ** 2 + W[..., 2] ** 2)
    if W.shape[-1] > 4:
        U[..., 4:] = rho[..., None] * W[..., 4:]
    return U


def temperature(U: np.ndarray, gamma: float, R: float) -> np.ndarray:
    return pressure_from_cons(U, gamma) / (np.maximum(U[..., 0], RHO_FLOOR) * R)


def mach(U: np.ndarray, gamma: float) -> np.ndarray:
    rho = np.maximum(U[..., 0], RHO_FLOOR)
    speed = np.sqrt(U[..., 1] ** 2 + U[..., 2] ** 2) / rho
    return speed / sound_speed(U, gamma)


def sutherland(T, mu_ref: float, T_ref: float, S: float):
    """mu(T) = mu_ref (T/T_ref)^{3/2} (T_ref+S)/(T+S)."""
    T = np.maximum(np.asarray(T, dtype=float), 1.0)
    return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)


__all__ = [
    "pressure_from_cons", "sound_speed", "cons_to_prim", "prim_to_cons",
    "temperature", "mach", "sutherland", "positivity_violations",
    "reset_positivity_counter", "P_FLOOR", "RHO_FLOOR",
]
