"""Approximate Riemann solvers, and the exact one used as an oracle.

Both approximate fluxes are required by the spec: **Rusanov** (local
Lax-Friedrichs) is the robust default for bring-up, **HLLC** is sharper and is
what the production corpus uses. They must agree to <1% on smooth flow, which is
an M1 test — a disagreement there means one of them is wrong, and the pair is
much easier to debug than either alone.

Everything here works on face-normal rotated states, so one implementation
serves both grid directions and any curvilinear face orientation: rotate
$(u,v)$ into $(u_n, u_t)$, solve the 1D problem, rotate the momentum flux back.
"""
from __future__ import annotations

import numpy as np

from .thermo import pressure_from_cons, sound_speed


def _rotate(U: np.ndarray, n: np.ndarray) -> np.ndarray:
    """[..., 4+] conserved state -> same with momentum in (normal, tangential)."""
    out = U.copy()
    mz, my = U[..., 1], U[..., 2]
    out[..., 1] = mz * n[..., 0] + my * n[..., 1]
    out[..., 2] = -mz * n[..., 1] + my * n[..., 0]
    return out


def _unrotate_flux(F: np.ndarray, n: np.ndarray) -> np.ndarray:
    out = F.copy()
    fn, ft = F[..., 1], F[..., 2]
    out[..., 1] = fn * n[..., 0] - ft * n[..., 1]
    out[..., 2] = fn * n[..., 1] + ft * n[..., 0]
    return out


def _flux_1d(U: np.ndarray, gamma: float) -> np.ndarray:
    """Normal-direction physical flux of an already-rotated state."""
    rho = U[..., 0]
    un = U[..., 1] / rho
    ut = U[..., 2] / rho
    p = pressure_from_cons(U, gamma)
    E = U[..., 3]
    F = np.empty_like(U)
    F[..., 0] = rho * un
    F[..., 1] = rho * un * un + p
    F[..., 2] = rho * un * ut
    F[..., 3] = (E + p) * un
    if U.shape[-1] > 4:                      # passive scalars: rho*Y
        F[..., 4:] = U[..., 4:] * un[..., None]
    return F


def rusanov(UL: np.ndarray, UR: np.ndarray, n: np.ndarray, gamma: float) -> np.ndarray:
    """Local Lax-Friedrichs. Maximally diffusive, essentially unbreakable."""
    L, R = _rotate(UL, n), _rotate(UR, n)
    aL = np.abs(L[..., 1] / L[..., 0]) + sound_speed(L, gamma)
    aR = np.abs(R[..., 1] / R[..., 0]) + sound_speed(R, gamma)
    s = np.maximum(aL, aR)[..., None]
    F = 0.5 * (_flux_1d(L, gamma) + _flux_1d(R, gamma)) - 0.5 * s * (R - L)
    return _unrotate_flux(F, n)


def hllc(UL: np.ndarray, UR: np.ndarray, n: np.ndarray, gamma: float) -> np.ndarray:
    """Harten-Lax-van Leer-Contact: three waves, so the contact discontinuity is
    resolved rather than smeared (Toro §10.4). Wave speeds by the Davis estimate."""
    L, R = _rotate(UL, n), _rotate(UR, n)
    rL, rR = L[..., 0], R[..., 0]
    uL, uR = L[..., 1] / rL, R[..., 1] / rR
    pL, pR = pressure_from_cons(L, gamma), pressure_from_cons(R, gamma)
    cL, cR = sound_speed(L, gamma), sound_speed(R, gamma)

    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)
    denom = rL * (SL - uL) - rR * (SR - uR)
    SM = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / np.where(
        np.abs(denom) < 1e-300, 1e-300, denom)

    FL, FR = _flux_1d(L, gamma), _flux_1d(R, gamma)

    def star(U, S, u, r, p):
        f = r * (S - u) / np.where(np.abs(S - SM) < 1e-300, 1e-300, S - SM)
        out = np.empty_like(U)
        out[..., 0] = f
        out[..., 1] = f * SM
        out[..., 2] = f * (U[..., 2] / U[..., 0])
        out[..., 3] = f * (U[..., 3] / r + (SM - u) * (SM + p / (r * (S - u))))
        if U.shape[-1] > 4:
            out[..., 4:] = f[..., None] * (U[..., 4:] / r[..., None])
        return out

    FsL = FL + SL[..., None] * (star(L, SL, uL, rL, pL) - L)
    FsR = FR + SR[..., None] * (star(R, SR, uR, rR, pR) - R)

    F = np.where(SL[..., None] >= 0, FL,
        np.where(SM[..., None] >= 0, FsL,
        np.where(SR[..., None] >= 0, FsR, FR)))
    return _unrotate_flux(F, n)


FLUXES = {"rusanov": rusanov, "hllc": hllc}


# --------------------------------------------------------------------------
# exact Riemann solver -- ORACLE ONLY, never in the time loop
# --------------------------------------------------------------------------


def exact_riemann(rhoL, uL, pL, rhoR, uR, pR, gamma, x_over_t):
    """Exact solution of the 1D Riemann problem (Toro §4), sampled at x/t.

    Used only to grade the approximate fluxes on the Sod shock tube (M1). Its
    Newton iteration on p* is not something to put in an inner loop."""
    cL, cR = np.sqrt(gamma * pL / rhoL), np.sqrt(gamma * pR / rhoR)
    g1, g2 = (gamma - 1.0) / (2.0 * gamma), (gamma + 1.0) / (2.0 * gamma)

    def f_side(p, rho, pk, ck):
        A, B = 2.0 / ((gamma + 1.0) * rho), (gamma - 1.0) / (gamma + 1.0) * pk
        if p > pk:                                        # shock
            return (p - pk) * np.sqrt(A / (p + B)), np.sqrt(A / (p + B)) * (
                1.0 - (p - pk) / (2.0 * (B + p)))
        return (2.0 * ck / (gamma - 1.0)) * ((p / pk) ** g1 - 1.0), \
            (1.0 / (rho * ck)) * (p / pk) ** (-g2)

    p = max(1e-8, 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rhoL + rhoR) * (cL + cR))
    for _ in range(60):
        fL, dfL = f_side(p, rhoL, pL, cL)
        fR, dfR = f_side(p, rhoR, pR, cR)
        dp = (fL + fR + uR - uL) / (dfL + dfR)
        p = max(1e-8, p - dp)
        if abs(dp) < 1e-12 * p:
            break
    us = 0.5 * (uL + uR) + 0.5 * (f_side(p, rhoR, pR, cR)[0] - f_side(p, rhoL, pL, cL)[0])

    S = np.atleast_1d(np.asarray(x_over_t, dtype=float))
    rho = np.empty_like(S); u = np.empty_like(S); pr = np.empty_like(S)
    gm = (gamma - 1.0) / (gamma + 1.0)

    for k, s in enumerate(S):
        if s <= us:                                        # left of the contact
            if p > pL:                                     # left shock
                SLs = uL - cL * np.sqrt(g2 * p / pL + g1)
                if s <= SLs:
                    rho[k], u[k], pr[k] = rhoL, uL, pL
                else:
                    rho[k] = rhoL * (p / pL + gm) / (gm * p / pL + 1.0)
                    u[k], pr[k] = us, p
            else:                                          # left fan
                cLs = cL * (p / pL) ** g1
                if s <= uL - cL:
                    rho[k], u[k], pr[k] = rhoL, uL, pL
                elif s >= us - cLs:
                    rho[k] = rhoL * (p / pL) ** (1.0 / gamma)
                    u[k], pr[k] = us, p
                else:
                    u[k] = 2.0 / (gamma + 1.0) * (cL + (gamma - 1.0) / 2.0 * uL + s)
                    c = 2.0 / (gamma + 1.0) * (cL + (gamma - 1.0) / 2.0 * (uL - s))
                    rho[k] = rhoL * (c / cL) ** (2.0 / (gamma - 1.0))
                    pr[k] = pL * (c / cL) ** (2.0 * gamma / (gamma - 1.0))
        else:
            if p > pR:                                     # right shock
                SRs = uR + cR * np.sqrt(g2 * p / pR + g1)
                if s >= SRs:
                    rho[k], u[k], pr[k] = rhoR, uR, pR
                else:
                    rho[k] = rhoR * (p / pR + gm) / (gm * p / pR + 1.0)
                    u[k], pr[k] = us, p
            else:                                          # right fan
                cRs = cR * (p / pR) ** g1
                if s >= uR + cR:
                    rho[k], u[k], pr[k] = rhoR, uR, pR
                elif s <= us + cRs:
                    rho[k] = rhoR * (p / pR) ** (1.0 / gamma)
                    u[k], pr[k] = us, p
                else:
                    u[k] = 2.0 / (gamma + 1.0) * (-cR + (gamma - 1.0) / 2.0 * uR + s)
                    c = 2.0 / (gamma + 1.0) * (cR - (gamma - 1.0) / 2.0 * (uR - s))
                    rho[k] = rhoR * (c / cR) ** (2.0 / (gamma - 1.0))
                    pr[k] = pR * (c / cR) ** (2.0 * gamma / (gamma - 1.0))

    return rho, u, pr, float(p), float(us)


__all__ = ["rusanov", "hllc", "FLUXES", "exact_riemann"]
