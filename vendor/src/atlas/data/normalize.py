"""Per-episode reference scales, frozen BEFORE the run.

    rho^ = rho/rho_ref,  u^ = u/u_ref,  p^ = p/(rho_ref u_ref^2),
    T^   = T/T_ref,      t^ = t u_ref/L_ref

Every scale is derived from the episode's *inputs* — chamber pressure, flame
temperature, throat dimension, launch altitude — never from statistics of the
realized trajectory. Normalizing by realized data leaks the future into the
model's input scaling: the network would see, at t=0, a number that depends on
what happens at t=10 s. This is the single most tempting shortcut in the phase
and it is silently wrong, so `refs_for_episode` takes the config and nothing else.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..solvers import atmosphere
from ..solvers.oracles import area_ratio, mach_from_area_ratio


@dataclass(frozen=True)
class Refs:
    rho: float
    u: float
    T: float
    L: float
    p: float          # = rho * u^2, the dynamic scale (NOT a static pressure)

    def nondim(self, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            "rho": fields["rho"] / self.rho,
            "u": fields["u"] / self.u,
            "v": fields["v"] / self.u,
            "p": fields["p"] / self.p,
            "T": fields["T"] / self.T,
            **({"Y": fields["Y"]} if "Y" in fields else {}),
        }

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeScales:
    """All per-agent references plus the derived chamber/exit quantities the
    conditioning scalars need."""
    per_agent: dict
    mdot: float               # per unit depth, kg/(s m)
    u_exit: float
    p_exit: float
    T_exit: float
    M_exit: float
    rho_exit: float


def chamber_gas(p_c: float, T_c: float, gamma: float, R: float) -> tuple[float, float]:
    return p_c / (R * T_c), float(np.sqrt(gamma * R * T_c))


def exit_conditions(p_c: float, T_c: float, gamma: float, R: float, eps: float):
    """Ideal quasi-1D nozzle exit state for expansion ratio `eps`.

    Known before the run from the episode inputs alone, which is what makes it
    usable as a frozen reference scale for the plume."""
    M_e = float(mach_from_area_ratio(np.array([eps]), gamma, supersonic=True)[0])
    tt = 1.0 + 0.5 * (gamma - 1.0) * M_e ** 2
    T_e = T_c / tt
    p_e = p_c / tt ** (gamma / (gamma - 1.0))
    rho_e = p_e / (R * T_e)
    u_e = M_e * float(np.sqrt(gamma * R * T_e))
    return M_e, T_e, p_e, rho_e, u_e


def refs_for_episode(cfg, ep) -> EpisodeScales:
    """`ep` is a SweepPoint: p_c, T_c, h0, M_inf, alpha."""
    geo = cfg.geometry
    gamma_g, R_g = ep.gamma_gas, ep.R_gas
    rho_c, a_c = chamber_gas(ep.p_c, ep.T_c, gamma_g, R_g)
    A_t = 2.0 * geo.throat_halfheight
    eps = geo.exit_halfheight / geo.throat_halfheight
    M_e, T_e, p_e, rho_e, u_e = exit_conditions(ep.p_c, ep.T_c, gamma_g, R_g, eps)

    mdot = ep.p_c * A_t * np.sqrt(gamma_g / (R_g * ep.T_c)) * (
        2.0 / (gamma_g + 1.0)) ** ((gamma_g + 1.0) / (2.0 * (gamma_g - 1.0)))

    T_inf, p_inf, rho_inf, c_inf = (float(x) for x in atmosphere.properties(ep.h0))
    v_flight = max(ep.M_inf * c_inf, 1.0)
    L_body = geo.shell_z[1] - geo.shell_z[0]

    engine = Refs(rho_c, a_c, ep.T_c, A_t, rho_c * a_c ** 2)
    plume = Refs(rho_e, u_e, T_e, 2.0 * geo.exit_halfheight, rho_e * u_e ** 2)
    ext = Refs(rho_inf, v_flight, T_inf, L_body, rho_inf * v_flight ** 2)
    solid = Refs(2700.0, v_flight, ep.T_c, geo.shell_thickness, 70.0e9)

    per_agent = {"a": engine, "b": engine, "e": engine,
                 "f": plume, "d": ext, "g": ext, "c": solid}
    return EpisodeScales(per_agent, float(mdot), u_e, p_e, T_e, M_e, rho_e)


def gas_conditioning(fields, scales: Refs, gamma: float, R: float, mu: float,
                     p_inf: float) -> np.ndarray:
    """(Ma, Re, Pr, gamma, p/p_inf) — the five scalars FiLM'd into a gas agent."""
    rho = np.mean(fields["rho"]); T = np.mean(fields["T"])
    speed = np.sqrt(np.mean(fields["u"]) ** 2 + np.mean(fields["v"]) ** 2)
    c = np.sqrt(gamma * R * max(T, 1.0))
    Re = rho * speed * scales.L / max(mu, 1e-12)
    return np.array([speed / c, Re, 0.72, gamma, np.mean(fields["p"]) / max(p_inf, 1.0)])


def solid_conditioning(T_field, h_conv: float, mat, L: float, dT: float) -> np.ndarray:
    """(Bi, alpha*dT, E/sigma_ref) — the three scalars for the shell."""
    Bi = h_conv * L / mat.k
    return np.array([Bi, mat.alpha * dT, 1.0])


__all__ = [
    "Refs", "EpisodeScales", "refs_for_episode", "exit_conditions", "chamber_gas",
    "gas_conditioning", "solid_conditioning",
]
