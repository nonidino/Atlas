"""Sweep design: Latin hypercube for train, deliberate CORNERS for test.

Full factorial over the five parameters is 2304 configs — too many. The spec's
design is 300 LHS train points plus 40 held-out test configs, where the held-out
set deliberately includes the *corners*:

  * highest p_c at highest altitude — the most under-expanded plume;
  * lowest p_c at sea level — the most over-expanded, flow-separating case.

Holding out corners rather than random points is the honest test. Interpolation
inside a swept box is the easy case, and the diversity principle in
[[transfer-learning-fine-tuning]] warns that in-distribution success overstates
generality.

Configs are screened for **thermal choking** before anything runs. If q_rxn is
too large for the inlet Mach number, Rayleigh flow drives M -> 1 inside the
reaction zone and the solver stalls or produces nonsense. That must be a rejected
config with a clear message, not something discovered 40 minutes into a run.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..solvers.oracles import mach_from_area_ratio, thermal_choking_margin

# spec §3.5
P_C_RANGE = (2.0e6, 8.0e6)
T_C_RANGE = (2200.0, 3400.0)
H_RANGE = (0.0, 25000.0)
M_INF_RANGE = (0.1, 3.5)
ALPHA_CHOICES = (0.0, 2.0, 5.0)

GAMMA_GAS = 1.22          # hot combustion products
R_GAS = 361.0             # J/(kg K), ~23 g/mol products

# Injector total temperature. NOT a free choice: Rayleigh flow ties it to the
# chamber area ratio and the top of the T_c range. With A_c/A_t = 2.5 the
# subsonic inlet sits at M = 0.238, whose thermal-choking limit is
# T0*/T0_in = 1/rayleigh_T0_ratio(0.238) = 4.10, so T_c/T_inject must stay under
# ~3.90 with margin. At 700 K that caps T_c at 2730 K and silently rejects 45% of
# the declared 2200-3400 K sweep -- truncating the corpus exactly at the hottest,
# most interesting configs. 900 K (regeneratively preheated propellant) makes the
# whole declared range attainable. See the implementation log, 2026-08-09.
T_INJECT = 900.0
CHAMBER_AREA_RATIO = 2.5  # chamber half-height 0.10 / throat half-height 0.04


@dataclass(frozen=True)
class SweepPoint:
    idx: int
    split: str                # 'train' | 'test'
    p_c: float
    T_c: float
    h0: float
    M_inf: float
    alpha: float              # degrees
    seed: int
    gamma_gas: float = GAMMA_GAS
    R_gas: float = R_GAS
    label: str = ""           # non-empty for the named corner cases

    @property
    def symmetric(self) -> bool:
        """The y-mirror BC is valid only at zero angle of attack. Silently wrong
        otherwise, so it is a property of the config, never a solver default."""
        return self.alpha == 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def q_rxn_for(T_c: float, T_inject: float, gamma: float, R: float) -> float:
    """Heat of reaction that produces flame temperature `T_c` from `T_inject`."""
    cp = gamma * R / (gamma - 1.0)
    return (T_c - T_inject) * cp


def choking_check(p: SweepPoint, chamber_ar: float = CHAMBER_AREA_RATIO,
                  T_inject: float = T_INJECT) -> float:
    """Rayleigh margin for this config; negative means it would thermally choke.

    `chamber_ar` is the chamber area over the throat area, which sets the
    subsonic inlet Mach number through the area-Mach relation."""
    M_in = float(mach_from_area_ratio(np.array([chamber_ar]), p.gamma_gas, supersonic=False)[0])
    tt = 1.0 + 0.5 * (p.gamma_gas - 1.0) * M_in ** 2
    T0_in = T_inject * tt
    T0_out = p.T_c * tt
    return thermal_choking_margin(M_in, T0_in, T0_out, p.gamma_gas)


def _lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Centred Latin hypercube on [0,1]^d."""
    cuts = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for k in range(d):
        rng.shuffle(cuts[:, k])
    return cuts


def _scale(u, lo, hi):
    return lo + u * (hi - lo)


def corner_cases(seed0: int = 90000) -> list[SweepPoint]:
    """The named corners the test split is built around, plus the extremes of the
    other three axes so the held-out set spans the box's vertices rather than
    only its p_c-h face."""
    out, i = [], 0
    named = [
        ("underexpanded_max", P_C_RANGE[1], T_C_RANGE[1], H_RANGE[1], 3.5, 0.0),
        ("overexpanded_max", P_C_RANGE[0], T_C_RANGE[0], 0.0, 0.1, 0.0),
        ("underexpanded_cold", P_C_RANGE[1], T_C_RANGE[0], H_RANGE[1], 2.0, 0.0),
        ("overexpanded_hot", P_C_RANGE[0], T_C_RANGE[1], 0.0, 0.8, 0.0),
        ("transonic_yaw", 5.0e6, 2800.0, 11000.0, 1.0, 5.0),
        ("high_alpha_low", 3.0e6, 2400.0, 2000.0, 0.6, 5.0),
        ("max_q_like", 6.0e6, 3000.0, 8000.0, 1.6, 2.0),
        ("ceiling_subsonic", 7.0e6, 3200.0, H_RANGE[1], 0.3, 0.0),
    ]
    for label, p_c, T_c, h0, M, al in named:
        out.append(SweepPoint(i, "test", p_c, T_c, h0, M, al, seed0 + i, label=label))
        i += 1
    return out


def build_sweep(n_train: int = 300, n_test: int = 40, seed: int = 20260809,
                chamber_ar: float = CHAMBER_AREA_RATIO) -> tuple[list[SweepPoint], list[SweepPoint], list[dict]]:
    """(train, test, rejected). Rejected configs carry the reason."""
    rng = np.random.default_rng(seed)
    rejected: list[dict] = []

    def keep(p: SweepPoint) -> bool:
        m = choking_check(p, chamber_ar)
        if m <= 0.05:                      # 5% margin, not merely non-negative
            rejected.append({**p.as_dict(), "reason": f"thermal choking margin {m:.3f}"})
            return False
        return True

    train: list[SweepPoint] = []
    tries = 0
    while len(train) < n_train and tries < 20 * n_train:
        need = n_train - len(train)
        u = _lhs(need, 4, rng)
        for k in range(need):
            p = SweepPoint(
                idx=len(train), split="train",
                p_c=float(_scale(u[k, 0], *P_C_RANGE)),
                T_c=float(_scale(u[k, 1], *T_C_RANGE)),
                h0=float(_scale(u[k, 2], *H_RANGE)),
                M_inf=float(_scale(u[k, 3], *M_INF_RANGE)),
                alpha=float(rng.choice(ALPHA_CHOICES)),
                seed=int(rng.integers(1, 2 ** 31)),
            )
            if keep(p):
                train.append(p)
        tries += need

    test = corner_cases()
    if n_test > len(test):
        u = _lhs(n_test - len(test), 4, rng)
        for k in range(u.shape[0]):
            # remaining test points sit on the box FACES (one coordinate pinned
            # to an extreme), so the held-out set stays an extrapolation test
            axis = int(rng.integers(0, 4))
            vals = [float(_scale(u[k, 0], *P_C_RANGE)), float(_scale(u[k, 1], *T_C_RANGE)),
                    float(_scale(u[k, 2], *H_RANGE)), float(_scale(u[k, 3], *M_INF_RANGE))]
            rng_hi = bool(rng.integers(0, 2))
            vals[axis] = [P_C_RANGE, T_C_RANGE, H_RANGE, M_INF_RANGE][axis][1 if rng_hi else 0]
            p = SweepPoint(len(test), "test", *vals, float(rng.choice(ALPHA_CHOICES)),
                           int(rng.integers(1, 2 ** 31)), label="face")
            if keep(p):
                test.append(p)
    return train, test, rejected


__all__ = [
    "SweepPoint", "build_sweep", "corner_cases", "choking_check", "q_rxn_for",
    "P_C_RANGE", "T_C_RANGE", "H_RANGE", "M_INF_RANGE", "ALPHA_CHOICES",
    "GAMMA_GAS", "R_GAS",
]
